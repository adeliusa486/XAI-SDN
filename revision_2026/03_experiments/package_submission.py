"""Assemble the resubmission package against the IEEE Access checklist.

The checklist the editor attached asks for six things: a clean manuscript PDF in
double-column format, the LaTeX source with biographies and photos, a version
with every individual change highlighted, the response-to-reviewers document, any
video submitted for peer review, and supplementary material. This script builds
all of them into 09_package/ and refuses to finish if the manuscript gate fails,
so an unfilled placeholder or an unresolved reference cannot reach a submission.

    python 03_experiments/package_submission.py
    python 03_experiments/package_submission.py --skip-compile
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))

from paths import (MANUSCRIPT, PACKAGE, PROJECT, RESULTS,  # noqa: E402
                   RESPONSE, SUBMITTED, VENV_PY)

TEX = "XAI-SDN-journal"


def run(cmd: list[str], cwd: Path, quiet: bool = True) -> int:
    return subprocess.run(cmd, cwd=str(cwd),
                          stdout=subprocess.DEVNULL if quiet else None,
                          stderr=subprocess.STDOUT).returncode


def compile_manuscript() -> None:
    print("compiling manuscript (4 passes)...", flush=True)
    run(["pdflatex", "-interaction=nonstopmode", "-draftmode", TEX + ".tex"], MANUSCRIPT)
    run(["bibtex", TEX], MANUSCRIPT)
    run(["pdflatex", "-interaction=nonstopmode", "-draftmode", TEX + ".tex"], MANUSCRIPT)
    run(["pdflatex", "-interaction=nonstopmode", TEX + ".tex"], MANUSCRIPT)


def gate() -> bool:
    print("running the manuscript gate...", flush=True)
    rc = subprocess.run([str(VENV_PY), str(HERE / "common" / "check_manuscript.py"),
                         str(MANUSCRIPT / (TEX + ".tex"))],
                        cwd=str(PROJECT)).returncode
    return rc == 0


def highlight() -> Path | None:
    old = SUBMITTED / (TEX + ".pdf")
    new = MANUSCRIPT / (TEX + ".pdf")
    out = PACKAGE / (TEX + "_HIGHLIGHTED.pdf")
    if not old.exists() or not new.exists():
        print("  cannot highlight: missing one of the two PDFs")
        return None
    print("building the highlighted PDF...", flush=True)
    rc = subprocess.run([str(VENV_PY), str(HERE / "common" / "highlight_pdf.py"),
                         str(old), str(new), str(out)],
                        cwd=str(PROJECT)).returncode
    return out if rc == 0 and out.exists() else None


def copy_source() -> list[str]:
    src = PACKAGE / "latex_source"
    src.mkdir(parents=True, exist_ok=True)
    taken = []
    for pat in ("*.tex", "*.cls", "*.bib", "*.bbl", "*.pdf", "*.jpg", "*.png"):
        for f in MANUSCRIPT.glob(pat):
            shutil.copy2(f, src / f.name)
            taken.append(f.name)
    figdir = MANUSCRIPT / "figures"
    if figdir.exists():
        shutil.copytree(figdir, src / "figures", dirs_exist_ok=True)
        taken += [f"figures/{f.name}" for f in figdir.iterdir()]
    return sorted(taken)


def copy_supplementary() -> list[str]:
    sup = PACKAGE / "supplementary"
    sup.mkdir(parents=True, exist_ok=True)
    taken = []
    for f in sorted(RESULTS.glob("*.json")):
        shutil.copy2(f, sup / f.name)
        taken.append(f.name)
    for f in sorted(RESULTS.glob("*.csv")):
        shutil.copy2(f, sup / f.name)
        taken.append(f.name)
    # Provenance notes belong with the results they qualify. E4b_PROVENANCE.md
    # records two defects in the surrounding code that were fixed without
    # re-running the experiment, and a reader of E4b_mininet_testbed.json needs
    # it to interpret two stale fields in that file.
    for f in sorted(RESULTS.glob("*_PROVENANCE.md")):
        shutil.copy2(f, sup / f.name)
        taken.append(f.name)
    ev = RESULTS / "mininet_evidence"
    if ev.exists():
        # Older runs captured the flow table after the topology had been torn
        # down, so those files hold "s1 is not a bridge or a socket" rather than
        # a dump. Shipping an error message as evidence is worse than shipping
        # nothing, so skip any capture that only records the failure.
        dest = sup / "mininet_evidence"
        dest.mkdir(parents=True, exist_ok=True)
        skipped = []
        for f in sorted(ev.iterdir()):
            if f.is_dir():
                continue
            if f.suffix == ".txt":
                head = f.read_text(encoding="utf-8", errors="ignore").strip()
                if "is not a bridge or a socket" in head or not head:
                    skipped.append(f.name)
                    continue
            shutil.copy2(f, dest / f.name)
        taken.append("mininet_evidence/")
        if skipped:
            print(f"  skipped {len(skipped)} stale evidence capture(s): "
                  f"{', '.join(skipped)}")
    return taken


def zip_supplementary() -> str | None:
    """IEEE Access requires supplementary material as a single .zip."""
    sup = PACKAGE / "supplementary"
    if not sup.exists():
        return None
    archive = shutil.make_archive(str(PACKAGE / "supplementary"), "zip",
                                  root_dir=str(sup))
    return Path(archive).name


def find_video() -> Path | None:
    for d in (Path.home() / "Downloads", PACKAGE):
        for f in d.glob("XAI-SDN_mininet_live_demo*.mp4"):
            return f
    return None


def main(argv: list[str]) -> int:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if "--skip-compile" not in argv:
        compile_manuscript()

    clean = MANUSCRIPT / (TEX + ".pdf")
    if not clean.exists():
        print("FAIL: no compiled PDF")
        return 1
    shutil.copy2(clean, PACKAGE / "Main_Manuscript.pdf")

    passed = gate()
    hl = highlight()
    source = copy_source()
    sup = copy_supplementary()
    sup_zip = zip_supplementary()

    for name in ("Response_to_Reviewers.docx", "Response_to_Reviewers.md"):
        f = RESPONSE / name
        if f.exists():
            shutil.copy2(f, PACKAGE / name)

    video = find_video()
    if video:
        shutil.copy2(video, PACKAGE / video.name)
        tx = video.with_name(video.stem + "_transcript.txt")
        if tx.exists():
            shutil.copy2(tx, PACKAGE / tx.name)

    import fitz
    with fitz.open(str(clean)) as doc:
        pages = doc.page_count

    checklist = {
        "Main Manuscript PDF (double column)": (PACKAGE / "Main_Manuscript.pdf").exists(),
        "LaTeX source with biographies and photos": len(source) > 0,
        "Manuscript with all changes highlighted": bool(hl),
        "Response to Reviewers document": (PACKAGE / "Response_to_Reviewers.docx").exists(),
        "Video submitted for peer review": bool(video),
        "Supplementary material": len(sup) > 0,
        "Supplementary material zipped for upload": bool(sup_zip),
        "Automated manuscript gate passes": passed,
    }
    manifest = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "page_count": pages,
        "checklist": checklist,
        "latex_source_files": source,
        "supplementary_files": sup,
        "supplementary_zip": sup_zip,
        "video": video.name if video else None,
        "build_seconds": round(time.time() - t0, 1),
    }
    (PACKAGE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")

    print(f"\nmanuscript: {pages} pages")
    for k, v in checklist.items():
        print(f"  [{'x' if v else ' '}] {k}")
    missing = [k for k, v in checklist.items() if not v]
    print(f"\npackage written to {PACKAGE}")
    if missing:
        print("NOT READY TO SUBMIT, missing:")
        for m in missing:
            print("  -", m)
        return 1
    print("package complete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
