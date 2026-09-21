"""Phase 0.2 - Inventory the submitted manuscript (tex + pdf).

Emits 00_submitted/inventory/{pdf_text.txt, inventory.json, defects.json}
so that every later edit can be diffed against a frozen ground truth.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "00_submitted"
OUT = SUB / "inventory"
OUT.mkdir(parents=True, exist_ok=True)


def pdf_text(path: Path) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc), doc.page_count


def scan_tex(tex: str) -> dict:
    inv: dict = {}
    inv["sections"] = [
        {"level": m.group(1), "title": m.group(2)}
        for m in re.finditer(r"\\(section|subsection|subsubsection)\*?\{([^}]*)\}", tex)
    ]
    inv["labels"] = re.findall(r"\\label\{([^}]*)\}", tex)
    inv["refs"] = re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", tex)
    inv["cites"] = sorted({c.strip() for m in re.finditer(r"\\cite\{([^}]*)\}", tex)
                           for c in m.group(1).split(",")})
    inv["figures"] = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)
    inv["equations"] = re.findall(r"\\label\{(eq:[^}]*)\}", tex)
    inv["tables"] = re.findall(r"\\label\{(tab:[^}]*)\}", tex)
    inv["algorithms"] = re.findall(r"\\label\{(alg:[^}]*)\}", tex)
    inv["figure_labels"] = re.findall(r"\\label\{(fig:[^}]*)\}", tex)
    return inv


# Conversational / informal constructions flagged by R1.2, R2.8, R6.2, R7.
INFORMAL_PATTERNS = [
    r"to its knees",
    r"things like",
    r"to be able to see the whole network",
    r"lightning speed",
    r"highly compelling",
    r"a bunch of",
    r"The key to making the system fast",
    r"we can update the calculation",
    r"That means",
    r"that means",
    r"It gets \d",
    r"It looked at",
    r"looked at \d",
    r"bring the entire network",
    r"Some of them use something called",
    r"There are also attacks",
    r"Here are the details",
    r"exceptional robustness",
    r"generalizes robustly",
    r"exceptionally robust",
    r"exceptionally negligible",
    r"This confirms",
    r"confirms that",
    r"surpasses",
    r"outperforming",
    r"real-world SDN deployment",
]

OVERCLAIM_PATTERNS = [
    r"\bconfirm(s|ed|ing)?\b",
    r"\bprove(s|d|n)?\b",
    r"\bsurpass(es|ed)?\b",
    r"\bexceptional\b",
    r"\brobustly\b",
    r"\bperfect\b",
    r"\bnear-perfect\b",
    r"\bguarantee(s|d)?\b",
    r"\balways\b",
    r"\bensuring\b",
    r"\bevery attack flow is caught\b",
]


def scan_defects(tex: str, ptext: str) -> dict:
    lines = tex.split("\n")
    d: dict = {"informal": [], "overclaim": [], "pdf_broken_refs": [], "notation": []}
    for i, line in enumerate(lines, 1):
        for pat in INFORMAL_PATTERNS:
            for m in re.finditer(pat, line):
                d["informal"].append({"line": i, "pattern": pat,
                                      "text": line.strip()[:220]})
        for pat in OVERCLAIM_PATTERNS:
            for m in re.finditer(pat, line, flags=re.I):
                d["overclaim"].append({"line": i, "pattern": pat,
                                       "match": m.group(0),
                                       "text": line.strip()[:220]})
    # Broken cross references visible in the rendered PDF (R1.1)
    for m in re.finditer(r"\(\?\?\)", ptext):
        d["pdf_broken_refs"].append(ptext[max(0, m.start() - 160):m.start() + 30].replace("\n", " "))
    # Notation defects (R1.3 Acc spacing, R2.8 (au) artefact)
    for m in re.finditer(r"\(au\)", ptext):
        d["notation"].append({"kind": "tau_rendered_as_au",
                              "ctx": ptext[max(0, m.start() - 120):m.start() + 30].replace("\n", " ")})
    for m in re.finditer(r"\\mathrm\{Acc\}", tex):
        d["notation"].append({"kind": "Acc_macro", "ctx": "Eq. (4) metrics block"})
    return d


def main() -> int:
    tex = (SUB / "XAI-SDN-journal.tex").read_text(encoding="utf-8", errors="replace")
    ptext, pages = pdf_text(SUB / "XAI-SDN-journal.pdf")
    (OUT / "pdf_text.txt").write_text(ptext, encoding="utf-8")

    bbl = (SUB / "XAI-SDN-journal.bbl").read_text(encoding="utf-8", errors="replace")
    bib_order = re.findall(r"\\bibitem\{([^}]*)\}", bbl)

    inv = scan_tex(tex)
    inv["pdf_pages"] = pages
    inv["pdf_chars"] = len(ptext)
    inv["bib_order"] = {k: i + 1 for i, k in enumerate(bib_order)}
    inv["n_refs"] = len(bib_order)
    inv["tex_lines"] = tex.count("\n") + 1
    inv["tex_words"] = len(re.findall(r"\b\w+\b", tex))

    # figure text (for the (au) and Eq.(??) defects living inside graphics)
    figtext = {}
    for p in sorted((SUB / "figures").glob("*.pdf")) + [SUB / "entropyengine_standalone.pdf"]:
        try:
            t, _ = pdf_text(p)
            figtext[p.name] = " ".join(t.split())
        except Exception as exc:  # pragma: no cover
            figtext[p.name] = f"<unreadable: {exc}>"
    inv["figure_text"] = figtext

    defects = scan_defects(tex, ptext)
    for name, t in figtext.items():
        if "??" in t:
            defects["pdf_broken_refs"].append(f"[in graphic {name}] ...{t[:200]}")
        if "(au)" in t:
            defects["notation"].append({"kind": "tau_rendered_as_au_in_graphic",
                                        "file": name, "ctx": t[:200]})

    (OUT / "inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")
    (OUT / "defects.json").write_text(json.dumps(defects, indent=2), encoding="utf-8")

    print(f"pages={pages}  refs={inv['n_refs']}  sections={len(inv['sections'])}")
    print(f"figures={len(inv['figure_labels'])}  tables={len(inv['tables'])}  "
          f"equations={len(inv['equations'])}  algorithms={len(inv['algorithms'])}")
    print(f"informal_hits={len(defects['informal'])}  "
          f"overclaim_hits={len(defects['overclaim'])}  "
          f"broken_refs={len(defects['pdf_broken_refs'])}  "
          f"notation={len(defects['notation'])}")

    # uncited floats (R5.6)
    uncited = {
        "figures": [l for l in inv["figure_labels"] if l not in inv["refs"]],
        "tables": [l for l in inv["tables"] if l not in inv["refs"]],
        "algorithms": [l for l in inv["algorithms"] if l not in inv["refs"]],
        "equations": [l for l in inv["equations"] if l not in inv["refs"]],
    }
    (OUT / "uncited_floats.json").write_text(json.dumps(uncited, indent=2), encoding="utf-8")
    print("uncited:", {k: len(v) for k, v in uncited.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
