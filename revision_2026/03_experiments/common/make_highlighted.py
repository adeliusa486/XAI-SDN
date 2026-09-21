"""Phase 7 - Build the yellow-highlighted revision PDF with latexdiff.

The Associate Editor asked specifically for changes marked with "the yellow
highlight tool". latexdiff's default markup is blue underline plus red strikeout,
which is not that. This script overrides latexdiff's add/delete commands so that
added and changed text is rendered with a yellow background via soul's \\hl,
and deleted text is left out of the highlighted PDF entirely rather than struck
through, which keeps the document readable at the length IEEE Access expects.

Usage:
    python make_highlighted.py OLD.tex NEW.tex OUTDIR
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# latexdiff is a Perl script and needs Algorithm::Diff, which MiKTeX does not
# ship. A local copy is installed outside the project tree; point Perl at it.
PERLLIB_WIN = r"C:/Users/adeel/xaisdn_revision/perllib"
# MiKTeX dispatches latexdiff through the MSYS perl shipped with Git for Windows,
# whose PERL5LIB separator is ":", so a drive-lettered path would be split at the
# colon. The MSYS-style path is therefore used instead.
PERLLIB_MSYS = "/c/Users/adeel/xaisdn_revision/perllib"
if Path(PERLLIB_WIN).exists():
    os.environ["PERL5LIB"] = PERLLIB_MSYS

# latexdiff preamble override: yellow background for additions.
# soulpos/soul cannot break across some boxes, so \hl is applied per-word by
# latexdiff's default CHANGEBAR-free markup; ulem is disabled.
PREAMBLE = r"""
%DIF PREAMBLE EXTENSION ADDED BY make_highlighted.py
\RequirePackage{xcolor}
\RequirePackage{soul}
\definecolor{revyellow}{rgb}{1,1,0}
\sethlcolor{revyellow}
\providecommand{\DIFaddtex}[1]{\hl{#1}}
\providecommand{\DIFdeltex}[1]{}
\providecommand{\DIFadd}[1]{\hl{#1}}
\providecommand{\DIFdel}[1]{}
\providecommand{\DIFaddbegin}{}
\providecommand{\DIFaddend}{}
\providecommand{\DIFdelbegin}{}
\providecommand{\DIFdelend}{}
\providecommand{\DIFaddbeginFL}{}
\providecommand{\DIFaddendFL}{}
\providecommand{\DIFdelbeginFL}{}
\providecommand{\DIFdelendFL}{}
\providecommand{\DIFaddFL}[1]{\hl{#1}}
\providecommand{\DIFdelFL}[1]{}
%DIF END PREAMBLE EXTENSION
"""

# latexdiff configuration: never mark inside these, or the document will not build.
SAFE_CMD = ("chapter,section,subsection,subsubsection,paragraph,"
            "caption,title,author,address,corresp,keywords")
EXCLUDE_ENV = ("tabular,tabular\\*,array,align,align\\*,equation,equation\\*,"
               "eqnarray,eqnarray\\*,aligned,algorithmic,split,cases,IEEEbiography")


def run(cmd: list[str], cwd: Path | None = None, out: Path | None = None) -> int:
    if out:
        with out.open("w", encoding="utf-8", errors="replace") as fh:
            return subprocess.run(cmd, cwd=cwd, stdout=fh,
                                  stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).returncode


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    old, new, outdir = Path(argv[0]), Path(argv[1]), Path(argv[2])
    outdir.mkdir(parents=True, exist_ok=True)

    # stage everything the document needs next to the diff
    for src in (new.parent,):
        for f in src.iterdir():
            if f.is_file() and f.suffix.lower() in (
                    ".cls", ".bst", ".bib", ".sty", ".pdf", ".png", ".jpg",
                    ".jpeg", ".eps", ".fd", ".map", ".pfb", ".tfm"):
                shutil.copy2(f, outdir / f.name)
        fig = src / "figures"
        if fig.is_dir():
            shutil.copytree(fig, outdir / "figures", dirs_exist_ok=True)

    diff_tex = outdir / "XAI-SDN-highlighted.tex"
    cmd = ["latexdiff",
           "--type=UNDERLINE",
           "--append-safecmd=" + SAFE_CMD,
           "--exclude-textcmd=" + SAFE_CMD,
           "--config", f"PICTUREENV=(?:picture|DIFnomarkup|tikzpicture|algorithm)[\\w\\d*@]*",
           "--math-markup=whole",
           "--graphics-markup=none",
           str(old), str(new)]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not proc.stdout.strip():
        print("latexdiff failed:", proc.stderr[:3000])
        return 1

    tex = proc.stdout
    # inject the yellow-highlight preamble immediately before \begin{document}
    tex = tex.replace("\\begin{document}", PREAMBLE + "\n\\begin{document}", 1)
    diff_tex.write_text(tex, encoding="utf-8")
    print(f"wrote {diff_tex} ({len(tex):,} chars)")

    log = outdir / "highlight_build.log"
    rc = run(["latexmk", "-pdf", "-interaction=nonstopmode", "-f",
              diff_tex.name], cwd=outdir, out=log)
    pdf = diff_tex.with_suffix(".pdf")
    print(f"latexmk rc={rc}; pdf exists={pdf.exists()}")
    if pdf.exists():
        try:
            import fitz
            d = fitz.open(pdf)
            text = "".join(p.get_text() for p in d)
            print(f"highlighted PDF: {d.page_count} pages, "
                  f"{len(text):,} chars, '??' count={text.count('??')}")
        except Exception as exc:
            print("pdf inspect failed:", exc)
    else:
        print("see", log)
    return 0 if pdf.exists() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
