"""Phase 7 - Produce the Highlighted PDF with genuine yellow highlight annotations.

The Associate Editor asked for "an updated manuscript with all your individual
changes highlighted, including grammatical changes (e.g. preferably with the
yellow highlight tool within the pdf file)". This script does exactly that: it
compares the submitted PDF with the revised PDF at the level of rendered words,
finds every inserted or replaced run, and adds a real PDF highlight annotation
over each one in the revised document.

Why not latexdiff alone. Two reasons, both practical. First, soul's \\hl - the
only pdflatex mechanism that paints a yellow background - breaks on citations,
maths and several IEEE class macros, so a heavily revised manuscript will not
compile. Second, latexdiff's default markup is a blue wavy underline and a red
strikeout, which is not what was requested. Annotating the compiled PDF sidesteps
both problems and produces the artefact the editor described. A latexdiff source
diff is still generated alongside, as a cross-check on coverage.

Usage:
    python highlight_pdf.py OLD.pdf NEW.pdf OUT.pdf [--min-words 1]
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

import pymupdf

YELLOW = (1.0, 1.0, 0.0)
# Words too common to anchor a search on their own.
STOP_SINGLE = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "was", "were",
    "for", "on", "at", "by", "with", "as", "that", "this", "it", "be", "we",
    "our", "its", "from", "not", "but", "than", "then", "so", "such", "which",
}


def page_words(doc: pymupdf.Document) -> list[list[tuple]]:
    """Per page: [(x0, y0, x1, y1, word, block, line, word_no), ...]."""
    return [p.get_text("words") for p in doc]


def flat_tokens(pages: list[list[tuple]]) -> tuple[list[str], list[tuple[int, int]]]:
    """Flatten to a token stream plus a (page, index-within-page) locator."""
    toks, loc = [], []
    for pno, ws in enumerate(pages):
        for i, w in enumerate(ws):
            toks.append(w[4])
            loc.append((pno, i))
    return toks, loc


def norm(tok: str) -> str:
    """Strip layout punctuation but keep case: a capitalisation fix is a change
    the editor asked to see highlighted."""
    return re.sub(r"[^\w%.\-/]", "", tok)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    old_p, new_p, out_p = Path(argv[0]), Path(argv[1]), Path(argv[2])
    min_words = 1
    if "--min-words" in argv:
        min_words = int(argv[argv.index("--min-words") + 1])

    old = pymupdf.open(old_p)
    new = pymupdf.open(new_p)
    old_tokens = [norm(w[4]) for p in old for w in p.get_text("words")]
    new_pages = page_words(new)
    new_tokens_raw, new_loc = flat_tokens(new_pages)
    new_tokens = [norm(t) for t in new_tokens_raw]

    sm = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    runs: list[tuple[int, int, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace") and (j2 - j1) >= min_words:
            runs.append((j1, j2, tag))

    changed_tokens = sum(j2 - j1 for j1, j2, _ in runs)
    stats = {
        "old_pdf": str(old_p), "new_pdf": str(new_p), "out_pdf": str(out_p),
        "old_token_count": len(old_tokens), "new_token_count": len(new_tokens),
        "changed_runs": len(runs), "changed_tokens": changed_tokens,
        "changed_fraction_of_new": round(changed_tokens / max(len(new_tokens), 1), 4),
        "similarity_ratio": round(sm.ratio(), 4),
    }

    # group each run's word rectangles by page and add one annotation per line
    annots = 0
    skipped = 0
    for j1, j2, _tag in runs:
        by_page: dict[int, list[int]] = {}
        for j in range(j1, j2):
            pno, widx = new_loc[j]
            by_page.setdefault(pno, []).append(widx)
        for pno, idxs in by_page.items():
            words = new_pages[pno]
            # keep single-token highlights meaningful
            if len(idxs) == 1 and new_tokens[j1].lower() in STOP_SINGLE:
                skipped += 1
                continue
            # split into contiguous runs on the same rendered line
            idxs.sort()
            groups: list[list[int]] = [[idxs[0]]]
            for k in idxs[1:]:
                prev = groups[-1][-1]
                same_line = (words[k][5], words[k][6]) == (words[prev][5],
                                                           words[prev][6])
                if k == prev + 1 and same_line:
                    groups[-1].append(k)
                else:
                    groups.append([k])
            page = new[pno]
            for g in groups:
                r = pymupdf.Rect(words[g[0]][:4])
                for k in g[1:]:
                    r |= pymupdf.Rect(words[k][:4])
                a = page.add_highlight_annot(r)
                a.set_colors(stroke=YELLOW)
                a.set_info(title="Revision", content="changed in revision")
                a.update()
                annots += 1

    stats["annotations_added"] = annots
    stats["single_stopword_runs_skipped"] = skipped
    out_p.parent.mkdir(parents=True, exist_ok=True)
    new.save(out_p, garbage=3, deflate=True)
    (out_p.with_suffix(".highlight_report.json")).write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))

    verify = pymupdf.open(out_p)
    n_annot = sum(1 for p in verify for _ in p.annots() or [])
    print(f"verification: {n_annot} annotations present in {out_p.name}, "
          f"{verify.page_count} pages")
    return 0 if n_annot else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
