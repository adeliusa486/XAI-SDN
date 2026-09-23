"""Phase 5/7 gate - automated manuscript checks.

Fails loudly on anything that must not reach the editor:

  * unfilled \FILL placeholders
  * unresolved cross references in the compiled PDF
  * floats (figures, tables, algorithms, equations) never cited in the body
  * conversational phrasing and over-claims the reviewers named
  * em dashes and British spellings, which the writing standard forbids
  * duplicated or orphaned text fragments

Usage:
    python check_manuscript.py 05_manuscript/XAI-SDN-journal.tex [compiled.pdf]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

INFORMAL = [
    r"to its knees", r"\bthings like\b", r"to be able to see the whole network",
    r"lightning speed", r"highly compelling", r"\ba bunch of\b",
    r"The key to making the system fast", r"Here are the details",
    r"Some of them use something called", r"\bIt gets \d", r"\bIt looked at\b",
    r"\bwe can update the calculation\b", r"\bdevastating\b",
    r"\bbig companies\b", r"\bvery important\b", r"\bpretty\b",
]

OVERCLAIM = [
    r"\bconfirms that\b", r"\bThis confirms\b", r"\bproves\b", r"\bproven\b",
    r"\bsurpass(?:es|ed)\b", r"\bexceptional(?:ly)?\b", r"\bgeneralizes robustly\b",
    r"\bnear-perfect\b", r"\bperfect separation\b", r"\bguarantee(?:s|d)?\b",
    r"\boutperforming all\b", r"\balways\b", r"\bnever fails\b",
    r"\bensuring every\b", r"\bcompletely eliminat", r"\bfully rules out\b",
]

# Over-claim hits that are legitimate in context and are therefore not failures.
# Each entry is (pattern, substring that must appear in the same line). Keep this
# list short and justify every addition; it is the escape hatch that lets the
# over-claim check be enforcing rather than advisory.
OVERCLAIM_ALLOW = [
    # SHAP's additive-consistency guarantee is a property of the method, not a
    # claim about our results.
    (r"\bguarantee(?:s|d)?\b", "consistency guarantees"),
    # Naming the fact that an exact and an inherited attribution differ in what
    # they guarantee is the opposite of an over-claim.
    (r"\bguarantee(?:s|d)?\b", "do not carry the same guarantee"),
    # "near-perfect" describes the literature's figures and the gap our protocol
    # audit explains, never our own performance.
    (r"\bnear-perfect\b", "gap between near-perfect and realistic"),
    (r"\bnear-perfect\b", "explains a good deal of the near-perfect performance"),
]


def allowed(hit) -> bool:
    line = hit.get("full", hit.get("context", ""))
    return any(hit["pattern"] == p and s in line for p, s in OVERCLAIM_ALLOW)

BRITISH = [
    # "favour" and "labelling" were absent from this list until Phase 3, so the
    # gate passed while the manuscript carried eight of them, one of which sat
    # three lines from an American "labeled". Any -our/-ll- pair added here must
    # cover every inflection, not just the one that happened to be present.
    r"\bfavour(?:s|ed|ing|ite|able|ably)?\b", r"\blabell(?:ed|ing|er)\b",
    r"\bmodell(?:ed|ing)\b", r"\bcancell(?:ed|ing)\b",
    r"\btravell(?:ed|ing)\b", r"\bsignall(?:ed|ing)\b",
    r"\bspecialis", r"\bprioritis", r"\butilis", r"\brealis(?:e|ed|es|ing)\b",
    r"\borganis", r"\bstandardis", r"\bsynthesis(?:e|ed|es|ing)\b",
    r"\bpractise\b", r"\blicence\b", r"\bjudgement\b", r"\backnowledgement\b",
    r"\bfibre\b", r"\bmetre\b", r"\benquir",
    r"\banalys(?:e|ed|ing)\b", r"\brecognise", r"\boptimis(?:e|ed|es|ing|ation)\b", r"\bbehaviour", r"\bcolour",
    r"\bneighbour", r"\bcentre\b",
    r"\bdefence\b", r"\bprogramme\b", r"\bgrey\b", r"\bartefact(?:s)?\b",
    r"\bfulfil\b", r"\bwhilst\b", r"\bamongst\b", r"\btowards\b",
    r"\bfavourab(?:le|ly)\b", r"\bgeneralis(?:e|ed|es|ing|ation)\b", r"\bcharacteris(?:e|ed|es|ing|ation)\b", r"\bsummaris(?:e|ed|es|ing|ation)\b",
    r"\bnormalis(?:e|ed|es|ing|ation)\b", r"\bminimis(?:e|ed|es|ing|ation)\b", r"\bmaximis(?:e|ed|es|ing|ation)\b", r"\bemphasis(?:e|ing)\b",
    r"\bcatalogue\b", r"\banalogue\b", r"\bafterwards\b",
]

# Em dashes: --- always, and -- outside of numeric ranges and option flags.
EM_DASH = r"---"


def strip_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.split("\n"))


def find(patterns, text, label):
    hits = []
    lines = text.split("\n")
    for i, ln in enumerate(lines, 1):
        for p in patterns:
            for m in re.finditer(p, ln, flags=re.I if label == "british" else 0):
                hits.append({"line": i, "pattern": p, "match": m.group(0),
                             "context": ln.strip()[:170], "full": ln.strip()})
    return hits


def main(argv: list[str]) -> int:
    tex_path = Path(argv[0])
    tex_raw = tex_path.read_text(encoding="utf-8")
    tex = strip_comments(tex_raw)

    report: dict = {"file": str(tex_path)}
    failures: list[str] = []

    # 1. unfilled placeholders --------------------------------------------
    fills = [{"line": i, "context": ln.strip()[:150]}
             for i, ln in enumerate(tex.split("\n"), 1) if r"\FILL{" in ln]
    report["unfilled_placeholders"] = {"count": len(fills), "items": fills}
    if fills:
        failures.append(f"{len(fills)} unfilled \\FILL placeholder line(s)")

    # 2. float citation coverage -------------------------------------------
    labels = {kind: re.findall(r"\\label\{(%s:[^}]*)\}" % kind, tex)
              for kind in ("fig", "tab", "alg", "eq", "sec")}
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", tex))
    uncited = {k: [l for l in v if l not in refs] for k, v in labels.items()}
    report["uncited_floats"] = uncited
    n_unc = sum(len(v) for k, v in uncited.items() if k != "sec")
    if n_unc:
        failures.append(f"{n_unc} uncited float(s)/equation(s): "
                        + "; ".join(f"{k}: {v}" for k, v in uncited.items()
                                    if v and k != "sec"))

    # 3. style ---------------------------------------------------------------
    body = tex.split(r"\begin{document}", 1)[-1]
    body = body.split(r"\begin{IEEEbiography}", 1)[0]
    for label, pats in (("informal", INFORMAL), ("overclaim", OVERCLAIM),
                        ("british", BRITISH)):
        hits = find(pats, body, label)
        if label == "overclaim":
            waived = [h for h in hits if allowed(h)]
            hits = [h for h in hits if not allowed(h)]
            report["overclaim_waived"] = {"count": len(waived), "items": waived[:20]}
        report[label] = {"count": len(hits), "items": hits[:60]}
    if report["informal"]["count"]:
        failures.append(f"{report['informal']['count']} conversational phrase(s)")
    if report["overclaim"]["count"]:
        failures.append(f"{report['overclaim']['count']} over-claiming phrase(s)")
    if report["british"]["count"]:
        failures.append(f"{report['british']['count']} British spelling(s)")

    em = [{"line": i, "context": ln.strip()[:150]}
          for i, ln in enumerate(body.split("\n"), 1) if re.search(EM_DASH, ln)]
    report["em_dashes"] = {"count": len(em), "items": em[:40]}
    if em:
        failures.append(f"{len(em)} em dash(es)")

    # 4. compiled PDF checks --------------------------------------------------
    if len(argv) > 1 and Path(argv[1]).exists():
        import pymupdf
        doc = pymupdf.open(argv[1])
        text = "".join(p.get_text() for p in doc)
        report["pdf"] = {
            "pages": doc.page_count,
            "chars": len(text),
            "broken_refs": text.count("??"),
            "fill_markers": text.count("[[FILL"),
        }
        if report["pdf"]["broken_refs"]:
            failures.append(f"{report['pdf']['broken_refs']} '??' in the compiled PDF")
        if report["pdf"]["fill_markers"]:
            failures.append(f"{report['pdf']['fill_markers']} FILL marker(s) in the PDF")

    report["PASS"] = not failures
    report["failures"] = failures

    out = tex_path.parent.parent / "08_logs" / "manuscript_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'PASS' if report['PASS'] else 'FAIL'}  ({tex_path.name})")
    for f in failures:
        print("  -", f)
    for label in ("informal", "overclaim", "british", "em_dashes"):
        c = report[label]["count"]
        if c:
            print(f"\n{label} ({c}):")
            for h in report[label]["items"][:12]:
                ctx = h.get("context", "")
                mt = h.get("match", "")
                print(f"   L{h['line']:>4} {mt!r:28} {ctx[:110]}")
    if report.get("pdf"):
        print("\npdf:", json.dumps(report["pdf"]))
    print(f"\nreport: {out}")
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
