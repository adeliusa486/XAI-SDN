"""Render TRACEABILITY.md from reviewer_comments.json + experiment_registry.json.

Re-run this whenever a comment's status changes so the human-readable table never
drifts from the machine-readable matrix.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

MATRIX = Path(__file__).resolve().parents[2] / "01_matrix"
comments = json.loads((MATRIX / "reviewer_comments.json").read_text(encoding="utf-8"))
registry = json.loads((MATRIX / "experiment_registry.json").read_text(encoding="utf-8"))

exp_by_id = {e["id"]: e for e in registry["experiments"]}

PHASE_NAME = {
    2: "P2 data/baseline",
    3: "P3 experiments",
    4: "P4 references",
    5: "P5 manuscript",
    6: "P6 figures",
}

lines: list[str] = []
a = lines.append

a("# Traceability Matrix — Access-2026-39885")
a("")
a(f"Manuscript: *{comments['title']}*  ")
a(f"Decision {comments['decision_date']} · {comments['n_reviewers']} reviewers · "
  f"**{comments['n_concerns']} atomic concerns**")
a("")
a("> Generated from `reviewer_comments.json` and `experiment_registry.json`. "
  "Do not hand-edit — re-run `render_traceability.py`.")
a("")

# ---- summary ----------------------------------------------------------------
by_sev = Counter(c["severity"] for c in comments["comments"])
by_type = Counter(c["type"] for c in comments["comments"])
by_phase = Counter(c["phase"] for c in comments["comments"])
by_status = Counter(c["status"] for c in comments["comments"])
n_exp = sum(1 for c in comments["comments"] if c["requires_experiment"])

a("## Summary")
a("")
a(f"- **Requires new experiments:** {n_exp} of {len(comments['comments'])} concerns")
a(f"- **Severity:** " + " · ".join(f"{k} {v}" for k, v in sorted(by_sev.items())))
a(f"- **Type:** " + " · ".join(f"{k} {v}" for k, v in sorted(by_type.items())))
a(f"- **Owning phase:** " + " · ".join(f"{PHASE_NAME.get(k, k)} {v}"
                                        for k, v in sorted(by_phase.items())))
a(f"- **Status:** " + " · ".join(f"{k} {v}" for k, v in sorted(by_status.items())))
a("")

# ---- per-reviewer tables ----------------------------------------------------
for r in range(1, comments["n_reviewers"] + 1):
    rows = [c for c in comments["comments"] if c["reviewer"] == r]
    if not rows:
        continue
    a(f"## Reviewer {r} — {len(rows)} concerns")
    a("")
    a("| ID | Severity | Type | Experiments | Owning phase | Target location(s) | Status |")
    a("|---|---|---|---|---|---|---|")
    for c in rows:
        exps = ", ".join(c["experiments"]) if c["experiments"] else "—"
        loc = "; ".join(c["target_locations"])
        if len(loc) > 95:
            loc = loc[:92] + "…"
        a(f"| **{c['id']}** | {c['severity']} | {c['type']} | {exps} | "
          f"{PHASE_NAME.get(c['phase'], c['phase'])} | {loc} | {c['status']} |")
    a("")
    # concerns with an adjudication or a verified finding get expanded
    for c in rows:
        if c.get("adjudication"):
            a(f"> **{c['id']} — adjudication.** {c['adjudication']}")
            a("")
        if c.get("verified_finding"):
            a(f"> **{c['id']} — verified against the submitted PDF.** {c['verified_finding']}")
            a("")
        if c.get("resolution_strategy"):
            a(f"> **{c['id']} — strategy.** {c['resolution_strategy']}")
            a("")

# ---- experiment coverage ----------------------------------------------------
a("## Experiment coverage")
a("")
a("| Exp | Name | Answers | Runtime (min) | Depends on | Status |")
a("|---|---|---|---|---|---|")
for e in registry["experiments"]:
    a(f"| **{e['id']}** | {e['name']} | {', '.join(e['answers'])} | "
      f"{e.get('runtime_estimate_min', '?')} | {', '.join(e['depends_on']) or '—'} | "
      f"{e['status']} |")
a("")
total = sum(e.get("runtime_estimate_min", 0) for e in registry["experiments"])
a(f"**Total estimated compute: {total} minutes (~{total/60:.1f} h).**")
a("")

# ---- orphan check -----------------------------------------------------------
referenced = {x for c in comments["comments"] for x in c["experiments"]}
declared = set(exp_by_id)
a("## Consistency checks")
a("")
missing = sorted(referenced - declared - {"prerequisite for all"})
unused = sorted(declared - referenced - {"E0"})
a(f"- Experiments referenced by a concern but not declared: "
  f"{', '.join(missing) if missing else '**none**'}")
a(f"- Experiments declared but answering no concern: "
  f"{', '.join(unused) if unused else '**none**'}")
uncovered = [c["id"] for c in comments["comments"]
             if c["requires_experiment"] and not c["experiments"]]
a(f"- Concerns needing an experiment but mapped to none: "
  f"{', '.join(uncovered) if uncovered else '**none**'}")
a("")

(MATRIX / "TRACEABILITY.md").write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {MATRIX / 'TRACEABILITY.md'}  ({len(lines)} lines)")
print(f"concerns={len(comments['comments'])}  needing_experiments={n_exp}  "
      f"experiments={len(registry['experiments'])}  est_compute_h={total/60:.1f}")
print("missing:", missing or "none", "| unused:", unused or "none",
      "| uncovered:", uncovered or "none")
