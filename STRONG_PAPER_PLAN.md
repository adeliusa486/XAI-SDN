# STRONG PAPER PLAN — Making XAI-SDN Acceptance-Ready for a Q1 Journal

**Companion to `IMPROVEMENT_PLAN.md`** (which holds the phase-by-phase experiment commands). This document is the complete strategy: everything the paper still needs, organized into seven pillars, with the reviewer objections each pillar neutralizes, a submission-package checklist, and a realistic timeline.

**Current state: 6.5/10.** Presentation is done (9/10). Every remaining point comes from evidence, rigor, and positioning — not writing.

---

## Pillar A — Evaluation breadth (the acceptance gatekeeper)

*Neutralizes: "single dataset, single attack type" — the #1 rejection reason for IDS papers at Computers & Security, Computer Networks, and TNSM.*

| # | Task | Output for the paper |
|---|---|---|
| A1 | Evaluate all major CIC-DDoS2019 partitions: **UDP, LDAP, MSSQL, NetBIOS, Portmap** (same 88-dim pipeline, same hyperparameters, temporal split) | Per-partition results table |
| A2 | **Cross-partition generalization matrix**: train on one attack vector, test on every other | Train×test macro-F1 heatmap figure — reviewers value this far more than another 99.99% cell |
| A3 | Second dataset: **InSDN** (already cited as [18]) — real SDN testbed traffic, perfect topical fit | External-validity table |
| A4 | **Cross-dataset transfer**: CIC-DDoS2019 → InSDN and reverse; report the degradation honestly | Transfer table + one honest paragraph; candor here buys credibility everywhere else |
| A5 | Mixed-prevalence stress test: rebalance test sets to 1%, 10%, 50% attack prevalence and report FPR/FNR at each | Prevalence-sensitivity figure; converts the 99.14%-prevalence caveat into a contribution |

## Pillar B — Statistical and measurement rigor

*Neutralizes: "results are single-run points on a saturated benchmark" and the repo/paper mismatch.*

| # | Task | Output |
|---|---|---|
| B1 | Re-run the full **10-seed** campaign (ablation + stratified FPR) so archived artifacts match every claim; publish per-seed JSONs | Fixes the p=0.0020 traceability gap |
| B2 | Report **95% confidence intervals** (bootstrap over test flows) for accuracy, macro F1, FPR — not just point estimates | CI columns in the headline table |
| B3 | Replace hand-placed ROC coordinates with **computed `roc_curve()` outputs**; add a **precision–recall curve** (mandatory under 0.86% minority class — PR-AUC is the honest metric here) | Real ROC + new PR figure |
| B4 | **McNemar's test** between XAI-SDN and each baseline on the shared test split (the correct paired test for classifier comparison on one split) | p-values in the comparison table |
| B5 | **Calibration analysis**: reliability diagram + Brier score for the RF vote fractions — the paper claims τ is tunable per SOC budget, so show the scores are calibrated enough to tune | Calibration figure; directly supports the deployment claims |
| B6 | Fix n_train drift (2,514,860 vs 2,514,862) and the DNN train-time gap | Clean consistency |

## Pillar C — Explanation quality evaluation (the differentiator most competitors lack)

*Neutralizes: "the paper claims explainability but never measures whether the explanations are any good" — an increasingly standard Q1 objection since Warnecke et al. [30].*

| # | Task | Output |
|---|---|---|
| C1 | **Fidelity (deletion test)**: remove top-k SHAP features per prediction, measure score drop vs. random-k removal | Deletion curve figure — proves attributions track real model evidence |
| C2 | **Stability**: SHAP attribution variance across the 10 seeds and across near-duplicate flows | Stability table |
| C3 | **Sparsity/actionability**: fraction of alerts where top-3 features carry >50% of total attribution (operators read 3 features, not 88) | One statistic + sentence |
| C4 | SHAP **runtime distribution** (p50/p95/p99), not just the mean 0.5 ms | Latency percentile table row |
| C5 | Case-study subsection: walk one real FP and one real FN through their SHAP payloads, showing what an operator would conclude | Half-page qualitative subsection with an alert-payload figure |

## Pillar D — Adversarial robustness (lifts 8.5 → 9)

*Neutralizes: "entropy features are trivially evadable by rotating spoofed IPs" — a question at least one reviewer will ask.*

| # | Task | Output |
|---|---|---|
| D1 | Implement entropy-aware evasion generator (IP rotation, IAT jitter, benign mimicry) parameterized by attacker budget | `attacks/entropy_evasion.py` |
| D2 | Recall-vs-evasion-budget curve; identify the crossover where evasion cost exceeds attack utility | Cost-to-evade figure — reframes evadability as a quantified attacker cost |
| D3 | SHAP drift under attack: which feature families keep discriminating | Figure/table tying Pillars C and D together |
| D4 | Optional: adversarial retraining, before/after | One table row |

## Pillar E — Systems evidence (alternative or complement to D)

*Neutralizes: "offline replay only; no evidence it works in a real controller."*

| # | Task | Output |
|---|---|---|
| E1 | Live Mininet + Ryu deployment (repo already has `sdn/controller/xai_sdn_app.py`, `sdn/topology/mininet_topo.py`; needs Linux/WSL2) | "Live deployment evaluation" subsection |
| E2 | Measure at the controller: poll→alert latency distribution, controller CPU/RAM under flood, time-to-first-detection, packet_in survival rate | 1 table + 1 figure |
| E3 | Demonstrate the mitigation loop: auto-install a drop rule on detection, measure time-to-mitigation | One end-to-end timeline figure — turns the dashed "policy feedback" arrow of Fig. 1 into measured reality |

## Pillar F — Positioning, writing, and reviewer management

*Neutralizes: "insufficient comparison with recent work" and citation-padding flags; maximizes the value of work already done.*

| # | Task | Detail |
|---|---|---|
| F1 | **Recent-SOTA comparison table**: survey 2023–2026 Q1 papers on SDN DDoS detection (search: Computers & Security, Computer Networks, TNSM, IEEE Access); tabulate dataset, split methodology, accuracy/F1, latency, XAI support — most will lack per-flow XAI and honest split reporting, which is exactly your niche. **Only cite verified papers.** | New related-work table |
| F2 | Replace the three tangential co-author citations (`toqeer2026climate`, `jan2025blockchain`, `akarma2026agents`) with genuinely relevant ones found in F1 | Removes desk-review red flag |
| F3 | Add explicit **research questions** (RQ1: does entropy augmentation add detection value? RQ2: can per-flow explanation run at line rate? RQ3: how split-sensitive are the results?) and answer each in the conclusion | Sharpened framing reviewers can tick off |
| F4 | **Anticipated-objections file** (`REBUTTAL_NOTES.md`, private): draft answers to the six predictable objections (saturated benchmark, RF not novel, SHAP cost, evadability, single dataset, FPR variance) so the response letter takes hours, not weeks | Faster, stronger major-revision response |
| F5 | Cover letter: disclose conference acceptance/withdrawal; explicitly claim the honest-reporting methodology (prevalence-aware SHAP latency + split-sensitivity) as a contribution | Sets reviewer expectations correctly |
| F6 | Suggest 4–5 reviewers (authors of cited methodological papers, no conflicts); exclude co-author collaborators | Journal form field |

## Pillar G — Reproducibility and artifact excellence (cheap points, real credibility)

| # | Task | Output |
|---|---|---|
| G1 | Archive the exact submission-state repo on **Zenodo** → citable DOI; reference it in Data Availability alongside GitHub | Permanent artifact DOI |
| G2 | **One-command reproduction**: `docker compose run reproduce` regenerates every table/figure from raw CSVs; document runtime | README badge + reviewer delight |
| G3 | `results/` → paper traceability map: table/figure number → generating script → artifact JSON | Table in REPO_REPORT.md |
| G4 | Tag `v2.0-submission`; freeze `requirements-lock.txt`; record hardware in every manifest | Version hygiene |

---

## Submission package checklist (final gate)

- [ ] Manuscript (review format) + highlights file (3–5 bullets, ≤85 chars each) + graphical abstract (reuse Fig. 1)
- [ ] Cover letter (F5) + suggested reviewers (F6)
- [ ] Conference withdrawal confirmation in hand (email PDF)
- [ ] Zenodo DOI live; GitHub tagged; README updated
- [ ] Every number in the paper traceable to a committed artifact (G3 map complete)
- [ ] All co-authors have approved the final PDF and author order; corresponding-author email verified
- [ ] Declarations: CRediT final, no-conflict statement, funding statement (add if any)
- [ ] arXiv preprint posted (allowed by Elsevier and IEEE)

## Score trajectory and effort

| Milestone | Rating | Cumulative effort |
|---|---|---|
| Today (writing/presentation complete, 33 refs) | 6.5 | — |
| + Pillar B (rigor debts: 10-seed, real ROC/PR, CIs) | 7.0 | ~4 days |
| + Pillar A (multi-partition + InSDN + transfer) | 8.0 | ~2.5 weeks |
| + Pillar C (explanation quality) | 8.5 | ~3 weeks |
| + Pillar D **or** E (adversarial or live SDN) | **9.0** | ~4–5 weeks |
| + Pillars F–G polish | 9.0, submission-ready | +3 days |

**Priority order if time is short:** B → A → C → F → G, with D/E as the stretch goal. B and A alone move you from "coin flip" to "likely major-revision-then-accept."

## The six predictable reviewer objections — and what answers them

| Objection | Answered by |
|---|---|
| "Benchmark is saturated; 99.99% is meaningless" | A2 cross-partition + A5 prevalence stress + B3 PR-AUC (none of these saturate) |
| "Random Forest + SHAP is not novel" | F3 RQs + C1–C5: the contribution is the *measured* explainability-throughput-honesty triad, not the classifier |
| "SHAP is too slow for line rate" | C4 percentiles + the alert-gated policy + E2 live measurements |
| "Entropy features are easily evaded" | D1–D3 cost-to-evade quantification |
| "Only one dataset" | A3–A4 InSDN + transfer analysis |
| "Results won't reproduce" | B1 artifacts + G1–G3 one-command reproduction with DOI |
