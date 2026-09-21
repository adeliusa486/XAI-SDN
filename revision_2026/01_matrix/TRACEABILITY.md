# Traceability Matrix — Access-2026-39885

Manuscript: *XAI-SDN: An explainable entropy-guided machine learning framework for real-time DDoS detection in software defined networks*  
Decision 2026-09-20 · 7 reviewers · **47 atomic concerns**

> Generated from `reviewer_comments.json` and `experiment_registry.json`. Do not hand-edit — re-run `render_traceability.py`.

## Summary

- **Requires new experiments:** 25 of 47 concerns
- **Severity:** critical 20 · major 15 · minor 12
- **Type:** experiment 19 · figure 2 · figure+experiment 1 · reference 4 · structure 3 · text 10 · text+experiment 5 · text+figure 2 · text+reference 1
- **Owning phase:** P2 data/baseline 1 · P3 experiments 24 · P4 references 4 · P5 manuscript 16 · P6 figures 2
- **Status:** open 47

## Reviewer 1 — 7 concerns

| ID | Severity | Type | Experiments | Owning phase | Target location(s) | Status |
|---|---|---|---|---|---|---|
| **R1.1** | minor | figure | — | P6 figures | Fig. 2 graphic (entropyengine_standalone); Sec. V | open |
| **R1.2** | minor | text | — | P5 manuscript | Sec. V-A | open |
| **R1.3** | minor | text | — | P5 manuscript | Eq. (4); Sec. IV-C; new notation table | open |
| **R1.4** | major | experiment | E1, E9 | P3 experiments | Table 8; Sec. VII-B; new experimental-environment table | open |
| **R1.5** | major | experiment | E1, E9 | P3 experiments | Table 8; Sec. VII-B | open |
| **R1.6** | major | experiment | E1, E9 | P3 experiments | Table 8; Sec. VI-C; Sec. VII-B | open |
| **R1.7** | minor | text | — | P5 manuscript | whole manuscript | open |

> **R1.1 — verified against the submitted PDF.** The literal string 'Eq. (??)' is rendered INSIDE the graphic entropyengine_standalone.pdf (Fig. 2), produced by \node{Eq.~(\ref{eq:rolling}) Applied Per Feature} in a standalone .tex with no eq:rolling label. Not a body-text defect.

> **R1.2 — verified against the submitted PDF.** Sec. V-A (Entropy feature extraction, tex L209-L233) contains 'a bunch of', 'that means', 'things like', 'The key to making the system fast', 'we can update the calculation'.

> **R1.3 — verified against the submitted PDF.** Eq. (4) uses \mathrm{Acc} inside an aligned block with \tfrac; spacing is cramped and the symbol is never defined in a notation table.

> **R1.4 — verified against the submitted PDF.** Table 8 caption states the asymmetry in full capitals; the DNN train-time cell is '---'. No hardware specification appears anywhere in the paper.

## Reviewer 2 — 9 concerns

| ID | Severity | Type | Experiments | Owning phase | Target location(s) | Status |
|---|---|---|---|---|---|---|
| **R2.1** | critical | text+experiment | E2, E3, E4 | P3 experiments | Abstract; Sec. VI-B; Sec. VIII; Sec. IX; new cross-dataset section; new deployment section | open |
| **R2.2** | critical | text+experiment | E5 | P3 experiments | Sec. VII-D (leakage ablation); Sec. VIII-B; Abstract | open |
| **R2.3** | major | text | — | P5 manuscript | Table 9; Sec. VII-E | open |
| **R2.4** | major | text+experiment | E10 | P3 experiments | Abstract; Sec. III-B; Algorithm 1; Table 6; Sec. VI-C; Sec. VIII-A | open |
| **R2.5** | major | text | — | P5 manuscript | Sec. V-A; Sec. II-A | open |
| **R2.6** | major | text+experiment | E11 | P3 experiments | Abstract; Sec. II-A; Sec. IV; Sec. V-A; Fig. 2 caption; Table 1 | open |
| **R2.7** | major | text | — | P5 manuscript | Sec. VI-B; Sec. VI-F; Sec. VII-A; Sec. VII-B; Sec. VII-D | open |
| **R2.8** | critical | text+figure | — | P5 manuscript | Abstract; Sec. I; Sec. V-A; Sec. VI-A; Fig. 2; Fig. 5 caption; Fig. 6; Fig. 9 caption; whole… | open |
| **R2.9** | minor | reference | — | P4 references | Sec. II; Sec. VIII | open |

> **R2.1 — strategy.** Do both: provide the cross-dataset, cross-environment and control-plane deployment experiments the reviewer names as the price of the claims, AND moderate the language regardless.

> **R2.7 — adjudication.** The manuscript contains no 'two-stage design', no 'edge-aware attention' and no 'post-fault/recovery transient' discussion; these appear to belong to a different manuscript. Respond politely, note the discrepancy, and apply the underlying principle in full: every mechanism-level explanation not isolated by an ablation is rehedged.

> **R2.8 — verified against the submitted PDF.** 28 informal-English hits located by line. '(au)' confirmed inside fig_sensitivity.pdf. 'UAV' does not appear in the submitted text - treat as a generic acronym-consistency request. t-SNE caption and radar-chart caption both overstate their evidential weight.

## Reviewer 3 — 5 concerns

| ID | Severity | Type | Experiments | Owning phase | Target location(s) | Status |
|---|---|---|---|---|---|---|
| **R3.1** | critical | experiment | E5 | P3 experiments | new Sec. VII leakage-audit subsection | open |
| **R3.2** | critical | experiment | E6 | P3 experiments | new Sec. VII collinearity/SHAP-stability subsection | open |
| **R3.3** | critical | experiment | E7, E4 | P3 experiments | new deployment section; Sec. VIII-A; Abstract | open |
| **R3.4** | critical | experiment | E8 | P3 experiments | new deployment section; Sec. IV (architecture) | open |
| **R3.5** | minor | reference | — | P4 references | Sec. II-C; Sec. VIII | open |

## Reviewer 4 — 3 concerns

| ID | Severity | Type | Experiments | Owning phase | Target location(s) | Status |
|---|---|---|---|---|---|---|
| **R4.1** | critical | experiment | E5, E2 | P3 experiments | new leakage-audit subsection; new cross-dataset subsection | open |
| **R4.2** | critical | experiment | E4, E9, E10 | P3 experiments | new deployment section; Table 8; new deep-baseline table | open |
| **R4.3** | major | text+reference | E9 | P5 manuscript | new Sec. VIII subsection | open |

> **R4.2 — adjudication.** Mininet/ONOS require hardware virtualization, which is disabled in firmware on the available machine (WSL2, Hyper-V and Docker all unavailable). Substitute a REAL OpenFlow 1.3 control plane built on os-ken (the maintained Ryu fork) driven by software switch agents over real TCP sockets. Describe it accurately in the paper as a controller-in-the-loop testbed with emulated switch agents; never call it Mininet. Disclose the substitution in the response letter.

## Reviewer 5 — 7 concerns

| ID | Severity | Type | Experiments | Owning phase | Target location(s) | Status |
|---|---|---|---|---|---|---|
| **R5.1** | minor | structure | — | P5 manuscript | Sec. II; Sec. III; Sec. V; Sec. VI; Sec. VII; Sec. VIII | open |
| **R5.2** | minor | structure | — | P5 manuscript | end of Sec. I; new Table | open |
| **R5.3** | major | text | — | P5 manuscript | Sec. V (experimental setup); new environment table | open |
| **R5.4** | minor | reference | — | P4 references | references.bib; Sec. II | open |
| **R5.5** | minor | reference | — | P4 references | references [23], [25], [31], [32] | open |
| **R5.6** | minor | structure | — | P5 manuscript | Fig. 7; Fig. 9; Table 5; Eq. (4); Eq. (5) | open |
| **R5.7** | minor | figure | — | P6 figures | all figures | open |

> **R5.5 — adjudication.** The submitted PDF contains only 32 references, so reference 34 does not exist. The preprint-flavoured entries are [23] Doshi-Velez & Kim, [25] Lundberg & Lee (SHAP), [31] Kumar et al., [32] You et al. All four are handled; the numbering discrepancy is explained in the response.

> **R5.6 — verified against the submitted PDF.** Never cited in body text: Fig. 7 (fig:ablation), Fig. 9 (fig:radar), Table 5 (tab:ablation), Eq. (4) (eq:metrics), Eq. (5) (eq:srcip).

## Reviewer 6 — 8 concerns

| ID | Severity | Type | Experiments | Owning phase | Target location(s) | Status |
|---|---|---|---|---|---|---|
| **R6.1** | critical | experiment | E5, E3, E2 | P3 experiments | new leakage-audit subsection; new cross-vector subsection; new cross-dataset subsection; Sec… | open |
| **R6.2** | critical | text | — | P5 manuscript | whole manuscript | open |
| **R6.3** | critical | experiment | E10, E4 | P3 experiments | Table 6; Algorithm 1; Sec. VI-C; new explanation-budget subsection; Sec. VIII-A | open |
| **R6.4** | critical | experiment | E11 | P3 experiments | Eq. (6); Sec. V-A; new numerical-stability subsection | open |
| **R6.5** | critical | experiment | E1, E9 | P3 experiments | Table 8; Table 4; Sec. VII-B | open |
| **R6.6** | major | experiment | E6 | P3 experiments | new collinearity subsection; Sec. V-B; Sec. VII-A | open |
| **R6.7** | critical | experiment | E12 | P3 experiments | Sec. VI-F; Fig. 6; new PR-curve figure; Table 3 | open |
| **R6.8** | critical | experiment | E5, E6 | P3 experiments | Sec. VII-D; new leakage-audit subsection | open |

> **R6.1 — strategy.** Answer with evidence, not rhetoric: a systematic leakage audit, cross-vector and cross-day evaluation inside CIC-DDoS2019, and genuine cross-dataset transfer to InSDN and CIC-IDS2017. Report degradation honestly where it occurs.

> **R6.3 — strategy.** The objection is correct. Convert it into a contribution: formally define the explanation trigger, measure the saturation it causes, then implement and measure episode-level deduplication, a token-bucket explanation budget and asynchronous off-path explanation, reporting a throughput-versus-coverage curve.

> **R6.4 — verified against the submitted PDF.** The implementation does not compute Eq. (6) as printed. features/entropy.py maintains S = sum(c*log2 c) and returns H = log2|W| - S/|W|. The manuscript equation and the executed code disagree; the drift study must target the executed form and the equation must be corrected.

> **R6.5 — strategy.** Run BOTH protocols for EVERY model: the full imbalanced temporal split, and an identically-applied subsampled protocol. Report them as separate tables, never mixed.

> **R6.8 — strategy.** The reviewer's alternative explanation is the more likely one. Test it directly: progressive removal curves and removal of the full objectively-flagged suspect set, plus redundancy quantification from E6.

## Reviewer 7 — 8 concerns

| ID | Severity | Type | Experiments | Owning phase | Target location(s) | Status |
|---|---|---|---|---|---|---|
| **R7.1** | critical | text | E10, E11, E4 | P5 manuscript | Sec. I (contributions); Table 1; Sec. II-D | open |
| **R7.2** | critical | text+experiment | E16 | P2 data/baseline | Sec. V-A; new feature-specification table; Eq. (7) | open |
| **R7.3** | major | text+figure | E11 | P5 manuscript | Sec. V-A; Fig. 2; Algorithm 1 | open |
| **R7.4** | critical | experiment | E3, E2 | P3 experiments | new cross-vector subsection; new cross-dataset subsection; Sec. VIII-B | open |
| **R7.5** | major | experiment | E13 | P3 experiments | new alert case-study subsection; new waterfall figure; new case-study table | open |
| **R7.6** | major | figure+experiment | E14 | P3 experiments | Sec. VI-E; Fig. 6; new ROC/PR figure | open |
| **R7.7** | major | text | — | P5 manuscript | Table 9; Sec. VII-E | open |
| **R7.8** | critical | experiment | E15 | P3 experiments | Sec. V-B; new protocol table; Sec. VI-A | open |

> **R7.1 — strategy.** The revision's genuine novelty is operational, not architectural: a numerically-characterised amortised-O(1) streaming entropy engine, a formally-defined and budgeted explanation trigger with a measured throughput-coverage frontier, and a controller-in-the-loop accounting of control-channel and TCAM cost. State that explicitly and support it with a positioning table.

> **R7.2 — verified against the submitted PDF.** features/pipeline.py:269 hard-codes "ttl": 64 for every flow record, so H_ttl is identically 0 and carries no information. CICFlowMeter CSVs contain no TTL column. H_tcp_flags is in fact the entropy of SYN_Flag_Count, and H_pkt_len / H_iat are computed from flow-level means rather than packet-level distributions.

> **R7.2 — strategy.** Confirm empirically that H_ttl is all-zero, replace it with source-port entropy (genuinely present and discriminative for randomised-source SYN floods), re-baseline, and give a complete feature specification with exact source columns and bin widths.

> **R7.6 — verified against the submitted PDF.** Sec. VI-E 'ROC analysis' contains prose but no figure whatsoever. Fig. 6 (fig_sensitivity) caption promises an ROC curve; the panels show accuracy vs threshold and accuracy vs window size.

## Experiment coverage

| Exp | Name | Answers | Runtime (min) | Depends on | Status |
|---|---|---|---|---|---|
| **E0** | Baseline reproduction of the submitted results | prerequisite for all | 25 | — | pending |
| **E16** | Feature-set correctness audit and repair | R7.2 | 30 | E0 | pending |
| **E5** | Leakage and dataset-artifact audit | R3.1, R4.1, R6.8, R2.2, R6.1 | 180 | E0, E16 | pending |
| **E6** | Collinearity, redundancy and SHAP attribution stability | R3.2, R6.6, R6.8 | 150 | E0, E16 | pending |
| **E12** | Precision-recall based threshold selection | R6.7, R1.3 | 45 | E0, E16 | pending |
| **E15** | Evaluation protocol formalisation | R7.8 | 120 | E0, E16 | pending |
| **E11** | Numerical stability of the rolling entropy engine | R6.4, R2.6, R7.3 | 90 | E16 | pending |
| **E3** | Cross-vector and cross-day generalization | R6.1, R7.4, R2.1 | 240 | E0, E16 | pending |
| **E2** | Cross-dataset validation with confidence intervals | R4.1, R2.1, R6.1, R7.4 | 180 | E0, E16 | pending |
| **E1** | Unified CPU-only classical baseline comparison | R1.4, R1.5, R1.6, R6.5 | 240 | E0, E16, E12 | pending |
| **E9** | Deep-learning baselines under the same CPU protocol | R4.2, R1.5, R6.5, R4.3 | 300 | E1 | pending |
| **E14** | ROC and precision-recall curves for all methods | R7.6, R6.7 | 20 | E1, E9 | pending |
| **E10** | Explanation trigger policy and budgeted SHAP throughput | R6.3, R2.4, R7.1 | 120 | E0, E16, E12 | pending |
| **E4** | Controller-in-the-loop OpenFlow deployment testbed | R4.2, R2.1, R3.3 | 240 | E10, E12 | pending |
| **E7** | OpenFlow control-channel capacity analysis | R3.3 | 120 | E4 | pending |
| **E8** | TCAM occupancy under aggregation and eviction policies | R3.4 | 90 | E4 | pending |
| **E13** | Individual alert case studies | R7.5 | 45 | E12, E6 | pending |

**Total estimated compute: 2235 minutes (~37.2 h).**

## Consistency checks

- Experiments referenced by a concern but not declared: **none**
- Experiments declared but answering no concern: **none**
- Concerns needing an experiment but mapped to none: **none**
