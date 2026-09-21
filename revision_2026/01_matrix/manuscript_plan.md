# Target Structure of the Revised Manuscript

Legend: **[NEW]** section did not exist · **[REWRITE]** substantially rewritten ·
**[EDIT]** targeted corrections · **[KEEP]** retained largely as-is

The submitted paper ran 11 pages with 9 figures, 9 tables, 7 equations and 1 algorithm.
The revision adds substantial new evidence, so the float budget is managed explicitly:
low-value floats are removed to make room for the ones reviewers demanded.

---

## Front matter

| Element | Action | Comments addressed |
|---|---|---|
| Title | KEEP | — |
| Abstract | REWRITE — remove "correct 99.9987% of the time", "looked at 60,606 flows"; state the explanation trigger; qualify the complexity claim; report cross-dataset and deployment evidence; drop absolute robustness language | R1.2, R2.1, R2.4, R2.6, R2.8, R6.2 |
| Keywords | EDIT — add cross-dataset validation, control-plane deployment | — |

## I. Introduction

| Element | Action | Comments |
|---|---|---|
| Opening paragraphs | REWRITE — remove "to be able to see the whole network", "Some of them use something called", "There are also attacks" | R1.2, R2.8, R6.2 |
| Contributions list | REWRITE — reframe around the revision's genuine operational novelty: a numerically characterised amortised-O(1) streaming entropy engine, a formally defined and budgeted explanation trigger with a measured throughput-coverage frontier, a controller-in-the-loop accounting of control-channel and TCAM cost, and a leakage audit with cross-dataset transfer | R7.1, R2.1 |
| **Table: Notation, symbols and acronyms** | **[NEW]** at end of §I | R5.2, R1.3 |
| Roadmap paragraph | EDIT — reflect the new section list | R5.1 |

## II. Related work

| Element | Action | Comments |
|---|---|---|
| **Section preamble** | **[NEW]** one paragraph before §II-A | R5.1 |
| II-A Entropy-based detection | EDIT — hedge the entropy/traffic relationship; qualify the O(1) claim | R2.5, R2.6 |
| II-B ML for SDN DDoS | EDIT — add recent 2023–2026 work | R5.4 |
| II-C XAI for intrusion detection | EDIT — integrate the Bayesian uncertainty-quantification reference where detector confidence is discussed | R3.5, R5.4 |
| **II-D Nonlinear dynamic indices for abnormal-state detection** | **[NEW]** — integrate the Tensor Poincaré plot index work substantively as a complementary feature paradigm, not a drive-by citation | R2.9 |
| Table 1 (qualitative comparison) | REWRITE — add the positioning axes that carry the novelty claim; correct the "0.5 ms per alert" cell to the measured budgeted value | R7.1, R2.4 |

## III. Threat model and problem formulation

| Element | Action | Comments |
|---|---|---|
| **Section preamble** | **[NEW]** | R5.1 |
| III-A Threat model | KEEP with light edits | — |
| III-B Problem formulation | REWRITE — add the formal definitions of *flow*, *positive prediction*, *alert*, *explanation trigger* and *per-decision explanation*, used identically everywhere thereafter | R2.4 |
| Eq. (1) decision rule | EDIT — make the explanation trigger a separate numbered condition | R2.4 |

## IV. XAI-SDN framework and architecture

| Element | Action | Comments |
|---|---|---|
| Architecture prose | EDIT — remove the orphaned "where {f_j}..." fragment that appears twice; remove the bulleted restatement of the architecture | R1.7, R2.8 |
| Fig. 1 | KEEP (style unchanged) | — |
| Algorithm 1 | REWRITE — explicit explanation-trigger condition, explanation budget, and the admit/evict window semantics | R2.4, R6.3, R7.3 |
| **IV-x Flow-rule management** | **[NEW]** — aggregation and eviction policy description feeding E8 | R3.4 |
| IV-C Evaluation metrics, Eq. (4) | EDIT — fix `\mathrm{Acc}` spacing, define every symbol, cite the equation in text, add PR-AUC and average precision | R1.3, R5.6, R6.7 |

## V. Mathematical foundation

| Element | Action | Comments |
|---|---|---|
| **Section preamble** | **[NEW]** | R5.1 |
| V-A Entropy feature extraction | **REWRITE in full** — this is the section Reviewer 1 singled out. Remove "a bunch of", "things like", "The key to making the system fast". Hedge the entropy/traffic relationship. | R1.2, R2.5, R2.8, R6.2 |
| Eq. (5) source-IP entropy | EDIT — cite it in text | R5.6 |
| **Eq. (6) rolling update** | **CORRECT** — the printed φ-form is not what the implementation computes. Replace with the running-sum form `H = log₂|W| − S/|W|`, `S = Σ c log₂ c`, with the admit/evict update actually executed. | R6.4, R7.3 |
| **Complexity statement** | REWRITE — expected/amortised O(1) per update, with the hash-table and buffer assumptions stated | R2.6 |
| **V-x Numerical stability** | **[NEW]** — drift bound, compensated summation, periodic exact recomputation, measured label-flip count | R6.4 |
| Eq. (7) feature vector | EDIT — `H_ttl` → `H_sport` after the E16 repair; cite in text | R7.2 |
| **Table: complete feature specification** | **[NEW]** — every entropy feature, its exact source column, its bin width, its units | R7.2, R7.3 |
| Fig. 2 entropy engine | REBUILD — fix `Eq. (??)`, keep style | R1.1, R2.8, R7.3 |

## VI. Experimental setup

| Element | Action | Comments |
|---|---|---|
| **Section preamble** | **[NEW]** | R5.1 |
| VI-A Datasets | REWRITE — now CIC-DDoS2019 (multiple vectors, two capture days), InSDN and CIC-IDS2017 | R6.1, R7.4, R4.1 |
| **Table: hardware and software environment** | **[NEW]** — CPU, cores, RAM, OS, Python, every library version, and the explicit statement that all timings are CPU-only on one machine and therefore comparable | R5.3, R1.4, R1.6 |
| **VI-x Evaluation protocol** | **[NEW]** — split definitions, nested CV, demonstrable tuning/testing separation | R7.8 |
| Table 3 hyperparameters | EDIT — add the search space, the selection criterion and the validation-selected τ | R7.8, R6.7 |

## VII. Results

| Element | Action | Comments |
|---|---|---|
| **Section preamble** | **[NEW]** | R5.1 |
| VII-A Detection performance | REWRITE — "It gets 99.9987% accuracy" removed; report with confidence intervals | R1.2, R2.2, R6.2 |
| VII-B Split sensitivity | REWRITE — "exceptional robustness", "This confirms", "generalizes robustly" all removed | R2.1, R2.2 |
| **VII-x Leakage and dataset-artifact audit** | **[NEW]** — E5: duplicates, near-duplicates, per-feature separability, progressive removal including the full flagged set, permutation control | R3.1, R4.1, R6.8, R2.2 |
| **VII-x Collinearity and SHAP stability** | **[NEW]** — E6 | R3.2, R6.6 |
| **VII-x Threshold selection by precision-recall** | **[NEW]** — E12, replaces the FNR-based argument | R6.7 |
| **VII-x Entropy numerical stability** | **[NEW]** — E11 | R6.4 |
| **VII-x Cross-vector and cross-day generalization** | **[NEW]** — E3 | R6.1, R7.4 |
| **VII-x Cross-dataset validation** | **[NEW]** — E2 with FPR/FNR confidence intervals | R4.1, R2.1 |
| VII-C Latency and throughput | REWRITE — CPU-only, p50/p95/p99, consistent terminology | R1.4, R1.6, R2.4 |
| **VII-x Explanation trigger and budgeted throughput** | **[NEW]** — E10, the throughput-coverage frontier | R6.3, R2.4 |
| VII-D SHAP attribution | EDIT — soften the t-SNE claim; cite the beeswarm properly | R2.8 |
| **VII-x Individual alert case studies** | **[NEW]** — E13, four real cases with waterfalls | R7.5 |
| VII-E ROC analysis | **REWRITE with an actual figure** — currently prose with no figure at all | R7.6 |
| VII-F Hyperparameter sensitivity | EDIT — caption fixed to match the panels; `(au)` → τ | R7.6, R2.8 |
| **VII-x Control-plane deployment** | **[NEW]** — E4/E7/E8: controller CPU and memory, end-to-end latency, control-channel saturation, TCAM occupancy, QoS impact | R4.2, R3.3, R3.4, R2.1 |

## VIII. Ablation and comparison

| Element | Action | Comments |
|---|---|---|
| **Section preamble** | **[NEW]** | R5.1 |
| VIII-A Feature ablation | EDIT — cite Table 5 and Fig. 7 in text (currently orphaned) | R5.6 |
| VIII-B Baseline comparison | **REWRITE** — CPU-only for all models, both protocols reported separately | R1.4, R1.5, R6.5 |
| **VIII-x Deep-learning baselines** | **[NEW]** — CNN, LSTM, autoencoder, GNN, transformer under the same protocol | R4.2 |
| VIII-C Leakage ablation | **SUPERSEDED** by the new VII leakage audit; retained only as a pointer | R6.8, R2.2 |
| VIII-D Comparison with published results (Table 9) | REWRITE — reframed as contextual, with an explicit note that the protocols differ; "surpasses" and "outperforming" removed | R2.3, R7.7 |
| Fig. 9 radar | EITHER cite it properly and soften the area claim, OR remove it to reclaim space | R5.6, R2.8 |

## IX. Discussion

| Element | Action | Comments |
|---|---|---|
| **Section preamble** | **[NEW]** | R5.1 |
| IX-A Deployment considerations | REWRITE — grounded in the measured testbed rather than extrapolation | R2.1, R3.3 |
| **IX-x Deep learning for SDN DDoS detection and QoS-aware SD-IoT** | **[NEW]** — accuracy/latency/explainability trade-offs versus RF+SHAP, adaptation to SD-IoT under latency, energy, reliability, scalability and heterogeneity constraints, lightweight and edge inference, cross-layer coordination between security enforcement and QoS policy | R4.3 |
| IX-B Limitations | REWRITE — rebuilt from what the new experiments actually showed, including what still cannot be excluded | R2.1, R2.2, R6.1 |

## X. Conclusion

REWRITE — remove "lightning speeds", "highly compelling", "devastating"; state only what
was measured. R1.2, R2.8, R6.2.

## Back matter

| Element | Action | Comments |
|---|---|---|
| Data availability | EDIT — replace the anonymous 4open.science link with the public GitHub repository | — |
| References | See Phase 4 | R2.9, R3.5, R5.4, R5.5 |

---

## Float budget

Submitted: 9 figures, 9 tables. The revision needs roughly 7 new figures and 8 new tables.
IEEE Access has no hard page limit but charges by page, and reviewers penalise bloat.
Management plan:

- **Merge** the sensitivity figure's two panels with the new PR-curve figure where it reads naturally.
- **Consider removing** Fig. 9 (radar) — it is uncited, and Reviewer 2 explicitly criticised
  treating radar-chart area as evidence. Removing it addresses R2.8 and R5.6 at once.
- **Consider removing** Fig. 5 (t-SNE) or demoting its claim — Reviewer 2 criticised it on the
  same grounds.
- New figures are allocated one per major new result, not one per sub-result.

Final float count to be fixed in Phase 5 once all results exist.

---

## Addendum: Mininet evidence figure (R4.2)

Reviewer 4 asked for a live proof of concept. The revision includes one, and the
paper shows what the session actually printed rather than only describing it.

E4b captures five verbatim transcripts per condition into
`04_results/mininet_evidence/`:

| File | What it evidences |
|---|---|
| `*__mn_version.txt` | Mininet and Open vSwitch versions actually installed |
| `*__ovs_show.txt` | the bridge, its ports and the controller connection state |
| `*__ofctl_show.txt` | the OpenFlow 1.3 datapath as the switch reports it |
| `*__flow_table.txt` | the rules the detector installed, including drop rules |
| `*__controller_tail.txt` | the controller log, showing switch connect and alerts |

Phase 6 typesets these as a console-transcript figure in a monospace frame,
captioned as console output from the live session. The caption says console
transcript rather than screenshot, because the text is typeset from the captured
output rather than photographed from a window. The content is verbatim and
unedited, which is what makes it evidence.

Placement: the deployment subsection of Sec. VII, beside the QoS table.
