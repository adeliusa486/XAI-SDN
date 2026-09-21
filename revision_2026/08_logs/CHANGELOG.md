# Change log: submitted version to revision

Every edit carries the reviewer concern ID that motivated it. Generated
alongside the manuscript and cross-checked against `01_matrix/TRACEABILITY.md`
and the Response to Reviewers document.

## Front matter

| Location | Change | IDs |
|---|---|---|
| Preamble | Added `\FILL` drafting guard (removed before submission) | internal |
| Abstract | Pending rewrite once all results land | R1.2, R2.1, R2.4, R2.6, R2.8, R6.2 |

## Sec. I Introduction

| Location | Change | IDs |
|---|---|---|
| Opening paragraphs | Rewritten. Removed "to be able to see the whole network", "Some of them use something called", "There are also attacks" | R1.2, R2.8, R6.2 |
| Contributions | Rewritten as four measurable operational claims rather than a generic list | R7.1, R2.1 |
| Closing paragraph | Added an explicit non-dominance statement | R2.1, R2.3 |
| **Table 1 (new)** | Notation, symbols and acronyms table at the end of Sec. I | R5.2 |

## Sec. II Related work

| Location | Change | IDs |
|---|---|---|
| Section opening | Added a scene-setting paragraph before the first subsection | R5.1 |
| II-A | Entropy interpretation rewritten as statistical, not deterministic | R2.5 |
| II-A | O(1) claim qualified as expected amortized | R2.6 |
| II-B | Added recent DL/SDN work; expanded the benchmark-critique paragraph | R5.4, R6.1 |
| II-C | Added the uncertainty-quantification reference and the attribution/confidence distinction | R3.5 |
| **II-D (new)** | Nonlinear dynamic indices for abnormal-state detection | R2.9 |
| **II-E (new)** | Positioning paragraph | R7.1 |
| Table 2 | Rewritten with the three axes that carry the novelty claim; removed the unsupported "0.5 ms per alert" cell | R7.1, R2.4 |

## Sec. III Threat model and problem formulation

| Location | Change | IDs |
|---|---|---|
| Section opening | Added preamble | R5.1 |
| **III-B (new)** | Formal definitions of flow, positive prediction, alert, per-decision explanation | R2.4 |
| III-C | Rewritten; threshold selection moved to the precision-recall plane | R6.7 |
| **Eq. (2) (new)** | Explanation trigger stated formally, including the budget term | R2.4, R6.3 |
| **Eq. (3) (new)** | Additive efficiency property given its own number and cited in text | R5.6 |

## Sec. IV Framework and architecture

| Location | Change | IDs |
|---|---|---|
| Algorithm 1 | Rewritten with the explanation budget and episode inheritance | R6.3, R2.4 |
| After Algorithm 1 | Added a paragraph separating classification rate from attribution rate | R2.4, R6.3 |
| Removed | Orphaned duplicate fragment "where $\{f_j\}$..." (appeared twice) | R1.7 |
| Removed | Bulleted restatement of the architecture | R2.8, R6.2 |
| IV-D Eq. (4) | Reformatted with displayed fractions; added AP and the rationale for preferring it to FNR | R1.3, R6.7 |

## Sec. V Mathematical foundation

| Location | Change | IDs |
|---|---|---|
| Section opening | Added preamble | R5.1 |
| V-A | Rewritten in full. Removed "a bunch of", "things like", "The key to making the system fast" | R1.2, R2.8, R6.2 |
| V-A | Entropy semantics hedged | R2.5 |
| **Table 4 (new)** | Feature specification: source field, transformation, bin width for all eight entropy features | R7.2, R7.3 |
| V-A | Disclosed the TTL defect and its replacement by source-port entropy | R7.2 |
| **V-B (new)** | Window maintenance: admit/evict, warm-up, per-partition state | R7.3 |
| **Eqs. (5)-(7)** | Corrected the update recurrence to the form the implementation executes | R6.4 |
| **V-C (new)** | Complexity and numerical behavior, with data-structure assumptions | R2.6, R6.4 |
| Eq. (8) | `H_ttl` replaced by `H_sport`; cited in text | R7.2, R5.6 |
| Fig. 2 caption | Corrected complexity claim; equation reference now resolves | R1.1, R2.6 |

## Sec. VI Experimental setup

| Location | Change | IDs |
|---|---|---|
| Section opening | Added preamble | R5.1 |
| VI-A | Rewritten; six corpora instead of one; cross-corpus schema stated | R6.1, R7.4, R4.1 |
| Table 5 | Replaced the single-partition table with all corpora | R6.1 |
| **VI-B (new)** | Evaluation protocol: both splits, validation block, nested CV, tuning/test separation | R7.8 |
| **VI-C, Table 6 (new)** | Hardware and software environment, with the CPU-only comparability statement | R5.3, R1.4, R1.6 |

## Sec. VII Results

| Location | Change | IDs |
|---|---|---|
| Section opening | Added preamble | R5.1 |
| **VII-A (new)** | Split methodology dominates the headline result; both protocols reported | R2.1, R4.1, R6.1, R7.8 |
| VII-B | Detection performance restated under the chronological protocol | R2.1, R2.2 |
| **VII-C (new)** | Feature-set audit: 12 constant features, entropy separability ranking | R6.6, R7.2 |
| **VII-D (new)** | Leakage audit with objective suspect definition and full-set removal | R3.1, R4.1, R6.8, R2.2 |
| VII-E to VII-J | New subsections pending results (E5 controls, E6, E12, E11, E3, E2) | multiple |
| Fig. 5 caption | t-SNE caption rewritten; no longer presented as evidence | R2.8 |
| Removed | Radar figure; reviewer objected to radar area as evidence and it was never cited | R2.8, R5.6 |

## Sec. VIII Ablation and comparison

| Location | Change | IDs |
|---|---|---|
| VIII-D | Table 9 reframed as contextual; both our protocols shown; "surpasses" removed | R2.3, R7.7 |
| Remainder | Pending rewrite once E1/E9/E14 land | R1.4, R1.5, R6.5, R4.2, R7.6 |

## Sec. IX Discussion

| Location | Change | IDs |
|---|---|---|
| Section opening | Added preamble | R5.1 |
| **IX-B (new)** | Deep learning for SDN DDoS detection: accuracy, latency, explainability trade | R4.3, R4.2 |
| **IX-C (new)** | QoS-aware SD-IoT: latency/energy, reliability/scalability, heterogeneity, cross-layer | R4.3 |
| IX-E | Limitations rewritten; six limits stated, including the protocol error | R2.1, R2.2, R6.1, R4.2 |

## References

| Change | IDs |
|---|---|
| [25] replaced with the published NeurIPS record | R5.5 |
| [23], [31], [32] recast as properly formatted preprints | R5.5 |
| Nine references added, all publisher-verified | R2.9, R3.5, R4.3, R5.4 |
| 2023+ share raised from 3/32 to 10/41 | R5.4 |
