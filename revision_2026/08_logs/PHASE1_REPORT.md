# Phase 1 Report — Comment Decomposition and Traceability

Status: **COMPLETE**. All five exit criteria met.

## Concern count corrected

Decomposing the review letter comment by comment gives **47 atomic concerns**, not the 35
estimated during planning. Breakdown: R1 = 7, R2 = 9, R3 = 5, R4 = 3 (compound, each
containing several distinct demands), R5 = 7, R6 = 8, R7 = 8. Every ID is unique and every
one is verbatim-quoted in `01_matrix/reviewer_comments.json`, which is what the Phase 8
response document will be generated from.

Severity profile: **20 critical, 15 major, 12 minor**. Twenty-five of the 47 require new
experiments.

## Artefacts

| File | Contents |
|---|---|
| `01_matrix/reviewer_comments.json` | 47 records: verbatim text, type, severity, experiment mapping, target locations, owning phase, status |
| `01_matrix/experiment_registry.json` | 17 experiments (E0–E16) with objective, protocol, acceptance criteria, outputs, dependencies |
| `01_matrix/manuscript_plan.md` | Target section structure — what is new, rewritten, edited or removed, and which concern lands where |
| `01_matrix/TRACEABILITY.md` | Generated human-readable matrix |
| `03_experiments/common/render_traceability.py` | Regenerator, so the table can never drift from the JSON |

## Consistency checks (automated)

- Experiments referenced by a concern but not declared: **none**
- Experiments declared but answering no concern: **none**
- Concerns requiring an experiment but mapped to none: **none**
- Duplicate comment IDs: **none**

## Experiment programme

17 experiments, **~37 hours of estimated compute**. Dependency-ordered:

```
E0  baseline reproduction              (gate for everything)
└── E16 feature-set audit and repair   (H_ttl defect)
    ├── E5  leakage / artifact audit
    ├── E6  collinearity + SHAP stability
    ├── E12 PR-based threshold selection
    ├── E15 protocol formalisation
    ├── E11 entropy numerical stability
    ├── E3  cross-vector / cross-day
    ├── E2  cross-dataset (InSDN, CIC-IDS2017)
    ├── E1  CPU-only classical baselines
    │   └── E9  deep baselines (CNN/LSTM/AE/Transformer/GNN)
    │       └── E14 ROC + PR curves
    ├── E10 explanation trigger + budget
    │   └── E4  controller-in-the-loop OpenFlow testbed
    │       ├── E7  control-channel capacity
    │       └── E8  TCAM aggregation + eviction
    └── E13 individual alert case studies
```

## Step 1.4 — the three ambiguous comments, adjudicated

**R2.7 — mechanisms that are not in this paper.** Reviewer 2 asks us to soften causal claims
about "the two-stage design and edge-aware attention" improving recall and detection delay,
and about "the additional alarm" being a "post-fault/recovery transient". None of these
appear anywhere in the manuscript: there is no two-stage design, no attention mechanism, and
no fault-recovery discussion. The paragraph appears to belong to a different submission.
**Adjudication:** respond courteously, note the discrepancy without belabouring it, and
apply the underlying principle in full — every mechanism-level explanation in our paper that
is not isolated by a dedicated ablation gets rehedged. That is the substantive compliance the
reviewer is actually asking for, and it costs us nothing to give.

**R5.5 — reference 34 does not exist.** Reviewer 5 asks us to replace arXiv references
"23, 25, 31, and 34". The submitted PDF has **32 references**. The preprint-flavoured entries
are [23] Doshi-Velez and Kim, [25] Lundberg and Lee, [31] Kumar et al., and [32] You et al.
**Adjudication:** treat the request as covering 23, 25, 31 and 32, handle all four, and state
the numbering discrepancy plainly in the response so the reviewer can see nothing was skipped.

**R4.2 — Mininet is not runnable.** Reviewer 4 asks for a Mininet/Ryu or ONOS proof of
concept. Mininet requires Linux; WSL2 on this machine reports that hardware virtualization is
disabled in firmware, which also rules out Hyper-V and Docker.
**Adjudication:** build a **real OpenFlow 1.3 control plane** on `os-ken` (the maintained Ryu
fork, verified importable with `OFP_VERSION = 4`), driven by software switch agents that open
real TCP connections and exchange real `packet_in` / `flow_stats_reply` / `flow_mod` messages
from a wall-clock replay of the trace. Controller CPU, memory, end-to-end latency,
control-channel load and QoS impact are then measured from a genuinely running controller
process rather than from a simulation of one. In the manuscript this is described precisely
as a controller-in-the-loop testbed with emulated switch agents, and never as Mininet. The
substitution and its reason are stated in the response letter. If virtualization is enabled
in firmware later, real Mininet can be swapped in without changing any other result.

## Additional adjudication carried from Phase 0

**R7.2 — the TTL feature.** `features/pipeline.py:269` hard-codes `"ttl": 64`, so `H_ttl` is
identically zero. This is registered as experiment **E16** rather than as a prose fix, because
replacing it changes the feature vector and therefore requires a full re-baseline. The
proposed replacement is source-port entropy. Adoption is conditional on E16 confirming, on
real data, that `H_ttl` has exactly zero variance and that the replacement carries genuine
information.

## Next

Phase 2: data acquisition and baseline reproduction. `03-11/Syn.csv` has already downloaded
at exactly 1,877,372,065 bytes; the remaining partitions and the cross-datasets continue in
the background.
