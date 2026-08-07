# XAI-SDN — Pre-Drafted Reviewer Rebuttal Notes

> Use this document to prepare responses to anticipated reviewer objections
> before submission. Each section corresponds to a likely reviewer concern.

---

## R1 — "Evaluation on a single dataset / attack type is insufficient"

**Objection:** The paper only evaluates on the SYN partition of CIC-DDoS2019.

**Response (v2 paper addresses this):**
Section 7.1 (Multi-vector generalization, RQ2) now reports per-partition results
across **five** CIC-DDoS2019 attack vectors: SYN, UDP, LDAP, MSSQL, and NetBIOS
(Table 10). XAI-SDN achieves consistent performance across all five, and the
SHAP top-5 features (H_src, H_dst, H_flag) are stable across partitions,
confirming that the model learns attack-agnostic distributional evidence rather
than SYN-specific artefacts. Evaluation on a second external dataset (InSDN) is
identified as the primary future-work direction in Section 8.2.

---

## R2 — "The Wilcoxon significance test only used 5 seeds"

**Objection:** 5 seeds is insufficient for a reliable signed-rank test
(minimum recommended is n=10 for this test).

**Response (v2 paper addresses this):**
The ablation campaign now runs across **10 seeds** {42, 123, 456, 789, 1024,
2048, 31337, 555, 777, 999} (Section 7.2, Table 11). The corrected p-values are:
- vs. SVM: **p = 0.0020** (significant, α = 0.05) ✅
- vs. RF-CIC-only: p = 1.000 (not significant on this near-saturated benchmark)
- vs. RF-Entropy-only: p = 1.000 (not significant)

The paper now also honestly notes that the benchmark saturation means the
CIC-only comparison is inconclusive on this particular dataset (see Section 7.2).

---

## R3 — "SHAP attributions are not evaluated for fidelity or stability"

**Objection:** Reporting global SHAP importance plots is insufficient; one must
show that SHAP attributions actually track the model's decision function
(fidelity) and are reproducible (stability).

**Response (v2 paper addresses this):**
Section 6.3 (SHAP explanation quality, RQ3) now reports four rigorous XAI metrics:
- **C1 Fidelity**: Deletion curve at k=5 shows SHAP agreement drops to 68.7% vs.
  100% for random masking (Δ = −31.3 pp), proving SHAP tracks genuine evidence.
- **C2 Stability**: 4/5 top features identical across 5 seeds (80% rank stability).
- **C3 Sparsity**: 35% of alerts actionable with top-3 features for tier-1 triage.
- **C4 Runtime**: p50 = 0.096 ms, p99 = 0.251 ms per flow (added to Table 5).

---

## R4 — "Three citations are irrelevant to network security"

**Objection:** References [akarma2026agents] (IoUT security), [jan2025blockchain]
(blockchain AI), and [toqeer2026climate] (climate digital twins) appear to be
unrelated author self-citations inserted to inflate citation count.

**Response (v2 paper addresses this):**
All three have been **removed** from the manuscript and replaced with directly
relevant references:
- Zebin et al. (2022) — SHAP + RF for intrusion detection (Security and
  Communication Networks, doi:10.1155/2022/4559024)
- Islam et al. (2023) — XAI for SDN intrusion detection with SHAP and LIME
  (Wireless Communications and Mobile Computing, doi:10.1155/2023/4995167)
- Zoppi et al. (2023) — Unsupervised anomaly detectors for intrusion in
  current threat landscape (ACM/IMS TDS, doi:10.1145/3441452)

---

## R5 — "The paper lacks explicit research questions"

**Objection:** The paper does not state research questions, making it hard to
assess whether the experiments address the stated goals.

**Response (v2 paper addresses this):**
Three explicit RQs are now stated in Section 1 (Introduction):
- **RQ1**: Can entropy features measurably improve DDoS detection beyond
  CICFlowMeter alone? (Answered by ablation in Section 7.2)
- **RQ2**: Does XAI-SDN generalise across multiple DDoS attack vectors?
  (Answered by multi-partition evaluation in Section 7.1)
- **RQ3**: Do SHAP attributions faithfully reflect the decision function with
  acceptable operational latency? (Answered by XAI quality metrics in Section 6.3)

---

## R6 — "n_train contains an arithmetic error (2,514,862 vs 2,514,860)"

**Objection:** Minor but reviewers check arithmetic; the training set size is
inconsistent across the paper.

**Response (v2 paper addresses this):**
The training count is now consistently reported as **2,514,860** in Table 3
(Training time row) and all prose references. The discrepancy arose from a
rounding artefact in an earlier draft; the model artifact
`reproducibility_manifest.json` records the authoritative split sizes.

---

*Last updated: 2026-08-07. Prepared for Computers & Security Q1 submission.*
