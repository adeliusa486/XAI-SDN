# Supplementary Material

**XAI-SDN: An explainable entropy-guided machine learning framework for real-time
DDoS detection in software defined networks**

Manuscript Access-2026-39885. This package contains the extended diagnostics, the
complete testbed configuration and the raw result files behind every number in the
main article. The article is self-contained: nothing here is required to follow the
argument or to check a reported value. This material is for readers who want the
full per-run evidence or who intend to reproduce the work.

`MANIFEST.json` lists all 130 files with SHA-256 checksums.

---

## Directory layout

```
Supplementary_XAI-SDN/
├── README.md                        this file
├── MANIFEST.json                    every file, size and SHA-256
├── S1_Extended_Experimental_Details/
│   ├── leakage/                     leakage audit, duplicate census, feature audit
│   ├── collinearity/                full Pearson matrix and VIF table
│   ├── shap_stability/              per-seed and bootstrap attribution stability
│   ├── transfer/                    cross-vector, cross-day, cross-corpus, few-shot
│   └── threshold/                   validation and test PR/ROC sweeps
├── S2_Mininet_OpenFlow_Testbed/
│   ├── topology/                    Mininet topology builder
│   ├── controller/                  OpenFlow 1.3 controller and testbed harness
│   ├── ovs/                         Open vSwitch setup
│   ├── traffic_generation/          attack and benign traffic drivers
│   ├── rate_sweeps/                 offered-rate sweeps and controller time series
│   ├── tcam_evaluation/             flow-table occupancy and aggregation policies
│   ├── evidence/                    captured console output and demo frames
│   └── run_instructions/            step-by-step reproduction
├── S3_Complete_Experiment_Results/  raw JSON and CSV for the remaining experiments
├── S4_Complete_Tables/              full-width versions of abridged tables
├── S5_Additional_Figures/           vector sources of the article figures
├── S6_Environment/                  pinned dependencies and provenance
└── S7_Reproducibility/              every experiment script and the run queue
```

---

## Experiment index

Each experiment writes result files under its own identifier. The identifiers are
the ones used in the repository and in the article's data-availability statement.

| ID | Experiment | Supports | Files |
|---|---|---|---:|
| `E0` | Baseline reproduction of the pre-revision pipeline | Sec. VII-A, Table 6 | 1 |
| `E1` | Classical baselines, CPU-only, both protocols | Sec. VIII-A, Table 19 | 2 |
| `E2` | Cross-dataset transfer (InSDN, CIC-IDS2017) | Sec. VII-J, Table 12 | 3 |
| `E3` | Cross-vector transfer (LDAP, NetBIOS, Portmap) | Sec. VII-J, Table 12 | 2 |
| `E4` | Live OpenFlow 1.3 controller testbed | Sec. VII-R, Table 17 | 3 |
| `E4b` | Mininet + Open vSwitch live deployment | Sec. VII-P, Table 15 | 2 |
| `E5` | Leakage and dataset-artifact audit | Sec. VII-D, Table 8 | 4 |
| `E5b` | Duplicate and near-duplicate census | Sec. VII-D1 | 2 |
| `E6` | Collinearity, VIF and SHAP attribution stability | Sec. VII-G, Table 11 | 2 |
| `E6s` | SHAP stability across seeds and bootstrap resamples | Sec. VII-G | 2 |
| `E7` | Control-channel offered-rate sweep | Sec. VII-R, Table 17 | 1 |
| `E8b` | Flow-table occupancy and TCAM aggregation policy | Sec. VII-S, Table 18 | 3 |
| `E9` | Deep-learning baselines (CNN, LSTM, Transformer, AE, GNN) | Sec. VIII-B, Table 20 | 2 |
| `E10` | Explanation budget and throughput frontier | Sec. VII-K, Table 13 | 2 |
| `E11` | Numerical stability of the streaming entropy engine | Sec. VII-I | 2 |
| `E12` | Threshold selection on the precision-recall plane | Sec. VII-H | 5 |
| `E13` | Individual alert case studies | Sec. VII-M, Table 14 | 2 |
| `E14` | ROC and precision-recall curves | Sec. VII-N, Fig. 4 | 3 |
| `E15` | Nested cross-validation and evaluation protocol | Sec. VI-B | 2 |
| `E16` | Feature-set audit and single-feature separability | Sec. VII-C | 3 |
| `E17` | Contribution of the entropy augmentation | Sec. VII-F, Table 10 | 2 |
| `E18` | Temporal structure of the benign class | Sec. VII-E | 2 |
| `E19` | Global SHAP attribution sample | Sec. VII-L, Fig. 3 | 1 |
| `E20` | Few-shot transfer and labeling budget | Sec. VII-J1 | 2 |
| `E21` | Entropy augmentation under distribution shift | Sec. VII-F1, Table 9 | 2 |
| `E22` | Model size / latency Pareto frontier | Sec. VII-Q, Table 16 | 2 |
| `E23` | Single-flow latency verification | Sec. VII-Q | 1 |

---

## Where each file lives

**E0 — Baseline reproduction of the pre-revision pipeline**  (Sec. VII-A, Table 6)

- `S3_Complete_Experiment_Results/E0_baseline_reproduction.json` (10 KB)

**E1 — Classical baselines, CPU-only, both protocols**  (Sec. VIII-A, Table 19)

- `S3_Complete_Experiment_Results/E1_baselines_cpu.csv` (5 KB)
- `S3_Complete_Experiment_Results/E1_baselines_cpu.json` (28 KB)

**E2 — Cross-dataset transfer (InSDN, CIC-IDS2017)**  (Sec. VII-J, Table 12)

- `S1_Extended_Experimental_Details/transfer/E2_cross_dataset.csv` (1 KB)
- `S1_Extended_Experimental_Details/transfer/E2_cross_dataset.json` (11 KB)
- `S1_Extended_Experimental_Details/transfer/E2_schema_map.csv` (7 KB)

**E3 — Cross-vector transfer (LDAP, NetBIOS, Portmap)**  (Sec. VII-J, Table 12)

- `S1_Extended_Experimental_Details/transfer/E3_cross_vector.json` (16 KB)
- `S1_Extended_Experimental_Details/transfer/E3_transfer_matrix.csv` (3 KB)

**E4 — Live OpenFlow 1.3 controller testbed**  (Sec. VII-R, Table 17)

- `S2_Mininet_OpenFlow_Testbed/rate_sweeps/E4_controller_testbed.json` (12 KB)
- `S2_Mininet_OpenFlow_Testbed/rate_sweeps/E4_timeseries_detection_only.csv` (27 KB)
- `S2_Mininet_OpenFlow_Testbed/rate_sweeps/E4_timeseries_with_explanations.csv` (26 KB)

**E4b — Mininet + Open vSwitch live deployment**  (Sec. VII-P, Table 15)

- `S2_Mininet_OpenFlow_Testbed/rate_sweeps/E4b_PROVENANCE.md` (1 KB)
- `S2_Mininet_OpenFlow_Testbed/rate_sweeps/E4b_mininet_testbed.json` (11 KB)

**E5 — Leakage and dataset-artifact audit**  (Sec. VII-D, Table 8)

- `S1_Extended_Experimental_Details/leakage/E5_distribution_checks.csv` (5 KB)
- `S1_Extended_Experimental_Details/leakage/E5_leakage_audit.json` (11 KB)
- `S1_Extended_Experimental_Details/leakage/E5_progressive_removal.csv` (1 KB)
- `S1_Extended_Experimental_Details/leakage/E5_single_feature_separability.csv` (4 KB)

**E5b — Duplicate and near-duplicate census**  (Sec. VII-D1)

- `S1_Extended_Experimental_Details/leakage/E5b_duplicate_leakage.json` (3 KB)
- `S1_Extended_Experimental_Details/leakage/E5b_straddle.csv` (1 KB)

**E6 — Collinearity, VIF and SHAP attribution stability**  (Sec. VII-G, Table 11)

- `S1_Extended_Experimental_Details/collinearity/E6_pearson.csv` (124 KB)
- `S1_Extended_Experimental_Details/collinearity/E6_vif.csv` (2 KB)

**E6s — SHAP stability across seeds and bootstrap resamples**  (Sec. VII-G)

- `S1_Extended_Experimental_Details/shap_stability/E6_collinearity_shap_stability.json` (17 KB)
- `S1_Extended_Experimental_Details/shap_stability/E6_global_shap.csv` (4 KB)

**E7 — Control-channel offered-rate sweep**  (Sec. VII-R, Table 17)

- `S2_Mininet_OpenFlow_Testbed/rate_sweeps/E7_rate_sweep.csv` (1 KB)

**E8b — Flow-table occupancy and TCAM aggregation policy**  (Sec. VII-S, Table 18)

- `S2_Mininet_OpenFlow_Testbed/tcam_evaluation/E8b_occupancy.csv` (527 KB)
- `S2_Mininet_OpenFlow_Testbed/tcam_evaluation/E8b_tcam_grid.csv` (5 KB)
- `S2_Mininet_OpenFlow_Testbed/tcam_evaluation/E8b_tcam_policy.json` (29 KB)

**E9 — Deep-learning baselines (CNN, LSTM, Transformer, AE, GNN)**  (Sec. VIII-B, Table 20)

- `S3_Complete_Experiment_Results/E9_deep_baselines.csv` (2 KB)
- `S3_Complete_Experiment_Results/E9_deep_baselines.json` (8 KB)

**E10 — Explanation budget and throughput frontier**  (Sec. VII-K, Table 13)

- `S3_Complete_Experiment_Results/E10_explanation_budget.json` (12 KB)
- `S3_Complete_Experiment_Results/E10_frontier.csv` (1 KB)

**E11 — Numerical stability of the streaming entropy engine**  (Sec. VII-I)

- `S3_Complete_Experiment_Results/E11_drift_curve.csv` (74 KB)
- `S3_Complete_Experiment_Results/E11_entropy_numerics.json` (4 KB)

**E12 — Threshold selection on the precision-recall plane**  (Sec. VII-H)

- `S1_Extended_Experimental_Details/threshold/E12_pr_threshold.json` (10 KB)
- `S1_Extended_Experimental_Details/threshold/E12_test_pr_curve.csv` (7 KB)
- `S1_Extended_Experimental_Details/threshold/E12_validation_pr_curve.csv` (1 KB)
- `S1_Extended_Experimental_Details/threshold/E12_validation_roc_curve.csv` (1 KB)
- `S1_Extended_Experimental_Details/threshold/E12_validation_sweep.csv` (1 KB)

**E13 — Individual alert case studies**  (Sec. VII-M, Table 14)

- `S3_Complete_Experiment_Results/E13_case_studies.csv` (3 KB)
- `S3_Complete_Experiment_Results/E13_case_studies.json` (11 KB)

**E14 — ROC and precision-recall curves**  (Sec. VII-N, Fig. 4)

- `S3_Complete_Experiment_Results/E14_auc_summary.csv` (7 KB)
- `S3_Complete_Experiment_Results/E14_curves.csv` (612 KB)
- `S3_Complete_Experiment_Results/E14_roc_pr_curves.json` (22 KB)

**E15 — Nested cross-validation and evaluation protocol**  (Sec. VI-B)

- `S3_Complete_Experiment_Results/E15_nested_cv_folds.csv` (1 KB)
- `S3_Complete_Experiment_Results/E15_protocol.json` (67 KB)

**E16 — Feature-set audit and single-feature separability**  (Sec. VII-C)

- `S1_Extended_Experimental_Details/leakage/E16_feature_audit.json` (3 KB)
- `S1_Extended_Experimental_Details/leakage/E16_feature_specification.csv` (1 KB)
- `S1_Extended_Experimental_Details/leakage/E16_feature_statistics.csv` (22 KB)

**E17 — Contribution of the entropy augmentation**  (Sec. VII-F, Table 10)

- `S3_Complete_Experiment_Results/E17_entropy_contribution.json` (7 KB)
- `S3_Complete_Experiment_Results/E17_variants.csv` (2 KB)

**E18 — Temporal structure of the benign class**  (Sec. VII-E)

- `S3_Complete_Experiment_Results/E18_temporal_structure.json` (3 KB)
- `S3_Complete_Experiment_Results/E18_timeline.csv` (1 KB)

**E19 — Global SHAP attribution sample**  (Sec. VII-L, Fig. 3)

- `S1_Extended_Experimental_Details/shap_stability/E19_shap_sample.json` (3 KB)

**E20 — Few-shot transfer and labeling budget**  (Sec. VII-J1)

- `S1_Extended_Experimental_Details/transfer/E20_few_shot_transfer.json` (31 KB)
- `S1_Extended_Experimental_Details/transfer/E20_transfer_budget.csv` (8 KB)

**E21 — Entropy augmentation under distribution shift**  (Sec. VII-F1, Table 9)

- `S1_Extended_Experimental_Details/transfer/E21_entropy_under_shift.csv` (11 KB)
- `S1_Extended_Experimental_Details/transfer/E21_entropy_under_shift.json` (4 KB)

**E22 — Model size / latency Pareto frontier**  (Sec. VII-Q, Table 16)

- `S3_Complete_Experiment_Results/E22_model_size_pareto.json` (29 KB)
- `S3_Complete_Experiment_Results/E22_size_pareto.csv` (6 KB)

**E23 — Single-flow latency verification**  (Sec. VII-Q)

- `S3_Complete_Experiment_Results/E23_latency_verification.json` (2 KB)

---

## Reproducing a result

### Requirements

| | |
|---|---|
| Python | 3.11 |
| OS for the analysis experiments | Windows 11 or Linux |
| OS for the testbed experiments | Linux (Mininet does not run on Windows) |
| Mininet | 2.3.0 |
| Open vSwitch | 3.7.1 |
| Hardware used | Intel Core i9-13900H, 14 cores / 20 threads, 47.6 GB RAM |
| GPU | none; the installed PyTorch build has no CUDA support |

Dependencies are pinned in `S6_Environment/`. The analysis experiments need about
16 GB of free memory, because the SYN partition holds 3,590,794 flows after
cleaning.

### Data

The corpora are public and are not redistributed here:

- CIC-DDoS2019 (primary; the `03-11/Syn.csv` capture)
- CIC-IDS2017
- InSDN

`S7_Reproducibility/download_data.py` fetches and verifies them.

### Running one experiment

```bash
pip install -r S6_Environment/requirements.txt
python S7_Reproducibility/download_data.py
python S7_Reproducibility/experiment_scripts/e5_leakage_audit.py
```

Each script writes its result files with a provenance block recording the host,
the library versions and the timestamp, so any output can be traced to the build
that produced it.

### Running the live testbed

The testbed needs root on Linux:

```bash
sudo bash S2_Mininet_OpenFlow_Testbed/ovs/setup_mininet.sh
sudo python3 S2_Mininet_OpenFlow_Testbed/controller/e4b_mininet_testbed.py
```

See `S2_Mininet_OpenFlow_Testbed/run_instructions/` for the full sequence,
including the offered-rate sweep and the flow-table occupancy evaluation.

---

## A note on the captured evidence

Six console captures under `S2_Mininet_OpenFlow_Testbed/evidence/` record a failed
command rather than a measurement, because the evidence harness ran them after the
Mininet namespace had been torn down. They are shipped unaltered and are documented
in `evidence/KNOWN_ISSUES.md`. No reported value depends on them.
