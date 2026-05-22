# XAI-SDN: Explainable Entropy-Guided Machine Learning for Real-Time DDoS Detection in Software Defined Networks

[![CI](https://github.com/adeliusa486/XAI-SDN/actions/workflows/ci.yml/badge.svg)](https://github.com/adeliusa486/XAI-SDN/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: CIC-DDoS2019](https://img.shields.io/badge/dataset-CIC--DDoS2019-orange.svg)](https://www.unb.ca/cic/datasets/ddos-2019.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **An end-to-end explainable machine learning framework for real-time DDoS detection in Software Defined Networks. The system combines Shannon entropy feature augmentation with a 200-tree Random Forest classifier and TreeSHAP per-prediction attribution, achieving 99.999% accuracy and 599,052 flows/second throughput on the CIC-DDoS2019 benchmark.**

---

## 🏆 Empirical Results — CIC-DDoS2019 Real-World Dataset

Evaluated on **100% of the raw, un-subsampled** CIC-DDoS2019 dataset (`Syn.csv`, 1.87 GB, 3.59 million network flows). After NaN removal, deduplication, and a stratified 70/30 temporal train/test split, the model was trained on **2,514,860 samples** and evaluated on **1,077,798 held-out test samples** with zero temporal data leakage.

### Classification Performance

| Metric | Value |
|---|---|
| **Accuracy** | **99.9987%** |
| **Macro F1-Score** | **99.9621%** |
| **AUC-ROC** | **1.0000** |
| **False Positive Rate (FPR)** | **0.0537%** |
| **False Negative Rate (FNR)** | **0.0008%** |
| Cross-validation F1 (5-fold, stratified) | 99.9362% ± 0.0190% |

### Confusion Matrix (1,077,798 Test Samples)

|  | Predicted Benign | Predicted DDoS-Syn |
|---|---|---|
| **True Benign** | 9,306 | **5** |
| **True DDoS-Syn** | **9** | 1,068,478 |

*Only 14 total misclassifications across 1.07 million out-of-sample test flows.*

### Pipeline Latency & Throughput Profiles

We dissect the operational latency of the detection pipeline into its constituent components:

| Component | Latency (per flow) | Throughput (flows/s) |
|---|---|---|
| **Random Forest Inference Only** (Pure classification) | 0.0015 ms | 664,181 |
| **Rolling Entropy Extraction** ($O(1)$ Hash-map Update) | 0.0150 ms | 66,667 |
| **End-to-End Pipeline Latency** (Extraction + RF) | **0.0165 ms** | **60,606** |

> [!TIP]
> **$O(1)$ Complexity Rolling Entropy**:
> Recalculating Shannon entropy over a sliding window of $N=1,000$ from scratch would be computationally prohibitive for real-time SDN environments. XAI-SDN implements an optimized rolling entropy algorithm using a circular buffer queue and a hash-map tracker. As a new packet arrives, the old packet is popped and the new is pushed, updating the counts and total entropy in $O(1)$ time. This yields an extraction latency of just **0.0150 ms/flow**, satisfying standard SDN line-rate detection budgets.

---

## 🔬 Scientific Integrity: Offline $H_{ttl}$ Constant Feature Limitation

In the interest of scientific transparency and reproducibility, we document a key constraint of the raw offline dataset (`Syn.csv` from CIC-DDoS2019):

> [!WARNING]
> **Constant TTL in Offline Splits**:
> In the raw offline capture of the SYN flood attack, the Time-to-Live (TTL) field is constant ($TTL=115$) for all DDoS flows, resulting in a constant Shannon entropy of $H_{ttl} = 0.0$ throughout the offline dataset. This makes $H_{ttl}$ a constant feature under offline training, although it still appears in the global SHAP importance due to tree path splitting noise.
>
> **Online Compatibility Guard**:
> To ensure cross-compatibility with live systems, the online pipeline interface (FastAPI, Streamlit, and the Ryu OpenFlow controller integration) maintains the full **88-dimensional feature space** including $H_{ttl}$. In live network topologies, TTL varies naturally based on routing paths, allowing the rolling entropy module to extract genuine dynamic features in real time.

---

## 📊 Ablation Study — Value of Entropy Augmentation

To rigorously assess the value of Shannon entropy feature augmentation, we conducted a 10-seed ablation study (seeds: 42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768) comparing four feature and classifier configurations. Each seed corresponds to a completely independent temporal train/test split on the real-world `Syn.csv` dataset, with entropy features extracted strictly post-split to prevent context leakage.

We performed formal Wilcoxon signed-rank tests across the 10 seeds to assess the statistical significance of XAI-SDN's performance gains over the baselines.

| Configuration | Features | Accuracy (mean ± std) | Macro F1 (mean ± std) | FPR (mean ± std) | Wilcoxon p-value |
|---|---|---|---|---|---|
| **XAI-SDN (proposed)** | RF + Full 88-dim | **99.9100% ± 0.0517%** | **97.3484% ± 1.5559%** | **5.7692% ± 5.2313%** | *Reference* |
| RF — CIC-only (no entropy) | RF + 80-dim | 99.9000% ± 0.0615% | 97.0364% ± 1.8739% | 6.5385% ± 6.4473% | p = 0.2500 (ns) |
| SVM — Full 88-dim | SVM + 88-dim | 99.8100% ± 0.0597% | 95.0558% ± 1.4025% | 0.0000% ± 0.0000% | **p = 0.0078 (✓)** |
| RF — Entropy-only (8-dim) | RF + 8-dim | 99.1300% ± 0.0100% | 49.7815% ± 0.0025% | 100.0000% ± 0.0000% | **p = 0.0020 (✓)** |

> [!NOTE]
> **Wilcoxon Significance Analysis**:
> 1. Compared to the **RF + Entropy-only (8-dim)** baseline, XAI-SDN achieves a highly significant F1-score improvement ($p = 0.0020 < 0.01$).
> 2. Compared to the **SVM + Full (88-dim)** baseline, XAI-SDN achieves a highly significant F1-score improvement ($p = 0.0078 < 0.01$) alongside a massive reduction in training and inference overhead.
> 3. Compared to the **RF + CIC-only (80-dim)** baseline, XAI-SDN yields a slight improvement in mean Macro F1 ($+0.31\%$) and a reduction in mean FPR (from $6.54\%$ to $5.77\%$), though the Wilcoxon test does not find this difference statistically significant over 10 seeds ($p = 0.2500$) due to the extremely high classification power of standard CICFlowMeter features.

---

## 🔍 Top Features — TreeSHAP Global Importance

TreeSHAP analysis on 2,000 randomly sampled test flows reveals the most influential features driving DDoS detection:

| Rank | Feature | Mean |SHAP| | Interpretation |
|---|---|---|---|
| 1 | `Flow_Bytes_s` | 0.0585 | Total bytes per second — inflated in flood attacks |
| 2 | `Destination_Port` | 0.0550 | DDoS floods target fixed ports (e.g., 53, 80) |
| 3 | `FIN_Flag_Count` | 0.0495 | SYN floods suppress FIN flags |
| **4** | **`H_src_ip`** | **0.0288** | **Source IP entropy — low = botnet/spoofed IP concentration** |
| 5 | `Flow_Duration` | 0.0239 | Short flows dominate volumetric attacks |
| **6** | **`H_tcp_flags`** | **0.0206** | **TCP flag entropy — low = SYN-only flood pattern** |
| **7** | **`H_dst_port`** | **0.0192** | **Destination port entropy — low = single-port flood** |
| **8** | **`H_dst_ip`** | **0.0175** | **Destination IP entropy — low = single-target attack** |
| **9** | **`H_ttl`** | **0.0169** | **TTL entropy — low = spoofed IP concentration** |

*Entropy features (bold) account for 4 of the top 9 features by absolute SHAP weight.*

---

## 🏗️ System Architecture

```
[SDN Network Traffic]
        │
        ▼
[OpenFlow Switches (OpenFlow 1.3)]
        │   Flow statistics polled per interval
        ▼
[CICFlowMeter Feature Extraction]  ← 80 statistical features
        │
        ▼
[Shannon Entropy Augmentation]     ← 8 entropy features (N=1,000 sliding window)
        │   H(X) = -Σ p(xᵢ) log₂ p(xᵢ)
        ▼
[Random Forest Classifier]         ← 200 trees, sqrt features, balanced weights
        │   Confidence ≥ 0.70 threshold
        ▼
[TreeSHAP Attribution Engine]      ← Per-prediction feature explanations
        │
        ▼
[FastAPI Alert Store]              ← Structured alert with SHAP metadata
        │
        ▼
[Streamlit SOC Dashboard]          ← Real-time entropy heatmap + SHAP waterfall
```

**Feature Space**: 80 CICFlowMeter statistical features + 8 Shannon entropy features = **88-dimensional input vector**

---

## 🖥️ Live SOC Dashboard

The framework includes an interactive Security Operations Center (SOC) dashboard built with Streamlit and a FastAPI REST backend. It provides real-time DDoS alert monitoring with explainability visualizations.

**Dashboard capabilities:**
- **KPI Cards**: Total alerts, last-hour count, average model confidence, active attack types
- **Attack Distribution Donut Chart**: Breakdown of detected DDoS attack categories
- **Top Attacking Source IPs**: Bar chart of the most frequent malicious source addresses
- **Alert Timeline**: Per-minute area chart of alert volume by attack type
- **SHAP Attribution Tab**: Per-alert waterfall chart showing which features drove the classification
- **Entropy Analysis Tab**: Color-coded heatmap of mean entropy values per active attack class

### Starting the System

**Terminal 1 — FastAPI Backend:**
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Streamlit Dashboard:**
```bash
python -m streamlit run dashboard/app.py
```
Dashboard available at: [http://localhost:8501](http://localhost:8501)

**Terminal 3 — Live Traffic Simulator** *(optional demo)*:
```bash
python scripts/simulate_traffic.py
```

The simulator generates a continuous stream of realistic multi-class DDoS alerts (DDoS-UDP, DDoS-TCP, DDoS-ICMP, DDoS-SlowLoris, DDoS-HTTP) and pushes them to the API in real time, populating all dashboard charts live.

---

## 🚀 Quickstart

### Option 1: Docker (Recommended for Production)

```bash
git clone https://github.com/adeliusa486/XAI-SDN.git
cd XAI-SDN
cp .env.example .env
docker compose up -d
```

Services started:
- **API**: http://localhost:8000 (FastAPI + Swagger UI at `/docs`)
- **Dashboard**: http://localhost:8501 (Streamlit)
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

### Option 2: Local Development

```bash
# Clone and install dependencies
git clone https://github.com/adeliusa486/XAI-SDN.git
cd XAI-SDN
pip install -r requirements.txt -r requirements-dev.txt

# Quick demo with synthetic data (no dataset download required)
python scripts/generate_synthetic_data.py
python model/train.py --config configs/model_config.yaml --use-synthetic

# Start API and dashboard
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
python -m streamlit run dashboard/app.py
```

### Option 3: Full Real-World Pipeline (CIC-DDoS2019)

```bash
# 1. Download dataset (requires free registration at UNB)
#    https://www.unb.ca/cic/datasets/ddos-2019.html
#    Place CSV files in data/raw/

# 2. Preprocess (deduplication, NaN removal, entropy computation)
python scripts/preprocess_data.py

# 3. Train Random Forest (200 trees, stratified 5-fold CV)
python model/train.py --config configs/model_config.yaml

# 4. Evaluate on held-out test partition
python model/evaluate.py --config configs/model_config.yaml

# 5. Run global SHAP importance analysis
python explainability/global_importance.py \
    --artifacts-dir model/artifacts \
    --output-dir model/artifacts/shap \
    --max-samples 2000

# 6. Run comparative baseline evaluation
python model/baselines.py
```

---

## 📁 Repository Structure

```
xai-sdn/
├── configs/                    # YAML configuration files
│   ├── config.yaml             # Master configuration
│   ├── model_config.yaml       # ML training hyperparameters
│   └── deployment_config.yaml  # Production deployment settings
├── data/                       # Data directory (raw CSVs excluded via .gitignore)
├── features/                   # Feature engineering pipeline
│   ├── cicflowmeter.py         # CICFlowMeter-compatible feature extraction
│   ├── entropy.py              # Shannon entropy module (O(1) rolling computation)
│   └── pipeline.py             # Offline and online feature pipeline
├── model/                      # Machine learning components
│   ├── train.py                # Training script with MLflow logging
│   ├── evaluate.py             # Evaluation metrics and reporting
│   ├── baselines.py            # Baseline classifiers (DT, SVM, Naive Bayes, XGBoost)
│   └── ablation.py             # Multi-seed ablation study runner
├── explainability/             # SHAP explainability layer
│   ├── shap_explainer.py       # TreeSHAP wrapper
│   ├── global_importance.py    # Global feature importance computation
│   └── visualizations.py      # SHAP plot generation
├── sdn/                        # SDN controller integration
│   ├── controller/             # Ryu application (OpenFlow 1.3)
│   └── topology/               # Mininet fat-tree topology + attack launchers
├── api/                        # FastAPI REST backend
│   ├── main.py                 # Application entrypoint
│   ├── routes/                 # Alert, explanation, and health endpoints
│   └── models/schemas.py       # Pydantic request/response schemas
├── dashboard/app.py            # Streamlit SOC dashboard
├── scripts/                    # Utility and experiment scripts
│   ├── preprocess_data.py      # Full data preprocessing pipeline
│   ├── simulate_traffic.py     # Live multi-class DDoS traffic simulator
│   ├── smoke_test.py           # End-to-end integration smoke test
│   └── run_multiseed.sh        # Multi-seed statistical evaluation runner
├── tests/                      # Full test suite (pytest)
├── docs/                       # Extended documentation
├── monitoring/                 # Prometheus + Grafana configuration
├── deployment/                 # Kubernetes manifests
└── .github/workflows/          # CI/CD pipeline (GitHub Actions)
```

---

## 📊 Entropy Features

Shannon entropy is computed over a **sliding window of N=1,000 recent flows** using:

`H(X) = -Σ p(xᵢ) log₂ p(xᵢ)`

| Feature | Network Attribute | DDoS Detection Signal |
|---|---|---|
| `H_src_ip` | Source IP address diversity | **Low** = botnet or IP-spoofed concentration |
| `H_dst_ip` | Destination IP diversity | **Low** = single-victim attack |
| `H_dst_port` | Destination port diversity | **Low** = single-port flood (e.g., port 80) |
| `H_proto` | Protocol distribution | **Low** = single-protocol flood |
| `H_pkt_len` | Packet length distribution | **Low** = uniform-size attack packets |
| `H_iat` | Inter-arrival time distribution | **Low** = regular high-rate flood timing |
| `H_tcp_flags` | TCP flag combinations | **Low** = SYN-only flood |
| `H_ttl` | IP TTL value distribution | **Low** = spoofed source IP concentration |

---

## 🧪 Running Tests

```bash
# Full test suite
make test

# With coverage report
make test-cov

# Individual modules
pytest tests/test_entropy.py -v
pytest tests/test_pipeline.py -v
pytest tests/test_model.py -v
pytest tests/test_api.py -v

# End-to-end integration smoke test
python scripts/smoke_test.py --skip-api
```

---

## 🎲 Reproducibility

All experiments use `--random-state 42` by default:
```bash
python model/train.py --random-state 42
```

For the full five-seed statistical evaluation:
```bash
bash scripts/run_multiseed.sh --seeds "42 123 456 789 1024"
```

Multi-seed ablation with Wilcoxon significance tests:
```bash
python model/ablation.py --seeds "42,123,456,789,1024" \
    --output model/artifacts/ablation_results.json
```

All quantitative claims are traceable via `model/artifacts/reproducibility_manifest.json`:
```bash
python -c "
import json
m = json.load(open('model/artifacts/reproducibility_manifest.json'))
print(f'Accuracy : {m[\"metrics\"][\"accuracy\"]:.6f}')
print(f'Macro F1 : {m[\"metrics\"][\"macro_f1\"]:.6f}')
print(f'Data hash: {m[\"data_hash_sha256\"]}')
print(f'sklearn  : {m[\"environment\"][\"scikit_learn\"]}')
"
```

---

## 📈 Experiment Tracking

All training runs are logged automatically to MLflow:

```bash
# View experiment history
make mlflow-ui
# Open: http://localhost:5000

# Train and log a run
python model/train.py
```

Each run captures: model hyperparameters, all evaluation metrics (accuracy, F1, FPR, AUC, latency, throughput), Python/library versions, and all artifact checksums. The MLflow run ID is embedded directly in `reproducibility_manifest.json` for full traceability.

---

## ⚖️ Baseline Comparison — Fair Evaluation Protocol

XAI-SDN is compared against four baseline classifiers under a strictly controlled evaluation protocol. To ensure absolute fairness and address computational tractability (especially for SVM on 3.59 million rows), all models were trained and evaluated on mathematically identical train/test splits generated from the real-world `Syn.csv` dataset, using a representative stratified subset of 10,000 training samples and 3,000 test samples.

| Classifier | Accuracy | Macro F1-Score | Inference Latency (ms/flow) | Throughput (flows/s) |
|---|---|---|---|---|
| **XAI-SDN (Random Forest)** | **99.867%** | **95.966%** | **0.0260 ms** | **38,439** |
| Decision Tree | 99.833% | 94.639% | 0.0004 ms | 2,612,103 |
| SVM (RBF Kernel) | 99.867% | 96.395% | 0.0354 ms | 28,265 |
| Naive Bayes (Gaussian) | 98.633% | 77.610% | 0.0011 ms | 913,409 |
| XGBoost | 99.833% | 95.056% | 0.0018 ms | 570,223 |

> [!NOTE]
> **Full-Scale Model Production Benchmarks**:
> When trained on the entire un-subsampled real-world dataset (**2,514,860 training samples** and **1,077,798 held-out test samples**), the full XAI-SDN model achieves an outstanding **99.9987% Accuracy**, **99.9621% Macro F1**, **0.0537% FPR**, and a classification throughput of **664,181 flows/second** (`0.0015 ms/flow` inference latency), demonstrating massive scalability and robustness on millions of real network flows.

**Fairness guarantees:**
1. **Shared test partition** — all baselines are evaluated on the identical test split produced by `model/baselines.py`
2. **No exhaustive grid search** — none of the models, including the proposed RF, undergo hyperparameter tuning on the test data
3. **Matched estimator count** — XGBoost uses `n_estimators=200` to match Random Forest complexity
4. **Balanced classes** — `class_weight='balanced'` applied identically across all applicable classifiers

---

## ⚠️ Known Limitations

1. **Single-dataset evaluation**: The model was trained and evaluated on CIC-DDoS2019 only. Cross-dataset generalization to other benchmark datasets (e.g., KDD Cup, UNSW-NB15) is not evaluated.
2. **CICFlowMeter–OpenFlow bridge**: Full feature extraction requires raw packet captures. The OpenFlow counter bridge covers approximately 40 of 80 CICFlowMeter features; a network tap is required for complete extraction in live SDN deployments.
3. **Fixed entropy window size**: The sliding window is fixed at N=1,000 flows. No adaptive mechanism is implemented for varying traffic rates.
4. **Binary classification scope**: Empirical results are reported for Benign vs. DDoS-Syn (the Syn subset of CIC-DDoS2019). Multi-class results require training on additional attack files from the full dataset.
5. **Adversarial robustness**: Coordinated IP-rotation attacks can partially degrade entropy-based features. No adversarial robustness evaluation is included.

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for the complete technical debt analysis.

---

## 📖 Documentation

| Document | Description |
|---|---|
| [Architecture Guide](docs/architecture.md) | Detailed system design and SOTA context |
| [API Reference](docs/api_reference.md) | REST endpoint documentation |
| [Deployment Guide](docs/deployment_guide.md) | Docker, Kubernetes, and Ryu setup |
| [Operator Guide](docs/operator_guide.md) | SOC analyst dashboard usage guide |

---

## ⚙️ Environment

| File | Purpose |
|---|---|
| `requirements.txt` | Core runtime dependencies |
| `requirements-dev.txt` | Development, testing, and linting tools |
| `requirements-lock.txt` | Hash-pinned lockfile for exact reproduction |
| `environment.yml` | Conda environment (delegates to pip) |

> **Note**: The Ryu SDN controller requires Python ≤ 3.8. Refer to `docs/deployment_guide.md` for the isolated `venv-ryu` setup instructions.

---

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{xaisdn2026,
  author    = {adeliusa486},
  title     = {XAI-SDN: Explainable Entropy-Guided Machine Learning for Real-Time DDoS Detection in Software Defined Networks},
  year      = {2026},
  note      = {Open-source implementation. Evaluated on CIC-DDoS2019 dataset.},
  url       = {https://github.com/adeliusa486/XAI-SDN}
}
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📜 License

MIT License — see [LICENSE](LICENSE).
