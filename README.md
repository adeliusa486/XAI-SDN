# XAI-SDN: Explainable Entropy-Guided ML for Real-Time DDoS Detection in SDNs

[![CI](https://github.com/yourusername/xai-sdn/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/xai-sdn/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **An end-to-end explainable machine learning framework for real-time DDoS detection in Software Defined Networks, combining Shannon entropy feature augmentation, Random Forest classification, and TreeSHAP per-prediction attribution.**

---

## 🔑 Key Results (Synthetic Demo — 5 seeds)

| Metric | Mean ± Std (5 seeds) |
|--------|----------------------|
| Accuracy | 1.0000 ± 0.0000 |
| Macro F1 | 1.0000 ± 0.0000 |

For real-data results, see `model/artifacts/multiseed_real/aggregate_results.json`.

†Full pipeline includes entropy window update, CIC feature extraction from
OpenFlow counters, RF inference, and HTTP alert dispatch. Hardware: [specify
CPU, RAM, Python version from manifest].

---

## 🏗️ Architecture

```
SDN Switches (OpenFlow) → Flow Collector → CICFlowMeter Extraction
→ Entropy Augmentation (8 features, N=1000 window) → Random Forest (200 trees)
→ [TreeSHAP if DDoS, p ≥ 0.70] → Structured Alert → Security Dashboard
```

**Feature Space**: 80 CICFlowMeter statistical features + 8 Shannon entropy features = **88-dimensional vector**

**Attack Classes**: Benign, DDoS-UDP, DDoS-TCP-SYN, DDoS-ICMP, DDoS-SlowLoris, DDoS-HTTP-Flood

---

## 🚀 Quickstart

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/yourusername/xai-sdn.git
cd xai-sdn
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
# Create conda environment
conda env create -f environment.yml
conda activate xai-sdn

# Or use pip (recommended for dev):
pip install -r requirements.txt -r requirements-dev.txt

# Or install with hash verification for exact reproduction:
pip install --require-hashes -r requirements-lock.txt

# Generate synthetic data and train a demo model
python scripts/generate_synthetic_data.py
python model/train.py --config configs/model_config.yaml --use-synthetic

# Run the API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run the dashboard
streamlit run dashboard/app.py
```

### Option 3: With Real CIC-DDoS2019 Dataset

```bash
# 1. Download dataset (requires registration at UNB)
# See: https://www.unb.ca/cic/datasets/ddos-2019.html
# Place CSV files in data/raw/

# 2. Preprocess
python scripts/preprocess_data.py

# 3. Train
python model/train.py --config configs/model_config.yaml

# 4. Evaluate
python model/evaluate.py --config configs/model_config.yaml

# 5. Run SHAP analysis (uses saved test split from step 3)
python explainability/global_importance.py \
    --artifacts-dir model/artifacts \
    --output-dir model/artifacts/shap \
    --max-samples 2000
```

---

## 📁 Repository Structure

```
xai-sdn/
├── configs/                    # All YAML configuration files
│   ├── config.yaml             # Master config
│   ├── model_config.yaml       # ML training hyperparameters
│   └── deployment_config.yaml  # Production deployment settings
├── data/                       # Data directory (raw CSVs gitignored)
├── features/                   # Feature engineering
│   ├── cicflowmeter.py         # CICFlowMeter-compatible feature extraction
│   ├── entropy.py              # Shannon entropy module
│   └── pipeline.py             # Full offline + online feature pipeline
├── model/                      # ML model components
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation + metrics
│   ├── baselines.py            # Baseline classifiers (DT, SVM, DNN, LSTM, XGB)
│   └── ablation.py             # Ablation study runner
├── explainability/             # SHAP explainability layer
│   ├── shap_explainer.py       # TreeSHAP wrapper
│   ├── global_importance.py    # Global feature importance
│   └── visualizations.py      # SHAP plot generation
├── sdn/                        # SDN controller integration
│   ├── controller/             # Ryu application modules
│   └── topology/               # Mininet topology + attack generators
├── api/                        # FastAPI REST backend
│   ├── main.py                 # Application entrypoint
│   ├── routes/                 # Endpoint handlers
│   └── models/schemas.py       # Pydantic schemas
├── dashboard/app.py            # Streamlit security dashboard
├── scripts/                    # Utility scripts
├── tests/                      # Full test suite
├── docs/                       # Documentation
├── monitoring/                 # Prometheus + Grafana configs
├── deployment/                 # Kubernetes manifests + Helm chart
└── .github/workflows/          # CI/CD pipelines
```

---

## 📊 Entropy Features

| Feature | Attribute | DDoS Signal |
|---|---|---|
| `H_src_ip` | Source IP diversity | Low = botnet/spoofed IP concentration |
| `H_dst_ip` | Destination IP diversity | Low = single-target attack |
| `H_dst_port` | Destination port diversity | Low = single-port flood |
| `H_proto` | Protocol distribution | Low = single-protocol flood |
| `H_pkt_len` | Packet length distribution | Low = uniform attack packets |
| `H_iat` | Inter-arrival time distribution | Low = regular flood timing |
| `H_tcp_flags` | TCP flag combinations | Low = SYN-only flood |
| `H_ttl` | IP TTL values | Low = spoofed IP concentration |

Computed via: `H(X) = -Σ p(xᵢ) log₂ p(xᵢ)` over a sliding window of N=1000 recent flows.

---

## 🧪 Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific module
pytest tests/test_entropy.py -v
pytest tests/test_model.py -v
pytest tests/test_api.py -v

# Smoke test
python scripts/smoke_test.py
```

---

## 🎲 Reproducibility

All experiments use `--random-state 42` by default. To override:
```bash
XAI_SDN_SEED=123 python model/train.py --use-synthetic
# or
python model/train.py --use-synthetic --random-state 123
```

For multi-seed experiments (Phase 7), use:
```bash
bash scripts/run_multiseed.sh --seeds "42 123 456 789 1024"
```

---

## 📊 Statistical Validity

All comparisons use 5 independent random seeds and Wilcoxon signed-rank tests.
```bash
bash scripts/run_multiseed.sh --synthetic --seeds "42 123 456 789 1024"
python model/ablation.py --use-synthetic --seeds "42,123,456,789,1024"
```
See `model/artifacts/multiseed_synthetic/aggregate_results.json` for significance
test results.

---

## 📈 Experiment Tracking

All training runs are logged to MLflow:

```bash
# Start MLflow UI
make mlflow-ui
# Open: http://localhost:5000

# Run and log training
python model/train.py --use-synthetic
```

Each run logs: hyperparameters, metrics (accuracy, F1, FPR, AUC, latency),
library versions, and all artifacts. Run ID is stored in
`model/artifacts/reproducibility_manifest.json`.

---

## 📖 Documentation

| Document | Description |
|---|---|
| [Architecture Guide](docs/architecture.md) | System design deep-dive |
| [API Reference](docs/api_reference.md) | REST endpoint documentation |
| [Deployment Guide](docs/deployment_guide.md) | Production deployment instructions |
| [Operator Guide](docs/operator_guide.md) | SOC analyst usage guide |

---

## ⚠️ Known Limitations

1. **CICFlowMeter–OpenFlow bridge**: CICFlowMeter requires raw packet captures; a partial feature bridge from OpenFlow counters is implemented but covers only ~40 of 80 features. A network tap is required for full feature extraction in live deployments.
2. **Single dataset evaluation**: Trained on CIC-DDoS2019 only. Cross-dataset generalization is unproven.
3. **No adversarial robustness testing**: IP rotation can partially evade entropy-based features.
4. **Fixed window size N=1000**: No adaptive mechanism for varying traffic rates.
5. **Label encoding**: `LabelEncoder` is fit exclusively on the training partition (post-split). All 6 attack classes appear in both splits due to stratified sampling.

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for full technical debt analysis.

---

## 📦 Repository Artifacts

The following files are **generated** (never committed):
- `SMOKE_TEST_REPORT.md` — produced by CI; download from GitHub Actions artifacts
- `model/artifacts/*.pkl` — produced by `python model/train.py`
- `model/artifacts/X_test.npy` — saved test split (produced by `model/train.py`)
- `model/artifacts/metrics.json` — training metrics (produced by `model/train.py`)

---

## 🔍 Metric Provenance

All quantitative claims are traceable to `model/artifacts/reproducibility_manifest.json`:

```bash
python -c "import json; m=json.load(open('model/artifacts/reproducibility_manifest.json')); \
print(f'Accuracy: {m[\"metrics\"][\"accuracy\"]:.4f}'); \
print(f'Dataset hash: {m[\"data_hash_sha256\"]}'); \
print(f'sklearn: {m[\"environment\"][\"scikit_learn\"]}')"
```

---

## ⚙️ Environment Notes

| File | Purpose |
|------|---------|
| `requirements.txt` | Runtime deps with version bounds |
| `requirements-dev.txt` | Test and lint tools |
| `requirements-lock.txt` | Hash-pinned exact lockfile for reproducibility |
| `environment.yml` | Conda environment (delegates to pip) |

**Ryu SDN controller** requires Python ≤ 3.8. See `docs/deployment_guide.md` for the separate `venv-ryu` setup.

---

## 📄 Citation

```bibtex
@misc{xai-sdn2024,
  title     = {XAI-SDN: Explainable Entropy-Guided Machine Learning for Real-Time DDoS Detection in Software Defined Networks},
  year      = {2024},
  note      = {Open-source implementation},
  url       = {https://github.com/yourusername/xai-sdn}
}
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

MIT License — see [LICENSE](LICENSE).
