# XAI-SDN — Full Repository Report

**Repository:** https://github.com/adeliusa486/XAI-SDN
**Report date:** 2026-07-17
**Scope:** Complete inventory of the codebase, experimental artifacts, verified results, reproduction instructions, and publication status.

---

## 1. Project overview

XAI-SDN is a lightweight, interpretable DDoS detection system for Software Defined Networks. It augments the 80 statistical flow features produced by CICFlowMeter with **eight Shannon entropy metrics** maintained by an **O(1) rolling-window algorithm**, classifies the resulting 88-dimensional flow vectors with a **200-tree Random Forest**, and attaches an **exact TreeSHAP attribution** to every alert so that security operators can audit each detection decision.

Evaluated on the complete SYN partition of CIC-DDoS2019 (3.59 M flows, no subsampling), the system attains **99.9987% accuracy, 99.9621% macro F1, AUC-ROC 1.0000**, with 14 total misclassifications across 1,077,798 held-out flows, at **0.0165 ms/flow** (60,606 flows/s) without explanation generation.

## 2. Repository layout

| Path | Purpose |
|---|---|
| `features/` | Feature engineering: `cicflowmeter.py` (80 statistical features), `entropy.py` (O(1) rolling entropy engine), `pipeline.py` (88-dim assembly) |
| `model/` | `train.py`, `evaluate.py`, `ablation.py`, `baselines.py`; trained artifacts under `model/artifacts/` |
| `explainability/` | `shap_explainer.py` (TreeSHAP per-prediction), `global_importance.py`, `visualizations.py` |
| `sdn/` | `controller/xai_sdn_app.py` (Ryu/OpenFlow controller app), `topology/mininet_topo.py` (Mininet test topology) |
| `api/` | FastAPI service: routes for `alerts`, `explanations`, `health`; Pydantic schemas |
| `dashboard/` | Streamlit SOC dashboard (`app.py`) |
| `scripts/` | `preprocess_data.py`, `run_experiment.sh`, `run_multiseed.sh`, `simulate_traffic.py`, `generate_synthetic_data.py`, `smoke_test.py` |
| `tests/` | Pytest suites: `test_entropy.py`, `test_model.py`, `test_pipeline.py`, `test_api.py` |
| `configs/` | `config.yaml`, `model_config.yaml`, `deployment_config.yaml` |
| `deployment/` | Kubernetes manifests (`deployment.yaml`, `service.yaml`); `Dockerfile` and `docker-compose.yml` at root |
| `monitoring/` | Prometheus config and Grafana dashboard JSON |
| `governance/` | `THREAT_MODEL.md` |
| `docs/` | Architecture, API reference, deployment guide, operator guide |
| `notebooks/` | `01_quickstart.ipynb` |
| `data/` | `raw/` (Syn.csv — not tracked), `splits/` (frozen train/test arrays, scaler, label encoder), `synthetic/` |
| `paper/` | **Journal manuscript (Elsevier `elsarticle`): LaTeX source, bibliography, figures, compiled PDFs, editorial report** |
| `mlruns/`, `mlflow.db` | MLflow experiment tracking |
| `.github/workflows/ci.yml` | CI pipeline |

## 3. Experimental artifacts and verified results

All headline numbers are regenerable from archived artifacts in `model/artifacts/`:

- **`metrics.json`** — full-dataset run (temporal split, seed 42): accuracy 0.9999870, macro F1 0.9996209, FPR 0.000537, FNR 8.42e-6, AUC 0.99999998, CV F1 0.99936 ± 0.00019, train time 92.0 s on 2,514,860 flows, RF-only latency 0.0015 ms (664,181 flows/s).
- **`confusion_matrix.csv`** — TN 9,306 / FP 5 / FN 9 / TP 1,068,478 (test n = 1,077,798).
- **`baseline_results.json`** — controlled protocol (10K train / 3K test, seed 42): Decision Tree, SVM (RBF), Naive Bayes, XGBoost, Random Forest with per-model accuracy, macro F1, latency, throughput, and train time.
- **`ablation_multiseed.json`** — 5 seeds (42, 123, 456, 789, 1024) × 4 configurations (entropy-only 8-dim, CIC-only 80-dim, SVM full 88-dim, RF full 88-dim) with Wilcoxon significance blocks.
- **`multiseed/`** — per-seed directories, each with model pickle, scaler, label encoder, metrics, confusion matrix, and a reproducibility manifest; `aggregate_results.json` summarizes.
- **`shap_global_importance.csv` / `.png`** — mean |SHAP| for all 88 features over 2,000 sampled test flows. Top features: Flow_Bytes/s (0.0585), Destination_Port (0.0550), FIN_Flag_Count (0.0495), H_src_ip (0.0288).
- **`shap/`** — per-prediction explanation artifacts.
- `data/splits/` — frozen `X_train/X_test/y_train/y_test` arrays, fitted scaler and label encoder, feature-name manifest, class-distribution record.

**Known integrity notes (also flagged in `paper/EDITORIAL_REPORT.md`):**
1. The paper prose cites a 10-seed Wilcoxon campaign (p = 0.0020/0.0078/0.2500) and 10-seed stratified FPR 5.77% ± 5.23%; the repo archives 5 seeds. The 10-seed run must be re-executed and archived before journal submission.
2. `metrics.json` records n_train = 2,514,860 vs 2,514,862 in the paper text (trivial drift; regenerate or align).
3. The ROC baseline curves in the paper should be regenerated from `roc_curve()` outputs of the archived models.
4. In the offline SYN capture, TTL is constant (115) for all DDoS flows, so H_ttl is degenerate offline; documented in README and paper.

## 4. Reproduction

```bash
python -m venv venv && venv\Scripts\activate        # Windows
pip install -r requirements.txt                       # or requirements-lock.txt for exact pins
python scripts/preprocess_data.py                     # expects data/raw/Syn.csv (CIC-DDoS2019)
python -m model.train                                 # full-dataset training, writes model/artifacts/
python -m model.evaluate                              # headline metrics
python -m model.baselines                             # controlled-protocol baselines
python -m model.ablation                              # multi-seed feature ablation
pytest                                                # unit tests
uvicorn api.main:app                                  # REST API
streamlit run dashboard/app.py                        # SOC dashboard
```

Dataset: CIC-DDoS2019 `Syn.csv` (1.87 GB) from https://www.unb.ca/cic/datasets/ddos-2019.html, placed at `data/raw/Syn.csv` (not tracked in git).

## 5. Publication status

- A conference version of this work was accepted (IEEE conference format); the authors did not register, and the paper is being withdrawn from those proceedings.
- **`paper/`** contains the upgraded **journal manuscript** targeting *Computers & Security* (Elsevier, no submission/publication fee):
  - `XAI-SDN-journal.tex` — final two-column layout (`5p`), compiled to `XAI-SDN-journal.pdf` (9 pp).
  - `XAI-SDN-journal-review.tex` — single-column 12 pt submission layout (`review`), compiled to `XAI-SDN-journal-review.pdf` (29 pp).
  - `references.bib` — cleaned bibliography (deduplicated, DOI-normalized).
  - `EDITORIAL_REPORT.md` — full list of changes versus the conference version and pre-submission action items.
- The journal version adds: threat model and formal problem statement, O(1) rolling-entropy update derivation, per-class metrics, confusion-matrix/latency/ablation/trade-off figures built exclusively from the artifacts in Section 3, deployment-considerations and limitations sections, and Elsevier declarations (CRediT, competing interests, data availability).

## 6. Pre-submission checklist

- [ ] Re-run and archive the 10-seed ablation + stratified-split FPR campaign (`scripts/run_multiseed.sh`), or align prose to 5 archived seeds.
- [ ] Regenerate the ROC figure from actual model outputs.
- [ ] Extend evaluation to at least one more CIC-DDoS2019 partition (UDP/LDAP) — strongest single improvement to acceptance odds.
- [ ] Remove or replace the three tangential citations flagged in the editorial report.
- [ ] Obtain written withdrawal confirmation from the prior conference; disclose in the cover letter.
- [ ] Verify author name spelling ("Hasan Razzaqi" vs "Hassan Ali Razzaqi") and fill DNN train time in the comparison table if available.
