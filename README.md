<p align="center">
  <img src="assets/fig10.png" width="600" alt="SHAP beeswarm plot of XAI-SDN feature attributions">
</p>

<h1 align="center">XAI-SDN: Explainable Entropy-Guided Machine Learning for Real-Time DDoS Detection</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.unb.ca/cic/datasets/ddos-2019.html"><img src="https://img.shields.io/badge/dataset-CIC--DDoS2019-orange.svg" alt="Dataset: CIC-DDoS2019"></a>
</p>

This repository contains the codebase for **XAI-SDN**, an explainable machine learning
framework for real-time DDoS detection in Software Defined Networks. It combines a
rolling Shannon entropy feature engine maintained at expected amortized
$\mathcal{O}(1)$ cost per update, a Random Forest classifier, and exact TreeSHAP
attribution, all running inside the controller process.

---

## ⚠️ Corrections in the 2026 revision

**Read this section before using any number from an earlier version of this
repository or of the manuscript.** A revision undertaken in response to peer
review found several errors in the originally reported results. They are listed
here rather than quietly overwritten.

| What was reported | What is correct | Why |
| --- | --- | --- |
| 99.999% accuracy, described as a temporal split | **99.7579% accuracy, 97.8813% macro F1** under a genuine chronological split | The pipeline sorted by timestamp and then called `train_test_split(..., stratify=...)`, discarding the ordering. The published figures came from a stratified random split. |
| 599,052 flows/s "detection throughput" | **27,647 flows/s** batch throughput; **7.05 ms** median single-flow latency | The original figure was an offline feature-extraction rate measured with no control channel and no per-flow inference. |
| 1,953 flows/s with SHAP | **661 flows/s** when every alert is attributed | Re-measured with the explanation trigger defined explicitly. Budgeting the trigger by attack episode restores 35,652 flows/s. |
| Entropy features improve accuracy ($p < 0.01$) | **They do not, in distribution.** The 80 flow statistics alone reach 98.28% macro F1 against 97.91% for the full 88-dimensional vector | The original ablation used stratified random resplits. Under the chronological protocol the effect reverses. |
| `FIN_Flag_Count` among the three most influential features | It is **constant** across all 3,590,794 flows and carries no information | 12 of the 80 exported statistics are constant on this partition. |
| Transfer to InSDN with accuracy 1.0 | **Zero-shot macro F1 0.6026 on InSDN, 0.2788 on CIC-IDS2017** | The file `model/artifacts/insdn_transfer.json` was flagged `"synthetic": true` and was produced from synthetic data by `scripts/run_insdn_transfer.py --use-synthetic`. It was never committed (artifacts are gitignored), was never used in the paper, and has been deleted from the working tree. |
| `H_ttl` among the eight entropy features | **`H_src_port` replaces it.** CICFlowMeter exports no TTL column, so the pipeline supplied a constant 64 and the feature was zero to float tolerance, with a single-feature AUC of 0.500021 | Source-port entropy is exported, is non-degenerate, reaches a single-feature AUC of 0.9876 and carries the highest mutual information of the eight. Dimensionality is unchanged at 88. `features/entropy.py` in this repository still computes the submitted eight, including `H_ttl`, because it reproduces the submitted results; every number in the revision comes from `revision_2026/03_experiments/common/data.py`, which computes the repaired eight. |

Two further findings from the revision that were not in the original work:

- **11.06% of the partition consists of exact duplicate flow records.** Under a
  stratified random split, 15.49% of test rows have a feature-identical twin in
  the training partition, and a lookup table that learns nothing reaches 99.28%
  accuracy. Under a chronological split that falls to 0.04%.
- **97.9% of benign flows occur in the final 30% of the capture.** A chronological
  split therefore leaves 582 benign flows for training against 30,432 at test.
  Any evaluation that splits this corpus randomly obtains a benign class that is
  easy by construction.

The full revision, including 20+ experiments, the response to reviewers, and the
raw result files, is in [`revision_2026/`](revision_2026/).

---

## What the system does

1. **Rolling entropy engine.** Eight Shannon entropy features over a sliding
   window of $N = 1{,}000$ flows, maintained by a circular buffer and a hash map
   of key multiplicities so that each admit/evict pair costs expected amortized
   $\mathcal{O}(1)$ rather than $\mathcal{O}(N)$. Accumulated drift over
   12,000,000 updates is at most $7.5 \times 10^{-14}$ bits and changes zero
   decisions.
2. **Exact per-decision explanation.** Every admitted alert carries a TreeSHAP
   attribution vector whose contributions sum exactly to the score, so an analyst
   can see which features drove a decision rather than inferring it.
3. **A budgeted explanation trigger.** Attributing every positive prediction is
   untenable when 97% of traffic is hostile. The trigger is defined formally and
   three budget policies are measured, giving a throughput-versus-coverage
   frontier rather than a single operating point.
4. **Measured control-plane cost.** The detector runs inside a live OpenFlow 1.3
   controller against Mininet and Open vSwitch, with controller CPU, memory,
   end-to-end detection latency, ingest saturation, and flow-table occupancy all
   measured rather than assumed.

---

## Headline results

All figures below are from the chronological protocol on the CIC-DDoS2019 SYN
partition (03-11), 2,513,556 training and 1,077,238 test flows, measured on one
CPU-only machine.

| Metric | Value |
| --- | --- |
| Accuracy | 99.7579% |
| Macro F1 | 97.8813% |
| False positive rate | 0.0953% (29 false alarms in 1,077,238 flows) |
| False negative rate | 0.2464% |
| ROC-AUC / average precision | 0.999728 / 0.999992 |
| Single-flow latency (p50) | 7.05 ms |
| Batch throughput | 27,647 flows/s |

The same table under a stratified random split gives 99.9981% accuracy and
99.9458% macro F1. Both are reported in the paper so the gap is visible.

---

## Reproducing the results

### 1. Environment

```bash
git clone https://github.com/adeliusa486/xAI-SDN.git
cd XAI-SDN
pip install -r requirements.txt
```

### 2. Dataset

Register and download `CIC-DDoS2019` from the
[UNB Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ddos-2019.html)
and place `Syn.csv` (03-11) into `data/raw/`.

A synthetic generator is provided for a functionality check only:

```bash
python scripts/generate_synthetic_data.py
```

Output produced from synthetic data is never a result. Any artifact generated
this way carries `"synthetic": true` and must not be reported.

### 3. Pipeline

```bash
# Preprocess: deduplication, NaN and infinity removal, rolling entropy
python scripts/preprocess_data.py

# Train (200 trees; see the revision for the depth and size frontier)
python model/train.py --config configs/model_config.yaml

# Evaluate on the held-out chronological test partition
python model/evaluate.py --config configs/model_config.yaml

# Global TreeSHAP attribution
python explainability/global_importance.py \
    --artifacts-dir model/artifacts \
    --output-dir model/artifacts/shap \
    --max-samples 2000

# Baselines, all CPU-only under one harness
python model/baselines.py
```

### 4. Live deployment on Mininet

The controller application, topology and narrated demonstration used for the
supplementary video are in [`sdn/`](sdn/). They require Mininet, Open vSwitch and
os-ken, and hardware virtualization enabled on the host.

```bash
sudo python sdn/run_topology.py          # 1 switch, 3 hosts, TCLink 100 Mbit/1 ms
sudo bash sdn/demo_session.sh            # six-step narrated demonstration
```

---

## Repository structure

```
XAI-SDN/
├── assets/            Figures and diagrams
├── configs/           YAML configuration
├── data/              Data directory (raw CSVs gitignored)
├── explainability/    TreeSHAP wrappers and global importance
├── features/          CICFlowMeter and rolling entropy extraction as
│                   submitted; the revision's repaired feature set is in
│                   revision_2026/03_experiments/common/data.py
├── model/             Training, evaluation, ablation, baselines
├── revision_2026/     The 2026 revision: experiments, raw results,
│                   manuscript, figures, response to reviewers
├── scripts/           Reproducibility and synthetic data scripts
├── sdn/               OpenFlow controller app, Mininet topology, live demo
└── tests/             pytest suite
```

---

## License

MIT. See [LICENSE](LICENSE).
