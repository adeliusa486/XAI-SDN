# XAI-SDN Architecture Guide

## Overview

XAI-SDN implements a three-tier explainable DDoS detection system operating
entirely within the SDN control plane, without requiring dedicated hardware or
out-of-band network taps.

```
┌─────────────────────────────────────────────────────────────┐
│                     SDN DATA PLANE                          │
│  OpenFlow Switches (OVS) ──── Flows ──→ Ryu Controller     │
└──────────────────────────────────────────┬──────────────────┘
                                           │ FlowStats (500ms)
                          ┌────────────────▼────────────────┐
                          │       FEATURE ENGINE             │
                          │  CICFlowMeter Bridge (80 feats)  │
                          │  + Shannon Entropy (8 feats)     │
                          │  = 88-dim feature vector         │
                          └────────────────┬────────────────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │    RANDOM FOREST CLASSIFIER      │
                          │  200 trees, sqrt(88)≈9 features  │
                          │  per split, balanced class weight│
                          │  Confidence ≥ τ=0.70 → DDoS     │
                          └────────────────┬────────────────┘
                                           │ DDoS detected
                          ┌────────────────▼────────────────┐
                          │        TreeSHAP EXPLAINER        │
                          │  φᵢ for each of 88 features     │
                          │  Top-10 ranked by |φᵢ|          │
                          └────────────────┬────────────────┘
                                           │ Structured alert
                          ┌────────────────▼────────────────┐
                          │          FastAPI + DB             │
                          │  POST /api/v1/alerts             │
                          │  SQLite (dev) / Postgres (prod)  │
                          └────────────────┬────────────────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │      Streamlit Dashboard          │
                          │  Real-time alert feed            │
                          │  SHAP waterfall charts           │
                          │  Entropy heatmaps                │
                          └─────────────────────────────────┘
```

## Component Details

### 1. Feature Engineering (88-dim vector)

**CICFlowMeter (80 features)**

CICFlowMeter computes statistical features over bidirectional network flows,
including:
- Packet length statistics (min, max, mean, std — per direction)
- Flow IAT (inter-arrival time) statistics
- TCP flag counts (SYN, ACK, FIN, RST, PSH, URG)
- Byte and packet rates
- Subflow features and window sizes

In the offline (training) mode, pre-extracted CSVs from CIC-DDoS2019 are
used directly. In online (production) mode, an OpenFlow bridge derives
approximately 40/80 features from counters; the remaining require raw PCAP.

**Shannon Entropy (8 features)**

For a sliding window of N=1000 recent flows, entropy is computed over:

```
H(X) = -Σ p(xᵢ) log₂ p(xᵢ)
```

| Feature | Attribute | DDoS Indicator |
|---------|-----------|----------------|
| H_src_ip | Source IP diversity | Low → botnet |
| H_dst_ip | Dest IP diversity | Low → single target |
| H_dst_port | Port diversity | Low → single port flood |
| H_proto | Protocol diversity | Low → single-proto flood |
| H_pkt_len | Packet size diversity | Low → fixed-size packets |
| H_iat | IAT diversity | Low → regular flood timing |
| H_tcp_flags | Flag combination diversity | Low → SYN-only flood |
| H_ttl | TTL diversity | Low → spoofed IP cluster |

### 2. Random Forest Classifier

**Hyperparameters (paper values)**:
- n_estimators = 200
- max_features = sqrt(88) ≈ 9
- class_weight = balanced
- bootstrap = True
- max_depth = None (fully grown)

**Design rationale**:
- Random Forest over DNN: 2.3ms latency vs. ~8ms for equivalent MLP
- Fully grown trees: no underfitting for complex DDoS patterns
- Balanced class weights: corrects for Benign-majority imbalance in CIC data
- Bootstrap sampling: reduces variance across attack type diversity

### 3. SHAP Explainability

**TreeSHAP** (Lundberg et al., 2020) provides exact Shapley values in
polynomial time for tree ensembles.

For a prediction f(x) on flow x:
```
f(x) = φ₀ + Σᵢ φᵢ
```

where φᵢ is feature i's marginal contribution to the prediction.

**Invocation strategy**: SHAP is only computed for flows classified as DDoS
with confidence p ≥ τ=0.70. This represents ~5% of flows in practice,
limiting average overhead to <0.09ms per flow.

**Alert structure**:
```json
{
  "label": "DDoS-UDP",
  "confidence": 0.95,
  "shap_top_features": [
    {"feature": "H_src_ip",      "shap_value": 0.412},
    {"feature": "H_dst_port",    "shap_value": 0.298},
    {"feature": "Flow_Bytes_s",  "shap_value": 0.187}
  ]
}
```

### 4. SDN Controller Integration

The Ryu application `xai_sdn_app.py` runs as a layer-2 learning switch with
a background statistics polling thread.

**Flow lifecycle**:
1. Table-miss → packet_in → install L2 rule
2. Background thread → OFPFlowStatsRequest every 500ms
3. FlowStatsReply → feature extraction → inference → (SHAP if DDoS)
4. Alert → HTTP POST to FastAPI

**Limitations**:
- OpenFlow only exposes aggregate counters, not per-packet data
- ~40/80 CICFlowMeter features are approximated from counters
- A network tap (port mirror to CICFlowMeter node) enables full feature coverage

### 5. Database Schema

Alerts are stored in SQLite (dev) or PostgreSQL (prod):

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-assigned |
| flow_id | VARCHAR | UUID from controller |
| src_ip, dst_ip | VARCHAR | Flow endpoints |
| src_port, dst_port | INTEGER | Port numbers |
| protocol | INTEGER | IP protocol |
| label | VARCHAR | Attack class |
| confidence | FLOAT | Model confidence |
| switch_id | VARCHAR | Source DPID |
| shap_top_features | JSON TEXT | Top-10 SHAP values |
| shap_full_attribution | JSON TEXT | Full 88-feature SHAP |
| feature_vector | JSON TEXT | Feature snapshot |
| timestamp | DATETIME | Flow timestamp |

## Performance Characteristics

| Metric | Value | Condition |
|--------|-------|-----------|
| Per-flow latency (CIC+Entropy+RF) | 2.3ms | 88-dim, 200 trees |
| SHAP per-invocation | ~1.8ms | 88-dim, 200 trees |
| Effective SHAP overhead | <0.09ms avg | Invoked for ~5% of flows |
| Throughput | 15,200 flows/s | Single-threaded Python |
| Entropy window update | O(1) | deque(maxlen=N) |
| Entropy computation | O(N) | N=1000 window scan |

## Security Considerations

1. **SHAP exposure**: SHAP values reveal which features cause DDoS
   classification. Exposing these externally could guide evasion attacks.
   The `/infer` endpoint should be restricted to internal/operator use.

2. **IP spoofing**: Entropy features are partially vulnerable to IP rotation.
   An attacker cycling through IPs can maintain H_src_ip above threshold.

3. **Confidence threshold τ**: At τ=0.70, the system has FPR=0.48%.
   Lowering τ reduces FNR but increases FPR — tune per deployment.

4. **OpenFlow security**: Ryu should communicate with switches over TLS
   (OpenFlow TLS) to prevent controller impersonation.

## Deployment Recommendations

**Development**: `docker compose up -d` — SQLite, single-worker API

**Production**:
- Replace SQLite with PostgreSQL (connection pool, concurrent writes)
- Run API with multiple uvicorn workers behind nginx
- Deploy Ryu controller with OpenFlow TLS
- Use a network tap for full CICFlowMeter feature coverage
- Enable Prometheus/Grafana monitoring
- Set SHAP_ENDPOINT_INTERNAL_ONLY=1

See [deployment_guide.md](deployment_guide.md) for full production setup.

## ⚖️ Baseline Fairness Methodology

When comparing XAI-SDN against baseline architectures (Decision Tree, SVM, Naive Bayes, XGBoost, DNN, LSTM):

1. **Test Set Invariance**: All baseline models are evaluated against the exact same test split (saved as `X_test.npy` and `y_test.npy` during RF training). This eliminates partition bias.
2. **Hyperparameter Tuning Policy**: No exhaustive grid search or randomized search is applied to any model (including the proposed Random Forest). This strict "no-tuning" policy prevents selection bias where the proposed model receives more optimization effort than the baselines.
3. **Parameter Matching**: Where parameters overlap between the proposed model and baselines, they are locked to identical values. For example, `n_estimators=200` is enforced for both RF and XGBoost.
4. **Class Weights**: To handle the class imbalance inherent in the CIC-DDoS2019 dataset, `class_weight='balanced'` is applied uniformly to all baseline classifiers that support it (DT, SVM, RF).
