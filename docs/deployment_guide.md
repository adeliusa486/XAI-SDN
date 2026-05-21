# XAI-SDN Deployment Guide

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| Python | 3.10+ |
| Docker | 24+ |
| Docker Compose | v2 |
| RAM | 4 GB minimum, 8 GB recommended |
| Disk | 5 GB (model artifacts + logs + data) |
| OS | Linux (Ubuntu 22.04 recommended) |

For SDN controller integration:
| Component | Requirement |
|-----------|-------------|
| Ryu | Python 3.8 environment (separate venv) |
| Open vSwitch | 2.17+ |
| Mininet | 2.3+ (testing only) |

---

## Option 1: Docker Compose (Recommended)

### Quick Start

```bash
git clone https://github.com/adeliusa486/XAI-SDN.git
cd XAI-SDN
cp .env.example .env
# Edit .env if needed

# Build and start all services
docker compose up -d

# Train model (first time only)
docker compose --profile train run --rm trainer

# Verify
curl http://localhost:8000/health
```

Services:
- **API**: http://localhost:8000 (Swagger: /docs)
- **Dashboard**: http://localhost:8501
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin / admin)

### Stopping

```bash
docker compose down           # Stop containers
docker compose down -v        # Stop + remove volumes (data loss!)
```

---

## Option 2: Local Python Environment

### Install

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/raw data/synthetic model/artifacts logs
```

### Train Model

```bash
# Option A: Synthetic data (no dataset needed)
python model/train.py --use-synthetic

# Option B: Real CIC-DDoS2019 data
# 1. Download from https://www.unb.ca/cic/datasets/ddos-2019.html
# 2. Extract CSV files to data/raw/
python model/train.py --config configs/model_config.yaml
```

### Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Option 3: Ryu SDN Controller (Requires Python 3.8)

The Ryu controller requires Python ≤ 3.8 due to eventlet compatibility.

```bash
# Create separate Python 3.8 environment for Ryu
python3.8 -m venv venv-ryu
source venv-ryu/bin/activate

# Install Ryu and ML dependencies
pip install ryu eventlet requests joblib scikit-learn shap numpy

# Copy model artifacts to accessible location
cp -r model/artifacts /opt/xaisdn/

# Start Ryu controller
MODEL_ARTIFACTS_DIR=/opt/xaisdn/artifacts \
API_URL=http://localhost:8000 \
DETECTION_THRESHOLD=0.70 \
ryu-manager sdn/controller/xai_sdn_app.py
```

### Mininet Test Topology

```bash
# Terminal 1: Start Ryu controller
ryu-manager sdn/controller/xai_sdn_app.py

# Terminal 2: Start Mininet topology
sudo python sdn/topology/mininet_topo.py --controller 127.0.0.1:6653 --cli

# Terminal 3 (Mininet CLI): Launch attack
mininet> atk hping3 --udp -p 53 --flood 10.0.0.1 &
```

---

## Production Hardening

### PostgreSQL (replace SQLite)

```bash
# .env
DATABASE_URL=postgresql+asyncpg://xaisdn:secret@postgres:5432/xaisdn
```

```yaml
# Add to docker-compose.yml
postgres:
  image: postgres:15
  environment:
    POSTGRES_DB: xaisdn
    POSTGRES_USER: xaisdn
    POSTGRES_PASSWORD: secret
  volumes:
    - postgres_data:/var/lib/postgresql/data
```

### TLS / Reverse Proxy (nginx)

```nginx
server {
    listen 443 ssl;
    server_name xaisdn.example.com;
    ssl_certificate /etc/letsencrypt/live/xaisdn.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/xaisdn.example.com/privkey.pem;

    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Multiple API Workers

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

> **Note**: With multiple workers and SQLite, use WAL mode or switch to PostgreSQL.

### Environment Hardening Checklist

- [ ] Change `API_SECRET_KEY` from default
- [ ] Set `ENFORCE_TLS=1`
- [ ] Set `SHAP_ENDPOINT_INTERNAL_ONLY=1`
- [ ] Restrict `/api/v1/infer` to internal network (nginx ACL)
- [ ] Enable OpenFlow TLS between Ryu and switches
- [ ] Set `GRAFANA_ADMIN_PASSWORD` from default `admin`
- [ ] Rotate database credentials
- [ ] Enable log rotation (configured in `configs/config.yaml`)
- [ ] Set up automated model retraining pipeline

---

## Kubernetes Deployment

See `deployment/k8s/` for manifests.

```bash
kubectl apply -f deployment/k8s/
kubectl get pods -n xaisdn
kubectl port-forward svc/xaisdn-api 8000:8000 -n xaisdn
```

---

## Monitoring

### Key Prometheus Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `http_requests_total` | API request count | — |
| `http_request_duration_seconds` | Request latency | p99 > 100ms |
| `xaisdn_alerts_total` | Total DDoS alerts | — |
| `xaisdn_detection_latency_ms` | Per-flow latency | > 5ms |

### Grafana

Import `monitoring/grafana_dashboard.json` for the pre-built XAI-SDN dashboard.
Default credentials: admin / admin (change immediately in production).

---

## Troubleshooting

### Model not loaded at API startup

```
API running in degraded mode — model not found
```

**Fix**: Run training first:
```bash
python model/train.py --use-synthetic
# or
docker compose --profile train run --rm trainer
```

### SHAP disabled / not ready

SHAP requires the `shap` package and a fitted model:
```bash
pip install shap
python model/train.py --use-synthetic
```

### Database locked (SQLite)

Increase WAL timeout or switch to PostgreSQL for concurrent write workloads.

### Ryu ImportError

Ryu is only compatible with Python ≤ 3.8. Use the separate `venv-ryu` environment.
The rest of XAI-SDN (API, training, dashboard) works on Python 3.10+.

### Port conflicts

```bash
# Check what's using port 8000
lsof -i :8000
# Change API port
API_PORT=8001 uvicorn api.main:app --port 8001
```
