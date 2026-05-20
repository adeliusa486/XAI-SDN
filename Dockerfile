# XAI-SDN Dockerfile
# Multi-stage build: Python 3.10, lightweight runtime image

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

LABEL org.opencontainers.image.title="XAI-SDN"
LABEL org.opencontainers.image.description="Explainable DDoS Detection for SDN"
LABEL org.opencontainers.image.version="0.1.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash xaisdn \
    && mkdir -p data logs model/artifacts \
    && chown -R xaisdn:xaisdn /app

USER xaisdn

# Environment defaults (override via docker-compose .env)
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    DATABASE_URL=sqlite+aiosqlite:///data/alerts.db \
    MODEL_ARTIFACTS_DIR=model/artifacts \
    LOG_LEVEL=INFO

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


# ─── Dashboard Stage ──────────────────────────────────────────────────────────
FROM runtime AS dashboard

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]


# ─── Training Stage ───────────────────────────────────────────────────────────
FROM runtime AS trainer

CMD ["python", "model/train.py", "--config", "configs/model_config.yaml", "--use-synthetic"]
