"""
main.py — FastAPI Application Entry Point for XAI-SDN.

Provides:
  - POST /api/v1/alerts        — Ingest DDoS alert from Ryu controller
  - GET  /api/v1/alerts        — Retrieve/filter stored alerts
  - GET  /api/v1/alerts/{id}   — Get single alert with full SHAP attribution
  - POST /api/v1/infer         — On-demand flow inference (demo/testing)
  - GET  /api/v1/model/info    — Model metadata
  - GET  /api/v1/stats         — Alert statistics
  - GET  /health               — Health check
  - GET  /metrics              — Prometheus metrics

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.dependencies import AppState, get_app_state
from api.routes import alerts, explanations, health

# ─── Application startup / shutdown ──────────────────────────────────────────

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize model + database on startup."""
    logger.info("=" * 60)
    logger.info("XAI-SDN API Server starting...")
    logger.info("=" * 60)

    state = AppState()
    await state.initialize()
    app.state.xaisdn = state

    logger.info("XAI-SDN API Server ready ✓")
    yield

    # Shutdown
    logger.info("Shutting down XAI-SDN API Server...")
    await state.shutdown()


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="XAI-SDN API",
    description=(
        "Explainable DDoS Detection for Software Defined Networks. "
        "Provides real-time DDoS alert ingestion, SHAP attribution retrieval, "
        "and on-demand inference with per-feature explainability."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

import os

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Prometheus Metrics (optional) ───────────────────────────────────────────

try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus metrics endpoint enabled at /metrics")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed; /metrics disabled.")

# ─── Routers ─────────────────────────────────────────────────────────────────

app.include_router(health.router, tags=["Health"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["Alerts"])
app.include_router(explanations.router, prefix="/api/v1", tags=["Explanations & Inference"])


# ─── Root ─────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "XAI-SDN API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


# ─── Global exception handler ─────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )
