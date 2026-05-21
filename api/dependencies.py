"""
dependencies.py — Shared Application State and Dependency Injection.

AppState holds the loaded model, SHAP explainer, and database engine.
It is initialized once at API startup and injected into route handlers.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import Depends, Request
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Integer, Float, String, DateTime, Text, JSON
from datetime import datetime

# ─── Database Models ──────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class AlertDB(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    flow_id: Mapped[str] = mapped_column(String(128), index=True)
    src_ip: Mapped[str] = mapped_column(String(45))
    dst_ip: Mapped[str] = mapped_column(String(45))
    src_port: Mapped[int] = mapped_column(Integer, default=0)
    dst_port: Mapped[int] = mapped_column(Integer, default=0, index=True)
    protocol: Mapped[int] = mapped_column(Integer, default=0)
    label: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    switch_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    flow_duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    packet_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    byte_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shap_top_features: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    shap_full_attribution: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    feature_vector: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ─── Application State ────────────────────────────────────────────────────────


class AppState:
    """Holds all shared application resources: model, explainer, DB."""

    def __init__(self) -> None:
        self.clf = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names: list[str] = []
        self.shap_explainer = None
        self.engine = None
        self.session_factory = None
        self._start_time = time.time()
        self._alert_count = 0

    async def initialize(self) -> None:
        """Load model artifacts and connect to database."""
        await self._load_model()
        await self._init_database()

    async def _load_model(self) -> None:
        artifacts_dir = Path(os.getenv("MODEL_ARTIFACTS_DIR", "model/artifacts"))

        rf_path = artifacts_dir / "rf_model.pkl"
        scaler_path = artifacts_dir / "scaler.pkl"
        le_path = artifacts_dir / "label_encoder.pkl"
        fn_path = artifacts_dir / "feature_names.json"

        if rf_path.exists():
            logger.info(f"Loading model from {artifacts_dir}...")
            self.clf = joblib.load(rf_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(le_path)
            with open(fn_path) as f:
                self.feature_names = json.load(f)
            logger.info(
                f"Model loaded: {type(self.clf).__name__}, {len(self.feature_names)} features"
            )

            # Initialize SHAP
            try:
                from explainability.shap_explainer import SHAPExplainer

                self.shap_explainer = SHAPExplainer(
                    model=self.clf,
                    feature_names=self.feature_names,
                    confidence_threshold=float(os.getenv("DETECTION_THRESHOLD", "0.70")),
                )
                logger.info(f"SHAP explainer ready: {self.shap_explainer.is_ready}")
            except Exception as e:
                logger.warning(f"SHAP init failed: {e}")
        else:
            logger.warning(
                f"No model found at {artifacts_dir}. "
                "API will serve 404 for inference endpoints. "
                "Run: python model/train.py --use-synthetic"
            )

    async def _init_database(self) -> None:
        db_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///data/alerts.db")
        # Ensure data directory exists
        if db_url.startswith("sqlite"):
            db_path = db_url.split("///")[-1]
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Connecting to database: {db_url.split('://')[0]}...")
        self.engine = create_async_engine(db_url, echo=False)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

        # Create tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables ready.")

    async def shutdown(self) -> None:
        if self.engine:
            await self.engine.dispose()

    @property
    def model_loaded(self) -> bool:
        return self.clf is not None

    @property
    def shap_ready(self) -> bool:
        return self.shap_explainer is not None and self.shap_explainer.is_ready

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def predict(self, feature_vector: dict) -> tuple:
        """Run inference on a named feature vector dict.

        Returns:
            (label_str, confidence_float, class_idx_int)
        """
        if self.clf is None:
            raise RuntimeError("Model not loaded.")

        from features.cicflowmeter import CIC_FEATURE_NAMES
        from features.entropy import ENTROPY_FEATURE_NAMES

        all_names = CIC_FEATURE_NAMES + ENTROPY_FEATURE_NAMES

        x = np.array([feature_vector.get(name, 0.0) for name in all_names], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_scaled = self.scaler.transform(x.reshape(1, -1))

        class_idx = int(self.clf.predict(x_scaled)[0])
        confidence = float(self.clf.predict_proba(x_scaled)[0].max())
        label = self.label_encoder.classes_[class_idx]

        return label, confidence, class_idx, x_scaled[0]


# ─── FastAPI Dependency ───────────────────────────────────────────────────────


def get_app_state(request: Request) -> AppState:
    """FastAPI dependency: inject AppState from request state."""
    return request.app.state.xaisdn


async def get_db_session(request: Request) -> AsyncSession:
    """FastAPI dependency: provide a database session."""
    state: AppState = request.app.state.xaisdn
    async with state.session_factory() as session:
        yield session
