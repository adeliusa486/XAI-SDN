"""
explanations.py — On-Demand Inference and Model Info Endpoints.

POST /api/v1/infer        — On-demand flow inference with SHAP
GET  /api/v1/model/info   — Model metadata
"""

from __future__ import annotations

import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from api.dependencies import get_app_state
from api.models.schemas import (
    AttackLabel,
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
    SHAPFeatureAttribution,
)

router = APIRouter()


@router.post("/infer", response_model=InferenceResponse)
async def infer_flow(
    request_body: InferenceRequest,
    request: Request,
):
    """Run inference on a submitted feature vector.

    Returns the predicted label, confidence, and (optionally) SHAP attribution.

    **Usage**: Useful for:
     - Testing the model without a live SDN environment
     - Dashboard demo mode
     - API integration testing

    **Note**: In production, inference is triggered by the Ryu controller
    posting alerts to POST /api/v1/alerts; this endpoint is for ad-hoc use.
    """
    state = get_app_state(request)

    if not state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run: python model/train.py --use-synthetic",
        )

    t0 = time.perf_counter()

    try:
        label_str, confidence, class_idx, x_scaled = state.predict(
            request_body.feature_vector
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    inference_latency_ms = (time.perf_counter() - t0) * 1000

    is_ddos = label_str != "Benign"

    # SHAP attribution (only for DDoS, only if requested and explainer is ready)
    shap_top = None
    shap_full = None

    if is_ddos and request_body.compute_shap and state.shap_ready:
        try:
            import numpy as np
            attribution = state.shap_explainer.format_alert_attribution(
                x_scaled, class_idx, top_k=10
            )
            shap_top = [
                SHAPFeatureAttribution(
                    feature=f["feature"],
                    shap_value=f["shap_value"],
                    abs_shap=abs(f["shap_value"]),
                )
                for f in attribution.get("top_features", [])
            ]
            shap_full = attribution.get("full_attribution")
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")

    try:
        label_enum = AttackLabel(label_str)
    except ValueError:
        label_enum = AttackLabel.UNKNOWN

    return InferenceResponse(
        label=label_enum,
        confidence=confidence,
        is_ddos=is_ddos,
        shap_top_features=shap_top,
        shap_full_attribution=shap_full,
        inference_latency_ms=round(inference_latency_ms, 3),
    )


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info(request: Request):
    """Return metadata about the currently loaded model."""
    state = get_app_state(request)

    if not state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    clf = state.clf
    metrics_path = "model/artifacts/metrics.json"

    training_acc = None
    training_f1 = None
    try:
        import json
        from pathlib import Path
        if Path(metrics_path).exists():
            with open(metrics_path) as f:
                m = json.load(f)
            training_acc = m.get("accuracy")
            training_f1 = m.get("macro_f1")
    except Exception:
        pass

    return ModelInfo(
        model_type=type(clf).__name__,
        n_estimators=getattr(clf, "n_estimators", 0),
        n_features=len(state.feature_names),
        feature_names=state.feature_names,
        classes=list(state.label_encoder.classes_) if state.label_encoder else [],
        training_accuracy=training_acc,
        training_macro_f1=training_f1,
        shap_enabled=state.shap_ready,
        artifacts_dir=str("model/artifacts"),
    )
