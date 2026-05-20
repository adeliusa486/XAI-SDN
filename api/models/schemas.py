"""
schemas.py — Pydantic Data Models for XAI-SDN REST API.

Defines request/response schemas for alert ingestion, retrieval,
and SHAP attribution queries.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class AttackLabel(str, Enum):
    BENIGN = "Benign"
    DDOS_UDP = "DDoS-UDP"
    DDOS_TCP = "DDoS-TCP"
    DDOS_ICMP = "DDoS-ICMP"
    DDOS_SLOWLORIS = "DDoS-SlowLoris"
    DDOS_HTTP = "DDoS-HTTP"
    UNKNOWN = "Unknown"


class SHAPFeatureAttribution(BaseModel):
    """Per-feature Shapley value attribution."""

    feature: str = Field(..., description="Feature name")
    shap_value: float = Field(
        ..., description="SHAP value (positive = toward DDoS, negative = toward Benign)"
    )
    abs_shap: float = Field(..., description="Absolute SHAP value (feature importance magnitude)")


class AlertCreate(BaseModel):
    """Schema for creating/ingesting a new DDoS alert."""

    flow_id: str = Field(..., description="Unique flow identifier")
    src_ip: str = Field(..., description="Source IP address")
    dst_ip: str = Field(..., description="Destination IP address")
    src_port: int = Field(0, ge=0, le=65535)
    dst_port: int = Field(0, ge=0, le=65535)
    protocol: int = Field(0, ge=0, le=255, description="IP protocol number")
    label: AttackLabel = Field(..., description="Predicted DDoS attack type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model prediction confidence")
    switch_id: Optional[str] = Field(None, description="Source OpenFlow switch DPID")
    flow_duration_ms: Optional[float] = Field(None, description="Flow duration in milliseconds")
    packet_count: Optional[int] = Field(None, description="Total packets in flow")
    byte_count: Optional[int] = Field(None, description="Total bytes in flow")

    # SHAP attribution
    shap_top_features: Optional[List[SHAPFeatureAttribution]] = Field(
        None, description="Top SHAP feature attributions (ranked by |φ|)"
    )
    shap_full_attribution: Optional[Dict[str, float]] = Field(
        None, description="Full 88-feature SHAP attribution vector"
    )
    shap_base_value: Optional[float] = Field(
        None, description="SHAP base value (expected model output)"
    )

    # Feature snapshot for audit
    feature_vector: Optional[Dict[str, float]] = Field(
        None, description="88-dim feature vector that triggered the alert"
    )

    timestamp: Optional[datetime] = Field(
        default_factory=datetime.utcnow, description="Alert generation timestamp (UTC)"
    )

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_valid(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class AlertResponse(AlertCreate):
    """Alert as returned from the API (includes DB-assigned ID)."""

    id: int = Field(..., description="Auto-assigned alert database ID")
    created_at: datetime = Field(..., description="Database insertion timestamp")

    class Config:
        from_attributes = True


class AlertListResponse(BaseModel):
    """Paginated list of alerts."""

    alerts: List[AlertResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class AlertFilterParams(BaseModel):
    """Query parameters for filtering alerts."""

    label: Optional[AttackLabel] = None
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)
    src_ip: Optional[str] = None
    dst_port: Optional[int] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=500)


class InferenceRequest(BaseModel):
    """Request schema for on-demand flow inference."""

    feature_vector: Dict[str, float] = Field(
        ..., description="Named feature values (88-dim CIC + entropy)"
    )
    flow_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata (src_ip, dst_ip, ports) for alert context"
    )
    compute_shap: bool = Field(
        True, description="Whether to compute SHAP attribution for DDoS predictions"
    )


class InferenceResponse(BaseModel):
    """Response from on-demand flow inference."""

    label: AttackLabel
    confidence: float
    is_ddos: bool
    shap_top_features: Optional[List[SHAPFeatureAttribution]] = None
    shap_full_attribution: Optional[Dict[str, float]] = None
    inference_latency_ms: float = Field(..., description="Server-side inference latency")


class ModelInfo(BaseModel):
    """Model metadata for the /model/info endpoint."""

    model_type: str
    n_estimators: int
    n_features: int
    feature_names: List[str]
    classes: List[str]
    training_accuracy: Optional[float] = None
    training_macro_f1: Optional[float] = None
    shap_enabled: bool
    artifacts_dir: str


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    version: str
    model_loaded: bool
    shap_ready: bool
    database_connected: bool
    alert_count: int
    uptime_seconds: float


class StatsResponse(BaseModel):
    """Alert statistics summary."""

    total_alerts: int
    alerts_last_hour: int
    alerts_last_24h: int
    by_label: Dict[str, int]
    avg_confidence: float
    top_src_ips: List[Dict[str, Any]]
    top_dst_ports: List[Dict[str, Any]]
