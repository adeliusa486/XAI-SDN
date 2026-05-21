"""
alerts.py — Alert Ingestion and Retrieval Endpoints.

POST /api/v1/alerts          — Ingest a new DDoS alert (from Ryu controller)
GET  /api/v1/alerts          — List/filter stored alerts (paginated)
GET  /api/v1/alerts/{id}     — Get single alert by ID
DELETE /api/v1/alerts/{id}   — Delete single alert (admin)
GET  /api/v1/alerts/stats    — Alert statistics summary
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import AlertDB, get_app_state, get_db_session
from api.models.schemas import (
    AlertCreate,
    AlertListResponse,
    AlertResponse,
    AttackLabel,
    StatsResponse,
)

router = APIRouter()


# ─── Ingest Alert ─────────────────────────────────────────────────────────────


@router.post("/", response_model=AlertResponse, status_code=201)
async def create_alert(
    alert: AlertCreate,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """Ingest a DDoS alert from the Ryu SDN controller.

    This endpoint is called by the controller's alert_dispatcher module
    whenever a flow is classified as DDoS with confidence ≥ τ.

    **Security note**: SHAP attribution values in the response must not
    be forwarded to untrusted clients — they could guide adversarial evasion.
    """
    db_alert = AlertDB(
        flow_id=alert.flow_id,
        src_ip=alert.src_ip,
        dst_ip=alert.dst_ip,
        src_port=alert.src_port,
        dst_port=alert.dst_port,
        protocol=alert.protocol,
        label=alert.label.value,
        confidence=alert.confidence,
        switch_id=alert.switch_id,
        flow_duration_ms=alert.flow_duration_ms,
        packet_count=alert.packet_count,
        byte_count=alert.byte_count,
        shap_top_features=(
            json.dumps([f.model_dump() for f in alert.shap_top_features])
            if alert.shap_top_features
            else None
        ),
        shap_full_attribution=(
            json.dumps(alert.shap_full_attribution) if alert.shap_full_attribution else None
        ),
        feature_vector=json.dumps(alert.feature_vector) if alert.feature_vector else None,
        timestamp=alert.timestamp or datetime.utcnow(),
        created_at=datetime.utcnow(),
    )

    db.add(db_alert)
    await db.commit()
    await db.refresh(db_alert)

    logger.info(
        f"Alert ingested: id={db_alert.id} label={db_alert.label} "
        f"src={db_alert.src_ip} dst={db_alert.dst_ip}:{db_alert.dst_port} "
        f"confidence={db_alert.confidence:.2f}"
    )

    return _db_to_response(db_alert)


# ─── List Alerts ──────────────────────────────────────────────────────────────


@router.get("/", response_model=AlertListResponse)
async def list_alerts(
    label: Optional[AttackLabel] = Query(None, description="Filter by attack label"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    src_ip: Optional[str] = Query(None),
    dst_port: Optional[int] = Query(None),
    since: Optional[datetime] = Query(None, description="ISO datetime string"),
    until: Optional[datetime] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db_session),
):
    """List stored DDoS alerts with optional filtering and pagination."""
    conditions = []

    if label is not None:
        conditions.append(AlertDB.label == label.value)
    if min_confidence > 0:
        conditions.append(AlertDB.confidence >= min_confidence)
    if src_ip is not None:
        conditions.append(AlertDB.src_ip == src_ip)
    if dst_port is not None:
        conditions.append(AlertDB.dst_port == dst_port)
    if since is not None:
        conditions.append(AlertDB.timestamp >= since)
    if until is not None:
        conditions.append(AlertDB.timestamp <= until)

    where_clause = and_(*conditions) if conditions else True

    # Count
    count_q = select(func.count(AlertDB.id)).where(where_clause)
    total_result = await db.execute(count_q)
    total = total_result.scalar() or 0

    # Fetch page
    offset = (page - 1) * page_size
    query = (
        select(AlertDB)
        .where(where_clause)
        .order_by(AlertDB.timestamp.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(query)
    rows = result.scalars().all()

    return AlertListResponse(
        alerts=[_db_to_response(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + len(rows)) < total,
    )


# ─── Get Single Alert ─────────────────────────────────────────────────────────


@router.get("/stats", response_model=StatsResponse)
async def get_stats(db: AsyncSession = Depends(get_db_session)):
    """Return aggregated alert statistics for the dashboard."""
    now = datetime.utcnow()
    hour_ago = now - timedelta(hours=1)
    day_ago = now - timedelta(days=1)

    total_r = await db.execute(select(func.count(AlertDB.id)))
    total = total_r.scalar() or 0

    h1_r = await db.execute(select(func.count(AlertDB.id)).where(AlertDB.timestamp >= hour_ago))
    alerts_1h = h1_r.scalar() or 0

    d1_r = await db.execute(select(func.count(AlertDB.id)).where(AlertDB.timestamp >= day_ago))
    alerts_24h = d1_r.scalar() or 0

    label_r = await db.execute(
        select(AlertDB.label, func.count(AlertDB.id)).group_by(AlertDB.label)
    )
    by_label = {row[0]: row[1] for row in label_r.all()}

    avg_conf_r = await db.execute(select(func.avg(AlertDB.confidence)))
    avg_conf = float(avg_conf_r.scalar() or 0.0)

    # Top source IPs
    top_src_r = await db.execute(
        select(AlertDB.src_ip, func.count(AlertDB.id).label("count"))
        .group_by(AlertDB.src_ip)
        .order_by(func.count(AlertDB.id).desc())
        .limit(10)
    )
    top_src_ips = [{"ip": r[0], "count": r[1]} for r in top_src_r.all()]

    # Top destination ports
    top_port_r = await db.execute(
        select(AlertDB.dst_port, func.count(AlertDB.id).label("count"))
        .group_by(AlertDB.dst_port)
        .order_by(func.count(AlertDB.id).desc())
        .limit(10)
    )
    top_dst_ports = [{"port": r[0], "count": r[1]} for r in top_port_r.all()]

    return StatsResponse(
        total_alerts=total,
        alerts_last_hour=alerts_1h,
        alerts_last_24h=alerts_24h,
        by_label=by_label,
        avg_confidence=round(avg_conf, 4),
        top_src_ips=top_src_ips,
        top_dst_ports=top_dst_ports,
    )


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Get a single alert by ID including full SHAP attribution."""
    result = await db.execute(select(AlertDB).where(AlertDB.id == alert_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return _db_to_response(row)


@router.delete("/{alert_id}", status_code=204)
async def delete_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Delete an alert by ID."""
    result = await db.execute(select(AlertDB).where(AlertDB.id == alert_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    await db.delete(row)
    await db.commit()


# ─── Helper ───────────────────────────────────────────────────────────────────


def _db_to_response(row: AlertDB) -> AlertResponse:
    """Convert AlertDB ORM object to AlertResponse schema."""
    shap_top = None
    if row.shap_top_features:
        try:
            shap_top = json.loads(row.shap_top_features)
        except Exception:
            pass

    shap_full = None
    if row.shap_full_attribution:
        try:
            shap_full = json.loads(row.shap_full_attribution)
        except Exception:
            pass

    feat_vec = None
    if row.feature_vector:
        try:
            feat_vec = json.loads(row.feature_vector)
        except Exception:
            pass

    return AlertResponse(
        id=row.id,
        flow_id=row.flow_id,
        src_ip=row.src_ip,
        dst_ip=row.dst_ip,
        src_port=row.src_port,
        dst_port=row.dst_port,
        protocol=row.protocol,
        label=AttackLabel(row.label),
        confidence=row.confidence,
        switch_id=row.switch_id,
        flow_duration_ms=row.flow_duration_ms,
        packet_count=row.packet_count,
        byte_count=row.byte_count,
        shap_top_features=shap_top,
        shap_full_attribution=shap_full,
        feature_vector=feat_vec,
        timestamp=row.timestamp,
        created_at=row.created_at,
    )
