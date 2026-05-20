"""Health check endpoints."""

from fastapi import APIRouter, Request
from api.dependencies import get_app_state
from api.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """API health check. Returns model, SHAP, and database status."""
    state = get_app_state(request)

    # Count alerts
    alert_count = 0
    try:
        from sqlalchemy import select, func
        from api.dependencies import AlertDB
        async with state.session_factory() as session:
            result = await session.execute(select(func.count(AlertDB.id)))
            alert_count = result.scalar() or 0
            state._alert_count = alert_count
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if state.model_loaded else "degraded",
        version="0.1.0",
        model_loaded=state.model_loaded,
        shap_ready=state.shap_ready,
        database_connected=state.engine is not None,
        alert_count=alert_count,
        uptime_seconds=round(state.uptime_seconds, 1),
    )
