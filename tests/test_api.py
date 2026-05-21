"""Integration tests for the FastAPI application."""

from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

# Use a test-only database
import os

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///data/test_alerts.db"


@pytest.fixture(scope="module")
async def client():
    """Async test client for the FastAPI app."""
    from api.main import app, lifespan

    transport = ASGITransport(app=app)
    async with lifespan(app):
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


class TestHealthEndpoint:
    async def test_health_returns_200(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_health_response_schema(self, client: AsyncClient):
        resp = await client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "shap_ready" in data
        assert "alert_count" in data
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], float)

    async def test_root_endpoint(self, client: AsyncClient):
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert data["name"] == "XAI-SDN API"


class TestAlertEndpoints:
    @pytest.fixture
    def sample_alert_payload(self):
        return {
            "flow_id": "test-flow-001",
            "src_ip": "10.0.1.100",
            "dst_ip": "10.0.0.1",
            "src_port": 12345,
            "dst_port": 53,
            "protocol": 17,
            "label": "DDoS-UDP",
            "confidence": 0.95,
            "switch_id": "0x0000000000000001",
            "packet_count": 1000,
            "byte_count": 64000,
        }

    async def test_create_alert_returns_201(self, client: AsyncClient, sample_alert_payload):
        resp = await client.post("/api/v1/alerts/", json=sample_alert_payload)
        assert resp.status_code == 201

    async def test_created_alert_has_id(self, client: AsyncClient, sample_alert_payload):
        resp = await client.post("/api/v1/alerts/", json=sample_alert_payload)
        data = resp.json()
        assert "id" in data
        assert isinstance(data["id"], int)
        assert data["id"] > 0

    async def test_created_alert_fields_match(self, client: AsyncClient, sample_alert_payload):
        resp = await client.post("/api/v1/alerts/", json=sample_alert_payload)
        data = resp.json()
        assert data["src_ip"] == sample_alert_payload["src_ip"]
        assert data["dst_port"] == sample_alert_payload["dst_port"]
        assert data["label"] == sample_alert_payload["label"]
        assert abs(data["confidence"] - sample_alert_payload["confidence"]) < 1e-6

    async def test_list_alerts_returns_200(self, client: AsyncClient):
        resp = await client.get("/api/v1/alerts/")
        assert resp.status_code == 200

    async def test_list_alerts_pagination(self, client: AsyncClient):
        resp = await client.get("/api/v1/alerts/?page=1&page_size=10")
        data = resp.json()
        assert "alerts" in data
        assert "total" in data
        assert "page" in data
        assert "has_more" in data
        assert isinstance(data["alerts"], list)

    async def test_get_alert_by_id(self, client: AsyncClient, sample_alert_payload):
        # Create first
        create_resp = await client.post("/api/v1/alerts/", json=sample_alert_payload)
        alert_id = create_resp.json()["id"]

        # Retrieve
        resp = await client.get(f"/api/v1/alerts/{alert_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == alert_id

    async def test_get_nonexistent_alert_returns_404(self, client: AsyncClient):
        resp = await client.get("/api/v1/alerts/999999")
        assert resp.status_code == 404

    async def test_list_alerts_filter_by_label(self, client: AsyncClient):
        resp = await client.get("/api/v1/alerts/?label=DDoS-UDP")
        assert resp.status_code == 200
        data = resp.json()
        for alert in data["alerts"]:
            assert alert["label"] == "DDoS-UDP"

    async def test_list_alerts_filter_by_confidence(self, client: AsyncClient):
        resp = await client.get("/api/v1/alerts/?min_confidence=0.9")
        data = resp.json()
        for alert in data["alerts"]:
            assert alert["confidence"] >= 0.9

    async def test_stats_endpoint(self, client: AsyncClient):
        resp = await client.get("/api/v1/alerts/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_alerts" in data
        assert "by_label" in data
        assert "top_src_ips" in data

    async def test_create_alert_with_shap(self, client: AsyncClient):
        payload = {
            "flow_id": "shap-test-001",
            "src_ip": "10.0.1.200",
            "dst_ip": "10.0.0.1",
            "dst_port": 80,
            "protocol": 6,
            "label": "DDoS-TCP",
            "confidence": 0.92,
            "shap_top_features": [
                {"feature": "H_src_ip", "shap_value": 0.412, "abs_shap": 0.412},
                {"feature": "SYN_Flag_Count", "shap_value": 0.298, "abs_shap": 0.298},
            ],
        }
        resp = await client.post("/api/v1/alerts/", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["shap_top_features"] is not None
        assert len(data["shap_top_features"]) == 2

    async def test_invalid_confidence_rejected(self, client: AsyncClient):
        payload = {
            "flow_id": "bad-001",
            "src_ip": "1.1.1.1",
            "dst_ip": "2.2.2.2",
            "label": "DDoS-UDP",
            "confidence": 1.5,  # invalid
        }
        resp = await client.post("/api/v1/alerts/", json=payload)
        assert resp.status_code == 422


class TestDocsEndpoints:
    async def test_swagger_ui_accessible(self, client: AsyncClient):
        resp = await client.get("/docs")
        assert resp.status_code == 200

    async def test_openapi_json_accessible(self, client: AsyncClient):
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "openapi" in data
        assert data["info"]["title"] == "XAI-SDN API"
