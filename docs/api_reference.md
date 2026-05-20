# XAI-SDN REST API Reference

Base URL: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Authentication

Currently no authentication is required (development mode).
In production, add bearer token or mTLS via reverse proxy.

---

## Endpoints

### `GET /health`

Health check. Returns model, SHAP, and database status.

**Response 200**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "model_loaded": true,
  "shap_ready": true,
  "database_connected": true,
  "alert_count": 1247,
  "uptime_seconds": 3600.0
}
```

Status values: `healthy` (model loaded) | `degraded` (model not loaded)

---

### `POST /api/v1/alerts/`

Ingest a new DDoS alert from the Ryu controller.

**Request body**
```json
{
  "flow_id": "550e8400-e29b-41d4-a716-446655440000",
  "src_ip": "10.0.1.100",
  "dst_ip": "10.0.0.1",
  "src_port": 12345,
  "dst_port": 53,
  "protocol": 17,
  "label": "DDoS-UDP",
  "confidence": 0.953,
  "switch_id": "0x0000000000000001",
  "flow_duration_ms": 1000.0,
  "packet_count": 5000,
  "byte_count": 320000,
  "shap_top_features": [
    {"feature": "H_src_ip", "shap_value": 0.412, "abs_shap": 0.412},
    {"feature": "H_dst_port", "shap_value": 0.298, "abs_shap": 0.298}
  ],
  "shap_full_attribution": {
    "H_src_ip": 0.412,
    "H_dst_port": 0.298,
    "Flow_Bytes_s": 0.187
  }
}
```

**Label values**: `Benign` | `DDoS-UDP` | `DDoS-TCP` | `DDoS-ICMP` | `DDoS-SlowLoris` | `DDoS-HTTP` | `Unknown`

**Response 201** — returns created alert with `id` and `created_at`.

---

### `GET /api/v1/alerts/`

List stored alerts with optional filtering.

**Query parameters**

| Param | Type | Description |
|-------|------|-------------|
| `label` | string | Filter by attack label |
| `min_confidence` | float | Minimum confidence (0.0–1.0) |
| `src_ip` | string | Filter by source IP |
| `dst_port` | int | Filter by destination port |
| `since` | datetime | ISO 8601 start datetime |
| `until` | datetime | ISO 8601 end datetime |
| `page` | int | Page number (default: 1) |
| `page_size` | int | Items per page (default: 50, max: 500) |

**Response 200**
```json
{
  "alerts": [...],
  "total": 1247,
  "page": 1,
  "page_size": 50,
  "has_more": true
}
```

---

### `GET /api/v1/alerts/stats`

Aggregated alert statistics for the dashboard.

**Response 200**
```json
{
  "total_alerts": 1247,
  "alerts_last_hour": 87,
  "alerts_last_24h": 412,
  "by_label": {
    "DDoS-UDP": 523,
    "DDoS-TCP": 489,
    "DDoS-ICMP": 235
  },
  "avg_confidence": 0.9312,
  "top_src_ips": [
    {"ip": "10.0.1.100", "count": 312},
    {"ip": "10.0.1.101", "count": 287}
  ],
  "top_dst_ports": [
    {"port": 53, "count": 523},
    {"port": 80, "count": 324}
  ]
}
```

---

### `GET /api/v1/alerts/{alert_id}`

Get a single alert by ID including full SHAP attribution.

**Response 200** — full AlertResponse schema.

**Response 404** if not found.

---

### `DELETE /api/v1/alerts/{alert_id}`

Delete an alert by ID. Returns 204 on success, 404 if not found.

---

### `POST /api/v1/infer`

On-demand flow inference. Accepts a named feature vector and returns the
predicted label, confidence, and (optionally) SHAP attribution.

> ⚠️ **Security**: Restrict this endpoint to internal/operator use only.
> SHAP values can guide adversarial evasion if exposed publicly.

**Request body**
```json
{
  "feature_vector": {
    "Destination_Port": 53.0,
    "Flow_Duration": 1000000.0,
    "Total_Fwd_Packets": 5000.0,
    "Flow_Bytes_s": 1000000.0,
    "H_src_ip": 0.2,
    "H_dst_ip": 0.0,
    "H_dst_port": 0.0,
    "H_proto": 0.0,
    "H_pkt_len": 0.5,
    "H_iat": 0.3,
    "H_tcp_flags": 0.0,
    "H_ttl": 0.2
  },
  "compute_shap": true
}
```

**Response 200**
```json
{
  "label": "DDoS-UDP",
  "confidence": 0.953,
  "is_ddos": true,
  "shap_top_features": [
    {"feature": "H_src_ip", "shap_value": 0.412, "abs_shap": 0.412},
    {"feature": "H_dst_port", "shap_value": 0.298, "abs_shap": 0.298},
    {"feature": "Flow_Bytes_s", "shap_value": 0.187, "abs_shap": 0.187}
  ],
  "inference_latency_ms": 2.341
}
```

**Response 503** if model not loaded.

---

### `GET /api/v1/model/info`

Returns metadata about the currently loaded model.

**Response 200**
```json
{
  "model_type": "RandomForestClassifier",
  "n_estimators": 200,
  "n_features": 88,
  "feature_names": ["Destination_Port", "Flow_Duration", ..., "H_ttl"],
  "classes": ["Benign", "DDoS-HTTP", "DDoS-ICMP", "DDoS-SlowLoris", "DDoS-TCP", "DDoS-UDP"],
  "training_accuracy": 0.9921,
  "training_macro_f1": 0.9914,
  "shap_enabled": true,
  "artifacts_dir": "model/artifacts"
}
```

---

### `GET /metrics`

Prometheus metrics endpoint (if `prometheus-fastapi-instrumentator` is installed).

---

## Error Responses

| Code | Meaning |
|------|---------|
| 422 | Validation error (check request schema) |
| 404 | Resource not found |
| 503 | Model not loaded (run training first) |
| 500 | Internal server error |

Error body:
```json
{
  "detail": "Human-readable error message"
}
```

---

## Rate Limits

No rate limiting is implemented by default. For production, add rate limiting
via nginx or a FastAPI middleware before deploying publicly.

---

## Webhooks / Streaming

Real-time alert streaming is not yet implemented. The dashboard polls the
`/api/v1/alerts/` endpoint at configurable intervals (default: 5s).

Future work: WebSocket endpoint at `/api/v1/alerts/stream` using Server-Sent Events.
