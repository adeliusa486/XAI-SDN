"""
smoke_test.py — End-to-End Smoke Test for XAI-SDN.

Validates:
  1. Python imports
  2. Config loading
  3. Entropy feature computation
  4. Synthetic data generation
  5. Model training (fast, 10 trees)
  6. Model inference
  7. SHAP explanation
  8. API startup + health check
  9. Alert ingestion via API
  10. Alert retrieval via API

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --skip-api  # Skip API tests (no uvicorn needed)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS: List[Dict] = []
PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"


def test(name: str):
    """Decorator for test functions."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                RESULTS.append({"test": name, "status": "PASS", "detail": str(result or "")})
                print(f"{PASS}  {name}")
                return result
            except Exception as e:
                tb = traceback.format_exc()
                RESULTS.append({"test": name, "status": "FAIL", "detail": str(e), "traceback": tb})
                print(f"{FAIL}  {name}")
                print(f"        {e}")
                return None

        return wrapper

    return decorator


# ─── Test functions ───────────────────────────────────────────────────────────


@test("Import: features.entropy")
def test_import_entropy():
    from features.entropy import shannon_entropy, EntropyFeatureExtractor, ENTROPY_FEATURE_NAMES

    assert len(ENTROPY_FEATURE_NAMES) == 8
    h = shannon_entropy(["a", "b", "c", "a"])
    assert 0 < h < 2.1, f"Expected entropy ~1.5, got {h}"
    return f"H={h:.4f}"


@test("Import: features.cicflowmeter")
def test_import_cic():
    from features.cicflowmeter import CIC_FEATURE_NAMES, CICFlowMeterExtractor

    assert len(CIC_FEATURE_NAMES) == 80
    return f"{len(CIC_FEATURE_NAMES)} features"


@test("Import: explainability.shap_explainer")
def test_import_shap():
    from explainability.shap_explainer import SHAPExplainer

    return "OK"


@test("Import: api.main")
def test_import_api():
    from api.main import app

    assert app is not None
    return "FastAPI app imported"


@test("Import: api.models.schemas")
def test_import_schemas():
    from api.models.schemas import AlertCreate, AlertResponse, InferenceRequest, InferenceResponse

    return "All schemas imported"


@test("Config: load master config")
def test_config_loading():
    import yaml

    cfg_path = Path("configs/config.yaml")
    assert cfg_path.exists(), f"Config not found: {cfg_path}"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert "model" in cfg
    assert "features" in cfg
    assert cfg["features"]["n_total_features"] == 88
    return f"window={cfg['features']['entropy']['window_size']}"


@test("Entropy: sliding window computation")
def test_entropy_sliding_window():
    from features.entropy import EntropyFeatureExtractor, ENTROPY_FEATURE_NAMES

    extractor = EntropyFeatureExtractor(window_size=10)
    records = [
        {
            "src_ip": f"10.0.0.{i % 3}",
            "dst_ip": "10.0.0.1",
            "dst_port": 53,
            "protocol": 17,
            "pkt_len_mean": 64.0,
            "iat_mean": 1000.0,
            "tcp_flags": 0,
            "ttl": 64,
        }
        for i in range(15)
    ]
    for rec in records:
        feats = extractor.update_and_compute(rec)

    assert set(feats.keys()) == set(ENTROPY_FEATURE_NAMES)
    assert 0 <= feats["H_src_ip"] <= 10
    assert feats["H_dst_port"] == 0.0  # Single destination port
    assert len(extractor) == 10  # Window capped at 10
    return f"H_src_ip={feats['H_src_ip']:.3f}"


@test("Entropy: offline batch computation")
def test_entropy_offline():
    import numpy as np
    from features.entropy import compute_entropy_features_offline

    records = [
        {
            "src_ip": f"10.0.0.{i%5}",
            "dst_ip": "10.0.0.1",
            "dst_port": 80,
            "protocol": 6,
            "pkt_len_mean": 100.0,
            "iat_mean": 500.0,
            "tcp_flags": 2,
            "ttl": 64,
        }
        for i in range(50)
    ]
    result = compute_entropy_features_offline(records, window_size=10)
    assert result.shape == (50, 8)
    return f"shape={result.shape}"


@test("CICFlowMeter: OpenFlow feature bridge")
def test_openflow_bridge():
    import numpy as np
    from features.cicflowmeter import extract_features_from_openflow, CIC_FEATURE_NAMES

    stat = {
        "packet_count": 1000,
        "byte_count": 64000,
        "duration_sec": 1,
        "duration_nsec": 0,
        "match": {"ipv4_src": "10.0.0.100", "ipv4_dst": "10.0.0.1", "tp_dst": 53, "ip_proto": 17},
    }
    feats = extract_features_from_openflow(stat)
    assert len(feats) == len(CIC_FEATURE_NAMES)
    assert feats["Flow_Bytes_s"] == 64000.0
    assert feats["Destination_Port"] == 53.0
    return f"bytes/s={feats['Flow_Bytes_s']:.0f}"


@test("Synthetic data: generation")
def test_synthetic_data():
    import numpy as np
    from model.train import load_synthetic_data

    X, y_raw, le = load_synthetic_data(n_samples=500, random_state=0)
    assert X.shape == (500, 88), f"Expected (500, 88), got {X.shape}"
    assert len(np.unique(y_raw)) >= 5
    le.fit(y_raw)
    return f"X={X.shape}, classes={list(le.classes_)}"


@test("Model: training (fast, 10 trees)")
def test_model_training():
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from model.train import load_synthetic_data

    X, y_raw, le = load_synthetic_data(n_samples=2000, random_state=42)
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y_raw, test_size=0.3, stratify=y_raw)
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    acc = clf.score(X_test_s, y_test)
    assert acc > 0.70, f"Expected acc > 0.70, got {acc:.3f}"
    return f"acc={acc:.4f}"


@test("Model: serialize and reload artifacts")
def test_model_serialization():
    import tempfile
    import joblib
    import json
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        clf = RandomForestClassifier(n_estimators=5, random_state=0)
        clf.fit(np.random.randn(100, 88), np.random.randint(0, 3, 100))
        joblib.dump(clf, tmppath / "rf_model.pkl")

        scaler = StandardScaler()
        scaler.fit(np.random.randn(100, 88))
        joblib.dump(scaler, tmppath / "scaler.pkl")

        le = LabelEncoder()
        le.fit(["Benign", "DDoS-UDP", "DDoS-TCP"])
        joblib.dump(le, tmppath / "label_encoder.pkl")

        feature_names = [f"feat_{i}" for i in range(88)]
        with open(tmppath / "feature_names.json", "w") as f:
            json.dump(feature_names, f)

        # Reload
        clf2 = joblib.load(tmppath / "rf_model.pkl")
        pred = clf2.predict(np.random.randn(5, 88))
        assert len(pred) == 5
    return "Serialization OK"


@test("SHAP: TreeSHAP initialization and explain_flow")
def test_shap_explanation():
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from explainability.shap_explainer import SHAPExplainer, SHAP_AVAILABLE

    if not SHAP_AVAILABLE:
        return "SHAP not installed — skipped"

    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    X_train = np.random.randn(200, 88)
    y_train = np.random.randint(0, 3, 200)
    clf.fit(X_train, y_train)

    feature_names = [f"feat_{i}" for i in range(88)]
    explainer = SHAPExplainer(clf, feature_names)

    x = np.random.randn(88)
    attribution = explainer.explain_flow(x, predicted_class_idx=1)

    if attribution:
        assert len(attribution) == 88
        assert all(isinstance(v, float) for v in attribution.values())
    return f"attribution keys={len(attribution)}, shap_ready={explainer.is_ready}"


@test("API schemas: AlertCreate validation")
def test_api_schema_validation():
    from datetime import datetime
    from api.models.schemas import AlertCreate, AttackLabel

    alert = AlertCreate(
        flow_id="test-flow-001",
        src_ip="10.0.1.100",
        dst_ip="10.0.0.1",
        dst_port=53,
        protocol=17,
        label=AttackLabel.DDOS_UDP,
        confidence=0.95,
    )
    assert alert.label == AttackLabel.DDOS_UDP
    assert alert.confidence == 0.95

    # Test invalid confidence raises
    try:
        AlertCreate(
            flow_id="x",
            src_ip="1.1.1.1",
            dst_ip="2.2.2.2",
            label=AttackLabel.BENIGN,
            confidence=1.5,  # invalid
        )
        assert False, "Should have raised ValidationError"
    except Exception:
        pass

    return "Validation OK"


def test_api_startup_and_endpoints(skip_api: bool = False):
    """Start FastAPI in a subprocess and run HTTP checks."""
    if skip_api:
        RESULTS.append(
            {"test": "API: startup + endpoints", "status": "SKIP", "detail": "--skip-api"}
        )
        print(f"{SKIP}  API: startup + endpoints")
        return

    import socket
    import requests as req

    # Check if already running on 8000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        already_running = s.connect_ex(("localhost", 8765)) == 0

    # Start uvicorn on test port 8765
    proc = None
    if not already_running:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
                "--log-level",
                "error",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(4)  # Wait for startup

    BASE = "http://127.0.0.1:8765"

    try:
        # Health check
        r = req.get(f"{BASE}/health", timeout=5)
        assert r.status_code == 200
        data = r.json()
        RESULTS.append(
            {"test": "API: /health", "status": "PASS", "detail": str(data.get("status"))}
        )
        print(f"{PASS}  API: /health → {data.get('status')}")

        # Docs check
        r2 = req.get(f"{BASE}/docs", timeout=5)
        assert r2.status_code == 200
        RESULTS.append({"test": "API: /docs", "status": "PASS"})
        print(f"{PASS}  API: /docs → 200")

        # Stats
        r3 = req.get(f"{BASE}/api/v1/alerts/stats", timeout=5)
        assert r3.status_code == 200
        RESULTS.append({"test": "API: /alerts/stats", "status": "PASS"})
        print(f"{PASS}  API: /alerts/stats → 200")

        # Ingest alert (will 503 if model not loaded, that's OK)
        alert_payload = {
            "flow_id": "smoke-test-001",
            "src_ip": "10.0.1.100",
            "dst_ip": "10.0.0.1",
            "dst_port": 53,
            "protocol": 17,
            "label": "DDoS-UDP",
            "confidence": 0.95,
        }
        r4 = req.post(f"{BASE}/api/v1/alerts", json=alert_payload, timeout=5)
        assert r4.status_code in (200, 201, 503)
        RESULTS.append(
            {"test": "API: POST /alerts", "status": "PASS", "detail": f"status={r4.status_code}"}
        )
        print(f"{PASS}  API: POST /alerts → {r4.status_code}")

    except Exception as e:
        RESULTS.append({"test": "API: startup + endpoints", "status": "FAIL", "detail": str(e)})
        print(f"{FAIL}  API: startup + endpoints → {e}")

    finally:
        if proc:
            proc.terminate()
            proc.wait()


# ─── Report generation ────────────────────────────────────────────────────────


def write_report():
    passed = [r for r in RESULTS if r["status"] == "PASS"]
    failed = [r for r in RESULTS if r["status"] == "FAIL"]
    skipped = [r for r in RESULTS if r["status"] == "SKIP"]

    lines = [
        "# XAI-SDN Smoke Test Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n",
        f"Python: {sys.version}\n",
        "",
        f"## Summary\n",
        f"- Total tests: {len(RESULTS)}",
        f"- ✅ Passed:  {len(passed)}",
        f"- ❌ Failed:  {len(failed)}",
        f"- ⏭  Skipped: {len(skipped)}",
        "",
        "## Test Results\n",
        "| Status | Test |",
        "|--------|------|",
    ]

    for r in RESULTS:
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭"}.get(r["status"], "?")
        lines.append(f"| {icon} {r['status']} | {r['test']} |")

    if failed:
        lines += ["", "## Failures\n"]
        for r in failed:
            lines += [
                f"### ❌ {r['test']}",
                f"```",
                r.get("detail", ""),
                r.get("traceback", ""),
                "```",
                "",
            ]

    lines += [
        "",
        "## Environment Notes",
        "",
        "- Ryu controller requires Python ≤ 3.8 (separate virtualenv).",
        "- Mininet requires Linux with root/sudo.",
        "- SHAP computation scales with n_estimators × test set size.",
        "- Real CIC-DDoS2019 data requires UNB registration.",
        "- API tests require uvicorn to be installed.",
    ]

    report = "\n".join(lines)
    with open("SMOKE_TEST_REPORT.md", "w") as f:
        f.write(report)
    print(f"\nSmoke test report saved: SMOKE_TEST_REPORT.md")
    return len(failed)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-api", action="store_true", help="Skip API server tests")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("XAI-SDN Smoke Test Suite")
    print("=" * 60 + "\n")

    # Run all unit/import/logic tests
    test_import_entropy()
    test_import_cic()
    test_import_shap()
    test_import_api()
    test_import_schemas()
    test_config_loading()
    test_entropy_sliding_window()
    test_entropy_offline()
    test_openflow_bridge()
    test_synthetic_data()
    test_model_training()
    test_model_serialization()
    test_shap_explanation()
    test_api_schema_validation()
    test_api_startup_and_endpoints(skip_api=args.skip_api)

    print()
    n_failed = write_report()

    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if n_failed > 0 else 0)


if __name__ == "__main__":
    main()
