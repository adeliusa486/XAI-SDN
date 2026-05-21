"""Unit and integration tests for model/train.py and model/evaluate.py"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


class TestSyntheticDataLoader:
    def test_load_synthetic_returns_correct_shapes(self):
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from model.train import load_synthetic_data

        X, y_raw, le = load_synthetic_data(n_samples=500, random_state=0)
        assert X.shape[0] == 500
        assert X.shape[1] == 88
        assert len(y_raw) == 500
        le.fit(y_raw)
        assert len(le.classes_) == 6

    def test_load_synthetic_has_all_classes(self):
        from model.train import load_synthetic_data

        X, y_raw, le = load_synthetic_data(n_samples=2000, random_state=42)
        unique_classes = np.unique(y_raw)
        assert len(unique_classes) == 6
        le.fit(y_raw)
        expected = {"Benign", "DDoS-UDP", "DDoS-TCP", "DDoS-ICMP", "DDoS-SlowLoris", "DDoS-HTTP"}
        assert set(le.classes_) == expected

    def test_load_synthetic_reproducible(self):
        from model.train import load_synthetic_data

        X1, y_raw1, _ = load_synthetic_data(n_samples=100, random_state=7)
        X2, y_raw2, _ = load_synthetic_data(n_samples=100, random_state=7)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y_raw1, y_raw2)

    def test_load_synthetic_different_seeds_differ(self):
        from model.train import load_synthetic_data

        X1, _, _ = load_synthetic_data(n_samples=200, random_state=1)
        X2, _, _ = load_synthetic_data(n_samples=200, random_state=2)
        assert not np.allclose(X1, X2)


class TestRandomForestTraining:
    @pytest.fixture(scope="class")
    def trained_model(self):
        from model.train import load_synthetic_data
        from sklearn.model_selection import train_test_split

        X, y_raw, le = load_synthetic_data(n_samples=2000, random_state=42)
        X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y_raw, test_size=0.3, stratify=y_raw)
        le.fit(y_train_raw)
        y_train = le.transform(y_train_raw)
        y_test = le.transform(y_test_raw)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
        clf.fit(X_train_s, y_train)

        return clf, scaler, le, X_test_s, y_test

    def test_accuracy_above_threshold(self, trained_model):
        clf, _, _, X_test, y_test = trained_model
        acc = clf.score(X_test, y_test)
        assert acc > 0.70, f"Expected acc > 0.70, got {acc:.4f}"

    def test_predict_proba_sums_to_one(self, trained_model):
        clf, _, _, X_test, _ = trained_model
        proba = clf.predict_proba(X_test[:10])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predictions_in_valid_range(self, trained_model):
        clf, _, le, X_test, _ = trained_model
        preds = clf.predict(X_test)
        assert all(0 <= p < len(le.classes_) for p in preds)

    def test_feature_importances_sum_to_one(self, trained_model):
        clf, _, _, _, _ = trained_model
        importances = clf.feature_importances_
        assert len(importances) == 88
        assert abs(importances.sum() - 1.0) < 1e-6

    def test_n_estimators_correct(self, trained_model):
        clf, _, _, _, _ = trained_model
        assert len(clf.estimators_) == 20


class TestModelSerialization:
    def test_serialize_and_reload(self):
        import joblib
        from model.train import load_synthetic_data
        from sklearn.model_selection import train_test_split

        X, y_raw, le = load_synthetic_data(n_samples=500, random_state=0)
        X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y_raw, test_size=0.3, stratify=y_raw)
        le.fit(y_train_raw)
        y_train = le.transform(y_train_raw)
        y_test = le.transform(y_test_raw)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = RandomForestClassifier(n_estimators=5, random_state=0)
        clf.fit(X_train_s, y_train)
        original_preds = clf.predict(X_test_s)

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            joblib.dump(clf, p / "model.pkl")
            joblib.dump(scaler, p / "scaler.pkl")
            joblib.dump(le, p / "le.pkl")
            with open(p / "features.json", "w") as f:
                json.dump([f"f_{i}" for i in range(88)], f)

            # Reload
            clf2 = joblib.load(p / "model.pkl")
            reloaded_preds = clf2.predict(X_test_s)

        np.testing.assert_array_equal(original_preds, reloaded_preds)


class TestConfigLoading:
    def test_load_model_config(self):
        import yaml

        cfg_path = Path("configs/model_config.yaml")
        assert cfg_path.exists()
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "random_forest" in cfg
        assert cfg["random_forest"]["n_estimators"] == 200
        assert cfg["random_forest"]["max_features"] == "sqrt"

    def test_load_master_config(self):
        import yaml

        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["features"]["n_total_features"] == 88
        assert cfg["features"]["entropy"]["window_size"] == 1000
        assert cfg["model"]["detection_threshold"] == 0.70
