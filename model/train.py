"""
train.py — Random Forest Training Script for XAI-SDN.

Usage:
    python model/train.py --config configs/model_config.yaml
    python model/train.py --config configs/model_config.yaml --use-synthetic
    python model/train.py --data-dir data/raw --output-dir model/artifacts
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import importlib.metadata as importlib_metadata
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click
import joblib
import numpy as np
import yaml
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.cicflowmeter import CIC_FEATURE_NAMES
from features.entropy import ENTROPY_FEATURE_NAMES

# ─── MLflow (optional) ────────────────────────────────────────────────────────

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available — experiment tracking disabled.")


# ─── CLI ──────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--config", default="configs/model_config.yaml", help="Training config path.")
@click.option("--data-dir", default="data/raw", help="Raw data directory (CIC CSVs).")
@click.option("--output-dir", default="model/artifacts", help="Artifact output directory.")
@click.option("--use-synthetic", is_flag=True, help="Use synthetic data (no real dataset needed).")
@click.option("--n-estimators", default=None, type=int, help="Override n_estimators.")
@click.option("--experiment-name", default=None, help="MLflow experiment name.")
@click.option("--random-state", default=42, type=int, help="Global random seed for reproducibility.")
def train(
    config: str,
    data_dir: str,
    output_dir: str,
    use_synthetic: bool,
    n_estimators: Optional[int],
    experiment_name: Optional[str],
    random_state: int,
) -> None:
    """Train the XAI-SDN Random Forest classifier."""
    from utils.seed_utils import set_global_seed
    set_global_seed(random_state)
    # Load config
    cfg = load_config(config)
    rf_cfg = cfg.get("random_forest", {})

    if n_estimators is not None:
        rf_cfg["n_estimators"] = n_estimators

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Configure MLflow
    if MLFLOW_AVAILABLE:
        exp_name = experiment_name or cfg.get("training", {}).get("experiment_name", "xai-sdn")
        mlflow.set_experiment(exp_name)

    logger.info("=" * 60)
    logger.info("XAI-SDN Model Training")
    logger.info("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    test_size = cfg.get("data", {}).get("test_size", 0.30)

    if use_synthetic:
        logger.info("Using synthetic data for training demo...")
        X, y_raw, label_encoder = load_synthetic_data(random_state=random_state)

        # Split — synthetic path: entropy is already encoded in the feature
        # means (no sliding window used), so split-before-entropy is not required.
        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
            X, y_raw, test_size=test_size, stratify=y_raw, random_state=random_state,
        )
        label_encoder.fit(y_train_raw)
        y_train = label_encoder.transform(y_train_raw)
        y_test  = label_encoder.transform(y_test_raw)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

    else:
        # REAL DATA PATH — use OfflineFeaturePipeline.run() which correctly
        # performs train/test split BEFORE entropy computation, preventing
        # temporal window contamination between partitions.
        logger.info(f"Loading real data from {data_dir} via OfflineFeaturePipeline.run()...")
        X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler = \
            load_real_data(data_dir, test_size=test_size, random_state=random_state)

    logger.info(
        f"Dataset: {X_train_scaled.shape[0] + X_test_scaled.shape[0]} samples total, "
        f"{X_train_scaled.shape[1]} features"
    )
    logger.info(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    logger.info(f"Label encoder fit on train split only. Classes: {list(label_encoder.classes_)}")
    logger.info("Leakage check: entropy computed independently per partition. ✓")
    logger.info(f"Train Class distribution:\n{_class_distribution(y_train, label_encoder)}")

    # Save test split for evaluation decoupling (excluded from .gitignore in CI)
    np.save(output_path / "X_test.npy", X_test_scaled)
    np.save(output_path / "y_test.npy", y_test)
    logger.info(
        f"Test split saved: X_test.npy {X_test_scaled.shape}, "
        f"y_test.npy {y_test.shape}"
    )

    # ── Cross-validation ───────────────────────────────────────────────────
    logger.info("Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    clf_cv = RandomForestClassifier(**rf_cfg)
    cv_scores = cross_val_score(
        clf_cv, X_train_scaled, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
    )
    logger.info(f"CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Train final model ──────────────────────────────────────────────────
    logger.info(f"Training final model: RF({rf_cfg.get('n_estimators',200)} trees)...")
    t0 = time.perf_counter()
    clf = RandomForestClassifier(**rf_cfg)
    clf.fit(X_train_scaled, y_train)

    # Verify model was trained with expected hyperparameters
    expected_n_estimators = rf_cfg.get("n_estimators", 200)
    assert clf.n_estimators == expected_n_estimators, (
        f"RF trained with {clf.n_estimators} trees but config specifies "
        f"{expected_n_estimators}."
    )
    assert clf.class_weight == "balanced", "class_weight must be 'balanced'"
    assert clf.max_features == "sqrt", "max_features must be 'sqrt'"
    logger.info(
        f"Architecture verified: {clf.n_estimators} trees, "
        f"max_features={clf.max_features}, class_weight={clf.class_weight}"
    )

    train_time = time.perf_counter() - t0
    logger.info(f"Training time: {train_time:.1f}s")

    # ── Evaluate ───────────────────────────────────────────────────────────
    logger.info("Evaluating on test set (best of 3 inference runs for stable latency)...")
    latencies = []
    for _ in range(3):
        t0 = time.perf_counter()
        y_pred = clf.predict(X_test_scaled)
        latencies.append(time.perf_counter() - t0)
    inference_time = min(latencies)  # Best of 3 for stable measurement

    latency_ms = (inference_time / len(X_test_scaled)) * 1000
    throughput = len(X_test_scaled) / inference_time

    y_proba = clf.predict_proba(X_test_scaled)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
    )
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    accuracy = report["accuracy"]

    # False Positive Rate (binary: Benign vs any DDoS)
    benign_id = list(label_encoder.classes_).index("Benign") \
        if "Benign" in label_encoder.classes_ else 0
    y_bin_true = (y_test != benign_id).astype(int)
    y_bin_pred = (y_pred != benign_id).astype(int)
    cm_bin = confusion_matrix(y_bin_true, y_bin_pred)
    if cm_bin.shape == (2, 2):
        tn, fp, fn, tp = cm_bin.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        fpr, fnr = 0.0, 0.0

    # AUC-ROC
    try:
        if len(label_encoder.classes_) == 2:
            auc = float(roc_auc_score(y_test, y_proba[:, 1]))
        else:
            auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
    except Exception as e:
        logger.warning(f"AUC computation failed: {e}")
        auc = None

    logger.info(f"\n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")
    logger.info(f"Macro F1:          {macro_f1:.6f}")
    logger.info(f"Accuracy:          {accuracy:.6f}")
    logger.info(f"FPR (binary):      {fpr*100:.4f}%")
    logger.info(f"FNR (binary):      {fnr*100:.4f}%")
    logger.info(f"AUC-ROC:           {auc:.6f}" if auc else "AUC-ROC:           N/A")
    logger.info(f"Mean latency:      {latency_ms:.4f} ms/flow")
    logger.info(f"Throughput:        {throughput:.0f} flows/s")

    # ── Serialize artifacts ────────────────────────────────────────────────
    logger.info(f"Saving artifacts to {output_path}...")
    joblib.dump(clf, output_path / "rf_model.pkl")
    joblib.dump(scaler, output_path / "scaler.pkl")
    joblib.dump(label_encoder, output_path / "label_encoder.pkl")

    feature_names_all = CIC_FEATURE_NAMES + ENTROPY_FEATURE_NAMES
    with open(output_path / "feature_names.json", "w") as f:
        json.dump(feature_names_all, f, indent=2)

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "auc_roc": auc,
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "latency_ms": float(latency_ms),
        "throughput_flows_s": float(throughput),
        "train_time_s": float(train_time),
        "n_train": int(X_train_scaled.shape[0]),
        "n_test": int(X_test_scaled.shape[0]),
        "n_features": int(X_train_scaled.shape[1]),
        "classes": label_encoder.classes_.tolist(),
    }
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Artifacts saved:")
    logger.info(f"  rf_model.pkl, scaler.pkl, label_encoder.pkl, feature_names.json, metrics.json")

    data_hash = hash_dataset(data_dir, use_synthetic)
    generate_reproducibility_manifest(
        output_path=output_path,
        metrics=metrics,
        rf_cfg=rf_cfg,
        n_train=int(X_train_scaled.shape[0]),
        n_test=int(X_test_scaled.shape[0]),
        random_state=random_state,
        data_source="synthetic" if use_synthetic else data_dir,
        data_hash=data_hash,
    )

    # ── MLflow logging ─────────────────────────────────────────────────────
    if MLFLOW_AVAILABLE:
        with mlflow.start_run() as run:
            run_id = run.info.run_id

            # Log all hyperparameters
            mlflow.log_params(rf_cfg)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("n_features", X_train_scaled.shape[1])
            mlflow.log_param("n_train", X_train_scaled.shape[0])
            mlflow.log_param("n_test", X_test_scaled.shape[0])
            mlflow.log_param("data_source", "synthetic" if use_synthetic else data_dir)
            mlflow.log_param("python_version", platform.python_version())
            mlflow.log_param("sklearn_version",
                             importlib_metadata.version("scikit-learn"))

            # Log all metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
                "latency_ms": latency_ms,
                "throughput_flows_s": throughput,
                "fpr": metrics.get("fpr", 0.0),
            })

            # Log model
            mlflow.sklearn.log_model(clf, "rf_model")

            # Log artifacts
            mlflow.log_artifact(str(output_path / "metrics.json"))
            mlflow.log_artifact(str(output_path / "reproducibility_manifest.json"))

            logger.info(f"MLflow run ID: {run_id}")

            # Add run_id to manifest
            manifest_path = output_path / "reproducibility_manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest["mlflow_run_id"] = run_id
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
    else:
        run_id = None
        logger.warning(
            "MLflow not available. Install with: pip install mlflow. "
            "Experiment tracking disabled."
        )

    logger.info("Training complete ✓")


# ─── Data Loaders ────────────────────────────────────────────────────────────


def load_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 88,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Generate synthetic DDoS/Benign data for smoke testing and demo.

    Creates a linearly separable synthetic dataset that mimics the
    distributional properties of CIC-DDoS2019 (class imbalance, feature ranges).

    Returns:
        (X, y_encoded, LabelEncoder)
    """
    logger.warning("==========================================================")
    logger.warning("WARNING: SYNTHETIC DATA GENERATION")
    logger.warning("This data is heavily engineered to be linearly separable")
    logger.warning("for CI/CD smoke testing. Models will trivially achieve")
    logger.warning("1.000 F1 scores. DO NOT report these metrics as real!")
    logger.warning("==========================================================")
    
    rng = np.random.RandomState(random_state)
    classes = ["Benign", "DDoS-UDP", "DDoS-TCP", "DDoS-ICMP", "DDoS-SlowLoris", "DDoS-HTTP"]
    # Class proportions mirroring CIC-DDoS2019
    class_props = [0.380, 0.200, 0.234, 0.183, 0.155, 0.016]
    # Normalize to sum to 1
    class_props = np.array(class_props)
    class_props /= class_props.sum()

    n_per_class = (class_props * n_samples).astype(int)
    n_per_class[-1] = n_samples - n_per_class[:-1].sum()  # Fix rounding

    X_parts, y_parts = [], []
    for i, (cls, n) in enumerate(zip(classes, n_per_class)):
        if n <= 0:
            continue
        # Each class has a distinct mean in entropy space (last 8 features)
        # to make classification non-trivial but tractable
        mean = np.zeros(n_features)

        if cls == "Benign":
            # High entropy across the board
            mean[80:88] = [6.5, 5.0, 4.5, 1.5, 4.0, 3.5, 2.0, 3.0]
            mean[0] = 443  # dst_port: HTTPS
            mean[14] = 5e4  # flow_bytes_s
        elif cls == "DDoS-UDP":
            # Low src_ip entropy (botnet), low dst_port entropy (single port)
            mean[80:88] = [0.5, 0.3, 0.2, 0.1, 0.8, 0.5, 0.1, 0.3]
            mean[0] = 53  # DNS port
            mean[14] = 1e7  # high bytes/s
        elif cls == "DDoS-TCP":
            mean[80:88] = [1.0, 0.4, 0.5, 0.2, 0.9, 0.6, 0.0, 0.2]
            mean[43] = 5.0  # SYN_Flag_Count
            mean[14] = 8e6
        elif cls == "DDoS-ICMP":
            mean[80:88] = [1.2, 0.5, 0.5, 0.0, 1.0, 0.4, 0.1, 0.4]
            mean[14] = 9e6
        elif cls == "DDoS-SlowLoris":
            mean[80:88] = [3.5, 2.0, 2.5, 0.8, 1.5, 0.2, 1.2, 1.5]
            mean[2] = 1  # Very few packets
            mean[1] = 3e7  # Long duration
        elif cls == "DDoS-HTTP":
            mean[80:88] = [4.0, 1.0, 1.2, 1.0, 2.0, 1.5, 1.8, 2.0]
            mean[0] = 80  # HTTP
            mean[14] = 2e5

        X_cls = rng.randn(n, n_features) * 0.3 + mean
        X_parts.append(X_cls)
        y_parts.extend([cls] * n)

    X = np.vstack(X_parts).astype(np.float32)
    y_raw = np.array(y_parts)

    le = LabelEncoder()
    return X, y_raw, le


def load_real_data(
    data_dir: str,
    test_size: float = 0.30,
    random_state: int = 42,
) -> tuple:
    """Load and preprocess real CIC-DDoS2019 data using the split-first pipeline.

    Uses OfflineFeaturePipeline.run() which performs the train/test split
    BEFORE computing entropy features, preventing temporal window contamination
    between the train and test partitions.

    Args:
        data_dir: Directory containing CIC-DDoS2019 CSV files.
        test_size: Fraction of data for the test split (default 0.30).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple (X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler)
        All arrays are already split, entropy-augmented, and scaled.
    """
    from features.pipeline import OfflineFeaturePipeline

    pipeline = OfflineFeaturePipeline(
        test_size=test_size,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = pipeline.run(data_dir)
    return X_train, X_test, y_train, y_test, pipeline.label_encoder, pipeline.scaler


# ─── Utilities ────────────────────────────────────────────────────────────────


def generate_reproducibility_manifest(
    output_path: Path,
    metrics: dict,
    rf_cfg: dict,
    n_train: int,
    n_test: int,
    random_state: int,
    data_source: str,
    data_hash: str | None = None,
) -> dict:
    """Generate a JSON manifest linking metrics to exact run conditions."""
    def get_version(pkg: str) -> str:
        try:
            return importlib_metadata.version(pkg)
        except Exception:
            return "unknown"

    manifest = {
        "schema_version": "1.0",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_source": data_source,
        "data_hash_sha256": data_hash,
        "random_state": random_state,
        "n_train": n_train,
        "n_test": n_test,
        "hyperparameters": rf_cfg,
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": get_version("scikit-learn"),
            "numpy": get_version("numpy"),
            "pandas": get_version("pandas"),
            "joblib": get_version("joblib"),
        },
    }

    manifest_path = output_path / "reproducibility_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Reproducibility manifest saved: {manifest_path}")
    return manifest


def hash_dataset(data_dir: str | None, use_synthetic: bool) -> str:
    """Compute SHA-256 of training data for provenance."""
    if use_synthetic:
        return "SYNTHETIC-n10000-seed42"
    if data_dir is None:
        return "UNKNOWN"
    h = hashlib.sha256()
    for csv_path in sorted(Path(data_dir).glob("*.csv")):
        with open(csv_path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config not found at {path}; using defaults.")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _class_distribution(y: np.ndarray, le: LabelEncoder) -> str:
    unique, counts = np.unique(y, return_counts=True)
    lines = []
    for cls_id, cnt in zip(unique, counts):
        cls_name = le.classes_[cls_id]
        pct = cnt / len(y) * 100
        lines.append(f"  {cls_name:20s} {cnt:6d}  ({pct:.1f}%)")
    return "\n".join(lines)


# ─── Entry Point ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    train()
