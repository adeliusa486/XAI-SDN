"""
baselines.py — Baseline Classifier Implementations for XAI-SDN.

Implements all baseline models from the paper:
  - Decision Tree
  - SVM (RBF kernel)
  - Naive Bayes
  - DNN (4×256, PyTorch)
  - LSTM (2×128, PyTorch)
  - XGBoost

Usage:
    python model/baselines.py --use-synthetic
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.train import load_synthetic_data

BASELINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "DecisionTree": {
        "model_class": "sklearn.tree.DecisionTreeClassifier",
        "params": {"max_depth": None, "class_weight": "balanced", "random_state": 42},
    },
    "SVM": {
        "model_class": "sklearn.svm.SVC",
        "params": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
            "random_state": 42,
        },
    },
    "NaiveBayes": {
        "model_class": "sklearn.naive_bayes.GaussianNB",
        "params": {},
    },
    "XGBoost": {
        "model_class": "xgboost.XGBClassifier",
        "params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "RandomForest": {
        "model_class": "sklearn.ensemble.RandomForestClassifier",
        "params": {
            "n_estimators": 200,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
}


def _import_class(dotted_path: str):
    """Dynamically import a class from a dotted path string."""
    parts = dotted_path.rsplit(".", 1)
    module = __import__(parts[0], fromlist=[parts[1]])
    return getattr(module, parts[1])


def train_and_evaluate_sklearn(
    model_name: str,
    config: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> Dict[str, Any]:
    """Train a sklearn-compatible baseline and return evaluation metrics."""
    logger.info(f"\n{'─'*50}")
    logger.info(f"Training baseline: {model_name}")

    try:
        cls = _import_class(config["model_class"])
        model = cls(**config["params"])
    except (ImportError, AttributeError) as e:
        logger.warning(f"  Could not import {model_name}: {e} — skipping.")
        return {"model": model_name, "error": str(e)}

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inf_time = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    latency_ms = (inf_time / len(X_test)) * 1000
    throughput = len(X_test) / inf_time

    logger.info(f"  Accuracy:   {acc:.4f}")
    logger.info(f"  Macro F1:   {macro_f1:.4f}")
    logger.info(f"  Latency:    {latency_ms:.3f} ms/flow")
    logger.info(f"  Throughput: {throughput:.0f} flows/s")
    logger.info(f"  Train time: {train_time:.1f}s")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")

    return {
        "model": model_name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "latency_ms": float(latency_ms),
        "throughput_flows_s": float(throughput),
        "train_time_s": float(train_time),
    }


# ─── PyTorch DNN Baseline ─────────────────────────────────────────────────────


def build_dnn(n_features: int, n_classes: int, hidden_size: int = 256, n_layers: int = 4):
    """Build a 4-layer DNN baseline (PyTorch)."""
    try:
        import torch
        import torch.nn as nn

        layers = []
        in_dim = n_features
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Dropout(0.3)])
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, n_classes))
        return nn.Sequential(*layers)
    except ImportError:
        return None


def build_lstm(n_features: int, n_classes: int, hidden_size: int = 128, n_layers: int = 2):
    """Build a 2-layer LSTM baseline (PyTorch)."""
    try:
        import torch
        import torch.nn as nn

        class LSTMClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    n_features, hidden_size, n_layers, batch_first=True, dropout=0.3
                )
                self.fc = nn.Linear(hidden_size, n_classes)

            def forward(self, x):
                # x shape: (batch, seq=1, features)
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        return LSTMClassifier()
    except ImportError:
        return None


def train_pytorch_model(model, X_train, y_train, X_test, y_test, model_name, epochs=20):
    """Train a PyTorch model and return metrics."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.warning(f"PyTorch not available — skipping {model_name}.")
        return {"model": model_name, "error": "PyTorch not available"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.LongTensor(y_train).to(device)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    logger.info(f"Training {model_name} for {epochs} epochs on {device}...")
    t0 = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            if "LSTM" in model_name:
                xb = xb.unsqueeze(1)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            logger.debug(f"  Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")
    train_time = time.perf_counter() - t0

    # Evaluate
    model.eval()
    X_te = torch.FloatTensor(X_test).to(device)
    if "LSTM" in model_name:
        X_te = X_te.unsqueeze(1)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(X_te)
    inf_time = time.perf_counter() - t0

    y_pred = logits.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    latency_ms = (inf_time / len(X_test)) * 1000

    logger.info(f"  {model_name} — Acc: {acc:.4f}, F1: {macro_f1:.4f}, Latency: {latency_ms:.3f}ms")
    return {
        "model": model_name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "latency_ms": float(latency_ms),
        "train_time_s": float(train_time),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--use-synthetic", is_flag=True, default=True, help="Use synthetic data.")
@click.option("--data-dir", default=None, help="Real data directory.")
@click.option("--skip-deep", is_flag=True, help="Skip DNN/LSTM baselines.")
@click.option("--output", default="model/artifacts/baseline_results.json", help="Output path.")
@click.option("--random-state", default=42, type=int, help="Global random seed for reproducibility.")
def run_baselines(
    use_synthetic: bool,
    data_dir: Optional[str],
    skip_deep: bool,
    output: str,
    random_state: int,
) -> None:
    """Run all baseline classifiers and compare with XAI-SDN."""
    from utils.seed_utils import set_global_seed
    set_global_seed(random_state)
    import json

    logger.info("=" * 60)
    logger.info("XAI-SDN Baseline Comparison")
    logger.info("=" * 60)

    if use_synthetic or data_dir is None:
        X, y_raw, le = load_synthetic_data(n_samples=8000, random_state=random_state)
    else:
        from model.train import load_real_data

        X, y_raw, le = load_real_data(data_dir)

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=0.30, stratify=y_raw, random_state=random_state
    )
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []

    # Sklearn baselines
    for name, cfg in BASELINE_CONFIGS.items():
        if "random_state" in cfg["params"]:
            cfg["params"]["random_state"] = random_state
        r = train_and_evaluate_sklearn(name, cfg, X_train_s, X_test_s, y_train, y_test, le)
        results.append(r)

    # PyTorch baselines
    if not skip_deep:
        n_f = X_train_s.shape[1]
        n_c = len(le.classes_)

        dnn = build_dnn(n_f, n_c)
        if dnn is not None:
            r = train_pytorch_model(dnn, X_train_s, y_train, X_test_s, y_test, "DNN-4x256")
            results.append(r)

        lstm = build_lstm(n_f, n_c)
        if lstm is not None:
            r = train_pytorch_model(lstm, X_train_s, y_train, X_test_s, y_test, "LSTM-2x128")
            results.append(r)

    # Print comparison table
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE COMPARISON SUMMARY")
    logger.info("=" * 70)
    header = (
        f"{'Model':<20} {'Accuracy':>10} {'Macro F1':>10} {'Latency ms':>12} {'Throughput':>12}"
    )
    logger.info(header)
    logger.info("-" * 70)
    for r in results:
        if "error" in r:
            logger.info(f"{r['model']:<20} ERROR: {r['error']}")
        else:
            logger.info(
                f"{r['model']:<20} "
                f"{r.get('accuracy', 0):>10.4f} "
                f"{r.get('macro_f1', 0):>10.4f} "
                f"{r.get('latency_ms', 0):>12.3f} "
                f"{r.get('throughput_flows_s', 0):>12.0f}"
            )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_baselines()
