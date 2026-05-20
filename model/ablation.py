"""
ablation.py — Ablation Study Runner for XAI-SDN.

Reproduces Table 4 from the paper:
  - RF + CIC features only (80-dim)     → baseline
  - SVM + Full features (88-dim)         → model swap
  - RF + Entropy only (8-dim)            → feature ablation
  - XAI-SDN (RF + 88-dim + SHAP)        → proposed system

Usage:
    python model/ablation.py --use-synthetic
    python model/ablation.py --data-dir data/raw
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import click
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.cicflowmeter import CIC_FEATURE_NAMES
from features.entropy import ENTROPY_FEATURE_NAMES
from model.train import load_synthetic_data

N_CIC = len(CIC_FEATURE_NAMES)   # 80
N_ENT = len(ENTROPY_FEATURE_NAMES)  # 8


@click.command()
@click.option("--use-synthetic", is_flag=True, default=True)
@click.option("--data-dir", default=None)
@click.option("--output", default="model/artifacts/ablation_results.json")
def run_ablation(use_synthetic: bool, data_dir, output: str) -> None:
    """Run the ablation study comparing feature subsets and model variants."""
    logger.info("=" * 60)
    logger.info("XAI-SDN Ablation Study")
    logger.info("=" * 60)

    if use_synthetic or data_dir is None:
        X_full, y, le = load_synthetic_data(n_samples=8000)
    else:
        from model.train import load_real_data
        X_full, y, le = load_real_data(data_dir)

    X_train_f, X_test_f, y_train, y_test = train_test_split(
        X_full, y, test_size=0.30, stratify=y, random_state=42
    )

    # Feature slicing
    def slice_features(X: np.ndarray, mode: str) -> np.ndarray:
        if mode == "cic_only":
            return X[:, :N_CIC]
        elif mode == "entropy_only":
            return X[:, N_CIC:]
        elif mode == "full":
            return X
        raise ValueError(f"Unknown feature mode: {mode}")

    RF_PARAMS = dict(
        n_estimators=200, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    SVM_PARAMS = dict(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", random_state=42, probability=True
    )

    configs = [
        {"name": "RF + CIC-only (80-dim)",       "features": "cic_only",    "model": "RF"},
        {"name": "RF + Entropy-only (8-dim)",     "features": "entropy_only","model": "RF"},
        {"name": "SVM + Full (88-dim)",           "features": "full",        "model": "SVM"},
        {"name": "XAI-SDN: RF + Full (88-dim)",  "features": "full",        "model": "RF"},
    ]

    results = []
    for cfg in configs:
        logger.info(f"\nRunning: {cfg['name']}")

        X_tr = slice_features(X_train_f, cfg["features"])
        X_te = slice_features(X_test_f, cfg["features"])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        if cfg["model"] == "RF":
            model = RandomForestClassifier(**RF_PARAMS)
        elif cfg["model"] == "SVM":
            model = SVC(**SVM_PARAMS)
        else:
            raise ValueError(f"Unknown model: {cfg['model']}")

        t0 = time.perf_counter()
        model.fit(X_tr_s, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(X_te_s)
        inf_time = time.perf_counter() - t0

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        latency_ms = (inf_time / len(X_te)) * 1000

        # Binary FPR
        benign_id = list(le.classes_).index("Benign") if "Benign" in le.classes_ else 0
        y_bin_true = (y_test != benign_id).astype(int)
        y_bin_pred = (y_pred != benign_id).astype(int)
        tn = int(((y_bin_true == 0) & (y_bin_pred == 0)).sum())
        fp = int(((y_bin_true == 0) & (y_bin_pred == 1)).sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        result = {
            "name": cfg["name"],
            "features": cfg["features"],
            "model": cfg["model"],
            "n_features": X_tr.shape[1],
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "fpr": float(fpr),
            "latency_ms": float(latency_ms),
            "train_time_s": float(train_time),
        }
        results.append(result)

        logger.info(
            f"  Acc={acc:.4f}  F1={f1:.4f}  FPR={fpr*100:.2f}%  "
            f"Lat={latency_ms:.2f}ms  Features={X_tr.shape[1]}"
        )

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("=" * 80)
    header = f"{'Configuration':<35} {'Acc%':>7} {'F1%':>7} {'FPR%':>7} {'ΔAcc':>7}"
    logger.info(header)
    logger.info("-" * 80)
    baseline_acc = results[0]["accuracy"]
    for r in results:
        delta = r["accuracy"] - baseline_acc
        delta_str = f"+{delta*100:.1f}%" if delta >= 0 else f"{delta*100:.1f}%"
        logger.info(
            f"{r['name']:<35} "
            f"{r['accuracy']*100:>7.1f} "
            f"{r['macro_f1']*100:>7.1f} "
            f"{r['fpr']*100:>7.2f} "
            f"{delta_str:>7}"
        )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nAblation results saved to {out_path}")


if __name__ == "__main__":
    run_ablation()
