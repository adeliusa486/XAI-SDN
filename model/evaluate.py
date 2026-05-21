"""
evaluate.py — Comprehensive Evaluation Script for XAI-SDN.

Computes:
  - Per-class Precision, Recall, F1
  - Macro accuracy and F1
  - False Positive Rate (binary: DDoS vs Benign)
  - AUC-ROC (one-vs-rest)
  - Confusion matrix
  - Per-flow detection latency and throughput
  - SHAP global feature importance

Usage:
    python model/evaluate.py --artifacts-dir model/artifacts --use-synthetic
    python model/evaluate.py --artifacts-dir model/artifacts --data-dir data/raw
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import click
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.train import load_synthetic_data


@click.command()
@click.option("--artifacts-dir", default="model/artifacts", help="Model artifacts directory.")
@click.option("--data-dir", default=None, help="Real data directory (optional).")
@click.option("--use-synthetic", is_flag=True, help="Use synthetic data.")
@click.option("--output-dir", default="model/artifacts", help="Where to write evaluation results.")
@click.option("--run-shap", is_flag=True, help="Compute global SHAP importance (slow).")
@click.option("--latency-runs", default=3, type=int, help="Latency measurement repetitions.")
def evaluate(
    artifacts_dir: str,
    data_dir: Optional[str],
    use_synthetic: bool,
    output_dir: str,
    run_shap: bool,
    latency_runs: int,
) -> None:
    """Evaluate the trained XAI-SDN model."""
    artifacts_path = Path(artifacts_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load artifacts ─────────────────────────────────────────────────────
    logger.info(f"Loading model artifacts from {artifacts_path}...")
    clf = joblib.load(artifacts_path / "rf_model.pkl")
    scaler = joblib.load(artifacts_path / "scaler.pkl")
    le = joblib.load(artifacts_path / "label_encoder.pkl")
    with open(artifacts_path / "feature_names.json") as f:
        feature_names = json.load(f)

    # ── Load data ──────────────────────────────────────────────────────────
    if use_synthetic or data_dir is None:
        test_X_path = artifacts_path / "X_test.npy"
        test_y_path = artifacts_path / "y_test.npy"

        if test_X_path.exists() and test_y_path.exists():
            logger.info(f"Loading saved test split from {artifacts_path}...")
            X_test = np.load(test_X_path)
            y_test = np.load(test_y_path)
            logger.info(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        else:
            logger.warning(
                "No saved test split found (X_test.npy / y_test.npy). "
                "Falling back to fresh synthetic data. "
                "For correct evaluation, run: python model/train.py --use-synthetic first."
            )
            X_all, y_raw, _ = load_synthetic_data(n_samples=5000, random_state=42)
            X_test = scaler.transform(X_all)
            y_test = le.transform(y_raw)
    else:
        from model.train import load_real_data
        from sklearn.model_selection import train_test_split

        X_all, y_all, _ = load_real_data(data_dir)
        _, X_test_raw, _, y_test = train_test_split(
            X_all, y_all, test_size=0.30, stratify=y_all, random_state=42
        )
        X_test = scaler.transform(X_test_raw)

    logger.info(f"Test set: {X_test.shape[0]} samples")

    # ── Classification metrics ─────────────────────────────────────────────
    logger.info("Computing classification metrics...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report_str = classification_report(y_test, y_pred, target_names=le.classes_)
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    logger.info(f"\n{report_str}")
    logger.info(f"Accuracy:    {acc:.4f}")
    logger.info(f"Macro F1:    {macro_f1:.4f}")

    # ── False Positive Rate ────────────────────────────────────────────────
    benign_class_id = list(le.classes_).index("Benign") if "Benign" in le.classes_ else 0
    y_binary_true = (y_test != benign_class_id).astype(int)
    y_binary_pred = (y_pred != benign_class_id).astype(int)
    cm = confusion_matrix(y_binary_true, y_binary_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        fpr = 0.0
    logger.info(f"FPR (binary): {fpr*100:.2f}%")

    # ── AUC-ROC ────────────────────────────────────────────────────────────
    try:
        if len(le.classes_) == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        logger.info(f"AUC (OvR):   {auc:.4f}")
    except Exception as e:
        logger.warning(f"AUC computation failed: {e}")
        auc = None

    # ── Latency benchmarks ─────────────────────────────────────────────────
    logger.info(f"Measuring inference latency ({latency_runs} runs)...")
    latencies = []
    for _ in range(latency_runs):
        t0 = time.perf_counter()
        clf.predict(X_test)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

    best_elapsed = min(latencies)
    latency_ms = (best_elapsed / len(X_test)) * 1000
    throughput = len(X_test) / best_elapsed

    logger.info(f"Mean per-flow latency: {latency_ms:.3f} ms")
    logger.info(f"Throughput:            {throughput:.0f} flows/s")

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm_full = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm_full,
        index=[f"True_{c}" for c in le.classes_],
        columns=[f"Pred_{c}" for c in le.classes_],
    )
    logger.info(f"\nConfusion Matrix:\n{cm_df}")

    # ── SHAP global importance ─────────────────────────────────────────────
    if run_shap:
        logger.info("Computing global SHAP importance (this may take several minutes)...")
        _compute_shap_global(clf, X_test, le, feature_names, output_path)

    # ── Save evaluation results ────────────────────────────────────────────
    results = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "fpr": float(fpr),
        "auc": float(auc) if auc is not None else None,
        "latency_ms": float(latency_ms),
        "throughput_flows_s": float(throughput),
        "n_test": int(len(y_test)),
        "per_class": {
            cls: {
                "precision": float(report_dict[cls]["precision"]),
                "recall": float(report_dict[cls]["recall"]),
                "f1": float(report_dict[cls]["f1-score"]),
                "support": int(report_dict[cls]["support"]),
            }
            for cls in le.classes_
            if cls in report_dict
        },
    }
    out_file = output_path / "evaluation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    cm_df.to_csv(output_path / "confusion_matrix.csv")

    logger.info(f"\nEvaluation results saved to {output_path}/")
    logger.info("Evaluation complete ✓")


def _compute_shap_global(clf, X_test, le, feature_names, output_path):
    """Compute and save global SHAP feature importance."""
    try:
        import shap

        # Subsample for speed
        max_samples = min(2000, len(X_test))
        X_sample = X_test[:max_samples]

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_sample)

        # Mean |SHAP| per feature across all DDoS classes (exclude Benign class 0)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            mean_abs_shap = np.mean(
                [np.abs(shap_values[i]).mean(axis=0) for i in range(1, len(shap_values))],
                axis=0,
            )
        else:
            shap_arr = np.array(shap_values)
            mean_abs_shap = (
                np.abs(shap_arr).mean(axis=0) if shap_arr.ndim == 2 else shap_arr.mean(axis=0)
            )

        importance_df = pd.DataFrame(
            {
                "feature": feature_names[: len(mean_abs_shap)],
                "mean_abs_shap": mean_abs_shap,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        importance_df.to_csv(output_path / "shap_global_importance.csv", index=False)
        logger.info(f"\nTop 10 SHAP features:\n{importance_df.head(10).to_string(index=False)}")

        # Save SHAP summary plot
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 8))
            top_n = 20
            top_df = importance_df.head(top_n).sort_values("mean_abs_shap")
            plt.barh(top_df["feature"], top_df["mean_abs_shap"], color="steelblue")
            plt.xlabel("Mean |SHAP value|")
            plt.title("Global SHAP Feature Importance (Top 20)")
            plt.tight_layout()
            plt.savefig(output_path / "shap_global_importance.png", dpi=150)
            plt.close()
            logger.info(f"SHAP importance plot saved.")
        except Exception as e:
            logger.warning(f"Could not save SHAP plot: {e}")

    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")


if __name__ == "__main__":
    evaluate()
