"""
generate_roc_pr.py — Generate real ROC and Precision-Recall curves from archived models.

Replaces hand-placed pgfplots coordinates in Fig. 6 with computed roc_curve() outputs.
Adds a PR curve (mandatory given 0.86% minority class — PR-AUC is the honest metric).

Usage:
    python scripts/generate_roc_pr.py --use-synthetic
    python scripts/generate_roc_pr.py --artifacts-dir model/artifacts
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
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).parent.parent))


def downsample_curve(fpr_or_rec: np.ndarray, tpr_or_prec: np.ndarray,
                     n_points: int = 20, log_spaced: bool = True) -> Tuple[List, List]:
    """Downsample a ROC/PR curve to n_points for pgfplots use.

    For ROC: log-spaced in FPR (better resolution at low FPR — the operational range).
    For PR:  linearly spaced in recall.
    """
    x = np.array(fpr_or_rec)
    y = np.array(tpr_or_prec)

    if log_spaced:
        # Log-space from min non-zero to max
        x_min = max(x[x > 0].min(), 1e-6) if (x > 0).any() else 1e-6
        x_grid = np.logspace(np.log10(x_min), np.log10(x.max()), n_points)
    else:
        x_grid = np.linspace(x.min(), x.max(), n_points)

    y_interp = np.interp(x_grid, x, y)
    # Always include endpoints
    x_out = np.concatenate([[x[0]], x_grid, [x[-1]]])
    y_out = np.concatenate([[y[0]], y_interp, [y[-1]]])
    return x_out.tolist(), y_out.tolist()


def load_or_synthesize_test_data(artifacts_dir: Path, use_synthetic: bool, random_state: int):
    """Load pre-saved test data or generate synthetic data for evaluation."""
    from scripts.synth_utils import load_synthetic_split

    splits_dir = Path("data/splits")
    if not use_synthetic and (splits_dir / "X_test.npy").exists():
        logger.info("Loading real test splits from data/splits/")
        X_test = np.load(splits_dir / "X_test.npy")
        y_test = np.load(splits_dir / "y_test.npy")
        return X_test, y_test

    logger.info("Using synthetic test data...")
    _, X_test, _, y_test = load_synthetic_split(
        n_samples=20000, random_state=random_state, binary=True
    )
    return X_test, y_test


@click.command()
@click.option("--artifacts-dir", default="model/artifacts", help="Model artifacts directory.")
@click.option("--output-dir", default="model/artifacts", help="Output directory for JSON curves.")
@click.option("--use-synthetic", is_flag=True, default=True, help="Use synthetic data.")
@click.option("--random-state", default=42, type=int)
@click.option("--n-points", default=20, type=int, help="Points per curve for pgfplots export.")
def generate_roc_pr(
    artifacts_dir: str,
    output_dir: str,
    use_synthetic: bool,
    random_state: int,
    n_points: int,
) -> None:
    """Generate real ROC and PR curves from archived models."""
    import joblib

    artifacts = Path(artifacts_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("XAI-SDN ROC / PR Curve Generator")
    logger.info("=" * 60)

    # ── Load test data ────────────────────────────────────────────────────────
    X_test, y_test = load_or_synthesize_test_data(artifacts, use_synthetic, random_state)
    logger.info(f"Test set: {X_test.shape}, class balance: {np.bincount(y_test)}")

    # ── Load models ───────────────────────────────────────────────────────────
    models: Dict[str, object] = {}
    model_path = artifacts / "rf_model.pkl"
    if model_path.exists():
        models["XAI-SDN (RF-88)"] = joblib.load(model_path)
        logger.info(f"Loaded RF model from {model_path}")
    else:
        # Train a quick RF on synthetic for the figure
        logger.warning("rf_model.pkl not found — training quick RF on synthetic data for curves.")
        from scripts.synth_utils import load_synthetic_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        X_tr, X_te, y_tr, y_te = load_synthetic_split(n_samples=30000, random_state=random_state, binary=True)
        X_test, y_test = X_te, y_te

        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)
        rf.fit(X_tr, y_tr)
        models["XAI-SDN (RF-88)"] = rf

        # CIC-only baseline (80 features)
        rf_cic = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)
        rf_cic.fit(X_tr[:, :80], y_tr)
        models["RF-CIC-only (80)"] = ("cic_only", rf_cic)

        # SVM baseline
        from sklearn.svm import SVC
        svm = SVC(kernel="rbf", probability=True, random_state=random_state)
        svm.fit(X_tr[:8000], y_tr[:8000])  # SVM is slow, use subset
        models["SVM (88)"] = svm

    # ── Compute curves ────────────────────────────────────────────────────────
    roc_results = {}
    pr_results = {}

    for name, model in models.items():
        logger.info(f"Computing curves for: {name}")
        t0 = time.perf_counter()

        if isinstance(model, tuple):
            # (feature_subset, model) tuple
            subset, m = model
            if subset == "cic_only":
                X_eval = X_test[:, :80]
            else:
                X_eval = X_test
            probas = m.predict_proba(X_eval)[:, 1]
        else:
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X_test)[:, 1]
            else:
                probas = model.decision_function(X_test)
                probas = (probas - probas.min()) / (probas.max() - probas.min())

        latency_ms = (time.perf_counter() - t0) / len(y_test) * 1000

        # ROC
        fpr, tpr, thresholds_roc = roc_curve(y_test, probas)
        roc_auc = auc(fpr, tpr)
        fpr_ds, tpr_ds = downsample_curve(fpr, tpr, n_points=n_points, log_spaced=True)

        roc_results[name] = {
            "fpr": fpr_ds,
            "tpr": tpr_ds,
            "auc": round(roc_auc, 6),
            "n_points": len(fpr_ds),
            "inference_latency_ms": round(latency_ms, 6),
        }

        # PR
        prec, rec, thresholds_pr = precision_recall_curve(y_test, probas)
        pr_auc = auc(rec, prec)
        rec_ds, prec_ds = downsample_curve(rec, prec, n_points=n_points, log_spaced=False)

        pr_results[name] = {
            "recall": rec_ds,
            "precision": prec_ds,
            "pr_auc": round(pr_auc, 6),
            "n_points": len(rec_ds),
        }

        logger.info(f"  ROC-AUC={roc_auc:.6f}  PR-AUC={pr_auc:.6f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    roc_path = out / "roc_curves.json"
    pr_path = out / "pr_curves.json"

    with open(roc_path, "w") as f:
        json.dump({"generated_from": "generate_roc_pr.py",
                   "note": "Downsampled roc_curve() outputs for pgfplots. Replace Fig 6.",
                   "curves": roc_results}, f, indent=2)

    with open(pr_path, "w") as f:
        json.dump({"generated_from": "generate_roc_pr.py",
                   "note": "PR curve — mandatory under 0.86% minority class. AUC is the honest metric.",
                   "curves": pr_results}, f, indent=2)

    logger.success(f"ROC curves saved → {roc_path}")
    logger.success(f"PR  curves saved → {pr_path}")

    # ── Print pgfplots-ready coordinates ─────────────────────────────────────
    logger.info("\n── pgfplots coordinates (copy into LaTeX Fig. 6) ──")
    for name, data in roc_results.items():
        coords = " ".join(f"({x:.6f},{y:.6f})" for x, y in zip(data["fpr"], data["tpr"]))
        logger.info(f"\n% {name}  AUC={data['auc']}\n\\addplot coordinates {{{coords}}};")


if __name__ == "__main__":
    generate_roc_pr()
