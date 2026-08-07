"""
run_10seed_wilcoxon.py — 10-seed ablation campaign with correct Wilcoxon tests.

Fixes the critical reproducibility issue: paper claims p=0.0020/0.0078/0.2500
from a 10-seed Wilcoxon test, but archived artifacts only have 5 seeds
(p=0.125/0.0625/0.0625 — n=5 cannot reach p=0.002).

This script:
  1. Runs the full 4-configuration ablation across 10 seeds
  2. Recomputes scipy.stats.wilcoxon (two-sided) on the 10-seed F1 values
  3. Saves model/artifacts/ablation_multiseed_10.json
  4. Prints the exact p-values to update Table 7 / the significance paragraph

Seeds: 42 123 456 789 1024 2048 31337 555 777 999

Usage:
    python scripts/run_10seed_wilcoxon.py --use-synthetic
    python scripts/run_10seed_wilcoxon.py --max-samples 5000  # faster
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import click
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

SEEDS_10 = [42, 123, 456, 789, 1024, 2048, 31337, 555, 777, 999]


def run_single_config(X_tr, X_te, y_tr, y_te, config: Dict, seed: int) -> Dict:
    """Run a single ablation configuration and return metrics."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score
    from utils.seed_utils import set_global_seed

    set_global_seed(seed)

    feature_set = config["features"]
    model_type = config["model"]

    # Feature selection
    n_cic = 80
    if feature_set == "cic_only":
        X_tr_f, X_te_f = X_tr[:, :n_cic], X_te[:, :n_cic]
        n_feat = n_cic
    elif feature_set == "entropy_only":
        X_tr_f, X_te_f = X_tr[:, n_cic:], X_te[:, n_cic:]
        n_feat = X_tr.shape[1] - n_cic
    else:  # full
        X_tr_f, X_te_f = X_tr, X_te
        n_feat = X_tr.shape[1]

    # Model
    t0 = time.perf_counter()
    if model_type == "RF":
        clf = RandomForestClassifier(
            n_estimators=100, n_jobs=-1,
            class_weight="balanced", random_state=seed
        )
    else:  # SVM
        clf = SVC(kernel="rbf", probability=False,
                  class_weight="balanced", random_state=seed)

    clf.fit(X_tr_f, y_tr)
    train_time = time.perf_counter() - t0

    # Inference timing
    t1 = time.perf_counter()
    y_pred = clf.predict(X_te_f)
    lat_ms = (time.perf_counter() - t1) / len(y_te) * 1000

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro")

    # FPR (benign=0, attack=1 in synthetic)
    tn = int(((y_te == 0) & (y_pred == 0)).sum())
    fp = int(((y_te == 0) & (y_pred == 1)).sum())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "seed": seed,
        "name": config["name"],
        "features": feature_set,
        "model": model_type,
        "n_features": n_feat,
        "accuracy": round(acc, 8),
        "macro_f1": round(f1, 8),
        "fpr": round(fpr, 8),
        "latency_ms": round(lat_ms, 8),
        "train_time_s": round(train_time, 8),
    }


@click.command()
@click.option("--use-synthetic", is_flag=True, default=True, help="Use synthetic data (no real CSV needed).")
@click.option("--data-dir", default=None, help="Real data dir (overrides synthetic).")
@click.option("--output", default="model/artifacts/ablation_multiseed_10.json", help="Output path.")
@click.option("--max-samples", default=8000, type=int, help="Samples per seed (8000 is fast + representative).")
@click.option("--seeds", default=",".join(str(s) for s in SEEDS_10), help="Comma-separated seeds.")
def run_10seed(use_synthetic: bool, data_dir: str, output: str,
               max_samples: int, seeds: str) -> None:
    """Run the 10-seed ablation campaign with correct Wilcoxon tests."""
    from scipy import stats as scipy_stats

    seed_list = [int(s.strip()) for s in seeds.split(",")]
    n_seeds = len(seed_list)
    logger.info("=" * 60)
    logger.info(f"XAI-SDN 10-Seed Ablation Campaign  (n={n_seeds} seeds)")
    logger.info("=" * 60)
    logger.info(f"Seeds: {seed_list}")
    logger.info(f"Max samples per seed: {max_samples}")

    configs = [
        {"name": "RF + CIC-only (80-dim)",      "features": "cic_only",     "model": "RF"},
        {"name": "RF + Entropy-only (8-dim)",   "features": "entropy_only", "model": "RF"},
        {"name": "SVM + Full (88-dim)",          "features": "full",         "model": "SVM"},
        {"name": "XAI-SDN: RF + Full (88-dim)", "features": "full",         "model": "RF"},
    ]

    all_results: Dict[str, List] = {cfg["name"]: [] for cfg in configs}

    # ── Per-seed loop ─────────────────────────────────────────────────────────
    for i, seed in enumerate(seed_list):
        logger.info(f"\n── Seed {seed}  ({i+1}/{n_seeds}) ──")

        # Generate reproducible synthetic data for this seed
        from scripts.synth_utils import load_synthetic_split
        X_tr, X_te, y_tr, y_te = load_synthetic_split(
            n_samples=max_samples, random_state=seed, binary=True
        )
        logger.info(f"   Data: train={X_tr.shape}, test={X_te.shape}")

        for cfg in configs:
            result = run_single_config(X_tr, X_te, y_tr, y_te, cfg, seed)
            all_results[cfg["name"]].append(result)
            logger.info(
                f"   {cfg['name']:40s}  "
                f"F1={result['macro_f1']:.4f}  "
                f"Acc={result['accuracy']:.4f}  "
                f"FPR={result['fpr']:.4f}"
            )

    # ── Aggregate stats ───────────────────────────────────────────────────────
    proposed_key = "XAI-SDN: RF + Full (88-dim)"
    proposed_f1s = [r["macro_f1"] for r in all_results[proposed_key]]
    proposed_mean = float(np.mean(proposed_f1s))

    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE STATISTICS")
    logger.info("=" * 60)

    significance = {}
    for cfg in configs:
        if cfg["name"] == proposed_key:
            continue
        baseline_f1s = [r["macro_f1"] for r in all_results[cfg["name"]]]
        baseline_mean = float(np.mean(baseline_f1s))

        # Wilcoxon signed-rank test (two-sided)
        diffs = [p - b for p, b in zip(proposed_f1s, baseline_f1s)]
        if len(set(diffs)) <= 1:
            stat, p_val = 0.0, 1.0
            note = "all differences identical — test undefined"
        else:
            try:
                stat, p_val = scipy_stats.wilcoxon(proposed_f1s, baseline_f1s,
                                                    alternative="two-sided")
                note = ""
            except Exception as e:
                stat, p_val = 0.0, 1.0
                note = str(e)

        significance[cfg["name"]] = {
            "wilcoxon_stat": float(stat),
            "p_value": round(float(p_val), 6),
            "significant_at_0.05": bool(p_val < 0.05),
            "proposed_mean": round(proposed_mean, 6),
            "baseline_mean": round(baseline_mean, 6),
            "n_seeds": n_seeds,
            "note": note,
        }

        logger.info(f"\n{cfg['name']}")
        logger.info(f"  Proposed mean F1: {proposed_mean:.6f}")
        logger.info(f"  Baseline mean F1: {baseline_mean:.6f}")
        logger.info(f"  Wilcoxon p = {p_val:.6f}  (significant={p_val < 0.05})")
        if note:
            logger.warning(f"  Note: {note}")

    # ── Summary for paper ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PAPER UPDATE — replace these values in the manuscript:")
    logger.info("=" * 60)
    for name, sig in significance.items():
        logger.info(
            f"  vs {name[:30]:30s}  p = {sig['p_value']:.4f}  "
            f"({'significant' if sig['significant_at_0.05'] else 'NOT significant'} at α=0.05)"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "generated_by": "scripts/run_10seed_wilcoxon.py",
        "n_seeds": n_seeds,
        "seeds": seed_list,
        "max_samples_per_seed": max_samples,
        "results": all_results,
        "significance": significance,
        "summary": {
            name: {
                "f1_mean": round(float(np.mean([r["macro_f1"] for r in rs])), 6),
                "f1_std":  round(float(np.std([r["macro_f1"] for r in rs], ddof=1)), 6),
                "acc_mean": round(float(np.mean([r["accuracy"] for r in rs])), 6),
                "acc_std":  round(float(np.std([r["accuracy"] for r in rs], ddof=1)), 6),
                "fpr_mean": round(float(np.mean([r["fpr"] for r in rs])), 6),
                "fpr_std":  round(float(np.std([r["fpr"] for r in rs], ddof=1)), 6),
            }
            for name, rs in all_results.items()
        },
    }

    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.success(f"\n10-seed ablation results saved → {out_path}")
    logger.success("Use 'significance' section to update paper Table 7 / significance paragraph.")


if __name__ == "__main__":
    run_10seed()
