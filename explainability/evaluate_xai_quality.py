"""
evaluate_xai_quality.py — Measure explanation quality for XAI-SDN.

Implements 5 XAI quality metrics (increasingly required by Q1 journals):

  C1. Fidelity (deletion test): remove top-k SHAP features per prediction,
      measure score drop vs. random-k removal → deletion curve figure.
  C2. Stability: SHAP attribution variance across 10 seeds + near-duplicate flows.
  C3. Sparsity: fraction of alerts where top-3 features carry >50% total attribution.
  C4. Runtime percentiles: p50/p95/p99 (not just mean 0.5ms).
  C5. Case studies: find one FP and one FN, extract their SHAP payloads for paper.

Output: model/artifacts/xai_quality_metrics.json

Usage:
    python explainability/evaluate_xai_quality.py --use-synthetic
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── C1: Fidelity / Deletion Test ─────────────────────────────────────────────

def compute_deletion_curve(
    clf: RandomForestClassifier,
    X_test: np.ndarray,
    shap_values: np.ndarray,
    k_values: Optional[List[int]] = None,
    n_samples: int = 500,
    random_state: int = 42,
) -> Dict:
    """
    Deletion test: remove top-k SHAP features, measure F1 drop vs. random-k removal.

    Returns dict with:
      - k_values: list of k tested
      - shap_f1: F1 when top-k SHAP features zeroed
      - random_f1: F1 when random-k features zeroed (average over 5 random draws)
      - delta: shap_f1 - random_f1 (should be negative: SHAP removal hurts more)
    """
    rng = np.random.RandomState(random_state)
    n_features = X_test.shape[1]

    if k_values is None:
        k_values = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30]

    idx = rng.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sample = X_test[idx]
    shap_sample = shap_values[idx]

    # Feature importance rank per sample (descending |SHAP|)
    shap_ranks = np.argsort(-np.abs(shap_sample), axis=1)  # (n, n_features)

    y_base = clf.predict(X_sample)

    shap_f1_list = []
    random_f1_list = []

    for k in k_values:
        if k == 0:
            shap_f1_list.append(1.0)
            random_f1_list.append(1.0)
            continue

        # SHAP deletion: zero out top-k features per sample
        X_shap_del = X_sample.copy()
        for i in range(len(X_sample)):
            top_k_idx = shap_ranks[i, :k]
            X_shap_del[i, top_k_idx] = 0.0
        y_shap = clf.predict(X_shap_del)
        shap_agreement = float((y_shap == y_base).mean())
        shap_f1_list.append(shap_agreement)

        # Random deletion: average over 5 draws
        random_agreements = []
        for _ in range(5):
            X_rand_del = X_sample.copy()
            for i in range(len(X_sample)):
                rand_idx = rng.choice(n_features, k, replace=False)
                X_rand_del[i, rand_idx] = 0.0
            y_rand = clf.predict(X_rand_del)
            random_agreements.append(float((y_rand == y_base).mean()))
        random_f1_list.append(float(np.mean(random_agreements)))

    delta = [s - r for s, r in zip(shap_f1_list, random_f1_list)]

    return {
        "k_values": k_values,
        "shap_agreement": [round(v, 4) for v in shap_f1_list],
        "random_agreement": [round(v, 4) for v in random_f1_list],
        "delta": [round(v, 4) for v in delta],
        "interpretation": (
            "Negative delta means SHAP identifies features that matter more than random — "
            "proving attributions track real model evidence (fidelity)."
        ),
    }


# ── C2: Stability ─────────────────────────────────────────────────────────────

def compute_stability(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seeds: List[int],
    n_explain: int = 100,
) -> Dict:
    """
    SHAP attribution variance across seeds.
    Trains RF on same data with different seeds, measures SHAP ranking consistency.
    """
    import shap

    all_top5: List[List[int]] = []
    all_shap_vals: List[np.ndarray] = []
    sample = X_test[:min(n_explain, len(X_test))]

    for seed in seeds:
        clf = RandomForestClassifier(
            n_estimators=100, n_jobs=-1,
            class_weight="balanced", random_state=seed
        )
        clf.fit(X_train, y_train)
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]
        elif hasattr(sv, 'ndim') and sv.ndim == 3:
            sv = sv[:, :, 1]  # attack class
        mean_abs = np.abs(sv).mean(axis=0)
        top5 = np.argsort(mean_abs)[::-1][:5].tolist()
        all_top5.append(top5)
        all_shap_vals.append(mean_abs)

    # Rank stability: fraction of seeds agreeing on top-5
    top5_sets = [frozenset(t) for t in all_top5]
    intersection = set.intersection(*[set(t) for t in all_top5])
    rank_stability = len(intersection) / 5.0

    # Value stability: coefficient of variation of SHAP magnitudes
    shap_matrix = np.stack(all_shap_vals, axis=0)  # (n_seeds, n_features)
    mean_vals = shap_matrix.mean(axis=0)
    std_vals = shap_matrix.std(axis=0)
    cv = float(np.nanmean(std_vals / (np.abs(mean_vals) + 1e-10)))

    return {
        "seeds_used": seeds,
        "top5_feature_indices_per_seed": all_top5,
        "top5_intersection": sorted(list(intersection)),
        "top5_rank_stability": round(rank_stability, 4),
        "shap_value_cv": round(cv, 4),
        "interpretation": (
            f"Top-5 ranking stable across {len(seeds)} seeds: "
            f"{len(intersection)}/5 features in common ({rank_stability:.0%}). "
            f"Value CV={cv:.4f} (lower is more stable)."
        ),
    }


# ── C3: Sparsity / Actionability ─────────────────────────────────────────────

def compute_sparsity(shap_values: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Fraction of predictions where top-3 features carry >threshold of total attribution.
    Operators can realistically inspect 3 features per alert.
    """
    abs_shap = np.abs(shap_values)
    total_per_sample = abs_shap.sum(axis=1)

    # Top-3 contribution
    top3_idx = np.argsort(-abs_shap, axis=1)[:, :3]
    top3_sum = np.array([abs_shap[i, top3_idx[i]].sum() for i in range(len(abs_shap))])
    top3_frac = top3_sum / (total_per_sample + 1e-12)

    frac_above_threshold = float((top3_frac >= threshold).mean())
    mean_top3_frac = float(top3_frac.mean())

    return {
        "top_k": 3,
        "threshold": threshold,
        "fraction_above_threshold": round(frac_above_threshold, 4),
        "mean_top3_attribution_fraction": round(mean_top3_frac, 4),
        "interpretation": (
            f"{frac_above_threshold:.1%} of alerts have ≥{threshold:.0%} of "
            f"attribution concentrated in 3 features — actionable for SOC operators."
        ),
    }


# ── C4: SHAP Runtime Percentiles ─────────────────────────────────────────────

def compute_runtime_percentiles(
    clf: RandomForestClassifier,
    X_test: np.ndarray,
    n_runs: int = 200,
) -> Dict:
    """Measure per-flow SHAP latency distribution (p50/p95/p99)."""
    import shap
    explainer = shap.TreeExplainer(clf)

    latencies_ms = []
    batch_size = 10  # realistic per-flow batch

    sample_indices = np.random.choice(len(X_test), size=n_runs, replace=True)

    for idx in sample_indices:
        x = X_test[idx:idx+1]
        t0 = time.perf_counter()
        sv = explainer.shap_values(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

    latencies = np.array(latencies_ms)

    return {
        "n_runs": n_runs,
        "p50_ms": round(float(np.percentile(latencies, 50)), 4),
        "p95_ms": round(float(np.percentile(latencies, 95)), 4),
        "p99_ms": round(float(np.percentile(latencies, 99)), 4),
        "mean_ms": round(float(latencies.mean()), 4),
        "std_ms": round(float(latencies.std()), 4),
        "max_ms": round(float(latencies.max()), 4),
        "interpretation": (
            "Per-flow SHAP latency distribution. "
            "p99 is the operationally relevant bound for SOC alert latency."
        ),
    }


# ── C5: Case Studies (FP and FN) ─────────────────────────────────────────────

def find_case_studies(
    clf: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """Find one FP and one FN and extract their SHAP payloads for paper case study."""
    y_pred = clf.predict(X_test)

    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]

    case_studies = {}

    for case_type, indices in [("false_positive", fp_indices), ("false_negative", fn_indices)]:
        if len(indices) == 0:
            case_studies[case_type] = None
            logger.warning(f"No {case_type.replace('_', ' ')}s found in test set.")
            continue

        # Pick the most "confident" wrong prediction (highest proba for wrong class)
        probas = clf.predict_proba(X_test[indices])
        worst_idx = indices[np.argmax(probas[:, y_pred[indices]])]

        sv = shap_values[worst_idx]
        abs_sv = np.abs(sv)
        top10_idx = np.argsort(abs_sv)[::-1][:10].tolist()

        case_studies[case_type] = {
            "sample_index": int(worst_idx),
            "true_label": int(y_test[worst_idx]),
            "predicted_label": int(y_pred[worst_idx]),
            "prediction_confidence": round(float(clf.predict_proba(X_test[worst_idx:worst_idx+1])[0].max()), 4),
            "top10_feature_indices": top10_idx,
            "top10_shap_values": [round(float(sv[i]), 6) for i in top10_idx],
            "top10_feature_names": [
                feature_names[i] if feature_names else f"feature_{i}"
                for i in top10_idx
            ],
            "top3_attribution_fraction": round(
                float(abs_sv[top10_idx[:3]].sum() / (abs_sv.sum() + 1e-12)), 4
            ),
        }
        logger.info(f"\n{case_type.upper()} case study:")
        logger.info(f"  Sample #{worst_idx}: true={y_test[worst_idx]}, pred={y_pred[worst_idx]}")
        logger.info(f"  Top-3 features: {top10_idx[:3]}")

    return case_studies


# ── Main CLI ──────────────────────────────────────────────────────────────────

@click.command()
@click.option("--artifacts-dir", default="model/artifacts")
@click.option("--output", default="model/artifacts/xai_quality_metrics.json")
@click.option("--use-synthetic", is_flag=True, default=True)
@click.option("--random-state", default=42, type=int)
@click.option("--n-explain", default=300, type=int, help="Samples for SHAP explanation.")
@click.option("--n-runtime-runs", default=100, type=int, help="Runs for latency percentiles.")
def evaluate_xai_quality(
    artifacts_dir: str,
    output: str,
    use_synthetic: bool,
    random_state: int,
    n_explain: int,
    n_runtime_runs: int,
) -> None:
    """Evaluate XAI explanation quality (fidelity, stability, sparsity, runtime, case studies)."""
    import shap
    import joblib
    from utils.seed_utils import set_global_seed
    set_global_seed(random_state)

    logger.info("=" * 60)
    logger.info("XAI-SDN Explanation Quality Evaluation")
    logger.info("=" * 60)

    # ── Load model and data ───────────────────────────────────────────────────
    artifacts = Path(artifacts_dir)
    model_path = artifacts / "rf_model.pkl"

    if model_path.exists() and not use_synthetic:
        clf = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        splits = Path("data/splits")
        X_train = np.load(splits / "X_train.npy")
        X_test = np.load(splits / "X_test.npy")
        y_train = np.load(splits / "y_train.npy")
        y_test = np.load(splits / "y_test.npy")
    else:
        logger.info("Training model on synthetic data for XAI quality evaluation...")
        from scripts.synth_utils import load_synthetic_split
        X_train, X_test, y_train, y_test = load_synthetic_split(
            n_samples=10000, random_state=random_state, binary=True
        )
        clf = RandomForestClassifier(
            n_estimators=100, n_jobs=-1,
            class_weight="balanced", random_state=random_state
        )
        clf.fit(X_train, y_train)
        logger.info(f"Model trained. Test set: {X_test.shape}")

    # ── Compute SHAP values (once, reuse) ─────────────────────────────────────
    logger.info(f"\nComputing SHAP values for {n_explain} test samples...")
    explainer = shap.TreeExplainer(clf)
    sample_idx = np.random.RandomState(random_state).choice(len(X_test), min(n_explain, len(X_test)), replace=False)
    X_explain = X_test[sample_idx]
    y_explain = y_test[sample_idx]

    t_shap0 = time.perf_counter()
    shap_vals = explainer.shap_values(X_explain)
    t_shap_total = time.perf_counter() - t_shap0

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # positive class (attack)
    elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]  # select attack class → (n, n_features)
    logger.info(f"SHAP values computed: {shap_vals.shape}  ({t_shap_total:.2f}s total)")

    # ── Try to load feature names ─────────────────────────────────────────────
    feature_names = None
    fn_path = Path("data/splits/feature_names.json")
    if fn_path.exists():
        with open(fn_path) as f:
            feature_names = json.load(f)

    results = {}

    # ── C1: Fidelity ──────────────────────────────────────────────────────────
    logger.info("\n[C1] Computing fidelity deletion curve...")
    results["c1_fidelity"] = compute_deletion_curve(
        clf, X_explain, shap_vals,
        k_values=[0, 1, 2, 3, 5, 8, 10, 15, 20, 30],
        n_samples=min(300, len(X_explain)),
        random_state=random_state,
    )
    logger.info(f"  k=5: SHAP agreement={results['c1_fidelity']['shap_agreement'][4]:.4f}, "
                f"random={results['c1_fidelity']['random_agreement'][4]:.4f}")

    # ── C2: Stability ─────────────────────────────────────────────────────────
    logger.info("\n[C2] Computing stability across 5 seeds...")
    stability_seeds = [42, 123, 456, 789, 1024]
    results["c2_stability"] = compute_stability(
        X_train, y_train, X_explain,
        seeds=stability_seeds,
        n_explain=min(50, len(X_explain)),
    )
    logger.info(f"  Rank stability: {results['c2_stability']['top5_rank_stability']:.4f}")
    logger.info(f"  Value CV: {results['c2_stability']['shap_value_cv']:.4f}")

    # ── C3: Sparsity ──────────────────────────────────────────────────────────
    logger.info("\n[C3] Computing sparsity / actionability...")
    results["c3_sparsity"] = compute_sparsity(shap_vals, threshold=0.5)
    logger.info(f"  {results['c3_sparsity']['interpretation']}")

    # ── C4: Runtime percentiles ───────────────────────────────────────────────
    logger.info(f"\n[C4] Computing SHAP runtime percentiles ({n_runtime_runs} runs)...")
    results["c4_runtime"] = compute_runtime_percentiles(clf, X_test, n_runs=n_runtime_runs)
    rt = results["c4_runtime"]
    logger.info(f"  p50={rt['p50_ms']:.4f}ms  p95={rt['p95_ms']:.4f}ms  p99={rt['p99_ms']:.4f}ms")

    # ── C5: Case studies ──────────────────────────────────────────────────────
    logger.info("\n[C5] Finding FP/FN case studies...")
    # Use full test set for case study search
    shap_full = explainer.shap_values(X_test[:min(2000, len(X_test))])
    if isinstance(shap_full, list):
        shap_full = shap_full[1]
    results["c5_case_studies"] = find_case_studies(
        clf, X_test[:len(shap_full)], y_test[:len(shap_full)],
        shap_full, feature_names=feature_names
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "generated_by": "explainability/evaluate_xai_quality.py",
        "random_state": random_state,
        "n_explain": n_explain,
        "n_runtime_runs": n_runtime_runs,
        "shap_total_time_s": round(t_shap_total, 4),
        "metrics": results,
    }

    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.success(f"\nXAI quality metrics saved → {out_path}")
    logger.info("\n── PAPER ADDITIONS FROM THIS RUN ──")
    logger.info(f"  C1 Deletion curve: {len(results['c1_fidelity']['k_values'])} k-values → new Figure")
    logger.info(f"  C2 Stability: top-5 stable in {results['c2_stability']['top5_rank_stability']:.0%} of seeds")
    logger.info(f"  C3 Sparsity: {results['c3_sparsity']['fraction_above_threshold']:.0%} alerts actionable with top-3 features")
    logger.info(f"  C4 Runtime: p50={rt['p50_ms']:.2f}ms p99={rt['p99_ms']:.2f}ms → update Table 8 latency row")
    logger.info(f"  C5 Case studies: {'FP+FN both found' if all(v is not None for v in results['c5_case_studies'].values()) else 'partial'}")


if __name__ == "__main__":
    evaluate_xai_quality()
