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

N_CIC = len(CIC_FEATURE_NAMES)  # 80
N_ENT = len(ENTROPY_FEATURE_NAMES)  # 8


@click.command()
@click.option("--use-synthetic/--no-use-synthetic", default=True, help="Use synthetic data.")
@click.option("--data-dir", default=None, help="Real data directory.")
@click.option("--output", default="model/artifacts/ablation_results.json", help="Output path.")
@click.option("--random-state", default=42, type=int, help="Global random seed for reproducibility.")
@click.option("--seeds", default="42", help="Comma-separated seeds for multi-run.")
@click.option("--max-samples", default=10000, type=int, help="Max samples for CPU-intensive ablation runs.")
def run_ablation(
    use_synthetic: bool,
    data_dir: Optional[str],
    output: str,
    random_state: int,
    seeds: str,
    max_samples: int,
) -> None:
    """Run the ablation study comparing feature subsets and model variants."""
    from utils.seed_utils import set_global_seed
    from scipy import stats as scipy_stats
    set_global_seed(random_state)
    logger.info("=" * 60)
    logger.info("XAI-SDN Ablation Study")
    logger.info("=" * 60)

    base_seed_list = [int(s.strip()) for s in seeds.split(",")]
    seed_list = []
    for s in base_seed_list:
        for i in range(10):
            seed_list.append(s + i)
            
    configs = [
        {"name": "RF + CIC-only (80-dim)", "features": "cic_only", "model": "RF"},
        {"name": "RF + Entropy-only (8-dim)", "features": "entropy_only", "model": "RF"},
        {"name": "SVM + Full (88-dim)", "features": "full", "model": "SVM"},
        {"name": "XAI-SDN: RF + Full (88-dim)", "features": "full", "model": "RF"},
    ]
    all_results = {cfg["name"]: [] for cfg in configs}

    for seed in seed_list:
        set_global_seed(seed)
        logger.info(f"\n--- Running Seed: {seed} ---")

        if use_synthetic or data_dir is None:
            # Synthetic path: completely self-contained, no mixing with real files.
            logger.info("Running synthetic ablation data path...")
            X_syn, y_syn_raw, le = load_synthetic_data(n_samples=8000, random_state=seed)
            X_train_f, X_test_f, y_train_raw, y_test_raw = train_test_split(
                X_syn, y_syn_raw, test_size=0.30, stratify=y_syn_raw, random_state=seed
            )
            le.fit(y_train_raw)
            y_train = le.transform(y_train_raw)
            y_test = le.transform(y_test_raw)
        else:
            # Real path: correctly unpack the 6-tuple returned by load_real_data.
            logger.info("Running real-world ablation data path...")
            from model.train import load_real_data
            X_train_f, X_test_f, y_train, y_test, le, scaler = load_real_data(
                data_dir, test_size=0.30, random_state=seed
            )
            
            # Subsample for tractability (SVM RBF training time/memory guard)
            if max_samples is not None and X_train_f.shape[0] > max_samples:
                logger.info(f"Subsampling real train split to {max_samples} for tractability...")
                _, X_train_f, _, y_train = train_test_split(
                    X_train_f, y_train, test_size=max_samples, stratify=y_train, random_state=seed
                )
                test_limit = int(max_samples * 0.3)
                if X_test_f.shape[0] > test_limit:
                    logger.info(f"Subsampling real test split to {test_limit} for tractability...")
                    _, X_test_f, _, y_test = train_test_split(
                        X_test_f, y_test, test_size=test_limit, stratify=y_test, random_state=seed
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
            n_estimators=200, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=-1
        )
        SVM_PARAMS = dict(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=seed,
            probability=True,
        )

        for cfg in configs:
            logger.info(f"Running: {cfg['name']}")

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
                "seed": seed,
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
            all_results[cfg["name"]].append(result)

            logger.info(
                f"  Acc={acc:.4f}  F1={f1:.4f}  FPR={fpr*100:.2f}%  "
                f"Lat={latency_ms:.2f}ms  Features={X_tr.shape[1]}"
            )

    # Statistical significance: proposed system vs each baseline
    proposed_key = "XAI-SDN: RF + Full (88-dim)"
    proposed_accs = [r["accuracy"] for r in all_results[proposed_key]]

    sig_results = {}
    for name, results in all_results.items():
        if name == proposed_key:
            continue
        baseline_accs = [r["accuracy"] for r in results]
        
        # If lengths are > 1 and all values are identically matched (which can happen with synthetic), wilcoxon might error
        try:
            stat, p_value = scipy_stats.wilcoxon(proposed_accs, baseline_accs)
        except ValueError:
            stat, p_value = 0.0, 1.0 # E.g. differences are all zero
            
        sig_results[name] = {
            "wilcoxon_stat": float(stat),
            "p_value": float(p_value),
            "significant_at_0.05": bool(p_value < 0.05),
            "proposed_mean": float(sum(proposed_accs)/len(proposed_accs)) if proposed_accs else 0.0,
            "baseline_mean": float(sum(baseline_accs)/len(baseline_accs)) if baseline_accs else 0.0,
        }
        logger.info(
            f"  {name}: p={p_value:.4f} "
            f"({'✓ significant' if p_value < 0.05 else '✗ not significant'})"
        )

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("=" * 80)
    header = f"{'Configuration':<35} {'Acc%':>7} {'F1%':>7} {'FPR%':>7} {'ΔAcc':>7}"
    logger.info(header)
    logger.info("-" * 80)
    baseline_acc = sum([r["accuracy"] for r in all_results["RF + CIC-only (80-dim)"]]) / len(seed_list)
    for name, r_list in all_results.items():
        mean_acc = sum([r["accuracy"] for r in r_list]) / len(r_list)
        mean_f1 = sum([r["macro_f1"] for r in r_list]) / len(r_list)
        mean_fpr = sum([r["fpr"] for r in r_list]) / len(r_list)
        
        delta = mean_acc - baseline_acc
        delta_str = f"+{delta*100:.1f}%" if delta >= 0 else f"{delta*100:.1f}%"
        logger.info(
            f"{name:<35} "
            f"{mean_acc*100:>7.1f} "
            f"{mean_f1*100:>7.1f} "
            f"{mean_fpr*100:>7.2f} "
            f"{delta_str:>7}"
        )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "significance": sig_results}, f, indent=2)
    logger.info(f"\nAblation results saved to {out_path}")


if __name__ == "__main__":
    run_ablation()
