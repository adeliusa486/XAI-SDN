"""E14 - ROC and precision-recall curves for every method.

Produces the curve data for every model evaluated in E1 and E9, and pairs each
ROC curve with a precision-recall curve, because under 99:1 prevalence the ROC
curve flatters every classifier and the PR curve does not.

Curves are downsampled to a fixed number of points so the figure stays a
reasonable size without distorting the shape, and AUC and average precision are
computed on the full score vectors with bootstrap confidence intervals.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

from paths import RESULTS, log_event, save_result  # noqa: E402

from sklearn.metrics import (average_precision_score, confusion_matrix,  # noqa: E402
                             f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

SCORES = RESULTS / "scores"
N_POINTS = 400
N_BOOT = 150


def thin(x: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= n:
        return x, y
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return x[idx], y[idx]



def sweep_macro_f1(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Threshold maximising macro F1, computed in one sorted pass.

    Calling f1_score once per candidate threshold re-scans the whole score vector
    every time. Sorting once and accumulating class counts gives the same answer
    for every candidate at once, which matters here because the vectors are of
    the order of a million rows.
    """
    order = np.argsort(s, kind="mergesort")[::-1]
    ys = y[order].astype(np.int64)
    ss = s[order]
    P = int(ys.sum())
    N = int(len(ys) - P)
    if P == 0 or N == 0:
        return 0.5, 0.0

    tp = np.cumsum(ys)                      # predicted positive = prefix
    fp = np.cumsum(1 - ys)
    fn = P - tp
    tn = N - fp

    # boundaries between distinct scores; a threshold inside a tie is meaningless
    keep = np.ones(len(ss), dtype=bool)
    keep[:-1] = ss[1:] != ss[:-1]
    idx = np.flatnonzero(keep)

    tp, fp, fn, tn = tp[idx], fp[idx], fn[idx], tn[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_pos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_neg = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    macro = 0.5 * (f1_pos + f1_neg)
    j = int(np.argmax(macro))
    return float(ss[idx][j]), float(macro[j])


def boot_auc(y, s, n_boot=N_BOOT, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, aps = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        aucs.append(roc_auc_score(y[i], s[i]))
        aps.append(average_precision_score(y[i], s[i]))

    def ci(a):
        if not a:
            return None
        a = np.asarray(a, float)
        return {"mean": float(a.mean()), "lo95": float(np.percentile(a, 2.5)),
                "hi95": float(np.percentile(a, 97.5))}
    return {"roc_auc": ci(aucs), "average_precision": ci(aps), "n_boot": len(aucs)}


def main() -> int:
    log_event("E14", "start")
    t_all = time.time()
    out: dict = {"experiment": "E14",
                 "objective": "ROC and precision-recall curves for every method",
                 "n_curve_points": N_POINTS, "n_bootstrap": N_BOOT}

    if not SCORES.exists():
        out["error"] = ("no score files found; run E1 and E9 first - they persist "
                        "per-model scores into 04_results/scores/")
        save_result("E14_roc_pr_curves", out)
        log_event("E14", "failed", reason=out["error"])
        print(out["error"])
        return 1

    score_files = sorted(SCORES.glob("*_score.npy"))
    print(f"found {len(score_files)} score files", flush=True)

    curves, summary = [], {}
    for f in score_files:
        stem = f.name[:-len("_score.npy")]
        parts = stem.split("_", 2)          # E1 | A/B | model-slug
        exp, proto, model = parts[0], parts[1], parts[2]

        yfile = SCORES / f"{exp}_{proto}_{model}_ytrue.npy"
        if not yfile.exists():
            yfile = SCORES / f"{exp}_{proto}_ytrue.npy"
        if not yfile.exists():
            print(f"  {stem}: no matching ytrue, skipped", flush=True)
            continue

        s = np.load(f).astype(np.float64)
        y = np.load(yfile).astype(np.int8)
        if len(s) != len(y):
            print(f"  {stem}: length mismatch {len(s)} vs {len(y)}, skipped",
                  flush=True)
            continue
        if len(np.unique(y)) < 2:
            print(f"  {stem}: single-class ground truth, skipped", flush=True)
            continue

        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        fx, ty = thin(fpr, tpr, N_POINTS)
        rx, py = thin(rec[::-1], prec[::-1], N_POINTS)

        # Per-model operating point. A single fixed tau across models is not a
        # fair comparison when their score distributions differ: the graph model
        # ranks almost perfectly (ROC-AUC 0.997) yet classifies almost nothing as
        # an attack at tau = 0.70, because its sigmoid outputs sit low. We
        # therefore select each model's threshold by maximising macro F1 on the
        # first half of its scores and report performance on the second half,
        # alongside the fixed-threshold figure, so both readings are available.
        half = len(y) // 2
        yv, sv = y[:half], s[:half]
        yt_, st_ = y[half:], s[half:]
        best_tau, best_f1 = 0.5, -1.0
        if len(np.unique(yv)) > 1:
            best_tau, best_f1 = sweep_macro_f1(yv, sv)
        pred_sel = (st_ >= best_tau).astype(np.int8)
        pred_fix = (st_ >= 0.70).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(yt_, pred_sel, labels=[0, 1]).ravel()
        operating = {
            "selected_tau": round(best_tau, 6),
            "macro_f1_at_selected_tau": float(f1_score(yt_, pred_sel,
                                                       average="macro",
                                                       zero_division=0)),
            "macro_f1_at_fixed_tau_0.70": float(f1_score(yt_, pred_fix,
                                                         average="macro",
                                                         zero_division=0)),
            "false_positives": int(fp), "false_negatives": int(fn),
            "n_eval": int(len(yt_)),
        }

        key = f"{exp}/{proto}/{model}"
        summary[key] = {
            "experiment": exp, "protocol": proto, "model": model.replace("_", " "),
            "n": int(len(y)), "positive_prevalence": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, s)),
            "average_precision": float(average_precision_score(y, s)),
            "baseline_average_precision": float(y.mean()),
            "bootstrap": boot_auc(y, s),
            "operating_point": operating,
        }
        for a, b in zip(fx, ty):
            curves.append({"key": key, "curve": "roc", "x": float(a), "y": float(b)})
        for a, b in zip(rx, py):
            curves.append({"key": key, "curve": "pr", "x": float(a), "y": float(b)})
        sm = summary[key]
        print(f"  {key:44s} AUC={sm['roc_auc']:.6f} AP={sm['average_precision']:.6f} "
              f"| F1@tau*={operating['macro_f1_at_selected_tau']:.4f} "
              f"(tau*={operating['selected_tau']:.3f}) "
              f"vs F1@0.70={operating['macro_f1_at_fixed_tau_0.70']:.4f}", flush=True)

    pd.DataFrame(curves).to_csv(RESULTS / "E14_curves.csv", index=False)
    pd.DataFrame([{k: v for k, v in s.items() if k != "bootstrap"} |
                  {"roc_auc_lo95": (s["bootstrap"]["roc_auc"] or {}).get("lo95"),
                   "roc_auc_hi95": (s["bootstrap"]["roc_auc"] or {}).get("hi95"),
                   "ap_lo95": (s["bootstrap"]["average_precision"] or {}).get("lo95"),
                   "ap_hi95": (s["bootstrap"]["average_precision"] or {}).get("hi95")}
                 for s in summary.values()]
                ).to_csv(RESULTS / "E14_auc_summary.csv", index=False)

    out["curves"] = summary
    out["interpretation_note"] = (
        "Under the test prevalence reported per row, a classifier that predicts at "
        "random attains an average precision equal to that prevalence and a ROC-AUC "
        "of 0.5. The gap between a model's ROC-AUC and its average precision is "
        "therefore the honest measure of how much the imbalance flatters the ROC "
        "curve, which is the point that matters for threshold selection.")
    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E14_roc_pr_curves", out)
    log_event("E14", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
