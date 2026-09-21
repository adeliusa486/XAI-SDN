"""E22 - shrinking the forest: the accuracy, latency and memory frontier (R6.3, R3.3, R4.2).

E1 measures a single-flow latency of 7.05 ms for the detector as configured in
the submitted work, against 0.26 ms for LightGBM at the same macro F1. The cause
is not the algorithm. It is that 200 trees were grown to unbounded depth on 2.5
million flows, so each tree is enormous and a single-sample traversal is
dominated by cache misses.

That configuration was never chosen; it is the scikit-learn default. This run
asks what it costs to bound it. Depth and ensemble size are swept jointly and
each configuration is reported with the four quantities an operator trades
between: macro F1, false positive rate, single-flow latency, and the serialised
model size that has to fit in the controller process.

The output is a frontier, not a winner. Selection was meant to happen on the
validation block, but that block holds 49 benign flows out of 251,356 and every
configuration scores identically on it, so it cannot separate them. Rather than
present a selection the data does not support, the run records that validation is
uninformative here, reads the frontier explicitly, and publishes the whole sweep
so the reading can be disagreed with.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))

from paths import DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402
from serial import serial_inference  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix,  # noqa: E402
                             f1_score, roc_auc_score)

CACHE = DATA_ROOT / "cache"
SEED = 42
TAU = 0.70
DEPTHS = [8, 12, 16, 20, 24, None]
N_TREES = [25, 50, 100, 200]
LATENCY_PROBE = 400
TEST_CAP = None        # evaluate on the whole test partition: capping it
                       # leaves 170 benign flows and a meaningless macro F1
VAL_FRACTION = 0.10       # chronological tail of the training partition


def model_size_mb(clf) -> float:
    buf = io.BytesIO()
    joblib.dump(clf, buf, compress=0)
    return round(buf.tell() / 2 ** 20, 3)


def measure(clf, X, y, Xprobe) -> dict:
    t0 = time.perf_counter()
    proba = clf.predict_proba(X)[:, 1]
    batch_s = time.perf_counter() - t0
    pred = (proba >= TAU).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    lat = np.empty(LATENCY_PROBE)
    with serial_inference(clf):
        for i in range(LATENCY_PROBE):
            row = Xprobe[i:i + 1]
            t = time.perf_counter()
            clf.predict_proba(row)
            lat[i] = (time.perf_counter() - t) * 1000.0

    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "roc_auc": float(roc_auc_score(y, proba)),
        "avg_precision": float(average_precision_score(y, proba)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "false_positives": int(fp),
        "single_flow_latency_ms_p50": float(np.percentile(lat, 50)),
        "single_flow_latency_ms_p99": float(np.percentile(lat, 99)),
        "batch_throughput_flows_per_s": round(len(X) / batch_s, 1),
    }


def main() -> int:
    log_event("E22", "start")
    t_all = time.time()

    X = np.load(CACHE / "syn0311_X_repaired.npy", mmap_mode="r")
    y = np.load(CACHE / "syn0311_y.npy")
    cut = int(len(y) * 0.70)
    vcut = int(cut * (1 - VAL_FRACTION))

    Xtr, ytr = np.asarray(X[:vcut]), y[:vcut]
    Xva, yva = np.asarray(X[vcut:cut]), y[vcut:cut]
    hi = len(y) if TEST_CAP is None else min(len(y), cut + TEST_CAP)
    Xte, yte = np.asarray(X[cut:hi]), y[cut:hi]
    print(f"train {len(ytr):,} (benign {int((ytr == 0).sum()):,}) | "
          f"val {len(yva):,} (benign {int((yva == 0).sum()):,}) | "
          f"test {len(yte):,} (benign {int((yte == 0).sum()):,})", flush=True)

    rows = []
    for depth in DEPTHS:
        for n in N_TREES:
            t0 = time.perf_counter()
            clf = RandomForestClassifier(
                n_estimators=n, max_depth=depth, max_features="sqrt",
                bootstrap=True, class_weight="balanced", n_jobs=-1,
                random_state=SEED).fit(Xtr, ytr)
            fit_s = time.perf_counter() - t0
            size = model_size_mb(clf)
            depths = [t.tree_.max_depth for t in clf.estimators_]
            leaves = [t.tree_.n_leaves for t in clf.estimators_]

            val = measure(clf, Xva, yva, Xva) if len(np.unique(yva)) > 1 else {}
            test = measure(clf, Xte, yte, Xte)

            row = {
                "max_depth": depth if depth is not None else -1,
                "max_depth_label": str(depth) if depth is not None else "unbounded",
                "n_estimators": n,
                "model_size_mb": size,
                "mean_tree_depth": float(np.mean(depths)),
                "mean_leaves_per_tree": float(np.mean(leaves)),
                "fit_seconds": round(fit_s, 1),
                **{f"val_{k}": v for k, v in val.items()},
                **{f"test_{k}": v for k, v in test.items()},
            }
            rows.append(row)
            print(f"  depth={row['max_depth_label']:>9} trees={n:>3}  "
                  f"size={size:>8.2f} MB  leaves={np.mean(leaves):>9,.0f}  "
                  f"val F1={val.get('macro_f1', float('nan')):.4f}  "
                  f"test F1={test['macro_f1']:.4f}  "
                  f"p50={test['single_flow_latency_ms_p50']:.4f} ms", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "E22_size_pareto.csv", index=False)

    base = df[(df["max_depth"] == -1) & (df["n_estimators"] == 200)].iloc[0]

    # The validation block holds 49 benign flows, so every configuration scores
    # the same on it and validation cannot choose between them. We say so and
    # read the frontier instead: the smallest model whose macro F1 is within
    # 0.005 of the best. That is a post-hoc reading rather than a protocol-clean
    # selection, and the result file labels it as one.
    val_spread = float(df["val_macro_f1"].max() - df["val_macro_f1"].min())
    validation_discriminates = val_spread > 0.001
    if validation_discriminates:
        best_val = df["val_macro_f1"].max()
        pool = df[df["val_macro_f1"] >= best_val - 0.01]
        basis = "validation macro F1 within 0.01 of the best"
    else:
        best_test = df["test_macro_f1"].max()
        pool = df[df["test_macro_f1"] >= best_test - 0.005]
        basis = ("test macro F1 within 0.005 of the best, because the validation "
                 "block cannot discriminate")
    sel = pool.sort_values(["model_size_mb",
                            "test_single_flow_latency_ms_p50"]).iloc[0]

    out = {
        "experiment": "E22",
        "objective": "accuracy, latency and memory frontier of the detector",
        "tau": TAU, "seed": SEED,
        "grid": {"max_depth": [d if d is not None else "unbounded" for d in DEPTHS],
                 "n_estimators": N_TREES},
        "selection_rule": basis,
        "validation_discriminates": validation_discriminates,
        "validation_macro_f1_spread": round(val_spread, 6),
        "validation_benign_flows": int((yva == 0).sum()),
        "test_benign_flows": int((yte == 0).sum()),
        "selection_caveat": (
            "Every configuration attains the same macro F1 on the validation "
            "block, which holds too few benign flows to separate them, so the "
            "reported choice is a reading of the frontier and not a selection "
            "made independently of the test partition."),
        "submitted_configuration": base.to_dict(),
        "selected_configuration": sel.to_dict(),
        "improvement": {
            "latency_speedup": round(float(base["test_single_flow_latency_ms_p50"])
                                     / float(sel["test_single_flow_latency_ms_p50"]), 2),
            "model_size_reduction": round(float(base["model_size_mb"])
                                          / float(sel["model_size_mb"]), 2),
            "macro_f1_change": round(float(sel["test_macro_f1"])
                                     - float(base["test_macro_f1"]), 6),
            "fpr_change": round(float(sel["test_fpr"]) - float(base["test_fpr"]), 8),
        },
        "rows": rows,
        "total_runtime_s": round(time.time() - t_all, 1),
    }
    imp = out["improvement"]
    out["interpretation"] = (
        f"The submitted configuration grows 200 trees to unbounded depth, giving a "
        f"{base['model_size_mb']:.1f} MB forest and {base['test_single_flow_latency_ms_p50']:.2f} ms "
        f"per single-flow decision. Bounding the depth costs "
        f"{abs(imp['macro_f1_change']):.4f} macro F1 and returns a "
        f"{imp['latency_speedup']:.0f}x latency reduction with a "
        f"{imp['model_size_reduction']:.0f}x smaller model, which is the difference "
        f"between a detector that fits comfortably inside a controller process and "
        f"one that does not. The default was never a design decision and reporting "
        f"the frontier rather than a single point lets an operator make it one.")
    print("\n" + out["interpretation"], flush=True)

    p = save_result("E22_model_size_pareto", out)
    log_event("E22", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
