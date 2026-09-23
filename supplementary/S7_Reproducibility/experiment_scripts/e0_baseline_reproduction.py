"""E0 - Establish the baseline under both candidate split protocols.

Gate for every other experiment. Runs both protocols on the same data:

  * 'stratified' - random stratified 70/30
  * 'temporal'   - all training flows precede all test flows

and reports how far apart they place the headline figures. Also records the
variance of the candidate entropy features needed by E16, and caches the
cleaned matrices so the 1.87 GB CSV is parsed exactly once.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

import data as D  # noqa: E402
from paths import DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_fscore_support, roc_auc_score,
                             average_precision_score)
from sklearn.model_selection import train_test_split  # noqa: E402

CACHE = DATA_ROOT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "n_train": 2514862, "n_test": 1077798,
    "test_benign": 9311, "test_attack": 1068487,
    "accuracy": 0.999987, "macro_f1": 0.999621,
    "fpr": 0.000537, "fnr": 0.0000084, "auc_roc": 1.0,
}

RF_KW = dict(n_estimators=200, max_depth=None, max_features="sqrt",
             bootstrap=True, class_weight="balanced", n_jobs=-1, random_state=42)
TAU = 0.70


def metrics(y_true, y_pred, y_score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1],
                                                 zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "avg_precision": float(average_precision_score(y_true, y_score)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_class": {
            "benign": {"precision": float(p[0]), "recall": float(r[0]),
                       "f1": float(f[0]), "support": int(s[0])},
            "attack": {"precision": float(p[1]), "recall": float(r[1]),
                       "f1": float(f[1]), "support": int(s[1])},
        },
    }


def run_split(X, y, kind: str, seed: int = 42) -> dict:
    n = len(y)
    if kind == "stratified":
        idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.30,
                                          stratify=y, random_state=seed)
    elif kind == "temporal":
        cut = int(round(n * 0.70))
        idx_tr, idx_te = np.arange(cut), np.arange(cut, n)
    else:
        raise ValueError(kind)

    Xtr, Xte = X[idx_tr], X[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]

    info = {
        "split": kind, "seed": seed,
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "train_benign": int((ytr == 0).sum()), "train_attack": int((ytr == 1).sum()),
        "test_benign": int((yte == 0).sum()), "test_attack": int((yte == 1).sum()),
        "test_benign_pct": round(100 * float((yte == 0).mean()), 4),
    }
    print(f"  [{kind}] train={info['n_train']:,} test={info['n_test']:,} "
          f"test_benign={info['test_benign']:,} test_attack={info['test_attack']:,}",
          flush=True)

    if info["test_benign"] == 0 or info["train_benign"] == 0:
        info["error"] = "a partition contains no benign flows; metrics undefined"
        print(f"  [{kind}] ABORT: {info['error']}", flush=True)
        return info

    t0 = time.time()
    clf = RandomForestClassifier(**RF_KW).fit(Xtr, ytr)
    info["train_time_s"] = round(time.time() - t0, 2)

    t0 = time.time()
    proba = clf.predict_proba(Xte)[:, 1]
    info["inference_time_s"] = round(time.time() - t0, 3)
    info["inference_latency_ms_per_flow"] = round(1000 * (time.time() - t0) / len(yte), 6)

    pred = (proba >= TAU).astype(np.int8)
    info["metrics_at_tau"] = metrics(yte, pred, proba)
    info["metrics_argmax"] = metrics(yte, clf.predict(Xte), proba)
    info["tau"] = TAU
    m = info["metrics_at_tau"]
    print(f"  [{kind}] acc={m['accuracy']:.6f} macroF1={m['macro_f1']:.6f} "
          f"AUC={m['auc_roc']:.6f} FPR={m['fpr']:.6f} FNR={m['fnr']:.8f}", flush=True)
    return info


def main() -> int:
    log_event("E0", "start")
    t_all = time.time()
    out: dict = {"experiment": "E0", "objective": "reproduce submitted headline numbers",
                 "targets": TARGETS}

    src = DATA_ROOT / "CICDDoS2019" / "03-11" / "Syn.csv"
    print(f"Loading {src}", flush=True)
    out["source_file"] = {"path": str(src), "bytes": src.stat().st_size}

    df = D.load_partition(src)
    df, clean_stats = D.clean_partition(df)
    out["cleaning"] = clean_stats

    print("Sorting temporally...", flush=True)
    df = D.sort_temporal(df)

    y = D.to_binary_label(df["Label"])
    out["cleaned_totals"] = {
        "n": int(len(y)), "benign": int((y == 0).sum()), "attack": int((y == 1).sum()),
        "benign_pct": round(100 * float((y == 0).mean()), 5),
    }
    print(f"  cleaned total={len(y):,}  benign={(y==0).sum():,}  attack={(y==1).sum():,}",
          flush=True)
    out["cleaned_total_matches_paper"] = int(len(y)) == TARGETS["n_train"] + TARGETS["n_test"]

    print("Building 88-dim matrix (ORIGINAL entropy set, with the H_ttl defect)...",
          flush=True)
    X, names = D.build_matrix(df, repaired=False)
    out["feature_names"] = names
    out["n_features"] = int(X.shape[1])

    # --- H_ttl degeneracy evidence for E16 -------------------------------
    ent_start = len(names) - 8
    ent_stats = {}
    for j in range(ent_start, len(names)):
        colv = X[:, j]
        ent_stats[names[j]] = {
            "min": float(colv.min()), "max": float(colv.max()),
            "mean": float(colv.mean()), "std": float(colv.std()),
            "n_unique": int(np.unique(colv).size),
        }
    out["entropy_feature_stats"] = ent_stats
    out["H_ttl_is_constant_zero"] = bool(
        ent_stats["H_ttl"]["max"] == 0.0 and ent_stats["H_ttl"]["min"] == 0.0)
    print(f"  H_ttl constant-zero: {out['H_ttl_is_constant_zero']}  "
          f"(std={ent_stats['H_ttl']['std']})", flush=True)

    np.save(CACHE / "syn0311_X_original.npy", X)
    np.save(CACHE / "syn0311_y.npy", y)
    df[["Source_IP", "Destination_IP", "Timestamp", "Label"]].to_parquet(
        CACHE / "syn0311_meta.parquet", index=False)
    (CACHE / "syn0311_feature_names.json").write_text(json.dumps(names, indent=2))
    print(f"  cached to {CACHE}", flush=True)

    out["runs"] = {}
    for kind in ("stratified", "temporal"):
        print(f"Running {kind} split...", flush=True)
        out["runs"][kind] = run_split(X, y, kind)

    # --- which protocol reproduces the submitted numbers? -------------------
    verdict = {}
    for kind, r in out["runs"].items():
        if "metrics_at_tau" not in r:
            verdict[kind] = {"reproduces_counts": False, "reason": r.get("error")}
            continue
        counts_ok = (r["n_train"] == TARGETS["n_train"] and
                     r["n_test"] == TARGETS["n_test"] and
                     r["test_benign"] == TARGETS["test_benign"] and
                     r["test_attack"] == TARGETS["test_attack"])
        m = r["metrics_at_tau"]
        verdict[kind] = {
            "reproduces_counts": bool(counts_ok),
            "acc_delta_pp": round(100 * (m["accuracy"] - TARGETS["accuracy"]), 6),
            "macro_f1_delta_pp": round(100 * (m["macro_f1"] - TARGETS["macro_f1"]), 6),
            "fpr_delta_pp": round(100 * (m["fpr"] - TARGETS["fpr"]), 6),
        }
    out["verdict"] = verdict
    out["total_runtime_s"] = round(time.time() - t_all, 1)

    p = save_result("E0_baseline_reproduction", out)
    log_event("E0", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
