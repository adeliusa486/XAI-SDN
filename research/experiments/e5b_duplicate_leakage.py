"""E5b - Duplicate-driven leakage across the split boundary.

This experiment provides a "duplicate/near-duplicate flow analysis". Running that on
the cleaned matrix answers nothing, because the pipeline removes duplicates
before modelling and the count is zero by construction. The question that
actually matters is what those duplicates would have done had they been kept,
and specifically whether a duplicate group straddles the train/test boundary -
because a duplicate on both sides of a split is a test answer handed to the
model during training.

This experiment therefore works on the RAW partition, before deduplication, and
measures the straddle rate under both candidate protocols:

  * the stratified random split that actually produced the submitted numbers, and
  * the temporal split the manuscript described and this work adopts.

It then quantifies the accuracy a trivial nearest-duplicate lookup would achieve
under each protocol. That number is the cleanest available statement of how much
of the reported performance a memorising baseline could have obtained without
learning anything about attacks at all.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

import data as D  # noqa: E402
from paths import DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402

from sklearn.model_selection import train_test_split  # noqa: E402

SEED = 42


def group_ids(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map identical feature rows to a shared group id."""
    Xb = np.ascontiguousarray(X)
    view = Xb.view([("", Xb.dtype)] * Xb.shape[1]).ravel()
    _, inverse, counts = np.unique(view, return_inverse=True, return_counts=True)
    return inverse, counts


def straddle_report(groups: np.ndarray, y: np.ndarray, tr: np.ndarray,
                    te: np.ndarray, label: str) -> dict:
    n_groups = int(groups.max()) + 1
    in_train = np.zeros(n_groups, bool)
    in_train[np.unique(groups[tr])] = True

    g_te = groups[te]
    memorisable = in_train[g_te]

    # what would a pure lookup predict? the majority training label of the group
    lut_sum = np.bincount(groups[tr], weights=y[tr], minlength=n_groups)
    lut_cnt = np.bincount(groups[tr], minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        lut = (lut_sum / np.maximum(lut_cnt, 1) >= 0.5).astype(np.int8)

    pred = np.where(memorisable, lut[g_te], 1)   # unseen -> majority class
    yte = y[te]
    acc = float((pred == yte).mean())
    acc_on_seen = float((pred[memorisable] == yte[memorisable]).mean()) \
        if memorisable.any() else None

    res = {
        "split": label,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "test_rows_with_identical_training_twin": int(memorisable.sum()),
        "straddle_rate": round(float(memorisable.mean()), 6),
        "lookup_baseline_accuracy": round(acc, 6),
        "lookup_accuracy_on_straddling_rows": round(acc_on_seen, 6)
        if acc_on_seen is not None else None,
        "test_benign": int((yte == 0).sum()),
        "benign_rows_with_training_twin": int(((yte == 0) & memorisable).sum()),
        "benign_straddle_rate": round(
            float(((yte == 0) & memorisable).sum() / max((yte == 0).sum(), 1)), 6),
    }
    print(f"  [{label}] straddle={res['straddle_rate']:.4%} "
          f"(benign {res['benign_straddle_rate']:.4%}), "
          f"lookup-only accuracy={res['lookup_baseline_accuracy']:.6f}", flush=True)
    return res


def main() -> int:
    log_event("E5b", "start")
    t_all = time.time()
    out: dict = {"experiment": "E5b",
                 "objective": "duplicate-driven leakage across the split boundary"}

    src = DATA_ROOT / "CICDDoS2019" / "03-11" / "Syn.csv"
    print(f"loading RAW {src.name} (no deduplication)...", flush=True)
    df = D.load_partition(src)

    feats = [c for c in D.CIC_FEATURE_NAMES if c in df.columns]
    for c in feats:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=feats).reset_index(drop=True)
    out["raw"] = {"rows_raw": int(before),
                  "rows_after_nan_inf_drop": int(len(df)),
                  "dropped_nan_inf": int(before - len(df))}

    df = D.sort_temporal(df)
    y = D.to_binary_label(df["Label"])
    X = df[feats].to_numpy(dtype=np.float32, copy=False)
    print(f"  {len(y):,} rows, {int((y==0).sum()):,} benign", flush=True)

    print("grouping identical feature rows...", flush=True)
    t0 = time.time()
    groups, counts = group_ids(X)
    n_dup_rows = int(len(y) - len(counts))
    out["duplicates_before_cleaning"] = {
        "n_rows": int(len(y)),
        "n_distinct_feature_vectors": int(len(counts)),
        "n_duplicate_rows": n_dup_rows,
        "duplicate_rate": round(n_dup_rows / len(y), 6),
        "largest_group": int(counts.max()),
        "groups_with_2_or_more": int((counts >= 2).sum()),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"  {n_dup_rows:,} duplicate rows "
          f"({out['duplicates_before_cleaning']['duplicate_rate']:.2%}), "
          f"largest group {int(counts.max()):,}", flush=True)

    # per-class duplicate behaviour
    ben = y == 0
    gb, cb = group_ids(X[ben])
    out["duplicates_benign_only"] = {
        "n_benign": int(ben.sum()),
        "n_distinct": int(len(cb)),
        "duplicate_rate": round(float((ben.sum() - len(cb)) / max(ben.sum(), 1)), 6),
        "largest_group": int(cb.max()),
    }
    print(f"  benign duplicate rate "
          f"{out['duplicates_benign_only']['duplicate_rate']:.2%}", flush=True)

    idx = np.arange(len(y))
    print("straddle analysis...", flush=True)
    reports = []
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    reports.append(straddle_report(groups, y, tr, te, "stratified random 70/30"))
    cut = int(round(len(y) * 0.70))
    reports.append(straddle_report(groups, y, idx[:cut], idx[cut:], "temporal 70/30"))
    out["straddle"] = reports

    a = reports[0]["lookup_baseline_accuracy"]
    b = reports[1]["lookup_baseline_accuracy"]
    out["interpretation"] = (
        f"Before deduplication the partition contains "
        f"{out['duplicates_before_cleaning']['duplicate_rate']:.1%} exactly duplicated "
        f"feature vectors. Under the stratified random split that produced the "
        f"submitted results, {reports[0]['straddle_rate']:.1%} of test rows have a "
        f"feature-identical twin in the training partition, and a lookup table that "
        f"learns nothing but the training labels of those twins already reaches "
        f"{a:.4f} accuracy. Under the temporal split the same lookup reaches {b:.4f}. "
        f"The difference isolates how much of the reported performance was available "
        f"to pure memorisation rather than to detection.")
    print("\n" + out["interpretation"], flush=True)

    pd.DataFrame(reports).to_csv(RESULTS / "E5b_straddle.csv", index=False)
    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E5b_duplicate_leakage", out)
    log_event("E5b", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
