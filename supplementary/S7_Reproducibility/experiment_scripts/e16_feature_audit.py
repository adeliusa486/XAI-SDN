"""E16 - Feature-set correctness audit and repair.

Establishes, on the real data rather than by inspection:

  1. that H_ttl is a constant-zero column, because the pipeline hard-codes
     ttl = 64 for every flow record;
  2. how much information each of the 88 features actually carries
     (variance, distinct values, single-feature AUC, mutual information);
  3. that source-port entropy is a well-posed replacement;
  4. what the repair does to the headline metrics.

Also emits the complete feature specification table the manuscript needs:
every feature, its exact source column, its transformation and its bin width.
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

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

CACHE = DATA_ROOT / "cache"
RF_KW = dict(n_estimators=200, max_depth=None, max_features="sqrt", bootstrap=True,
             class_weight="balanced", n_jobs=-1, random_state=42)
TAU = 0.70
MI_SAMPLE = 200_000          # mutual information is O(n log n) per feature
AUC_SAMPLE = 500_000


def single_feature_auc(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """AUC of each feature used alone as a score (and of its negation)."""
    n = len(y)
    if n > AUC_SAMPLE:
        idx = rng.choice(n, AUC_SAMPLE, replace=False)
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y
    out = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        col = Xs[:, j]
        if np.all(col == col[0]):
            out[j] = 0.5
            continue
        try:
            a = roc_auc_score(ys, col)
            out[j] = max(a, 1.0 - a)       # direction-agnostic separability
        except Exception:
            out[j] = np.nan
    return out


def describe(X: np.ndarray, names: list[str], y: np.ndarray,
             rng: np.random.Generator) -> pd.DataFrame:
    n = len(y)
    mi_idx = rng.choice(n, min(MI_SAMPLE, n), replace=False)
    print(f"  mutual information on {len(mi_idx):,} rows...", flush=True)
    t0 = time.time()
    mi = mutual_info_classif(X[mi_idx], y[mi_idx], discrete_features=False,
                             random_state=42, n_neighbors=3)
    print(f"    {time.time()-t0:.1f}s", flush=True)
    print("  single-feature AUC...", flush=True)
    t0 = time.time()
    auc = single_feature_auc(X, y, rng)
    print(f"    {time.time()-t0:.1f}s", flush=True)

    rows = []
    for j, nm in enumerate(names):
        col = X[:, j]
        rows.append({
            "feature": nm,
            "family": "entropy" if nm.startswith("H_") else "cicflowmeter",
            "min": float(col.min()), "max": float(col.max()),
            "mean": float(col.mean()), "std": float(col.std()),
            "n_unique": int(np.unique(col).size),
            "is_constant": bool(col.min() == col.max()),
            "mutual_info": float(mi[j]),
            "single_feature_auc": float(auc[j]),
        })
    return pd.DataFrame(rows)


def fit_eval(X, y, cut, tag):
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    t0 = time.time()
    clf = RandomForestClassifier(**RF_KW).fit(Xtr, ytr)
    tt = time.time() - t0
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= TAU).astype(np.int8)
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 average_precision_score)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    res = {
        "tag": tag,
        "n_features": int(X.shape[1]),
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "auc_roc": float(roc_auc_score(yte, proba)),
        "avg_precision": float(average_precision_score(yte, proba)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "train_time_s": round(tt, 2),
    }
    print(f"  [{tag}] acc={res['accuracy']:.6f} macroF1={res['macro_f1']:.6f} "
          f"AUC={res['auc_roc']:.6f} AP={res['avg_precision']:.6f} "
          f"FPR={res['fpr']} FNR={res['fnr']}", flush=True)
    return res, clf


FEATURE_SPEC = [
    # name, source column(s), transformation, bin width, units
    ("H_src_ip", "Source IP", "Shannon entropy over the sliding window of source addresses",
     "none (categorical)", "bits"),
    ("H_dst_ip", "Destination IP", "Shannon entropy over destination addresses",
     "none (categorical)", "bits"),
    ("H_dst_port", "Destination Port", "Shannon entropy over destination port numbers",
     "none (integer)", "bits"),
    ("H_proto", "Protocol", "Shannon entropy over IP protocol numbers",
     "none (integer)", "bits"),
    ("H_pkt_len", "Packet Length Mean", "floor-binned, then Shannon entropy",
     f"{D.BIN_PKT_LEN} bytes", "bits"),
    ("H_iat", "Flow IAT Mean", "floor-binned, then Shannon entropy",
     f"{D.BIN_IAT} microseconds", "bits"),
    ("H_tcp_flags", "SYN Flag Count", "Shannon entropy over SYN flag counts",
     "none (integer)", "bits"),
    ("H_ttl [SUBMITTED, DEFECTIVE]", "none - hard-coded constant 64",
     "Shannon entropy of a constant, therefore identically zero",
     f"{D.BIN_TTL} TTL units (inoperative)", "bits"),
    ("H_src_port [REPAIRED]", "Source Port",
     "Shannon entropy over source port numbers", "none (integer)", "bits"),
]


def main() -> int:
    log_event("E16", "start")
    t_all = time.time()
    rng = np.random.default_rng(42)
    out: dict = {"experiment": "E16", "objective": "feature-set correctness audit and repair"}

    Xo = np.load(CACHE / "syn0311_X_original.npy")
    y = np.load(CACHE / "syn0311_y.npy")
    names_o = json.loads((CACHE / "syn0311_feature_names.json").read_text())
    print(f"loaded cached original matrix {Xo.shape}", flush=True)

    # ---- 1. the H_ttl defect, measured --------------------------------------
    j_ttl = names_o.index("H_ttl")
    ttl_col = Xo[:, j_ttl]
    out["h_ttl_defect"] = {
        "column_index": int(j_ttl),
        "min": float(ttl_col.min()), "max": float(ttl_col.max()),
        "std": float(ttl_col.std()), "n_unique": int(np.unique(ttl_col).size),
        "is_identically_zero": bool(np.all(ttl_col == 0.0)),
        "is_zero_within_float_tolerance": bool(np.abs(ttl_col).max() < 1e-12),
        "max_abs_value": float(np.abs(ttl_col).max()),
        "root_cause": "features/pipeline.py hard-codes 'ttl': 64 for every flow record; "
                      "CICFlowMeter CSV output contains no TTL column.",
    }
    print(f"H_ttl identically zero: {out['h_ttl_defect']['is_identically_zero']}", flush=True)

    # ---- 2. per-feature information content ---------------------------------
    print("Describing original 88-feature set...", flush=True)
    desc_o = describe(Xo, names_o, y, rng)
    desc_o.insert(0, "feature_set", "original")
    out["n_constant_features_original"] = int(desc_o["is_constant"].sum())
    out["constant_features_original"] = desc_o.loc[desc_o["is_constant"],
                                                   "feature"].tolist()
    print(f"  constant features: {out['constant_features_original']}", flush=True)

    # ---- 3. build the repaired matrix ---------------------------------------
    print("Rebuilding with source-port entropy in place of TTL entropy...", flush=True)
    meta = pd.read_parquet(CACHE / "syn0311_meta.parquet")
    # Source_Port lives in the CIC block; recompute only the 8th entropy stream.
    j_sport = names_o.index("Source_Port")
    sport = Xo[:, j_sport].astype(np.int64)
    t0 = time.time()
    H_sport = D.rolling_entropy([list(sport)], window=D.DEFAULT_WINDOW)[:, 0]
    print(f"  H_src_port computed in {time.time()-t0:.1f}s", flush=True)

    Xr = Xo.copy()
    Xr[:, j_ttl] = H_sport.astype(np.float32)
    names_r = list(names_o)
    names_r[j_ttl] = "H_src_port"

    out["h_src_port"] = {
        "min": float(H_sport.min()), "max": float(H_sport.max()),
        "mean": float(H_sport.mean()), "std": float(H_sport.std()),
        "n_unique": int(np.unique(H_sport.astype(np.float32)).size),
        "is_constant": bool(H_sport.min() == H_sport.max()),
    }
    print(f"  H_src_port std={out['h_src_port']['std']:.4f} "
          f"unique={out['h_src_port']['n_unique']:,}", flush=True)

    print("Describing repaired 88-feature set...", flush=True)
    desc_r = describe(Xr, names_r, y, rng)
    desc_r.insert(0, "feature_set", "repaired")

    pd.concat([desc_o, desc_r]).to_csv(RESULTS / "E16_feature_statistics.csv", index=False)

    # ---- 4. effect of the repair on the headline metrics --------------------
    cut = int(round(len(y) * 0.70))
    out["protocol"] = {"split": "temporal", "train_frac": 0.70, "cut_index": int(cut),
                       "note": "temporal split adopted as this work's primary protocol"}
    print("Re-baselining (temporal split)...", flush=True)
    res_o, _ = fit_eval(Xo, y, cut, "original (H_ttl constant)")
    res_r, clf_r = fit_eval(Xr, y, cut, "repaired (H_src_port)")
    out["baseline_original"] = res_o
    out["baseline_repaired"] = res_r
    out["repair_delta"] = {
        k: round(res_r[k] - res_o[k], 8)
        for k in ("accuracy", "macro_f1", "auc_roc", "avg_precision")
    }

    # importance of the replaced slot, before and after
    out["impurity_importance_of_slot"] = {
        "H_src_port": float(clf_r.feature_importances_[j_ttl]),
    }

    np.save(CACHE / "syn0311_X_repaired.npy", Xr)
    (CACHE / "syn0311_feature_names_repaired.json").write_text(json.dumps(names_r, indent=2))

    # ---- 5. publishable feature specification -------------------------------
    pd.DataFrame(FEATURE_SPEC, columns=["feature", "source_column", "transformation",
                                        "bin_width", "units"]
                 ).to_csv(RESULTS / "E16_feature_specification.csv", index=False)

    out["decision"] = (
        "Adopt the repaired feature set. H_ttl carries exactly zero information; "
        "H_src_port is well-posed, non-constant and available in every CICFlowMeter "
        "export. The 88-dimensional representation is preserved."
        if out["h_ttl_defect"]["is_zero_within_float_tolerance"]
        and not out["h_src_port"]["is_constant"]
        else "Repair NOT adopted - preconditions failed; investigate before proceeding."
    )
    out["total_runtime_s"] = round(time.time() - t_all, 1)

    p = save_result("E16_feature_audit", out)
    log_event("E16", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    print(out["decision"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
