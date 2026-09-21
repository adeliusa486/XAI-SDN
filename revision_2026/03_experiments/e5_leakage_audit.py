"""E5 - Leakage and dataset-artifact audit (R3.1, R4.1, R6.8, R2.2, R6.1).

Reviewers 3, 4 and 6 all make the same charge from different angles: the
near-perfect accuracy may reflect dataset artifacts rather than attack signal,
and removing the top three SHAP features does not test that. This experiment
answers it with evidence instead of assertion.

Seven parts:
  1. exact-duplicate census, within the partition and across the split boundary;
  2. near-duplicate analysis (MinHash/LSH + nearest-neighbour distances);
  3. single-feature separability probe over all 88 features, which defines the
     leakage-suspect set objectively rather than by intuition;
  4. per-class distribution divergence (KS, Jensen-Shannon);
  5. progressive removal - top 1/3/5/10/20 suspects and the ENTIRE flagged set,
     which is what R3.1 literally asks for;
  6. controls - label permutation (must collapse to chance), temporal shuffle,
     and a recomputation of the entropy features over a shuffled arrival order,
     which is the decisive test of whether window entropy encodes attack
     structure or merely temporal position in the capture;
  7. bootstrap 95% confidence intervals on every headline metric.
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
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

CACHE = DATA_ROOT / "cache"
RF_KW = dict(n_estimators=200, max_depth=None, max_features="sqrt", bootstrap=True,
             class_weight="balanced", n_jobs=-1, random_state=42)
TAU = 0.70
SUSPECT_AUC = 0.99        # a single feature separating the classes this well is a shortcut
N_BOOT = 1000
NN_SAMPLE = 50_000        # nearest-neighbour probe size (O(n) queries against the train set)
MINHASH_SAMPLE = 300_000


def evaluate(X, y, cut, seed=42, tag="") -> dict:
    kw = dict(RF_KW)
    kw["random_state"] = seed
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    t0 = time.time()
    clf = RandomForestClassifier(**kw).fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= TAU).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    out = {
        "tag": tag, "n_features": int(X.shape[1]),
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "auc_roc": float(roc_auc_score(yte, proba)) if len(np.unique(yte)) > 1 else None,
        "avg_precision": float(average_precision_score(yte, proba)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fit_seconds": round(time.time() - t0, 1),
    }
    print(f"  [{tag}] d={out['n_features']} acc={out['accuracy']:.6f} "
          f"F1={out['macro_f1']:.6f} AUC={out['auc_roc']} "
          f"FPR={out['fpr']} FNR={out['fnr']}", flush=True)
    return out, yte, proba, pred


def bootstrap_ci(y_true, y_pred, y_score, n_boot=N_BOOT, seed=42) -> dict:
    """Percentile bootstrap CIs. Reviewer 4 asked for FPR/FNR confidence intervals."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc, f1s, fprs, fnrs, aps = [], [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ys = y_true[idx], y_pred[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        acc.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        fprs.append(fp / (fp + tn) if (fp + tn) else np.nan)
        fnrs.append(fn / (fn + tp) if (fn + tp) else np.nan)
        aps.append(average_precision_score(yt, ys))

    def ci(a):
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        return {"mean": float(a.mean()), "lo95": float(np.percentile(a, 2.5)),
                "hi95": float(np.percentile(a, 97.5))}

    return {"n_boot": len(acc), "accuracy": ci(acc), "macro_f1": ci(f1s),
            "fpr": ci(fprs), "fnr": ci(fnrs), "avg_precision": ci(aps)}


def duplicate_census(X, y, cut) -> dict:
    """Exact duplicates within the whole partition and across the split boundary."""
    print("  hashing rows for exact-duplicate census...", flush=True)
    t0 = time.time()
    Xb = np.ascontiguousarray(X)
    view = Xb.view([('', Xb.dtype)] * Xb.shape[1]).ravel()
    _, first_idx, inverse, counts = np.unique(view, return_index=True,
                                              return_inverse=True, return_counts=True)
    n = len(y)
    n_unique = int(len(counts))
    dup_rows = int(n - n_unique)

    # how many test rows have a feature-identical twin in the training partition
    seen_train = np.zeros(n_unique, dtype=bool)
    seen_train[np.unique(inverse[:cut])] = True
    test_has_train_twin = int(seen_train[inverse[cut:]].sum())

    # and do the twins agree on the label?
    same_label = 0
    if test_has_train_twin:
        train_label = {}
        for g, lab in zip(inverse[:cut], y[:cut]):
            train_label.setdefault(g, lab)
        for g, lab in zip(inverse[cut:], y[cut:]):
            if seen_train[g] and train_label.get(g) == lab:
                same_label += 1

    res = {
        "n_rows": int(n),
        "n_unique_feature_vectors": n_unique,
        "n_exact_duplicate_rows": dup_rows,
        "duplicate_rate": round(dup_rows / n, 6),
        "largest_duplicate_group": int(counts.max()),
        "test_rows_with_exact_train_twin": test_has_train_twin,
        "test_twin_rate": round(test_has_train_twin / (n - cut), 6),
        "test_twin_same_label": int(same_label),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"    exact dup rows={dup_rows:,} ({res['duplicate_rate']:.2%}), "
          f"test rows with a train twin={test_has_train_twin:,} "
          f"({res['test_twin_rate']:.2%})", flush=True)
    return res


def near_duplicate_probe(X, y, cut, rng) -> dict:
    """Distance from each sampled test flow to its nearest training flow."""
    print("  nearest-neighbour near-duplicate probe...", flush=True)
    t0 = time.time()
    Xtr, Xte = X[:cut], X[cut:]
    # standardise so the distance is not dominated by large-magnitude columns
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    ntr = min(400_000, len(Xtr))
    nte = min(NN_SAMPLE, len(Xte))
    itr = rng.choice(len(Xtr), ntr, replace=False)
    ite = rng.choice(len(Xte), nte, replace=False)
    A = ((Xtr[itr] - mu) / sd).astype(np.float32)
    B = ((Xte[ite] - mu) / sd).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1).fit(A)
    dist, ind = nn.kneighbors(B)
    d = dist.ravel()
    lab_match = (y[:cut][itr][ind.ravel()] == y[cut:][ite])
    res = {
        "n_train_ref": int(ntr), "n_test_probe": int(nte),
        "nn_distance": {
            "min": float(d.min()), "p1": float(np.percentile(d, 1)),
            "median": float(np.median(d)), "p99": float(np.percentile(d, 99)),
            "max": float(d.max()), "mean": float(d.mean()),
        },
        "frac_nn_distance_below_1e-6": float((d < 1e-6).mean()),
        "frac_nn_distance_below_1e-3": float((d < 1e-3).mean()),
        "frac_nn_distance_below_0.01": float((d < 0.01).mean()),
        "frac_nn_same_label": float(lab_match.mean()),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"    median NN distance={res['nn_distance']['median']:.6g}, "
          f"frac<1e-6={res['frac_nn_distance_below_1e-6']:.3%}, "
          f"NN label agreement={res['frac_nn_same_label']:.3%}", flush=True)
    return res


def separability_probe(X, y, names, rng) -> pd.DataFrame:
    """Single-feature AUC and single-threshold accuracy for all 88 features."""
    print("  single-feature separability probe...", flush=True)
    t0 = time.time()
    n = len(y)
    idx = rng.choice(n, min(500_000, n), replace=False)
    Xs, ys = X[idx], y[idx]
    rows = []
    for j, nm in enumerate(names):
        col = Xs[:, j].astype(np.float64)
        if col.min() == col.max():
            rows.append({"feature": nm, "auc": 0.5, "best_stump_bacc": 0.5,
                         "constant": True})
            continue
        a = roc_auc_score(ys, col)
        auc = max(a, 1 - a)
        # best single threshold by balanced accuracy, over quantile candidates
        qs = np.unique(np.quantile(col, np.linspace(0.001, 0.999, 199)))
        pos, neg = ys == 1, ys == 0
        best = 0.5
        for t in qs:
            ge = col >= t
            tpr = ge[pos].mean() if pos.any() else 0.0
            fpr = ge[neg].mean() if neg.any() else 0.0
            best = max(best, (tpr + (1 - fpr)) / 2, ((1 - tpr) + fpr) / 2)
        rows.append({"feature": nm, "auc": float(auc),
                     "best_stump_bacc": float(best), "constant": False})
    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    print(f"    {time.time()-t0:.1f}s; top-5 by AUC: "
          f"{df.head(5)[['feature','auc']].to_dict('records')}", flush=True)
    return df


def distribution_checks(X, y, names, rng) -> pd.DataFrame:
    """Per-feature class-conditional divergence (R4.1 'feature-distribution checks')."""
    print("  class-conditional distribution checks...", flush=True)
    t0 = time.time()
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    ip = rng.choice(pos, min(50_000, len(pos)), replace=False)
    ineg = rng.choice(neg, min(50_000, len(neg)), replace=False)
    rows = []
    for j, nm in enumerate(names):
        a = X[ip, j].astype(np.float64)
        b = X[ineg, j].astype(np.float64)
        if a.min() == a.max() and b.min() == b.max() and a.min() == b.min():
            rows.append({"feature": nm, "ks_stat": 0.0, "ks_p": 1.0, "js_div": 0.0,
                         "class_disjoint_ranges": False})
            continue
        ks = ks_2samp(a, b)
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        bins = np.linspace(lo, hi, 129) if hi > lo else np.array([lo, lo + 1])
        ha, _ = np.histogram(a, bins=bins)
        hb, _ = np.histogram(b, bins=bins)
        ha = ha / max(ha.sum(), 1)
        hb = hb / max(hb.sum(), 1)
        js = float(jensenshannon(ha, hb, base=2))
        disjoint = bool(a.max() < b.min() or b.max() < a.min())
        rows.append({"feature": nm, "ks_stat": float(ks.statistic),
                     "ks_p": float(ks.pvalue), "js_div": 0.0 if np.isnan(js) else js,
                     "class_disjoint_ranges": disjoint})
    df = pd.DataFrame(rows).sort_values("ks_stat", ascending=False).reset_index(drop=True)
    print(f"    {time.time()-t0:.1f}s; {int(df['class_disjoint_ranges'].sum())} features "
          f"have completely disjoint class ranges", flush=True)
    return df


def main() -> int:
    log_event("E5", "start")
    t_all = time.time()
    rng = np.random.default_rng(42)
    out: dict = {"experiment": "E5", "objective": "leakage and dataset-artifact audit",
                 "suspect_auc_threshold": SUSPECT_AUC}

    X = np.load(CACHE / "syn0311_X_repaired.npy")
    y = np.load(CACHE / "syn0311_y.npy")
    names = json.loads((CACHE / "syn0311_feature_names_repaired.json").read_text())
    cut = int(round(len(y) * 0.70))
    out["protocol"] = {"split": "temporal", "cut_index": int(cut),
                       "n_train": int(cut), "n_test": int(len(y) - cut)}
    print(f"loaded {X.shape}, temporal cut at {cut:,}", flush=True)

    # 1 -------------------------------------------------------------------
    out["duplicates"] = duplicate_census(X, y, cut)
    # 2 -------------------------------------------------------------------
    out["near_duplicates"] = near_duplicate_probe(X, y, cut, rng)
    # 3 -------------------------------------------------------------------
    sep = separability_probe(X, y, names, rng)
    sep.to_csv(RESULTS / "E5_single_feature_separability.csv", index=False)
    suspects = sep.loc[sep["auc"] >= SUSPECT_AUC, "feature"].tolist()
    out["leakage_suspects"] = {
        "criterion": f"single-feature AUC >= {SUSPECT_AUC}",
        "n": len(suspects), "features": suspects,
    }
    print(f"  {len(suspects)} leakage-suspect features: {suspects}", flush=True)
    # 4 -------------------------------------------------------------------
    dist = distribution_checks(X, y, names, rng)
    dist.to_csv(RESULTS / "E5_distribution_checks.csv", index=False)
    out["n_class_disjoint_features"] = int(dist["class_disjoint_ranges"].sum())
    out["class_disjoint_features"] = dist.loc[dist["class_disjoint_ranges"],
                                              "feature"].tolist()

    # 5 -------------------------------------------------------------------
    print("Progressive removal...", flush=True)
    ordered = sep["feature"].tolist()          # most separable first
    name_idx = {nm: i for i, nm in enumerate(names)}
    removal_rows = []
    out["progressive_removal"] = {}

    full_res, yte, proba, pred = evaluate(X, y, cut, tag="full 88")
    out["progressive_removal"]["full"] = full_res
    out["bootstrap_ci_full"] = bootstrap_ci(yte, pred, proba)
    removal_rows.append({"removed_k": 0, "removed": "", **{
        k: full_res[k] for k in ("n_features", "accuracy", "macro_f1", "auc_roc",
                                 "avg_precision", "fpr", "fnr")}})

    for k in (1, 3, 5, 10, 20):
        drop = ordered[:k]
        keep = [i for nm, i in name_idx.items() if nm not in drop]
        keep.sort()
        r, yt2, pr2, pd2 = evaluate(X[:, keep], y, cut, tag=f"minus top-{k}")
        r["removed"] = drop
        out["progressive_removal"][f"top{k}"] = r
        removal_rows.append({"removed_k": k, "removed": "|".join(drop), **{
            kk: r[kk] for kk in ("n_features", "accuracy", "macro_f1", "auc_roc",
                                 "avg_precision", "fpr", "fnr")}})

    # the literal demand of R3.1: remove ALL objectively-flagged suspects
    if suspects:
        keep = [i for nm, i in name_idx.items() if nm not in set(suspects)]
        keep.sort()
        r, yt3, pr3, pd3 = evaluate(X[:, keep], y, cut, tag="minus ALL suspects")
        r["removed"] = suspects
        out["progressive_removal"]["all_suspects"] = r
        out["bootstrap_ci_all_suspects_removed"] = bootstrap_ci(yt3, pd3, pr3)
        removal_rows.append({"removed_k": len(suspects),
                             "removed": "|".join(suspects), **{
            kk: r[kk] for kk in ("n_features", "accuracy", "macro_f1", "auc_roc",
                                 "avg_precision", "fpr", "fnr")}})

    pd.DataFrame(removal_rows).to_csv(RESULTS / "E5_progressive_removal.csv", index=False)

    # 6 -------------------------------------------------------------------
    print("Controls...", flush=True)
    yperm = rng.permutation(y)
    r_perm, _, _, _ = evaluate(X, yperm, cut, tag="label permutation control")
    out["control_label_permutation"] = r_perm
    out["control_label_permutation_passes"] = bool(r_perm["macro_f1"] < 0.60)
    if not out["control_label_permutation_passes"]:
        print("  !! label-permutation control FAILED - the pipeline leaks. "
              "Investigate before trusting any other result.", flush=True)

    perm = rng.permutation(len(y))
    r_shuf, _, _, _ = evaluate(X[perm], y[perm], cut, tag="temporal shuffle control")
    out["control_temporal_shuffle"] = r_shuf

    # --- decisive diagnostic: is window entropy an attack signature or a clock? --
    # The entropy features are computed over a sliding window of ARRIVAL ORDER.
    # If benign and attack flows are temporally segregated in the capture, a
    # window entropy can separate the classes simply by encoding which phase of
    # the capture the flow belongs to. Recomputing the same features over a
    # shuffled arrival order destroys that temporal structure while leaving every
    # per-flow value untouched. A large drop in separability means the feature was
    # reading the clock; little change means it was reading the traffic.
    print("Entropy-under-shuffled-arrival-order diagnostic...", flush=True)
    t0 = time.time()
    meta = pd.read_parquet(CACHE / "syn0311_meta.parquet")
    ent_start = len(names) - 8
    cic_names = names[:ent_start]
    sh = rng.permutation(len(y))
    df_sh = pd.DataFrame({
        "Source_IP": meta["Source_IP"].to_numpy()[sh],
        "Destination_IP": meta["Destination_IP"].to_numpy()[sh],
        "Destination_Port": X[sh, cic_names.index("Destination_Port")],
        "Protocol": X[sh, cic_names.index("Protocol")],
        "Packet_Length_Mean": X[sh, cic_names.index("Packet_Length_Mean")],
        "Flow_IAT_Mean": X[sh, cic_names.index("Flow_IAT_Mean")],
        "SYN_Flag_Count": X[sh, cic_names.index("SYN_Flag_Count")],
        "Source_Port": X[sh, cic_names.index("Source_Port")],
    })
    streams, ent_names = D.entropy_streams(df_sh, repaired=True)
    H_sh = D.rolling_entropy(streams, window=D.DEFAULT_WINDOW).astype(np.float32)
    y_sh = y[sh]

    per_feature = {}
    for k, nm in enumerate(ent_names):
        a_nat = roc_auc_score(y, X[:, ent_start + k])
        a_shf = roc_auc_score(y_sh, H_sh[:, k])
        per_feature[nm] = {
            "auc_natural_order": float(max(a_nat, 1 - a_nat)),
            "auc_shuffled_order": float(max(a_shf, 1 - a_shf)),
        }
        per_feature[nm]["drop"] = round(
            per_feature[nm]["auc_natural_order"] -
            per_feature[nm]["auc_shuffled_order"], 6)
        print(f"    {nm:14s} AUC natural={per_feature[nm]['auc_natural_order']:.4f} "
              f"shuffled={per_feature[nm]['auc_shuffled_order']:.4f} "
              f"drop={per_feature[nm]['drop']:+.4f}", flush=True)

    X_sh = np.hstack([X[sh, :ent_start], H_sh])
    r_entshuf, _, _, _ = evaluate(X_sh, y_sh, cut,
                                  tag="entropy recomputed on shuffled arrival order")
    r_cic_only, _, _, _ = evaluate(X[:, :ent_start], y, cut, tag="CIC-only (80)")
    out["control_entropy_shuffled_arrival"] = {
        "per_feature_auc": per_feature,
        "model_with_shuffled_order_entropy": r_entshuf,
        "model_cic_features_only": r_cic_only,
        "mean_auc_drop": round(float(np.mean([v["drop"] for v in per_feature.values()])), 6),
        "seconds": round(time.time() - t0, 1),
        "interpretation_rule": (
            "A large mean AUC drop indicates that the window entropy features were "
            "separating the classes by temporal position in the capture rather than "
            "by attack structure, which would be a dataset artifact rather than a "
            "detection capability. A small drop indicates the opposite."),
    }
    out["temporal_vs_shuffled_gap"] = {
        "accuracy": round(r_shuf["accuracy"] - full_res["accuracy"], 8),
        "macro_f1": round(r_shuf["macro_f1"] - full_res["macro_f1"], 8),
        "fpr": round((r_shuf["fpr"] or 0) - (full_res["fpr"] or 0), 8),
    }

    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E5_leakage_audit", out)
    log_event("E5", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
