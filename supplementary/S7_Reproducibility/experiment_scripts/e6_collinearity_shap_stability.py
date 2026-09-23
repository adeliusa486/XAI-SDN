"""E6 - Collinearity, redundancy and SHAP attribution stability.

An open question is what collinearity diagnostics were applied to make the SHAP
attributions "causally stable and operationally trustworthy". It has been argued that the 88-dimensional input ignores severe multicollinearity. The deployment question further
argues that the small accuracy drop after removing the top three SHAP features
merely shows the remaining features are redundant. All three are answered by
measuring redundancy directly and then measuring whether the attributions
survive it.

Parts:
  1. Pearson and Spearman correlation, hierarchical clustering on 1-|rho|;
  2. Variance Inflation Factor and design-matrix condition number;
  3. reduced feature sets by correlation clustering, retrained and compared;
  4. SHAP stability across seeds and bootstrap resamples: top-k Jaccard
     overlap and Spearman rank correlation of global mean |SHAP|.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

from paths import DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             confusion_matrix, f1_score, roc_auc_score)

CACHE = DATA_ROOT / "cache"
RF_KW = dict(n_estimators=200, max_depth=None, max_features="sqrt", bootstrap=True,
             class_weight="balanced", n_jobs=-1, random_state=42)
TAU = 0.70
CORR_SAMPLE = 400_000
VIF_SAMPLE = 100_000
SHAP_BACKGROUND = 200
SHAP_EXPLAIN = 3_000
N_SEEDS = 10
N_BOOT = 10
# The stability study needs 20 independent fits. Running them at full scale would
# cost many hours and would not change the quantity being measured, which is the
# variability of the attribution ranking. They are therefore fitted on a
# documented, temporally-coherent tail of the training partition; the size is
# reported in the result file so the compromise is visible.
STABILITY_TRAIN = 600_000
CUT_HEIGHTS = [0.05, 0.10, 0.20, 0.40]     # on 1 - |spearman|


def quick_eval(X, y, cut, tag, seed=42) -> dict:
    kw = dict(RF_KW); kw["random_state"] = seed
    t0 = time.time()
    clf = RandomForestClassifier(**kw).fit(X[:cut], y[:cut])
    fit_s = time.time() - t0
    t0 = time.time()
    proba = clf.predict_proba(X[cut:])[:, 1]
    inf_s = time.time() - t0
    pred = (proba >= TAU).astype(np.int8)
    yte = y[cut:]
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    r = {
        "tag": tag, "n_features": int(X.shape[1]),
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "auc_roc": float(roc_auc_score(yte, proba)),
        "avg_precision": float(average_precision_score(yte, proba)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "train_time_s": round(fit_s, 2),
        "inference_latency_ms_per_flow": round(1000 * inf_s / len(yte), 6),
    }
    print(f"  [{tag}] d={r['n_features']} acc={r['accuracy']:.6f} "
          f"F1={r['macro_f1']:.6f} AP={r['avg_precision']:.6f} "
          f"lat={r['inference_latency_ms_per_flow']:.5f} ms", flush=True)
    return r, clf


def global_shap(clf, Xs: np.ndarray) -> np.ndarray:
    """Mean |TreeSHAP| per feature over a fixed explain sample."""
    import shap
    ex = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    sv = ex.shap_values(Xs, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                      # (n, d, n_classes) in newer shap
        sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
    return np.abs(sv).mean(axis=0)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def main() -> int:
    log_event("E6", "start")
    t_all = time.time()
    rng = np.random.default_rng(42)
    out: dict = {"experiment": "E6",
                 "objective": "collinearity, redundancy and SHAP attribution stability"}

    X = np.load(CACHE / "syn0311_X_repaired.npy")
    y = np.load(CACHE / "syn0311_y.npy")
    names = json.loads((CACHE / "syn0311_feature_names_repaired.json").read_text())
    cut = int(round(len(y) * 0.70))
    d = X.shape[1]
    print(f"loaded {X.shape}", flush=True)

    # ---- 1. correlation structure ------------------------------------------
    print("Correlation structure...", flush=True)
    t0 = time.time()
    idx = rng.choice(len(y), min(CORR_SAMPLE, len(y)), replace=False)
    Xs = X[idx].astype(np.float64)
    const = np.array([Xs[:, j].min() == Xs[:, j].max() for j in range(d)])
    out["constant_features"] = [names[j] for j in np.flatnonzero(const)]

    P = np.corrcoef(Xs, rowvar=False)
    P = np.nan_to_num(P, nan=0.0)
    S, _ = spearmanr(Xs)
    S = np.nan_to_num(np.atleast_2d(S), nan=0.0)
    np.fill_diagonal(P, 1.0)
    np.fill_diagonal(S, 1.0)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    iu = np.triu_indices(d, k=1)
    absP = np.abs(P[iu])
    pairs = [(names[i], names[j], float(P[i, j]))
             for i, j in zip(*iu) if abs(P[i, j]) >= 0.95]
    out["correlation"] = {
        "n_pairs": int(len(absP)),
        "pairs_abs_r_ge_0.99": int((absP >= 0.99).sum()),
        "pairs_abs_r_ge_0.95": int((absP >= 0.95).sum()),
        "pairs_abs_r_ge_0.90": int((absP >= 0.90).sum()),
        "mean_abs_r": float(absP.mean()),
        "median_abs_r": float(np.median(absP)),
        "examples_r_ge_0.95": sorted(pairs, key=lambda t: -abs(t[2]))[:25],
    }
    print(f"  |r|>=0.95 pairs: {out['correlation']['pairs_abs_r_ge_0.95']} "
          f"of {out['correlation']['n_pairs']}", flush=True)
    pd.DataFrame(P, index=names, columns=names).to_csv(RESULTS / "E6_pearson.csv")

    # ---- 2. VIF and conditioning -------------------------------------------
    print("VIF and conditioning...", flush=True)
    t0 = time.time()
    vidx = rng.choice(len(y), min(VIF_SAMPLE, len(y)), replace=False)
    Xv = X[vidx].astype(np.float64)
    keep = ~np.array([Xv[:, j].min() == Xv[:, j].max() for j in range(d)])
    Xk = Xv[:, keep]
    mu, sd = Xk.mean(0), Xk.std(0)
    sd[sd == 0] = 1.0
    Z = (Xk - mu) / sd
    R = np.corrcoef(Z, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    try:
        Rinv = np.linalg.pinv(R)
        vif = np.clip(np.diag(Rinv), 0, None)
    except np.linalg.LinAlgError:
        vif = np.full(Z.shape[1], np.nan)
    kept_names = [names[j] for j in np.flatnonzero(keep)]
    vif_df = pd.DataFrame({"feature": kept_names, "vif": vif}
                          ).sort_values("vif", ascending=False)
    vif_df.to_csv(RESULTS / "E6_vif.csv", index=False)
    sv = np.linalg.svd(Z, compute_uv=False)
    out["vif"] = {
        "n_evaluated": int(len(vif)),
        "n_vif_gt_5": int((vif > 5).sum()),
        "n_vif_gt_10": int((vif > 10).sum()),
        "n_vif_gt_100": int((vif > 100).sum()),
        "median_vif": float(np.nanmedian(vif)),
        "top10": vif_df.head(10).to_dict("records"),
        "condition_number": float(sv.max() / sv[sv > 0].min()),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"  VIF>10: {out['vif']['n_vif_gt_10']}/{out['vif']['n_evaluated']}, "
          f"condition number={out['vif']['condition_number']:.3g}", flush=True)

    # ---- 3. reduced feature sets -------------------------------------------
    print("Correlation clustering and reduced feature sets...", flush=True)
    Dm = 1.0 - np.abs(S)
    np.fill_diagonal(Dm, 0.0)
    Dm = (Dm + Dm.T) / 2
    Z_link = linkage(squareform(Dm, checks=False), method="average")

    out["reduced_sets"] = {}
    full_res, full_clf = quick_eval(X, y, cut, "full 88")
    out["reduced_sets"]["full"] = full_res

    reduced_sets: dict[str, list[int]] = {"full": list(range(d))}
    for h in CUT_HEIGHTS:
        lab = fcluster(Z_link, t=h, criterion="distance")
        reps = []
        for c in np.unique(lab):
            members = np.flatnonzero(lab == c)
            # representative = highest-variance member of the cluster
            reps.append(int(members[np.argmax([Xs[:, m].std() for m in members])]))
        reps = sorted(set(reps))
        key = f"cut{h}_d{len(reps)}"
        reduced_sets[key] = reps
        r, _ = quick_eval(X[:, reps], y, cut, key)
        r["features"] = [names[i] for i in reps]
        r["cut_height"] = h
        out["reduced_sets"][key] = r

    # ---- 4. SHAP stability --------------------------------------------------
    print("SHAP attribution stability...", flush=True)
    t0 = time.time()
    ex_idx = rng.choice(np.arange(cut, len(y)), SHAP_EXPLAIN, replace=False)
    X_explain = X[ex_idx]

    s_start = max(0, cut - STABILITY_TRAIN)
    Xst, yst = X[s_start:cut], y[s_start:cut]
    out["stability_protocol"] = {
        "train_range": [int(s_start), int(cut)],
        "n_train": int(cut - s_start),
        "n_benign": int((yst == 0).sum()),
        "rationale": "20 independent fits are required; a temporally-coherent tail of "
                     "the training partition is used so the study is affordable, and "
                     "the size is reported rather than hidden",
    }
    print(f"  stability fits on {len(yst):,} flows "
          f"({int((yst == 0).sum()):,} benign)", flush=True)

    imps = []
    for si in range(N_SEEDS):
        kw = dict(RF_KW); kw["random_state"] = 1000 + si
        clf = RandomForestClassifier(**kw).fit(Xst, yst)
        imps.append(global_shap(clf, X_explain))
        print(f"  seed {si+1}/{N_SEEDS}", flush=True)
    imps = np.vstack(imps)

    boot = []
    nst = len(yst)
    for b in range(N_BOOT):
        bidx = rng.integers(0, nst, nst)
        kw = dict(RF_KW); kw["random_state"] = 42
        clf = RandomForestClassifier(**kw).fit(Xst[bidx], yst[bidx])
        boot.append(global_shap(clf, X_explain))
        print(f"  bootstrap {b+1}/{N_BOOT}", flush=True)
    boot = np.vstack(boot)

    def stability(M: np.ndarray) -> dict:
        res: dict = {}
        for k in (5, 10, 20):
            tops = [set(np.argsort(-row)[:k]) for row in M]
            js = [jaccard(tops[i], tops[j])
                  for i in range(len(tops)) for j in range(i + 1, len(tops))]
            res[f"top{k}_jaccard_mean"] = float(np.mean(js))
            res[f"top{k}_jaccard_min"] = float(np.min(js))
        rhos = []
        for i in range(len(M)):
            for j in range(i + 1, len(M)):
                rhos.append(spearmanr(M[i], M[j]).statistic)
        res["spearman_mean"] = float(np.mean(rhos))
        res["spearman_min"] = float(np.min(rhos))
        res["mean_cv_of_importance"] = float(
            np.nanmean(M.std(0) / np.where(M.mean(0) == 0, np.nan, M.mean(0))))
        return res

    out["shap_stability_across_seeds"] = stability(imps)
    out["shap_stability_across_bootstraps"] = stability(boot)
    mean_imp = imps.mean(0)
    order = np.argsort(-mean_imp)
    out["global_shap_ranking"] = [
        {"rank": int(i + 1), "feature": names[j],
         "mean_abs_shap": float(mean_imp[j]),
         "std_across_seeds": float(imps[:, j].std())}
        for i, j in enumerate(order[:25])
    ]
    pd.DataFrame({"feature": names, "mean_abs_shap": mean_imp,
                  "std_across_seeds": imps.std(0)}
                 ).sort_values("mean_abs_shap", ascending=False
                               ).to_csv(RESULTS / "E6_global_shap.csv", index=False)
    out["shap_seconds"] = round(time.time() - t0, 1)
    print(f"  seed top-10 Jaccard="
          f"{out['shap_stability_across_seeds']['top10_jaccard_mean']:.3f}, "
          f"bootstrap top-10 Jaccard="
          f"{out['shap_stability_across_bootstraps']['top10_jaccard_mean']:.3f}",
          flush=True)

    # SHAP agreement between the full and the most aggressive reduced set
    smallest = min((k for k in reduced_sets if k != "full"),
                   key=lambda k: len(reduced_sets[k]))
    reps = reduced_sets[smallest]
    kw = dict(RF_KW)
    clf_red = RandomForestClassifier(**kw).fit(Xst[:, reps], yst)
    imp_red = global_shap(clf_red, X_explain[:, reps])
    shared = {names[i] for i in np.argsort(-mean_imp)[:10]} & \
             {names[reps[i]] for i in np.argsort(-imp_red)[:10]}
    out["shap_full_vs_reduced"] = {
        "reduced_set": smallest, "n_reduced": len(reps),
        "top10_overlap_count": len(shared), "top10_overlap_features": sorted(shared),
    }

    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E6_collinearity_shap_stability", out)
    log_event("E6", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
