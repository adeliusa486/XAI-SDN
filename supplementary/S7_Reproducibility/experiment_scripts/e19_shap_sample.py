"""E19 - per-flow TreeSHAP sample for the attribution figure.

A beeswarm needs one attribution vector per flow, which E6 does not persist
because it only needs the mean. This run produces that matrix under the
primary chronological protocol.

This run fits the detector under the primary chronological protocol and writes
the attribution matrix, the matching feature values, and the base value, for a
sample of test flows drawn so that both classes are visible. It exists to
produce figure data, so it is deliberately small and does no analysis of its
own; every number quoted in the text comes from E5, E6 or E17.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))

from paths import DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

CACHE = DATA_ROOT / "cache"
RF_KW = dict(n_estimators=200, max_depth=None, max_features="sqrt", bootstrap=True,
             class_weight="balanced", n_jobs=-1, random_state=42)
N_EXPLAIN = 1500          # flows drawn for the figure
BENIGN_SHARE = 0.35       # a beeswarm of 97% attack traffic shows one blob


def main() -> int:
    log_event("E19", "start")
    t_all = time.time()

    names = json.loads((CACHE / "syn0311_feature_names_repaired.json")
                       .read_text(encoding="utf-8"))
    X = np.load(CACHE / "syn0311_X_repaired.npy", mmap_mode="r")
    y = np.load(CACHE / "syn0311_y.npy")
    cut = int(len(y) * 0.70)
    print(f"loaded {X.shape}, chronological cut at {cut:,}", flush=True)

    t0 = time.time()
    clf = RandomForestClassifier(**RF_KW).fit(np.asarray(X[:cut]), y[:cut])
    fit_s = time.time() - t0
    print(f"fitted in {fit_s:.1f}s", flush=True)

    rng = np.random.default_rng(42)
    te = np.arange(cut, len(y))
    ben = te[y[cut:] == 0]
    att = te[y[cut:] == 1]
    n_ben = min(len(ben), int(N_EXPLAIN * BENIGN_SHARE))
    n_att = N_EXPLAIN - n_ben
    idx = np.sort(np.concatenate([rng.choice(ben, n_ben, replace=False),
                                  rng.choice(att, n_att, replace=False)]))
    Xs = np.asarray(X[idx])
    ys = y[idx]
    print(f"explaining {len(idx):,} flows ({n_ben:,} benign, {n_att:,} attack)",
          flush=True)

    import shap
    ex = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    t0 = time.time()
    sv = ex.shap_values(Xs, check_additivity=False)
    shap_s = time.time() - t0
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
    base = ex.expected_value
    base = float(base[1] if np.ndim(base) else base)

    out_dir = RESULTS / "shap_sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "shap_values.npy", sv.astype(np.float32))
    np.save(out_dir / "feature_values.npy", Xs.astype(np.float32))
    np.save(out_dir / "y.npy", ys.astype(np.int8))
    (out_dir / "feature_names.json").write_text(json.dumps(names), encoding="utf-8")

    order = np.argsort(np.abs(sv).mean(axis=0))[::-1]
    top = [{"feature": names[j], "mean_abs_shap": float(np.abs(sv[:, j]).mean())}
           for j in order[:15]]
    for r in top:
        print(f"  {r['feature']:26s} {r['mean_abs_shap']:.6f}", flush=True)

    out = {
        "experiment": "E19",
        "objective": "per-flow TreeSHAP sample for the attribution figure",
        "protocol": "chronological 70/30, repaired 88-dimensional feature set",
        "n_train": int(cut),
        "n_explained": int(len(idx)),
        "n_explained_benign": int(n_ben),
        "n_explained_attack": int(n_att),
        "benign_oversampled": True,
        "oversampling_note": (
            "Test flows are 2.8% benign, so a sample drawn at the natural ratio "
            "would show one class only. Benign flows are deliberately "
            "over-represented in this figure sample; no metric is computed from "
            "it, and every reported number comes from the full test partition."),
        "shap_base_value": base,
        "top_features": top,
        "fit_seconds": round(fit_s, 1),
        "shap_seconds": round(shap_s, 1),
        "artifacts": sorted(p.name for p in out_dir.iterdir()),
        "total_runtime_s": round(time.time() - t_all, 1),
    }
    p = save_result("E19_shap_sample", out)
    log_event("E19", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
