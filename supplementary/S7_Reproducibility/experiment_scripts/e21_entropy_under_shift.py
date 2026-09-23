"""E21 - does the entropy augmentation help under distribution shift?

E17 answers the in-distribution question and the answer is negative: on the SYN
partition the eighty exported flow statistics alone reach a slightly higher macro
F1 than the full 88-dimensional vector, and scrambling the arrival order of the
entropy features changes nothing. Taken alone that result leaves the augmentation
with no demonstrated value.

It is the wrong place to look. A distributional summary of a traffic window is a
claim about structure that should survive a change of attack vector, of capture
day, or of network, where a raw per-flow counter such as a byte rate need not.
E17 measures the one setting in which CICFlowMeter has already encoded everything
the entropy features could add. This run measures the settings in which it may
not have.

Three feature sets are trained identically and evaluated on every shift that the
revision already has data for:

  full          80 flow statistics + 8 entropy features
  cic_only      80 flow statistics
  entropy_only  8 entropy features

Shifts: three held-out attack vectors from the same capture, a second capture
day, and, where the schema allows, the two external corpora. The comparison of
interest is full against cic_only on each shift. If the augmentation earns its
place it will show up here or nowhere, and if it does not we report that the
augmentation is not justified by any measurement in this paper.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))

from paths import DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix,  # noqa: E402
                             f1_score, roc_auc_score)

CACHE = DATA_ROOT / "cache"
RF_KW = dict(n_estimators=200, max_depth=None, max_features="sqrt", bootstrap=True,
             class_weight="balanced", n_jobs=-1, random_state=42)
TAU = 0.70
TRAIN_CAP = 600_000
TEST_CAP = 400_000
N_BOOT = 300
SEEDS = [42, 123, 456]

VECTORS = ["LDAP_0311", "NetBIOS_0311", "Portmap_0311", "SYN_0112"]


def split_sets(names: list[str]) -> dict[str, np.ndarray]:
    ent = np.array([i for i, n in enumerate(names) if n.startswith("H_")])
    cic = np.array([i for i, n in enumerate(names) if not n.startswith("H_")])
    return {"full": np.arange(len(names)), "cic_only": cic, "entropy_only": ent}


def score(y, s) -> dict:
    pred = (s >= TAU).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None,
        "avg_precision": float(average_precision_score(y, s)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "n_test": int(len(y)),
    }


def boot(y, s, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    a = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        a.append(f1_score(y[i], (s[i] >= TAU).astype(np.int8), average="macro",
                          zero_division=0))
    if not a:
        return {}
    a = np.asarray(a)
    return {"macro_f1_lo95": float(np.percentile(a, 2.5)),
            "macro_f1_hi95": float(np.percentile(a, 97.5))}


def load_vector(name: str):
    fx, fy, fn = (CACHE / f"{name}_X.npy", CACHE / f"{name}_y.npy",
                  CACHE / f"{name}_names.json")
    if not all(p.exists() for p in (fx, fy, fn)):
        return None
    X = np.load(fx, mmap_mode="r")
    y = np.load(fy)
    names = json.loads(fn.read_text(encoding="utf-8"))
    if len(y) > TEST_CAP:                      # tail: benign traffic is late
        X, y = X[-TEST_CAP:], y[-TEST_CAP:]
    return np.asarray(X), y, names


def main() -> int:
    log_event("E21", "start")
    t_all = time.time()
    out: dict = {
        "experiment": "E21",
        "objective": "value of the entropy augmentation under distribution shift",
        "tau": TAU, "seeds": SEEDS,
        "train_cap": TRAIN_CAP, "test_cap": TEST_CAP,
        "rationale": ("E17 shows the augmentation does not help in distribution. "
                      "This run asks whether it helps where a distributional summary "
                      "should: across attack vectors, capture days and corpora."),
    }

    names = json.loads((CACHE / "syn0311_feature_names_repaired.json")
                       .read_text(encoding="utf-8"))
    sets = split_sets(names)
    out["feature_set_sizes"] = {k: int(len(v)) for k, v in sets.items()}
    print("feature sets:", out["feature_set_sizes"], flush=True)

    Xs = np.load(CACHE / "syn0311_X_repaired.npy", mmap_mode="r")
    ys = np.load(CACHE / "syn0311_y.npy")
    cut = int(len(ys) * 0.70)
    lo = max(0, cut - TRAIN_CAP)
    Xtr = np.asarray(Xs[lo:cut])
    ytr = ys[lo:cut]
    print(f"source train {Xtr.shape}, benign {int((ytr == 0).sum()):,}", flush=True)

    # in-distribution reference on the source test partition, for the same models
    Xte_src = np.asarray(Xs[cut:cut + TEST_CAP])
    yte_src = ys[cut:cut + TEST_CAP]

    rows = []
    models: dict[tuple[str, int], RandomForestClassifier] = {}
    for fs, cols in sets.items():
        for seed in SEEDS:
            t0 = time.time()
            m = RandomForestClassifier(**{**RF_KW, "random_state": seed})
            m.fit(Xtr[:, cols], ytr)
            models[(fs, seed)] = m
            s = m.predict_proba(Xte_src[:, cols])[:, 1]
            r = score(yte_src, s) | boot(yte_src, s)
            rows.append({"shift": "in_distribution", "target": "SYN_0311",
                         "feature_set": fs, "seed": seed,
                         "fit_seconds": round(time.time() - t0, 1), **r})
            print(f"  [in_distribution ] {fs:13s} seed={seed:<5} "
                  f"F1={r['macro_f1']:.4f}", flush=True)

    for vec in VECTORS:
        loaded = load_vector(vec)
        if loaded is None:
            print(f"{vec}: no cache, skipped", flush=True)
            continue
        Xv, yv, vnames = loaded
        if vnames != names or len(np.unique(yv)) < 2:
            print(f"{vec}: schema mismatch or single class, skipped", flush=True)
            continue
        shift = "cross_day" if vec.endswith("0112") else "cross_vector"
        for fs, cols in sets.items():
            for seed in SEEDS:
                s = models[(fs, seed)].predict_proba(Xv[:, cols])[:, 1]
                r = score(yv, s) | boot(yv, s)
                rows.append({"shift": shift, "target": vec, "feature_set": fs,
                             "seed": seed, **r})
            f1s = [x["macro_f1"] for x in rows
                   if x["target"] == vec and x["feature_set"] == fs]
            print(f"  [{shift:14s}] {vec:14s} {fs:13s} "
                  f"F1={np.mean(f1s):.4f}", flush=True)

    for slug in ("insdn", "ids2017"):
        fx = CACHE / f"{slug}_X.npy"
        ff = CACHE / f"{slug}_features.json"
        if not (fx.exists() and ff.exists()):
            print(f"{slug}: no cache, skipped", flush=True)
            continue
        tnames = json.loads(ff.read_text(encoding="utf-8"))
        Xt = np.load(fx)
        yt = np.load(CACHE / f"{slug}_y.npy")
        Xsrc = np.load(CACHE / f"{slug}_src_X.npy")
        ysrc = np.load(CACHE / f"{slug}_src_y.npy")
        if len(Xsrc) > TRAIN_CAP:
            Xsrc, ysrc = Xsrc[-TRAIN_CAP:], ysrc[-TRAIN_CAP:]
        tsets = split_sets(tnames)
        if len(tsets["entropy_only"]) == 0 or len(np.unique(yt)) < 2:
            print(f"{slug}: no shared entropy features, skipped", flush=True)
            continue
        for fs, cols in tsets.items():
            for seed in SEEDS:
                m = RandomForestClassifier(**{**RF_KW, "random_state": seed})
                m.fit(Xsrc[:, cols], ysrc)
                s = m.predict_proba(Xt[:, cols])[:, 1]
                r = score(yt, s) | boot(yt, s)
                rows.append({"shift": "cross_dataset", "target": slug,
                             "feature_set": fs, "seed": seed,
                             "transfer_dimension": int(len(tnames)), **r})
            f1s = [x["macro_f1"] for x in rows
                   if x["target"] == slug and x["feature_set"] == fs]
            print(f"  [cross_dataset  ] {slug:14s} {fs:13s} "
                  f"F1={np.mean(f1s):.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "E21_entropy_under_shift.csv", index=False)

    verdicts = {}
    for shift in sorted(df["shift"].unique()):
        sub = df[df["shift"] == shift]
        agg = sub.groupby("feature_set")["macro_f1"].agg(["mean", "std", "count"])
        full = float(agg.loc["full", "mean"])
        cic = float(agg.loc["cic_only", "mean"]) if "cic_only" in agg.index else None
        verdicts[shift] = {
            "full_macro_f1_mean": full,
            "cic_only_macro_f1_mean": cic,
            "entropy_helps": bool(cic is not None and full > cic),
            "delta": (round(full - cic, 6) if cic is not None else None),
            "per_set": {k: {"mean": float(v["mean"]), "std": float(v["std"]),
                            "n": int(v["count"])}
                        for k, v in agg.iterrows()},
        }
        print(f"\n{shift}: full={full:.4f} cic_only={cic:.4f} "
              f"delta={full - cic:+.4f}", flush=True)

    helps = [k for k, v in verdicts.items()
             if k != "in_distribution" and v["entropy_helps"]]
    out["by_shift"] = verdicts
    out["shifts_where_entropy_helps"] = helps
    out["verdict"] = (
        "The entropy augmentation improves macro F1 under "
        + ", ".join(helps) + " while not improving it in distribution, which is "
        "the pattern a distributional feature should show: it contributes nothing "
        "where the exported counters already separate the classes, and contributes "
        "where they no longer transfer."
        if helps else
        "The entropy augmentation does not improve macro F1 under any shift "
        "measured here, nor in distribution. On the evidence in this paper the "
        "augmentation is not justified by detection performance, and its case rests "
        "only on its negligible cost. We state this rather than searching for a "
        "setting in which the sign is favourable.")
    print("\nVERDICT:", out["verdict"], flush=True)

    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E21_entropy_under_shift", out)
    log_event("E21", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
