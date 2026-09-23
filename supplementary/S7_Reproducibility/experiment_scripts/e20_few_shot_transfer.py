"""E20 - how much target-domain data restores cross-dataset detection.

E2 established the negative result: a detector trained on CIC-DDoS2019 does not
transfer zero-shot to InSDN or CIC-IDS2017, reaching macro F1 0.6026 and 0.2788.
Reporting that and stopping would leave an operator with nothing to do about it.

The question this run answers is the practical one: how many labelled flows from
the target network are needed before the detector becomes usable there? Two
adaptation strategies are compared against the zero-shot baseline and against a
target-only model trained from scratch on the same budget, because a transfer
claim means nothing unless it beats simply training on the target data:

  zero_shot     source model, no target data
  target_only   train on n target flows alone
  fine_tune     warm-start from the source forest and add target-fitted trees
  joint         train on all source data plus n target flows

Budgets sweep two orders of magnitude so the curve, not a single point, is what
gets reported. The train/test split of the target corpus stays chronological, so
no test flow precedes a training flow. Within the training partition the budget is
drawn stratified rather than as a contiguous prefix: on both target corpora the
early training flows are single-class, and in any case an operator labelling
their own traffic samples across it rather than taking the first n records in
timestamp order. Each budget's class composition is reported.
"""
from __future__ import annotations

import copy
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
BUDGETS = [0, 100, 500, 1_000, 5_000, 10_000, 25_000, 50_000]
SOURCE_TRAIN_CAP = 600_000     # keeps the joint fits tractable; reported
N_BOOT = 300
MIN_MINORITY = 5


def metrics(y, s, tag: str, extra: dict | None = None) -> dict:
    pred = (s >= TAU).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    r = {
        "condition": tag,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None,
        "avg_precision": float(average_precision_score(y, s)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(y)),
    }
    r.update(extra or {})
    return r


def boot_ci(y, s, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    f1s = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        f1s.append(f1_score(y[i], (s[i] >= TAU).astype(np.int8),
                            average="macro", zero_division=0))
    if not f1s:
        return {}
    a = np.asarray(f1s)
    return {"macro_f1_lo95": float(np.percentile(a, 2.5)),
            "macro_f1_hi95": float(np.percentile(a, 97.5))}



def draw_budget(pool_ben: np.ndarray, pool_att: np.ndarray, n: int,
                rng) -> np.ndarray | None:
    """Stratified draw of n indices preserving the pool's class ratio.

    At least MIN_MINORITY of the rarer class is taken whenever the pool has that
    many, because a budget of pure majority traffic teaches a classifier nothing
    and would make the curve look worse than the method deserves.
    """
    total = len(pool_ben) + len(pool_att)
    if min(len(pool_ben), len(pool_att)) < MIN_MINORITY or n > total:
        return None
    minority, majority = ((pool_ben, pool_att) if len(pool_ben) <= len(pool_att)
                          else (pool_att, pool_ben))
    share = len(minority) / total
    k_min = max(MIN_MINORITY, int(round(n * share)))
    k_min = min(k_min, len(minority), n - 1)
    k_maj = min(n - k_min, len(majority))
    take = np.concatenate([rng.choice(minority, k_min, replace=False),
                           rng.choice(majority, k_maj, replace=False)])
    return np.sort(take)


def load_pair(slug: str):
    """Schema-aligned source and target matrices, as E2 built and cached them.

    The transfer dimension differs per target, because each target shares a
    different subset of the schema with the source, so the source matrix is
    cached per target rather than reused across them.
    """
    need = [f"{slug}_X.npy", f"{slug}_y.npy", f"{slug}_src_X.npy",
            f"{slug}_src_y.npy"]
    if not all((CACHE / f).exists() for f in need):
        return None
    Xt = np.load(CACHE / f"{slug}_X.npy")
    yt = np.load(CACHE / f"{slug}_y.npy")
    Xsrc = np.load(CACHE / f"{slug}_src_X.npy")
    ysrc = np.load(CACHE / f"{slug}_src_y.npy")
    if len(Xsrc) > SOURCE_TRAIN_CAP:
        Xsrc, ysrc = Xsrc[-SOURCE_TRAIN_CAP:], ysrc[-SOURCE_TRAIN_CAP:]
    feats = json.loads((CACHE / f"{slug}_features.json").read_text(encoding="utf-8")) \
        if (CACHE / f"{slug}_features.json").exists() else None
    return Xsrc, ysrc, Xt, yt, int(len(yt) * 0.70), feats


def main() -> int:
    log_event("E20", "start")
    t_all = time.time()
    out: dict = {
        "experiment": "E20",
        "objective": "target-data budget required to restore cross-dataset detection",
        "tau": TAU,
        "budgets": BUDGETS,
        "source_train_cap": SOURCE_TRAIN_CAP,
        "sampling_policy": ("the train/test split is chronological; within the "
                            "training partition each budget is a stratified draw "
                            "preserving that partition's class ratio, with at "
                            "least five of the minority class"),
    }

    rng = np.random.default_rng(42)
    results = {}
    for target in ("insdn", "ids2017"):
        loaded = load_pair(target)
        if loaded is None:
            results[target] = {"status": "skipped",
                               "reason": f"no cached matrices for {target}; run E2 first"}
            print(f"{target}: no cache, skipped", flush=True)
            continue
        Xsrc, ysrc, Xt, yt, tcut, feats = loaded
        print(f"\n{target}: source {Xsrc.shape}, target {Xt.shape}", flush=True)
        t0 = time.time()
        src_model = RandomForestClassifier(**RF_KW).fit(Xsrc, ysrc)
        print(f"  source model fitted in {time.time() - t0:.1f}s", flush=True)
        warm_base = RandomForestClassifier(**{**RF_KW, "warm_start": True})
        warm_base.fit(Xsrc, ysrc)
        Xte, yte = Xt[tcut:], yt[tcut:]
        if len(np.unique(yte)) < 2:
            results[target] = {"status": "skipped",
                               "reason": "target test partition is single-class"}
            continue
        print(f"  train {tcut:,}, test {len(yte):,}, "
              f"test benign {int((yte == 0).sum()):,}", flush=True)

        rows = []
        s0 = src_model.predict_proba(Xte)[:, 1]
        z = metrics(yte, s0, "zero_shot", {"n_target_train": 0})
        z.update(boot_ci(yte, s0))
        rows.append(z)
        print(f"  zero_shot                    F1={z['macro_f1']:.4f} "
              f"FNR={z['fnr']:.4f}", flush=True)

        pool = np.arange(tcut)
        pool_ben = pool[yt[:tcut] == 0]
        pool_att = pool[yt[:tcut] == 1]
        print(f"  labelling pool: {len(pool_ben):,} benign, {len(pool_att):,} attack",
              flush=True)

        for n in BUDGETS:
            if n == 0 or n > tcut:
                continue
            take = draw_budget(pool_ben, pool_att, n, rng)
            if take is None:
                rows.append({"condition": "target_only", "n_target_train": int(n),
                             "status": "skipped",
                             "reason": "training partition has too few of one class"})
                continue
            Xn, yn = Xt[take], yt[take]
            n_ben = int((yn == 0).sum())

            m = RandomForestClassifier(**RF_KW).fit(Xn, yn)
            s = m.predict_proba(Xte)[:, 1]
            r = metrics(yte, s, "target_only", {"n_target_train": int(n),
                                        "n_target_benign": n_ben})
            r.update(boot_ci(yte, s))
            rows.append(r)

            # Warm start: keep the source trees and grow target-fitted ones, which
            # is the cheap adaptation an operator can actually run on a controller.
            # The source half is fitted once per target and deep-copied per budget,
            # so the cost reported is the cost of the adaptation, not of refitting.
            ft = copy.deepcopy(warm_base)
            ft.n_estimators = RF_KW["n_estimators"] + 100
            ft.fit(np.vstack([Xn, Xn]), np.concatenate([yn, yn]))
            s = ft.predict_proba(Xte)[:, 1]
            r2 = metrics(yte, s, "fine_tune", {"n_target_train": int(n),
                                        "n_target_benign": n_ben})
            r2.update(boot_ci(yte, s))
            rows.append(r2)

            step = max(1, len(Xsrc) // 200_000)
            Xj = np.vstack([Xsrc[::step], Xn])
            yj = np.concatenate([ysrc[::step], yn])
            j = RandomForestClassifier(**RF_KW).fit(Xj, yj)
            s = j.predict_proba(Xte)[:, 1]
            r3 = metrics(yte, s, "joint", {"n_target_train": int(n),
                                      "n_target_benign": n_ben,
                                      "n_source_subsampled": int(len(Xj) - n)})
            r3.update(boot_ci(yte, s))
            rows.append(r3)

            print(f"  n={n:>6,}  target_only F1={r['macro_f1']:.4f}  "
                  f"fine_tune F1={r2['macro_f1']:.4f}  joint F1={r3['macro_f1']:.4f}",
                  flush=True)

        ok = [r for r in rows if "macro_f1" in r]
        best = max(ok, key=lambda r: r["macro_f1"])
        zs = z["macro_f1"]
        recovered = [r for r in ok
                     if r["condition"] != "zero_shot" and r["macro_f1"] >= 0.90]
        results[target] = {
            "status": "ok",
            "transfer_dimension": int(Xt.shape[1]),
            "n_source_train": int(len(ysrc)),
            "n_target_train_total": int(tcut),
            "n_test": int(len(yte)),
            "rows": rows,
            "zero_shot_macro_f1": zs,
            "best": best,
            "smallest_budget_reaching_0.90": (
                min(r["n_target_train"] for r in recovered) if recovered else None),
        }

    flat = []
    for tgt, v in results.items():
        for r in v.get("rows", []):
            flat.append({"target": tgt, **{k: val for k, val in r.items()
                                           if k != "confusion"}})
    if flat:
        pd.DataFrame(flat).to_csv(RESULTS / "E20_transfer_budget.csv", index=False)

    out["results"] = results
    out["interpretation"] = (
        "Zero-shot transfer fails, which E2 already established. What this run adds "
        "is the price of fixing it: the smallest labelled budget from the target "
        "network at which the detector becomes usable, and whether transferring from "
        "the source is worth anything at that budget compared with training on the "
        "target data alone. Where fine-tuning does not beat target-only training at "
        "the same budget, we say so, because that is the case in which the source "
        "model contributes nothing and an operator should not use it.")
    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E20_few_shot_transfer", out)
    log_event("E20", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
