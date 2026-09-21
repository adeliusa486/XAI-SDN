"""E15 - Evaluation protocol formalisation (R7.8).

Reviewer 7: "Data splitting, cross-validation, and separation of tuning from
testing are insufficiently documented."

This experiment does not merely document the protocol, it executes it, so the
manuscript can report measured fold-level variance rather than an assurance.

  * Outer loop: blocked, forward-chaining temporal folds. Fold k trains on an
    expanding prefix and tests on the block that follows it, so no future
    observation ever informs a past decision.
  * Inner loop: grid search over the Random Forest hyperparameters, evaluated
    on a validation block carved from the END of each outer training prefix.
    The outer test block is untouched during selection.
  * Hyperparameter search runs on a temporally-coherent subsample of the
    training prefix, because a full grid at 2.5M x 88 is not tractable; the
    selected configuration is then refit on the complete prefix. The subsample
    size is reported so the compromise is visible rather than hidden.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

from paths import DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             confusion_matrix, f1_score, roc_auc_score)

CACHE = DATA_ROOT / "cache"
N_OUTER = 5
INNER_VAL_FRAC = 0.15          # tail of each outer training prefix
SEARCH_SUBSAMPLE = 300_000     # temporally-coherent tail of the prefix
TAU = 0.70

GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 20],
    "max_features": ["sqrt", 0.3],
    "min_samples_leaf": [1, 5],
}
FIXED = dict(bootstrap=True, class_weight="balanced", n_jobs=-1, random_state=42)


def score(y, proba, tau=TAU) -> dict:
    pred = (proba >= tau).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "avg_precision": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None,
        "precision": float(tp / (tp + fp)) if (tp + fp) else None,
        "recall": float(tp / (tp + fn)) if (tp + fn) else None,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> int:
    log_event("E15", "start")
    t_all = time.time()
    out: dict = {"experiment": "E15", "objective": "evaluation protocol formalisation"}

    X = np.load(CACHE / "syn0311_X_repaired.npy")
    y = np.load(CACHE / "syn0311_y.npy")
    n = len(y)

    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    out["design"] = {
        "outer_folds": N_OUTER,
        "outer_scheme": "blocked forward-chaining temporal split (expanding window)",
        "inner_scheme": f"validation block = last {INNER_VAL_FRAC:.0%} of the outer "
                        f"training prefix",
        "search_subsample": SEARCH_SUBSAMPLE,
        "search_subsample_rule": "temporally-coherent tail of the training prefix",
        "grid": GRID,
        "n_configurations": len(combos),
        "tuning_test_separation": "the outer test block is never read during "
                                  "hyperparameter selection or model fitting",
        "decision_threshold": TAU,
    }
    print(f"{N_OUTER} outer folds x {len(combos)} configurations", flush=True)

    # Forward-chaining blocks: reserve the first 40% as the initial training base.
    base = int(n * 0.40)
    block = (n - base) // N_OUTER
    folds = []
    for k in range(N_OUTER):
        tr_end = base + k * block
        te_end = tr_end + block if k < N_OUTER - 1 else n
        folds.append((0, tr_end, tr_end, te_end))

    rows, fold_results = [], []
    for k, (a, b, c, d_) in enumerate(folds, 1):
        ytr_full = y[a:b]
        yte = y[c:d_]
        info = {
            "fold": k, "train_range": [a, b], "test_range": [c, d_],
            "n_train": b - a, "n_test": d_ - c,
            "train_benign": int((ytr_full == 0).sum()),
            "test_benign": int((yte == 0).sum()),
        }
        print(f"\nfold {k}: train[{a}:{b}] ({b-a:,}, benign {info['train_benign']:,})  "
              f"test[{c}:{d_}] ({d_-c:,}, benign {info['test_benign']:,})", flush=True)
        if info["train_benign"] < 2 or info["test_benign"] < 1:
            info["skipped"] = "insufficient benign flows in this temporal block"
            print(f"  skipped: {info['skipped']}", flush=True)
            fold_results.append(info)
            continue

        v_start = b - max(1, int((b - a) * INNER_VAL_FRAC))
        s_start = max(a, v_start - SEARCH_SUBSAMPLE)
        Xs, ys = X[s_start:v_start], y[s_start:v_start]
        Xv, yv = X[v_start:b], y[v_start:b]
        info["inner"] = {"search_range": [s_start, v_start],
                         "validation_range": [v_start, b],
                         "n_search": int(v_start - s_start),
                         "n_validation": int(b - v_start),
                         "search_benign": int((ys == 0).sum()),
                         "validation_benign": int((yv == 0).sum())}
        if info["inner"]["search_benign"] < 2 or info["inner"]["validation_benign"] < 1:
            info["skipped"] = "inner split has no benign flows; cannot select"
            print(f"  skipped: {info['skipped']}", flush=True)
            fold_results.append(info)
            continue

        best, best_score, trials = None, -1.0, []
        for cfg in combos:
            t0 = time.time()
            m = RandomForestClassifier(**{**FIXED, **cfg}).fit(Xs, ys)
            pv = m.predict_proba(Xv)[:, 1]
            sc = score(yv, pv)
            sc["config"] = cfg
            sc["fit_seconds"] = round(time.time() - t0, 2)
            trials.append(sc)
            if sc["macro_f1"] > best_score:
                best, best_score = cfg, sc["macro_f1"]
        info["inner"]["trials"] = trials
        info["selected_config"] = best
        info["selected_validation_macro_f1"] = best_score
        print(f"  selected {best} (inner macro-F1={best_score:.6f})", flush=True)

        t0 = time.time()
        final = RandomForestClassifier(**{**FIXED, **best}).fit(X[a:b], y[a:b])
        info["refit_seconds"] = round(time.time() - t0, 1)
        pte = final.predict_proba(X[c:d_])[:, 1]
        info["outer_test"] = score(yte, pte)
        m = info["outer_test"]
        print(f"  OUTER acc={m['accuracy']:.6f} macroF1={m['macro_f1']:.6f} "
              f"AP={m['avg_precision']:.6f} FPR={m['fpr']}", flush=True)
        fold_results.append(info)
        rows.append({"fold": k, **{kk: m[kk] for kk in
                                   ("accuracy", "macro_f1", "avg_precision",
                                    "roc_auc", "precision", "recall", "fpr", "fnr")},
                     "selected_config": json.dumps(best)})

    out["folds"] = fold_results
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(RESULTS / "E15_nested_cv_folds.csv", index=False)
        num = df.select_dtypes(include=[np.number]).drop(columns=["fold"], errors="ignore")
        out["outer_summary"] = {
            c: {"mean": float(num[c].mean()), "std": float(num[c].std(ddof=1))
                if len(num) > 1 else 0.0,
                "min": float(num[c].min()), "max": float(num[c].max())}
            for c in num.columns
        }
        sel = [json.loads(s) for s in df["selected_config"]]
        out["selected_config_agreement"] = {
            "unique_configurations": len({json.dumps(s, sort_keys=True) for s in sel}),
            "per_fold": sel,
        }
        print("\nouter-fold summary:",
              json.dumps({k: round(v["mean"], 6) for k, v in out["outer_summary"].items()},
                         indent=2), flush=True)
    else:
        out["outer_summary"] = None
        out["note"] = ("no outer fold produced a usable evaluation; the benign class "
                       "is too temporally concentrated for forward-chaining folds. "
                       "This is itself a reportable property of the partition.")
        print("\n" + out["note"], flush=True)

    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E15_protocol", out)
    log_event("E15", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
