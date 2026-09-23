"""E12 - Precision-recall based threshold selection.

Selecting a threshold from a sweep reporting "zero FNR below tau=0.5" is
uninformative under severe class imbalance. Under 99:1 prevalence, FNR is
computed over the
majority class and is therefore insensitive, while the quantity an operator
actually cares about - how many alerts are wrong - lives on the precision axis.

This experiment:
  * builds a chronological train / validation / test partition so that the
    threshold is never selected on test data;
  * reports the precision-recall curve and average precision on validation;
  * selects tau by three defensible criteria and reports each;
  * re-expresses the threshold sweep in precision, recall, F1 and alert volume
    instead of FNR;
  * reports the test operating point separately, once, at the selected tau.
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

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

CACHE = DATA_ROOT / "cache"
RF_KW = dict(n_estimators=200, max_depth=None, max_features="sqrt", bootstrap=True,
             class_weight="balanced", n_jobs=-1, random_state=42)
TRAIN_FRAC, VAL_FRAC = 0.60, 0.10        # test = 0.30, chronological
PRECISION_TARGET = 0.999
SWEEP = np.round(np.arange(0.05, 1.00, 0.05), 2)


def operating_point(y, score, tau) -> dict:
    pred = (score >= tau).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return {
        "tau": float(tau),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "alerts": int(tp + fp), "false_alerts": int(fp), "missed_attacks": int(fn),
        "alert_rate": float((tp + fp) / len(y)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> int:
    log_event("E12", "start")
    t_all = time.time()
    out: dict = {"experiment": "E12",
                 "objective": "precision-recall based threshold selection"}

    X = np.load(CACHE / "syn0311_X_repaired.npy")
    y = np.load(CACHE / "syn0311_y.npy")
    n = len(y)
    i_tr = int(round(n * TRAIN_FRAC))
    i_va = int(round(n * (TRAIN_FRAC + VAL_FRAC)))
    out["protocol"] = {
        "type": "chronological train / validation / test",
        "train": [0, i_tr], "validation": [i_tr, i_va], "test": [i_va, n],
        "n_train": i_tr, "n_validation": i_va - i_tr, "n_test": n - i_va,
        "train_benign": int((y[:i_tr] == 0).sum()),
        "validation_benign": int((y[i_tr:i_va] == 0).sum()),
        "test_benign": int((y[i_va:] == 0).sum()),
        "statement": "the threshold is selected on the validation partition only; "
                     "no test observation participates in any tuning decision",
    }
    print(json.dumps({k: out["protocol"][k] for k in
                      ("n_train", "n_validation", "n_test", "train_benign",
                       "validation_benign", "test_benign")}, indent=2), flush=True)

    if out["protocol"]["validation_benign"] == 0 or out["protocol"]["test_benign"] == 0:
        out["error"] = ("a chronological partition contains no benign flows; "
                        "threshold selection is undefined under this split")
        print("ABORT:", out["error"], flush=True)
        save_result("E12_pr_threshold", out)
        log_event("E12", "failed", reason=out["error"])
        return 1

    print("fitting on the training partition...", flush=True)
    t0 = time.time()
    clf = RandomForestClassifier(**RF_KW).fit(X[:i_tr], y[:i_tr])
    out["train_time_s"] = round(time.time() - t0, 2)

    s_va = clf.predict_proba(X[i_tr:i_va])[:, 1]
    s_te = clf.predict_proba(X[i_va:])[:, 1]
    y_va, y_te = y[i_tr:i_va], y[i_va:]

    # ---- validation PR curve ------------------------------------------------
    prec, rec, thr = precision_recall_curve(y_va, s_va)
    ap = average_precision_score(y_va, s_va)
    fpr_c, tpr_c, thr_r = roc_curve(y_va, s_va)
    out["validation_curves"] = {
        "average_precision": float(ap),
        "pr_auc_note": "average precision is the PR-AUC estimator used here",
        "roc_auc": float(roc_auc_score(y_va, s_va)),
        "positive_prevalence": float(y_va.mean()),
        "baseline_precision_of_random_classifier": float(y_va.mean()),
    }
    print(f"  validation AP={ap:.6f}  ROC-AUC={out['validation_curves']['roc_auc']:.6f} "
          f"prevalence={y_va.mean():.4f}", flush=True)

    pd.DataFrame({"threshold": np.append(thr, np.nan),
                  "precision": prec, "recall": rec}
                 ).to_csv(RESULTS / "E12_validation_pr_curve.csv", index=False)
    pd.DataFrame({"threshold": thr_r, "fpr": fpr_c, "tpr": tpr_c}
                 ).to_csv(RESULTS / "E12_validation_roc_curve.csv", index=False)

    # ---- threshold sweep, expressed on the precision-recall plane -------------
    sweep_rows = [operating_point(y_va, s_va, t) for t in SWEEP]
    pd.DataFrame([{k: v for k, v in r.items() if k != "confusion"}
                  for r in sweep_rows]).to_csv(RESULTS / "E12_validation_sweep.csv",
                                               index=False)
    out["validation_sweep"] = [{k: v for k, v in r.items() if k != "confusion"}
                               for r in sweep_rows]

    # ---- three selection criteria ------------------------------------------
    f1s = np.nan_to_num([r["f1"] for r in sweep_rows], nan=-1)
    tau_f1 = float(SWEEP[int(np.argmax(f1s))])

    # finest-grained max-F1 directly on the PR curve
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_curve = 2 * prec * rec / (prec + rec)
    f1_curve = np.nan_to_num(f1_curve, nan=-1)
    k = int(np.argmax(f1_curve[:-1])) if len(thr) else 0
    tau_f1_fine = float(thr[k]) if len(thr) else 0.5

    ok = np.flatnonzero(prec[:-1] >= PRECISION_TARGET) if len(thr) else np.array([])
    if ok.size:
        best = ok[int(np.argmax(rec[:-1][ok]))]
        tau_prec = float(thr[best])
        rec_at_target = float(rec[:-1][best])
    else:
        tau_prec, rec_at_target = float("nan"), float("nan")

    out["threshold_selection"] = {
        "criterion_max_f1_grid": {"tau": tau_f1,
                                  "note": "0.05 grid, for comparability with the sweep table"},
        "criterion_max_f1_fine": {"tau": tau_f1_fine,
                                  "f1": float(f1_curve[k]) if len(thr) else None},
        "criterion_precision_target": {"precision_target": PRECISION_TARGET,
                                       "tau": tau_prec,
                                       "recall_at_target": rec_at_target},
        "submitted_paper_tau": 0.70,
        "selected": tau_f1_fine,
        "selected_on": "validation partition",
    }
    print(f"  tau(max-F1 grid)={tau_f1}  tau(max-F1 fine)={tau_f1_fine:.4f}  "
          f"tau(precision>={PRECISION_TARGET})={tau_prec}", flush=True)

    # ---- single test evaluation at the selected threshold -------------------
    tau_star = tau_f1_fine
    out["test_operating_point"] = operating_point(y_te, s_te, tau_star)
    out["test_operating_point_at_submitted_tau_0.70"] = operating_point(y_te, s_te, 0.70)
    out["test_curves"] = {
        "average_precision": float(average_precision_score(y_te, s_te)),
        "roc_auc": float(roc_auc_score(y_te, s_te)),
        "positive_prevalence": float(y_te.mean()),
    }
    tp_ = out["test_operating_point"]
    print(f"  TEST @tau*={tau_star:.4f}: precision={tp_['precision']:.6f} "
          f"recall={tp_['recall']:.6f} F1={tp_['f1']:.6f} "
          f"false_alerts={tp_['false_alerts']:,} missed={tp_['missed_attacks']:,}",
          flush=True)
    print(f"  TEST AP={out['test_curves']['average_precision']:.6f} "
          f"ROC-AUC={out['test_curves']['roc_auc']:.6f}", flush=True)

    p_te, r_te, t_te = precision_recall_curve(y_te, s_te)
    pd.DataFrame({"threshold": np.append(t_te, np.nan),
                  "precision": p_te, "recall": r_te}
                 ).to_csv(RESULTS / "E12_test_pr_curve.csv", index=False)

    out["why_fnr_is_misleading_here"] = {
        "test_prevalence_positive": float(y_te.mean()),
        "negatives_in_test": int((y_te == 0).sum()),
        "note": ("FNR is normalised by the positive class, which holds "
                 f"{float(y_te.mean()):.2%} of the test flows, so a change of one "
                 "false negative moves it by a negligible amount. Precision is "
                 f"normalised by the alert volume and is driven by the "
                 f"{int((y_te == 0).sum()):,} negatives, which is the quantity an "
                 "operator experiences as false alarms."),
    }

    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E12_pr_threshold", out)
    log_event("E12", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
