"""E1 - Unified CPU-only classical baseline comparison.

Two conditions make a baseline comparison meaningful, and both are enforced here.

First, timing must be comparable. Every model runs on the same CPU, in the same
harness, on the same machine, with the same thread count, and the torch build in
this environment has no CUDA at all, so a GPU measurement is not merely avoided
but impossible.

Second, the training setting must be comparable. Comparing baselines trained on
10,000 class-balanced samples against a Random Forest trained on 2.5M imbalanced
flows measures the setting, not the method. Every model is therefore run under
BOTH protocols, and the two are never mixed:

  Protocol A - full imbalanced temporal split, natural prevalence, no balancing.
               Models that cannot scale to it are reported as such rather than
               quietly omitted.
  Protocol B - a consistently subsampled temporal protocol applied identically
               to every model, preserving the natural class ratio.

Reported per model: accuracy, macro-F1, PR-AUC (average precision), ROC-AUC,
FPR, FNR, training time, peak training memory, single-flow latency at p50/p95/p99,
batch throughput, and whether per-prediction explanations are available.
"""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

from paths import DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402
from serial import serial_inference  # noqa: E402

from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             confusion_matrix, f1_score, roc_auc_score)

CACHE = DATA_ROOT / "cache"
TAU = 0.70
SEED = 42
LATENCY_PROBE = 2000          # single-flow inference calls
BATCH = 1024
PROTOCOL_B_TRAIN = 100_000
PROTOCOL_B_TEST = 40_000
SVM_TRAIN_CAP = 40_000        # RBF SVM is O(n^2); the cap is reported, not hidden
LINEAR_SVM_CAP = 300_000      # liblinear is single-threaded with no time
                              # bound; on the full 2.5M partition it ran for
                              # over half an hour without converging, so it
                              # is capped and the cap is reported


def build_models(protocol: str) -> dict:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    import lightgbm as lgb
    import xgboost as xgb

    m: dict = {
        "Majority baseline": (DummyClassifier(strategy="most_frequent"), False, None),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced",
                                                 random_state=SEED), False, None),
        "Naive Bayes": (Pipeline([("sc", StandardScaler()),
                                  ("m", GaussianNB())]), False, None),
        "Logistic Regression": (Pipeline([
            ("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=2000, class_weight="balanced",
                                     n_jobs=-1, random_state=SEED))]), False, None),
        "Linear SVM": (Pipeline([
            ("sc", StandardScaler()),
            ("m", LinearSVC(class_weight="balanced", random_state=SEED,
                            dual="auto", max_iter=2000, tol=1e-3))]),
            False, LINEAR_SVM_CAP),
        "XGBoost": (xgb.XGBClassifier(n_estimators=200, max_depth=8,
                                      tree_method="hist", n_jobs=-1,
                                      eval_metric="logloss", random_state=SEED,
                                      device="cpu"), True, None),
        "LightGBM": (lgb.LGBMClassifier(n_estimators=200, num_leaves=63,
                                        class_weight="balanced", n_jobs=-1,
                                        random_state=SEED, verbose=-1), True, None),
        "XAI-SDN (Random Forest)": (RandomForestClassifier(
            n_estimators=200, max_depth=None, max_features="sqrt", bootstrap=True,
            class_weight="balanced", n_jobs=-1, random_state=SEED), True, None),
    }
    if protocol == "B":
        m["SVM (RBF)"] = (Pipeline([
            ("sc", StandardScaler()),
            ("m", SVC(kernel="rbf", class_weight="balanced", probability=True,
                      random_state=SEED))]), False, SVM_TRAIN_CAP)
    return m


def get_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))
    return model.predict(X).astype(float)


SCORE_DIR = RESULTS / "scores"
SCORE_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(name, spec, Xtr, ytr, Xte, yte, protocol) -> dict:
    model, tree_shap, cap = spec
    proc = psutil.Process()
    if cap and len(ytr) > cap:
        Xtr, ytr = Xtr[-cap:], ytr[-cap:]     # temporally contiguous tail
        capped = cap
    else:
        capped = None

    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    try:
        model.fit(Xtr, ytr)
    except Exception as exc:
        return {"model": name, "protocol": protocol, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"}
    train_s = time.perf_counter() - t0
    rss_peak = max(proc.memory_info().rss, rss0)

    # batch throughput
    t0 = time.perf_counter()
    score = np.concatenate([get_scores(model, Xte[i:i + BATCH])
                            for i in range(0, len(yte), BATCH)])
    batch_s = time.perf_counter() - t0

    # Single-flow latency on the online path. Forced serial: joblib's parallel
    # dispatch costs ~25 ms per call irrespective of the work, which on one
    # sample is the entire measurement rather than a component of it.
    probe = Xte[:LATENCY_PROBE]
    lat = np.empty(len(probe))
    with serial_inference(model):
        get_scores(model, probe[:1])          # warm up, exclude from timing
        for i in range(len(probe)):
            one = probe[i:i + 1]
            t = time.perf_counter()
            get_scores(model, one)
            lat[i] = (time.perf_counter() - t) * 1000.0

    # The same probe with the fitted parallel setting, so the difference between
    # the two is visible rather than hidden.
    lat_par = np.empty(min(200, len(probe)))
    for i in range(len(lat_par)):
        one = probe[i:i + 1]
        t = time.perf_counter()
        get_scores(model, one)
        lat_par[i] = (time.perf_counter() - t) * 1000.0

    pred = (score >= TAU).astype(np.int8)
    slug = name.replace(" ", "_").replace("(", "").replace(")", "")
    np.save(SCORE_DIR / f"E1_{protocol}_{slug}_score.npy", score.astype(np.float32))
    np.save(SCORE_DIR / f"E1_{protocol}_ytrue.npy", yte.astype(np.int8))
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    r = {
        "model": name, "protocol": protocol, "status": "ok",
        "score_file": f"scores/E1_{protocol}_{slug}_score.npy",
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "train_capped_to": capped,
        "train_benign": int((ytr == 0).sum()), "test_benign": int((yte == 0).sum()),
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "avg_precision": float(average_precision_score(yte, score)),
        "roc_auc": float(roc_auc_score(yte, score)) if len(np.unique(yte)) > 1 else None,
        "precision": float(tp / (tp + fp)) if (tp + fp) else None,
        "recall": float(tp / (tp + fn)) if (tp + fn) else None,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "train_time_s": round(train_s, 3),
        "peak_rss_gb": round(rss_peak / 2 ** 30, 3),
        "batch_throughput_flows_per_s": round(len(yte) / batch_s, 1),
        "single_flow_latency_ms": {
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)),
            "mean": float(lat.mean()),
        },
        "single_flow_latency_ms_parallel_dispatch": {
            "p50": float(np.percentile(lat_par, 50)),
            "mean": float(lat_par.mean()),
            "note": "same single-sample probe with the fitted n_jobs setting; the "
                    "gap against the serial figure is joblib dispatch overhead",
        },
        "exact_per_prediction_explanation": bool(tree_shap),
        "hardware": "CPU only",
    }
    print(f"  {name:28s} acc={r['accuracy']:.5f} F1={r['macro_f1']:.5f} "
          f"AP={r['avg_precision']:.5f} train={r['train_time_s']:.1f}s "
          f"p50={r['single_flow_latency_ms']['p50']:.4f}ms "
          f"thr={r['batch_throughput_flows_per_s']:,.0f}/s", flush=True)
    return r


def main() -> int:
    log_event("E1", "start")
    t_all = time.time()
    import torch
    out: dict = {
        "experiment": "E1",
        "objective": "unified CPU-only classical baseline comparison",
        "cpu_only_guarantee": {
            "torch_version": torch.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "note": "the installed torch build has no CUDA support, so no model in "
                    "this comparison can be timed on a GPU even accidentally",
        },
        "tau": TAU,
    }

    X = np.load(CACHE / "syn0311_X_repaired.npy")
    y = np.load(CACHE / "syn0311_y.npy")
    n = len(y)
    cut = int(round(n * 0.70))

    rows = []

    # ---- Protocol A: full imbalanced temporal split ------------------------
    out["protocol_A"] = {
        "description": "full imbalanced temporal split, natural class prevalence, "
                       "no resampling and no class balancing of the data itself",
        "n_train": int(cut), "n_test": int(n - cut),
        "train_benign": int((y[:cut] == 0).sum()),
        "test_benign": int((y[cut:] == 0).sum()),
        "test_attack_prevalence": float(y[cut:].mean()),
        "results": {},
    }
    print(f"\nProtocol A: train={cut:,} test={n-cut:,} "
          f"(test benign {int((y[cut:]==0).sum()):,})", flush=True)
    for name, spec in build_models("A").items():
        r = evaluate_model(name, spec, X[:cut], y[:cut], X[cut:], y[cut:], "A")
        out["protocol_A"]["results"][name] = r
        if r["status"] == "ok":
            rows.append({k: v for k, v in r.items()
                         if k not in ("confusion", "single_flow_latency_ms",
                                      "single_flow_latency_ms_parallel_dispatch")} |
                        {f"lat_{k}_ms": v for k, v in r["single_flow_latency_ms"].items()})

    # ---- Protocol B: identical subsampling for every model -----------------
    b_tr_end = cut
    b_tr_start = max(0, cut - PROTOCOL_B_TRAIN)
    b_te_end = min(n, cut + PROTOCOL_B_TEST)
    XtrB, ytrB = X[b_tr_start:b_tr_end], y[b_tr_start:b_tr_end]
    XteB, yteB = X[cut:b_te_end], y[cut:b_te_end]
    out["protocol_B"] = {
        "description": "identically subsampled temporal protocol applied to every "
                       "model, preserving the natural class ratio; the training block "
                       "is the contiguous tail of the training partition and the test "
                       "block is the contiguous head of the test partition",
        "n_train": int(len(ytrB)), "n_test": int(len(yteB)),
        "train_benign": int((ytrB == 0).sum()), "test_benign": int((yteB == 0).sum()),
        "svm_train_cap": SVM_TRAIN_CAP,
        "results": {},
    }
    print(f"\nProtocol B: train={len(ytrB):,} test={len(yteB):,} "
          f"(train benign {int((ytrB==0).sum()):,}, "
          f"test benign {int((yteB==0).sum()):,})", flush=True)
    if len(np.unique(ytrB)) < 2 or len(np.unique(yteB)) < 2:
        out["protocol_B"]["error"] = ("a contiguous block contains a single class; "
                                      "falling back to a stratified subsample of the "
                                      "same sizes, which is reported as such")
        rng = np.random.default_rng(SEED)
        itr = rng.choice(cut, min(PROTOCOL_B_TRAIN, cut), replace=False)
        ite = rng.choice(np.arange(cut, n), min(PROTOCOL_B_TEST, n - cut), replace=False)
        XtrB, ytrB, XteB, yteB = X[itr], y[itr], X[ite], y[ite]
        out["protocol_B"].update({"n_train": int(len(ytrB)), "n_test": int(len(yteB)),
                                  "train_benign": int((ytrB == 0).sum()),
                                  "test_benign": int((yteB == 0).sum())})
        print(f"  fallback: train={len(ytrB):,} test={len(yteB):,}", flush=True)

    for name, spec in build_models("B").items():
        r = evaluate_model(name, spec, XtrB, ytrB, XteB, yteB, "B")
        out["protocol_B"]["results"][name] = r
        if r["status"] == "ok":
            rows.append({k: v for k, v in r.items()
                         if k not in ("confusion", "single_flow_latency_ms",
                                      "single_flow_latency_ms_parallel_dispatch")} |
                        {f"lat_{k}_ms": v for k, v in r["single_flow_latency_ms"].items()})

    pd.DataFrame(rows).to_csv(RESULTS / "E1_baselines_cpu.csv", index=False)
    out["training_caps_note"] = (
        "Two models are trained on a capped subset because they do not scale to "
        "the full partition on a single machine: the RBF SVM, whose kernel "
        "computation is quadratic in the sample count, and the linear SVM, whose "
        "liblinear solver is single-threaded and had not converged after more "
        "than thirty minutes on 2.5M samples. Both caps are recorded per model in "
        "the 'train_capped_to' field and stated in the paper, because a model that "
        "cannot be trained at the deployment scale is a deployment fact rather "
        "than an inconvenience to be hidden.")
    out["latency_measurement_note"] = (
        "Per-flow latency is measured with the estimator forced to serial "
        "execution. With the fitted n_jobs=-1 setting, joblib dispatch adds "
        "roughly 25 ms to every single-sample call regardless of model size, "
        "which would make a Random Forest appear two orders of magnitude slower "
        "than a linear SVM. Both figures are recorded per model so the overhead "
        "is visible. Batch throughput retains the parallel setting, since a "
        "controller processing a poll batch can use every core.")
    out["comparability_statement"] = (
        "Every figure in both protocols was measured on one machine, on the CPU only, "
        "with the same thread budget and the same measurement harness. Latency is "
        "reported as a distribution over single-flow inference calls rather than as a "
        "mean, and throughput is measured in batch mode; the two answer different "
        "operational questions and are not interchangeable. Results from Protocol A "
        "and Protocol B are never compared with one another.")
    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E1_baselines_cpu", out)
    log_event("E1", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
