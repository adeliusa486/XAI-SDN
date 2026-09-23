"""E23 - back-to-back latency check for the two configurations that matter.

E22 sweeps 24 configurations and fits a model immediately before probing each
one, so its absolute latencies sit above those of E1, which probes a model fitted
much earlier. Ratios within E22 are sound because every row is measured the same
way, but the section's argument is a comparison against the baseline table, and
two tables that disagree on the same quantity invite a reader to distrust both.

This run removes the question. It fits exactly two forests, the configuration
inherited from the submitted work and the one the frontier selects, then probes
them alternately in the same process with the same harness so that any machine
state affects both equally. Interleaving matters: measuring one and then the
other lets a thermal or scheduling drift land entirely on one of them.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))

from paths import DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402
from serial import serial_inference  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

CACHE = DATA_ROOT / "cache"
SEED = 42
TAU = 0.70
LATENCY_PROBE = 2000
ROUNDS = 5             # interleaved passes over both models


def size_mb(clf) -> float:
    buf = io.BytesIO()
    joblib.dump(clf, buf, compress=0)
    return round(buf.tell() / 2 ** 20, 3)


def probe(clf, rows: np.ndarray) -> np.ndarray:
    lat = np.empty(len(rows))
    with serial_inference(clf):
        clf.predict_proba(rows[:1])
        for i in range(len(rows)):
            one = rows[i:i + 1]
            t = time.perf_counter()
            clf.predict_proba(one)
            lat[i] = (time.perf_counter() - t) * 1000.0
    return lat


def quality(clf, X, y) -> dict:
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= TAU).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
            "false_positives": int(fp), "false_negatives": int(fn)}


def main() -> int:
    log_event("E23", "start")
    t_all = time.time()

    sweep = RESULTS / "E22_model_size_pareto.json"
    if not sweep.exists():
        print("E22 has not run; nothing to verify")
        return 1
    e22 = json.loads(sweep.read_text(encoding="utf-8"))
    sel = e22["selected_configuration"]
    chosen = {"max_depth": None if sel["max_depth"] == -1 else int(sel["max_depth"]),
              "n_estimators": int(sel["n_estimators"])}
    baseline = {"max_depth": None, "n_estimators": 200}
    print(f"baseline {baseline} against selected {chosen}", flush=True)

    X = np.load(CACHE / "syn0311_X_repaired.npy", mmap_mode="r")
    y = np.load(CACHE / "syn0311_y.npy")
    cut = int(len(y) * 0.70)
    vcut = int(cut * 0.9)
    Xtr, ytr = np.asarray(X[:vcut]), y[:vcut]
    Xte, yte = np.asarray(X[cut:]), y[cut:]
    rows = Xte[:LATENCY_PROBE]

    models = {}
    for name, kw in (("baseline", baseline), ("selected", chosen)):
        t0 = time.perf_counter()
        models[name] = RandomForestClassifier(
            max_features="sqrt", bootstrap=True, class_weight="balanced",
            n_jobs=-1, random_state=SEED, **kw).fit(Xtr, ytr)
        print(f"  fitted {name} in {time.perf_counter() - t0:.0f}s, "
              f"{size_mb(models[name]):.2f} MB", flush=True)

    # Let the fitting thread pool settle before timing anything.
    time.sleep(5)

    lat = {k: [] for k in models}
    for r in range(ROUNDS):
        for name, clf in models.items():
            lat[name].append(probe(clf, rows))
        print(f"  round {r + 1}/{ROUNDS}: "
              + "  ".join(f"{k} p50={np.percentile(np.concatenate(v), 50):.3f} ms"
                          for k, v in lat.items()), flush=True)

    out = {
        "experiment": "E23",
        "objective": "directly comparable latency for the baseline and selected "
                     "configurations",
        "method": ("both forests are fitted in one process and probed alternately "
                   "over five rounds, so machine state affects both equally"),
        "latency_probe": LATENCY_PROBE, "rounds": ROUNDS, "tau": TAU,
        "n_train": int(vcut), "n_test": int(len(yte)),
        "configurations": {},
    }
    for name, clf in models.items():
        a = np.concatenate(lat[name])
        out["configurations"][name] = {
            "max_depth": (baseline if name == "baseline" else chosen)["max_depth"],
            "n_estimators": (baseline if name == "baseline" else chosen)["n_estimators"],
            "model_size_mb": size_mb(clf),
            "mean_leaves_per_tree": float(np.mean([t.tree_.n_leaves
                                                   for t in clf.estimators_])),
            "single_flow_latency_ms": {
                "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)),
                "mean": float(a.mean()), "n": int(len(a))},
            **quality(clf, Xte, yte),
        }

    b = out["configurations"]["baseline"]
    s = out["configurations"]["selected"]
    out["speedup"] = round(b["single_flow_latency_ms"]["p50"]
                           / s["single_flow_latency_ms"]["p50"], 2)
    out["size_reduction"] = round(b["model_size_mb"] / s["model_size_mb"], 1)
    out["macro_f1_change"] = round(s["macro_f1"] - b["macro_f1"], 6)
    out["interpretation"] = (
        f"Measured in one process with interleaved probes, the selected "
        f"configuration is {out['speedup']:.1f}x faster per flow and "
        f"{out['size_reduction']:.0f}x smaller, for a macro F1 change of "
        f"{out['macro_f1_change']:+.4f}.")
    print("\n" + out["interpretation"], flush=True)

    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E23_latency_verification", out)
    log_event("E23", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
