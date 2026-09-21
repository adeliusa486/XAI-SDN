"""E17 - Does the entropy augmentation actually contribute? (R6.6, R6.1, R7.1, R2.1)

E5 produced two results that together question the paper's central claim, and
neither can be interpreted from that experiment alone.

First, under the chronological protocol a model using only the 80 exported flow
statistics reached macro F1 0.9812, slightly ABOVE the full 88-dimensional model
at 0.9788. A single run cannot distinguish that from noise.

Second, recomputing the entropy features over a shuffled arrival order collapsed
every one of them from an AUC near 0.99 to near 0.55. That looks damning, but
the diagnostic as run was confounded: shuffling the rows changed the entropy
computation AND the train/test split at the same time, and we already know the
split alone moves macro F1 by two points. The comparison therefore cannot
attribute the collapse to either cause.

This experiment separates them.

  A  natural      entropy over the true chronological arrival order (the system)
  B  scrambled    entropy computed over a random permutation, then each flow's
                  value returned to its chronological position, so the SPLIT is
                  identical to A and only the temporal coherence of the entropy
                  features is destroyed
  C  cic_only     the 80 exported statistics, no entropy features
  D  entropy_only the 8 entropy features alone

All four use the same chronological split and the same hyperparameters, and each
is run over five seeds so the differences can be tested rather than eyeballed.

Reading the outcome:
  A > C significantly            the entropy augmentation contributes
  A ~= C                         it does not contribute in distribution
  A > B significantly            what it contributes is temporal structure
  B ~= C                         entropy values stripped of arrival order are inert
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

from scipy.stats import wilcoxon  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             confusion_matrix, f1_score, roc_auc_score)

CACHE = DATA_ROOT / "cache"
RF_KW = dict(n_estimators=200, max_depth=None, max_features="sqrt", bootstrap=True,
             class_weight="balanced", n_jobs=-1)
TAU = 0.70
SEEDS = [42, 123, 456, 789, 1024]


def evaluate(X, y, cut, seed, tag) -> dict:
    kw = dict(RF_KW); kw["random_state"] = seed
    t0 = time.time()
    clf = RandomForestClassifier(**kw).fit(X[:cut], y[:cut])
    proba = clf.predict_proba(X[cut:])[:, 1]
    pred = (proba >= TAU).astype(np.int8)
    yte = y[cut:]
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    r = {
        "variant": tag, "seed": seed, "n_features": int(X.shape[1]),
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "avg_precision": float(average_precision_score(yte, proba)),
        "roc_auc": float(roc_auc_score(yte, proba)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "fit_seconds": round(time.time() - t0, 1),
    }
    print(f"  [{tag:12s} seed={seed:>4}] d={r['n_features']:>2} "
          f"acc={r['accuracy']:.6f} F1={r['macro_f1']:.6f} "
          f"AP={r['avg_precision']:.6f} FPR={r['fpr']:.6f}", flush=True)
    return r


def main() -> int:
    log_event("E17", "start")
    t_all = time.time()
    out: dict = {"experiment": "E17",
                 "objective": "isolate the contribution of the entropy augmentation",
                 "seeds": SEEDS, "tau": TAU}

    X = np.load(CACHE / "syn0311_X_repaired.npy")
    y = np.load(CACHE / "syn0311_y.npy")
    names = json.loads((CACHE / "syn0311_feature_names_repaired.json").read_text())
    cut = int(round(len(y) * 0.70))
    ent_start = len(names) - 8
    cic_names = names[:ent_start]
    print(f"loaded {X.shape}, chronological cut at {cut:,}", flush=True)

    # ---- build variant B: entropy over a scrambled order, values returned
    #      to their chronological positions so the split is unchanged ---------
    scrambled_path = CACHE / "syn0311_H_scrambled.npy"
    if scrambled_path.exists():
        H_scr = np.load(scrambled_path)
        print("  loaded cached scrambled-order entropy", flush=True)
    else:
        print("Computing entropy over a scrambled arrival order...", flush=True)
        t0 = time.time()
        rng = np.random.default_rng(2026)
        perm = rng.permutation(len(y))
        meta = pd.read_parquet(CACHE / "syn0311_meta.parquet")
        df_scr = pd.DataFrame({
            "Source_IP": meta["Source_IP"].to_numpy()[perm],
            "Destination_IP": meta["Destination_IP"].to_numpy()[perm],
            "Destination_Port": X[perm, cic_names.index("Destination_Port")],
            "Protocol": X[perm, cic_names.index("Protocol")],
            "Packet_Length_Mean": X[perm, cic_names.index("Packet_Length_Mean")],
            "Flow_IAT_Mean": X[perm, cic_names.index("Flow_IAT_Mean")],
            "SYN_Flag_Count": X[perm, cic_names.index("SYN_Flag_Count")],
            "Source_Port": X[perm, cic_names.index("Source_Port")],
        })
        streams, _ = D.entropy_streams(df_scr, repaired=True)
        H_perm = D.rolling_entropy(streams, window=D.DEFAULT_WINDOW).astype(np.float32)
        # return each flow's scrambled-order entropy to its chronological slot
        H_scr = np.empty_like(H_perm)
        H_scr[perm] = H_perm
        np.save(scrambled_path, H_scr)
        print(f"  {time.time()-t0:.1f}s", flush=True)

    variants = {
        "natural":      X,
        "scrambled":    np.hstack([X[:, :ent_start], H_scr]),
        "cic_only":     X[:, :ent_start],
        "entropy_only": X[:, ent_start:],
    }
    out["variant_dims"] = {k: int(v.shape[1]) for k, v in variants.items()}
    out["variant_description"] = {
        "natural": "80 flow statistics + 8 entropy features over the true arrival order",
        "scrambled": "80 flow statistics + 8 entropy features computed over a random "
                     "permutation and returned to chronological positions; the split "
                     "is identical to 'natural'",
        "cic_only": "80 exported flow statistics only",
        "entropy_only": "8 entropy features only",
    }

    rows = []
    for tag, Xv in variants.items():
        for seed in SEEDS:
            rows.append(evaluate(Xv, y, cut, seed, tag))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "E17_variants.csv", index=False)

    summary = {}
    for tag in variants:
        sub = df[df.variant == tag]
        summary[tag] = {m: {"mean": float(sub[m].mean()),
                            "std": float(sub[m].std(ddof=1)),
                            "min": float(sub[m].min()), "max": float(sub[m].max())}
                        for m in ("accuracy", "macro_f1", "avg_precision", "fpr", "fnr")}
    out["summary"] = summary
    print("\nmacro F1 by variant (mean +/- std over 5 seeds):", flush=True)
    for tag in variants:
        s = summary[tag]["macro_f1"]
        print(f"  {tag:12s} d={out['variant_dims'][tag]:>2}  "
              f"{100*s['mean']:.4f} +/- {100*s['std']:.4f}", flush=True)

    # ---- paired tests over the shared seeds --------------------------------
    def paired(a, b):
        va = df[df.variant == a].sort_values("seed")["macro_f1"].to_numpy()
        vb = df[df.variant == b].sort_values("seed")["macro_f1"].to_numpy()
        diff = va - vb
        try:
            stat, p = wilcoxon(va, vb)
            p = float(p)
        except ValueError:
            stat, p = float("nan"), float("nan")
        return {"pair": f"{a} vs {b}",
                "mean_macro_f1_difference": float(diff.mean()),
                "difference_per_seed": [float(x) for x in diff],
                "wilcoxon_p": p,
                "a_better": bool(diff.mean() > 0)}

    tests = [paired("natural", "cic_only"),
             paired("natural", "scrambled"),
             paired("scrambled", "cic_only"),
             paired("natural", "entropy_only"),
             paired("cic_only", "entropy_only")]
    out["paired_tests"] = tests
    print("\npaired comparisons (macro F1):", flush=True)
    for t in tests:
        print(f"  {t['pair']:28s} delta={100*t['mean_macro_f1_difference']:+.4f} pp  "
              f"p={t['wilcoxon_p']:.4f}", flush=True)

    # ---- verdict -----------------------------------------------------------
    nat_cic = tests[0]
    nat_scr = tests[1]
    alpha = 0.05
    if nat_cic["mean_macro_f1_difference"] > 0 and nat_cic["wilcoxon_p"] < alpha:
        verdict = ("The entropy augmentation improves in-distribution detection "
                   "under the chronological protocol, and the improvement is "
                   "statistically significant over five seeds.")
    elif nat_cic["mean_macro_f1_difference"] <= 0:
        verdict = ("The entropy augmentation does NOT improve in-distribution "
                   "detection under the chronological protocol. The flow statistics "
                   "alone match or exceed the full feature set. The claim that the "
                   "entropy features improve accuracy is not supported in this "
                   "setting and must be restated.")
    else:
        verdict = ("The entropy augmentation shows a positive but statistically "
                   "insignificant in-distribution difference over five seeds.")
    out["verdict_in_distribution"] = verdict

    if nat_scr["mean_macro_f1_difference"] > 0 and nat_scr["wilcoxon_p"] < alpha:
        out["verdict_temporal_coherence"] = (
            "Destroying the temporal coherence of the entropy features while holding "
            "the split fixed degrades performance, so whatever the features "
            "contribute depends on arrival order rather than on per-flow values.")
    else:
        out["verdict_temporal_coherence"] = (
            "Destroying the temporal coherence of the entropy features while holding "
            "the split fixed does not measurably change performance. The collapse in "
            "single-feature AUC observed in E5 therefore reflects the loss of "
            "arrival-order information in the features themselves rather than a "
            "change in what the classifier can do with the full representation.")

    print("\nVERDICT (in distribution): " + verdict, flush=True)
    print("VERDICT (temporal coherence): " + out["verdict_temporal_coherence"],
          flush=True)
    print("\nNote: this experiment measures in-distribution contribution only. "
          "Whether the entropy features help under transfer to other attack "
          "vectors, capture days or corpora is answered by E3 and E2.", flush=True)

    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E17_entropy_contribution", out)
    log_event("E17", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
