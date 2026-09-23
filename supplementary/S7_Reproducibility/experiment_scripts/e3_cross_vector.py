"""E3 - Cross-vector and cross-day generalization.

One may reasonably call SYN-only evaluation "methodologically bankrupt"; It has been argued that it "provides limited evidence of generalization"; It has been argued that the
generalization language outruns the evidence. All three are answered the same
way: train on SYN and test on attack vectors and a capture day the model has
never seen, and report whatever happens.

Conditions:
  A. zero-shot  - train on 03-11 SYN, test on each other 03-11 vector
  B. leave-one-vector-out - train on all vectors but one, test on the held-out one
  C. cross-day  - train on 03-11 SYN, test on 01-12 SYN (different capture day,
                  different background traffic, same attack class)

Every result carries bootstrap 95% confidence intervals on FPR and FNR.
Degradation is the expected outcome and is reported as a finding.
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

CACHE = DATA_ROOT / "cache"
RF_KW = dict(n_estimators=200, max_depth=None, max_features="sqrt", bootstrap=True,
             class_weight="balanced", n_jobs=-1, random_state=42)
TAU = 0.70
N_BOOT = 500
MAX_ROWS_PER_PARTITION = 1_200_000     # keeps every partition on an equal footing

PARTITIONS = {
    "SYN_0311":     "CICDDoS2019/03-11/Syn.csv",
    "LDAP_0311":    "CICDDoS2019/03-11/LDAP.csv",
    "NetBIOS_0311": "CICDDoS2019/03-11/NetBIOS.csv",
    "Portmap_0311": "CICDDoS2019/03-11/Portmap.csv",
    "SYN_0112":     "CICDDoS2019/01-12/Syn.csv",
}


def scores(y, proba, tau=TAU) -> dict:
    pred = (proba >= tau).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "avg_precision": float(average_precision_score(y, proba))
        if len(np.unique(y)) > 1 else None,
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None,
        "precision": float(tp / (tp + fp)) if (tp + fp) else None,
        "recall": float(tp / (tp + fn)) if (tp + fn) else None,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y)), "n_benign": int((y == 0).sum()), "n_attack": int((y == 1).sum()),
    }


def boot_ci(y, proba, tau=TAU, n_boot=N_BOOT, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    pred = (proba >= tau).astype(np.int8)
    n = len(y)
    fprs, fnrs, f1s = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yt, yp = y[i], pred[i]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        if (fp + tn):
            fprs.append(fp / (fp + tn))
        if (fn + tp):
            fnrs.append(fn / (fn + tp))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))

    def ci(a):
        if not a:
            return None
        a = np.asarray(a, float)
        return {"mean": float(a.mean()), "lo95": float(np.percentile(a, 2.5)),
                "hi95": float(np.percentile(a, 97.5))}
    return {"fpr": ci(fprs), "fnr": ci(fnrs), "macro_f1": ci(f1s), "n_boot": n_boot}


def load_built(key: str, rel: str) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Load, clean, temporally order and featurise one partition, with caching."""
    xc = CACHE / f"{key}_X.npy"
    yc = CACHE / f"{key}_y.npy"
    sc = CACHE / f"{key}_stats.json"
    nc = CACHE / f"{key}_names.json"
    if xc.exists() and yc.exists():
        return (np.load(xc), np.load(yc), json.loads(nc.read_text()),
                json.loads(sc.read_text()))
    path = DATA_ROOT / rel
    print(f"  building {key} from {path.name}", flush=True)
    df = D.load_partition(path)
    df, st = D.clean_partition(df)
    df = D.sort_temporal(df)
    if len(df) > MAX_ROWS_PER_PARTITION:
        # Keep a contiguous temporal block so the window semantics stay valid,
        # and take it from the END of the capture. E18 establishes that benign
        # traffic in these captures is concentrated in the final third, so
        # truncating from the head would discard almost the entire benign class
        # and make the transfer target degenerate.
        df = df.iloc[-MAX_ROWS_PER_PARTITION:].reset_index(drop=True)
        st["truncated_to"] = MAX_ROWS_PER_PARTITION
        st["truncation_side"] = "tail (benign traffic is concentrated late)"
    y = D.to_binary_label(df["Label"])
    X, names = D.build_matrix(df, repaired=True)
    st["n_final"] = int(len(y))
    st["benign"] = int((y == 0).sum())
    st["attack"] = int((y == 1).sum())
    np.save(xc, X); np.save(yc, y)
    sc.write_text(json.dumps(st, indent=2)); nc.write_text(json.dumps(names, indent=2))
    return X, y, names, st


def main() -> int:
    log_event("E3", "start")
    t_all = time.time()
    out: dict = {"experiment": "E3",
                 "objective": "cross-vector and cross-day generalization",
                 "tau": TAU, "max_rows_per_partition": MAX_ROWS_PER_PARTITION,
                 "truncation_policy": "contiguous tail of each capture, because benign traffic is concentrated late (see E18)"}

    data = {}
    out["partitions"] = {}
    for key, rel in PARTITIONS.items():
        if not (DATA_ROOT / rel).exists():
            print(f"  {key}: FILE MISSING, skipped", flush=True)
            out["partitions"][key] = {"status": "missing"}
            continue
        X, y, names, st = load_built(key, rel)
        data[key] = (X, y, names)
        out["partitions"][key] = {"status": "ok", "n": int(len(y)),
                                  "benign": int((y == 0).sum()),
                                  "attack": int((y == 1).sum()),
                                  "benign_pct": round(100 * float((y == 0).mean()), 4),
                                  "cleaning": st}
        print(f"  {key}: n={len(y):,} benign={int((y==0).sum()):,} "
              f"attack={int((y==1).sum()):,}", flush=True)

    if "SYN_0311" not in data:
        out["error"] = "SYN_0311 unavailable"
        save_result("E3_cross_vector", out)
        return 1

    Xs, ys, names = data["SYN_0311"]
    cut = int(round(len(ys) * 0.70))
    print("\nA. zero-shot transfer from SYN_0311", flush=True)
    t0 = time.time()
    src = RandomForestClassifier(**RF_KW).fit(Xs[:cut], ys[:cut])
    print(f"  source model fit in {time.time()-t0:.1f}s", flush=True)

    rows = []
    out["zero_shot"] = {}
    for key, (X, y, nm) in data.items():
        if nm != names:
            out["zero_shot"][key] = {"error": "feature-name mismatch"}
            continue
        Xe, ye = (X[cut:], ys[cut:]) if key == "SYN_0311" else (X, y)
        if len(np.unique(ye)) < 2:
            out["zero_shot"][key] = {"skipped": "target has a single class",
                                     "n": int(len(ye)),
                                     "n_benign": int((ye == 0).sum())}
            print(f"  {key}: skipped (single class)", flush=True)
            continue
        pr = src.predict_proba(Xe)[:, 1]
        r = scores(ye, pr)
        r["bootstrap_ci"] = boot_ci(ye, pr)
        r["condition"] = "in-distribution held-out" if key == "SYN_0311" else "zero-shot"
        out["zero_shot"][key] = r
        rows.append({"condition": r["condition"], "train": "SYN_0311", "test": key,
                     **{k: r[k] for k in ("accuracy", "macro_f1", "avg_precision",
                                          "roc_auc", "precision", "recall",
                                          "fpr", "fnr", "n", "n_benign", "n_attack")},
                     "fpr_lo95": (r["bootstrap_ci"]["fpr"] or {}).get("lo95"),
                     "fpr_hi95": (r["bootstrap_ci"]["fpr"] or {}).get("hi95"),
                     "fnr_lo95": (r["bootstrap_ci"]["fnr"] or {}).get("lo95"),
                     "fnr_hi95": (r["bootstrap_ci"]["fnr"] or {}).get("hi95")})
        print(f"  -> {key}: acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} "
              f"FPR={r['fpr']} FNR={r['fnr']}", flush=True)

    # ---- B. leave-one-vector-out -------------------------------------------
    print("\nB. leave-one-vector-out", flush=True)
    out["leave_one_out"] = {}
    keys_0311 = [k for k in data if k.endswith("_0311")]
    for held in keys_0311:
        tr_keys = [k for k in keys_0311 if k != held]
        Xtr = np.vstack([data[k][0] for k in tr_keys])
        ytr = np.concatenate([data[k][1] for k in tr_keys])
        Xte, yte, _ = data[held]
        if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
            out["leave_one_out"][held] = {"skipped": "single-class partition"}
            print(f"  hold out {held}: skipped", flush=True)
            continue
        t0 = time.time()
        m = RandomForestClassifier(**RF_KW).fit(Xtr, ytr)
        pr = m.predict_proba(Xte)[:, 1]
        r = scores(yte, pr)
        r["bootstrap_ci"] = boot_ci(yte, pr)
        r["train_partitions"] = tr_keys
        r["fit_seconds"] = round(time.time() - t0, 1)
        out["leave_one_out"][held] = r
        rows.append({"condition": "leave-one-vector-out",
                     "train": "+".join(tr_keys), "test": held,
                     **{k: r[k] for k in ("accuracy", "macro_f1", "avg_precision",
                                          "roc_auc", "precision", "recall",
                                          "fpr", "fnr", "n", "n_benign", "n_attack")},
                     "fpr_lo95": (r["bootstrap_ci"]["fpr"] or {}).get("lo95"),
                     "fpr_hi95": (r["bootstrap_ci"]["fpr"] or {}).get("hi95"),
                     "fnr_lo95": (r["bootstrap_ci"]["fnr"] or {}).get("lo95"),
                     "fnr_hi95": (r["bootstrap_ci"]["fnr"] or {}).get("hi95")})
        print(f"  hold out {held}: acc={r['accuracy']:.4f} "
              f"macroF1={r['macro_f1']:.4f} FPR={r['fpr']} FNR={r['fnr']}", flush=True)

    # ---- C. cross-day -------------------------------------------------------
    print("\nC. cross-day 03-11 -> 01-12", flush=True)
    if "SYN_0112" in data:
        Xd, yd, _ = data["SYN_0112"]
        if len(np.unique(yd)) > 1:
            pr = src.predict_proba(Xd)[:, 1]
            r = scores(yd, pr)
            r["bootstrap_ci"] = boot_ci(yd, pr)
            out["cross_day"] = r
            rows.append({"condition": "cross-day", "train": "SYN_0311",
                         "test": "SYN_0112",
                         **{k: r[k] for k in ("accuracy", "macro_f1", "avg_precision",
                                              "roc_auc", "precision", "recall",
                                              "fpr", "fnr", "n", "n_benign",
                                              "n_attack")},
                         "fpr_lo95": (r["bootstrap_ci"]["fpr"] or {}).get("lo95"),
                         "fpr_hi95": (r["bootstrap_ci"]["fpr"] or {}).get("hi95"),
                         "fnr_lo95": (r["bootstrap_ci"]["fnr"] or {}).get("lo95"),
                         "fnr_hi95": (r["bootstrap_ci"]["fnr"] or {}).get("hi95")})
            print(f"  acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} "
                  f"FPR={r['fpr']} FNR={r['fnr']}", flush=True)
        else:
            out["cross_day"] = {"skipped": "single-class target"}

    pd.DataFrame(rows).to_csv(RESULTS / "E3_transfer_matrix.csv", index=False)
    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E3_cross_vector", out)
    log_event("E3", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
