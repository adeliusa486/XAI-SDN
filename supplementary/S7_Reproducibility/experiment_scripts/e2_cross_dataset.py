"""E2 - Cross-dataset validation with confidence intervals.

This experiment provides cross-dataset validation on InSDN and CIC-IDS2017/2018 with
FPR/FNR confidence intervals. This experiment performs validation beyond CIC-DDoS2019.
It has been argued that the generalization claims are not licensed by the evidence.

This also replaces the repository artefact `insdn_transfer.json`, which is
flagged `"synthetic": true` and reports accuracy 1.0 on every scenario. Nothing
from that file is used here, and it is deleted from the repository in Phase 9.

Because the three corpora were produced by different CICFlowMeter versions, a
transfer is only meaningful on the intersection of their schemas. The mapping is
computed explicitly, published as a table, and applied symmetrically: a feature
absent from the target is dropped from the source as well, so the source model
is trained on exactly the columns it will be asked to predict from.

Known asymmetries, stated rather than hidden:
  * CIC-IDS2017 ISCX exports carry no Source IP, Destination IP, Source Port or
    Timestamp, so only five of the eight entropy features can be computed there.
  * The available InSDN export is already binarised into attack and benign and
    mixes several attack families, so transfer to it measures attack detection
    in general rather than DDoS detection specifically.
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
from profile_datasets import canonical  # noqa: E402
from paths import DATA_META, DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             confusion_matrix, f1_score, roc_auc_score)

CACHE = DATA_ROOT / "cache"
RF_KW = dict(n_estimators=200, max_depth=None, max_features="sqrt", bootstrap=True,
             class_weight="balanced", n_jobs=-1, random_state=42)
TAU = 0.70
N_BOOT = 1000
MAX_SOURCE_ROWS = 1_500_000

ENT_REQUIRES = {
    "H_src_ip": "Source_IP", "H_dst_ip": "Destination_IP",
    "H_dst_port": "Destination_Port", "H_proto": "Protocol",
    "H_pkt_len": "Packet_Length_Mean", "H_iat": "Flow_IAT_Mean",
    "H_tcp_flags": "SYN_Flag_Count", "H_src_port": "Source_Port",
}


def load_canonical(path: Path, label_col_hint: str | None = None,
                   nrows: int | None = None) -> pd.DataFrame:
    """Read any of the three corpora into canonical column names."""
    head = pd.read_csv(path, nrows=0)
    ren = {c: canonical(c) for c in head.columns}
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df = df.rename(columns=ren)
    if "target" in df.columns and "Label" not in df.columns:
        df = df.rename(columns={"target": "Label"})
    return df


def clean_generic(df: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, dict]:
    st = {"rows_in": int(len(df))}
    for c in feats:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=feats)
    st["dropped_nan_inf"] = int(before - len(df))
    before = len(df)
    df = df.drop_duplicates(subset=feats + ["Label"])
    st["dropped_duplicate"] = int(before - len(df))
    st["rows_out"] = int(len(df))
    return df.reset_index(drop=True), st


def binary_label(s: pd.Series) -> np.ndarray:
    v = s.astype(str).str.strip()
    if set(v.unique()) <= {"0", "1"}:
        return v.astype(int).astype(np.int8).values
    return (~v.str.upper().str.startswith("BENIGN")).astype(np.int8).values


def entropy_block(df: pd.DataFrame, ent_names: list[str]) -> np.ndarray:
    streams = []
    n = len(df)
    for e in ent_names:
        src = ENT_REQUIRES[e]
        col = df[src]
        if e in ("H_src_ip", "H_dst_ip"):
            streams.append(list(col.astype(str).values))
        elif e == "H_pkt_len":
            v = np.nan_to_num(col.to_numpy(dtype=np.float64))
            streams.append(list((v // D.BIN_PKT_LEN).astype(np.int64)))
        elif e == "H_iat":
            v = np.nan_to_num(col.to_numpy(dtype=np.float64))
            streams.append(list((v // D.BIN_IAT).astype(np.int64)))
        else:
            streams.append(list(col.to_numpy(dtype=np.float64).astype(np.int64)))
    return D.rolling_entropy(streams, window=D.DEFAULT_WINDOW).astype(np.float32)


def scores(y, proba, tau=TAU) -> dict:
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
        "n": int(len(y)), "n_benign": int((y == 0).sum()), "n_attack": int((y == 1).sum()),
    }


def boot_ci(y, proba, tau=TAU, n_boot=N_BOOT, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    pred = (proba >= tau).astype(np.int8)
    n = len(y)
    acc, f1s, fprs, fnrs = [], [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yt, yp = y[i], pred[i]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        acc.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        if (fp + tn):
            fprs.append(fp / (fp + tn))
        if (fn + tp):
            fnrs.append(fn / (fn + tp))

    def ci(a):
        if not a:
            return None
        a = np.asarray(a, float)
        return {"mean": float(a.mean()), "lo95": float(np.percentile(a, 2.5)),
                "hi95": float(np.percentile(a, 97.5))}
    return {"accuracy": ci(acc), "macro_f1": ci(f1s), "fpr": ci(fprs),
            "fnr": ci(fnrs), "n_boot": n_boot}


def main() -> int:
    log_event("E2", "start")
    t_all = time.time()
    out: dict = {"experiment": "E2",
                 "objective": "cross-dataset validation with confidence intervals",
                 "tau": TAU,
                 "repudiated_artifact": (
                     "model/artifacts/insdn_transfer.json in the repository is flagged "
                     "synthetic:true and reports accuracy 1.0; it is not used here and "
                     "is removed from the repository")}

    targets = {
        "InSDN": {"path": DATA_ROOT / "InSDN" / "Dataset.csv",
                  "caveat": "the available export is pre-binarised and mixes several "
                            "attack families, so this measures attack detection in "
                            "general rather than DDoS detection specifically"},
        "CICIDS2017_DDoS": {"path": DATA_ROOT / "CICIDS2017" /
                            "Friday-WorkingHours-Afternoon-DDos.csv",
                            "caveat": "ISCX exports carry no address, source-port or "
                                      "timestamp fields, so three of the eight entropy "
                                      "features cannot be computed"},
    }

    print("loading source corpus (CIC-DDoS2019 03-11 SYN)...", flush=True)
    src_df = D.load_partition(DATA_ROOT / "CICDDoS2019" / "03-11" / "Syn.csv")
    src_df, src_stats = D.clean_partition(src_df)
    src_df = D.sort_temporal(src_df)
    if len(src_df) > MAX_SOURCE_ROWS:
        src_df = src_df.iloc[-MAX_SOURCE_ROWS:].reset_index(drop=True)
        src_stats["truncated_to"] = MAX_SOURCE_ROWS
    out["source"] = {"name": "CICDDoS2019_0311_Syn", "cleaning": src_stats,
                     "n": int(len(src_df))}
    src_cols = set(src_df.columns)

    out["transfers"] = {}
    rows = []
    schema_rows = []

    for tname, meta in targets.items():
        path = meta["path"]
        if not path.exists():
            out["transfers"][tname] = {"status": "missing"}
            continue
        print(f"\n=== transfer to {tname} ===", flush=True)
        tdf = load_canonical(path)
        t_cols = set(tdf.columns)

        shared_cic = [c for c in D.CIC_FEATURE_NAMES if c in src_cols and c in t_cols]
        shared_ent = [e for e, need in ENT_REQUIRES.items()
                      if need in src_cols and need in t_cols]
        missing_ent = [e for e in ENT_REQUIRES if e not in shared_ent]
        print(f"  shared schema: {len(shared_cic)} CIC + {len(shared_ent)} entropy "
              f"= {len(shared_cic)+len(shared_ent)} dims; "
              f"unavailable entropy: {missing_ent}", flush=True)

        for c in D.CIC_FEATURE_NAMES:
            schema_rows.append({"target": tname, "feature": c,
                                "in_source": c in src_cols, "in_target": c in t_cols,
                                "used": c in shared_cic})
        for e in ENT_REQUIRES:
            schema_rows.append({"target": tname, "feature": e,
                                "in_source": ENT_REQUIRES[e] in src_cols,
                                "in_target": ENT_REQUIRES[e] in t_cols,
                                "used": e in shared_ent})

        if not shared_ent or len(shared_cic) < 40:
            out["transfers"][tname] = {"status": "schema too disjoint",
                                       "n_shared_cic": len(shared_cic),
                                       "n_shared_entropy": len(shared_ent)}
            continue

        tdf, t_stats = clean_generic(tdf, shared_cic)
        if "Timestamp" in tdf.columns:
            tdf = D.sort_temporal(tdf)
        yt = binary_label(tdf["Label"])

        Xs = np.hstack([src_df[shared_cic].to_numpy(dtype=np.float32),
                        entropy_block(src_df, shared_ent)])
        ys = D.to_binary_label(src_df["Label"])
        Xt = np.hstack([tdf[shared_cic].to_numpy(dtype=np.float32),
                        entropy_block(tdf, shared_ent)])

        # Cache the aligned matrices so the few-shot study (E20) can reuse this
        # schema alignment instead of reimplementing it and risking a drift
        # between the two experiments.
        slug = {"InSDN": "insdn", "CICIDS2017_DDoS": "ids2017"}.get(tname, tname.lower())
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(CACHE / f"{slug}_X.npy", Xt)
        np.save(CACHE / f"{slug}_y.npy", yt.astype(np.int8))
        np.save(CACHE / f"{slug}_src_X.npy", Xs)
        np.save(CACHE / f"{slug}_src_y.npy", ys.astype(np.int8))
        (CACHE / f"{slug}_features.json").write_text(
            json.dumps(shared_cic + shared_ent), encoding="utf-8")

        info = {
            "status": "ok",
            "caveat": meta["caveat"],
            "schema": {"n_shared_cic_features": len(shared_cic),
                       "n_shared_entropy_features": len(shared_ent),
                       "shared_entropy_features": shared_ent,
                       "unavailable_entropy_features": missing_ent,
                       "transfer_dimension": Xs.shape[1]},
            "target_cleaning": t_stats,
            "target_class_balance": {"n": int(len(yt)),
                                     "benign": int((yt == 0).sum()),
                                     "attack": int((yt == 1).sum())},
        }
        print(f"  target: n={len(yt):,} benign={int((yt==0).sum()):,} "
              f"attack={int((yt==1).sum()):,}", flush=True)

        if len(np.unique(yt)) < 2:
            info["status"] = "target has a single class"
            out["transfers"][tname] = info
            continue

        # --- 1. zero-shot ---------------------------------------------------
        print("  zero-shot...", flush=True)
        t0 = time.time()
        m = RandomForestClassifier(**RF_KW).fit(Xs, ys)
        pr = m.predict_proba(Xt)[:, 1]
        r = scores(yt, pr); r["bootstrap_ci"] = boot_ci(yt, pr)
        r["fit_seconds"] = round(time.time() - t0, 1)
        info["zero_shot"] = r
        print(f"    acc={r['accuracy']:.4f} F1={r['macro_f1']:.4f} "
              f"FPR={r['fpr']:.4f} [{r['bootstrap_ci']['fpr']['lo95']:.4f}, "
              f"{r['bootstrap_ci']['fpr']['hi95']:.4f}] "
              f"FNR={r['fnr']:.4f}", flush=True)

        # --- 2. in-distribution reference -----------------------------------
        print("  in-distribution reference...", flush=True)
        c2 = int(round(len(yt) * 0.70))
        # A chronological split is only a usable reference if BOTH partitions
        # retain a workable share of the minority class. On InSDN the attack
        # class is concentrated early, so a 70/30 chronological cut leaves 92
        # attack flows out of 67,012 at test and the resulting macro F1 measures
        # the split rather than the model. Require at least 5% of each class on
        # each side, and fall back to a stratified split of the same sizes with
        # the substitution recorded.
        def _viable(seg):
            if len(np.unique(seg)) < 2:
                return False
            minority = min((seg == 0).sum(), (seg == 1).sum())
            return minority >= max(50, 0.05 * min((yt == 0).sum(), (yt == 1).sum()))

        rng_t = np.random.default_rng(42)
        perm = rng_t.permutation(len(yt))
        if _viable(yt[:c2]) and _viable(yt[c2:]):
            m2 = RandomForestClassifier(**RF_KW).fit(Xt[:c2], yt[:c2])
            pr2 = m2.predict_proba(Xt[c2:])[:, 1]
            r2 = scores(yt[c2:], pr2); r2["bootstrap_ci"] = boot_ci(yt[c2:], pr2)
            r2["split"] = "chronological"
            info["in_distribution"] = r2
            print(f"    acc={r2['accuracy']:.4f} F1={r2['macro_f1']:.4f}", flush=True)
        else:
            m2 = RandomForestClassifier(**RF_KW).fit(Xt[perm[:c2]], yt[perm[:c2]])
            pr2 = m2.predict_proba(Xt[perm[c2:]])[:, 1]
            r2 = scores(yt[perm[c2:]], pr2)
            r2["bootstrap_ci"] = boot_ci(yt[perm[c2:]], pr2)
            r2["note"] = ("a chronological split of this corpus leaves too few "
                          "minority-class flows in one partition to support a "
                          "meaningful reference, so a stratified split of the same "
                          "sizes was used; the in-distribution row is therefore an "
                          "upper bound and is labelled accordingly")
            r2["split"] = "stratified (chronological split degenerate)"
            info["in_distribution"] = r2
            print(f"    (stratified fallback) acc={r2['accuracy']:.4f} "
                  f"F1={r2['macro_f1']:.4f}", flush=True)

        # --- 3. joint training ----------------------------------------------
        print("  joint training...", flush=True)
        # Use whichever index split the in-distribution reference used, so the
        # three conditions are evaluated on identical test flows.
        if info["in_distribution"].get("split", "").startswith("stratified"):
            tr_idx, te_idx = perm[:c2], perm[c2:]
        else:
            tr_idx, te_idx = np.arange(c2), np.arange(c2, len(yt))
        Xj = np.vstack([Xs, Xt[tr_idx]])
        yj = np.concatenate([ys, yt[tr_idx]])
        m3 = RandomForestClassifier(**RF_KW).fit(Xj, yj)
        pr3 = m3.predict_proba(Xt[te_idx])[:, 1]
        r3 = scores(yt[te_idx], pr3); r3["bootstrap_ci"] = boot_ci(yt[te_idx], pr3)
        info["joint"] = r3
        print(f"    acc={r3['accuracy']:.4f} F1={r3['macro_f1']:.4f}", flush=True)

        info["transfer_gap_macro_f1"] = round(
            info["in_distribution"]["macro_f1"] - info["zero_shot"]["macro_f1"], 6)
        info["joint_gain_macro_f1"] = round(
            r3["macro_f1"] - info["in_distribution"]["macro_f1"], 6)
        out["transfers"][tname] = info

        for cond, rr in (("zero-shot", info["zero_shot"]),
                         ("in-distribution", info["in_distribution"]),
                         ("joint", r3)):
            rows.append({"target": tname, "condition": cond,
                         **{k: rr[k] for k in ("accuracy", "macro_f1", "avg_precision",
                                               "roc_auc", "precision", "recall",
                                               "fpr", "fnr", "n", "n_benign",
                                               "n_attack")},
                         "fpr_lo95": (rr["bootstrap_ci"]["fpr"] or {}).get("lo95"),
                         "fpr_hi95": (rr["bootstrap_ci"]["fpr"] or {}).get("hi95"),
                         "fnr_lo95": (rr["bootstrap_ci"]["fnr"] or {}).get("lo95"),
                         "fnr_hi95": (rr["bootstrap_ci"]["fnr"] or {}).get("hi95")})

    pd.DataFrame(rows).to_csv(RESULTS / "E2_cross_dataset.csv", index=False)
    pd.DataFrame(schema_rows).to_csv(RESULTS / "E2_schema_map.csv", index=False)

    # sanity gate: a perfect zero-shot transfer is a bug signal, not a result
    suspicious = [t for t, v in out["transfers"].items()
                  if v.get("status") == "ok" and v["zero_shot"]["accuracy"] >= 0.9999]
    out["sanity_gate"] = {
        "targets_with_near_perfect_zero_shot": suspicious,
        "action": ("investigate before reporting - a zero-shot transfer at or above "
                   "0.9999 accuracy across corpora is far more likely to indicate a "
                   "leaked column or a degenerate label mapping than genuine transfer")
        if suspicious else "none",
    }
    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E2_cross_dataset", out)
    log_event("E2", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    if suspicious:
        print(f"SANITY GATE TRIPPED for {suspicious}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
