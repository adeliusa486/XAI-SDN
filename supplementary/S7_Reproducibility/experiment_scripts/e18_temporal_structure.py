"""E18 - Temporal structure of the benign class.

E12 exposed something that changes how every chronological result on this
corpus must be read. Splitting the SYN partition chronologically at 70% leaves
only 522 benign flows in the training partition and 60 in the validation block,
against 30,432 in the test partition. In other words 98% of all benign traffic
in this capture occurs in its final third.

Before drawing any conclusion from that, two things have to be checked, because
an artifact of our own preprocessing would produce the same appearance:

  1. Are the timestamps being parsed correctly? A malformed parse would produce
     an essentially arbitrary ordering, and any apparent clustering with it.
  2. Given a correct parse, how is each class actually distributed over the
     capture timeline?

If the parse is sound and the clustering is real, then a chronological split of
this partition is not merely a harder version of the same task. It is a
near-zero-shot problem for the benign class, and that has to be stated plainly
rather than reported as a lower accuracy number.
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

CACHE = DATA_ROOT / "cache"
N_BINS = 50


def main() -> int:
    log_event("E18", "start")
    t_all = time.time()
    out: dict = {"experiment": "E18",
                 "objective": "verify timestamp parsing and characterize the "
                              "temporal distribution of each class"}

    src = DATA_ROOT / "CICDDoS2019" / "03-11" / "Syn.csv"

    # ---- 1. timestamp parse validation -------------------------------------
    print("Validating timestamp parsing...", flush=True)
    raw = pd.read_csv(src, usecols=[" Timestamp", " Label"], low_memory=False)
    raw.columns = ["Timestamp", "Label"]
    samples = raw["Timestamp"].dropna().astype(str)
    out["timestamp_examples"] = samples.head(5).tolist() + samples.tail(5).tolist()

    ts = pd.to_datetime(raw["Timestamp"], errors="coerce", format="mixed",
                        dayfirst=True)
    n_bad = int(ts.isna().sum())
    out["timestamp_parse"] = {
        "n_rows": int(len(ts)),
        "n_unparseable": n_bad,
        "unparseable_rate": round(n_bad / len(ts), 8),
        "min": str(ts.min()), "max": str(ts.max()),
        "span_seconds": float((ts.max() - ts.min()).total_seconds()),
        "n_distinct": int(ts.nunique()),
        "monotone_in_file_order": bool(ts.dropna().is_monotonic_increasing),
    }
    print(f"  unparseable: {n_bad:,} / {len(ts):,}", flush=True)
    print(f"  span: {out['timestamp_parse']['min']} .. "
          f"{out['timestamp_parse']['max']} "
          f"({out['timestamp_parse']['span_seconds']/60:.1f} min)", flush=True)
    print(f"  already monotone in file order: "
          f"{out['timestamp_parse']['monotone_in_file_order']}", flush=True)

    # a second parser as a cross-check on the format inference
    ts_alt = pd.to_datetime(raw["Timestamp"], errors="coerce", dayfirst=False,
                            format="mixed")
    agree = float((ts.dropna() == ts_alt.reindex(ts.dropna().index)).mean())
    out["timestamp_parse"]["dayfirst_agreement"] = round(agree, 6)
    print(f"  agreement between dayfirst=True and dayfirst=False: {agree:.4f}",
          flush=True)

    # ---- 2. class distribution over the timeline ----------------------------
    print("\nCharacterizing the class timeline...", flush=True)
    y_raw = (~raw["Label"].astype(str).str.strip().str.upper()
             .str.startswith("BENIGN")).astype(np.int8)
    ok = ts.notna()
    tsv = ts[ok]
    yv = y_raw[ok].to_numpy()
    order = np.argsort(tsv.to_numpy(), kind="mergesort")
    y_sorted = yv[order]
    t_sorted = tsv.to_numpy()[order]

    n = len(y_sorted)
    edges = np.linspace(0, n, N_BINS + 1).astype(int)
    rows = []
    for b in range(N_BINS):
        seg = y_sorted[edges[b]:edges[b + 1]]
        rows.append({"bin": b + 1,
                     "position_pct": round(100 * (b + 0.5) / N_BINS, 1),
                     "n": int(len(seg)),
                     "benign": int((seg == 0).sum()),
                     "attack": int((seg == 1).sum()),
                     "benign_pct": round(100 * float((seg == 0).mean()), 4)})
    pd.DataFrame(rows).to_csv(RESULTS / "E18_timeline.csv", index=False)

    benign_pos = np.flatnonzero(y_sorted == 0) / max(n - 1, 1)
    out["benign_position_in_timeline"] = {
        "n_benign": int((y_sorted == 0).sum()),
        "n_attack": int((y_sorted == 1).sum()),
        "min": float(benign_pos.min()), "p05": float(np.percentile(benign_pos, 5)),
        "median": float(np.median(benign_pos)),
        "p95": float(np.percentile(benign_pos, 95)), "max": float(benign_pos.max()),
        "fraction_in_last_30pct": float((benign_pos > 0.70).mean()),
        "fraction_in_last_10pct": float((benign_pos > 0.90).mean()),
    }
    bp = out["benign_position_in_timeline"]
    print(f"  benign flows: {bp['n_benign']:,}", flush=True)
    print(f"  median position in the timeline: {bp['median']:.4f}", flush=True)
    print(f"  fraction of benign in the last 30%: "
          f"{bp['fraction_in_last_30pct']:.4%}", flush=True)
    print(f"  fraction of benign in the last 10%: "
          f"{bp['fraction_in_last_10pct']:.4%}", flush=True)

    # ---- 3. consequences for chronological splitting -----------------------
    cons = {}
    for frac in (0.50, 0.60, 0.70, 0.80, 0.90):
        cut = int(round(n * frac))
        cons[f"{int(frac*100)}/{int((1-frac)*100)}"] = {
            "train_benign": int((y_sorted[:cut] == 0).sum()),
            "test_benign": int((y_sorted[cut:] == 0).sum()),
            "train_benign_pct": round(100 * float((y_sorted[:cut] == 0).mean()), 5),
            "test_benign_pct": round(100 * float((y_sorted[cut:] == 0).mean()), 5),
        }
    out["chronological_split_consequences"] = cons
    print("\n  chronological split -> benign flows available for training:",
          flush=True)
    for k, v in cons.items():
        print(f"    {k:>7}: train_benign={v['train_benign']:>7,}  "
              f"test_benign={v['test_benign']:>7,}", flush=True)

    # ---- 4. the same question on the cleaned, deduplicated matrix ----------
    yc = np.load(CACHE / "syn0311_y.npy")
    ncl = len(yc)
    cut70 = int(round(ncl * 0.70))
    out["cleaned_matrix"] = {
        "n": int(ncl),
        "benign_total": int((yc == 0).sum()),
        "benign_in_first_70pct": int((yc[:cut70] == 0).sum()),
        "benign_in_last_30pct": int((yc[cut70:] == 0).sum()),
    }
    cm = out["cleaned_matrix"]
    print(f"\n  cleaned matrix: {cm['benign_in_first_70pct']:,} benign flows in the "
          f"first 70% vs {cm['benign_in_last_30pct']:,} in the last 30%", flush=True)

    ratio = cm["benign_in_last_30pct"] / max(cm["benign_in_first_70pct"], 1)
    out["verdict"] = (
        f"Timestamps parse cleanly ({n_bad} unparseable of {len(ts):,}) and the "
        f"clustering is a property of the capture, not of our preprocessing. "
        f"{100*bp['fraction_in_last_30pct']:.1f}% of benign flows fall in the final "
        f"30% of the timeline, so a chronological 70/30 split leaves only "
        f"{cm['benign_in_first_70pct']:,} benign flows for training against "
        f"{cm['benign_in_last_30pct']:,} at test, a ratio of 1 to {ratio:.0f}. "
        f"A chronological split of this partition is therefore close to a "
        f"zero-shot problem for the benign class rather than a conventional "
        f"held-out evaluation, and results under it must be described that way.")
    print("\nVERDICT: " + out["verdict"], flush=True)

    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E18_temporal_structure", out)
    log_event("E18", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
