"""Canonical CIC-DDoS2019 loading, cleaning and feature construction.

Ported from the XAI-SDN repository with three deliberate changes, each tied to a
open question:

  * the entropy engine is instrumented so that the H_ttl defect can be
    measured rather than assumed;
  * both the temporal split described in the manuscript and the stratified split
    implemented in the repository are available, so that E0 can determine which
    one actually produced the submitted numbers;
  * the running-sum entropy update is exposed in three numerical variants so the
    drift study can target the form that actually executes.

Everything is cached to .npy so the 1.87 GB CSV is parsed once.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from paths import DATA_ROOT

# --------------------------------------------------------------------------- #
# Feature definitions
# --------------------------------------------------------------------------- #

CIC_FEATURE_NAMES: list[str] = [
    "Destination_Port", "Flow_Duration", "Total_Fwd_Packets", "Total_Backward_Packets",
    "Total_Length_of_Fwd_Packets", "Total_Length_of_Bwd_Packets",
    "Fwd_Packet_Length_Max", "Fwd_Packet_Length_Min", "Fwd_Packet_Length_Mean",
    "Fwd_Packet_Length_Std", "Bwd_Packet_Length_Max", "Bwd_Packet_Length_Min",
    "Bwd_Packet_Length_Mean", "Bwd_Packet_Length_Std",
    "Flow_Bytes_s", "Flow_Packets_s",
    "Flow_IAT_Mean", "Flow_IAT_Std", "Flow_IAT_Max", "Flow_IAT_Min",
    "Fwd_IAT_Total", "Fwd_IAT_Mean", "Fwd_IAT_Std", "Fwd_IAT_Max", "Fwd_IAT_Min",
    "Bwd_IAT_Total", "Bwd_IAT_Mean", "Bwd_IAT_Std", "Bwd_IAT_Max", "Bwd_IAT_Min",
    "Fwd_PSH_Flags", "Bwd_PSH_Flags", "Fwd_URG_Flags", "Bwd_URG_Flags",
    "Fwd_Header_Length", "Bwd_Header_Length", "Fwd_Packets_s", "Bwd_Packets_s",
    "Min_Packet_Length", "Max_Packet_Length", "Packet_Length_Mean",
    "Packet_Length_Std", "Packet_Length_Variance",
    "FIN_Flag_Count", "SYN_Flag_Count", "RST_Flag_Count", "PSH_Flag_Count",
    "ACK_Flag_Count", "URG_Flag_Count", "CWE_Flag_Count", "ECE_Flag_Count",
    "Down_Up_Ratio", "Average_Packet_Size", "Avg_Fwd_Segment_Size",
    "Avg_Bwd_Segment_Size", "Fwd_Header_Length_2",
    "Fwd_Avg_Bytes_Bulk", "Fwd_Avg_Packets_Bulk", "Fwd_Avg_Bulk_Rate",
    "Bwd_Avg_Bytes_Bulk", "Bwd_Avg_Packets_Bulk", "Bwd_Avg_Bulk_Rate",
    "Subflow_Fwd_Packets", "Subflow_Fwd_Bytes", "Subflow_Bwd_Packets",
    "Subflow_Bwd_Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active_Mean", "Active_Std", "Active_Max", "Active_Min",
    "Idle_Mean", "Idle_Std", "Idle_Max", "Idle_Min",
    "Protocol", "Source_Port",
]
assert len(CIC_FEATURE_NAMES) == 80, len(CIC_FEATURE_NAMES)

# Legacy entropy set. H_ttl is degenerate on CICFlowMeter exports.
ENTROPY_FEATURES_ORIGINAL = [
    "H_src_ip", "H_dst_ip", "H_dst_port", "H_proto",
    "H_pkt_len", "H_iat", "H_tcp_flags", "H_ttl",
]
# Repaired set: TTL entropy replaced by source-port entropy, which is genuinely
# present in CICFlowMeter output and discriminative for randomised-source floods.
ENTROPY_FEATURES_REPAIRED = [
    "H_src_ip", "H_dst_ip", "H_dst_port", "H_proto",
    "H_pkt_len", "H_iat", "H_tcp_flags", "H_src_port",
]

DEFAULT_WINDOW = 1000
BIN_PKT_LEN = 10        # bytes per bucket
BIN_IAT = 1000          # microseconds per bucket
BIN_TTL = 5             # TTL units per bucket (legacy; TTL is constant)
BIN_SRC_PORT = 1        # source port used at full resolution


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, collapse spaces and slashes, drop punctuation."""
    ren = {}
    for col in df.columns:
        clean = re.sub(r"[^A-Za-z0-9_]", "", re.sub(r"[\s/]+", "_", col.strip()))
        ren[col] = clean
    out = df.rename(columns=ren)
    # CICFlowMeter emits a duplicated header 'Fwd Header Length.1'
    if "Fwd_Header_Length1" in out.columns:
        out = out.rename(columns={"Fwd_Header_Length1": "Fwd_Header_Length_2"})
    return out


# --------------------------------------------------------------------------- #
# Rolling entropy: three numerical variants for the drift study
# --------------------------------------------------------------------------- #

def _log_caches(window: int):
    m = max(1005, window + 5)
    clogc = np.array([0.0] + [c * math.log2(c) for c in range(1, m)])
    log2len = np.array([0.0] + [math.log2(l) for l in range(1, m)])
    invlen = np.array([0.0] + [1.0 / l for l in range(1, m)])
    return clogc.tolist(), log2len.tolist(), invlen.tolist()


def rolling_entropy(
    streams: Sequence[Sequence[Any]],
    window: int = DEFAULT_WINDOW,
    variant: str = "running_sum",
    recompute_every: int | None = None,
) -> np.ndarray:
    """Entropy of each stream over a sliding window of the last `window` items.

    variant:
      'running_sum'  -- the form the repository implements: S = sum(c log2 c),
                        H = log2|W| - S/|W|, S updated incrementally.
      'kahan'        -- same recurrence with Neumaier compensated summation.
      'exact'        -- recompute S from the multiset every step (ground truth,
                        O(distinct) per step; slow, used for drift checkpoints).

    `recompute_every` forces an exact resynchronisation of S every R updates,
    modelling the periodic-recomputation mitigation.
    """
    n_streams = len(streams)
    n = len(streams[0]) if n_streams else 0
    out = np.zeros((n, n_streams), dtype=np.float64)
    clogc, log2len, invlen = _log_caches(window)

    for s_idx in range(n_streams):
        vals = streams[s_idx]
        q: deque = deque(maxlen=window)
        counts: dict[Any, int] = {}
        S = 0.0
        comp = 0.0  # Neumaier compensation

        def add(x: float, S: float, comp: float):
            if variant == "kahan":
                t = S + x
                if abs(S) >= abs(x):
                    comp += (S - t) + x
                else:
                    comp += (x - t) + S
                return t, comp
            return S + x, comp

        for i in range(n):
            v = vals[i]
            if len(q) >= window:
                vo = q.popleft()
                co = counts[vo]
                S, comp = add(-clogc[co] + clogc[co - 1], S, comp)
                if co == 1:
                    del counts[vo]
                else:
                    counts[vo] = co - 1
            ci = counts.get(v, 0)
            S, comp = add(-clogc[ci] + clogc[ci + 1], S, comp)
            counts[v] = ci + 1
            q.append(v)

            if variant == "exact" or (recompute_every and (i + 1) % recompute_every == 0):
                S = 0.0
                for c in counts.values():
                    S += clogc[c]
                comp = 0.0

            L = len(q)
            Seff = S + comp if variant == "kahan" else S
            out[i, s_idx] = max(0.0, log2len[L] - Seff * invlen[L])
    return out


def entropy_streams(df: pd.DataFrame, repaired: bool) -> tuple[list[list[Any]], list[str]]:
    """Build the eight discrete streams the entropy engine consumes.

    The exact source column and bin width of every stream is returned so the
    manuscript can publish a complete feature specification.
    """
    n = len(df)

    def col(name: str, default):
        return df[name].values if name in df.columns else np.full(n, default)

    src_ip = df["Source_IP"].astype(str).values if "Source_IP" in df.columns else None
    if src_ip is None:
        raise KeyError("Source_IP column missing; refusing to synthesise addresses.")
    dst_ip = df["Destination_IP"].astype(str).values
    dst_port = col("Destination_Port", 0).astype(np.int64)
    proto = col("Protocol", 0).astype(np.int64)
    pkt_len = np.nan_to_num(col("Packet_Length_Mean", 0.0).astype(np.float64))
    iat = np.nan_to_num(col("Flow_IAT_Mean", 0.0).astype(np.float64))
    syn = col("SYN_Flag_Count", 0).astype(np.int64)

    streams = [
        list(src_ip),
        list(dst_ip),
        list(dst_port),
        list(proto),
        list((pkt_len // BIN_PKT_LEN).astype(np.int64)),
        list((iat // BIN_IAT).astype(np.int64)),
        list(syn),
    ]
    if repaired:
        src_port = col("Source_Port", 0).astype(np.int64)
        streams.append(list(src_port))
        names = ENTROPY_FEATURES_REPAIRED
    else:
        # The submitted pipeline hard-codes ttl = 64 for every record.
        streams.append([64 // BIN_TTL] * n)
        names = ENTROPY_FEATURES_ORIGINAL
    return streams, list(names)


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #

KEEP_META = ["Source_IP", "Destination_IP", "Timestamp", "Label", "Flow_ID"]


def load_partition(csv_path: Path, nrows: int | None = None,
                   verbose: bool = True) -> pd.DataFrame:
    """Load one CIC CSV, normalise columns, keep features + the metadata we need."""
    t0 = time.time()
    # Read the header to build a usecols list; avoids materialising junk columns.
    head = pd.read_csv(csv_path, nrows=0)
    head = normalize_columns(head)
    available = set(head.columns)
    want = [c for c in CIC_FEATURE_NAMES if c in available] + \
           [c for c in KEEP_META if c in available]
    raw_by_clean = {}
    orig = pd.read_csv(csv_path, nrows=0).columns
    for o in orig:
        clean = re.sub(r"[^A-Za-z0-9_]", "", re.sub(r"[\s/]+", "_", o.strip()))
        if clean == "Fwd_Header_Length1":
            clean = "Fwd_Header_Length_2"
        raw_by_clean[clean] = o
    usecols = [raw_by_clean[c] for c in want]

    df = pd.read_csv(csv_path, usecols=usecols, nrows=nrows, low_memory=False)
    df = normalize_columns(df)
    if verbose:
        print(f"  loaded {len(df):,} rows x {df.shape[1]} cols in {time.time()-t0:.1f}s",
              flush=True)
    return df


def clean_partition(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Apply the manuscript's stated cleaning: drop NaN/Inf rows, drop duplicates."""
    stats: dict[str, Any] = {"rows_in": int(len(df))}
    feat = [c for c in CIC_FEATURE_NAMES if c in df.columns]
    stats["n_cic_features_present"] = len(feat)
    stats["missing_cic_features"] = [c for c in CIC_FEATURE_NAMES if c not in df.columns]

    for c in feat:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)

    nan_by_col = df[feat].isna().sum()
    stats["nan_cells_total"] = int(nan_by_col.sum())
    stats["nan_top_columns"] = {k: int(v) for k, v in
                                nan_by_col.sort_values(ascending=False).head(10).items()
                                if v > 0}
    before = len(df)
    df = df.dropna(subset=feat)
    stats["rows_dropped_nan_inf"] = int(before - len(df))

    before = len(df)
    df = df.drop_duplicates(subset=feat + (["Label"] if "Label" in df.columns else []))
    stats["rows_dropped_duplicate"] = int(before - len(df))

    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip()
        stats["label_counts"] = {k: int(v) for k, v in df["Label"].value_counts().items()}
    stats["rows_out"] = int(len(df))
    if verbose:
        print(f"  cleaned: {stats['rows_in']:,} -> {stats['rows_out']:,} "
              f"(-{stats['rows_dropped_nan_inf']:,} nan/inf, "
              f"-{stats['rows_dropped_duplicate']:,} dup)", flush=True)
    return df.reset_index(drop=True), stats


def to_binary_label(labels: pd.Series) -> np.ndarray:
    """1 = attack, 0 = benign. Everything that is not BENIGN is an attack."""
    return (~labels.str.upper().str.startswith("BENIGN")).astype(np.int8).values


def sort_temporal(df: pd.DataFrame) -> pd.DataFrame:
    if "Timestamp" not in df.columns:
        return df
    ts = pd.to_datetime(df["Timestamp"], errors="coerce", format="mixed", dayfirst=True)
    df = df.assign(_ts=ts).sort_values("_ts", kind="mergesort").drop(columns=["_ts"])
    return df.reset_index(drop=True)


def build_matrix(df: pd.DataFrame, repaired: bool, window: int = DEFAULT_WINDOW,
                 variant: str = "running_sum", verbose: bool = True
                 ) -> tuple[np.ndarray, list[str]]:
    """Concatenate the 80 CIC features with the 8 rolling-entropy features."""
    t0 = time.time()
    feat = [c for c in CIC_FEATURE_NAMES if c in df.columns]
    X_cic = df[feat].to_numpy(dtype=np.float32, copy=False)
    streams, ent_names = entropy_streams(df, repaired=repaired)
    H = rolling_entropy(streams, window=window, variant=variant).astype(np.float32)
    X = np.hstack([X_cic, H])
    del X_cic, H
    if verbose:
        print(f"  features: {X.shape} in {time.time()-t0:.1f}s", flush=True)
    return X, feat + ent_names


def file_sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    read = 0
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
            read += len(blk)
            if limit and read >= limit:
                break
    return h.hexdigest()
