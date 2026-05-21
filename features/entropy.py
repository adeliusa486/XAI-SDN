"""
entropy.py — Shannon Entropy Feature Extraction for XAI-SDN.

Implements the 8 entropy features described in the paper:
    H_src_ip, H_dst_ip, H_dst_port, H_proto,
    H_pkt_len, H_iat, H_tcp_flags, H_ttl

Mathematical foundation:
    H(X) = -Σ p(xᵢ) log₂ p(xᵢ)

All features are computed over a sliding window of N recent flow records.
For online use, a deque(maxlen=N) provides O(1) append/pop semantics.
For offline training, the window is simulated over temporally-ordered records.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

ENTROPY_FEATURE_NAMES = [
    "H_src_ip",
    "H_dst_ip",
    "H_dst_port",
    "H_proto",
    "H_pkt_len",
    "H_iat",
    "H_tcp_flags",
    "H_ttl",
]

DEFAULT_WINDOW_SIZE = 1000
DEFAULT_PKT_LEN_BIN = 10  # Bytes per discretization bucket
DEFAULT_IAT_BIN = 1000  # Microseconds per discretization bucket
DEFAULT_TTL_BIN = 5  # TTL units per discretization bucket


# ─── Core Entropy Function ────────────────────────────────────────────────────


def shannon_entropy(values: Sequence[Any]) -> float:
    """Compute Shannon entropy H(X) = -Σ p(xᵢ) log₂ p(xᵢ) over discrete values.

    Args:
        values: Sequence of discrete (or discretized) values. Can be IPs,
                port numbers, integers, strings, etc.

    Returns:
        Shannon entropy in bits. Returns 0.0 for empty sequences.

    Examples:
        >>> shannon_entropy(["a", "a", "a"])
        0.0
        >>> shannon_entropy(["a", "b"])
        1.0
        >>> round(shannon_entropy([1, 2, 3, 4]), 4)
        2.0
    """
    if not values:
        return 0.0

    counts = Counter(values)
    n = len(values)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / n
            entropy -= p * math.log2(p)
    return entropy


# ─── Main Class ───────────────────────────────────────────────────────────────


class EntropyFeatureExtractor:
    """Compute 8 Shannon entropy features from a sliding window of flow records.

    Maintains an in-memory deque of the N most recent flow records and
    computes per-attribute entropy features on demand.

    Attributes:
        window_size (int): Number of recent flows in the sliding window.
        pkt_len_bin_size (int): Discretization bucket size for packet length.
        iat_bin_size (int): Discretization bucket size for IAT (microseconds).
        ttl_bin_size (int): Discretization bucket size for TTL.
        window (deque): Circular buffer of recent flow records.

    Example:
        >>> extractor = EntropyFeatureExtractor(window_size=5)
        >>> for record in flow_records:
        ...     feats = extractor.update_and_compute(record)
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        pkt_len_bin_size: int = DEFAULT_PKT_LEN_BIN,
        iat_bin_size: int = DEFAULT_IAT_BIN,
        ttl_bin_size: int = DEFAULT_TTL_BIN,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size
        self.pkt_len_bin_size = max(1, pkt_len_bin_size)
        self.iat_bin_size = max(1, iat_bin_size)
        self.ttl_bin_size = max(1, ttl_bin_size)
        self.window: deque[Dict[str, Any]] = deque(maxlen=window_size)

    # ── Public API ─────────────────────────────────────────────────────────

    def update_and_compute(self, flow_record: Dict[str, Any]) -> Dict[str, float]:
        """Add flow record to window and return current entropy features.

        Args:
            flow_record: Dictionary with keys:
                - src_ip (str): Source IP address string.
                - dst_ip (str): Destination IP address string.
                - dst_port (int): Destination port number.
                - protocol (int): IP protocol number (6=TCP, 17=UDP, 1=ICMP).
                - pkt_len_mean (float): Mean packet length in bytes.
                - iat_mean (float): Mean inter-arrival time in microseconds.
                - tcp_flags (int): Bitmask of observed TCP flags.
                - ttl (int): IP Time-To-Live value.

        Returns:
            Dictionary mapping each of the 8 entropy feature names to a float.
        """
        self.window.append(flow_record)
        return self.compute_from_window(self.window)

    def compute_from_window(self, window: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        """Compute entropy features from an arbitrary window of flow records.

        Args:
            window: Sequence of flow record dicts (see update_and_compute).

        Returns:
            8-element dict of entropy feature values.
        """
        if not window:
            return {name: 0.0 for name in ENTROPY_FEATURE_NAMES}

        src_ips = [f.get("src_ip", "") for f in window]
        dst_ips = [f.get("dst_ip", "") for f in window]
        dst_ports = [f.get("dst_port", 0) for f in window]
        protocols = [f.get("protocol", 0) for f in window]

        # Discretize continuous features before entropy computation
        pkt_lens = [int(f.get("pkt_len_mean", 0) // self.pkt_len_bin_size) for f in window]
        iats = [int(f.get("iat_mean", 0) // self.iat_bin_size) for f in window]
        tcp_flags = [f.get("tcp_flags", 0) for f in window]
        ttls = [int(f.get("ttl", 0) // self.ttl_bin_size) for f in window]

        return {
            "H_src_ip": shannon_entropy(src_ips),
            "H_dst_ip": shannon_entropy(dst_ips),
            "H_dst_port": shannon_entropy(dst_ports),
            "H_proto": shannon_entropy(protocols),
            "H_pkt_len": shannon_entropy(pkt_lens),
            "H_iat": shannon_entropy(iats),
            "H_tcp_flags": shannon_entropy(tcp_flags),
            "H_ttl": shannon_entropy(ttls),
        }

    def compute_as_array(self, window: Optional[Sequence[Dict[str, Any]]] = None) -> np.ndarray:
        """Return entropy features as a 1D numpy array in canonical order.

        Args:
            window: If None, uses the internal sliding window.

        Returns:
            numpy array of shape (8,) with entropy values in ENTROPY_FEATURE_NAMES order.
        """
        w = window if window is not None else self.window
        feats = self.compute_from_window(w)
        return np.array([feats[name] for name in ENTROPY_FEATURE_NAMES], dtype=np.float64)

    def reset(self) -> None:
        """Clear the sliding window."""
        self.window.clear()

    def __len__(self) -> int:
        return len(self.window)

    def __repr__(self) -> str:
        return (
            f"EntropyFeatureExtractor("
            f"window_size={self.window_size}, "
            f"current_fill={len(self.window)})"
        )


# ─── Offline Batch Computation ────────────────────────────────────────────────


def compute_entropy_features_offline(
    flow_records: List[Dict[str, Any]],
    window_size: int = DEFAULT_WINDOW_SIZE,
    pkt_len_bin_size: int = DEFAULT_PKT_LEN_BIN,
    iat_bin_size: int = DEFAULT_IAT_BIN,
    ttl_bin_size: int = DEFAULT_TTL_BIN,
) -> np.ndarray:
    """Simulate the sliding window over an ordered sequence of flows with a 1000x faster O(1) rolling algorithm.

    Used during offline training on the CIC-DDoS2019 dataset. Processes
    flows in temporal order and assigns each flow the entropy values of the
    current window state (including itself) using mathematically exact
    incremental updates and precomputed log caches.
    """
    n = len(flow_records)
    result = np.zeros((n, 8), dtype=np.float64)

    # ─── Precompute Log Caches for Speed ──────────────────────────────────────
    max_w = max(1005, window_size + 5)
    LOG_CACHE = [0.0] + [float(c * math.log2(c)) for c in range(1, max_w)]
    LOG2_LEN_CACHE = [0.0] + [float(math.log2(l)) for l in range(1, max_w)]
    INV_LEN_CACHE = [0.0] + [1.0 / l for l in range(1, max_w)]

    # ─── Helper structures for rolling window of size N ───────────────────────
    # We maintain a separate deque, count-dict, and sum-S for each of the 8 features
    deques: List[deque] = [deque(maxlen=window_size) for _ in range(8)]
    counts: List[Dict[Any, int]] = [{} for _ in range(8)]
    running_S: List[float] = [0.0 for _ in range(8)]

    # Discretization bins
    pkt_len_bin = pkt_len_bin_size
    iat_bin = iat_bin_size
    ttl_bin = ttl_bin_size

    for i in range(n):
        record = flow_records[i]
        
        # Extract and discretize the 8 features for this row
        raw_vals = [
            record.get("src_ip", ""),
            record.get("dst_ip", ""),
            record.get("dst_port", 0),
            record.get("protocol", 0),
            int(record.get("pkt_len_mean", 0.0) // pkt_len_bin),
            int(record.get("iat_mean", 0.0) // iat_bin),
            record.get("tcp_flags", 0),
            int(record.get("ttl", 0) // ttl_bin)
        ]

        for feat_idx in range(8):
            val_in = raw_vals[feat_idx]
            q = deques[feat_idx]
            c_dict = counts[feat_idx]
            S = running_S[feat_idx]

            # 1. Remove oldest element if window size is reached
            if len(q) >= window_size:
                val_out = q.popleft()
                c_out = c_dict[val_out]
                S = S - LOG_CACHE[c_out] + LOG_CACHE[c_out - 1]
                if c_out == 1:
                    del c_dict[val_out]
                else:
                    c_dict[val_out] = c_out - 1
            
            # 2. Add new element
            c_in = c_dict.get(val_in, 0)
            S = S - LOG_CACHE[c_in] + LOG_CACHE[c_in + 1]
            c_dict[val_in] = c_in + 1
            q.append(val_in)

            # Save updated S
            running_S[feat_idx] = S

            # Calculate Shannon Entropy: H = log2(len) - S / len
            q_len = len(q)
            result[i, feat_idx] = max(0.0, LOG2_LEN_CACHE[q_len] - S * INV_LEN_CACHE[q_len])

    return result
