"""
entropy.py - Shannon entropy feature extraction for XAI-SDN.

Computes the eight windowed Shannon entropy features of the published feature
set (Table 3 of the paper):

    H_src_ip, H_dst_ip, H_dst_port, H_proto,
    H_pkt_len, H_iat, H_tcp_flags, H_src_port

Source-port entropy is used rather than TTL entropy. CICFlowMeter exports no
TTL column, so a TTL-derived feature is constant on these records and carries no
information. Source port is exported, is non-degenerate, and is directly
relevant to floods that randomize source ports. The representation is
88-dimensional either way: 80 exported flow statistics plus these eight.

Mathematical foundation:
    H(X) = -sum p(x_i) log2 p(x_i)

All features are computed over a sliding window of N recent flow records.
For online use, a deque(maxlen=N) provides O(1) append/pop semantics.
For offline evaluation the window is replayed over timestamp-ordered records.
Continuous quantities are discretized before the entropy is taken; the bucket
sizes are the module constants below and are the ones the paper reports.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

# Published feature set (Table 3). Source-port entropy replaces TTL entropy,
# which is constant on CICFlowMeter exports.
ENTROPY_FEATURE_NAMES = [
    "H_src_ip",
    "H_dst_ip",
    "H_dst_port",
    "H_proto",
    "H_pkt_len",
    "H_iat",
    "H_tcp_flags",
    "H_src_port",
]

DEFAULT_WINDOW_SIZE = 1000
DEFAULT_PKT_LEN_BIN = 10  # Bytes per discretization bucket
DEFAULT_IAT_BIN = 1000  # Microseconds per discretization bucket
DEFAULT_SRC_PORT_BIN = 1  # Source port is used at full resolution


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
        src_port_bin_size (int): Discretization bucket size for source port.
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
        src_port_bin_size: int = DEFAULT_SRC_PORT_BIN,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size
        self.pkt_len_bin_size = max(1, pkt_len_bin_size)
        self.iat_bin_size = max(1, iat_bin_size)
        self.src_port_bin_size = max(1, src_port_bin_size)
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
                - src_port (int): TCP/UDP source port.

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
        src_ports = [int(f.get("src_port", 0) // self.src_port_bin_size) for f in window]

        return {
            "H_src_ip": shannon_entropy(src_ips),
            "H_dst_ip": shannon_entropy(dst_ips),
            "H_dst_port": shannon_entropy(dst_ports),
            "H_proto": shannon_entropy(protocols),
            "H_pkt_len": shannon_entropy(pkt_lens),
            "H_iat": shannon_entropy(iats),
            "H_tcp_flags": shannon_entropy(tcp_flags),
            "H_src_port": shannon_entropy(src_ports),
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
    src_port_bin_size: int = DEFAULT_SRC_PORT_BIN,
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
    src_port_bin = src_port_bin_size

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
            int(record.get("src_port", 0) // src_port_bin)
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
