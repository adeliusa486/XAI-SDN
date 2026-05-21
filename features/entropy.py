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
    """Simulate the sliding window over an ordered sequence of flows.

    Used during offline training on the CIC-DDoS2019 dataset.  Processes
    flows in temporal order and assigns each flow the entropy values of the
    current window state (including itself).

    Args:
        flow_records: List of flow record dicts ordered by timestamp.
        window_size: Sliding window size N.
        pkt_len_bin_size: Packet length discretization bucket size.
        iat_bin_size: IAT discretization bucket size (microseconds).
        ttl_bin_size: TTL discretization bucket size.

    Returns:
        numpy array of shape (n_flows, 8) with entropy features per flow.
    """
    extractor = EntropyFeatureExtractor(
        window_size=window_size,
        pkt_len_bin_size=pkt_len_bin_size,
        iat_bin_size=iat_bin_size,
        ttl_bin_size=ttl_bin_size,
    )

    n = len(flow_records)
    result = np.zeros((n, 8), dtype=np.float64)

    for i, record in enumerate(flow_records):
        feat_dict = extractor.update_and_compute(record)
        result[i] = [feat_dict[name] for name in ENTROPY_FEATURE_NAMES]

    return result
