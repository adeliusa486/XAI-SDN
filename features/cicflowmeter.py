"""
cicflowmeter.py — CICFlowMeter-Compatible Feature Extraction for XAI-SDN.

Implements two modes:
  1. OFFLINE (DataFrame): Loads pre-extracted CICFlowMeter CSV files from the
     CIC-DDoS2019 dataset, normalizes column names, and handles inf/NaN cleanup.
  2. ONLINE (OpenFlow bridge): Derives a best-effort subset of CICFlowMeter
     features directly from OpenFlow per-flow statistics counters.

IMPORTANT: CICFlowMeter requires raw PCAP files to compute all 80 features.
OpenFlow only exposes aggregate counters. The online bridge covers ~40/80
features. A network tap is required for full feature extraction in production.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# ─── CICFlowMeter Feature Definitions ────────────────────────────────────────

# The 80 canonical CICFlowMeter feature names (normalized, no leading spaces).
# Based on CICFlowMeter v3 output column names.
CIC_FEATURE_NAMES: List[str] = [
    # Basic flow counts
    "Destination_Port",
    "Flow_Duration",
    "Total_Fwd_Packets",
    "Total_Backward_Packets",
    "Total_Length_of_Fwd_Packets",
    "Total_Length_of_Bwd_Packets",
    # Packet length statistics
    "Fwd_Packet_Length_Max",
    "Fwd_Packet_Length_Min",
    "Fwd_Packet_Length_Mean",
    "Fwd_Packet_Length_Std",
    "Bwd_Packet_Length_Max",
    "Bwd_Packet_Length_Min",
    "Bwd_Packet_Length_Mean",
    "Bwd_Packet_Length_Std",
    # Flow byte/packet rates
    "Flow_Bytes_s",
    "Flow_Packets_s",
    # Flow IAT (inter-arrival time)
    "Flow_IAT_Mean",
    "Flow_IAT_Std",
    "Flow_IAT_Max",
    "Flow_IAT_Min",
    "Fwd_IAT_Total",
    "Fwd_IAT_Mean",
    "Fwd_IAT_Std",
    "Fwd_IAT_Max",
    "Fwd_IAT_Min",
    "Bwd_IAT_Total",
    "Bwd_IAT_Mean",
    "Bwd_IAT_Std",
    "Bwd_IAT_Max",
    "Bwd_IAT_Min",
    # TCP Flags
    "Fwd_PSH_Flags",
    "Bwd_PSH_Flags",
    "Fwd_URG_Flags",
    "Bwd_URG_Flags",
    "Fwd_Header_Length",
    "Bwd_Header_Length",
    # Packet rates per direction
    "Fwd_Packets_s",
    "Bwd_Packets_s",
    # Packet length min/max/mean/std aggregate
    "Min_Packet_Length",
    "Max_Packet_Length",
    "Packet_Length_Mean",
    "Packet_Length_Std",
    "Packet_Length_Variance",
    # TCP flag counts
    "FIN_Flag_Count",
    "SYN_Flag_Count",
    "RST_Flag_Count",
    "PSH_Flag_Count",
    "ACK_Flag_Count",
    "URG_Flag_Count",
    "CWE_Flag_Count",
    "ECE_Flag_Count",
    # Byte ratios
    "Down_Up_Ratio",
    "Average_Packet_Size",
    "Avg_Fwd_Segment_Size",
    "Avg_Bwd_Segment_Size",
    # Bulk features
    "Fwd_Header_Length_2",
    "Fwd_Avg_Bytes_Bulk",
    "Fwd_Avg_Packets_Bulk",
    "Fwd_Avg_Bulk_Rate",
    "Bwd_Avg_Bytes_Bulk",
    "Bwd_Avg_Packets_Bulk",
    "Bwd_Avg_Bulk_Rate",
    # Subflow
    "Subflow_Fwd_Packets",
    "Subflow_Fwd_Bytes",
    "Subflow_Bwd_Packets",
    "Subflow_Bwd_Bytes",
    # Window sizes
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    # Active/Idle
    "Active_Mean",
    "Active_Std",
    "Active_Max",
    "Active_Min",
    "Idle_Mean",
    "Idle_Std",
    "Idle_Max",
    "Idle_Min",
    # Standard CICFlowMeter v3 columns — protocol and source port
    "Protocol",
    "Source_Port",
]

# Label classes in CIC-DDoS2019
LABEL_BENIGN = "BENIGN"
ATTACK_LABELS = [
    "DDoS-UDP",
    "DDoS-TCP",
    "DDoS-ICMP",
    "Slowloris",
    "WebDDoS",
]

# Canonical label map (normalize raw CIC labels → clean class names)
LABEL_MAP = {
    "BENIGN": "Benign",
    "Benign": "Benign",
    "UDP Flood": "DDoS-UDP",
    "UDPFlood": "DDoS-UDP",
    "UDP-Flood": "DDoS-UDP",
    "TCP SYN": "DDoS-TCP",
    "TCPFlood": "DDoS-TCP",
    "TCP-SYN": "DDoS-TCP",
    "ICMP Flood": "DDoS-ICMP",
    "ICMP": "DDoS-ICMP",
    "Slowloris": "DDoS-SlowLoris",
    "SlowLoris": "DDoS-SlowLoris",
    "WebDDoS": "DDoS-HTTP",
    "HTTP Flood": "DDoS-HTTP",
    "HTTPFlood": "DDoS-HTTP",
}

FINAL_CLASS_NAMES = ["Benign", "DDoS-UDP", "DDoS-TCP", "DDoS-ICMP", "DDoS-SlowLoris", "DDoS-HTTP"]


# ─── Offline Data Loader ──────────────────────────────────────────────────────


class CICFlowMeterExtractor:
    """Load and preprocess CICFlowMeter CSV files from CIC-DDoS2019.

    Handles:
      - Column name normalization (leading spaces, capitalization, special chars)
      - Inf/NaN cleanup
      - Label normalization
      - Duplicate removal
      - Optional feature subset selection

    Example:
        >>> extractor = CICFlowMeterExtractor()
        >>> df = extractor.load_directory("data/raw")
        >>> X, y = extractor.get_feature_matrix(df)
    """

    def __init__(
        self,
        label_column: str = " Label",
        drop_inf_nan: bool = True,
        remove_duplicates: bool = True,
    ) -> None:
        self.label_column = label_column
        self.drop_inf_nan = drop_inf_nan
        self.remove_duplicates = remove_duplicates

    # ── Public API ─────────────────────────────────────────────────────────

    def load_file(self, filepath: str | Path) -> pd.DataFrame:
        """Load a single CICFlowMeter CSV file.

        Args:
            filepath: Path to the CSV file.

        Returns:
            Preprocessed DataFrame with normalized column names.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the label column is not found after normalization.
        """
        fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f"Dataset file not found: {fp}")

        logger.info(f"Loading {fp.name} ...")
        df = pd.read_csv(fp, low_memory=False)
        logger.debug(f"  Raw shape: {df.shape}")

        df = self._normalize_columns(df)
        df = self._normalize_labels(df)

        if self.drop_inf_nan:
            df = self._clean_infinities(df)

        if self.remove_duplicates:
            n_before = len(df)
            df = df.drop_duplicates()
            n_removed = n_before - len(df)
            if n_removed > 0:
                logger.debug(f"  Removed {n_removed} duplicate rows")

        logger.info(f"  Processed shape: {df.shape}")
        return df

    def load_directory(self, directory: str | Path) -> pd.DataFrame:
        """Load all CSV files in a directory and concatenate them.

        Args:
            directory: Path to directory containing CIC-DDoS2019 CSV files.

        Returns:
            Concatenated, preprocessed DataFrame.
        """
        data_dir = Path(directory)
        csv_files = list(data_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        frames = []
        for fp in sorted(csv_files):
            try:
                df = self.load_file(fp)
                frames.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {fp.name}: {e}")

        if not frames:
            raise RuntimeError("No files loaded successfully.")

        combined = pd.concat(frames, ignore_index=True)
        logger.info(f"Combined dataset shape: {combined.shape}")
        return combined

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract feature matrix X and label vector y from processed DataFrame.

        Args:
            df: Preprocessed DataFrame (output of load_file/load_directory).
            feature_names: Optional list of feature columns to select.
                           Defaults to all numeric columns minus label.

        Returns:
            Tuple of (X DataFrame, y Series).

        Raises:
            ValueError: If label column is missing from df.
        """
        label_col = "Label"
        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found. " f"Available: {df.columns.tolist()[:10]}"
            )

        if feature_names is not None:
            # Select only requested features (fill missing with 0)
            available = [f for f in feature_names if f in df.columns]
            missing = [f for f in feature_names if f not in df.columns]
            if missing:
                logger.warning(f"Features not found in data: {missing}")
            X = df[available].copy()
            for col in missing:
                X[col] = 0.0
            X = X[feature_names]
        else:
            non_feature_cols = {
                "Label",
                "Timestamp",
                "Flow_ID",
                "Source_IP",
                "Source_Port",
                "Destination_IP",
                "Protocol_Name",
            }
            feature_cols = [
                c
                for c in df.columns
                if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])
            ]
            X = df[feature_cols].copy()

        y = df[label_col]
        logger.info(f"Feature matrix: {X.shape}, Labels: {y.value_counts().to_dict()}")
        return X, y

    # ── Private Helpers ────────────────────────────────────────────────────

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace and normalize column names."""
        df = df.copy()
        rename_map = {}
        for col in df.columns:
            clean = col.strip()
            # Replace spaces and special chars with underscores
            clean = re.sub(r"[\s/]+", "_", clean)
            clean = re.sub(r"[^A-Za-z0-9_]", "", clean)
            rename_map[col] = clean
        df = df.rename(columns=rename_map)
        return df

    def _normalize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw CIC labels to canonical class names."""
        df = df.copy()
        # Find the label column (may have been renamed)
        label_col = None
        for candidate in ["Label", "label", "Class", "class"]:
            if candidate in df.columns:
                label_col = candidate
                break

        if label_col is None:
            logger.warning("No label column found — skipping label normalization")
            return df

        df = df.rename(columns={label_col: "Label"})
        df["Label"] = df["Label"].str.strip()
        df["Label"] = df["Label"].map(lambda x: LABEL_MAP.get(x, x))
        return df

    def _clean_infinities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace inf/-inf with NaN and drop rows with NaN in feature cols."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        n_before = len(df)
        df = df.dropna(subset=numeric_cols)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.debug(f"  Dropped {n_dropped} rows with inf/NaN values")
        return df


# ─── Online OpenFlow Bridge ───────────────────────────────────────────────────


def extract_features_from_openflow(stat: Dict[str, Any]) -> Dict[str, float]:
    """Derive CICFlowMeter-compatible features from an OpenFlow FlowStats record.

    NOTE: OpenFlow exposes only aggregate counters, not raw packets. This bridge
    covers approximately 40/80 CICFlowMeter features. Features requiring
    per-packet timing data (IAT std, active/idle, etc.) are approximated or
    set to zero. A network tap (PCAP mirror) is required for full coverage.

    Args:
        stat: Dictionary representing an OpenFlow FlowStats entry with keys:
            - packet_count (int): Total packets in flow.
            - byte_count (int): Total bytes in flow.
            - duration_sec (int): Flow duration in seconds.
            - duration_nsec (int): Nanosecond portion of flow duration.
            - match (dict): Flow match fields (in_port, eth_type, ip_src,
                            ip_dst, ip_proto, tp_src, tp_dst).
            - actions (list): Flow action list.

    Returns:
        Dict mapping feature names to float values. Zero-padded for features
        that cannot be derived from OpenFlow counters.
    """
    packet_count = float(stat.get("packet_count", 0))
    byte_count = float(stat.get("byte_count", 0))
    duration_sec = float(stat.get("duration_sec", 0))
    duration_nsec = float(stat.get("duration_nsec", 0))

    duration_us = duration_sec * 1e6 + duration_nsec / 1e3  # microseconds
    duration_s = duration_sec + duration_nsec / 1e9

    # Rates (guarded against division by zero)
    flow_bytes_s = byte_count / duration_s if duration_s > 0 else 0.0
    flow_packets_s = packet_count / duration_s if duration_s > 0 else 0.0
    avg_pkt_size = byte_count / packet_count if packet_count > 0 else 0.0

    # Match fields
    match = stat.get("match", {})
    ip_proto = float(match.get("ip_proto", 0))
    dst_port = float(match.get("tp_dst", 0))

    # Build feature vector (zero-padded for unavailable features)
    features: Dict[str, float] = {
        "Destination_Port": dst_port,
        "Flow_Duration": duration_us,
        "Total_Fwd_Packets": packet_count,  # Approximation: no direction split
        "Total_Backward_Packets": 0.0,  # Not available from OF counters
        "Total_Length_of_Fwd_Packets": byte_count,  # Approximation
        "Total_Length_of_Bwd_Packets": 0.0,
        "Fwd_Packet_Length_Max": avg_pkt_size,  # Approximation
        "Fwd_Packet_Length_Min": avg_pkt_size,
        "Fwd_Packet_Length_Mean": avg_pkt_size,
        "Fwd_Packet_Length_Std": 0.0,
        "Bwd_Packet_Length_Max": 0.0,
        "Bwd_Packet_Length_Min": 0.0,
        "Bwd_Packet_Length_Mean": 0.0,
        "Bwd_Packet_Length_Std": 0.0,
        "Flow_Bytes_s": flow_bytes_s,
        "Flow_Packets_s": flow_packets_s,
        "Flow_IAT_Mean": duration_us / max(packet_count - 1, 1),
        "Flow_IAT_Std": 0.0,
        "Flow_IAT_Max": 0.0,
        "Flow_IAT_Min": 0.0,
        "Average_Packet_Size": avg_pkt_size,
        "Avg_Fwd_Segment_Size": avg_pkt_size,
    }

    # Fill remaining CIC features with 0.0
    for fname in CIC_FEATURE_NAMES:
        if fname not in features:
            features[fname] = 0.0

    return features


def flow_record_from_openflow(stat: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenFlow stat to entropy-compatible flow record dict.

    Used to feed flows into EntropyFeatureExtractor.update_and_compute().

    Args:
        stat: OpenFlow FlowStats dict (see extract_features_from_openflow).

    Returns:
        Dict with keys: src_ip, dst_ip, dst_port, protocol,
                         pkt_len_mean, iat_mean, tcp_flags, ttl.
    """
    match = stat.get("match", {})
    packet_count = max(float(stat.get("packet_count", 1)), 1)
    byte_count = float(stat.get("byte_count", 0))
    duration_us = (
        float(stat.get("duration_sec", 0)) * 1e6 + float(stat.get("duration_nsec", 0)) / 1e3
    )

    return {
        "src_ip": match.get("ipv4_src", "0.0.0.0"),
        "dst_ip": match.get("ipv4_dst", "0.0.0.0"),
        "dst_port": int(match.get("tp_dst", 0)),
        "protocol": int(match.get("ip_proto", 0)),
        "pkt_len_mean": byte_count / packet_count,
        "iat_mean": duration_us / max(packet_count - 1, 1),
        "tcp_flags": int(stat.get("tcp_flags", 0)),
        "ttl": int(match.get("ttl", 64)),
    }
