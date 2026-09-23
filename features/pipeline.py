"""
pipeline.py — Full Feature Pipeline for XAI-SDN.

Combines CICFlowMeter statistical features (80-dim) with Shannon entropy
features (8-dim) into the final 88-dimensional feature vector used by the
Random Forest classifier.

Two modes:
  - Offline: processes pandas DataFrames loaded from CIC-DDoS2019 CSVs.
  - Online:  processes individual flow records in streaming / real-time mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder, StandardScaler

from features.cicflowmeter import (
    CIC_FEATURE_NAMES,
    CICFlowMeterExtractor,
    extract_features_from_openflow,
    flow_record_from_openflow,
)
from features.entropy import (
    ENTROPY_FEATURE_NAMES,
    EntropyFeatureExtractor,
    compute_entropy_features_offline,
)

ALL_FEATURE_NAMES: List[str] = CIC_FEATURE_NAMES + ENTROPY_FEATURE_NAMES
N_FEATURES = len(ALL_FEATURE_NAMES)  # 88


# ─── Offline Pipeline ─────────────────────────────────────────────────────────


class OfflineFeaturePipeline:
    """Full offline feature pipeline for training on CIC-DDoS2019.

    Steps:
      1. Load and clean CICFlowMeter CSV files.
      2. Sort by timestamp for temporal ordering.
      3. Simulate sliding window entropy computation.
      4. Concatenate CIC (80) + entropy (8) = 88-dim feature matrix.
      5. Encode labels with LabelEncoder.
      6. Apply StandardScaler (fit on train, transform on test).

    Example:
        >>> pipeline = OfflineFeaturePipeline()
        >>> X_train, X_test, y_train, y_test = pipeline.run("data/raw")
    """

    def __init__(
        self,
        window_size: int = 1000,
        pkt_len_bin_size: int = 10,
        iat_bin_size: int = 1000,
        src_port_bin_size: int = 1,
        test_size: float = 0.30,
        random_state: int = 42,
    ) -> None:
        self.window_size = window_size
        self.pkt_len_bin_size = pkt_len_bin_size
        self.iat_bin_size = iat_bin_size
        self.src_port_bin_size = src_port_bin_size
        self.test_size = test_size
        self.random_state = random_state

        self.cic_extractor = CICFlowMeterExtractor()
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        self._fitted = False

    def run(
        self,
        data_dir: str | Path,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute the full offline feature pipeline.

        NOTE ON ENTROPY ORDERING: The train/test split is performed on the raw
        CIC features BEFORE entropy computation. Shannon entropy features are
        then computed independently on each partition using a fresh extractor
        state, which eliminates temporal contamination between partitions.

        Args:
            data_dir: Directory containing CIC-DDoS2019 CSV files.

        Returns:
            Tuple (X_train, X_test, y_train, y_test) as numpy arrays.
        """
        from sklearn.model_selection import train_test_split

        # Step 1: Load data
        logger.info("Step 1/6: Loading CICFlowMeter CSV files...")
        df = self.cic_extractor.load_directory(data_dir)
        X_cic, y_raw = self.cic_extractor.get_feature_matrix(df)

        # Step 2: Sort by timestamp if available
        logger.info("Step 2/6: Sorting by timestamp for temporal ordering...")
        if "Timestamp" in df.columns:
            sort_idx = df["Timestamp"].argsort().values
            X_cic = X_cic.iloc[sort_idx].reset_index(drop=True)
            y_raw = y_raw.iloc[sort_idx].reset_index(drop=True)
            df = df.iloc[sort_idx].reset_index(drop=True)
        else:
            logger.warning("No Timestamp column found; using file order.")

        # Step 3: Split CIC features FIRST — prevents entropy temporal contamination
        # Entropy is computed on the full time-ordered dataset before the split,
        # which would allow test-partition flow context to leak into train-partition
        # entropy windows. Splitting first and computing entropy separately on each
        # partition is the correct approach.
        logger.info("Step 3/6: Splitting CIC features before entropy computation...")
        indices = np.arange(len(X_cic))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            stratify=y_raw.values,
            random_state=self.random_state,
        )
        X_cic_train = X_cic.iloc[train_idx].reset_index(drop=True)
        X_cic_test = X_cic.iloc[test_idx].reset_index(drop=True)
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        y_train_raw = y_raw.iloc[train_idx].values
        y_test_raw = y_raw.iloc[test_idx].values

        # Step 4: Compute entropy independently on each partition
        # Each partition starts with a fresh extractor state — no cross-partition leakage.
        logger.info(f"Step 4/6: Computing entropy features per partition (window={self.window_size})...")
        train_records = self._dataframe_to_flow_records(df_train, X_cic_train)
        entropy_train = compute_entropy_features_offline(
            train_records,
            window_size=self.window_size,
            pkt_len_bin_size=self.pkt_len_bin_size,
            iat_bin_size=self.iat_bin_size,
            src_port_bin_size=self.src_port_bin_size,
        )
        test_records = self._dataframe_to_flow_records(df_test, X_cic_test)
        entropy_test = compute_entropy_features_offline(
            test_records,
            window_size=self.window_size,
            pkt_len_bin_size=self.pkt_len_bin_size,
            iat_bin_size=self.iat_bin_size,
            src_port_bin_size=self.src_port_bin_size,
        )

        # Step 5: Concatenate CIC + entropy features per partition
        logger.info("Step 5/6: Concatenating feature matrix (88-dim per partition)...")
        entropy_train_df = pd.DataFrame(entropy_train, columns=ENTROPY_FEATURE_NAMES)
        entropy_test_df = pd.DataFrame(entropy_test, columns=ENTROPY_FEATURE_NAMES)
        X_train_full = pd.concat([X_cic_train, entropy_train_df], axis=1)
        X_test_full = pd.concat([X_cic_test, entropy_test_df], axis=1)
        X_train_full = self._ensure_feature_columns(X_train_full)
        X_test_full = self._ensure_feature_columns(X_test_full)

        # Step 6: Encode labels and scale features
        logger.info("Step 6/6: Encoding labels and scaling features...")
        self.label_encoder.fit(y_train_raw)
        y_train = self.label_encoder.transform(y_train_raw)
        y_test = self.label_encoder.transform(y_test_raw)

        X_train = self.scaler.fit_transform(X_train_full.values)
        X_test = self.scaler.transform(X_test_full.values)

        self._fitted = True
        logger.info(
            f"Pipeline complete. Train: {X_train.shape}, Test: {X_test.shape}. "
            f"Classes: {list(self.label_encoder.classes_)}"
        )
        return X_train, X_test, y_train, y_test


    def run_on_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Run pipeline on a pre-loaded DataFrame (without split or scaling).

        Useful for adding entropy features to already-loaded data for
        inspection or notebook use.

        Returns:
            (X_full array shape [n,88], y_encoded array shape [n]).
        """
        X_cic, y_raw = self.cic_extractor.get_feature_matrix(df)
        flow_records = self._dataframe_to_flow_records(df, X_cic)
        entropy_arr = compute_entropy_features_offline(flow_records, window_size=self.window_size)
        entropy_df = pd.DataFrame(entropy_arr, columns=ENTROPY_FEATURE_NAMES)
        X_full = pd.concat([X_cic.reset_index(drop=True), entropy_df], axis=1)
        X_full = self._ensure_feature_columns(X_full)
        return X_full.values, y_raw.values

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _dataframe_to_flow_records(df: pd.DataFrame, X_cic: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame rows to flow record dicts for entropy computation."""
        n = len(df)
        
        # 1. Source IP
        src_ip_col = None
        for candidate in ["Source_IP", "Src_IP", "Source IP", "Src IP"]:
            if candidate in df.columns:
                src_ip_col = candidate
                break
        src_ips = df[src_ip_col].astype(str).values if src_ip_col is not None else [f"10.0.0.{i % 254}" for i in range(n)]

        # 2. Destination IP
        dst_ip_col = None
        for candidate in ["Destination_IP", "Dst_IP", "Destination IP", "Dst IP"]:
            if candidate in df.columns:
                dst_ip_col = candidate
                break
        dst_ips = df[dst_ip_col].astype(str).values if dst_ip_col is not None else ["10.0.0.1"] * n

        # 3. Destination Port
        dst_port_col = None
        for candidate in ["Destination_Port", "Destination Port"]:
            if candidate in X_cic.columns:
                dst_port_col = candidate
                break
        dst_ports = X_cic[dst_port_col].values.astype(int) if dst_port_col is not None else np.zeros(n, dtype=int)

        # 4. Protocol
        proto_col = None
        for candidate in ["Protocol", "protocol", "Proto", "proto"]:
            if candidate in df.columns:
                proto_col = candidate
                break
        protocols = df[proto_col].values.astype(int) if proto_col is not None else np.zeros(n, dtype=int)

        # 5. Packet Length Mean
        pkt_len_col = None
        for candidate in ["Packet_Length_Mean", "Fwd_Packet_Length_Mean", "Packet Length Mean", "Fwd Packet Length Mean"]:
            if candidate in X_cic.columns:
                pkt_len_col = candidate
                break
        pkt_lens = X_cic[pkt_len_col].values.astype(float) if pkt_len_col is not None else np.zeros(n, dtype=float)

        # 6. Flow IAT Mean
        iat_col = None
        for candidate in ["Flow_IAT_Mean", "Flow IAT Mean"]:
            if candidate in X_cic.columns:
                iat_col = candidate
                break
        iats = X_cic[iat_col].values.astype(float) if iat_col is not None else np.zeros(n, dtype=float)

        # 7. TCP Flags
        tcp_col = None
        for candidate in ["SYN_Flag_Count", "SYN Flag Count"]:
            if candidate in X_cic.columns:
                tcp_col = candidate
                break
        tcp_flags = X_cic[tcp_col].values.astype(int) if tcp_col is not None else np.zeros(n, dtype=int)

        # 8. Source Port (published feature set uses source-port entropy)
        src_port_col = None
        for candidate in ["Source_Port", "Src_Port", "Source Port", "Src Port", "source_port"]:
            if candidate in df.columns:
                src_port_col = candidate
                break
        src_ports = (df[src_port_col].values.astype(int)
                     if src_port_col is not None else np.zeros(n, dtype=int))

        return [
            {
                "src_ip": src_ips[i],
                "dst_ip": dst_ips[i],
                "dst_port": int(dst_ports[i]),
                "protocol": int(protocols[i]),
                "pkt_len_mean": float(pkt_lens[i]),
                "iat_mean": float(iats[i]),
                "tcp_flags": int(tcp_flags[i]),
                "src_port": int(src_ports[i]),
            }
            for i in range(n)
        ]

    def _ensure_feature_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure all 88 feature columns exist, adding zeros for missing."""
        for col in ALL_FEATURE_NAMES:
            if col not in X.columns:
                X[col] = 0.0
        return X[ALL_FEATURE_NAMES]


# ─── Online (Real-Time) Pipeline ──────────────────────────────────────────────


class OnlineFeaturePipeline:
    """Stateful real-time feature pipeline for SDN controller integration.

    Maintains a sliding entropy window and processes one flow at a time.
    Must be initialized with a fitted scaler from offline training.

    Example:
        >>> pipeline = OnlineFeaturePipeline.from_artifacts("model/artifacts")
        >>> for of_stat in openflow_stats:
        ...     x = pipeline.process_openflow_stat(of_stat)
        ...     # x is ready for clf.predict([x])
    """

    def __init__(
        self,
        scaler: StandardScaler,
        window_size: int = 1000,
        pkt_len_bin_size: int = 10,
        iat_bin_size: int = 1000,
        src_port_bin_size: int = 1,
    ) -> None:
        self.scaler = scaler
        self.entropy_extractor = EntropyFeatureExtractor(
            window_size=window_size,
            pkt_len_bin_size=pkt_len_bin_size,
            iat_bin_size=iat_bin_size,
            src_port_bin_size=src_port_bin_size,
        )

    @classmethod
    def from_artifacts(cls, artifacts_dir: str | Path, **kwargs) -> "OnlineFeaturePipeline":
        """Load scaler from serialized artifacts.

        Args:
            artifacts_dir: Directory containing scaler.pkl.
            **kwargs: Forwarded to __init__.
        """
        import joblib

        artifacts_dir = Path(artifacts_dir)
        scaler = joblib.load(artifacts_dir / "scaler.pkl")
        return cls(scaler=scaler, **kwargs)

    def process_openflow_stat(self, stat: Dict[str, Any]) -> np.ndarray:
        """Process a single OpenFlow FlowStats record into a scaled 88-dim vector.

        Args:
            stat: OpenFlow FlowStats dict.

        Returns:
            Scaled feature vector of shape (88,), ready for RF prediction.
        """
        # Extract CIC features from OpenFlow counters
        cic_features = extract_features_from_openflow(stat)
        cic_array = np.array(
            [cic_features.get(name, 0.0) for name in CIC_FEATURE_NAMES],
            dtype=np.float64,
        )

        # Update entropy window and compute entropy features
        flow_record = flow_record_from_openflow(stat)
        entropy_dict = self.entropy_extractor.update_and_compute(flow_record)
        entropy_array = np.array(
            [entropy_dict[name] for name in ENTROPY_FEATURE_NAMES],
            dtype=np.float64,
        )

        # Concatenate → 88-dim
        x = np.concatenate([cic_array, entropy_array])

        # Scale (replace any inf/nan first)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_scaled = self.scaler.transform(x.reshape(1, -1))[0]
        return x_scaled

    def process_flow_dict(
        self, cic_dict: Dict[str, float], flow_record: Dict[str, Any]
    ) -> np.ndarray:
        """Process pre-computed CIC features + update entropy window.

        Used when CIC features are computed externally (e.g. by a network tap
        running CICFlowMeter directly).

        Args:
            cic_dict: Dict mapping CIC feature names to float values.
            flow_record: Flow record dict for entropy computation
                         (keys: src_ip, dst_ip, dst_port, protocol,
                          pkt_len_mean, iat_mean, tcp_flags, src_port).

        Returns:
            Scaled 88-dim feature vector.
        """
        cic_array = np.array(
            [cic_dict.get(name, 0.0) for name in CIC_FEATURE_NAMES],
            dtype=np.float64,
        )
        entropy_dict = self.entropy_extractor.update_and_compute(flow_record)
        entropy_array = np.array(
            [entropy_dict[name] for name in ENTROPY_FEATURE_NAMES],
            dtype=np.float64,
        )
        x = np.concatenate([cic_array, entropy_array])
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(x.reshape(1, -1))[0]


# ─── Convenience Alias ────────────────────────────────────────────────────────

FeaturePipeline = OfflineFeaturePipeline
