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
        ttl_bin_size: int = 5,
        test_size: float = 0.30,
        random_state: int = 42,
    ) -> None:
        self.window_size = window_size
        self.pkt_len_bin_size = pkt_len_bin_size
        self.iat_bin_size = iat_bin_size
        self.ttl_bin_size = ttl_bin_size
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

        Args:
            data_dir: Directory containing CIC-DDoS2019 CSV files.

        Returns:
            Tuple (X_train, X_test, y_train, y_test) as numpy arrays.
        """
        from sklearn.model_selection import train_test_split

        # Step 1: Load data
        logger.info("Step 1/5: Loading CICFlowMeter CSV files...")
        df = self.cic_extractor.load_directory(data_dir)
        X_cic, y_raw = self.cic_extractor.get_feature_matrix(df)

        # Step 2: Sort by timestamp if available
        logger.info("Step 2/5: Sorting by timestamp for temporal ordering...")
        if "Timestamp" in df.columns:
            sort_idx = df["Timestamp"].argsort().values
            X_cic = X_cic.iloc[sort_idx].reset_index(drop=True)
            y_raw = y_raw.iloc[sort_idx].reset_index(drop=True)
            df = df.iloc[sort_idx].reset_index(drop=True)
        else:
            logger.warning("No Timestamp column found; using file order.")

        # Step 3: Compute entropy features
        logger.info(f"Step 3/5: Computing entropy features (window={self.window_size})...")
        flow_records = self._dataframe_to_flow_records(df, X_cic)
        entropy_arr = compute_entropy_features_offline(
            flow_records,
            window_size=self.window_size,
            pkt_len_bin_size=self.pkt_len_bin_size,
            iat_bin_size=self.iat_bin_size,
            ttl_bin_size=self.ttl_bin_size,
        )
        entropy_df = pd.DataFrame(entropy_arr, columns=ENTROPY_FEATURE_NAMES)

        # Step 4: Concatenate
        logger.info("Step 4/5: Concatenating feature matrix (88-dim)...")
        X_full = pd.concat([X_cic.reset_index(drop=True), entropy_df], axis=1)
        # Ensure canonical feature order
        X_full = self._ensure_feature_columns(X_full)

        # Step 5: Encode labels and split
        logger.info("Step 5/5: Encoding labels and splitting...")
        y_enc = self.label_encoder.fit_transform(y_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X_full.values,
            y_enc,
            test_size=self.test_size,
            stratify=y_enc,
            random_state=self.random_state,
        )

        # Scale features (fit on train only)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

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
        y_enc = self.label_encoder.fit_transform(y_raw)
        return X_full.values, y_enc

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _dataframe_to_flow_records(df: pd.DataFrame, X_cic: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame rows to flow record dicts for entropy computation."""
        records = []
        for i in range(len(df)):
            row = df.iloc[i] if len(df) == len(X_cic) else X_cic.iloc[i]
            xrow = X_cic.iloc[i]
            records.append(
                {
                    "src_ip": str(df.iloc[i].get("Source_IP", f"10.0.0.{i % 254}")),
                    "dst_ip": str(df.iloc[i].get("Destination_IP", "10.0.0.1")),
                    "dst_port": int(xrow.get("Destination_Port", 0)),
                    "protocol": int(df.iloc[i].get("Protocol", xrow.get("Protocol", 0))),
                    "pkt_len_mean": float(
                        xrow.get("Packet_Length_Mean", xrow.get("Fwd_Packet_Length_Mean", 0))
                    ),
                    "iat_mean": float(xrow.get("Flow_IAT_Mean", 0)),
                    "tcp_flags": int(xrow.get("SYN_Flag_Count", 0)),
                    "ttl": 64,  # TTL not available in CIC CSVs; use default
                }
            )
        return records

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
        ttl_bin_size: int = 5,
    ) -> None:
        self.scaler = scaler
        self.entropy_extractor = EntropyFeatureExtractor(
            window_size=window_size,
            pkt_len_bin_size=pkt_len_bin_size,
            iat_bin_size=iat_bin_size,
            ttl_bin_size=ttl_bin_size,
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
                          pkt_len_mean, iat_mean, tcp_flags, ttl).

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
