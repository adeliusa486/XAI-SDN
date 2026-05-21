"""
preprocess_data.py — Preprocess CIC-DDoS2019 CSV files for training.

Steps:
  1. Load all CSV files from data/raw/
  2. Clean inf/NaN, normalize labels
  3. Sort by timestamp
  4. Compute entropy features (offline simulation)
  5. Save processed X, y arrays to data/splits/

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --data-dir data/raw --output-dir data/splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))


def main(data_dir: str = "data/raw", output_dir: str = "data/splits") -> None:
    """Full preprocessing pipeline for CIC-DDoS2019."""
    from features.cicflowmeter import CICFlowMeterExtractor
    from features.entropy import compute_entropy_features_offline, ENTROPY_FEATURE_NAMES
    from features.cicflowmeter import CIC_FEATURE_NAMES
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("XAI-SDN Data Preprocessing")
    logger.info("=" * 60)

    # Step 1: Load
    logger.info(f"Loading CSVs from {data_dir}...")
    extractor = CICFlowMeterExtractor()
    df = extractor.load_directory(data_dir)

    # Step 2: Sort by timestamp
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp").reset_index(drop=True)
        logger.info("Sorted by Timestamp.")
    else:
        logger.warning("No Timestamp column — using file order.")

    # Step 3: Extract CIC features
    X_cic, y_raw = extractor.get_feature_matrix(df)
    logger.info(f"CIC features: {X_cic.shape}")

    # Step 4: Build flow records for entropy
    logger.info("Building flow records for entropy computation...")
    flow_records = []
    for i in range(len(df)):
        row = X_cic.iloc[i]
        flow_records.append(
            {
                "src_ip": str(df.iloc[i].get("Source_IP", f"10.0.0.{i%254}")),
                "dst_ip": str(df.iloc[i].get("Destination_IP", "10.0.0.1")),
                "dst_port": int(row.get("Destination_Port", 0)),
                "protocol": int(df.iloc[i].get("Protocol", 0)),
                "pkt_len_mean": float(
                    row.get("Packet_Length_Mean", row.get("Fwd_Packet_Length_Mean", 0))
                ),
                "iat_mean": float(row.get("Flow_IAT_Mean", 0)),
                "tcp_flags": int(row.get("SYN_Flag_Count", 0)),
                "ttl": 64,
            }
        )

    # Step 5: Compute entropy features
    logger.info("Computing entropy features (N=1000 window)...")
    entropy_arr = compute_entropy_features_offline(flow_records, window_size=1000)
    entropy_df = pd.DataFrame(entropy_arr, columns=ENTROPY_FEATURE_NAMES)

    # Step 6: Assemble full 88-dim matrix
    X_full = pd.concat([X_cic.reset_index(drop=True), entropy_df], axis=1)
    logger.info(f"Full feature matrix: {X_full.shape}")

    # Step 7: Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    # Step 8: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full.values, y_enc, test_size=0.30, stratify=y_enc, random_state=42
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Step 9: Save
    np.save(out_path / "X_train.npy", X_train)
    np.save(out_path / "X_test.npy", X_test)
    np.save(out_path / "y_train.npy", y_train)
    np.save(out_path / "y_test.npy", y_test)

    import json, joblib

    joblib.dump(le, out_path / "label_encoder.pkl")
    with open(out_path / "feature_names.json", "w") as f:
        all_names = CIC_FEATURE_NAMES + ENTROPY_FEATURE_NAMES
        json.dump(all_names, f, indent=2)

    with open(out_path / "class_distribution.txt", "w") as f:
        for cls_id, cls_name in enumerate(le.classes_):
            n_train = int((y_train == cls_id).sum())
            n_test = int((y_test == cls_id).sum())
            f.write(f"{cls_name:20s}  train={n_train:6d}  test={n_test:5d}\n")

    logger.info(f"Splits saved to {out_path}/")
    logger.info("Preprocessing complete ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/splits")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
