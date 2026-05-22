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
    """Full preprocessing pipeline for CIC-DDoS2019 using OfflineFeaturePipeline."""
    import json
    import joblib
    from features.pipeline import OfflineFeaturePipeline, ALL_FEATURE_NAMES

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("XAI-SDN Data Preprocessing")
    logger.info("=" * 60)

    # Use the unified, non-leaky OfflineFeaturePipeline
    logger.info("Initializing OfflineFeaturePipeline...")
    pipeline = OfflineFeaturePipeline(test_size=0.30, random_state=42)
    
    logger.info("Executing pipeline (split-first, independent entropy, scaling)...")
    X_train, X_test, y_train, y_test = pipeline.run(data_dir)

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Step 9: Save
    np.save(out_path / "X_train.npy", X_train)
    np.save(out_path / "X_test.npy", X_test)
    np.save(out_path / "y_train.npy", y_train)
    np.save(out_path / "y_test.npy", y_test)

    # Save encoders & metadata
    joblib.dump(pipeline.label_encoder, out_path / "label_encoder.pkl")
    joblib.dump(pipeline.scaler, out_path / "scaler.pkl")
    
    with open(out_path / "feature_names.json", "w") as f:
        json.dump(ALL_FEATURE_NAMES, f, indent=2)

    with open(out_path / "class_distribution.txt", "w") as f:
        for cls_id, cls_name in enumerate(pipeline.label_encoder.classes_):
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
