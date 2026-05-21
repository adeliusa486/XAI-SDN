"""
generate_synthetic_data.py — Synthetic CIC-DDoS2019 Dataset Generator.

Generates a synthetic dataset that mimics the structure and distribution
of the CIC-DDoS2019 dataset, allowing the full pipeline to run without
access to the proprietary dataset.

Output: data/synthetic/synthetic_ddos.csv

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --n-samples 50000 --output data/synthetic/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.cicflowmeter import CIC_FEATURE_NAMES, FINAL_CLASS_NAMES, LABEL_MAP
from features.entropy import ENTROPY_FEATURE_NAMES


def generate_synthetic_dataset(
    n_samples: int = 20000,
    random_state: int = 42,
    output_dir: str = "data/synthetic",
) -> pd.DataFrame:
    """Generate a realistic synthetic DDoS dataset.

    Mimics CIC-DDoS2019 statistical properties:
      - Class imbalance (benign ~38%, UDP ~25%, TCP ~23%, ICMP ~9%, Slow ~4%, HTTP ~1%)
      - Attack flows: higher packet rates, lower entropy
      - Benign flows: lower packet rates, higher entropy

    Returns:
        pd.DataFrame with 80 CIC features + 8 entropy features + Label column.
    """
    logger.warning("==========================================================")
    logger.warning("WARNING: SYNTHETIC DATA GENERATION")
    logger.warning("This data is heavily engineered to be linearly separable")
    logger.warning("for CI/CD smoke testing. Models will trivially achieve")
    logger.warning("1.000 F1 scores. DO NOT report these metrics as real!")
    logger.warning("==========================================================")
    
    rng = np.random.RandomState(random_state)

    classes = ["Benign", "DDoS-UDP", "DDoS-TCP", "DDoS-ICMP", "DDoS-SlowLoris", "DDoS-HTTP"]
    props = np.array([0.38, 0.25, 0.23, 0.09, 0.04, 0.01])
    props /= props.sum()

    n_per_class = (props * n_samples).astype(int)
    n_per_class[-1] = n_samples - n_per_class[:-1].sum()

    all_rows = []

    for cls, n in zip(classes, n_per_class):
        if n <= 0:
            continue
        rows = _generate_class_samples(cls, n, rng)
        all_rows.append(rows)

    df = pd.concat(all_rows, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "synthetic_ddos.csv"
    df.to_csv(csv_path, index=False)

    logger.info(f"Synthetic dataset saved: {csv_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Class distribution:\n{df['Label'].value_counts()}")
    return df


def _generate_class_samples(cls: str, n: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Generate n samples for a given class with realistic feature distributions."""
    n_cic = len(CIC_FEATURE_NAMES)
    n_ent = len(ENTROPY_FEATURE_NAMES)

    # Base CIC feature matrix (random, with class-specific biases)
    X_cic = rng.exponential(scale=100, size=(n, n_cic)).astype(np.float32)

    # Entropy features (class-specific)
    X_ent = np.zeros((n, n_ent), dtype=np.float32)

    if cls == "Benign":
        # Legitimate traffic: high diversity → high entropy
        X_cic[:, 0] = rng.choice([80, 443, 8080, 22, 53, 3306], size=n)  # dst_port
        X_cic[:, 14] = rng.exponential(5e4, size=n)  # bytes/s
        X_cic[:, 2] = rng.randint(5, 200, size=n).astype(float)  # fwd packets
        X_ent[:, 0] = rng.uniform(4.0, 7.0, size=n)  # H_src_ip  — high: many sources
        X_ent[:, 1] = rng.uniform(3.5, 6.0, size=n)  # H_dst_ip
        X_ent[:, 2] = rng.uniform(3.0, 5.5, size=n)  # H_dst_port — high: many ports
        X_ent[:, 3] = rng.uniform(1.0, 2.5, size=n)  # H_proto
        X_ent[:, 4] = rng.uniform(3.0, 5.0, size=n)  # H_pkt_len
        X_ent[:, 5] = rng.uniform(2.5, 4.5, size=n)  # H_iat
        X_ent[:, 6] = rng.uniform(1.5, 3.0, size=n)  # H_tcp_flags
        X_ent[:, 7] = rng.uniform(2.0, 4.0, size=n)  # H_ttl

    elif cls == "DDoS-UDP":
        # UDP flood: single source cluster → low src_ip entropy, single port
        X_cic[:, 0] = 53.0  # DNS port
        X_cic[:, 14] = rng.exponential(1e7, size=n)  # very high bytes/s
        X_cic[:, 2] = rng.randint(500, 5000, size=n).astype(float)
        X_ent[:, 0] = rng.uniform(0.1, 1.5, size=n)  # H_src_ip  — LOW: botnet
        X_ent[:, 1] = rng.uniform(0.0, 0.5, size=n)  # H_dst_ip  — very low: single target
        X_ent[:, 2] = rng.uniform(0.0, 0.8, size=n)  # H_dst_port — low: single port
        X_ent[:, 3] = rng.uniform(0.0, 0.2, size=n)  # H_proto   — 0: only UDP
        X_ent[:, 4] = rng.uniform(0.5, 1.5, size=n)  # H_pkt_len
        X_ent[:, 5] = rng.uniform(0.2, 1.0, size=n)  # H_iat
        X_ent[:, 6] = rng.uniform(0.0, 0.5, size=n)  # H_tcp_flags
        X_ent[:, 7] = rng.uniform(0.1, 0.8, size=n)  # H_ttl

    elif cls == "DDoS-TCP":
        # SYN flood: high SYN flags, low H_tcp_flags (only SYN)
        X_cic[:, 0] = rng.choice([80, 443, 22], size=n).astype(float)
        X_cic[:, 43] = rng.randint(100, 1000, size=n).astype(float)  # SYN_Flag_Count
        X_cic[:, 14] = rng.exponential(8e6, size=n)
        X_ent[:, 0] = rng.uniform(0.5, 2.0, size=n)
        X_ent[:, 1] = rng.uniform(0.0, 0.5, size=n)
        X_ent[:, 2] = rng.uniform(0.3, 1.2, size=n)
        X_ent[:, 3] = rng.uniform(0.0, 0.3, size=n)
        X_ent[:, 4] = rng.uniform(0.5, 1.5, size=n)
        X_ent[:, 5] = rng.uniform(0.1, 0.8, size=n)
        X_ent[:, 6] = rng.uniform(0.0, 0.3, size=n)  # H_tcp_flags — very low (SYN only)
        X_ent[:, 7] = rng.uniform(0.2, 1.0, size=n)

    elif cls == "DDoS-ICMP":
        X_cic[:, 14] = rng.exponential(9e6, size=n)
        X_ent[:, 0] = rng.uniform(0.8, 2.0, size=n)
        X_ent[:, 1] = rng.uniform(0.0, 0.5, size=n)
        X_ent[:, 2] = rng.uniform(0.0, 0.2, size=n)  # No ports for ICMP
        X_ent[:, 3] = rng.uniform(0.0, 0.1, size=n)  # Only ICMP
        X_ent[:, 4] = rng.uniform(0.8, 2.0, size=n)
        X_ent[:, 5] = rng.uniform(0.3, 1.0, size=n)
        X_ent[:, 6] = rng.uniform(0.0, 0.0, size=n)  # No TCP flags
        X_ent[:, 7] = rng.uniform(0.5, 1.5, size=n)

    elif cls == "DDoS-SlowLoris":
        # Few packets, long duration, moderate diversity
        X_cic[:, 2] = rng.randint(1, 5, size=n).astype(float)  # Very few fwd packets
        X_cic[:, 1] = rng.exponential(3e7, size=n)  # Long duration
        X_cic[:, 0] = 80.0
        X_ent[:, 0] = rng.uniform(2.5, 4.0, size=n)  # Moderate diversity
        X_ent[:, 1] = rng.uniform(0.0, 0.5, size=n)
        X_ent[:, 2] = rng.uniform(2.0, 3.5, size=n)
        X_ent[:, 3] = rng.uniform(0.5, 1.5, size=n)
        X_ent[:, 4] = rng.uniform(1.0, 2.5, size=n)
        X_ent[:, 5] = rng.uniform(0.1, 0.5, size=n)  # Low IAT entropy (slow)
        X_ent[:, 6] = rng.uniform(0.8, 2.0, size=n)
        X_ent[:, 7] = rng.uniform(1.0, 2.5, size=n)

    elif cls == "DDoS-HTTP":
        X_cic[:, 0] = rng.choice([80, 8080], size=n).astype(float)
        X_cic[:, 14] = rng.exponential(2e5, size=n)
        X_ent[:, 0] = rng.uniform(3.5, 5.0, size=n)
        X_ent[:, 1] = rng.uniform(0.0, 0.5, size=n)
        X_ent[:, 2] = rng.uniform(1.0, 2.0, size=n)
        X_ent[:, 3] = rng.uniform(0.8, 1.5, size=n)
        X_ent[:, 4] = rng.uniform(1.5, 3.0, size=n)
        X_ent[:, 5] = rng.uniform(1.0, 2.5, size=n)
        X_ent[:, 6] = rng.uniform(1.2, 2.5, size=n)
        X_ent[:, 7] = rng.uniform(1.5, 3.0, size=n)

    # Add Gaussian noise
    X_cic += rng.randn(n, n_cic) * 5
    X_ent += rng.randn(n, n_ent) * 0.1
    X_ent = np.clip(X_ent, 0, None)

    # Assemble DataFrame
    df_cic = pd.DataFrame(X_cic, columns=CIC_FEATURE_NAMES)
    df_ent = pd.DataFrame(X_ent, columns=ENTROPY_FEATURE_NAMES)
    df = pd.concat([df_cic, df_ent], axis=1)
    df["Label"] = cls
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic CIC-DDoS2019-style dataset")
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_synthetic_dataset(
        n_samples=args.n_samples,
        random_state=args.seed,
        output_dir=args.output,
    )
