"""
synth_utils.py — Shared helper for loading synthetic data with proper split.

This fixes the API difference: load_synthetic_data() returns (X, y_raw, le)
where y_raw contains string labels. This util encodes and splits properly.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_synthetic_split(
    n_samples: int = 8000,
    random_state: int = 42,
    test_size: float = 0.30,
    binary: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load synthetic data and return (X_train, X_test, y_train, y_test).

    Args:
        binary: If True, binarize labels: Benign=0, all DDoS variants=1
    """
    from model.train import load_synthetic_data

    X, y_raw, le = load_synthetic_data(n_samples=n_samples, random_state=random_state)

    if binary:
        y = (y_raw != "Benign").astype(int)
    else:
        le2 = LabelEncoder()
        y = le2.fit_transform(y_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_tr, X_te, y_tr, y_te
