"""Centralized seed initialization for reproducible experiments."""

from __future__ import annotations
import os
import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: Integer seed value (use 42 for canonical runs).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional: fix scikit-learn global state
    # sklearn uses numpy random state internally

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not required for RF pipeline


def seed_from_env(default: int = 42) -> int:
    """Read seed from XAI_SDN_SEED env var or return default."""
    return int(os.environ.get("XAI_SDN_SEED", default))
