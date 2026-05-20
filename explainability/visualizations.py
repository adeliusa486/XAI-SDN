"""
visualizations.py — SHAP Plot Generation for XAI-SDN.

Generates:
  - Global SHAP bar chart (mean |SHAP| across test set)
  - Local waterfall chart (single-flow attribution)
  - Beeswarm summary plot
  - Feature importance heatmap (per class)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def plot_global_importance(
    importance: Dict[str, float],
    output_path: Optional[str] = None,
    top_n: int = 20,
    title: str = "Global SHAP Feature Importance",
) -> None:
    """Horizontal bar chart of mean absolute SHAP values.

    Args:
        importance: Dict mapping feature name → mean |SHAP| value.
        output_path: File path to save the figure (PNG).
        top_n: Number of top features to display.
        title: Plot title.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available.")
        return

    sorted_feats = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [k for k, _ in sorted_feats]
    values = [v for _, v in sorted_feats]

    # Colour entropy features differently
    colors = ["coral" if name.startswith("H_") else "steelblue" for name in names]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Legend
    handles = [
        mpatches.Patch(color="coral", label="Entropy feature"),
        mpatches.Patch(color="steelblue", label="CICFlowMeter feature"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=10)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()


def plot_local_waterfall(
    attribution: Dict[str, float],
    flow_label: str,
    confidence: float,
    output_path: Optional[str] = None,
    top_k: int = 15,
    base_value: float = 0.0,
) -> None:
    """Waterfall chart of per-feature SHAP values for a single flow.

    Args:
        attribution: Dict mapping feature name → SHAP value.
        flow_label: Predicted class label string.
        confidence: Model confidence probability.
        output_path: Save path (PNG).
        top_k: Top features to display.
        base_value: SHAP base value (expected model output).
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    # Sort by absolute value
    sorted_items = sorted(attribution.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    names = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    colors = ["#d73027" if v > 0 else "#4575b4" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.45)))
    ax.barh(range(len(names)), values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
    ax.set_title(
        f"Local Attribution — Predicted: {flow_label} (confidence: {confidence:.2%})",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    # Annotate bars
    for i, v in enumerate(values):
        ax.text(
            v + (0.005 if v >= 0 else -0.005),
            i,
            f"{v:+.3f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
        )

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Heatmap confusion matrix.

    Args:
        cm: Confusion matrix array (n_classes × n_classes).
        class_names: List of class name strings.
        output_path: Save path (PNG).
        title: Plot title.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
) -> None:
    """Multi-class one-vs-rest ROC curves.

    Args:
        y_true: Integer encoded ground truth labels.
        y_score: Probability matrix (n_samples, n_classes).
        class_names: List of class name strings.
        output_path: Save path (PNG).
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        if y_bin.shape[1] > i:
            fpr_vals, tpr_vals, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr_vals, tpr_vals)
            ax.plot(fpr_vals, tpr_vals, color=color, lw=2, label=f"{cls} (AUC={roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()
