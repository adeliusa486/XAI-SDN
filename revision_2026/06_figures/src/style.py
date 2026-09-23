"""The two house styles already used by this paper's figures, reproduced exactly.

Reviewer 5 asked for larger type in the figures, and no reviewer asked for a
different visual language, so the colours, hatching, spines, grid, legend frames
and line weights here are copied from the original generators
(paper/figures/generate_ieee_figs.py and generate_new_plots.py). Only the type
sizes change, and they change through SCALE so the change is one number.

  ieee_style()   serif, IEEE blue/orange, dashed grey grid, white axes
                 used by the ablation figure
  okabe_style()  sans-serif, Okabe-Ito palette, white grid on #F8F9FA axes
                 used by the throughput and attribution figures
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCALE = 1.35          # Reviewer 5: the submitted figures were too small to read

# IEEE palette (generate_ieee_figs.py)
C_BLUE = "#1F6BB0"
C_ORANGE = "#E07B28"
C_GREEN = "#2CA02C"
C_RED = "#D62728"
C_GRAY = "#7F7F7F"
C_PURPLE = "#9467BD"

# Okabe-Ito palette (generate_new_plots.py)
O_BLUE = "#0072B2"
O_TEAL = "#009E73"
O_ORANGE = "#E69F00"
O_RED = "#D55E00"
O_GRAY = "#999999"
O_SKY = "#56B4E9"
O_PINK = "#CC79A7"

HATCH_ACC = ""
HATCH_F1 = "//"


def _sized(base: dict) -> dict:
    out = dict(base)
    for k in ("axes.labelsize", "axes.titlesize", "xtick.labelsize",
              "ytick.labelsize", "legend.fontsize", "font.size"):
        if k in out:
            out[k] = round(out[k] * SCALE, 1)
    return out


# IEEE PDF eXpress flags Type 3 fonts, which is what matplotlib embeds by
# default. Type 42 embeds the glyphs as TrueType instead: same drawing, same
# metrics, but a font the IEEE production toolchain accepts.
_TRUETYPE = {"pdf.fonttype": 42, "ps.fonttype": 42}


def ieee_style() -> None:
    plt.rcParams.update(_TRUETYPE)
    plt.rcParams.update(_sized({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#AAAAAA",
        "legend.framealpha": 1.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }))


def okabe_style() -> None:
    plt.rcParams.update(_TRUETYPE)
    plt.rcParams.update(_sized({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FA",
        "axes.edgecolor": "#D3D3D3",
        "axes.grid": True,
        "grid.color": "#FFFFFF",
        "grid.linewidth": 1.5,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#D3D3D3",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }))


def save(fig_dir, name: str) -> None:
    from pathlib import Path
    d = Path(fig_dir)
    d.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(d / f"{name}.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  wrote {d / (name + '.pdf')}")
