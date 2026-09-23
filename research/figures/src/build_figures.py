"""Rebuild every figure in the paper from this repository's own results.

The submitted figures were carried over unchanged and several of them now
contradict the tables: the ablation figure shows 100.00% accuracy under a split
the paper no longer uses, and the attribution figure ranks a constant column
third. Each builder below reads a result file and redraws one figure in the
house style of the original generator, with only the type size increased
(The deployment question). A builder whose input is missing is skipped and reported, so this
can be run repeatedly while experiments are still landing.

    python 06_figures/src/build_figures.py            everything available
    python 06_figures/src/build_figures.py fig7 fig8  only these
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "03_experiments" / "common"))

import matplotlib.pyplot as plt  # noqa: E402
import style as S  # noqa: E402
from paths import RESULTS  # noqa: E402

OUT = HERE.parent.parent / "05_manuscript" / "figures"


def load(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return pd.read_csv(p)


# --------------------------------------------------------------------------- #
# fig7: contribution of the entropy augmentation (E17)
# --------------------------------------------------------------------------- #
def fig7() -> str | None:
    d = load("E17_entropy_contribution.json")
    if not d:
        return "E17 not available"
    s = d["summary"]
    order = ["entropy_only", "scrambled", "natural", "cic_only"]
    labels = ["Entropy only\n(8-dim)", "Entropy scrambled\n(88-dim)",
              "Full\n(88-dim)", "Flow statistics only\n(80-dim)"]
    acc = [s[k]["accuracy"]["mean"] * 100 for k in order]
    acc_e = [s[k]["accuracy"]["std"] * 100 for k in order]
    f1 = [s[k]["macro_f1"]["mean"] * 100 for k in order]
    f1_e = [s[k]["macro_f1"]["std"] * 100 for k in order]

    S.ieee_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, acc, w, yerr=acc_e, capsize=3, color=S.C_BLUE,
           hatch=S.HATCH_ACC, edgecolor="white", linewidth=0.6,
           label="Accuracy (%)", error_kw=dict(ecolor="#333333", lw=1.0))
    ax.bar(x + w / 2, f1, w, yerr=f1_e, capsize=3, color=S.C_ORANGE,
           hatch=S.HATCH_F1, edgecolor="white", linewidth=0.6,
           label="Macro F1 (%)", error_kw=dict(ecolor="#333333", lw=1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(40, 104)
    ax.axhline(100, color=S.C_BLUE, linestyle=":", linewidth=0.9, alpha=0.7)
    best = max(range(len(order)), key=lambda i: f1[i])
    ax.annotate(f"{f1[best]:.2f}", (x[best] + w / 2, f1[best]),
                textcoords="offset points", xytext=(0, 6), ha="center",
                fontsize=plt.rcParams["xtick.labelsize"])
    ax.legend(loc="lower right", ncol=2)
    S.save(OUT, "fig7")
    return None


# --------------------------------------------------------------------------- #
# fig4: per-flow cost of each pipeline stage (E10)
# --------------------------------------------------------------------------- #
def fig4() -> str | None:
    d = load("E10_explanation_budget.json")
    if not d:
        return "E10 not available"
    c = d["stage_costs"]
    stages = ["Rolling entropy\n+ RF inference", "TreeSHAP\nexplanation"]
    ms = [c["inference_ms_per_flow"], c["shap_ms_per_explanation"]]

    S.okabe_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.barh(stages, ms, color=[S.O_BLUE, S.O_ORANGE], height=0.5,
                   edgecolor="white", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Per-flow cost (ms, log scale)")
    ax.set_xlim(min(ms) / 3, max(ms) * 6)
    for b, v in zip(bars, ms):
        ax.text(v * 1.25, b.get_y() + b.get_height() / 2,
                f"{v:.4f} ms\n{1000 / v:,.0f} flows/s", va="center",
                fontsize=plt.rcParams["xtick.labelsize"])
    ax.set_title(f"Explanation costs {ms[1] / ms[0]:.0f}$\\times$ a classification")
    S.save(OUT, "fig4")
    return None


# --------------------------------------------------------------------------- #
# fig8: macro F1 against sustained throughput (E1)
# --------------------------------------------------------------------------- #
def fig8() -> str | None:
    d = load("E1_baselines_cpu.json")
    if not d:
        return "E1 not available"
    rows = []
    for proto, res in (d.get("protocols") or d.get("results") or {}).items():
        if not isinstance(res, dict):
            continue
        for name, r in res.items():
            if not isinstance(r, dict) or "macro_f1" not in r:
                continue
            thr = r.get("throughput_flows_per_s") or r.get("batch_throughput_flows_per_s")
            if not thr:
                continue
            rows.append((proto, name, r["macro_f1"] * 100, float(thr)))
    rows = [r for r in rows if r[0] in ("A", "protocol_A", "full")] or rows
    if not rows:
        return "E1 present but no usable throughput rows"

    S.okabe_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    palette = [S.O_BLUE, S.O_TEAL, S.O_ORANGE, S.O_RED, S.O_GRAY, S.O_SKY, S.O_PINK]
    for i, (_, name, f1, thr) in enumerate(sorted(rows, key=lambda r: -r[2])):
        ours = "XAI-SDN" in name or "Random Forest" in name
        ax.scatter(thr, f1, s=180 if ours else 90,
                   color=S.O_RED if ours else palette[i % len(palette)],
                   marker="*" if ours else "o", zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(name, (thr, f1), textcoords="offset points",
                    xytext=(7, 4), fontsize=plt.rcParams["xtick.labelsize"] - 1)
    ax.set_xscale("log")
    ax.set_xlabel("Sustained throughput (flows/s, log scale)")
    ax.set_ylabel("Macro F1 (%)")
    S.save(OUT, "fig8")
    return None


# --------------------------------------------------------------------------- #
# fig10: per-flow attribution beeswarm (E19)
# --------------------------------------------------------------------------- #
PRETTY = {
    "H_src_ip": "$H_{src}$ (IP)", "H_dst_ip": "$H_{dst}$ (IP)",
    "H_dst_port": "$H_{dport}$", "H_src_port": "$H_{sport}$",
    "H_proto": "$H_{proto}$", "H_pkt_len": "$H_{len}$",
    "H_iat": "$H_{iat}$", "H_tcp_flags": "$H_{flags}$",
}


def pretty(name: str) -> str:
    if name in PRETTY:
        return PRETTY[name]
    label = name.replace("_", " ").title()
    # CICFlowMeter names rates as '..._s'; title-casing turns that into a
    # trailing ' S', which reads as a word rather than a unit.
    for suffix in (" S", " Ms"):
        if label.endswith(suffix):
            label = label[: -len(suffix)] + "/" + suffix.strip().lower()
    return label


def fig10() -> str | None:
    d = RESULTS / "shap_sample"
    if not (d / "shap_values.npy").exists():
        return "E19 not available"
    sv = np.load(d / "shap_values.npy")
    fv = np.load(d / "feature_values.npy")
    names = json.loads((d / "feature_names.json").read_text(encoding="utf-8"))

    top = np.argsort(np.abs(sv).mean(axis=0))[::-1][:12][::-1]
    S.okabe_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("coolwarm")
    for row, j in enumerate(top):
        v = fv[:, j].astype(float)
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        c = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
        y = row + rng.uniform(-0.17, 0.17, len(v))
        ax.scatter(sv[:, j], y, c=cmap(c), s=6, alpha=0.65, linewidths=0)
    ax.axvline(0, color="#888888", linewidth=1.0, zorder=1)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([pretty(names[j]) for j in top])
    ax.set_xlabel("SHAP value (impact on model output)")
    lim = np.percentile(np.abs(sv[:, top]), 99.5)
    ax.set_xlim(-lim, lim)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.03)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Feature value")
    S.save(OUT, "fig10")
    return None


# --------------------------------------------------------------------------- #
# fig_rocpr: ROC and precision-recall for every method (E14)
# --------------------------------------------------------------------------- #
def fig_rocpr() -> str | None:
    c = load("E14_curves.csv")
    d = load("E14_roc_pr_curves.json")
    if c is None or not d:
        return "E14 not available"

    # Full-scale protocol only. Mixing protocols on one axis would compare
    # models evaluated on different test partitions.
    keys = [k for k in dict.fromkeys(c["key"]) if k.startswith("E1/A/")]
    if not keys:
        return "E14 present but no full-protocol curves"
    keys.sort(key=lambda k: -d["curves"].get(k, {}).get("roc_auc", 0))

    S.okabe_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    palette = [S.O_RED, S.O_BLUE, S.O_TEAL, S.O_ORANGE, S.O_PINK, S.O_SKY,
               S.O_GRAY, "#444444"]
    styles = ["-", "-", "--", "-.", "--", "-.", ":", ":"]
    prev = None
    handles = []
    for i, k in enumerate(keys):
        sub = c[c["key"] == k]
        info = d["curves"].get(k, {})
        name = info.get("model", k.split("/")[-1]).replace("XAI-SDN ", "")
        ours = "Random Forest" in name
        col = S.O_RED if ours else palette[(i + 1) % len(palette)]
        lw = 2.1 if ours else 1.3
        ls = "-" if ours else styles[i % len(styles)]
        r = sub[sub["curve"] == "roc"]
        ln, = axes[0].plot(r["x"], r["y"], color=col, linewidth=lw, linestyle=ls,
                           label=f"{name}  AUC {info.get('roc_auc', float('nan')):.6f}"
                                 f"  AP {info.get('average_precision', float('nan')):.6f}")
        handles.append(ln)
        pr = sub[sub["curve"] == "pr"]
        axes[1].plot(pr["x"], pr["y"], color=col, linewidth=lw, linestyle=ls)
        prev = info.get("baseline_average_precision", prev)

    axes[0].plot([0, 1], [0, 1], color="#BBBBBB", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_xlim(-0.01, 0.35)
    axes[0].set_ylim(0.0, 1.01)
    axes[0].set_title("ROC")

    if prev:
        axes[1].axhline(prev, color="#666666", linestyle="--", linewidth=1.1)
        # Anchored in axes coordinates. The data-coordinate anchor this used to
        # carry sat at recall 0.02, outside the panel's own x range, so the
        # label was clipped away and the chance line went unlabeled.
        axes[1].annotate(f"chance AP = {prev:.4f}", xy=(0.03, prev),
                         xycoords=("axes fraction", "data"),
                         textcoords="offset points", xytext=(0, 4),
                         fontsize=plt.rcParams["xtick.labelsize"] - 2,
                         color="#444444")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0.55, 1.005)
    axes[1].set_ylim(min(0.955, (prev or 0.97) - 0.015), 1.002)
    axes[1].set_title("Precision-recall")

    # One column, not two. With two columns the legend is wider than the canvas,
    # so bbox_inches="tight" grows the saved figure and LaTeX scales it further
    # down, which shrinks the printed legend text however large it is set here.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=1, frameon=False,
               fontsize=plt.rcParams["legend.fontsize"] - 1.5)
    plt.tight_layout()
    plt.savefig(OUT / "fig_rocpr.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  wrote {OUT / 'fig_rocpr.pdf'} ({len(keys)} models)")
    return None


# --------------------------------------------------------------------------- #
# fig_mininet: stills from the live demonstration
# --------------------------------------------------------------------------- #
def fig_mininet() -> str | None:
    """The emulated-network result (E4b).

    The earlier version of this figure stacked raw terminal screenshots. Those
    frames are dominated by repeated sch_htb quantum warnings from Mininet's
    traffic shaper, which read as failures on the page, so the figure is built
    from the measured values instead. The stills remain in the supplementary
    video, where the surrounding context makes the warnings legible as noise.
    """
    d = load("E4b_mininet_testbed.json")
    if not d or not d.get("results"):
        return "no E4b result yet"
    # Short tick labels. At the type size The deployment question asked for, "No detector"
    # and "Detect + explain" are wider than the bar spacing and run into each
    # other; the caption and Table 15 carry the full condition names.
    order = [("no_controller", "None"),
             ("detect", "Detect"),
             ("detect_explain", "Detect\n+expl.")]
    idle, attack, pkt, labels = [], [], [], []
    for key, lab in order:
        r = (d["results"].get(key) or {})
        mn, ctl = r.get("mininet") or {}, r.get("controller") or {}
        if not mn.get("baseline"):
            return f"condition {key} has no measurement"
        idle.append(mn["baseline"]["iperf"]["throughput_mbps"])
        attack.append(mn["under_attack"]["iperf"]["throughput_mbps"])
        pkt.append(ctl.get("packet_in", 0))
        labels.append(lab)

    S.okabe_style()
    # Taller than wide enough: the legend sits above ax1 and the x labels run to
    # two lines, so at the larger type size the panel needs the vertical room.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.5))
    x = np.arange(len(labels))
    w = 0.38

    ax1.bar(x - w / 2, idle, w, label="Network idle",
            color=S.O_SKY, edgecolor="white", linewidth=0.8)
    ax1.bar(x + w / 2, attack, w, label="Under SYN flood",
            color=S.O_RED, edgecolor="white", linewidth=0.8)
    for xi, v in zip(x + w / 2, attack):
        ax1.annotate(f"{v:.0f}", (xi, v), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8 * S.SCALE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Benign throughput (Mbit/s)")
    ax1.set_ylim(0, max(idle + attack) * 1.34)
    # Above the whole figure and in one row, so it can neither sit on top of the
    # annotated bars nor overrun ax1's own width at the larger type size.
    fig.legend(*ax1.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False,
               borderpad=0.3, handlelength=1.4, columnspacing=1.8)

    ax2.bar(x, pkt, 0.55, color=S.O_BLUE, edgecolor="white", linewidth=0.8)
    for xi, v in zip(x, pkt):
        ax2.annotate(f"{v:,}", (xi, v), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8 * S.SCALE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Packet-ins at controller")
    ax2.set_ylim(0, max(pkt) * 1.28)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    S.save(OUT, "fig_mininet")
    return None


BUILDERS = {"fig4": fig4, "fig7": fig7, "fig8": fig8, "fig10": fig10,
            "fig_rocpr": fig_rocpr, "fig_mininet": fig_mininet}


def main(argv: list[str]) -> int:
    want = [a for a in argv if a in BUILDERS] or list(BUILDERS)
    OUT.mkdir(parents=True, exist_ok=True)
    skipped = []
    for name in want:
        print(f"{name}:", flush=True)
        try:
            why = BUILDERS[name]()
        except Exception as e:                      # a bad figure must not stop the rest
            why = f"{type(e).__name__}: {e}"
        if why:
            print(f"  skipped - {why}")
            skipped.append((name, why))
    print()
    if skipped:
        print("not built:")
        for n, w in skipped:
            print(f"  {n:12s} {w}")
    else:
        print("all requested figures built")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
