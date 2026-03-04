"""
panels_diversity.py — Panels J (diversity diagnostics) and K (expression diversity).

Plotting only; scripts run diagnostics and pass all_results from diversity_diagnostics.json.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .style import apply_style, save_with_vcd

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def plot_diagnostics(
    all_results: Dict,
    output_dir: str = "results/figures",
    dpi: int = 300,
) -> List[Path]:
    """Generate Panel J (diversity diagnostics) and Panel K (expression diversity) from test results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    apply_style()

    # ── Panel J: Diversity Diagnostics (4 subplots) ──
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.5),
                             gridspec_kw={"hspace": 0.60, "wspace": 0.55})

    ax = axes[0, 0]
    t1 = all_results.get("test1_intratype_diversity", {}).get("per_type", {})
    if t1:
        names = list(t1.keys())
        div_ratios = [t1[n]["diversity_ratio"] for n in names]
        sorted_idx = np.argsort(div_ratios)
        sorted_names = [names[i][:25] for i in sorted_idx]
        sorted_divs = [div_ratios[i] for i in sorted_idx]

        colors = ["#F44336" if d < 0.5 else "#FF9800" if d < 0.8 else "#4CAF50" if d < 1.2 else "#2196F3"
                  for d in sorted_divs]
        ax.barh(range(len(sorted_divs)), sorted_divs, color=colors, height=0.8)
        ax.set_yticks(range(len(sorted_divs)))
        _step_j1 = max(1, len(sorted_divs) // 12)
        _ytl_j1 = [n if i % _step_j1 == 0 else "" for i, n in enumerate(sorted_names)]
        ax.set_yticklabels(_ytl_j1, fontsize=7)
        ax.axvline(x=1.0, color="black", ls="--", lw=1, alpha=0.5, label="ratio=1 (equal)")
        ax.axvline(x=0.5, color="red", ls=":", lw=1, alpha=0.5, label="ratio=0.5 (collapse)")
        ax.set_xlabel("Diversity Ratio (gen / real)")
        ax.legend(fontsize=8, loc="lower right")
        summary = all_results["test1_intratype_diversity"]["summary"]
        ax.set_title(
            f"Intra-Type Diversity Ratio\n"
            f"mean={summary['mean_diversity_ratio']:.4f}, "
            f"collapsed={summary['n_collapsed']}/{summary['n_collapsed'] + summary['n_healthy']}"
        )
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Intra-Type Diversity Ratio")

    ax = axes[0, 1]
    t2 = all_results.get("test2_memorization", {})
    if t2:
        nn = t2["nn_cosine_distance"]
        labels = ["p5", "p25", "median", "mean", "p75", "p95"]
        vals = [nn["p5"], nn["p25"], nn["median"], nn["mean"], nn["p75"], nn["p95"]]
        ax.bar(labels, vals, color=["#F44336", "#FF9800", "#4CAF50", "#2196F3", "#FF9800", "#F44336"],
               alpha=0.8, edgecolor="white")
        ax.set_ylabel("Cosine Distance to Nearest Real Cell")
        ax.axhline(y=0.01, color="red", ls=":", alpha=0.5, label="memorization threshold")
        ax.legend(fontsize=8)
        ax.set_title(
            f"Nearest-Neighbour Distance\n"
            f"n_near_copy(d<0.001)={t2['n_very_close']}, "
            f"same_type_nn={t2['nn_same_type_frac']:.1%}"
        )
    else:
        ax.set_title("Nearest-Neighbour Distance")

    ax = axes[1, 0]
    t3 = all_results.get("test3_cfg_sweep", {})
    if t3:
        cfg_vals = sorted(t3.keys(), key=lambda k: t3[k]["cfg_scale"])
        scales = [t3[k]["cfg_scale"] for k in cfg_vals]
        divs = [t3[k]["mean_intra_diversity"] for k in cfg_vals]
        norms = [t3[k]["mean_norm"] for k in cfg_vals]

        color = "#1976D2"
        ax.plot(scales, divs, "o-", color=color, lw=2, markersize=8, label="Diversity")
        ax.set_xlabel("CFG Scale")
        ax.set_ylabel("Mean Intra-Type Diversity", color=color)
        ax.tick_params(axis="y", labelcolor=color)

        ax2 = ax.twinx()
        color2 = "#FF7043"
        ax2.plot(scales, norms, "s--", color=color2, lw=2, markersize=8, label="Norm")
        ax2.set_ylabel("Mean Embedding Norm", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        ax.set_title("CFG Scale vs Diversity & Norm")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    else:
        ax.set_title("CFG Scale vs Diversity")

    ax = axes[1, 1]
    t5 = all_results.get("test5_condition_sensitivity", {})
    if t5:
        per_type = t5.get("per_type", {})
        type_ids = sorted(per_type.keys(), key=lambda k: per_type[k]["diversity_gain"])
        cent_divs = [per_type[k]["centroid_diversity"] for k in type_ids]
        noise_divs = [per_type[k]["noise_diversity"] for k in type_ids]

        x = np.arange(len(type_ids))
        w = 0.35
        ax.bar(x - w / 2, cent_divs, w, label="Centroid Cond", color="#1976D2", alpha=0.8)
        ax.bar(x + w / 2, noise_divs, w, label="Centroid + Noise", color="#FF7043", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in type_ids], fontsize=8)
        ax.set_xlabel("Type ID")
        ax.set_ylabel("Intra-Type Diversity (1 - mean cosine)")
        gain = t5["summary"]["mean_diversity_gain"]
        eps = t5["summary"].get("noise_scale", "?")
        ax.set_title(f"Centroid vs Noisy Conditioning\ngain={gain:.2f}x, \u03b5={eps}")
        ax.legend(fontsize=8)
    else:
        ax.set_title("Centroid vs Noisy Conditioning")

    path = out / "panel_j_diversity_diagnostics.png"
    save_with_vcd(fig, path, dpi)
    logger.info(f"Saved Panel J → {path}")
    saved.append(path)
    plt.close(fig)

    # ── Panel K: Expression Diversity ──
    t6 = all_results.get("test6_expression_diversity", {})
    if t6 and t6.get("overall"):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.0))

        o = t6["overall"]
        fig.suptitle(
            f"Expression-Level Diversity — "
            f"gene_std_ratio={o['gene_std_ratio']:.4f}",
            fontsize=11,
        )

        ax = axes[0]
        labels = ["Cell Std\n(across genes)", "Gene Std\n(across cells)"]
        real_vals = [o["real_mean_cell_std"], o["real_mean_gene_std"]]
        gen_vals = [o["gen_mean_cell_std"], o["gen_mean_gene_std"]]
        x = np.arange(2)
        w = 0.35
        ax.bar(x - w / 2, real_vals, w, label="Real", color="#1976D2", alpha=0.8)
        ax.bar(x + w / 2, gen_vals, w, label="Generated", color="#FF7043", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Standard Deviation")
        ax.set_title("Expression Variability Summary")
        ax.legend()

        ax = axes[1]
        pt_ratio = t6.get("per_type_gene_std_ratio", {})
        if pt_ratio:
            vals = [pt_ratio["min"], pt_ratio["mean"], pt_ratio["max"]]
            lbls = ["Min", "Mean", "Max"]
            colors = ["#F44336" if v < 0.5 else "#4CAF50" for v in vals]
            ax.bar(lbls, vals, color=colors, alpha=0.8, edgecolor="white")
            ax.axhline(y=1.0, color="black", ls="--", lw=1, alpha=0.5,
                       label="ratio=1 (equal diversity)")
            ax.set_ylabel("Gene Std Ratio (gen / real)")
            ax.set_title("Per-Type Gene Std Ratio")
            ax.legend(fontsize=8)

        path = out / "panel_k_expression_diversity.png"
        save_with_vcd(fig, path, dpi)
        logger.info(f"Saved Panel K → {path}")
        saved.append(path)
        plt.close(fig)

    return saved
