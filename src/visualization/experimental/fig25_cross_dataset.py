"""
fig25_cross_dataset.py — Fig 25: Cross-dataset gene-level validation.

Shows that CLOP-DiT generated expression profiles are compatible with
5 independent tissue datasets (lung, gastric, skin, liver, blood) at the
gene-expression level.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..direct_layout import bind_figure_region
from ..style import (
    COLORS,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TICK,
    FONT_TITLE,
    FONT_ANNOTATION,
    add_panel_label,
    apply_style,
    save_panel,
    style_axes,
)
from src.utils.paths import RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)


def plot_cross_dataset_validation(
    data_path: str | Path = "results/downstream/cross_dataset_validation.json",
    output_dir: str | Path = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Fig 25: Cross-dataset gene-level validation.

    Panel (a): Grouped bars — Pearson r + Spearman ρ (mean expression) per tissue.
    Panel (b): Grouped bars — PCA subspace overlap + frac similar distribution.
    Panel (c): Horizontal bars — median variance ratio per tissue.
    """
    apply_style()
    data_path = Path(data_path)
    output_dir = Path(output_dir)

    if not data_path.exists():
        logger.warning("Cross-dataset results not found: %s", data_path)
        return None

    with open(data_path) as f:
        all_data = json.load(f)

    # Filter tissues with successful results
    tissues = []
    for tissue, info in all_data.items():
        if isinstance(info, dict) and info.get("status") == "ok":
            tissues.append(tissue)

    if not tissues:
        logger.warning("No tissues with valid results — skipping Fig 25")
        return None

    # Extract metrics
    tissue_labels = [t.capitalize() for t in tissues]
    pearson_vals = [all_data[t]["mean_expr_pearson_r"] for t in tissues]
    spearman_vals = [all_data[t]["mean_expr_spearman_r"] for t in tissues]
    subspace_vals = [all_data[t]["pca_subspace_overlap"] for t in tissues]
    frac_sim_vals = [all_data[t]["frac_genes_similar_distribution"] for t in tissues]
    var_ratios = [all_data[t]["median_variance_ratio"] for t in tissues]
    n_genes_vals = [all_data[t]["n_shared_genes"] for t in tissues]

    # Layout: 3 panels
    fig = plt.figure(figsize=(14.5, 5.0))
    layout = bind_figure_region(fig, (0.07, 0.18, 0.97, 0.90))
    p_a, p_b, p_c = layout.split_cols([1.1, 1.1, 0.8], gap=0.06)

    x = np.arange(len(tissues))
    bar_w = 0.35

    # ── Panel (a): Pearson + Spearman on mean expression ──
    ax_a = p_a.add_axes(fig)
    add_panel_label(ax_a, "a", x=-0.12, y=1.06)

    bars1 = ax_a.bar(x - bar_w / 2, pearson_vals, bar_w, label="Pearson r",
                     color=COLORS["real"], alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax_a.bar(x + bar_w / 2, spearman_vals, bar_w, label="Spearman ρ",
                     color=COLORS["generated"], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(tissue_labels, fontsize=FONT_TICK, rotation=25, ha="right")
    ax_a.set_ylabel("Correlation", fontsize=FONT_LABEL)
    ax_a.set_title("Mean Gene Expression", fontsize=FONT_TITLE)
    ax_a.set_ylim(0, 1.15)
    ax_a.legend(fontsize=FONT_LEGEND, loc="upper right", frameon=False)
    style_axes(ax_a)

    # ── Panel (b): PCA overlap (single bar series) ──
    ax_b = p_b.add_axes(fig)
    add_panel_label(ax_b, "b", x=-0.12, y=1.06)

    bars3 = ax_b.bar(x, subspace_vals, bar_w, label="PCA Overlap",
                     color=COLORS["good"], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(tissue_labels, fontsize=FONT_TICK, rotation=25, ha="right")
    ax_b.set_ylabel("Score", fontsize=FONT_LABEL)
    ax_b.set_title("Distribution Similarity", fontsize=FONT_TITLE)
    ax_b.set_ylim(0, 1.15)
    ax_b.legend(fontsize=FONT_LEGEND, loc="upper right", frameon=False)
    ax_b.text(0.5, -0.22, "KS test: 0 genes similar across all tissues",
              transform=ax_b.transAxes, ha="center", fontsize=FONT_ANNOTATION,
              style="italic", color=COLORS["neutral"])
    style_axes(ax_b)

    # ── Panel (c): Variance ratio + n_shared_genes ──
    ax_c = p_c.add_axes(fig)
    add_panel_label(ax_c, "c", x=-0.15, y=1.06)

    # Use log10 scale to handle extreme variance ratios (can be ~0.0002 or >100)
    import math
    safe_var_ratios = [max(v, 1e-6) for v in var_ratios]
    log_ratios = [math.log10(v) for v in safe_var_ratios]
    bar_colors = [COLORS["good"] if 0.5 <= v <= 2.0 else COLORS["bad"] for v in safe_var_ratios]
    ax_c.barh(x, log_ratios, height=0.6,
              color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax_c.set_yticks(x)
    ax_c.set_yticklabels(tissue_labels, fontsize=FONT_TICK)
    ax_c.set_xlabel("Median Var Ratio log10(gen/ref)", fontsize=FONT_LABEL)
    ax_c.set_title("Variance Match (log scale)", fontsize=FONT_TITLE)
    ax_c.axvline(0.0, color=COLORS["neutral"], ls="--", lw=0.8, alpha=0.7,
                 label="ideal (ratio=1)")
    # Shade the acceptable range [0.5, 2.0] → log10 [-0.3, 0.3]
    ax_c.axvspan(math.log10(0.5), math.log10(2.0), color=COLORS["good"], alpha=0.08)
    ax_c.invert_yaxis()
    style_axes(ax_c)
    ax_c.text(0.5, -0.22, "Ratio << 1: scGPT decoder compresses variance",
              transform=ax_c.transAxes, ha="center", fontsize=FONT_ANNOTATION,
              style="italic", color=COLORS["neutral"])

    # Note: per-bar value annotations omitted to avoid overlap with y-tick labels.
    # The log-scale x-axis and bar lengths directly encode the variance ratio magnitudes.

    # Save
    if save:
        if save_panel_fn:
            save_panel_fn(fig, "fig25_cross_dataset")
        else:
            out = output_dir / "fig25_cross_dataset.png"
            save_panel(fig, out, dpi=dpi)
        logger.info("Saved Fig 25 → %s", output_dir)

    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    logging.basicConfig(level=logging.INFO)
    plot_cross_dataset_validation()
