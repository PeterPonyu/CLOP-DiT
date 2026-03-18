"""Fig 24: Per-gene variance deep-dive — diagnosing variance collapse.

Loads real vs generated expression matrices and computes per-gene variance
statistics to visualize:
  (a) Scatter plot of real vs generated per-gene variance
  (b) Histogram of variance ratios (gen/real)
  (c) Top-N most under-dispersed genes ranked by variance gap

This figure directly addresses the per-gene variance weakness (r ≈ 0)
discussed in the manuscript.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from .direct_layout import bind_figure_region
from .style import (
    COLORS,
    FONT_LABEL,
    FONT_LEGEND_DENSE,
    FONT_TITLE,
    add_panel_label,
    apply_style,
    save_with_vcd,
    set_figure_suptitle,
)

logger = logging.getLogger(__name__)


def plot_variance_deepdive(
    real_expr_path: str = "results/real_expression.npy",
    gen_expr_path: str = "results/generated_expression.npy",
    gene_names_path: str = "results/expression_gene_names.json",
    output_dir: str | Path = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Render per-gene variance analysis.

    Parameters
    ----------
    real_expr_path : path to real expression matrix (cells × genes)
    gen_expr_path : path to generated expression matrix
    gene_names_path : path to JSON list of gene names
    output_dir : figure output directory
    dpi : export resolution
    save : whether to save figure
    save_panel_fn : optional callback for custom saving

    Returns
    -------
    fig : Figure or None on error
    """
    apply_style()
    output_dir = Path(output_dir)

    # Load data
    try:
        real_expr = np.load(real_expr_path)
        gen_expr = np.load(gen_expr_path)
    except FileNotFoundError as e:
        logger.warning("Expression data not found: %s", e)
        return None

    gene_names = None
    try:
        with open(gene_names_path) as f:
            gene_names = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.info("Gene names not available; using indices")

    # Compute per-gene variance
    real_var = np.var(real_expr, axis=0)
    gen_var = np.var(gen_expr, axis=0)
    n_genes = len(real_var)

    # Variance ratio (avoid division by zero)
    eps = 1e-10
    var_ratio = gen_var / (real_var + eps)

    # Correlation
    valid = (real_var > eps) & (gen_var > eps)
    if valid.sum() > 2:
        r = np.corrcoef(real_var[valid], gen_var[valid])[0, 1]
    else:
        r = 0.0

    # Figure: 3 panels
    fig = plt.figure(figsize=(16.0, 5.0))
    layout = bind_figure_region(fig, (0.07, 0.14, 0.97, 0.88))
    cols = layout.split_cols([1, 1, 1.2], wspace=0.38)

    # ── Panel (a): Scatter ──
    ax_a = cols[0].add_axes(fig)
    ax_a.scatter(real_var[valid], gen_var[valid], s=4, alpha=0.4,
                 color=COLORS["real"], edgecolors="none")
    # Identity line
    lim = max(real_var[valid].max(), gen_var[valid].max()) * 1.05
    ax_a.plot([0, lim], [0, lim], "--", color=COLORS["neutral"], linewidth=1, alpha=0.7)
    ax_a.set_xlabel("Real per-gene variance", fontsize=FONT_LABEL)
    ax_a.set_ylabel("Generated per-gene variance", fontsize=FONT_LABEL)
    ax_a.set_title("Variance Scatter", fontsize=FONT_TITLE)
    ax_a.text(0.05, 0.92, f"r = {r:.3f}\nn = {valid.sum()} genes",
              transform=ax_a.transAxes, fontsize=FONT_LEGEND_DENSE,
              va="top", color=COLORS["annotation_dark"])
    ax_a.locator_params(axis='x', nbins=4)
    ax_a.locator_params(axis='y', nbins=4)
    add_panel_label(ax_a, "a", x=-0.14, y=1.08)

    # ── Panel (b): Histogram of variance ratios ──
    ax_b = cols[1].add_axes(fig)
    log_ratio = np.log2(var_ratio[valid] + eps)
    ax_b.hist(log_ratio, bins=50, color=COLORS["generated"], alpha=0.75, edgecolor="white",
              linewidth=0.3)
    ax_b.axvline(0, color=COLORS["neutral"], linewidth=1, linestyle="--", alpha=0.7)
    median_lr = np.median(log_ratio)
    ax_b.axvline(median_lr, color=COLORS["bad"], linewidth=1.2, linestyle="-")
    ax_b.set_xlabel("log₂(σ²_gen / σ²_real)", fontsize=FONT_LABEL)
    ax_b.set_ylabel("Number of genes", fontsize=FONT_LABEL)
    ax_b.set_title("Variance Ratio Distribution", fontsize=FONT_TITLE)
    ax_b.text(0.05, 0.92, f"Median: {median_lr:.2f}",
              transform=ax_b.transAxes, fontsize=FONT_LEGEND_DENSE,
              va="top", color=COLORS["bad"])
    pct_under = (log_ratio < -1).sum() / len(log_ratio) * 100
    ax_b.text(0.05, 0.82, f"{pct_under:.0f}% genes >2× under-dispersed",
              transform=ax_b.transAxes, fontsize=FONT_LEGEND_DENSE,
              va="top", color=COLORS["annotation_dark"])
    add_panel_label(ax_b, "b", x=-0.12, y=1.06)

    # ── Panel (c): Top under-dispersed genes ──
    ax_c = cols[2].add_axes(fig)
    top_n = 20
    gaps = real_var - gen_var
    worst_idx = np.argsort(gaps)[-top_n:][::-1]  # largest gap first

    if gene_names and len(gene_names) == n_genes:
        labels = [gene_names[i] for i in worst_idx]
    else:
        labels = [f"Gene {i}" for i in worst_idx]

    y_pos = np.arange(top_n)
    gap_vals = gaps[worst_idx]
    real_vals = real_var[worst_idx]
    gen_vals = gen_var[worst_idx]

    bar_height = 0.35
    ax_c.barh(y_pos - bar_height / 2, real_vals, bar_height, label="Real",
              color=COLORS["real"], alpha=0.85)
    ax_c.barh(y_pos + bar_height / 2, gen_vals, bar_height, label="Generated",
              color=COLORS["generated"], alpha=0.85)

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(labels, fontsize=FONT_LEGEND_DENSE - 2)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Per-gene variance", fontsize=FONT_LABEL)
    ax_c.set_title(f"Top-{top_n} Under-dispersed Genes", fontsize=FONT_TITLE)
    ax_c.legend(fontsize=FONT_LEGEND_DENSE, loc="lower right")
    ax_c.locator_params(axis='x', nbins=4)
    add_panel_label(ax_c, "c", x=-0.20, y=1.08)

    set_figure_suptitle(fig, "Per-Gene Variance Analysis: Real vs Generated", y=0.98)

    if save:
        out = output_dir / "fig24_variance_deepdive.png"
        if save_panel_fn:
            save_panel_fn(fig, "fig24_variance_deepdive")
        else:
            save_with_vcd(fig, out, dpi=dpi)

    return fig
