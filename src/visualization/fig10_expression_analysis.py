"""
fig10_expression_analysis.py -- Article Figure 10: Expression Decoder Analysis.

  I1: CV scatter (real vs gen per-gene) with identity line and outlier genes
  I2: Expression range ribbon with percentile bands (10-90th, 25-75th)
  I3: Per-cell expression std as overlaid KDE-style histograms
  I4: Top variable genes heatmap (genes with highest CV difference)

Standalone function extracted from panels_expression for composability and
VCD-integrated saving via an optional ``save_panel_fn`` callback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, add_colorbar_safe, add_panel_label, quality_color, save_with_vcd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# FIGURE 10: Expression Decoder Analysis
# ──────────────────────────────────────────────────────────────

def plot_expression_analysis(
    real_expr_path: str = "results/real_expression.npy",
    gen_expr_path: str = "results/generated_expression.npy",
    gene_names_path: str = "results/expression_gene_names.json",
    metrics_path: str = "results/expression_metrics.json",
    output_dir: Path = Path("results/figures"),
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Enhanced expression decoder analysis with denser information.

    I1: CV scatter (real vs gen per-gene) with identity line and outlier genes
    I2: Expression range ribbon with percentile bands (10-90th, 25-75th)
    I3: Per-cell expression std as overlaid KDE-style histograms
    I4: Top variable genes heatmap (genes with highest CV difference)

    Parameters
    ----------
    real_expr_path : path to real expression matrix (.npy, cells x genes)
    gen_expr_path : path to generated expression matrix (.npy, cells x genes)
    gene_names_path : path to gene names JSON list
    metrics_path : path to expression metrics JSON
    output_dir : directory for saved figures
    dpi : figure resolution
    save : whether to save figure to disk
    save_panel_fn : optional callable(fig, basename, output_dir, dpi) for
                    VCD-integrated saving; falls back to ``save_with_vcd``
    """
    paths = [real_expr_path, gen_expr_path, gene_names_path, metrics_path]
    if not all(Path(p).exists() for p in paths):
        logger.info("Expression data not found -- skipping Panel I")
        return None

    real = np.load(real_expr_path)
    gen = np.load(gen_expr_path)
    with open(gene_names_path) as f:
        gene_names = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)

    overall = metrics.get("overall", {})

    fig = plt.figure(figsize=(9.8, 7.4))
    gs = fig.add_gridspec(2, 2, wspace=0.58, hspace=0.56)
    fig._clop_layout_rect = (0.03, 0.04, 0.98, 0.96)
    # Note: Figure-level title removed per revision requirements; stats moved to caption

    # -- I1: CV scatter (real vs gen) with gene labels --
    ax1 = fig.add_subplot(gs[0, 0])
    real_cv = real.std(axis=0) / (np.abs(real.mean(axis=0)) + 1e-8)
    gen_cv = gen.std(axis=0) / (np.abs(gen.mean(axis=0)) + 1e-8)
    cv_diff = np.abs(gen_cv - real_cv)

    sc = ax1.scatter(real_cv, gen_cv, c=cv_diff, cmap="YlOrRd", s=10,
                     alpha=0.7, edgecolors="none",
                     vmin=0, vmax=np.percentile(cv_diff, 95))
    lo = 0
    hi = max(real_cv.max(), gen_cv.max()) * 1.05
    ax1.plot([lo, hi], [lo, hi], color=COLORS["bad"], linestyle="--", lw=1.5,
             alpha=0.6, label="y = x")
    ax1.set_xlabel("Real CV (std/|mean|)", fontsize=11)
    ax1.set_ylabel("Generated CV", fontsize=11)
    ax1.set_title("Per-Gene Variability (CV)", fontsize=12)
    add_colorbar_safe(sc, ax=ax1, label="|\u0394CV|", shrink=0.50, pad=0.03, aspect=16)

    # Annotate top 3 divergent genes with staggered offsets
    top_cv_idx = np.argsort(cv_diff)[-3:]
    _offsets_cv = [(-85, -40), (30, 30), (-90, 30)]
    for j, i in enumerate(top_cv_idx):
        if i < len(gene_names):
            ax1.annotate(gene_names[i], (real_cv[i], gen_cv[i]),
                         fontsize=10, xytext=_offsets_cv[j % len(_offsets_cv)],
                         textcoords="offset points",
                         arrowprops=dict(arrowstyle="->", lw=0.5, color="#555"),
                         color="#333",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    cv_corr = np.corrcoef(real_cv, gen_cv)[0, 1]
    ax1.legend(fontsize=10, frameon=False)
    ax1.text(0.98, 0.02, f"r = {cv_corr:.3f}", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9))
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    add_panel_label(ax1, 'a', x=-0.10, y=1.05)

    # -- I2: Expression range with percentile bands --
    ax2 = fig.add_subplot(gs[0, 1])
    real_means = real.mean(axis=0)
    gen_means = gen.mean(axis=0)
    sort_idx = np.argsort(real_means)

    # Percentile bands
    rp10, rp90 = np.percentile(real, [10, 90], axis=0)
    rp25, rp75 = np.percentile(real, [25, 75], axis=0)
    gp10, gp90 = np.percentile(gen, [10, 90], axis=0)
    gp25, gp75 = np.percentile(gen, [25, 75], axis=0)

    x_range = np.arange(len(sort_idx))
    ax2.fill_between(x_range, rp10[sort_idx], rp90[sort_idx],
                     alpha=0.08, color=COLORS["real"], label="Real 10\u201390%")
    ax2.fill_between(x_range, rp25[sort_idx], rp75[sort_idx],
                     alpha=0.15, color=COLORS["real"])
    ax2.fill_between(x_range, gp10[sort_idx], gp90[sort_idx],
                     alpha=0.08, color=COLORS["generated"], label="Gen 10\u201390%")
    ax2.fill_between(x_range, gp25[sort_idx], gp75[sort_idx],
                     alpha=0.15, color=COLORS["generated"])
    ax2.plot(real_means[sort_idx], color=COLORS["real"], lw=1.2, label="Real mean", zorder=3)
    ax2.plot(gen_means[sort_idx], color=COLORS["generated"], lw=1.2, ls="--",
             label="Gen mean", zorder=3)
    ax2.set_xlabel("Gene index (sorted by real mean)", fontsize=11)
    ax2.set_ylabel("Expression", fontsize=11)
    ax2.set_title("Expression Range", fontsize=12)
    ax2.legend(fontsize=10, loc="upper left", ncol=2, frameon=False)
    add_panel_label(ax2, 'b', x=-0.10, y=1.05)

    # -- I3: Per-cell std as overlaid smooth histograms --
    ax3 = fig.add_subplot(gs[1, 0])
    real_cell_std = real.std(axis=1)
    gen_cell_std = gen.std(axis=1)
    real_cell_mean = real.mean(axis=1)
    gen_cell_mean = gen.mean(axis=1)

    bins3 = np.linspace(
        min(real_cell_std.min(), gen_cell_std.min()) - 0.005,
        max(real_cell_std.max(), gen_cell_std.max()) + 0.005,
        80,
    )
    ax3.hist(real_cell_std, bins=bins3, alpha=0.5, color=COLORS["real"],
             label=f"Real (\u03bc={real_cell_std.mean():.3f})",
             edgecolor="white", linewidth=0.3, density=True)
    ax3.hist(gen_cell_std, bins=bins3, alpha=0.5, color=COLORS["generated"],
             label=f"Gen (\u03bc={gen_cell_std.mean():.3f})",
             edgecolor="white", linewidth=0.3, density=True)
    ax3.axvline(x=real_cell_std.mean(), color=COLORS["real"], linestyle="--", lw=1.5)
    ax3.axvline(x=gen_cell_std.mean(), color=COLORS["generated"], linestyle="--", lw=1.5)
    ax3.set_xlabel("Per-Cell Std Dev", fontsize=11)
    ax3.set_ylabel("Density", fontsize=11)
    ax3.set_title("Per-Cell Variability Distribution", fontsize=12)
    ax3.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0.0, 1.0),
               frameon=False)
    ax3.locator_params(axis='x', nbins=4)
    add_panel_label(ax3, 'c', x=-0.10, y=1.05)

    std_ratio = gen_cell_std.mean() / (real_cell_std.mean() + 1e-8)
    ax3.text(0.02, 0.95,
             f"Std ratio: {std_ratio:.3f}",
             transform=ax3.transAxes, ha="left", va="top", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9))

    # -- I4: Top variable genes ranked bar chart --
    ax4 = fig.add_subplot(gs[1, 1])
    real_stds = real.std(axis=0)
    gen_stds = gen.std(axis=0)
    valid_mask = real_stds > 1e-4
    std_ratio_per_gene = np.ones(len(real_stds))
    std_ratio_per_gene[valid_mask] = gen_stds[valid_mask] / real_stds[valid_mask]

    n_show = min(12, len(gene_names))
    clipped_ratio = np.clip(std_ratio_per_gene, 0, 5.0)
    diff_score = np.abs(clipped_ratio - 1.0)
    top_diff_idx = np.argsort(diff_score)[-n_show:]
    top_diff_idx = top_diff_idx[np.argsort(clipped_ratio[top_diff_idx])]

    ratios_show = clipped_ratio[top_diff_idx]
    names_show = [gene_names[i] if i < len(gene_names) else f"G{i}"
                  for i in top_diff_idx]
    # Std ratio thresholds (0.9, 1.1) good band; (0.7, 0.9)/(1.1, 1.3) warn
    colors_i4 = [quality_color(min(r, 1 / (r + 1e-8)), (0.9, 0.7)) for r in ratios_show]
    ax4.barh(range(n_show), ratios_show, color=colors_i4, height=0.7,
             edgecolor="white", linewidth=0.3)
    ax4.axvline(x=1.0, color="#333", linestyle="-", linewidth=1.5)
    ax4.axvspan(0.9, 1.1, alpha=0.08, color=COLORS["good"])
    ax4.set_yticks(range(n_show))
    ax4.set_yticklabels(names_show, fontsize=10, ha="right")
    ax4.set_xlabel("Std Ratio (Gen / Real, clipped at 5\u00d7)", fontsize=11)
    ax4.set_title("Most Divergent Genes\n(over- & under-dispersed)", fontsize=11, pad=10)
    add_panel_label(ax4, 'd', x=-0.12, y=1.12)
    placed_annotations: list = []
    for i, r in enumerate(ratios_show):
        # Skip annotations within 0.05 of an already-placed one to avoid overlap
        too_close = any(abs(r - pr) < 0.05 and abs(i - pi) <= 1
                        for pr, pi in placed_annotations)
        if not too_close:
            ax4.text(r + 0.02, i, f"{r:.2f}\u00d7", va="center", fontsize=10,
                     fontweight="normal")
            placed_annotations.append((r, i))

    if save:
        if save_panel_fn:
            save_panel_fn(fig, "fig10_expression_analysis", Path(output_dir), dpi)
        else:
            save_with_vcd(fig, Path(output_dir) / "fig10_expression_analysis.png", dpi)
    return fig
