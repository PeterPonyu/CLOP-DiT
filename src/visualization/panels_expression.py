"""
panels_expression.py -- Panels H, I, N: gene expression fidelity and marker analysis.

  Panel H: Gene Expression Correlation (scatter + per-type + markers + residuals)
  Panel I: Expression Decoder Analysis (CV scatter + range ribbon + per-cell std + variable genes)
  Panel N: Marker Gene Comparison (per-type real vs generated expression, focused heatmap)

Standalone functions extracted from ResultsVisualizer for composability and
VCD-integrated saving via an optional ``save_panel_fn`` callback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, add_colorbar_safe, quality_color, save_with_vcd, set_figure_suptitle, style_axes

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Module-level constants for Panel N
# ──────────────────────────────────────────────────────────────

# Biologically meaningful markers covering major lineages
MARKER_PANEL_GENES: Dict[str, List[str]] = {
    "CD8+ T":       ["CD8A", "GZMB"],
    "Myeloid":      ["CD68", "CD163"],
    "Epithelial":   ["EPCAM", "KRT8"],
    "Stromal":      ["COL1A1", "COL1A2"],
}

# Representative types to show per-type breakdown
MARKER_PANEL_TYPES: List[str] = [
    "CD8+ cytotoxic T lymphocytes",
    "Tumor-associated macrophages",
    "Epithelial tumor cells",
    "Fibroblasts and mesenchymal stromal cell",
]


# ──────────────────────────────────────────────────────────────
# PANEL H: Gene Expression Correlation
# ──────────────────────────────────────────────────────────────

def plot_expression_correlation(
    real_expr_path: str = "results/real_expression.npy",
    gen_expr_path: str = "results/generated_expression.npy",
    real_labels_path: str = "results/real_expression_labels.npy",
    gen_labels_path: str = "results/generated_expression_labels.npy",
    gene_names_path: str = "results/expression_gene_names.json",
    metrics_path: str = "results/expression_metrics.json",
    output_dir: Path = Path("results/figures"),
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Enhanced gene expression fidelity with density-aware scatter and richer bars.

    H1: Density scatter of per-gene mean expression with residual coloring
    H2: Per-type Pearson r lollipop chart with tier shading
    H3: Marker gene grouped bars with error bars and fold-change annotation
    H4: Per-gene residual distribution (gen - real)

    Parameters
    ----------
    real_expr_path : path to real expression matrix (.npy, cells x genes)
    gen_expr_path : path to generated expression matrix (.npy, cells x genes)
    real_labels_path : path to real cell-type label array (.npy)
    gen_labels_path : path to generated cell-type label array (.npy)
    gene_names_path : path to gene names JSON list
    metrics_path : path to expression metrics JSON
    output_dir : directory for saved figures
    dpi : figure resolution
    save : whether to save figure to disk
    save_panel_fn : optional callable(fig, basename, output_dir, dpi) for
                    VCD-integrated saving; falls back to ``save_with_vcd``
    """
    paths = [real_expr_path, gen_expr_path, metrics_path, gene_names_path]
    if not all(Path(p).exists() for p in paths):
        logger.info("Expression data not found -- skipping Panel H")
        return None

    real = np.load(real_expr_path)
    gen = np.load(gen_expr_path)
    with open(gene_names_path) as f:
        gene_names = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)

    real_means = real.mean(axis=0)
    gen_means = gen.mean(axis=0)
    residuals = gen_means - real_means
    pearson_r = metrics["gene_correlation"]["pearson_r"]
    spearman_rho = metrics["gene_correlation"]["spearman_rho"]

    fig = plt.figure(figsize=(11.2, 7.8))
    gs = fig.add_gridspec(2, 2, wspace=0.62, hspace=0.56)
    set_figure_suptitle(
        fig,
        f"Gene Expression Recovery \u2014 r={pearson_r:.6f}, "
        f"\u03c1={spearman_rho:.6f}, n={len(gene_names)}",
        fontsize=11,
    )

    # -- H1: Density scatter with residual coloring --
    ax1 = fig.add_subplot(gs[0, 0])
    abs_res = np.abs(residuals)
    sc = ax1.scatter(real_means, gen_means, c=abs_res, cmap="magma_r",
                     s=12, alpha=0.7, edgecolors="none",
                     vmin=0, vmax=np.percentile(abs_res, 95))
    lo = min(real_means.min(), gen_means.min()) - 0.2
    hi = max(real_means.max(), gen_means.max()) + 0.2
    ax1.plot([lo, hi], [lo, hi], color=COLORS["bad"], linestyle="--", lw=1.5,
             alpha=0.7, label="y = x", zorder=1)
    ax1.fill_between([lo, hi], [lo - 0.1, hi - 0.1], [lo + 0.1, hi + 0.1],
                     alpha=0.06, color=COLORS["good"], zorder=0)
    ax1.set_xlabel("Real Mean Expression", fontsize=10)
    ax1.set_ylabel("Generated Mean Expression", fontsize=10)
    ax1.set_title("Per-Gene Correlation", fontsize=11)
    ax1.legend(fontsize=8, loc="upper left", frameon=False)
    from matplotlib.ticker import MaxNLocator as _MaxNLoc
    ax1.xaxis.set_major_locator(_MaxNLoc(nbins=3, prune="both"))
    ax1.yaxis.set_major_locator(_MaxNLoc(nbins=3, prune="both"))
    cbar = add_colorbar_safe(sc, ax=ax1, label="|Resid|", shrink=0.78, pad=0.03, aspect=24)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.set_major_locator(_MaxNLoc(nbins=2, prune="both"))

    # Annotate outlier genes (top 2 residuals)
    outlier_idx = np.argsort(abs_res)[-2:]
    for i in outlier_idx:
        if i < len(gene_names):
            ax1.annotate(gene_names[i], (real_means[i], gen_means[i]),
                         fontsize=8, xytext=(5, 5), textcoords="offset points",
                         arrowprops=dict(arrowstyle="->", lw=0.5, color="#555"),
                         color="#333")

    # -- H2: Per-type Pearson r lollipop chart --
    ax2 = fig.add_subplot(gs[0, 1])
    per_type = metrics.get("per_type_expression_fidelity", {})
    if per_type:
        type_names_sorted = sorted(per_type.keys(),
                                   key=lambda k: per_type[k]["pearson_r"])
        type_rs = [per_type[n]["pearson_r"] for n in type_names_sorted]

        # Show only bottom 6 and top 6 types to avoid 69-label overlap
        n_show_each = min(6, len(type_rs) // 2)
        if len(type_rs) > 2 * n_show_each + 2:
            show_idx = list(range(n_show_each)) + list(range(len(type_rs) - n_show_each, len(type_rs)))
            type_names_sorted = [type_names_sorted[i] for i in show_idx]
            type_rs = [type_rs[i] for i in show_idx]
        short_names = [n[:20] for n in type_names_sorted]

        mean_r = metrics.get("per_type_summary", {}).get("mean_pearson_r", 0)
        min_r = min(type_rs)
        y_pos = np.arange(len(type_rs))

        # Threshold shading
        ax2.axvspan(0.9999, 1.00005, alpha=0.08, color=COLORS["good"])
        ax2.axvspan(0.999, 0.9999, alpha=0.08, color=COLORS["warn"])
        ax2.axvspan(min_r - 0.001, 0.999, alpha=0.08, color=COLORS["bad"])

        # Pearson r thresholds (0.9999, 0.999) per FIGURE_PRESENTATION_POLICY
        colors_h2 = [quality_color(r, (0.9999, 0.999)) for r in type_rs]
        ax2.hlines(y_pos, min_r - 0.0005, type_rs, color="#DDD", linewidth=0.8, zorder=1)
        ax2.scatter(type_rs, y_pos, c=colors_h2, s=30, zorder=3, edgecolors="white",
                    linewidths=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(short_names, fontsize=8, ha="right")
        ax2.axvline(x=mean_r, color=COLORS["bad"], linestyle="--", alpha=0.6, linewidth=1.5,
                    label="mean (see caption)")
        ax2.set_xlim(min_r - 0.0005, 1.00005)
        ax2.set_xlabel("Pearson r", fontsize=10)
        ax2.legend(
            fontsize=8,
            loc="lower left",
            frameon=False,
            bbox_to_anchor=(1.02, 0.0),
            borderaxespad=0.0,
        )
    else:
        ax2.text(0.5, 0.5, "No per-type data", ha="center", va="center",
                 transform=ax2.transAxes)
    ax2.set_title("Per-Type Expression Fidelity", fontsize=11)
    ax2.xaxis.set_major_locator(_MaxNLoc(nbins=3, prune="both"))
    ax2.yaxis.set_major_locator(_MaxNLoc(nbins=4, prune="both"))

    # -- H3: Marker gene expression with error bars --
    ax3 = fig.add_subplot(gs[1, 0])
    marker_dict = metrics.get("marker_genes", {})
    selected_cats: List[str] = []
    selected_genes: List[List[str]] = []
    for cat, genes_list in marker_dict.items():
        found = [g for g in genes_list if g in gene_names]
        if found:
            selected_cats.append(cat)
            selected_genes.append(found[:2])
        if len(selected_cats) >= 6:
            break

    if selected_cats:
        all_marker_genes: List[str] = []
        cat_labels: List[str] = []
        for cat, gg in zip(selected_cats, selected_genes):
            for g in gg:
                all_marker_genes.append(g)
                cat_labels.append(cat.replace("_", " "))

        gidx = [gene_names.index(g) for g in all_marker_genes]
        r_means = np.array([real[:, i].mean() for i in gidx])
        g_means = np.array([gen[:, i].mean() for i in gidx])
        r_stds = np.array([real[:, i].std() for i in gidx])
        g_stds = np.array([gen[:, i].std() for i in gidx])

        x = np.arange(len(all_marker_genes))
        width = 0.35
        ax3.bar(x - width / 2, r_means, width, yerr=r_stds * 0.5,
                label="Real", color=COLORS["real"], alpha=0.85, edgecolor="white",
                capsize=3, error_kw=dict(lw=0.8))
        ax3.bar(x + width / 2, g_means, width, yerr=g_stds * 0.5,
                label="Generated", color=COLORS["generated"], alpha=0.85, edgecolor="white",
                capsize=3, error_kw=dict(lw=0.8))

        # Fold-change annotations -- only annotate outliers (fc > 1.05 or < 0.95)
        for i, (rm, gm) in enumerate(zip(r_means, g_means)):
            fc = gm / (rm + 1e-8)
            if abs(fc - 1.0) > 0.10:
                color = COLORS["good"] if 0.95 <= fc <= 1.05 else COLORS["bad"]
                ax3.text(i, max(rm, gm) + max(r_stds[i], g_stds[i]) * 0.5 + 0.01,
                         f"{fc:.2f}\u00d7", ha="center", fontsize=8, color=color)

        ax3.set_xticks(x)
        ax3.set_xticklabels(
            [g for g in all_marker_genes],
            fontsize=9, rotation=90, ha="center",
        )
        ax3.set_ylabel("Expression (mean \u00b1 0.5\u00d7std)", fontsize=10)
        ax3.legend(fontsize=8, loc="upper left", ncol=1, frameon=False, bbox_to_anchor=(1.02, 1.0))
    else:
        ax3.text(0.5, 0.5, "No marker genes found", ha="center", va="center",
                 transform=ax3.transAxes)
    ax3.set_title("Marker Gene Expression", fontsize=11)
    ax3.xaxis.set_major_locator(_MaxNLoc(nbins=4, prune="both"))
    ax3.yaxis.set_major_locator(_MaxNLoc(nbins=4, prune="both"))

    # -- H4: Residual distribution --
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(residuals, bins=60, color=COLORS["real"], alpha=0.7, edgecolor="white",
             density=True)
    ax4.tick_params(axis='x', labelsize=9, rotation=30)
    ax4.axvline(x=0, color=COLORS["bad"], linestyle="--", linewidth=1.5, label="Zero")
    ax4.axvline(x=residuals.mean(), color=COLORS["warn"], linestyle="-", linewidth=1.5,
                label=f"Mean={residuals.mean():.4f}")
    ax4.set_xlabel("Residual (Gen \u2212 Real)", fontsize=10)
    ax4.set_ylabel("Density", fontsize=10)
    ax4.set_title("Per-Gene Residual Distribution", fontsize=11)
    ax4.legend(fontsize=8, frameon=False)
    ax4.xaxis.set_major_locator(_MaxNLoc(nbins=4, prune="both"))
    ax4.yaxis.set_major_locator(_MaxNLoc(nbins=4, prune="both"))

    pct_within_01 = (np.abs(residuals) < 0.1).mean() * 100
    pct_within_001 = (np.abs(residuals) < 0.01).mean() * 100
    ax4.text(0.95, 0.95,
             f"|\u0394|<0.01: {pct_within_001:.0f}%\n|\u0394|<0.10: {pct_within_01:.0f}%",
             transform=ax4.transAxes, ha="right", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9))

    if save:
        if save_panel_fn:
            save_panel_fn(fig, "panel_h_expression_correlation", Path(output_dir), dpi)
        else:
            save_with_vcd(fig, Path(output_dir) / "panel_h_expression_correlation.png", dpi)
    return fig


# ──────────────────────────────────────────────────────────────
# PANEL I: Expression Decoder Analysis
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
    set_figure_suptitle(
        fig,
        f"Expression Decoder \u2014 "
        f"{real.shape[0]} real, {gen.shape[0]} gen, "
        f"{len(gene_names)} genes",
        fontsize=11,
    )

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
    ax1.set_xlabel("Real CV (std/|mean|)", fontsize=10)
    ax1.set_ylabel("Generated CV", fontsize=10)
    ax1.set_title("Per-Gene Variability (CV)", fontsize=11)
    add_colorbar_safe(sc, ax=ax1, label="|\u0394CV|", shrink=0.8, pad=0.02)

    # Annotate top 2 divergent genes (annotation budget: at most two per subplot)
    top_cv_idx = np.argsort(cv_diff)[-2:]
    for i in top_cv_idx:
        if i < len(gene_names):
            ax1.annotate(gene_names[i], (real_cv[i], gen_cv[i]),
                         fontsize=8, xytext=(4, 4), textcoords="offset points",
                         arrowprops=dict(arrowstyle="->", lw=0.5, color="#555"),
                         color="#333")

    cv_corr = np.corrcoef(real_cv, gen_cv)[0, 1]
    ax1.legend(fontsize=8, frameon=False)
    ax1.text(0.98, 0.02, f"CV corr = {cv_corr:.4f}", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", alpha=0.9))
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))

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
    ax2.set_xlabel("Gene index (sorted by real mean)", fontsize=10)
    ax2.set_ylabel("Expression", fontsize=10)
    ax2.set_title("Expression Range", fontsize=11)
    ax2.legend(fontsize=7, loc="upper left", ncol=2, frameon=False)

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
             label=f"Real (\u03bc={real_cell_std.mean():.4f})",
             edgecolor="white", linewidth=0.3, density=True)
    ax3.hist(gen_cell_std, bins=bins3, alpha=0.5, color=COLORS["generated"],
             label=f"Gen (\u03bc={gen_cell_std.mean():.4f})",
             edgecolor="white", linewidth=0.3, density=True)
    ax3.axvline(x=real_cell_std.mean(), color=COLORS["real"], linestyle="--", lw=1.5)
    ax3.axvline(x=gen_cell_std.mean(), color=COLORS["generated"], linestyle="--", lw=1.5)
    ax3.set_xlabel("Per-Cell Std Dev", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.set_title("Per-Cell Variability Distribution", fontsize=11)
    ax3.legend(fontsize=8, loc='upper right', ncol=1, frameon=False)
    ax3.locator_params(axis='x', nbins=4)

    std_ratio = gen_cell_std.mean() / (real_cell_std.mean() + 1e-8)
    ax3.text(0.02, 0.95,
             f"Std ratio: {std_ratio:.3f}",
             transform=ax3.transAxes, ha="left", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", alpha=0.9))

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
    ax4.set_yticklabels(names_show, fontsize=9, ha="right")
    ax4.set_xlabel("Std Ratio (Gen / Real, clipped at 5\u00d7)", fontsize=10)
    ax4.set_title("Top Variable Genes (std gen/real)", fontsize=11)
    placed_annotations: list = []
    for i, r in enumerate(ratios_show):
        # Skip annotations within 0.05 of an already-placed one to avoid overlap
        too_close = any(abs(r - pr) < 0.05 and abs(i - pi) <= 1
                        for pr, pi in placed_annotations)
        if not too_close:
            ax4.text(r + 0.02, i, f"{r:.2f}\u00d7", va="center", fontsize=8,
                     fontweight="bold" if abs(r - 1.0) > 0.3 else "normal")
            placed_annotations.append((r, i))

    if save:
        if save_panel_fn:
            save_panel_fn(fig, "panel_i_expression_analysis", Path(output_dir), dpi)
        else:
            save_with_vcd(fig, Path(output_dir) / "panel_i_expression_analysis.png", dpi)
    return fig


# ──────────────────────────────────────────────────────────────
# PANEL N: Marker Gene Comparison (per-type real vs generated)
# ──────────────────────────────────────────────────────────────

def plot_marker_gene_comparison(
    real_expr_path: str = "results/real_expression.npy",
    gen_expr_path: str = "results/generated_expression.npy",
    real_labels_path: str = "results/real_expression_labels.npy",
    gen_labels_path: str = "results/generated_expression_labels.npy",
    gene_names_path: str = "results/expression_gene_names.json",
    metrics_path: str = "results/expression_metrics.json",
    type_names: Optional[Dict[int, str]] = None,
    output_dir: Path = Path("results/figures"),
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Rich marker-gene comparison with violin plots and annotated heatmaps.

    N1: Violin + strip plot -- per-marker expression distribution (real vs gen)
    N2: Dual heatmap with cell-value annotations and row-normalized coloring
    N3: Difference heatmap with statistical significance indicators
    N4: Fold-change waterfall for all markers

    Parameters
    ----------
    real_expr_path : path to real expression matrix (.npy, cells x genes)
    gen_expr_path : path to generated expression matrix (.npy, cells x genes)
    real_labels_path : path to real cell-type label array (.npy)
    gen_labels_path : path to generated cell-type label array (.npy)
    gene_names_path : path to gene names JSON list
    metrics_path : path to expression metrics JSON
    type_names : dict mapping integer type IDs to human-readable names;
                 if ``None``, falls back to ``"Type_<id>"`` labels
    output_dir : directory for saved figures
    dpi : figure resolution
    save : whether to save figure to disk
    save_panel_fn : optional callable(fig, basename, output_dir, dpi) for
                    VCD-integrated saving; falls back to ``save_with_vcd``
    """
    if type_names is None:
        type_names = {}

    paths = [real_expr_path, gen_expr_path, gene_names_path, metrics_path]
    if not all(Path(p).exists() for p in paths):
        logger.info("Expression data not found -- skipping Panel N")
        return None

    real = np.load(real_expr_path)
    gen = np.load(gen_expr_path)
    real_labels = np.load(real_labels_path) if Path(real_labels_path).exists() else None
    gen_labels = np.load(gen_labels_path) if Path(gen_labels_path).exists() else None
    with open(gene_names_path) as f:
        gene_names = json.load(f)

    all_marker_genes: List[str] = []
    marker_cats: List[str] = []
    for cat, genes in MARKER_PANEL_GENES.items():
        for g in genes:
            if g in gene_names:
                all_marker_genes.append(g)
                marker_cats.append(cat)
    if len(all_marker_genes) < 2:
        logger.warning("Too few markers found -- skipping Panel N")
        return None

    gene_idx = [gene_names.index(g) for g in all_marker_genes]

    selected_type_ids: List[int] = []
    selected_type_names: List[str] = []
    if real_labels is not None:
        for target_name in MARKER_PANEL_TYPES:
            for tid, tname in type_names.items():
                if target_name.lower() in tname.lower():
                    if tid in np.unique(real_labels):
                        selected_type_ids.append(tid)
                        selected_type_names.append(tname[:30])
                        break
    if len(selected_type_ids) < 3 and real_labels is not None:
        unique, counts = np.unique(real_labels, return_counts=True)
        top4 = unique[np.argsort(counts)[-4:]]
        selected_type_ids = top4.tolist()
        selected_type_names = [type_names.get(int(t), f"Type_{t}")[:30]
                               for t in selected_type_ids]

    n_markers = len(all_marker_genes)
    n_sel_types = len(selected_type_ids)

    fig = plt.figure(figsize=(11.0, 7.5))
    gs = fig.add_gridspec(2, 2, wspace=0.60, hspace=0.50)
    set_figure_suptitle(fig, "Marker Gene Comparison", fontsize=11)

    # -- N1: Paired bars with error whiskers and category coloring --
    ax1 = fig.add_subplot(gs[0, 0])
    real_marker_means = np.array([real[:, gi].mean() for gi in gene_idx])
    gen_marker_means = np.array([gen[:, gi].mean() for gi in gene_idx])
    real_marker_stds = np.array([real[:, gi].std() for gi in gene_idx])
    gen_marker_stds = np.array([gen[:, gi].std() for gi in gene_idx])

    cat_colors = {"CD8+ T": "#1565C0", "Myeloid": "#C62828",
                  "Epithelial": "#2E7D32", "Stromal": "#6A1B9A"}
    x = np.arange(n_markers)
    w = 0.35
    for i, (g, c) in enumerate(zip(all_marker_genes, marker_cats)):
        bc = cat_colors.get(c, "#666")
        ax1.bar(i - w / 2, real_marker_means[i], w, yerr=real_marker_stds[i] * 0.3,
                color=bc, alpha=0.75, edgecolor="white", capsize=3,
                error_kw=dict(lw=0.8))
        ax1.bar(i + w / 2, gen_marker_means[i], w, yerr=gen_marker_stds[i] * 0.3,
                color=bc, alpha=0.4, edgecolor=bc, linewidth=1.5,
                capsize=3, error_kw=dict(lw=0.8), hatch="///")
        fc = gen_marker_means[i] / (real_marker_means[i] + 1e-8)
        # Only annotate fc when noticeably different from 1.0
        if abs(fc - 1.0) > 0.02:
            color_fc = COLORS["good"] if 0.95 <= fc <= 1.05 else COLORS["bad"]
            ax1.text(i, max(real_marker_means[i], gen_marker_means[i]) +
                     max(real_marker_stds[i], gen_marker_stds[i]) * 0.3 + 0.005,
                     f"{fc:.3f}\u00d7", ha="center", fontsize=8, color=color_fc)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{g[:15]}" for g, c in zip(all_marker_genes, marker_cats)],
                        fontsize=10, rotation=90, ha="center")
    ax1.set_ylabel("Expression", fontsize=10)
    ax1.set_title("Marker Expression by Lineage", fontsize=11)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#666", alpha=0.75, label="Real"),
                       Patch(facecolor="#666", alpha=0.4, hatch="///", label="Gen")]
    ax1.legend(handles=legend_elements, fontsize=8, loc="upper right",
               bbox_to_anchor=(1.0, 1.0))

    # -- N2 & N3: Heatmaps (if per-type labels) --
    if n_sel_types >= 2 and real_labels is not None and gen_labels is not None:
        real_heat = np.zeros((n_sel_types, n_markers))
        gen_heat = np.zeros((n_sel_types, n_markers))
        real_heat_std = np.zeros((n_sel_types, n_markers))
        gen_heat_std = np.zeros((n_sel_types, n_markers))
        for i, tid in enumerate(selected_type_ids):
            r_mask = real_labels == tid
            g_mask = gen_labels == tid
            for j, gi in enumerate(gene_idx):
                if r_mask.any():
                    real_heat[i, j] = real[r_mask][:, gi].mean()
                    real_heat_std[i, j] = real[r_mask][:, gi].std()
                if g_mask.any():
                    gen_heat[i, j] = gen[g_mask][:, gi].mean()
                    gen_heat_std[i, j] = gen[g_mask][:, gi].std()

        # N2: Side-by-side annotated heatmaps
        ax2 = fig.add_subplot(gs[0, 1])
        combined = np.hstack([real_heat, gen_heat])
        vmin, vmax = combined.min(), combined.max()
        gap_col = np.full((n_sel_types, 1), np.nan)
        display = np.hstack([real_heat, gap_col, gen_heat])

        cmap_n2 = mcolors.LinearSegmentedColormap.from_list(
            "expr_heat", ["#fff3e0", "#ffcc80", "#ff9800", "#e65100", "#bf360c"], N=256)
        im = ax2.imshow(display, cmap=cmap_n2, aspect="auto", vmin=vmin, vmax=vmax)
        ax2.set_yticks(range(n_sel_types))
        ax2.set_yticklabels([n[:20] for n in selected_type_names], fontsize=8)
        xtick_pos = list(range(n_markers)) + [n_markers] + list(range(n_markers + 1, 2 * n_markers + 1))
        xtick_labels = all_marker_genes + ["|"] + all_marker_genes
        ax2.set_xticks(xtick_pos)
        ax2.set_xticklabels(xtick_labels, fontsize=8, rotation=90, ha="center")
        ax2.set_title("Per-Type \u00d7 Marker (Real | Gen)", fontsize=10, pad=8)
        ax2.text(n_markers / 2 - 0.5, -1.2, "Real", ha="center",
             fontsize=10, color=COLORS["real"])
        ax2.text(n_markers + 0.5 + n_markers / 2 - 0.5, -1.2, "Generated",
             ha="center", fontsize=10, color=COLORS["generated"])

        add_colorbar_safe(im, ax=ax2, label="Expr.", shrink=0.6, pad=0.05)

        # N3: Difference heatmap with significance
        ax3 = fig.add_subplot(gs[1, 0])
        diff = gen_heat - real_heat
        pct_diff = diff / (np.abs(real_heat) + 1e-8) * 100
        max_abs = max(abs(diff.min()), abs(diff.max()), 0.01)

        im3 = ax3.imshow(diff, cmap="RdBu_r", aspect="auto",
                         vmin=-max_abs, vmax=max_abs)
        ax3.set_yticks(range(n_sel_types))
        ax3.set_yticklabels([n[:20] for n in selected_type_names], fontsize=8)
        ax3.set_xticks(range(n_markers))
        ax3.set_xticklabels(all_marker_genes, fontsize=10, rotation=90, ha="center")
        ax3.set_title("\u0394 Expression (Gen \u2212 Real)", fontsize=10)
        add_colorbar_safe(im3, ax=ax3, label="\u0394", shrink=0.6, pad=0.12)
        # Only annotate cells with large differences
        for i in range(n_sel_types):
            for j in range(n_markers):
                if abs(diff[i, j]) > max_abs * 0.3:
                    txt_color = "white" if abs(diff[i, j]) > max_abs * 0.5 else "black"
                    ax3.text(j, i, f"{diff[i, j]:+.2f}",
                             ha="center", va="center", fontsize=8, color=txt_color)

        # N4: Fold-change waterfall
        ax4 = fig.add_subplot(gs[1, 1])
        fc_all = gen_marker_means / (real_marker_means + 1e-8)
        sort_fc = np.argsort(fc_all)
        fc_sorted = fc_all[sort_fc]
        names_sorted = [all_marker_genes[i] for i in sort_fc]
        cats_sorted = [marker_cats[i] for i in sort_fc]
        fc_colors = [cat_colors.get(c, "#666") for c in cats_sorted]

        bars = ax4.barh(range(n_markers), fc_sorted - 1.0, left=1.0,
                        color=fc_colors, height=0.6, edgecolor="white")
        ax4.axvline(x=1.0, color="#333", linewidth=1.5, linestyle="-")
        ax4.axvspan(0.95, 1.05, alpha=0.1, color=COLORS["good"])
        ax4.set_yticks(range(n_markers))
        ax4.set_yticklabels([f"{n} ({c})" for n, c in zip(names_sorted, cats_sorted)],
                            fontsize=8)
        ax4.set_xlabel("Fold Change (Gen / Real)", fontsize=10)
        ax4.set_title("Marker Fold Change", fontsize=11)
        for i, fc in enumerate(fc_sorted):
            # Skip annotations where fold change is negligibly close to 1.0
            if abs(fc - 1.0) < 0.01:
                continue
            ax4.text(fc + 0.002 if fc >= 1.0 else fc - 0.002, i,
                     f"{fc:.3f}\u00d7", va="center", fontsize=8,
                     ha="left" if fc >= 1.0 else "right",
                     fontweight="bold" if abs(fc - 1.0) > 0.05 else "normal")
    else:
        ax_fallback = fig.add_subplot(gs[0, 1])
        ax_fallback.text(0.5, 0.5, "Per-type labels not available",
                         ha="center", va="center", transform=ax_fallback.transAxes)

    if save:
        if save_panel_fn:
            save_panel_fn(fig, "panel_n_marker_gene_comparison", Path(output_dir), dpi)
        else:
            save_with_vcd(fig, Path(output_dir) / "panel_n_marker_gene_comparison.png", dpi)
    return fig
