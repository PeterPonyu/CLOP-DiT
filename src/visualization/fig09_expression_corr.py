"""
fig09_expression_corr.py -- Article Figure 9: Gene Expression Correlation.

  H1: Density scatter of per-gene mean expression with residual coloring
  H2: Per-type Pearson r lollipop chart with tier shading
  H3: Marker gene grouped bars with error bars and fold-change annotation
  H4: Per-gene residual distribution (gen - real)

Standalone function extracted from panels_expression for composability and
VCD-integrated saving via an optional ``save_panel_fn`` callback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .direct_layout import bind_figure_region
from .explicit_positioning import add_axes_next_to
from .style import COLORS, FONT_DENSE_YTICK, abbreviate_cell_type, add_panel_label, reserve_annotation_slot, save_with_vcd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# FIGURE 9: Gene Expression Correlation
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

    fig = plt.figure(figsize=(12.0, 6.8))
    # Note: Figure-level title removed per revision requirements; stats moved to caption
    layout = bind_figure_region(fig, (0.07, 0.12, 0.93, 0.94))
    top_row, bottom_row = layout.split_rows(2, hspace=0.50)
    top_left, top_right = top_row.split_cols([0.86, 1.14], gap=0.050)
    bottom_left, bottom_right = bottom_row.split_cols([1.04, 0.92], gap=0.050)

    # -- H1: Density scatter with residual coloring --
    ax1 = top_left.inset(right=0.028).add_axes(fig)
    abs_res = np.abs(residuals)
    resid_vmax = float(np.percentile(abs_res, 95)) if len(abs_res) else 1.0
    sc = ax1.scatter(real_means, gen_means, c=abs_res, cmap="magma_r",
                     s=10, alpha=0.55, edgecolors="none",
                     vmin=0, vmax=resid_vmax)
    lo = min(real_means.min(), gen_means.min()) - 0.2
    hi = max(real_means.max(), gen_means.max()) + 0.2
    ax1.plot([lo, hi], [lo, hi], color=COLORS["bad"], linestyle="--", lw=1.5,
             alpha=0.7, label="y = x", zorder=1)
    ax1.fill_between([lo, hi], [lo - 0.1, hi - 0.1], [lo + 0.1, hi + 0.1],
                     alpha=0.06, color=COLORS["good"], zorder=0)
    ax1.set_xlabel("Real Mean Expression", fontsize=11)
    ax1.set_ylabel("Generated Mean Expression", fontsize=11)
    ax1.set_title("Per-Gene Correlation", fontsize=12)
    from matplotlib.ticker import MaxNLocator as _MaxNLoc, ScalarFormatter as _ScalarFormatter
    ax1.xaxis.set_major_locator(_MaxNLoc(nbins=3, prune="both"))
    ax1.yaxis.set_major_locator(_MaxNLoc(nbins=3, prune="both"))
    cax = add_axes_next_to(
        fig,
        ax1,
        side="right",
        width=0.012,
        height=ax1.get_position().height * 0.32,
        pad=0.014,
        align="bottom",
        y_offset=0.018,
    )
    cbar = fig.colorbar(sc, cax=cax)
    if getattr(cbar, "solids", None) is not None:
        try:
            cbar.solids.set_edgecolor("face")
        except Exception:
            pass
    cbar.set_label("|Resid|", fontsize=9)
    cbar.set_ticks(np.linspace(0, resid_vmax, 3))
    cbar_fmt = _ScalarFormatter(useMathText=True)
    cbar_fmt.set_scientific(True)
    cbar_fmt.set_powerlimits((0, 0))
    cbar.formatter = cbar_fmt
    cbar.update_ticks()
    # Apply tick styling AFTER update_ticks() which rebuilds tick label objects
    cbar.ax.tick_params(labelsize=7, length=2, pad=1)
    cbar.ax.yaxis.get_offset_text().set_fontsize(7)
    cbar.ax.yaxis.get_offset_text().set_visible(True)
    add_panel_label(ax1, 'e', x=-0.12, y=1.04)

    # Gene callouts for top-3 residual outliers. Labels placed adjacent to
    # their data points using display-space offsets (not axes-fraction slots)
    # so leader lines stay short and labels do not collide with the colorbar.
    outlier_idx = np.argsort(abs_res)[-3:]
    outlier_idx = outlier_idx[np.argsort(abs_res[outlier_idx])[::-1]]
    valid_idx = [i for i in outlier_idx if i < len(gene_names)]
    x_lo, x_hi = ax1.get_xlim()
    y_lo, y_hi = ax1.get_ylim()
    x_span = x_hi - x_lo + 1e-12
    y_span = y_hi - y_lo + 1e-12
    for i in valid_idx:
        xv, yv = real_means[i], gen_means[i]
        x_frac = (xv - x_lo) / x_span
        y_frac = (yv - y_lo) / y_span
        # Canonical adjacent-offset callout pattern shared across Figs 4E / 5A /
        # 10D / 10H. 14 pt + slightly larger offsets and padding per user spec.
        dx_pt = -24 if x_frac > 0.62 else 18
        dy_pt = -16 if y_frac > 0.62 else 14
        ax1.annotate(
            gene_names[i],
            xy=(xv, yv),
            xycoords="data",
            xytext=(dx_pt, dy_pt),
            textcoords="offset points",
            fontsize=14,
            ha="right" if dx_pt < 0 else "left",
            va="center",
            color="#333",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#BBB",
                      lw=0.4, alpha=0.95),
            arrowprops=dict(
                arrowstyle="-",
                lw=0.5,
                color="#888",
                alpha=0.65,
                shrinkA=1,
                shrinkB=1,
            ),
            zorder=6,
            annotation_clip=True,
        )

    # -- H2: Per-type deviation lollipop chart (1 - r, log scale) --
    ax2 = top_right.inset(left=0.17, right=0.05).add_axes(fig)
    per_type = metrics.get("per_type_expression_fidelity", {})
    if per_type:
        type_names_sorted = sorted(per_type.keys(),
                                   key=lambda k: per_type[k]["pearson_r"])
        type_rs = [per_type[n]["pearson_r"] for n in type_names_sorted]

        # Show only bottom 5 and top 5 types to avoid 69-label overlap
        n_show_each = min(5, len(type_rs) // 2)
        if len(type_rs) > 2 * n_show_each + 2:
            show_idx = list(range(n_show_each)) + list(range(len(type_rs) - n_show_each, len(type_rs)))
            type_names_sorted = [type_names_sorted[i] for i in show_idx]
            type_rs = [type_rs[i] for i in show_idx]
        short_names = [abbreviate_cell_type(n, 24) for n in type_names_sorted]

        # Plot deviation (1 - r) instead of raw Pearson r
        type_devs = [max(1 - r, 1e-12) for r in type_rs]
        y_pos = np.arange(len(type_devs))

        dev_color = COLORS["real"]
        ax2.hlines(y_pos, min(type_devs) * 0.5, type_devs, color="#DDD", linewidth=0.8, zorder=1)
        ax2.scatter(type_devs, y_pos, c=dev_color, s=30, zorder=3, edgecolors="white",
                    linewidths=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(short_names, fontsize=FONT_DENSE_YTICK, ha="right")
        ax2.set_xscale('log')
        ax2.set_xlabel("Deviation (1 \u2212 r)", fontsize=11)

        # Annotation about expression space
        ax2.text(0.95, 0.05, "Expression in scGPT binned space",
                 transform=ax2.transAxes, ha="right", va="bottom",
                 fontsize=9, style="italic", color=COLORS["neutral"])
    else:
        ax2.text(0.5, 0.5, "No per-type data", ha="center", va="center",
                 transform=ax2.transAxes)
    ax2.set_title("Per-Type Expression Fidelity", fontsize=12)
    add_panel_label(ax2, 'f', x=-0.12, y=1.04)

    # -- H3: Marker gene expression with error bars --
    ax3 = bottom_left.add_axes(fig)
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
        n_real = real.shape[0]
        n_gen = gen.shape[0]

        x = np.arange(len(all_marker_genes))
        width = 0.35
        ax3.bar(x - width / 2, r_means, width, yerr=r_stds / np.sqrt(n_real),
                label="Real", color=COLORS["real"], alpha=0.85, edgecolor="white",
                capsize=3, error_kw=dict(lw=0.8))
        ax3.bar(x + width / 2, g_means, width, yerr=g_stds / np.sqrt(n_gen),
                label="Gen", color=COLORS["generated"], alpha=0.85, edgecolor="white",
                capsize=3, error_kw=dict(lw=0.8))

        # Fold-change annotations -- only annotate outliers (fc > 1.05 or < 0.95)
        for i, (rm, gm) in enumerate(zip(r_means, g_means)):
            fc = gm / (rm + 1e-8)
            if abs(fc - 1.0) > 0.10:
                color = COLORS["good"] if 0.95 <= fc <= 1.05 else COLORS["bad"]
                ax3.text(i, max(rm, gm) + max(r_stds[i], g_stds[i]) * 0.5 + 0.01,
                         f"{fc:.2f}\u00d7", ha="center", fontsize=10, color=color)

        ax3.set_xticks(x)
        ax3.set_xticklabels(
            [g for g in all_marker_genes],
            fontsize=10, rotation=90, ha="center",
        )
        ax3.set_ylabel("Expression (mean \u00b1 SEM)", fontsize=11)
        legend_handles_c, legend_labels_c = ax3.get_legend_handles_labels()
    else:
        ax3.text(0.5, 0.5, "No marker genes found", ha="center", va="center",
                 transform=ax3.transAxes)
    ax3.set_title("Marker Gene Expression", fontsize=12)
    # Note: do NOT set xaxis MaxNLocator here -- it would override the explicit
    # gene-name tick labels set above (set_xticks / set_xticklabels).
    ax3.yaxis.set_major_locator(_MaxNLoc(nbins=4, prune="both"))
    add_panel_label(ax3, 'g', x=-0.12, y=1.04)

    # -- H4: Residual distribution --
    ax4 = bottom_right.inset(left=0.07, right=0.02).add_axes(fig)
    ax4.hist(residuals, bins=60, color=COLORS["real"], alpha=0.7, edgecolor="white",
             density=True)
    ax4.tick_params(axis='x', labelsize=10, rotation=30)
    ax4.axvline(x=0, color=COLORS["bad"], linestyle="--", linewidth=1.5, label="Zero")
    ax4.axvline(x=residuals.mean(), color=COLORS["warn"], linestyle="-", linewidth=1.5,
                label=f"Mean={residuals.mean():.3f}")
    ax4.set_xlabel("Residual (Gen \u2212 Real)", fontsize=11)
    ax4.set_ylabel("Density", fontsize=11)
    ax4.set_title("Per-Gene Residual Distribution", fontsize=12)
    ax4.legend(fontsize=10, frameon=False)
    ax4.xaxis.set_major_locator(_MaxNLoc(nbins=4, prune="both"))
    ax4.yaxis.set_major_locator(_MaxNLoc(nbins=4, prune="both"))
    add_panel_label(ax4, 'h', x=-0.12, y=1.04)

    pct_within_01 = (np.abs(residuals) < 0.1).mean() * 100
    pct_within_001 = (np.abs(residuals) < 0.01).mean() * 100
    ax4.text(0.95, 0.95,
             f"|\u0394|<0.01: {pct_within_001:.0f}%\n|\u0394|<0.10: {pct_within_01:.0f}%",
             transform=ax4.transAxes, ha="right", va="top", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="none", alpha=0.9))

    # Real/Gen color semantics are consistent across the paper; avoid adding a
    # separate legend here because it masks neighbouring panel content.

    if save:
        if save_panel_fn:
            save_panel_fn(fig, "fig04b_expression_correlation", Path(output_dir), dpi)
        else:
            save_with_vcd(fig, Path(output_dir) / "fig04b_expression_correlation.png", dpi)
    return fig
