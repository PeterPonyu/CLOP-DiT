"""
panels_de_concordance.py — Panel R: Differential expression concordance (real vs generated).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, add_colorbar_safe, save_panel, set_figure_suptitle, style_axes, add_panel_label

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def plot_de_concordance_panel(
    de_data: Dict,
    output_dir: Path,
    dpi: int = 300,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Panel R: Differential expression concordance between real and generated.

    R1: effect-size weighted logFC scatter for the first contrast
    R2: Top-gene overlap heatmap across all contrasts
    R3: Summary bars (Pearson r, Jaccard, sign agreement per contrast)
    """
    if not de_data:
        logger.info("No DE data — skipping Panel R")
        return None

    contrasts = list(de_data.keys())
    n_contrasts = len(contrasts)

    fig = plt.figure(figsize=(14.8, 7.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.0], wspace=0.62)
    # Title moved to LaTeX caption
    # set_figure_suptitle(fig, "DE Concordance — Real vs Generated", fontsize=11)
    fig._clop_layout_rect = (0.02, 0.03, 0.98, 0.95)

    ax = fig.add_subplot(gs[0])
    add_panel_label(ax, 'a')
    first_key = contrasts[0]
    first = de_data[first_key]
    real_logfc = np.array(first.get("_real_logfc", []))
    gen_logfc = np.array(first.get("_gen_logfc", []))
    real_padj = np.array(first.get("_real_padj", []), dtype=float)
    gen_padj = np.array(first.get("_gen_padj", []), dtype=float)
    shared_genes = first.get("_shared_genes", [])

    if len(real_logfc) > 0 and len(gen_logfc) > 0:
        effect_size = (np.abs(real_logfc) + np.abs(gen_logfc)) / 2.0
        effect_scale = np.percentile(effect_size, 95) if len(effect_size) > 4 else effect_size.max()
        effect_scale = effect_scale if effect_scale > 0 else 1.0
        marker_sizes = 12 + 140 * np.clip(effect_size / effect_scale, 0, 1.5)

        if len(real_padj) == len(real_logfc) and len(gen_padj) == len(gen_logfc):
            significance = -np.log10(np.clip(np.minimum(real_padj, gen_padj), 1e-300, 1.0))
            cbar_label = "min -log10(adj p)"
        else:
            significance = effect_size
            cbar_label = "Effect-size proxy"

        sc = ax.scatter(
            real_logfc,
            gen_logfc,
            c=significance,
            cmap="magma",
            s=marker_sizes,
            alpha=0.70,
            edgecolors="none",
        )

        discordant = np.sign(real_logfc) != np.sign(gen_logfc)
        if discordant.any():
            strong_discordant = discordant & (effect_size >= np.percentile(effect_size, 85))
            ax.scatter(
                real_logfc[strong_discordant],
                gen_logfc[strong_discordant],
                s=marker_sizes[strong_discordant] * 1.2,
                facecolors="none",
                edgecolors=COLORS["bad"],
                linewidths=0.8,
                label="sign disagreement",
            )

        lo = min(real_logfc.min(), gen_logfc.min()) * 1.1
        hi = max(real_logfc.max(), gen_logfc.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], color=COLORS["bad"], linestyle="--", lw=1.5,
                alpha=0.7, label="y = x")
        ax.fill_between([lo, hi], [lo - 0.4, hi - 0.4], [lo + 0.4, hi + 0.4],
                        alpha=0.05, color=COLORS["good"])
        ax.axhline(0, color="#666", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.axvline(0, color="#666", linestyle=":", linewidth=1.0, alpha=0.6)

        residuals = np.abs(gen_logfc - real_logfc) * np.maximum(effect_size, 1e-6)
        top_idx = np.argsort(residuals)[-min(2, len(residuals)):]
        for i in top_idx:
            if i < len(shared_genes):
                ax.annotate(shared_genes[i], (real_logfc[i], gen_logfc[i]),
                            fontsize=8, xytext=(8, 8), textcoords="offset points",
                            arrowprops=dict(arrowstyle="->", lw=0.5, color="#555"),
                            color="#333")

        r_val = first.get("logfc_pearson_r", 0)
        sign_agreement = first.get("top_k_sign_agreement", 0)
        ax.legend(
            fontsize=8,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
        )
        discord_pct = 100 * discordant.mean() if len(discordant) > 0 else 0
        n_genes_shown = len(shared_genes)
        ax.text(
            0.98, 0.98,
            f"r = {r_val:.3f}\nsign = {sign_agreement:.3f}\ndiscord = {discord_pct:.1f}%\nn = {n_genes_shown} genes",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=COLORS["neutral"],
        )
        add_colorbar_safe(sc, ax=ax, label=cbar_label,
                         shrink=0.5, pad=0.14, orientation="horizontal", aspect=20)

    contrast_display = first_key.replace("_", " ")[:28]
    style_axes(ax, "scatter", title="Effect-Size Concordance",
               xlabel="Real logFC", ylabel="Generated logFC")
    ax.text(
        0.02,
        0.90,
        contrast_display,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="medium",
        color=COLORS["neutral"],
    )
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    ax.tick_params(axis='x', labelsize=8)

    ax2 = fig.add_subplot(gs[1])
    add_panel_label(ax2, 'b')
    # Display labels for heatmap and bar chart — kept short to avoid tick overlap
    metric_names = ["Pears. r", "Spear. \u03c1", "Jacc.@50", "Sign agr."]
    metric_keys = ["logfc_pearson_r", "logfc_spearman_rho", "top_k_jaccard", "top_k_sign_agreement"]
    heatmap_data = np.zeros((n_contrasts, len(metric_names)))
    contrast_labels = []
    for i, cname in enumerate(contrasts):
        cd = de_data[cname]
        for j, mk in enumerate(metric_keys):
            heatmap_data[i, j] = cd.get(mk, 0)
        parts = cname.split("_vs_")
        short = f"{parts[0][:8]} v {parts[1][:8]}" if len(parts) == 2 else cname[:18]
        contrast_labels.append(short)

    im = ax2.imshow(heatmap_data, cmap="PiYG", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(len(metric_names)))
    ax2.set_xticklabels(metric_names, fontsize=8, rotation=45, ha="right")
    ax2.set_yticks(range(n_contrasts))
    ax2.set_yticklabels(contrast_labels, fontsize=7)

    for i in range(n_contrasts):
        for j in range(len(metric_names)):
            val = heatmap_data[i, j]
            if val >= 0.7:
                continue
            color = "white" if val < 0.3 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=8, color=color)

    add_colorbar_safe(im, ax=ax2, shrink=0.6)
    style_axes(ax2, "heatmap", title="Concordance Across Contrasts")

    ax3 = fig.add_subplot(gs[2])
    add_panel_label(ax3, 'c')
    x = np.arange(n_contrasts)
    n_metrics = len(metric_names)
    w = 0.8 / n_metrics
    bar_colors = [COLORS["real"], COLORS["baseline_gauss"], COLORS["warn"], COLORS["baseline_shuffle"]]

    for j, (mname, mk) in enumerate(zip(metric_names, metric_keys)):
        vals = [de_data[c].get(mk, 0) for c in contrasts]
        offset = (j - n_metrics / 2 + 0.5) * w
        ax3.bar(x + offset, vals, w, label=mname, color=bar_colors[j],
                alpha=0.85, edgecolor="white")

    short_xlabels = []
    for cname in contrasts:
        parts = cname.split("_vs_")
        short_xlabels.append(f"{parts[0][:15]}…" if len(parts) == 2 else cname[:15])

    ax3.set_xticks(x)
    ax3.set_xticklabels(short_xlabels, fontsize=7, rotation=45, ha="right")
    ax3.set_ylim(0, 1.15)
    ax3.legend(fontsize=8, ncol=2, loc="upper right", frameon=False)
    style_axes(ax3, "bar", title="Per-Contrast Summary", ylabel="Score")

    if save:
        path = save_panel(fig, output_dir / "panel_r_de_concordance.png", dpi)
        logger.info(f"Saved Panel R → {path}")
    return fig
