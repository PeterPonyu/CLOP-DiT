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
from scipy import stats as scipy_stats

from .style import (
    COLORS, add_colorbar_safe, save_panel, set_figure_suptitle, style_axes,
    add_panel_label, abbreviate_cell_type,
    FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_TICK_DENSE, FONT_ANNOTATION,
    FONT_HEATMAP_CELL,
)

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def _abbrev_contrast(name: str, max_len: int = 18) -> str:
    """Shorten a contrast name like 'CD8_cytotoxic_T_vs_CD4_helper_T'
    into a readable two-part label using biology-aware abbreviation."""
    parts = name.split("_vs_")
    if len(parts) == 2:
        a = abbreviate_cell_type(parts[0].replace("_", " "), max_len=max_len)
        b = abbreviate_cell_type(parts[1].replace("_", " "), max_len=max_len)
        return f"{a}\nvs {b}"
    return abbreviate_cell_type(name.replace("_", " "), max_len=max_len * 2)


def plot_de_concordance_panel(
    de_data: Dict,
    output_dir: Path,
    dpi: int = 300,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Panel R: Differential expression concordance between real and generated.

    R1 (a): effect-size weighted logFC scatter for the first contrast
    R2 (b): concordance heatmap across all contrasts
    R3 (c): summary bars per contrast
    """
    if not de_data:
        logger.info("No DE data — skipping Panel R")
        return None

    from matplotlib.ticker import MaxNLocator

    contrasts = list(de_data.keys())
    n_contrasts = len(contrasts)

    fig = plt.figure(figsize=(15.0, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.0], wspace=0.55)
    fig._clop_layout_rect = (0.02, 0.08, 0.98, 0.96)

    # ── Panel (a): effect-size weighted logFC scatter ──
    ax = fig.add_subplot(gs[0])
    add_panel_label(ax, 'a', x=-0.10, y=1.05)

    first_key = contrasts[0]
    first = de_data[first_key]
    real_logfc = np.array(first.get("_real_logfc", []))
    gen_logfc  = np.array(first.get("_gen_logfc",  []))
    real_padj  = np.array(first.get("_real_padj",  []), dtype=float)
    gen_padj   = np.array(first.get("_gen_padj",   []), dtype=float)
    shared_genes = first.get("_shared_genes", [])

    if len(real_logfc) > 0 and len(gen_logfc) > 0:
        effect_size  = (np.abs(real_logfc) + np.abs(gen_logfc)) / 2.0
        effect_scale = np.percentile(effect_size, 95) if len(effect_size) > 4 else effect_size.max()
        effect_scale = effect_scale if effect_scale > 0 else 1.0
        marker_sizes = 12 + 140 * np.clip(effect_size / effect_scale, 0, 1.5)

        if len(real_padj) == len(real_logfc) and len(gen_padj) == len(gen_logfc):
            significance = -np.log10(np.clip(np.minimum(real_padj, gen_padj), 1e-300, 1.0))
            cbar_label = "min \u2212log\u2081\u2080(adj. p)"
        else:
            significance = effect_size
            cbar_label   = "Effect size"

        # Density shading behind the scatter
        ax.hexbin(real_logfc, gen_logfc, gridsize=30, cmap="Blues", alpha=0.25,
                  mincnt=1, linewidths=0, zorder=1)

        sc = ax.scatter(
            real_logfc, gen_logfc,
            c=significance, cmap="magma",
            s=marker_sizes, alpha=0.70, edgecolors="none",
            zorder=3,
        )

        discordant = np.sign(real_logfc) != np.sign(gen_logfc)
        if discordant.any():
            strong_discordant = discordant & (effect_size >= np.percentile(effect_size, 85))
            ax.scatter(
                real_logfc[strong_discordant], gen_logfc[strong_discordant],
                s=marker_sizes[strong_discordant] * 1.2,
                facecolors="none", edgecolors=COLORS["bad"],
                linewidths=0.8, label="sign disagreement", zorder=4,
            )

        lo = min(real_logfc.min(), gen_logfc.min()) * 1.1
        hi = max(real_logfc.max(), gen_logfc.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], color=COLORS["bad"], linestyle="--",
                lw=1.5, alpha=0.7, label="y = x")
        ax.fill_between([lo, hi], [lo - 0.4, hi - 0.4], [lo + 0.4, hi + 0.4],
                        alpha=0.05, color=COLORS["good"])
        ax.axhline(0, color="#666", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.axvline(0, color="#666", linestyle=":", linewidth=1.0, alpha=0.6)

        # ── Gene name labels: top-6 by weighted residual, staggered to avoid pileup ──
        residuals = np.abs(gen_logfc - real_logfc) * np.maximum(effect_size, 1e-6)
        n_label   = min(8, len(residuals))
        top_idx   = np.argsort(residuals)[-n_label:]
        # Staggered offsets: alternate quadrants so labels fan out from the origin
        _offsets = [(35, 25), (-38, 30), (35, -28), (-38, -30), (48, 12), (-50, 12), (30, -35), (-35, 35)]
        for _k, idx in enumerate(top_idx):
            if idx < len(shared_genes):
                ox, oy = _offsets[_k % len(_offsets)]
                ax.annotate(
                    shared_genes[idx],
                    (real_logfc[idx], gen_logfc[idx]),
                    fontsize=FONT_ANNOTATION,
                    xytext=(ox, oy), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="#666",
                                    connectionstyle="arc3,rad=0.15"),
                    color=COLORS["annotation_dark"],
                    ha="center",
                )

        # ── Minimal in-plot annotation: 2 key stats only ──
        pearson_r,    pearson_p    = scipy_stats.pearsonr(real_logfc, gen_logfc)
        sign_agreement = first.get("top_k_sign_agreement", 0)
        discord_pct    = 100 * discordant.mean() if len(discordant) > 0 else 0

        def _fmt_p(p: float) -> str:
            return f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"

        ax.text(
            0.97, 0.97,
            f"r = {pearson_r:.3f}  (p {_fmt_p(pearson_p)})\n"
            f"Sign agr. = {sign_agreement:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FONT_ANNOTATION,
            color=COLORS["neutral"],
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88),
        )

        # Horizontal colorbar below scatter
        add_colorbar_safe(sc, ax=ax, label=cbar_label,
                          shrink=0.55, pad=0.15,
                          orientation="horizontal", aspect=22)

        # Small legend (sign-disagreement + y=x) inside lower-left; sparse there
        ax.legend(fontsize=FONT_ANNOTATION, frameon=True,
                  framealpha=0.85, edgecolor="none",
                  loc="lower left", bbox_to_anchor=(0.01, 0.01))

    # Contrast name embedded in title as a smaller second line — no in-axes text needed
    parts = first_key.split("_vs_")
    if len(parts) == 2:
        a_short = abbreviate_cell_type(parts[0].replace("_", " "), max_len=20)
        b_short = abbreviate_cell_type(parts[1].replace("_", " "), max_len=20)
        sub_line = f"{a_short}  vs  {b_short}"
    else:
        sub_line = abbreviate_cell_type(first_key.replace("_", " "), max_len=38)

    style_axes(ax, "scatter", xlabel="Real logFC", ylabel="Generated logFC")
    ax.set_title(f"Effect-Size Concordance\n{sub_line}",
                 fontsize=FONT_TITLE, pad=6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.tick_params(labelsize=FONT_TICK)

    # ── Panel (b): concordance heatmap ──
    ax2 = fig.add_subplot(gs[1])
    add_panel_label(ax2, 'b', x=-0.10, y=1.05)

    metric_names = ["Pears. r", "Spear. \u03c1", "Jacc.@50", "Sign agr."]
    metric_keys  = ["logfc_pearson_r", "logfc_spearman_rho",
                    "top_k_jaccard", "top_k_sign_agreement"]
    heatmap_data   = np.zeros((n_contrasts, len(metric_names)))
    contrast_labels = []
    for i, cname in enumerate(contrasts):
        cd = de_data[cname]
        for j, mk in enumerate(metric_keys):
            heatmap_data[i, j] = cd.get(mk, 0)
        contrast_labels.append(_abbrev_contrast(cname, max_len=16))

    im = ax2.imshow(heatmap_data, cmap="PiYG", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(len(metric_names)))
    ax2.set_xticklabels(metric_names, fontsize=FONT_TICK_DENSE,
                         rotation=40, ha="right")
    ax2.set_yticks(range(n_contrasts))
    ax2.set_yticklabels(contrast_labels, fontsize=FONT_TICK_DENSE)

    # Annotate every cell (dark text on light cells, white on dark)
    for i in range(n_contrasts):
        for j in range(len(metric_names)):
            val   = heatmap_data[i, j]
            color = "white" if val < 0.35 or val > 0.72 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=FONT_HEATMAP_CELL, fontweight="bold", color=color)

    add_colorbar_safe(im, ax=ax2, shrink=0.55, pad=0.03)
    style_axes(ax2, "heatmap", title="Concordance Across Contrasts")
    ax2.set_title("Concordance Across Contrasts", fontsize=FONT_TITLE)

    # ── Panel (c): grouped bar chart ──
    ax3 = fig.add_subplot(gs[2])
    add_panel_label(ax3, 'c', x=-0.10, y=1.05)

    x        = np.arange(n_contrasts)
    n_metrics = len(metric_names)
    w         = 0.72 / n_metrics
    bar_colors = [COLORS["real"], COLORS["baseline_gauss"],
                  COLORS["warn"], COLORS["baseline_shuffle"]]

    for j, (mname, mk) in enumerate(zip(metric_names, metric_keys)):
        vals   = [de_data[c].get(mk, 0) for c in contrasts]
        offset = (j - n_metrics / 2 + 0.5) * w
        ax3.bar(x + offset, vals, w, label=mname,
                color=bar_colors[j], alpha=0.85, edgecolor="white")

    # Build short x-labels from contrast names (two-line, biology-aware)
    xs_labels = [_abbrev_contrast(c, max_len=14) for c in contrasts]
    ax3.set_xticks(x)
    ax3.set_xticklabels(xs_labels, fontsize=FONT_TICK_DENSE,
                         rotation=30, ha="right", multialignment="center")
    ax3.set_ylim(0, 1.12)
    ax3.legend(fontsize=FONT_ANNOTATION, ncol=2,
               loc="upper right", frameon=False)
    style_axes(ax3, "bar", title="Per-Contrast Summary", ylabel="Score")
    ax3.set_title("Per-Contrast Summary", fontsize=FONT_TITLE)
    ax3.set_ylabel("Score", fontsize=FONT_LABEL)
    ax3.tick_params(labelsize=FONT_TICK)

    if save:
        path = save_panel(fig, output_dir / "panel_r_de_concordance.png", dpi)
        logger.info("Saved Panel R \u2192 %s", path)
    return fig
