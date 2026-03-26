"""
fig18_de_concordance.py — Fig 18: Differential expression concordance (real vs generated).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from .direct_layout import bind_figure_region
from .explicit_positioning import add_axes_next_to
from .style import (
    COLORS, save_panel, set_figure_suptitle, style_axes,
    add_panel_label, abbreviate_cell_type,
    FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_TICK_DENSE, FONT_ANNOTATION,
    FONT_HEATMAP_CELL,
)

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
    label_offset: int = 0,
) -> Optional[plt.Figure]:
    """Fig 18: Differential expression concordance between real and generated.

    R1 (a): effect-size weighted logFC scatter for the first contrast
    R2 (b): concordance heatmap across all contrasts
    R3 (c): summary bars per contrast
    """
    if not de_data:
        logger.info("No DE data — skipping Fig 18")
        return None

    from matplotlib.ticker import MaxNLocator

    contrasts = list(de_data.keys())
    n_contrasts = len(contrasts)

    fig = plt.figure(figsize=(15.8, 6.8))
    panel_a, panel_b, panel_c = bind_figure_region(fig, (0.06, 0.10, 0.98, 0.94)).split_cols(
        [1.26, 0.92, 0.98],
        gap=[0.060, 0.055],
    )
    ax_rect_a = panel_a.inset(left=0.014, right=0.050)
    ax_rect_b = panel_b.inset(left=0.070, right=0.020)
    ax_rect_c = panel_c.inset(left=0.032, right=0.028)

    # ── Panel (a): effect-size weighted logFC scatter ──
    ax = ax_rect_a.add_axes(fig)
    add_panel_label(ax, chr(ord('a') + label_offset), x=-0.14, y=1.03)

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
            significance = -np.log10(np.clip(np.minimum(real_padj, gen_padj), 1e-50, 1.0))
            cbar_label = r"min $-\log_{10}(\mathrm{adj.\;p})$"
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
        n_label = min(6, len(residuals))
        top_idx = np.argsort(residuals)[-n_label:]
        placed_points = []
        for _k, idx in enumerate(top_idx[np.argsort(residuals[top_idx])[::-1]]):
            if idx < len(shared_genes):
                x_pt = real_logfc[idx]
                y_pt = gen_logfc[idx]
                if any(abs(x_pt - px) < 0.35 and abs(y_pt - py) < 0.35 for px, py in placed_points):
                    continue
                ox = (28 + 6 * _k) * (-1 if x_pt > np.median(real_logfc) else 1)
                oy = (18 + 5 * (_k % 3)) * (-1 if y_pt > np.median(gen_logfc) else 1)
                ax.annotate(
                    shared_genes[idx],
                    (x_pt, y_pt),
                    fontsize=FONT_ANNOTATION,
                    xytext=(ox, oy), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="#666",
                                    connectionstyle="arc3,rad=0.15"),
                    color=COLORS["annotation_dark"],
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="none", ec="none"),
                )
                placed_points.append((x_pt, y_pt))

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
        )

        # DE logFC scale limitation annotation
        max_lfc = max(abs(real_logfc).max(), abs(gen_logfc).max())
        ax.text(0.03, 0.97, f"logFC in scGPT output space\nmax |logFC| \u2248 {max_lfc:.1e}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=FONT_ANNOTATION - 1, style="italic", color=COLORS["neutral"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"))

        # Horizontal colorbar below scatter
        cax = add_axes_next_to(
            fig,
            ax,
            side="right",
            width=0.010,
            height=ax.get_position().height * 0.44,
            pad=0.014,
            align="bottom",
            y_offset=0.02,
        )
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("")
        cbar.ax.set_title("")
        cbar.ax.tick_params(labelsize=FONT_HEATMAP_CELL)

        # Small legend (sign-disagreement + y=x) inside lower-left; sparse there
        ax.legend(fontsize=FONT_ANNOTATION, frameon=False,
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
    ax.xaxis.labelpad = 10
    ax.set_title(f"Effect-Size Concordance\n{sub_line}",
                 fontsize=FONT_TITLE - 1, pad=4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.tick_params(labelsize=FONT_TICK)

    # ── Panel (b): concordance heatmap ──
    ax2 = ax_rect_b.add_axes(fig)
    add_panel_label(ax2, chr(ord('a') + label_offset + 1), x=-0.14, y=1.03)

    metric_names = ["Pears. r", "Spear. \u03c1", "Jacc.@50", "Sign agr."]
    metric_tick_labels = ["Pears.\nr", "Spear.\n\u03c1", "Jacc.\n50", "Sign\nagr."]
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
    ax2.set_xticklabels(metric_tick_labels, fontsize=FONT_TICK_DENSE,
                         rotation=0, ha="center")
    ax2.set_yticks(range(n_contrasts))
    ax2.set_yticklabels(contrast_labels, fontsize=FONT_TICK_DENSE)

    # Annotate every cell (dark text on light cells, white on dark)
    for i in range(n_contrasts):
        for j in range(len(metric_names)):
            val   = heatmap_data[i, j]
            color = "white" if val < 0.35 or val > 0.72 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=FONT_HEATMAP_CELL, fontweight="bold", color=color)

    cax2 = add_axes_next_to(
        fig,
        ax2,
        side="right",
        width=0.010,
        height=ax2.get_position().height * 0.48,
        pad=0.012,
        align="bottom",
        y_offset=0.02,
    )
    cbar2 = fig.colorbar(im, cax=cax2)
    cbar2.ax.tick_params(labelsize=FONT_HEATMAP_CELL)
    style_axes(ax2, "heatmap", title="Concordance Across Contrasts")
    ax2.set_title("Concordance Across Contrasts", x=0.60, pad=6)

    # ── Panel (c): grouped bar chart ──
    ax3 = ax_rect_c.add_axes(fig)
    add_panel_label(ax3, chr(ord('a') + label_offset + 2), x=-0.14, y=1.03)

    x        = np.arange(n_contrasts)
    n_metrics = len(metric_names)
    w         = 0.72 / n_metrics
    bar_colors = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A"]

    for j, (mname, mk) in enumerate(zip(metric_names, metric_keys)):
        vals   = [de_data[c].get(mk, 0) for c in contrasts]
        offset = (j - n_metrics / 2 + 0.5) * w
        ax3.bar(x + offset, vals, w, label=mname,
                color=bar_colors[j], alpha=0.85, edgecolor="white")

    # Build short x-labels from contrast names (biology-aware)
    xs_labels = [_abbrev_contrast(c, max_len=8) for c in contrasts]
    ax3.set_xticks(x)
    ax3.set_xticklabels(xs_labels, fontsize=6,
                         rotation=0, ha="center", multialignment="center")
    ax3.set_ylim(0, 1.12)
    ax3.legend(fontsize=FONT_ANNOTATION, ncol=2,
               loc="upper right", frameon=False)
    style_axes(ax3, "bar", title="Per-Contrast Summary", ylabel="Score")
    ax3.tick_params(labelsize=FONT_TICK)

    if save:
        path = save_panel(fig, output_dir / "fig08b_de_concordance.png", dpi)
        logger.info("Saved Fig 18 \u2192 %s", path)
    return fig
