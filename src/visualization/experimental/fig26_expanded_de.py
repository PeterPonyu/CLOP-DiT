"""
fig26_expanded_de.py — Fig 26: Expanded DE concordance across 5 contrasts.

Shows differential expression concordance between real and generated cells
for all 5 biologically meaningful contrasts including the 2 previously
unused ones (B vs plasma, activated vs resting T).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from ..direct_layout import bind_figure_region
from ..style import (
    COLORS,
    FONT_ANNOTATION,
    FONT_HEATMAP_CELL,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TICK,
    FONT_TICK_DENSE,
    FONT_TITLE,
    abbreviate_cell_type,
    add_panel_label,
    apply_style,
    save_panel,
    style_axes,
)
from ..explicit_positioning import add_axes_next_to
from src.utils.paths import RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)


def _abbrev_contrast(name: str, max_len: int = 16) -> str:
    """Shorten a contrast name for axis display."""
    parts = name.split("_vs_")
    if len(parts) == 2:
        a = abbreviate_cell_type(parts[0].replace("_", " "), max_len=max_len)
        b = abbreviate_cell_type(parts[1].replace("_", " "), max_len=max_len)
        return f"{a}\nvs {b}"
    return abbreviate_cell_type(name.replace("_", " "), max_len=max_len * 2)


def plot_expanded_de(
    de_path: str | Path = "results/downstream/expanded_de_concordance.json",
    output_dir: str | Path = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Fig 26: Expanded DE concordance.

    Panel (a): logFC scatter for the two NEW contrasts (B vs plasma, activated vs resting T).
    Panel (b): Concordance summary heatmap across all 5 contrasts.
    Panel (c): Top-K Jaccard overlap bars.
    """
    apply_style()
    de_path = Path(de_path)
    output_dir = Path(output_dir)

    if not de_path.exists():
        logger.warning("Expanded DE results not found: %s", de_path)
        return None

    with open(de_path) as f:
        de_data = json.load(f)

    contrasts = list(de_data.keys())
    if not contrasts:
        logger.warning("No DE contrasts found")
        return None

    n_contrasts = len(contrasts)

    # Figure layout: scatter pair | heatmap | bars
    fig = plt.figure(figsize=(16.5, 7.5))
    layout = bind_figure_region(fig, (0.12, 0.14, 0.96, 0.90))

    p_a, p_b, p_c = layout.split_cols([0.85, 1.1, 0.7], gap=0.17)

    # ── Panel (a): logFC scatters for new contrasts ──
    # Find the new contrasts (B vs plasma, Activated vs Resting) or use last 2
    new_contrasts = []
    for c in contrasts:
        c_lower = c.lower()
        if any(kw in c_lower for kw in ["plasma", "b_lymph", "activated", "resting"]):
            new_contrasts.append(c)
    if len(new_contrasts) < 2:
        new_contrasts = contrasts[-2:] if len(contrasts) >= 2 else contrasts

    scatter_regions = p_a.split_rows(len(new_contrasts), gap=0.18)

    for i, contrast in enumerate(new_contrasts[:2]):
        data = de_data[contrast]
        real_logfc = np.array(data.get("_real_logfc", []))
        gen_logfc = np.array(data.get("_gen_logfc", []))

        if len(real_logfc) == 0 or len(gen_logfc) == 0:
            continue

        ax = scatter_regions[i].add_axes(fig)
        if i == 0:
            add_panel_label(ax, "a", x=-0.20, y=1.06)

        ax.scatter(real_logfc, gen_logfc, s=4, alpha=0.3, c=COLORS["real"],
                   edgecolors="none", rasterized=True)

        # Fit line
        mask = np.isfinite(real_logfc) & np.isfinite(gen_logfc)
        if mask.sum() > 10:
            slope, intercept, r_val, p_val, _ = scipy_stats.linregress(
                real_logfc[mask], gen_logfc[mask])
            xline = np.linspace(real_logfc[mask].min(), real_logfc[mask].max(), 100)
            ax.plot(xline, slope * xline + intercept, color=COLORS["generated"],
                    lw=1.2, alpha=0.8)
            ax.text(0.03, 0.95, f"r = {r_val:.3f}",
                    transform=ax.transAxes, fontsize=FONT_ANNOTATION,
                    va="top", ha="left", color=COLORS["generated"],
                    fontweight="bold")

        lim = max(abs(real_logfc[mask]).max(), abs(gen_logfc[mask]).max()) * 1.15 if mask.sum() else 5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.locator_params(axis="x", nbins=3)  # fewer ticks to reduce density
        ax.axhline(0, color="grey", lw=0.5, alpha=0.3)
        ax.axvline(0, color="grey", lw=0.5, alpha=0.3)
        ax.locator_params(axis="y", nbins=3)
        # Use compact tick format to avoid label truncation at figure border
        import matplotlib.ticker as mticker
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1g"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1g"))

        short_name = _abbrev_contrast(contrast, max_len=22).replace("\n", " ")
        ax.set_title(short_name, fontsize=FONT_TICK_DENSE)
        if i == len(new_contrasts[:2]) - 1:
            ax.set_xlabel("Real logFC", fontsize=FONT_LABEL)
        ax.set_ylabel("Gen. logFC", fontsize=FONT_TICK_DENSE)
        style_axes(ax)

        # DE logFC scale limitation annotation (first scatter only)
        if i == 0 and len(real_logfc) > 0 and len(gen_logfc) > 0:
            max_lfc = max(abs(real_logfc).max(), abs(gen_logfc).max())
            ax.text(0.97, 0.03, f"max |logFC| \u2248 {max_lfc:.1e}\n(scGPT output space)",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=FONT_ANNOTATION - 1, style="italic", color=COLORS["neutral"],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"))

    # ── Panel (b): Concordance heatmap ──
    ax_b = p_b.add_axes(fig)
    add_panel_label(ax_b, "b", x=-0.12, y=1.06)

    metrics = ["logfc_pearson_r", "logfc_spearman_rho", "top_k_jaccard", "top_k_sign_agreement"]
    metric_labels = ["Pearson r", "Spearman ρ", "Jaccard@100", "Sign Agree."]
    contrast_labels = [_abbrev_contrast(c, max_len=10) for c in contrasts]

    heatmap_data = np.full((n_contrasts, len(metrics)), np.nan)
    for i, c in enumerate(contrasts):
        for j, m in enumerate(metrics):
            heatmap_data[i, j] = de_data[c].get(m, np.nan)

    im = ax_b.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax_b.set_xticks(range(len(metrics)))
    ax_b.set_xticklabels(metric_labels, fontsize=FONT_TICK_DENSE - 1, rotation=35, ha="right")
    ax_b.set_yticks(range(n_contrasts))
    ax_b.set_yticklabels(contrast_labels, fontsize=FONT_HEATMAP_CELL)
    ax_b.set_title("DE Concordance Metrics", fontsize=FONT_TITLE)

    for ri in range(heatmap_data.shape[0]):
        for ci in range(heatmap_data.shape[1]):
            val = heatmap_data[ri, ci]
            if np.isfinite(val):
                color = "white" if val > 0.6 else "black"
                ax_b.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                          fontsize=max(FONT_HEATMAP_CELL - 1, 7), color=color, clip_on=True)

    cax = add_axes_next_to(fig, ax_b, side="right", width=0.008,
                           height=ax_b.get_position().height * 0.55,
                           pad=0.014, align="center")
    cbar = fig.colorbar(im, cax=cax)
    # No label to avoid overlap with heatmap ytick labels
    cax.tick_params(labelsize=6)

    # ── Panel (c): Jaccard overlap bars ──
    ax_c = p_c.add_axes(fig)
    add_panel_label(ax_c, "c", x=-0.18, y=1.06)

    jaccard_vals = [de_data[c].get("top_k_jaccard", 0) for c in contrasts]
    sign_vals = [de_data[c].get("top_k_sign_agreement", 0) for c in contrasts]
    y_pos = np.arange(n_contrasts)

    bars = ax_c.barh(y_pos - 0.18, jaccard_vals, height=0.35, label="Jaccard@100",
                     color=COLORS["real"], alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax_c.barh(y_pos + 0.18, sign_vals, height=0.35, label="Sign Agree.",
                      color=COLORS["generated"], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax_c.set_yticks(y_pos)
    short_labels = [_abbrev_contrast(c, max_len=9).replace("\n", " ") for c in contrasts]
    ax_c.set_yticklabels(short_labels, fontsize=FONT_TICK_DENSE - 2)
    ax_c.set_xlabel("Score", fontsize=FONT_LABEL)
    ax_c.set_title("Top-K Overlap", fontsize=FONT_TITLE)
    ax_c.set_xlim(0, 1.0)
    ax_c.invert_yaxis()
    ax_c.legend(fontsize=FONT_LEGEND - 2, loc="upper center",
                bbox_to_anchor=(0.5, -0.08), frameon=False, ncol=2)
    style_axes(ax_c)

    # Save
    if save:
        if save_panel_fn:
            save_panel_fn(fig, "fig26_expanded_de")
        else:
            out = output_dir / "fig26_expanded_de.png"
            save_panel(fig, out, dpi=dpi)
        logger.info("Saved Fig 26 → %s", output_dir)

    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    logging.basicConfig(level=logging.INFO)
    plot_expanded_de()
