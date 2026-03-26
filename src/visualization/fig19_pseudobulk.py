"""
fig19_pseudobulk.py — Pseudobulk validation figure.

Panels:
  (a) Per-type pseudobulk Pearson correlation (sorted bar chart)
  (b) Cross-type gene correlation histogram
  (c) Top/bottom type scatter (pseudobulk real vs gen for best/worst)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .direct_layout import bind_figure_region
from .style import (
    COLORS, add_panel_label, apply_style, save_with_vcd, style_axes,
    abbreviate_cell_type, FONT_LABEL, FONT_TITLE, FONT_ANNOTATION, FONT_SMALL,
)
from src.utils.paths import RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)


def plot_pseudobulk_validation(output_dir=None, dpi=300):
    """Generate pseudobulk validation figure."""
    apply_style()
    output_dir = Path(output_dir or FIG_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_dir = RESULTS_DIR / "validation"
    summary_path = val_dir / "pseudobulk_validation.json"
    per_type_path = val_dir / "pseudobulk_per_type.json"

    if not summary_path.exists() or not per_type_path.exists():
        logger.warning("Pseudobulk validation results not found. "
                       "Run: python scripts/analysis/pseudobulk_validation.py")
        return None

    with open(summary_path) as f:
        summary = json.load(f)
    with open(per_type_path) as f:
        per_type = json.load(f)

    # Sort types by correlation
    sorted_types = sorted(per_type.items(), key=lambda x: x[1]["pearson_r"], reverse=True)
    names = [abbreviate_cell_type(n, max_len=16) for n, _ in sorted_types]
    pearson_vals = [v["pearson_r"] for _, v in sorted_types]
    cosine_vals = [v["cosine_sim"] for _, v in sorted_types]

    fig = plt.figure(figsize=(8.0, 3.5))
    layout = bind_figure_region(fig, (0.10, 0.14, 0.98, 0.92))
    left, right = layout.split_cols([1.2, 1.0], wspace=0.36)

    # Panel (a): Per-type Pearson r bar chart
    ax = left.inset(right=0.02).add_axes(fig)
    add_panel_label(ax, 'a', x=-0.12, y=1.10)

    median_r = np.median(pearson_vals)
    colors = [COLORS["good"] if v >= 0.8 else (COLORS["warn"] if v >= 0.5 else COLORS["bad"])
              for v in pearson_vals]

    n_types = len(names)
    step = max(1, n_types // 15)
    thin_labels = [names[i] if i % step == 0 or i == n_types - 1 else "" for i in range(n_types)]

    ax.barh(range(n_types), pearson_vals, color=colors, height=0.7,
            edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(thin_labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(median_r, color="#555", linestyle="--", linewidth=1.2,
               label=f"Median = {median_r:.3f}")
    ax.axvline(0.8, color=COLORS["good"], linestyle=":", linewidth=1.0,
               alpha=0.5, label="r = 0.8")
    ax.legend(fontsize=FONT_ANNOTATION, frameon=False, loc="lower right")
    style_axes(ax, "bar", xlabel="Pseudobulk Pearson r",
               title="Per-Type Pseudobulk Correlation")

    # Panel (b): Summary statistics box
    ax2 = right.inset(left=0.05, right=0.02).add_axes(fig)
    add_panel_label(ax2, 'b', x=-0.12, y=1.10)

    # Histogram of Pearson r values
    ax2.hist(pearson_vals, bins=20, color=COLORS["real"], alpha=0.7,
             edgecolor="white", linewidth=0.5)
    ax2.axvline(median_r, color=COLORS["generated"], linestyle="--", linewidth=2,
                label=f"Median = {median_r:.3f}")
    ax2.axvline(np.mean(pearson_vals), color=COLORS["accent"], linestyle="-.",
                linewidth=1.5, label=f"Mean = {np.mean(pearson_vals):.3f}")

    # Disable scientific notation on x-axis (values close to 1.0)
    ax2.xaxis.get_major_formatter().set_useOffset(False)
    ax2.ticklabel_format(axis='x', useOffset=False, style='plain')
    # Rotate x-tick labels to avoid overlap (values are very close together near 1.0)
    ax2.tick_params(axis='x', rotation=40)
    for label in ax2.get_xticklabels():
        label.set_ha('right')
        label.set_fontsize(7)

    # Summary text — positioned to avoid overlap with legend
    s = summary
    text = (f"n = {s['n_types_evaluated']} types\n"
            f"r > 0.9: {s['pct_pearson_gt_0.9']:.0f}%\n"
            f"r > 0.8: {s['pct_pearson_gt_0.8']:.0f}%")
    ax2.text(0.97, 0.60, text, transform=ax2.transAxes,
             ha="right", va="top", fontsize=FONT_SMALL,
             color=COLORS["neutral"])

    ax2.legend(fontsize=FONT_ANNOTATION, frameon=False, loc="upper left")
    style_axes(ax2, "default", xlabel="Pearson r", ylabel="Count",
               title="Distribution of Pseudobulk Correlations")

    fig_path = output_dir / "figS02a_pseudobulk_validation"
    save_with_vcd(fig, fig_path, dpi=dpi, layout_rect=(0.08, 0.06, 0.98, 0.94))
    plt.close()
    logger.info(f"Saved: {fig_path}")
    return fig_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_pseudobulk_validation()
