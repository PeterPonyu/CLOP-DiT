"""
fig22_discriminator.py — Discriminator feature importance figure.

Panels (sequential within composed Figure A2):
  (h) Coefficient magnitude vs mean shift correlation
  (i) Coefficient magnitude vs variance shift correlation
  (j) Per-type discriminator AUC (sorted bar chart)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..direct_layout import bind_figure_region
from ..style import (
    COLORS, add_panel_label, apply_style, save_with_vcd, style_axes,
    abbreviate_cell_type, FONT_LABEL, FONT_TITLE, FONT_ANNOTATION, FONT_SMALL,
)
from src.utils.paths import RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)


def plot_discriminator_analysis(output_dir=None, dpi=300):
    """Generate discriminator feature importance figure."""
    apply_style()
    output_dir = Path(output_dir or FIG_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_dir = RESULTS_DIR / "validation"
    results_path = val_dir / "discriminator_analysis.json"

    if not results_path.exists():
        logger.warning("Discriminator analysis results not found. "
                       "Run: python scripts/analysis/discriminator_importance.py")
        return None

    with open(results_path) as f:
        results = json.load(f)

    per_type = results.get("per_type", {})
    global_disc = results.get("global_discriminator", {})
    mv = results.get("mean_vs_variance", {})
    coef = results.get("coefficient_analysis", {})

    fig = plt.figure(figsize=(5.0, 3.2))
    layout = bind_figure_region(fig, (0.12, 0.18, 0.96, 0.86))
    left, mid, right = layout.split_cols([0.8, 0.8, 1.4], wspace=0.55)

    # Panel (h): Separability driver summary (mean vs variance)
    ax = left.inset(right=0.02).add_axes(fig)
    add_panel_label(ax, 'h', x=-0.12, y=1.12)

    # Bar chart: mean vs variance contribution
    labels_bar = ["Mean", "Variance"]
    r_mean = abs(mv.get("coef_vs_mean_diff", {}).get("pearson_r", 0))
    r_var = abs(mv.get("coef_vs_var_diff", {}).get("pearson_r", 0))
    vals = [r_mean, r_var]
    bar_colors = [COLORS["real"], COLORS["generated"]]

    bars = ax.bar(labels_bar, vals, color=bar_colors, width=0.5,
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"r={val:.3f}", ha="center", fontsize=FONT_ANNOTATION,
                color=COLORS["neutral"])

    ax.set_ylim(0, max(vals) * 1.3)
    driver = "Mean shift" if r_mean > r_var else "Variance shift"
    ax.text(0.5, 0.95, f"Primary driver: {driver}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, fontweight="bold", color=COLORS["annotation_dark"])
    ax.set_xticks(range(len(labels_bar)))
    ax.set_xticklabels(labels_bar, rotation=0, ha="center")
    style_axes(ax, "bar", ylabel="|Pearson r|",
               title="Separability Driver")

    # Panel (i): Coefficient concentration
    ax2 = mid.inset(left=0.05, right=0.02).add_axes(fig)
    add_panel_label(ax2, 'i', x=-0.12, y=1.12)

    conc = coef.get("concentration", {})
    x_labels = ["10", "20", "50"]
    x_vals = [conc.get("top10_fraction", 0), conc.get("top20_fraction", 0),
              conc.get("top50_fraction", 0)]

    bars2 = ax2.bar(x_labels, [v * 100 for v in x_vals], color=COLORS["accent"],
                    width=0.5, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars2, x_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.0%}", ha="center", fontsize=FONT_ANNOTATION,
                 color=COLORS["neutral"])

    gini = conc.get("gini", 0)
    ax2.text(0.97, 0.97, f"Gini = {gini:.3f}",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=11, color=COLORS["neutral"])

    ax2.set_ylim(0, 100)
    ax2.set_xticks(range(len(x_labels)))
    ax2.set_xticklabels(x_labels, rotation=0, ha="center")
    style_axes(ax2, "bar", xlabel="Top-K dims", ylabel="% discrim. wt.",
               title="Dim. Concentration")

    # Panel (j): Per-type AUC bar chart
    ax3 = right.inset(left=0.05, right=0.01).add_axes(fig)
    add_panel_label(ax3, 'j', x=-0.10, y=1.12)

    sorted_types = sorted(per_type.items(), key=lambda x: x[1]["auc"], reverse=True)
    type_names = [abbreviate_cell_type(n, max_len=10) for n, _ in sorted_types]
    auc_vals = [v["auc"] for _, v in sorted_types]

    n_types = len(type_names)
    step = max(1, n_types // 8)
    thin = [type_names[i] if i % step == 0 or i == n_types - 1 else ""
            for i in range(n_types)]

    pt_colors = [COLORS["bad"] if a > 0.7 else (COLORS["warn"] if a > 0.6 else COLORS["good"])
                 for a in auc_vals]

    ax3.barh(range(n_types), auc_vals, color=pt_colors, height=0.7,
             edgecolor="white", linewidth=0.3)
    ax3.set_yticks(range(n_types))
    ax3.set_yticklabels(thin, fontsize=10)
    ax3.invert_yaxis()
    ax3.axvline(0.5, color="#999", linestyle=":", linewidth=1.0, alpha=0.5,
                label="Chance (0.5)")
    auc_mean = np.mean(auc_vals)
    ax3.axvline(auc_mean, color="#555", linestyle="--", linewidth=1.2,
                label=f"Mean = {auc_mean:.3f}")
    ax3.legend(fontsize=FONT_SMALL, frameon=False, loc="upper right")
    style_axes(ax3, "bar", xlabel="Discriminator AUC",
               title="Per-Type Separability")

    fig_path = output_dir / "figS02d_discriminator_analysis"
    save_with_vcd(fig, fig_path, dpi=dpi, layout_rect=(0.10, 0.08, 0.96, 0.92))
    plt.close()
    logger.info(f"Saved: {fig_path}")
    return fig_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_discriminator_analysis()
