"""
fig21_field_ablation.py — Prompt field disentanglement figure.

Panels (sequential within composed Figure A2):
  (f) KNN accuracy by variant (horizontal bar chart)
  (g) KNN retention percentage relative to full prompt
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
    FONT_LABEL, FONT_TICK, FONT_TITLE, FONT_ANNOTATION,
)
from src.utils.paths import RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)

VARIANT_DISPLAY = {
    "full": "Full prompt",
    "markers_only": "Markers only",
    "celltype_only": "Cell type only",
    "tissue_only": "Tissue only",
    "organism_only": "Organism only",
    "disease_only": "Disease only",
    "no_markers": "No markers",
    "no_celltype": "No cell type",
}


def plot_field_ablation(output_dir=None, dpi=300):
    """Generate prompt field ablation figure."""
    apply_style()
    output_dir = Path(output_dir or FIG_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_dir = RESULTS_DIR / "validation"
    results_path = val_dir / "field_ablation_results.json"

    if not results_path.exists():
        logger.warning("Field ablation results not found. "
                       "Run: python scripts/analysis/prompt_field_ablation.py")
        return None

    with open(results_path) as f:
        results = json.load(f)

    # Variant ordering (full first, then single-field, then leave-one-out)
    order = ["full", "markers_only", "celltype_only", "no_markers", "no_celltype",
             "tissue_only", "organism_only", "disease_only"]
    present_variants = [v for v in order if v in results]

    labels = [VARIANT_DISPLAY.get(v, v) for v in present_variants]
    knn_vals = [results[v]["knn_accuracy"] for v in present_variants]
    retention = [results[v].get("knn_retention_pct", 100.0) for v in present_variants]

    fig = plt.figure(figsize=(5.0, 3.0))
    layout = bind_figure_region(fig, (0.18, 0.18, 0.94, 0.88))
    left, right = layout.split_cols([1.0, 1.0], wspace=0.28)

    # Panel (f): KNN accuracy bars
    ax = left.inset(right=0.02).add_axes(fig)
    add_panel_label(ax, 'f', x=-0.12, y=1.10)

    y_pos = np.arange(len(labels))
    colors = []
    for v in present_variants:
        if v == "full":
            colors.append(COLORS["real"])
        elif "only" in v:
            colors.append(COLORS["generated"])
        else:
            colors.append(COLORS["accent"])

    bars = ax.barh(y_pos, knn_vals, color=colors, height=0.6,
                   edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()

    # Value annotations
    for i, (bar, val) in enumerate(zip(bars, knn_vals)):
        ax.text(val + 0.005, i, f"{val:.3f}", va="center",
                fontsize=FONT_ANNOTATION, color=COLORS["neutral"])

    style_axes(ax, "bar", xlabel="KNN Accuracy",
               title="Steering by Variant")

    # Panel (g): Retention percentage
    ax2 = right.inset(left=0.05, right=0.02).add_axes(fig)
    add_panel_label(ax2, 'g', x=-0.12, y=1.10)

    # Skip "full" for retention chart (it's always 100%)
    ret_variants = [v for v in present_variants if v != "full"]
    ret_labels = [VARIANT_DISPLAY.get(v, v) for v in ret_variants]
    ret_vals = [results[v].get("knn_retention_pct", 0) for v in ret_variants]

    ret_y = np.arange(len(ret_labels))
    ret_colors = [COLORS["good"] if r >= 80 else (COLORS["warn"] if r >= 50 else COLORS["bad"])
                  for r in ret_vals]

    bars2 = ax2.barh(ret_y, ret_vals, color=ret_colors, height=0.6,
                     edgecolor="white", linewidth=0.3)
    ax2.set_yticks(ret_y)
    ax2.set_yticklabels(ret_labels, fontsize=11)
    ax2.invert_yaxis()
    ax2.axvline(100, color="#999", linestyle=":", linewidth=1.0, alpha=0.5)

    for i, (bar, val) in enumerate(zip(bars2, ret_vals)):
        ax2.text(val + 1.0, i, f"{val:.1f}%", va="center",
                 fontsize=FONT_ANNOTATION, color=COLORS["neutral"])

    style_axes(ax2, "bar", xlabel="Retention (%)",
               title="Field Contribution")

    fig_path = output_dir / "figS02c_field_ablation"
    save_with_vcd(fig, fig_path, dpi=dpi, layout_rect=(0.12, 0.08, 0.97, 0.93))
    plt.close()
    logger.info(f"Saved: {fig_path}")
    return fig_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_field_ablation()
