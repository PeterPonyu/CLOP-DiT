"""
fig14_expr_diversity.py — Article Figure 14: expression diversity.

Extracted from panels_diversity.py (formerly Panel K / plot_expression_diversity_panel).
Plotting only; scripts run diagnostics and pass pre-computed data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .direct_layout import bind_figure_region
from .style import COLORS, apply_style, save_with_vcd, add_panel_label

logger = logging.getLogger(__name__)


def plot_expression_diversity_panel(
    t6_data: Dict,
    output_dir: str = "results/figures",
    dpi: int = 300,
    label_offset: int = 0,
    save: bool = True,
) -> "Optional[plt.Figure]":
    """Panel K / Figure 14: Expression diversity (standalone, supports label_offset for merged figures).

    Parameters
    ----------
    t6_data : dict
        The ``test6_expression_diversity`` section from diversity_diagnostics.json.
    output_dir : str
        Directory for saved figures.
    dpi : int
        Resolution for raster output.
    label_offset : int
        Offset for panel labels (0 -> 'a','b'; 1 -> 'b','c'; etc.)
    save : bool
        Whether to persist the figure to disk.

    Returns
    -------
    fig or None
    """

    if not t6_data or not t6_data.get("overall"):
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()

    o = t6_data["overall"]
    fig = plt.figure(figsize=(7.6, 4.9))
    left_rect, right_rect = bind_figure_region(fig, (0.08, 0.16, 0.98, 0.90)).split_cols(2, wspace=0.38)
    axes = [left_rect.add_axes(fig), right_rect.add_axes(fig)]
    add_panel_label(axes[0], chr(ord('a') + label_offset), x=-0.04, y=1.05)
    add_panel_label(axes[1], chr(ord('a') + label_offset + 1), x=-0.04, y=1.05)

    ax = axes[0]
    labels = ["Cell Std\n(across genes)", "Gene Std\n(across cells)"]
    real_vals = [o["real_mean_cell_std"], o["real_mean_gene_std"]]
    gen_vals = [o["gen_mean_cell_std"], o["gen_mean_gene_std"]]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, real_vals, w, label="Real", color=COLORS["real"], alpha=0.8)
    ax.bar(x + w / 2, gen_vals, w, label="Generated", color=COLORS["generated"], alpha=0.8)
    # Add gen/real ratio annotations above each bar pair
    annotation_tops = []
    for _bi in range(len(real_vals)):
        if real_vals[_bi] > 0:
            ratio = gen_vals[_bi] / real_vals[_bi]
            max_h = max(real_vals[_bi], gen_vals[_bi])
            if ratio >= 1.03:
                y_offset = 1.12
            elif ratio <= 0.97:
                y_offset = 1.07
            else:
                y_offset = 1.02
            ann_y = max_h * y_offset
            annotation_tops.append(ann_y)
            ax.text(_bi, ann_y + 0.02, f"ratio={ratio:.2f}",
                    ha="center", fontsize=8, fontweight="normal",
                    color=COLORS["annotation_dark"])
    if annotation_tops:
        ax.set_ylim(0, max(max(real_vals + gen_vals) * 1.08, max(annotation_tops) * 1.10))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Expression Variability Summary", x=0.58)
    ax.legend(frameon=False)

    ax = axes[1]
    pt_ratio = t6_data.get("per_type_gene_std_ratio", {})
    if pt_ratio:
        vals = [pt_ratio["min"], pt_ratio["mean"], pt_ratio["max"]]
        lbls = ["Min", "Mean", "Max"]
        colors = [COLORS["bad"] if v < 0.5 else COLORS["good"] for v in vals]
        ax.bar(lbls, vals, color=colors, alpha=0.8, edgecolor="white")
        ax.axhline(y=1.0, color="black", ls="--", lw=1, alpha=0.5,
                   label="ratio=1 (equal diversity)")
        ax.set_ylabel("Gene Std Ratio (gen / real)")
        ax.set_title("Per-Type Gene Std Ratio")
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "No per-type data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=COLORS["neutral"])
        ax.set_title("Per-Type Gene Std Ratio")

    if save:
        path = out / "fig14_expression_diversity.png"
        save_with_vcd(fig, path, dpi)
        logger.info(f"Saved Fig 14 \u2192 {path}")

    return fig
