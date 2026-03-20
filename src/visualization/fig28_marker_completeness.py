"""
fig28_marker_completeness.py — Fig 28: Marker gene program completeness.

Shows whether generated cells recapitulate known marker programs for
representative cell types.  Reads results from the marker completeness
experiment and renders precision/recall curves and marker heatmaps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from .direct_layout import bind_figure_region
from .explicit_positioning import add_axes_next_to
from .style import (
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
from src.utils.paths import RESULTS_DIR

logger = logging.getLogger(__name__)


def plot_marker_completeness(
    data_path: str | Path = "results/downstream/marker_completeness.json",
    output_dir: str | Path = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Fig 28: Marker gene program completeness.

    Panel (a): Recall@K curves for canonical markers across cell types.
    Panel (b): Heatmap — per-type canonical marker recovery (top-50 / top-100).
    Panel (c): Bar chart — mean log2FC of canonical markers vs background.
    """
    apply_style()
    data_path = Path(data_path)
    output_dir = Path(output_dir)

    if not data_path.exists():
        logger.warning("Marker completeness not found: %s", data_path)
        return None

    with open(data_path) as f:
        data = json.load(f)

    cell_types = list(data.keys())
    if not cell_types:
        logger.warning("No cell types in marker completeness data")
        return None

    # Layout
    fig = plt.figure(figsize=(16.5, 8.0))
    layout = bind_figure_region(fig, (0.07, 0.18, 0.97, 0.92))
    p_a, p_b, p_c = layout.split_cols([1.0, 1.2, 0.7], gap=0.07)

    # Discover available K values dynamically from data
    _candidate_ks = [10, 20, 50, 100]
    _first_canonical = data[cell_types[0]].get("canonical", {})
    ks = [k for k in _candidate_ks if f"recall@{k}" in _first_canonical]
    if not ks:
        ks = _candidate_ks  # fallback
    ct_short = [abbreviate_cell_type(ct, max_len=12) for ct in cell_types]

    # Generate distinct colors for cell types
    import matplotlib
    _n_ct = max(len(cell_types), 1)
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(_n_ct)
    ct_colors = [cmap(i) for i in range(_n_ct)]

    # ── Panel (a): Recall@K curves ──
    ax_a = p_a.add_axes(fig)
    add_panel_label(ax_a, "a", x=-0.12, y=1.06)

    for i, ct in enumerate(cell_types):
        ct_data = data[ct].get("canonical", {})
        recalls = [ct_data.get(f"recall@{k}", 0) for k in ks]
        ax_a.plot(ks, recalls, marker="o", markersize=4, linewidth=1.5,
                  color=ct_colors[i], label=ct_short[i], alpha=0.8)

    ax_a.set_xlabel("K (top-K genes)", fontsize=FONT_LABEL)
    ax_a.set_ylabel("Recall", fontsize=FONT_LABEL)
    ax_a.set_title("Canonical Marker Recall@K", fontsize=FONT_TITLE)
    ax_a.set_xticks(ks)
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.set_xlim(5, 110)

    # Legend below the plot to avoid masking data lines
    ax_a.legend(fontsize=FONT_LEGEND - 3, loc="upper center",
                bbox_to_anchor=(0.5, -0.15), frameon=False,
                ncol=5, columnspacing=0.5, handlelength=1.0)
    style_axes(ax_a)

    # ── Panel (b): Recovery heatmap ──
    ax_b = p_b.add_axes(fig)
    add_panel_label(ax_b, "b", x=-0.10, y=1.06)

    # Build matrix: rows=cell types, cols=recall@K for canonical + extended
    col_labels = [f"C@{k}" for k in ks] + [f"E@{k}" for k in ks]
    n_cols = len(col_labels)
    heatmap = np.full((len(cell_types), n_cols), np.nan)

    for i, ct in enumerate(cell_types):
        for j, k in enumerate(ks):
            heatmap[i, j] = data[ct].get("canonical", {}).get(f"recall@{k}", np.nan)
            heatmap[i, j + len(ks)] = data[ct].get("extended", {}).get(f"recall@{k}", np.nan)

    im = ax_b.imshow(heatmap, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    ax_b.set_xticks(range(n_cols))
    ax_b.set_xticklabels(col_labels, fontsize=FONT_TICK_DENSE - 1, rotation=40, ha="right")
    ax_b.set_yticks(range(len(cell_types)))
    ax_b.set_yticklabels(ct_short, fontsize=FONT_TICK_DENSE)
    ax_b.set_title("Marker Recovery (Canonical + Extended)", fontsize=FONT_TITLE)

    # Add separator line between canonical and extended
    ax_b.axvline(len(ks) - 0.5, color="white", lw=2)

    for ri in range(heatmap.shape[0]):
        for ci in range(heatmap.shape[1]):
            val = heatmap[ri, ci]
            if np.isfinite(val):
                color = "white" if val > 0.5 else "black"
                ax_b.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                          fontsize=max(FONT_HEATMAP_CELL - 1, 5), color=color)

    cax = add_axes_next_to(fig, ax_b, side="right", width=0.008,
                           height=ax_b.get_position().height * 0.6,
                           pad=0.010, align="center")
    fig.colorbar(im, cax=cax, label="Recall")
    cax.tick_params(labelsize=7)

    # ── Panel (c): Mean logFC of canonical markers ──
    ax_c = p_c.add_axes(fig)
    add_panel_label(ax_c, "c", x=-0.18, y=1.06)

    mean_lfcs = []
    for ct in cell_types:
        canonical = data[ct].get("canonical", {})
        marker_details = canonical.get("marker_details", {})
        lfcs = [v.get("log2fc_vs_background", 0) for v in marker_details.values()
                if isinstance(v, dict) and "log2fc_vs_background" in v]
        mean_lfcs.append(np.mean(lfcs) if lfcs else 0)

    y_pos = np.arange(len(cell_types))
    bar_colors = [COLORS["good"] if lfc > 0 else COLORS["bad"] for lfc in mean_lfcs]
    ax_c.barh(y_pos, mean_lfcs, height=0.6, color=bar_colors,
              alpha=0.85, edgecolor="white", linewidth=0.5)

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(ct_short, fontsize=FONT_TICK_DENSE)
    ax_c.set_xlabel("Mean log2FC vs bg", fontsize=FONT_LABEL)
    ax_c.set_title("Marker Enrichment", fontsize=FONT_TITLE)
    ax_c.axvline(0, color="grey", lw=0.5, alpha=0.5)
    ax_c.invert_yaxis()
    style_axes(ax_c)

    # Auto-scale x-axis to data range (values are ~1e-4 to 7e-4, invisible at normal scale)
    if mean_lfcs:
        lfc_min = min(mean_lfcs)
        lfc_max = max(mean_lfcs)
        lfc_pad = (lfc_max - lfc_min) * 0.15 if lfc_max != lfc_min else abs(lfc_max) * 0.2
        ax_c.set_xlim(lfc_min - lfc_pad, lfc_max + lfc_pad)

    # Use scientific notation for tiny log2FC values
    ax_c.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))

    for i, v in enumerate(mean_lfcs):
        ha = "left" if v >= 0 else "right"
        offset_pts = 4 if v >= 0 else -4
        ax_c.annotate(f"{v:.4f}", xy=(v, i), xytext=(offset_pts, 0),
                      textcoords="offset points", ha=ha, va="center",
                      fontsize=FONT_ANNOTATION - 1, annotation_clip=True)

    ax_c.text(0.98, 0.02,
              "Effect sizes < 0.001 log2FC\n(scGPT decoder near-uniform)",
              transform=ax_c.transAxes, ha="right", va="bottom",
              fontsize=FONT_ANNOTATION - 1, style="italic",
              color=COLORS["neutral"])

    # Save
    if save:
        if save_panel_fn:
            save_panel_fn(fig, "fig28_marker_completeness")
        else:
            out = output_dir / "fig28_marker_completeness.png"
            save_panel(fig, out, dpi=dpi)
        logger.info("Saved Fig 28 → %s", output_dir)

    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_marker_completeness()
