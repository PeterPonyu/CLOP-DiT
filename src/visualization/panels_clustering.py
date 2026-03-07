"""
panels_clustering.py — Panel P: Clustering alignment and real-generated mixing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .style import (
    add_panel_label,
    set_figure_suptitle,
    TYPE_PALETTE,
    quality_color,
    save_panel,
    set_dense_tick_labels,
    style_axes,
)
logger = logging.getLogger(__name__)


def plot_clustering_panel(
    clustering_data: Dict,
    type_names: Dict[int, str],
    output_dir: Path,
    dpi: int = 300,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Panel P: Clustering alignment and real-generated mixing.

    P1: UMAP of real+gen coloured by cell type (generated outlined)
    P2: Per-type kNN mixing score bars
    P3: ARI/NMI/cluster-purity gauges + gen cluster accuracy
    """
    umap_coords = clustering_data.get("_umap_coords")
    source = clustering_data.get("_source")
    cell_type = clustering_data.get("_cell_type")

    if umap_coords is None:
        logger.info("No clustering UMAP data — skipping Panel P")
        return None

    umap_coords = np.asarray(umap_coords)
    source = np.asarray(source)
    cell_type = np.asarray(cell_type)

    fig = plt.figure(figsize=(13.0, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 0.8],
                          wspace=0.45)
    # suptitle removed per revision; title information moved to LaTeX caption

    ax = fig.add_subplot(gs[0])
    add_panel_label(ax, 'a')
    unique_types = np.unique(cell_type)
    ct_colors = {}
    for i, ct in enumerate(sorted(unique_types)):
        ct_colors[ct] = TYPE_PALETTE[i % len(TYPE_PALETTE)]

    real_mask = source == "real"
    gen_mask = source == "generated"
    for ct in unique_types:
        mask = real_mask & (cell_type == ct)
        if mask.any():
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[ct_colors[ct]], s=4, alpha=0.3, rasterized=True)
    for ct in unique_types:
        mask = gen_mask & (cell_type == ct)
        if mask.any():
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[ct_colors[ct]], s=12, alpha=0.6, marker="^",
                       edgecolors="black", linewidths=0.3, rasterized=True)

    ax.scatter([], [], c="gray", s=15, marker="o", label="Real")
    ax.scatter([], [], c="gray", s=15, marker="^", edgecolors="black",
               linewidths=0.3, label="Generated")
    ax.legend(fontsize=9, loc="upper left", markerscale=2)
    style_axes(ax, "umap", title="UMAP Overlay",
               xlabel="UMAP 1", ylabel="UMAP 2")

    ax2 = fig.add_subplot(gs[1])
    add_panel_label(ax2, 'b')
    if mixing:
        sorted_types = sorted(mixing.keys(), key=lambda k: mixing[k])
        vals = [mixing[t] for t in sorted_types]
        short_names = [t[:25] for t in sorted_types]
        bar_colors = [quality_color(v, (0.3, 0.15)) for v in vals]

        y_pos = np.arange(len(sorted_types))
        ax2.barh(y_pos, vals, color=bar_colors, height=0.7, edgecolor="white", linewidth=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(short_names, fontsize=8)
        set_dense_tick_labels(ax2, axis="y", max_labels=18, fontsize=8, rotation=0)
        ax2.axvline(x=clustering_data.get("mean_mixing_score", 0),
                     color="#D32F2F", linestyle="--", alpha=0.7, linewidth=1.5,
                     label=f"mean={clustering_data.get('mean_mixing_score', 0):.3f}")
        ax2.set_xlim(0, max(max(vals) * 1.1, 0.5))
        ax2.legend(fontsize=9, loc="upper left", frameon=False)
        style_axes(ax2, "bar", title="kNN Mixing",
                   xlabel="Fraction Real Neighbours")
    else:
        ax2.text(0.5, 0.5, "No mixing data", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=10)
        ax2.set_title("kNN Mixing Score")

    ax3 = fig.add_subplot(gs[2])
    add_panel_label(ax3, 'c')
    ax3.axis("off")

    gauge_items = [
        ("ARI", clustering_data.get("ari_gt_vs_leiden", 0), (0.6, 0.3)),
        ("NMI", clustering_data.get("nmi_gt_vs_leiden", 0), (0.6, 0.3)),
        ("C.Purity", clustering_data.get("mean_cluster_purity", 0), (0.8, 0.5)),
        ("Mean Mix", clustering_data.get("mean_mixing_score", 0), (0.3, 0.15)),
    ]

    n_items = len(gauge_items)
    for i, (label, val, thresh) in enumerate(gauge_items):
        y = 0.90 - i * (0.82 / max(n_items - 1, 1))
        color = quality_color(val, thresh)
        ax3.text(0.55, y, f"{val:.3f}", fontsize=9,
                 color="#222222", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=color,
                           alpha=1.0, linewidth=2.0),
                 transform=ax3.transAxes)
        ax3.text(0.1, y, label, fontsize=9, ha="left", va="center",
                 transform=ax3.transAxes, color="#333")

    ax3.set_title("Alignment Gauges", fontsize=11, pad=8)
    n_clusters = clustering_data.get("n_leiden_clusters", "?")
    ax3.text(0.5, 0.02, f"Leiden clusters: {n_clusters}",
             transform=ax3.transAxes, ha="center", fontsize=8, color="#666")

    if save:
        path = save_panel(fig, output_dir / "panel_p_clustering_mixing.png", dpi)
        logger.info(f"Saved Panel P → {path}")
    return fig
