"""Fig 21: Ablation heatmap — comparison of CLOP ablation variants.

Reads ``results/ablations/all_summaries.json`` and renders a category-grouped
heatmap of key metrics (proto accuracy, top-5, temperature, loss) for each
ablation variant.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .direct_layout import bind_figure_region
from .style import (
    COLORS,
    FONT_LABEL,
    FONT_LEGEND_DENSE,
    FONT_TITLE,
    add_panel_label,
    apply_style,
    save_with_vcd,
    set_figure_suptitle,
)

logger = logging.getLogger(__name__)

# Metrics to display in the heatmap columns
_METRIC_COLS = [
    ("best_val_proto_acc", "Proto Acc"),
    ("best_val_proto_top5", "Top-5 Acc"),
    ("best_val_proto_top10", "Top-10 Acc"),
    ("best_val_loss", "Val Loss"),
    ("best_epoch", "Best Epoch"),
]

# Category display order and colours
_CATEGORY_COLORS = {
    "architecture": "#1565C0",
    "loss": "#C62828",
    "regularization": "#2E7D32",
    "temperature": "#6A1B9A",
    "training": "#E65100",
    "other": COLORS["neutral"],
}


def plot_ablation_heatmap(
    ablation_path: str | Path = "results/ablations/all_summaries.json",
    output_dir: str | Path = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Render ablation heatmap from cached summaries.

    Parameters
    ----------
    ablation_path : path to all_summaries.json
    output_dir : figure output directory
    dpi : export resolution
    save : whether to save figure
    save_panel_fn : optional callback for custom saving

    Returns
    -------
    fig : Figure or None on error
    """
    apply_style()
    ablation_path = Path(ablation_path)
    output_dir = Path(output_dir)

    if not ablation_path.exists():
        logger.warning("Ablation summaries not found: %s", ablation_path)
        return None

    with open(ablation_path) as f:
        summaries = json.load(f)

    if not summaries:
        logger.warning("No ablation entries found")
        return None

    # Sort by category then name
    summaries.sort(key=lambda s: (s.get("category", "other"), s.get("ablation_name", "")))

    # Extract data
    names = [s["ablation_name"].replace("_", " ") for s in summaries]
    categories = [s.get("category", "other") for s in summaries]
    n_variants = len(summaries)
    n_metrics = len(_METRIC_COLS)

    # Build matrix
    data = np.full((n_variants, n_metrics), np.nan)
    for i, s in enumerate(summaries):
        metrics = s.get("metrics", {})
        for j, (key, _label) in enumerate(_METRIC_COLS):
            val = metrics.get(key)
            if val is not None:
                data[i, j] = float(val)

    # Normalize columns to [0, 1] for heatmap (loss is inverted)
    norm_data = np.zeros_like(data)
    for j in range(n_metrics):
        col = data[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            norm_data[:, j] = 0.5
            continue
        vmin, vmax = col[valid].min(), col[valid].max()
        if vmax - vmin < 1e-12:
            norm_data[:, j] = 0.5
        else:
            norm_data[:, j] = (col - vmin) / (vmax - vmin)
        # Invert loss columns (lower is better)
        if "loss" in _METRIC_COLS[j][0].lower():
            norm_data[:, j] = 1.0 - norm_data[:, j]

    # Create figure
    fig_height = max(5.0, 0.35 * n_variants + 1.5)
    fig = plt.figure(figsize=(8.0, fig_height))
    layout = bind_figure_region(fig, (0.22, 0.10, 0.92, 0.90))
    region = layout
    ax = region.add_axes(fig)

    im = ax.imshow(norm_data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    # Axis labels
    metric_labels = [label for _, label in _METRIC_COLS]
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_labels, rotation=30, ha="right", fontsize=FONT_LABEL - 1)
    ax.set_yticks(range(n_variants))
    ax.set_yticklabels(names, fontsize=FONT_LABEL - 2)

    # Annotate cells with raw values
    for i in range(n_variants):
        for j in range(n_metrics):
            val = data[i, j]
            if np.isnan(val):
                continue
            color = "white" if norm_data[i, j] < 0.3 or norm_data[i, j] > 0.7 else "black"
            fmt = ".3f" if val < 10 else ".1f"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                    fontsize=FONT_LEGEND_DENSE - 1, color=color)

    # Category color bar on left
    for i, cat in enumerate(categories):
        color = _CATEGORY_COLORS.get(cat, COLORS["neutral"])
        ax.plot(-0.7, i, "s", color=color, markersize=6, clip_on=False,
                transform=ax.transData)

    # Legend for categories
    seen = {}
    for cat in categories:
        if cat not in seen:
            seen[cat] = _CATEGORY_COLORS.get(cat, COLORS["neutral"])
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                   markersize=7, label=cat.capitalize())
        for cat, c in seen.items()
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0, 0.01),
              ncol=min(len(seen), 3), fontsize=FONT_LEGEND_DENSE, frameon=False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Normalized score (higher = better)", fontsize=FONT_LABEL - 1)

    add_panel_label(ax, "a", x=-0.18, y=1.04)
    set_figure_suptitle(fig, "CLOP Ablation Comparison", y=0.96)

    if save:
        out = output_dir / "fig21_ablation_heatmap.png"
        if save_panel_fn:
            save_panel_fn(fig, "fig21_ablation_heatmap")
        else:
            save_with_vcd(fig, out, dpi=dpi)

    return fig
