"""Fig 22: Multi-seed robustness — mean ± std across seeds per regime.

Reads ``results/multi_seed/multi_seed_report.json`` and renders grouped
bar charts with error bars for each metric across the high-fidelity and
high-diversity regimes.
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

# Metrics to plot (key → display label)
_METRICS = [
    ("knn_top1", "kNN Top-1"),
    ("knn_top5", "kNN Top-5"),
    ("steering_accuracy", "Steering Acc"),
    ("diversity_ratio", "Diversity Ratio"),
    ("linear_accuracy", "Linear Acc"),
    ("centroid_cosine", "Centroid Cos"),
    ("frechet_distance", "FD ↓"),
]

_REGIME_COLORS = {
    "high_fidelity_regime": COLORS["real"],
    "high_diversity_regime": COLORS["generated"],
}

_REGIME_LABELS = {
    "high_fidelity_regime": "High Fidelity",
    "high_diversity_regime": "High Diversity",
}


def plot_multi_seed_robustness(
    report_path: str | Path = "results/multi_seed/multi_seed_report.json",
    output_dir: str | Path = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Render multi-seed robustness figure.

    Parameters
    ----------
    report_path : path to multi_seed_report.json
    output_dir : figure output directory
    dpi : export resolution
    save : whether to save figure
    save_panel_fn : optional callback for custom saving

    Returns
    -------
    fig : Figure or None on error
    """
    apply_style()
    report_path = Path(report_path)
    output_dir = Path(output_dir)

    if not report_path.exists():
        logger.warning("Multi-seed report not found: %s", report_path)
        return None

    with open(report_path) as f:
        report = json.load(f)

    regimes = [k for k in report if k in _REGIME_LABELS]
    if not regimes:
        logger.warning("No recognized regimes in multi-seed report")
        return None

    # Figure: grouped bar chart
    fig = plt.figure(figsize=(12.0, 5.5))
    layout = bind_figure_region(fig, (0.08, 0.18, 0.96, 0.88))

    # Panel (a): grouped bars
    ax = layout.add_axes(fig)

    n_metrics = len(_METRICS)
    n_regimes = len(regimes)
    bar_width = 0.35
    x = np.arange(n_metrics)

    for idx, regime in enumerate(regimes):
        agg = report[regime].get("aggregated", {})
        means = []
        stds = []
        for key, _label in _METRICS:
            m = agg.get(key, {})
            means.append(m.get("mean", 0))
            stds.append(m.get("std", 0))

        offset = (idx - (n_regimes - 1) / 2) * bar_width
        color = _REGIME_COLORS.get(regime, COLORS["neutral"])
        label = _REGIME_LABELS.get(regime, regime)
        ax.bar(x + offset, means, bar_width, yerr=stds, label=label,
               color=color, alpha=0.85, capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in _METRICS], rotation=25, ha="right",
                       fontsize=FONT_LABEL - 1)
    ax.set_ylabel("Score", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_LEGEND_DENSE, loc="upper right")

    # Annotate number of seeds
    first_regime = regimes[0]
    n_seeds = report[first_regime].get("aggregated", {}).get("n_seeds", "?")
    ax.text(0.98, 0.02, f"n = {n_seeds} seeds",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FONT_LEGEND_DENSE, color=COLORS["neutral"])

    add_panel_label(ax, "a", x=-0.06, y=1.06)
    set_figure_suptitle(fig, "Multi-Seed Generation Robustness", y=0.96)

    if save:
        out = output_dir / "fig22_multi_seed_robustness.png"
        if save_panel_fn:
            save_panel_fn(fig, "fig22_multi_seed_robustness")
        else:
            save_with_vcd(fig, out, dpi=dpi)

    return fig
