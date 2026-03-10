"""
fig16_benchmark.py — Fig 16: Comprehensive model benchmarking visualization.

  S1: Metrics heatmap (methods x metrics, colour-coded)
  S2: Composite score bar chart with rank badges
  S3: Key metrics comparison — grouped bar chart
  S4: FD & Centroid Cosine CI comparison (error-bar plot)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .explicit_positioning import add_axes_next_to, add_shared_legend_axes
from .panel_geometry import apply_layout_rect
from .style import (
    COLORS, FONT_SMALL, FONT_ANNOTATION, METHOD_COLORS, abbreviate_cell_type,
    add_panel_label, save_panel, save_with_vcd,
    set_figure_suptitle, style_axes
)
from src.utils.paths import RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)


def plot_benchmark_panel(
    report_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    dpi: int = 300,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Fig 16: Comprehensive model benchmarking dashboard (2x2).

    S1: Metrics heatmap — methods x metrics, cell-coloured by normalised value
    S2: Composite score — horizontal bars with score labels
    S3: Key metrics comparison — grouped bar chart
    S4: Confidence interval comparison — error-bar plot
    """
    report_path = report_path or str(RESULTS_DIR / "benchmark_report.json")
    output_dir = output_dir or FIG_DIR
    rpath = Path(report_path)
    if not rpath.exists():
        logger.info("No benchmark report — skipping Fig 16")
        return None

    with open(rpath) as f:
        report = json.load(f)

    methods_data = report.get("methods", {})
    composite = report.get("composite_score", {})
    rankings = report.get("rankings", {})

    if not methods_data:
        logger.warning("Empty benchmark report — skipping Fig 16")
        return None

    # Derive composite scores from normalised metrics when the report
    # does not include pre-computed composite scores.
    if not composite and methods_data:
        _dir_map = {"lower": -1, "higher": 1}
        _hm = [
            ("frechet_distance", "lower"),
            ("coverage", "higher"),
            ("mean_centroid_cosine", "higher"),
            ("diversity_ratio", "higher"),
            ("gene_pearson_r", "higher"),
        ]
        for mname in methods_data:
            _score = 0.0
            _count = 0
            for mk, direction in _hm:
                v = methods_data[mname].get(mk)
                if v is not None:
                    _score += v * _dir_map[direction]
                    _count += 1
            composite[mname] = _score / max(_count, 1)

    method_names = list(methods_data.keys())
    n_methods = len(method_names)

    # Metrics to display in the heatmap — readable names with direction arrows
    heatmap_metrics = [
        ("frechet_distance",     "Fr\u00e9chet \u2193",     "lower"),
        ("mmd_rbf",              "MMD \u2193",              "lower"),
        ("mean_kl",              "KL \u2193",               "lower"),
        ("coverage",             "Coverage \u2191",         "higher"),
        ("density",              "Density \u2191",          "higher"),
        ("mean_centroid_cosine", "Centroid Cos \u2191",     "higher"),
        ("min_centroid_cosine",  "Min Cos \u2191",          "higher"),
        ("diversity_ratio",      "Diversity \u2191",        "higher"),
        ("fraction_collapsed",   "Collapsed \u2193",        "lower"),
        ("gene_pearson_r",       "Gene r \u2191",           "higher"),
        ("gene_spearman_rho",    "Gene \u03c1 \u2191",      "higher"),
    ]

    fig = plt.figure(figsize=(16.0, 9.4))
    gs = fig.add_gridspec(2, 2, wspace=0.52, hspace=0.24, height_ratios=[1.0, 1.12])
    apply_layout_rect(fig, (0.08, 0.16, 0.98, 0.95))

    # ── S1: Heatmap (methods x metrics) ──
    ax1 = fig.add_subplot(gs[0, 0])
    add_panel_label(ax1, 'a', x=-0.10, y=1.05)
    metric_labels = [m[1] for m in heatmap_metrics]
    metric_keys = [m[0] for m in heatmap_metrics]
    directions = [m[2] for m in heatmap_metrics]

    # Build raw values matrix
    raw = np.full((n_methods, len(metric_keys)), np.nan)
    for i, mname in enumerate(method_names):
        for j, mk in enumerate(metric_keys):
            v = methods_data[mname].get(mk)
            if v is not None:
                raw[i, j] = v

    # Normalise each column to [0, 1] with direction awareness
    norm = np.zeros_like(raw)
    for j in range(len(metric_keys)):
        col = raw[:, j]
        valid_mask = ~np.isnan(col)
        if valid_mask.sum() == 0:
            continue
        valid_vals = col[valid_mask]
        mn, mx = valid_vals.min(), valid_vals.max()
        rng = max(mx - mn, 1e-8)
        for i in range(n_methods):
            if np.isnan(raw[i, j]):
                norm[i, j] = 0.0
            elif directions[j] == "lower":
                norm[i, j] = 1.0 - (raw[i, j] - mn) / rng
            else:
                norm[i, j] = (raw[i, j] - mn) / rng

    cmap = matplotlib.colormaps.get_cmap("PiYG")
    im = ax1.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    short_method_names = [abbreviate_cell_type(n, 20) for n in method_names]
    ax1.set_xticks(range(len(metric_labels)))
    ax1.set_xticklabels(metric_labels, rotation=55, ha="right", fontsize=8)
    ax1.set_yticks(range(n_methods))
    ax1.set_yticklabels(short_method_names, fontsize=9)

    # Highlight best cell in each column
    for j in range(len(metric_keys)):
        best_i = norm[:, j].argmax()
        ax1.add_patch(plt.Rectangle((j - 0.5, best_i - 0.5), 1, 1,
                                    fill=False, edgecolor=COLORS["good"], linewidth=2.5))

    cax1 = add_axes_next_to(
        fig,
        ax1,
        side="right",
        width=0.010,
        height=ax1.get_position().height * 0.56,
        pad=0.012,
        align="bottom",
        y_offset=0.01,
    )
    cbar1 = fig.colorbar(im, cax=cax1)
    cbar1.set_label("Normalised Score (1 = best)", fontsize=FONT_ANNOTATION)
    cbar1.ax.tick_params(labelsize=FONT_SMALL)
    style_axes(ax1, "heatmap", title="Metrics Comparison Heatmap")

    # ── S2: Composite score bars ──
    ax2 = fig.add_subplot(gs[0, 1])
    add_panel_label(ax2, 'b', x=-0.10, y=1.05)
    composite_common = report.get("composite_score_common_metrics_only", composite)
    sorted_methods = sorted(composite_common.keys(), key=lambda k: composite_common.get(k, 0.0), reverse=True)
    scores = [composite[m] for m in sorted_methods]
    scores_common = [composite_common.get(m, 0.0) for m in sorted_methods]
    bar_colors = [METHOD_COLORS.get(m, COLORS["neutral"]) for m in sorted_methods]
    short_sorted = [abbreviate_cell_type(m, 20) for m in sorted_methods]

    ranked_labels = []
    for i, name in enumerate(short_sorted):
        badge = f"#{i+1}"
        ranked_labels.append(f"{badge} {name}")

    y_pos = np.arange(len(sorted_methods))
    bar_height = 0.35
    bars_full = ax2.barh(y_pos + bar_height / 2, scores, color=bar_colors,
                         height=bar_height, edgecolor="white", linewidth=0.8,
                         alpha=0.40, hatch="//", label="All metrics (full)")
    bars_common = ax2.barh(y_pos - bar_height / 2, scores_common, color=bar_colors,
                           height=bar_height, edgecolor="white", linewidth=0.8,
                           alpha=0.90, label="Common metrics (primary)")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(ranked_labels, fontsize=9)
    ax2.invert_yaxis()

    for bar, score in zip(bars_full, scores):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{score:.3f}", va="center", fontsize=FONT_SMALL,
                 color=COLORS["neutral"])
    for bar, score in zip(bars_common, scores_common):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{score:.3f}", va="center", fontsize=FONT_SMALL, fontstyle="italic",
                 color=COLORS["neutral"])

    ax2.set_xlim(0, max(scores) * 1.25)
    ax2.legend(fontsize=FONT_SMALL, loc="lower right", framealpha=0.7)
    style_axes(ax2, "bar", title="Composite Score (higher = better)",
               xlabel="Normalised Aggregate Score")

    # ── S3: Grouped bar chart for key metrics ──
    ax3 = fig.add_subplot(gs[1, 0])
    add_panel_label(ax3, 'c', x=-0.10, y=1.05)
    key_metrics = [
        ("frechet_distance",     "FD \u2193"),
        ("mean_centroid_cosine", "Cent Cos \u2191"),
        ("diversity_ratio",      "Div \u2191"),
        ("coverage",             "Cover \u2191"),
        ("gene_pearson_r",       "Gene r \u2191"),
    ]

    x = np.arange(len(key_metrics))
    w = 0.8 / n_methods

    for i, mname in enumerate(method_names):
        vals = [methods_data[mname].get(km[0]) for km in key_metrics]
        vals_plot = [v if v is not None else 0.0 for v in vals]
        offset = (i - n_methods / 2 + 0.5) * w
        color = METHOD_COLORS.get(mname, f"C{i}")
        ax3.bar(x + offset, vals_plot, w, label=mname[:18],
                color=color, alpha=0.85, edgecolor="white")

    ax3.set_xticks(x)
    ax3.set_xticklabels([km[1] for km in key_metrics], fontsize=8, rotation=12, ha="right")
    handles_s3, labels_s3 = ax3.get_legend_handles_labels()
    style_axes(ax3, "bar", title="Key Metrics", ylabel="Value")

    # ── S4: CI comparison — error-bar plot ──
    ax4 = fig.add_subplot(gs[1, 1])
    add_panel_label(ax4, 'd', x=-0.10, y=1.05)
    ci_metrics = [
        ("frechet_distance",     "fd_ci",              "Fr\u00e9chet Distance"),
        ("mean_centroid_cosine", "centroid_cosine_ci",  "Centroid Cosine"),
        ("diversity_ratio",      "diversity_ratio_ci",  "Diversity Ratio"),
    ]

    group_positions = []
    group_labels = []
    all_y = 0

    for mi, (metric_key, ci_key, label) in enumerate(ci_metrics):
        # Add metric group title
        if mi > 0:
            ax4.axhline(y=all_y - 0.8, color="#DDD", linewidth=1, linestyle="--")
            all_y += 0.6

        for mname in method_names:
            val = methods_data[mname].get(metric_key, 0)
            ci = methods_data[mname].get(ci_key, None)
            has_ci = ci is not None
            if ci is None:
                ci = [val, val]
            color = METHOD_COLORS.get(mname, COLORS["neutral"])

            lo_err = max(0, val - ci[0])
            hi_err = max(0, ci[1] - val)
            ax4.errorbar(val, all_y, xerr=[[lo_err], [hi_err]],
                         fmt="o" if has_ci else "D", color=color,
                         markerfacecolor=color if has_ci else "none",
                         markersize=7,
                         capsize=4, capthick=1.5, linewidth=1.5,
                         label=mname[:18] if mi == 0 else None)
            group_positions.append(all_y)
            group_labels.append(f"{mname[:14]}")
            all_y += 1.6

    ax4.set_yticks(group_positions)
    ax4.set_yticklabels(group_labels, fontsize=8)
    ax4.invert_yaxis()

    ax4.legend(fontsize=FONT_SMALL, loc="lower right", ncol=1, frameon=False)
    style_axes(ax4, "default", title="95% Bootstrap CI Comparison",
               xlabel="Metric Value")

    legend_ax_s3 = add_shared_legend_axes(fig, (ax3.get_position().x0, ax3.get_position().y0 - 0.085, ax3.get_position().width, 0.055))
    legend_ax_s3.legend(handles_s3, labels_s3, fontsize=8, loc="center",
                        ncol=min(n_methods, 4), frameon=False, columnspacing=0.8)

    if save:
        path = save_with_vcd(fig, output_dir / "fig16_benchmark.png", dpi, layout_rect=(0.08, 0.16, 0.98, 0.95))
        logger.info(f"Saved Fig 16 \u2192 {path}")
    return fig
