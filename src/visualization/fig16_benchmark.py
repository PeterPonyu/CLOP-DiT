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

from .direct_layout import bind_figure_region
from .explicit_positioning import add_axes_next_to, add_shared_legend_axes
from .style import (
    COLORS, FONT_SMALL, FONT_ANNOTATION, METHOD_COLORS, abbreviate_cell_type,
    add_panel_label, save_panel, save_with_vcd,
    style_axes
)
from src.utils.paths import RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)


def plot_benchmark_panel(
    report_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    dpi: int = 300,
    save: bool = True,
    label_offset: int = 0,
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
        ("frechet_distance",     "FD \u2193",    "lower"),
        ("mmd_rbf",              "MMD \u2193",   "lower"),
        ("mean_kl",              "KL \u2193",    "lower"),
        ("coverage",             "Cov \u2191",   "higher"),
        ("density",              "Den \u2191",   "higher"),
        ("mean_centroid_cosine", "Cent \u2191",  "higher"),
        ("min_centroid_cosine",  "Min \u2191",   "higher"),
        ("diversity_ratio",      "Div \u2191",   "higher"),
        ("fraction_collapsed",   "Coll \u2193",  "lower"),
        ("gene_pearson_r",       "r \u2191",     "higher"),
        ("gene_spearman_rho",    "\u03c1 \u2191",     "higher"),
    ]

    fig = plt.figure(figsize=(16.2, 8.8))
    layout = bind_figure_region(fig, (0.05, 0.12, 0.96, 0.95))
    top_row, bottom_row = layout.split_rows([1.00, 1.28], hspace=0.26)
    top_left, top_right = top_row.split_cols(2, wspace=0.44)
    bottom_left, bottom_right = bottom_row.split_cols([1.04, 0.96], wspace=0.28)

    # ── S1: Heatmap (methods x metrics) ──
    ax1 = top_left.inset(left=0.04).add_axes(fig)
    add_panel_label(ax1, chr(ord('a') + label_offset), x=-0.08, y=1.02)
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

    short_method_names = [abbreviate_cell_type(n, 12) for n in method_names]
    assert len(set(short_method_names)) == len(short_method_names), (
        f"abbreviate_cell_type(max_len=12) collision: {short_method_names}"
    )
    ax1.set_xticks(range(len(metric_labels)))
    ax1.set_xticklabels(metric_labels, rotation=20, ha="right", fontsize=8)
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
    ax2 = top_right.inset(left=0.05, right=0.01).add_axes(fig)
    add_panel_label(ax2, chr(ord('a') + label_offset + 1), x=-0.08, y=1.02)
    composite_common = report.get("composite_score_common_metrics_only", composite)
    sorted_methods = sorted(composite_common.keys(), key=lambda k: composite_common.get(k, 0.0), reverse=True)
    scores = [composite[m] for m in sorted_methods]
    scores_common = [composite_common.get(m, 0.0) for m in sorted_methods]
    bar_colors = [METHOD_COLORS.get(m, COLORS["neutral"]) for m in sorted_methods]
    short_sorted = [abbreviate_cell_type(m, 14) for m in sorted_methods]
    assert len(set(short_sorted)) == len(short_sorted), (
        f"abbreviate_cell_type(max_len=14) collision: {short_sorted}"
    )

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
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, prune="upper"))
    style_axes(ax2, "bar", title="Composite Score (higher = better)",
               xlabel="Normalised Aggregate Score")

    # ── S3: Grouped bar chart for key metrics ──
    ax3 = bottom_left.add_axes(fig)
    target_label_x = ax1.get_position().x0 - 0.12 * ax1.get_position().width
    ax3_label_x = (target_label_x - ax3.get_position().x0) / ax3.get_position().width
    add_panel_label(ax3, chr(ord('a') + label_offset + 2), x=ax3_label_x, y=1.02)
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
    ax3.tick_params(axis='x', pad=4)
    handles_s3, labels_s3 = ax3.get_legend_handles_labels()
    style_axes(ax3, "bar", title="Key Metrics", ylabel="Value")

    # ── S4: CI comparison — error-bar plot ──
    ax4 = bottom_right.add_axes(fig)
    add_panel_label(ax4, chr(ord('a') + label_offset + 3), x=-0.08, y=1.02)
    ci_metrics = [
        ("frechet_distance",     "fd_ci",              "Fr\u00e9chet Distance"),
        ("mean_centroid_cosine", "centroid_cosine_ci",  "Centroid Cosine"),
        ("diversity_ratio",      "diversity_ratio_ci",  "Diversity Ratio"),
    ]

    group_positions = []
    ytick_labels = []
    group_ranges = []
    all_y = 0

    for mi, (metric_key, ci_key, label) in enumerate(ci_metrics):
        # Add metric group title
        if mi > 0:
            ax4.axhline(y=all_y - 0.8, color="#DDD", linewidth=1, linestyle="--")
            all_y += 1.0

        group_start = all_y
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
                         label=mname[:18])
            group_positions.append(all_y)
            ytick_labels.append(abbreviate_cell_type(mname, 10))
            all_y += 1.6
        group_ranges.append((group_start - 0.75, all_y - 1.6 + 0.75, label))

    _ytick_unique = [abbreviate_cell_type(m, 10) for m in method_names]
    assert len(set(_ytick_unique)) == len(_ytick_unique), (
        f"abbreviate_cell_type(max_len=10) collision: {_ytick_unique}"
    )
    for gi, (lo, hi, _label) in enumerate(group_ranges):
        if gi % 2 == 0:
            ax4.axhspan(lo, hi, color=COLORS["bg_gauge"], alpha=0.30, zorder=0)
    ax4.set_yticks(group_positions)
    ax4.set_yticklabels(ytick_labels, fontsize=7)
    ax4.invert_yaxis()
    group_span = len(method_names)
    for gi, (_, _, label) in enumerate(ci_metrics):
        center = gi * (group_span * 1.6 + 1.0) + ((group_span - 1) * 1.6) / 2.0
        ax4.text(
            -0.18,
            center,
            label.replace("Distance", "Dist.").replace("Cosine", "Cos."),
            transform=ax4.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=FONT_SMALL,
            fontweight="bold",
            color=COLORS["neutral"],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.86),
            clip_on=False,
        )

    handles, labels = ax4.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax4.legend(by_label.values(), by_label.keys(), fontsize=FONT_SMALL, loc="lower right", ncol=1, frameon=False)
    style_axes(ax4, "default", title="95% Bootstrap CI by Metric",
               xlabel="Metric Value")

    legend_ax_s3 = add_shared_legend_axes(fig, (ax3.get_position().x0, ax3.get_position().y0 - 0.110, ax3.get_position().width, 0.055))
    legend_ax_s3.legend(handles_s3, labels_s3, fontsize=8, loc="center",
                        ncol=min(n_methods, 4), frameon=False, columnspacing=0.8)

    if save:
        path = save_with_vcd(fig, output_dir / "fig07c_benchmark.png", dpi, layout_rect=(0.05, 0.08, 0.98, 0.95))
        logger.info(f"Saved Fig 16 \u2192 {path}")
    return fig
