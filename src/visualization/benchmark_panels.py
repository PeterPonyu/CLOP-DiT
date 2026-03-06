"""
benchmark_panels.py — Panel S: Comprehensive model benchmarking visualization.

  S1: Metrics heatmap (methods × metrics, colour-coded)
  S2: Composite score bar chart with bootstrap CIs
  S3: Per-metric ranking strip with rank badges
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

from .style import (
    COLORS, save_panel, set_dense_tick_labels, style_axes
)
from src.utils.paths import RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)

METHOD_COLORS = {
    "CLOP-DiT": "#1976D2",
    "Gaussian N(μ,σ²I)": "#FF7043",
    "Shuffled Labels": "#4CAF50",
    "Random N(0,I)": "#9C27B0",
    "Mean-only (collapse)": "#FFC107",
}


def plot_benchmark_panel(
    report_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    dpi: int = 300,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Panel S: Comprehensive model benchmarking dashboard (2×2).

    S1: Metrics heatmap — methods × metrics, cell-coloured by normalised value
    S2: Composite score — horizontal bars with score labels
    S3: Key metrics comparison — grouped bar chart for FD, Cosine, Diversity, Coverage
    S4: Confidence interval comparison — error-bar plot for FD and Centroid Cosine
    """
    report_path = report_path or str(RESULTS_DIR / "benchmark_report.json")
    output_dir = output_dir or FIG_DIR
    rpath = Path(report_path)
    if not rpath.exists():
        logger.info("No benchmark report — skipping Panel S")
        return None

    with open(rpath) as f:
        report = json.load(f)

    methods_data = report.get("methods", {})
    composite = report.get("composite_score", {})
    rankings = report.get("rankings", {})

    if not methods_data:
        logger.warning("Empty benchmark report — skipping Panel S")
        return None

    method_names = list(methods_data.keys())
    n_methods = len(method_names)

    # Metrics to display in the heatmap
    heatmap_metrics = [
        ("frechet_distance", "FD\u2193", "lower"),
        ("mmd_rbf", "MMD\u2193", "lower"),
        ("mean_kl", "KL\u2193", "lower"),
        ("coverage", "Cov\u2191", "higher"),
        ("density", "Den\u2191", "higher"),
        ("mean_centroid_cosine", "Cos\u2191", "higher"),
        ("min_centroid_cosine", "Min\u2191", "higher"),
        ("diversity_ratio", "Div\u2191", "higher"),
        ("fraction_collapsed", "Col\u2193", "lower"),
        ("gene_pearson_r", "gPr\u2191", "higher"),
        ("gene_spearman_rho", "gSp\u2191", "higher"),
    ]

    fig = plt.figure(figsize=(15.0, 9.0))
    gs = fig.add_gridspec(2, 2, wspace=0.55, hspace=0.50)
    fig.suptitle(
        "Model Benchmarking — CLOP-DiT vs Baselines",
        fontsize=11,
    )

    # ── S1: Heatmap (methods × metrics) ──
    ax1 = fig.add_subplot(gs[0, 0])
    metric_labels = [m[1] for m in heatmap_metrics]
    metric_keys = [m[0] for m in heatmap_metrics]
    directions = [m[2] for m in heatmap_metrics]

    # Build raw values matrix (None → NaN for proper handling)
    raw = np.full((n_methods, len(metric_keys)), np.nan)
    for i, mname in enumerate(method_names):
        for j, mk in enumerate(metric_keys):
            v = methods_data[mname].get(mk)
            if v is not None:
                raw[i, j] = v

    # Normalise each column to [0, 1] with direction awareness
    # NaN (missing capability) → normalised score 0.0 (worst)
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
                norm[i, j] = 0.0  # penalise missing capability
            elif directions[j] == "lower":
                norm[i, j] = 1.0 - (raw[i, j] - mn) / rng
            else:
                norm[i, j] = (raw[i, j] - mn) / rng

    # Plot heatmap
    cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    im = ax1.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Labels
    short_method_names = [n[:20] for n in method_names]
    ax1.set_xticks(range(len(metric_labels)))
    ax1.set_xticklabels(metric_labels, rotation=55, ha="right", fontsize=8)
    ax1.set_yticks(range(n_methods))
    ax1.set_yticklabels(short_method_names, fontsize=8)

    # Highlight best cell in each column (no cell text — colours tell the story)
    for j in range(len(metric_keys)):
        best_i = norm[:, j].argmax()
        ax1.add_patch(plt.Rectangle((j - 0.5, best_i - 0.5), 1, 1,
                                    fill=False, edgecolor="#1B5E20", linewidth=2.5))

    plt.colorbar(im, ax=ax1, shrink=0.6, pad=0.02, label="Normalised Score (1=best)")
    style_axes(ax1, "heatmap", title="Metrics Comparison Heatmap")

    # ── S2: Composite score bars ──
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_methods = sorted(composite.keys(), key=lambda k: composite[k], reverse=True)
    scores = [composite[m] for m in sorted_methods]
    bar_colors = [METHOD_COLORS.get(m, "#999") for m in sorted_methods]
    short_sorted = [m[:20] for m in sorted_methods]

    # Merge rank badges directly into ytick labels to avoid overlap
    ranked_labels = []
    for i, name in enumerate(short_sorted):
        badge = "#1" if i == 0 else "#2" if i == 1 else "#3" if i == 2 else f"#{i+1}"
        ranked_labels.append(f"{badge} {name}")

    y_pos = np.arange(len(sorted_methods))
    bars = ax2.barh(y_pos, scores, color=bar_colors, height=0.6,
                    edgecolor="white", linewidth=0.8, alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(ranked_labels, fontsize=8)
    ax2.invert_yaxis()

    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{score:.4f}", va="center", fontsize=8,
                 color="#333333")

    ax2.set_xlim(0, max(scores) * 1.25)
    style_axes(ax2, "bar", title="Composite Score (higher = better)",
               xlabel="Normalised Aggregate Score")

    # ── S3: Grouped bar chart for key metrics ──
    ax3 = fig.add_subplot(gs[1, 0])
    key_metrics = [
        ("frechet_distance", "FD ↓"),
        ("mean_centroid_cosine", "Cos ↑"),
        ("diversity_ratio", "Div ↑"),
        ("coverage", "Cov ↑"),
        ("gene_pearson_r", "Gene r ↑"),
    ]

    x = np.arange(len(key_metrics))
    w = 0.8 / n_methods

    for i, mname in enumerate(method_names):
        vals = [methods_data[mname].get(km[0]) for km in key_metrics]
        # Replace None with 0 for bar plotting
        vals_plot = [v if v is not None else 0.0 for v in vals]
        offset = (i - n_methods / 2 + 0.5) * w
        color = METHOD_COLORS.get(mname, f"C{i}")
        ax3.bar(x + offset, vals_plot, w, label=mname[:18],
                color=color, alpha=0.85, edgecolor="white")

    ax3.set_xticks(x)
    ax3.set_xticklabels([km[1] for km in key_metrics], fontsize=10)
    ax3.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    style_axes(ax3, "bar", title="Key Metrics Comparison", ylabel="Value")

    # ── S4: CI comparison — error-bar plot ──
    ax4 = fig.add_subplot(gs[1, 1])
    ci_metrics = [
        ("frechet_distance", "fd_ci", "FD"),
        ("mean_centroid_cosine", "centroid_cosine_ci", "Centroid Cos"),
        ("diversity_ratio", "diversity_ratio_ci", "Diversity"),
    ]

    group_positions = []
    group_labels = []
    all_y = 0

    for mi, (metric_key, ci_key, label) in enumerate(ci_metrics):
        for mname in method_names:
            val = methods_data[mname].get(metric_key, 0)
            ci = methods_data[mname].get(ci_key, [val, val])
            color = METHOD_COLORS.get(mname, "#999")

            lo_err = max(0, val - ci[0])
            hi_err = max(0, ci[1] - val)
            ax4.errorbar(val, all_y, xerr=[[lo_err], [hi_err]],
                         fmt="o", color=color, markersize=7,
                         capsize=4, capthick=1.5, linewidth=1.5,
                         label=mname[:18] if mi == 0 else None)
            group_positions.append(all_y)
            group_labels.append(f"{mname[:14]}")
            all_y += 1.6
        # Add metric group separator
        if mi < len(ci_metrics) - 1:
            ax4.axhline(y=all_y - 0.8, color="#DDD", linewidth=1, linestyle="--")
            all_y += 1.2

    ax4.set_yticks(group_positions)
    ax4.set_yticklabels(group_labels, fontsize=8)
    ax4.invert_yaxis()

    # Add metric group titles on the right — removed: y-labels already convey grouping

    ax4.legend(fontsize=8, loc="upper left", ncol=1)
    style_axes(ax4, "default", title="95% Bootstrap CI Comparison",
               xlabel="Metric Value")

    if save:
        path = save_panel(fig, output_dir / "panel_s_benchmark.png", dpi)
        logger.info(f"Saved Panel S → {path}")
    return fig
