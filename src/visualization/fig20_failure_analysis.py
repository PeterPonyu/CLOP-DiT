"""
fig20_failure_analysis.py — Per-cell-type failure analysis figure.

Panels (sequential within composed Figure A2):
  (c) Pass/Warn/Fail tier stacked bar by biological family
  (d) Quality score vs sample size scatter
  (e) Quality score vs embedding overlap scatter
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
    FONT_LABEL, FONT_TITLE, FONT_ANNOTATION, FONT_SMALL,
)
from src.utils.paths import RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)

TIER_COLORS = {"pass": "#1B5E20", "warn": "#F9A825", "fail": "#D84315"}


def plot_failure_analysis(output_dir=None, dpi=300):
    """Generate failure analysis figure."""
    apply_style()
    output_dir = Path(output_dir or FIG_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_dir = RESULTS_DIR / "validation"
    analysis_path = val_dir / "failure_analysis.json"
    tiers_path = val_dir / "failure_tiers.json"

    if not analysis_path.exists() or not tiers_path.exists():
        logger.warning("Failure analysis results not found. "
                       "Run: python scripts/analysis/failure_analysis.py")
        return None

    with open(analysis_path) as f:
        analysis = json.load(f)
    with open(tiers_path) as f:
        tiers = json.load(f)

    per_type = analysis["per_type_metrics"]
    patterns = analysis["summary"]["patterns"]

    fig = plt.figure(figsize=(8.0, 3.5))
    layout = bind_figure_region(fig, (0.08, 0.16, 0.98, 0.88))
    left, mid, right = layout.split_cols([1.0, 1.0, 1.0], wspace=0.26)

    # Panel (c): Stacked bar by biological family
    ax = left.inset(right=0.02).add_axes(fig)
    add_panel_label(ax, 'c', x=-0.12, y=1.14)

    by_family = patterns.get("by_family", {})
    families = sorted(by_family.keys(), key=lambda f: by_family[f]["mean_score"], reverse=True)
    y_pos = np.arange(len(families))

    pass_counts = [by_family[f]["tier_counts"].get("pass", 0) for f in families]
    warn_counts = [by_family[f]["tier_counts"].get("warn", 0) for f in families]
    fail_counts = [by_family[f]["tier_counts"].get("fail", 0) for f in families]

    ax.barh(y_pos, pass_counts, color=TIER_COLORS["pass"], height=0.6, label="Pass")
    ax.barh(y_pos, warn_counts, left=pass_counts, color=TIER_COLORS["warn"],
            height=0.6, label="Warn")
    cumulative = [p + w for p, w in zip(pass_counts, warn_counts)]
    ax.barh(y_pos, fail_counts, left=cumulative, color=TIER_COLORS["fail"],
            height=0.6, label="Fail")

    short_families = [f.replace("Immune: ", "").replace("Stem/", "S/")[:18] for f in families]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_families, fontsize=9)
    ax.invert_yaxis()
    ax.legend(fontsize=FONT_ANNOTATION, frameon=False, loc="lower right")
    style_axes(ax, "bar", xlabel="Number of cell types",
               title="Quality Tiers by Biological Family")

    # Panel (d): Quality score vs log(sample size)
    ax2 = mid.inset(left=0.05, right=0.02).add_axes(fig)
    add_panel_label(ax2, 'd', x=-0.12, y=1.14)

    names = list(per_type.keys())
    n_reals = np.array([per_type[n]["n_real"] for n in names])
    scores = np.array([tiers[n]["score"] for n in names])
    tier_labels = [tiers[n]["tier"] for n in names]
    point_colors = [TIER_COLORS[t] for t in tier_labels]

    ax2.scatter(n_reals, scores, c=point_colors, s=30, alpha=0.7,
                edgecolors="white", linewidth=0.3, zorder=3)
    ax2.set_xscale("log")

    # Regression line
    from scipy import stats as scipy_stats
    log_n = np.log10(n_reals + 1)
    valid = np.isfinite(log_n) & np.isfinite(scores)
    if valid.sum() >= 3:
        slope, intercept, r_val, p_val, _ = scipy_stats.linregress(log_n[valid], scores[valid])
        x_fit = np.linspace(log_n[valid].min(), log_n[valid].max(), 50)
        ax2.plot(10**x_fit, slope * x_fit + intercept, color="#555",
                 linewidth=1.5, linestyle="--", alpha=0.7)
        ax2.text(0.03, 0.04, f"r={r_val:.3f}", transform=ax2.transAxes,
                 fontsize=FONT_ANNOTATION, color=COLORS["neutral"])

    ax2.set_ylim(4.5, 8.3)
    ax2.set_yticks([5, 6, 7, 8])
    style_axes(ax2, "scatter", xlabel="Training cells (log scale)",
               ylabel="Quality score (0-8)", title="Quality vs Sample Size")

    # Panel (e): Quality score vs max overlap (confusability)
    ax3 = right.inset(left=0.05, right=0.02).add_axes(fig)
    add_panel_label(ax3, 'e', x=-0.12, y=1.14)

    overlaps = np.array([per_type[n]["max_overlap_with_other_type"] for n in names])
    ax3.scatter(overlaps, scores, c=point_colors, s=30, alpha=0.7,
                edgecolors="white", linewidth=0.3, zorder=3)

    if len(overlaps) >= 5:
        slope, intercept, r_val, _, _ = scipy_stats.linregress(overlaps[valid], scores[valid])
        x_fit = np.linspace(overlaps.min(), overlaps.max(), 50)
        ax3.plot(x_fit, slope * x_fit + intercept, color="#555",
                 linewidth=1.5, linestyle="--", alpha=0.7)
        ax3.text(0.03, 0.04, f"r={r_val:.3f}", transform=ax3.transAxes,
                 fontsize=FONT_ANNOTATION, color=COLORS["neutral"])

    ax3.set_ylim(4.5, 8.3)
    ax3.set_yticks([5, 6, 7, 8])
    style_axes(ax3, "scatter", xlabel="Max cosine overlap with other type",
               ylabel="Quality score (0-8)", title="Quality vs Confusability")

    fig_path = output_dir / "figS02b_failure_analysis"
    save_with_vcd(fig, fig_path, dpi=dpi, layout_rect=(0.06, 0.06, 0.98, 0.94))
    plt.close()
    logger.info(f"Saved: {fig_path}")
    return fig_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_failure_analysis()
