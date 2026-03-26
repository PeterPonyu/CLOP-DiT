"""Fig S1: Robustness & Ablation composite — three-row supplementary figure.

Merges three standalone figures into a single 3-row layout:
  Row 1 (panel a): CLOP ablation heatmap          — from all_summaries.json
  Row 2 (panel b): Multi-seed robustness bars      — from multi_seed_report.json
  Row 3 (panel c): OOD showcase table              — from ood_results.json

Each panel degrades gracefully: if its data file is missing the row displays
a "Data not available" placeholder and the figure is still produced.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from .direct_layout import bind_figure_region
from .style import (
    COLORS,
    FONT_ANNOTATION,
    FONT_LABEL,
    FONT_LEGEND_DENSE,
    FONT_TICK,
    FONT_TITLE,
    add_panel_label,
    apply_style,
    save_with_vcd,
    style_axes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Panel (a) constants — ablation heatmap
# ---------------------------------------------------------------------------
_METRIC_COLS = [
    ("best_val_proto_acc", "Proto Acc"),
    ("best_val_proto_top5", "Top-5 Acc"),
    ("best_val_proto_top10", "Top-10 Acc"),
    ("best_val_loss", "Val Loss"),
    ("best_epoch", "Best Epoch"),
]

_CATEGORY_COLORS = {
    "architecture": "#1565C0",
    "conditioning": "#0097A7",
    "loss": "#C62828",
    "optimization": "#6A1B9A",
    "reference": "#455A64",
    "regularization": "#2E7D32",
    "temperature": "#6A1B9A",
    "training": "#E65100",
    "other": COLORS["neutral"],
}

# ---------------------------------------------------------------------------
# Panel (b) constants — multi-seed robustness
# ---------------------------------------------------------------------------
_SEED_METRICS = [
    ("knn_top1", "kNN Top-1"),
    ("knn_top5", "kNN Top-5"),
    ("steering_accuracy", "Steering Acc"),
    ("diversity_ratio", "Diversity Ratio"),
    ("linear_accuracy", "Linear Acc"),
    ("centroid_cosine", "Centroid Cos"),
    ("frechet_distance", "FD \u2193"),
]

_REGIME_COLORS = {
    "high_fidelity_regime": COLORS["real"],
    "high_diversity_regime": COLORS["generated"],
}

_REGIME_LABELS = {
    "high_fidelity_regime": "High Fidelity",
    "high_diversity_regime": "High Diversity",
}


# ---------------------------------------------------------------------------
# Helper: placeholder for missing data
# ---------------------------------------------------------------------------

def _draw_placeholder(ax: plt.Axes, message: str = "Data not available") -> None:
    """Fill an axes with a centred 'not available' message."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center",
        fontsize=FONT_TITLE,
        color=COLORS["neutral"],
        style="italic",
        transform=ax.transAxes,
    )
    ax.axis("off")


# ---------------------------------------------------------------------------
# Panel (a): Ablation heatmap
# ---------------------------------------------------------------------------

def _draw_ablation_heatmap(fig: plt.Figure, region, ablation_path: Path) -> None:
    """Render the ablation heatmap into *region*."""
    if not ablation_path.exists():
        logger.warning("Ablation summaries not found: %s", ablation_path)
        ax = region.add_axes(fig)
        _draw_placeholder(ax, "Ablation data not available")
        add_panel_label(ax, "a", x=-0.10, y=1.04)
        return

    with open(ablation_path) as f:
        summaries = json.load(f)

    if not summaries:
        logger.warning("No ablation entries found")
        ax = region.add_axes(fig)
        _draw_placeholder(ax, "No ablation entries")
        add_panel_label(ax, "a", x=-0.10, y=1.04)
        return

    # Sort by category then name
    summaries.sort(key=lambda s: (s.get("category", "other"), s.get("ablation_name", "")))

    names = [s["ablation_name"].replace("_", " ") for s in summaries]
    categories = [s.get("category", "other") for s in summaries]
    n_variants = len(summaries)
    n_metrics = len(_METRIC_COLS)

    # Build data matrix
    data = np.full((n_variants, n_metrics), np.nan)
    for i, s in enumerate(summaries):
        metrics = s.get("metrics", {})
        for j, (key, _label) in enumerate(_METRIC_COLS):
            val = metrics.get(key)
            if val is not None:
                data[i, j] = float(val)

    # Column-wise normalisation to [0, 1]; loss columns inverted
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

    # Split region: heatmap (left, wide) + colorbar gutter (right, narrow)
    ax = region.add_axes(fig)
    im = ax.imshow(norm_data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    # Axis labels
    metric_labels = [label for _, label in _METRIC_COLS]
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_labels, rotation=0, ha="center", fontsize=FONT_TICK)
    ax.set_yticks(range(n_variants))
    ax.set_yticklabels(names, fontsize=FONT_TICK)

    # Cell annotations
    for i in range(n_variants):
        for j in range(n_metrics):
            val = data[i, j]
            if np.isnan(val):
                continue
            color = "white" if norm_data[i, j] < 0.3 or norm_data[i, j] > 0.7 else "black"
            fmt = ".3f" if val < 10 else ".1f"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                    fontsize=FONT_ANNOTATION, color=color)

    # Category colour markers on left
    for i, cat in enumerate(categories):
        color = _CATEGORY_COLORS.get(cat, COLORS["neutral"])
        ax.plot(-0.7, i, "s", color=color, markersize=5, clip_on=False,
                transform=ax.transData)

    # Category legend
    seen: dict[str, str] = {}
    for cat in categories:
        if cat not in seen:
            seen[cat] = _CATEGORY_COLORS.get(cat, COLORS["neutral"])
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                   markersize=6, label=cat.capitalize())
        for cat, c in seen.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              ncol=1, fontsize=FONT_LEGEND_DENSE, frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.85)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Normalized (higher = better)", fontsize=FONT_ANNOTATION)

    ax.set_title("CLOP Ablation Comparison", fontsize=FONT_TITLE, fontweight="normal", pad=8)
    add_panel_label(ax, "a", x=-0.10, y=1.04)


# ---------------------------------------------------------------------------
# Panel (b): Multi-seed robustness
# ---------------------------------------------------------------------------

def _draw_multi_seed(fig: plt.Figure, region, report_path: Path) -> None:
    """Render grouped bar chart of multi-seed metrics into *region*."""
    if not report_path.exists():
        logger.warning("Multi-seed report not found: %s", report_path)
        ax = region.add_axes(fig)
        _draw_placeholder(ax, "Multi-seed data not available")
        add_panel_label(ax, "b", x=-0.10, y=1.04)
        return

    with open(report_path) as f:
        report = json.load(f)

    regimes = [k for k in report if k in _REGIME_LABELS]
    if not regimes:
        logger.warning("No recognized regimes in multi-seed report")
        ax = region.add_axes(fig)
        _draw_placeholder(ax, "No regime data")
        add_panel_label(ax, "b", x=-0.10, y=1.04)
        return

    ax = region.add_axes(fig)

    n_metrics = len(_SEED_METRICS)
    n_regimes = len(regimes)
    bar_width = 0.35
    x = np.arange(n_metrics)

    for idx, regime in enumerate(regimes):
        agg = report[regime].get("aggregated", {})
        means = []
        stds = []
        for key, _label in _SEED_METRICS:
            m = agg.get(key, {})
            means.append(m.get("mean", 0))
            stds.append(m.get("std", 0))

        offset = (idx - (n_regimes - 1) / 2) * bar_width
        color = _REGIME_COLORS.get(regime, COLORS["neutral"])
        label = _REGIME_LABELS.get(regime, regime)
        ax.bar(x + offset, means, bar_width, yerr=stds, label=label,
               color=color, alpha=0.85, capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in _SEED_METRICS], rotation=0, ha="center",
                       fontsize=FONT_TICK)
    ax.set_ylabel("Score", fontsize=FONT_LABEL)
    ax.set_ylim(0, 1.15)  # cap y-axis so outliers don't dominate
    ax.legend(fontsize=FONT_LEGEND_DENSE, loc="upper right", frameon=False)
    style_axes(ax)

    # Annotate bars that exceed the y-axis limit (e.g. Frechet Distance)
    for bar_group in ax.containers:
        for bar in bar_group:
            if bar is None or not hasattr(bar, 'get_height'):
                continue
            if bar.get_height() > 1.15:
                ax.text(bar.get_x() + bar.get_width() / 2, 1.12,
                        f"{bar.get_height():.1f}",
                        ha="center", va="top", fontsize=FONT_ANNOTATION,
                        color="white", fontweight="bold")

    # Annotate seed count
    first_regime = regimes[0]
    n_seeds = report[first_regime].get("aggregated", {}).get("n_seeds", "?")
    ax.text(0.98, 0.02, f"n = {n_seeds} seeds",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FONT_LEGEND_DENSE, color=COLORS["neutral"])

    ax.set_title("Multi-Seed Generation Robustness", fontsize=FONT_TITLE, fontweight="normal", pad=8)
    add_panel_label(ax, "b", x=-0.10, y=1.04)


# ---------------------------------------------------------------------------
# Panel (c): OOD showcase table
# ---------------------------------------------------------------------------

def _draw_ood_showcase(fig: plt.Figure, region, ood_path: Path) -> None:
    """Render OOD showcase as a styled text table into *region*."""
    if not ood_path.exists():
        logger.warning("OOD results not found: %s", ood_path)
        ax = region.add_axes(fig)
        _draw_placeholder(ax, "OOD data not available")
        add_panel_label(ax, "c", x=-0.10, y=1.04)
        return

    with open(ood_path) as f:
        ood = json.load(f)

    novel_types = ood.get("novel_types", {})
    free_form = ood.get("free_form", {})

    if not novel_types and not free_form:
        logger.warning("No OOD entries found")
        ax = region.add_axes(fig)
        _draw_placeholder(ax, "No OOD entries")
        add_panel_label(ax, "c", x=-0.10, y=1.04)
        return

    # Split region into left (novel types) and right (free-form)
    left, right = region.split_cols([1, 1], wspace=0.08)

    # ── Left: Novel cell types ──
    ax_l = left.add_axes(fig)
    ax_l.set_xlim(0, 10)
    ax_l.set_ylim(-1.2, max(len(novel_types), 1) - 0.5)
    ax_l.invert_yaxis()
    ax_l.axis("off")

    # Header
    ax_l.text(0.0, -1.0, "Cell Type", fontsize=FONT_LABEL, fontweight="medium", va="center")
    ax_l.text(4.5, -1.0, "Prompt Excerpt", fontsize=FONT_LABEL, fontweight="medium", va="center")

    for i, (type_name, info) in enumerate(novel_types.items()):
        prompt = info.get("prompt", "")
        excerpt = prompt[:50] + "\u2026" if len(prompt) > 50 else prompt
        ax_l.text(0.0, i, type_name, fontsize=FONT_TICK, va="center",
                  color=COLORS["real"], fontweight="medium")
        ax_l.text(4.5, i, excerpt, fontsize=FONT_ANNOTATION, va="center",
                  color=COLORS["annotation_dark"], style="italic")
        if i < len(novel_types) - 1:
            ax_l.axhline(y=i + 0.5, color=COLORS["border_light"], linewidth=0.5,
                         xmin=0, xmax=1)

    ax_l.set_title("Novel Cell Types", fontsize=FONT_TITLE - 1, fontweight="normal", pad=10)

    # ── Right: Free-form prompts ──
    ax_r = right.add_axes(fig)
    ax_r.set_xlim(0, 10)
    n_ff = max(len(free_form), 1)
    ax_r.set_ylim(-1.2, n_ff - 0.5)
    ax_r.invert_yaxis()
    ax_r.axis("off")

    ax_r.text(0.0, -1.0, "Prompt ID", fontsize=FONT_LABEL, fontweight="medium", va="center")
    ax_r.text(3.5, -1.0, "Free-form Description", fontsize=FONT_LABEL, fontweight="medium",
              va="center")

    for i, (prompt_id, info) in enumerate(free_form.items()):
        ff_prompt = info.get("free_form_prompt", "")
        excerpt = ff_prompt[:45] + "\u2026" if len(ff_prompt) > 45 else ff_prompt
        display_id = prompt_id.replace("_", " ")
        ax_r.text(0.0, i, display_id, fontsize=FONT_TICK, va="center",
                  color=COLORS["generated"], fontweight="medium")
        ax_r.text(3.5, i, excerpt, fontsize=FONT_ANNOTATION, va="center",
                  color=COLORS["annotation_dark"], style="italic")
        if i < len(free_form) - 1:
            ax_r.axhline(y=i + 0.5, color=COLORS["border_light"], linewidth=0.5,
                         xmin=0, xmax=1)

    ax_r.set_title("Free-form Prompts", fontsize=FONT_TITLE - 1, fontweight="normal", pad=10)

    # Overall panel label on left sub-panel
    add_panel_label(ax_l, "c", x=-0.10, y=1.04)


# ===========================================================================
# Main composite entry point
# ===========================================================================

def plot_robustness_ablation(
    output_dir: str | Path = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Render the 3-row robustness & ablation composite figure.

    Parameters
    ----------
    output_dir : str | Path
        Directory for output PNG + PDF.
    dpi : int
        Export resolution.
    save : bool
        Whether to write files to disk.
    save_panel_fn : Callable, optional
        Custom save callback ``fn(fig, stem)``.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    apply_style()
    output_dir = Path(output_dir)

    # Data file paths
    ablation_path = Path("results/ablations/all_summaries.json")
    seed_path = Path("results/multi_seed/multi_seed_report.json")
    ood_path = Path("results/ood_evaluation/ood_results.json")

    # Create figure — height sized for three rows with adequate inter-row gaps
    fig = plt.figure(figsize=(16, 12.5))

    # Divide figure into three rows: heatmap (tall), bars (medium), table (medium)
    # Increased inter-row gaps to prevent x-axis labels overlapping titles below
    layout = bind_figure_region(fig, (0.12, 0.04, 0.96, 0.96))
    row_a, row_b, row_c = layout.split_rows([1.28, 0.94, 0.84], gap=0.10)

    # Draw each panel
    _draw_ablation_heatmap(fig, row_a, ablation_path)
    _draw_multi_seed(fig, row_b, seed_path)
    _draw_ood_showcase(fig, row_c, ood_path)

    # Save
    if save:
        out = output_dir / "figS01a_robustness_ablation.png"
        if save_panel_fn:
            save_panel_fn(fig, "figS01a_robustness_ablation")
        else:
            save_with_vcd(fig, out, dpi=dpi)

    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    logging.basicConfig(level=logging.INFO)
    plot_robustness_ablation()
