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

from ..direct_layout import bind_figure_region
from ..style import (
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
    ("knn_top1", "kNN-1"),
    ("knn_top5", "kNN-5"),
    ("steering_accuracy", "Steer."),
    ("diversity_ratio", "DivR"),
    ("linear_accuracy", "LinAcc"),
    ("centroid_cosine", "Cen.Cos"),
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
        add_panel_label(ax, "a", x=-0.08, y=1.05)
        return

    with open(ablation_path) as f:
        summaries = json.load(f)

    if not summaries:
        logger.warning("No ablation entries found")
        ax = region.add_axes(fig)
        _draw_placeholder(ax, "No ablation entries")
        add_panel_label(ax, "a", x=-0.08, y=1.05)
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
    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(1.12, 0.5), ncol=1,
              fontsize=FONT_LEGEND_DENSE, frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.9)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.04)
    cbar.set_label("Normalized (higher = better)", fontsize=FONT_ANNOTATION)

    ax.set_title("CLOP Ablation Comparison", fontsize=FONT_TITLE, fontweight="normal", pad=8)
    add_panel_label(ax, "a", x=-0.08, y=1.05)


# ---------------------------------------------------------------------------
# Panel (b): Multi-seed robustness
# ---------------------------------------------------------------------------

def _draw_multi_seed(fig: plt.Figure, region, report_path: Path) -> None:
    """Render grouped bar chart of multi-seed metrics into *region*."""
    if not report_path.exists():
        logger.warning("Multi-seed report not found: %s", report_path)
        ax = region.add_axes(fig)
        _draw_placeholder(ax, "Multi-seed data not available")
        add_panel_label(ax, "b", x=-0.08, y=1.08)
        return

    with open(report_path) as f:
        report = json.load(f)

    regimes = [k for k in report if k in _REGIME_LABELS]
    if not regimes:
        logger.warning("No recognized regimes in multi-seed report")
        ax = region.add_axes(fig)
        _draw_placeholder(ax, "No regime data")
        add_panel_label(ax, "b", x=-0.08, y=1.08)
        return

    ax = region.add_axes(fig)

    n_metrics = len(_SEED_METRICS)
    n_regimes = len(regimes)
    bar_width = 0.35
    x = np.arange(n_metrics)

    _Y_CAP = 1.05
    original_means_by_container: list[list[float]] = []
    for idx, regime in enumerate(regimes):
        agg = report[regime].get("aggregated", {})
        means = []
        stds = []
        for key, _label in _SEED_METRICS:
            m = agg.get(key, {})
            means.append(m.get("mean", 0))
            stds.append(m.get("std", 0))

        # Cap bar heights AND error bars at the axis ceiling so Rectangle /
        # Line2D artists do not extend past set_ylim(0, 1.05) — avoids VCD
        # axes_overflow while annotation below surfaces the true value.
        draw_means = [min(mv, _Y_CAP) for mv in means]
        draw_stds = [min(sv, max(0.0, _Y_CAP - mv)) for sv, mv in zip(stds, draw_means)]
        original_means_by_container.append(list(means))

        offset = (idx - (n_regimes - 1) / 2) * bar_width
        color = _REGIME_COLORS.get(regime, COLORS["neutral"])
        label = _REGIME_LABELS.get(regime, regime)
        ax.bar(x + offset, draw_means, bar_width, yerr=draw_stds, label=label,
               color=color, alpha=0.85, capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in _SEED_METRICS], rotation=0, ha="center",
                       fontsize=FONT_TICK)
    ax.set_ylabel("Score", fontsize=FONT_LABEL)
    ax.set_ylim(0, 1.05)  # cap y-axis so outliers don't dominate
    ax.legend(fontsize=FONT_LEGEND_DENSE, loc="upper right", frameon=False)
    style_axes(ax)

    # Annotate bars whose original value was capped at the axis ceiling
    # (e.g. Frechet Distance) — surface the true value via text.
    for container, orig_means in zip(ax.containers, original_means_by_container):
        for bar, orig in zip(container, orig_means):
            if bar is None or not hasattr(bar, 'get_height') or orig <= _Y_CAP:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, 1.02,
                    f"{orig:.1f}",
                    ha="center", va="top", fontsize=FONT_ANNOTATION,
                    color="white", fontweight="bold")

    # Annotate seed count
    first_regime = regimes[0]
    n_seeds = report[first_regime].get("aggregated", {}).get("n_seeds", "?")
    ax.text(0.98, 0.02, f"n = {n_seeds} seeds",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FONT_LEGEND_DENSE, color=COLORS["neutral"])

    ax.set_title("Multi-Seed Robustness", fontsize=FONT_TITLE, fontweight="normal", pad=8)
    add_panel_label(ax, "b", x=-0.08, y=1.08)


# ---------------------------------------------------------------------------
# Panel (c): OOD showcase table
# ---------------------------------------------------------------------------

def _draw_ood_showcase(fig: plt.Figure, region, ood_path: Path) -> None:
    """Render OOD showcase as a styled text table into *region*."""
    if not ood_path.exists():
        logger.warning("OOD results not found: %s", ood_path)
        ax = region.add_axes(fig)
        _draw_placeholder(ax, "OOD data not available")
        add_panel_label(ax, "c", x=-0.08, y=1.08)
        return

    with open(ood_path) as f:
        ood = json.load(f)

    novel_types = ood.get("novel_types", {})
    free_form = ood.get("free_form", {})

    if not novel_types and not free_form:
        logger.warning("No OOD entries found")
        ax = region.add_axes(fig)
        _draw_placeholder(ax, "No OOD entries")
        add_panel_label(ax, "c", x=-0.08, y=1.08)
        return

    # Split region into left (novel types) and right (free-form). The right
    # side needs extra width because free-form prompt IDs are longer and were
    # colliding with the description column at article scale.
    left, right = region.split_cols([0.98, 1.22], wspace=0.10)

    # Cap entries to avoid text overcrowding at article scale
    _MAX_OOD_ROWS = 7
    novel_items = list(novel_types.items())[:_MAX_OOD_ROWS]
    ff_items = list(free_form.items())[:_MAX_OOD_ROWS]

    # Vertical spacing factor — spread rows apart for readability
    _ROW_STEP = 1.25

    # ── Left: Novel cell types ──
    ax_l = left.add_axes(fig)
    ax_l.set_xlim(0, 11.5)
    n_novel = max(len(novel_items), 1)
    ax_l.set_ylim(-1.8, (n_novel - 1) * _ROW_STEP + 0.8)
    ax_l.invert_yaxis()
    ax_l.axis("off")

    # Header
    ax_l.text(0.0, -1.1, "Cell Type", fontsize=FONT_LABEL + 1, fontweight="medium", va="center")
    ax_l.text(5.1, -1.1, "Prompt Excerpt", fontsize=FONT_LABEL + 1, fontweight="medium", va="center")

    for i, (type_name, info) in enumerate(novel_items):
        y_pos = i * _ROW_STEP
        prompt = info.get("prompt", "")
        excerpt = prompt[:38] + "\u2026" if len(prompt) > 38 else prompt
        short_name = type_name if len(type_name) <= 20 else type_name[:18] + "\u2026"
        ax_l.text(0.0, y_pos, short_name, fontsize=FONT_LABEL, va="center",
                  color=COLORS["real"], fontweight="medium")
        ax_l.text(5.1, y_pos, excerpt, fontsize=FONT_TICK, va="center",
                  color=COLORS["annotation_dark"], style="italic")
        if i < len(novel_items) - 1:
            ax_l.axhline(y=y_pos + _ROW_STEP * 0.5, color=COLORS["border_light"],
                         linewidth=0.5, xmin=0, xmax=1)

    ax_l.set_title("Novel Cell Types", fontsize=FONT_TITLE, fontweight="normal", pad=10)

    # ── Right: Free-form prompts ──
    ax_r = right.add_axes(fig)
    ax_r.set_xlim(0, 12.8)
    n_ff = max(len(ff_items), 1)
    ax_r.set_ylim(-1.8, (n_ff - 1) * _ROW_STEP + 0.8)
    ax_r.invert_yaxis()
    ax_r.axis("off")

    ax_r.text(0.0, -1.1, "Target cell type", fontsize=FONT_LABEL + 1, fontweight="medium", va="center")
    ax_r.text(4.6, -1.1, "Prompt style", fontsize=FONT_LABEL + 1, fontweight="medium",
              va="center")

    # Sanitized, publication-quality display labels for free-form prompt IDs.
    # The raw prompt text is deliberately NOT rendered in the figure to avoid
    # baking internal test strings into the submission manuscript PDF.
    display_labels = {
        "informal cd8": ("CD8 T cell", "Colloquial sentence"),
        "informal macrophage": ("Macrophage", "Colloquial sentence"),
        "informal stem": ("Stem cell", "Colloquial sentence"),
        "informal neuron": ("Neuron", "Colloquial sentence"),
        "verbose treg": ("Treg", "Verbose paragraph"),
        "question style": ("Fibroblast", "Interrogative"),
        "shorthand nk": ("NK cell", "Shorthand list"),
        "expert sergio": ("Expert note", "Free-text note"),
        "clinical note style": ("Clinical", "Clinical note"),
    }

    for i, (prompt_id, _info) in enumerate(ff_items):
        y_pos = i * _ROW_STEP
        key = prompt_id.replace("_", " ")
        label, style_tag = display_labels.get(
            key, (key.title()[:16], "Free-form prompt")
        )
        if len(label) > 16:
            label = label[:14] + "\u2026"
        ax_r.text(0.0, y_pos, label, fontsize=FONT_TICK, va="center",
                  color=COLORS["generated"], fontweight="medium")
        ax_r.text(4.6, y_pos, style_tag, fontsize=FONT_TICK, va="center",
                  color=COLORS["annotation_dark"], style="italic")
        if i < len(ff_items) - 1:
            ax_r.axhline(y=y_pos + _ROW_STEP * 0.5, color=COLORS["border_light"],
                         linewidth=0.5, xmin=0, xmax=1)

    ax_r.set_title("Free-form Prompt Styles", fontsize=FONT_TITLE, fontweight="normal", pad=10)

    # Overall panel label on left sub-panel
    add_panel_label(ax_l, "c", x=-0.08, y=1.08)


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

    # Create figure — height sized for three rows with tighter inter-row gaps
    fig = plt.figure(figsize=(10.0, 7.5))

    # Divide figure into three rows: heatmap (tall), bars (medium), table (medium)
    # Gap increased to prevent cross-row text overlaps at article scale
    layout = bind_figure_region(fig, (0.14, 0.06, 0.96, 0.95))
    row_a, row_b, row_c = layout.split_rows([1.28, 0.94, 0.84], gap=0.12)

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
