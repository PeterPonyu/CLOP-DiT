"""
fig29_embedding_augmentation.py — Fig 29: Embedding-level augmentation analysis.

Shows whether augmenting rare cell type training data with generated embeddings
(rather than decoded expression profiles) improves downstream classification.
This tests the hypothesis that the 512-d embedding space is better preserved
than gene-expression space through the scGPT decoder.

Key finding: synthetic embedding augmentation yields marginal improvement
(delta F1 ~ +0.001) even at embedding level, confirming the negative result.
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
    FONT_LEGEND,
    FONT_TICK,
    FONT_TICK_DENSE,
    FONT_TITLE,
    FONT_SMALL,
    add_panel_label,
    apply_style,
    save_with_vcd,
    style_axes,
)
from src.utils.paths import RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)


def plot_embedding_augmentation(
    data_path: str | Path = "results/downstream/embedding_augmentation.json",
    output_dir: str | Path = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Fig 29: Embedding-level augmentation of rare cell types.

    Panel (a): Dot + whisker — baseline vs. augmented macro-F1 at 1x/5x/10x ratios.
    Panel (b): Delta F1 bars — incremental F1 gain from augmentation.
    Panel (c): Accuracy delta bars — similar analysis for classification accuracy.
    """
    apply_style()
    data_path = Path(data_path)
    output_dir = Path(output_dir)

    if not data_path.exists():
        logger.warning("Embedding augmentation results not found: %s", data_path)
        return None

    with open(data_path) as f:
        data = json.load(f)

    ratios, baseline_f1, augmented_f1 = [], [], []
    baseline_f1_std, augmented_f1_std = [], []
    delta_f1, baseline_acc, augmented_acc = [], [], []

    for key in ["ratio_1x", "ratio_5x", "ratio_10x"]:
        if key not in data:
            continue
        label = key.replace("ratio_", "").replace("x", "x aug")
        ratios.append(label)
        d = data[key]
        baseline_f1.append(d.get("baseline_f1_mean", 0))
        augmented_f1.append(d.get("augmented_f1_mean", 0))
        baseline_f1_std.append(d.get("baseline_f1_std", 0))
        augmented_f1_std.append(d.get("augmented_f1_std", 0))
        delta_f1.append(d.get("delta_f1", 0))
        baseline_acc.append(d.get("baseline_acc_mean", 0))
        augmented_acc.append(d.get("augmented_acc_mean", 0))

    if not ratios:
        logger.warning("No augmentation ratio data found — skipping Fig 29")
        return None

    n_rats = len(ratios)
    rare_threshold = data.get("rare_threshold", "?")
    n_rare = data.get("n_rare_types", "?")

    fig = plt.figure(figsize=(14.0, 5.2))
    layout = bind_figure_region(fig, (0.08, 0.14, 0.97, 0.88))
    p_a, p_b, p_c = layout.split_cols([1.2, 0.9, 0.9], gap=0.07)

    x = np.arange(n_rats)

    # ── Panel (a): Dot plot — baseline vs. augmented F1 ──
    ax_a = p_a.add_axes(fig)
    add_panel_label(ax_a, "a", x=-0.12, y=1.06)

    # Compute y range with generous padding so annotations fit
    all_f1 = baseline_f1 + augmented_f1
    all_std = baseline_f1_std + augmented_f1_std
    y_min = min(f - s for f, s in zip(all_f1, all_std)) - 0.003
    y_max = max(f + s for f, s in zip(all_f1, all_std)) + 0.006

    # Draw connecting lines first (grey)
    for i in range(n_rats):
        ax_a.plot([i - 0.15, i + 0.15], [baseline_f1[i], augmented_f1[i]],
                  color="grey", lw=0.8, alpha=0.5, zorder=1)

    # Error bars + dots for baseline
    ax_a.errorbar(x - 0.15, baseline_f1, yerr=baseline_f1_std,
                  fmt="o", markersize=7, color=COLORS["real"], capsize=4,
                  linewidth=1.0, elinewidth=0.8, label="Baseline", zorder=3)
    # Error bars + dots for augmented
    ax_a.errorbar(x + 0.15, augmented_f1, yerr=augmented_f1_std,
                  fmt="s", markersize=7, color=COLORS["generated"], capsize=4,
                  linewidth=1.0, elinewidth=0.8, label="Augmented", zorder=3)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(ratios, fontsize=FONT_TICK)
    ax_a.set_ylabel("Macro-F1 (5-fold CV)", fontsize=FONT_LABEL)
    ax_a.set_title("F1 Score: Baseline vs. Augmented", fontsize=FONT_TITLE)
    ax_a.set_ylim(y_min, y_max)
    ax_a.legend(fontsize=FONT_LEGEND - 1, loc="lower right", frameon=False)
    ax_a.text(0.03, 0.05,
              f"n_rare={n_rare} types (<={rare_threshold} cells)",
              transform=ax_a.transAxes,
              fontsize=FONT_SMALL, color=COLORS["neutral"], style="italic")
    style_axes(ax_a, kind="scatter")

    # ── Panel (b): Delta F1 bars ──
    ax_b = p_b.add_axes(fig)
    add_panel_label(ax_b, "b", x=-0.18, y=1.06)

    delta_colors = [COLORS["good"] if d >= 0 else COLORS["bad"] for d in delta_f1]
    ax_b.bar(x, delta_f1, color=delta_colors, alpha=0.85,
             edgecolor="white", linewidth=0.5)
    ax_b.axhline(0, color=COLORS["neutral"], lw=1.0, alpha=0.6, zorder=3)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(ratios, fontsize=FONT_TICK_DENSE, rotation=15, ha="right")
    ax_b.set_ylabel("delta F1 (aug - base)", fontsize=FONT_LABEL)
    ax_b.set_title("F1 Gain from Augmentation", fontsize=FONT_TITLE)
    style_axes(ax_b)

    _d_range = max(abs(d) for d in delta_f1) * 1.6 or 0.005
    ax_b.set_ylim(-_d_range, _d_range)

    for i, d in enumerate(delta_f1):
        va = "bottom" if d >= 0 else "top"
        offset = _d_range * 0.04 if d >= 0 else -_d_range * 0.04
        ax_b.text(i, d + offset, f"{d:+.4f}",
                  ha="center", va=va, fontsize=FONT_ANNOTATION)

    # Annotate if 5x and 10x deltas are identical (saturated)
    if len(delta_f1) >= 3 and abs(delta_f1[1] - delta_f1[2]) < 1e-6:
        mid_x = (x[1] + x[2]) / 2
        mid_val = delta_f1[1]
        y_ann = mid_val + _d_range * 0.25 if mid_val >= 0 else mid_val - _d_range * 0.25
        ax_b.annotate("identical\n(saturated)",
                       xy=(mid_x, y_ann),
                       fontsize=FONT_ANNOTATION - 1, ha="center",
                       style="italic", color=COLORS["neutral"])

    # ── Panel (c): Accuracy delta ──
    ax_c = p_c.add_axes(fig)
    add_panel_label(ax_c, "c", x=-0.18, y=1.06)

    delta_acc = [a - b for a, b in zip(augmented_acc, baseline_acc)]
    acc_colors = [COLORS["good"] if d >= 0 else COLORS["bad"] for d in delta_acc]
    ax_c.bar(x, delta_acc, color=acc_colors, alpha=0.85,
             edgecolor="white", linewidth=0.5)
    ax_c.axhline(0, color=COLORS["neutral"], lw=1.0, alpha=0.6, zorder=3)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(ratios, fontsize=FONT_TICK_DENSE, rotation=15, ha="right")
    ax_c.set_ylabel("delta Accuracy (aug - base)", fontsize=FONT_LABEL)
    ax_c.set_title("Accuracy Gain from Augmentation", fontsize=FONT_TITLE)
    style_axes(ax_c)

    _da_range = max(abs(d) for d in delta_acc) * 1.6 or 0.005
    ax_c.set_ylim(-_da_range, _da_range)

    for i, d in enumerate(delta_acc):
        va = "bottom" if d >= 0 else "top"
        offset = _da_range * 0.04 if d >= 0 else -_da_range * 0.04
        ax_c.text(i, d + offset, f"{d:+.4f}",
                  ha="center", va=va, fontsize=FONT_ANNOTATION)

    # Annotate if 5x and 10x accuracy deltas are identical (saturated)
    if len(delta_acc) >= 3 and abs(delta_acc[1] - delta_acc[2]) < 1e-6:
        mid_x = (x[1] + x[2]) / 2
        mid_val = delta_acc[1]
        y_ann = mid_val + _da_range * 0.25 if mid_val >= 0 else mid_val - _da_range * 0.25
        ax_c.annotate("identical\n(saturated)",
                       xy=(mid_x, y_ann),
                       fontsize=FONT_ANNOTATION - 1, ha="center",
                       style="italic", color=COLORS["neutral"])

    # Save
    if save:
        out = output_dir / "fig29_embedding_augmentation.png"
        if save_panel_fn:
            save_panel_fn(fig, "fig29_embedding_augmentation")
        else:
            save_with_vcd(fig, out, dpi=dpi)
        logger.info("Saved Fig 29 → %s", output_dir)

    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_embedding_augmentation()
