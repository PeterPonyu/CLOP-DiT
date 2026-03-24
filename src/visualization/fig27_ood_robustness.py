"""
fig27_ood_robustness.py — Fig 27: Quantitative OOD robustness evaluation.

Three-row panel summarising CLOP-DiT's out-of-distribution handling:
  (a) Marker hit rate by OOD category (novel / free-form / cross-tissue)
  (b) Embedding coherence (intra-sample cosine similarity)
  (c) Summary statistics table
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
    add_panel_label,
    apply_style,
    save_panel,
    style_axes,
)
from src.utils.paths import RESULTS_DIR

logger = logging.getLogger(__name__)


def plot_ood_robustness(
    data_path: str | Path = "results/downstream/ood_robustness_combined.json",
    marker_path: str | Path = "results/ood_evaluation/ood_marker_analysis.json",
    output_dir: str | Path = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Fig 27: Quantitative OOD robustness evaluation.

    Panel (a): Stacked bar — marker hit rate per prompt, colour-coded by category.
    Panel (b): Dot plot — embedding coherence per prompt.
    Panel (c): Category-level summary bars.
    """
    apply_style()
    data_path = Path(data_path)
    marker_path = Path(marker_path)
    output_dir = Path(output_dir)

    # Try loading from combined results first, fall back to marker analysis
    marker_data = {}
    if marker_path.exists():
        with open(marker_path) as f:
            marker_data = json.load(f)
    elif data_path.exists():
        with open(data_path) as f:
            marker_data = json.load(f)

    if not marker_data:
        logger.warning("No OOD data found")
        return None

    # Collect per-prompt metrics from all categories
    categories = ["novel_types", "free_form", "cross_tissue"]
    cat_colors = {
        "novel_types": COLORS["real"],
        "free_form": COLORS["generated"],
        "cross_tissue": COLORS["accent"],
    }
    cat_labels = {
        "novel_types": "Novel Types",
        "free_form": "Free-Form",
        "cross_tissue": "Cross-Tissue",
    }

    all_prompts = []
    all_hit_rates = []
    all_coherence = []
    all_categories = []
    all_n_vocab = []

    for cat in categories:
        cat_data = marker_data.get(cat, {})
        # Gracefully handle cases where category data is a list instead of a dict
        if not isinstance(cat_data, dict):
            continue
        for name, info in cat_data.items():
            if not isinstance(info, dict):
                continue
            # Skip entries with no actual evaluation data
            if "marker_hit_rate" not in info and "markers_in_vocabulary" not in info:
                continue
            all_prompts.append(name)
            all_hit_rates.append(float(info.get("marker_hit_rate", 0) or 0))
            all_coherence.append(float(info.get("intra_sample_cosine_sim", 0) or 0))
            all_categories.append(cat)
            all_n_vocab.append(int(info.get("markers_in_vocabulary", 0) or 0))

    n_prompts = len(all_prompts)
    if n_prompts == 0:
        logger.warning("No OOD prompts found in data")
        return None

    # Count how many free-form were skipped
    n_ff_skipped = len(marker_data.get("free_form", {})) - sum(1 for c in all_categories if c == "free_form")

    # Layout
    fig = plt.figure(figsize=(15.0, 6.5))
    layout = bind_figure_region(fig, (0.07, 0.15, 0.97, 0.90))
    p_a, p_b, p_c = layout.split_cols([1.3, 0.8, 0.7], gap=0.06)

    x = np.arange(n_prompts)

    # ── Panel (a): Marker hit rate per prompt ──
    ax_a = p_a.add_axes(fig)
    add_panel_label(ax_a, "a", x=-0.08, y=1.06)

    bar_colors = [cat_colors.get(c, COLORS["neutral"]) for c in all_categories]
    bars = ax_a.bar(x, all_hit_rates, color=bar_colors, alpha=0.85,
                    edgecolor="white", linewidth=0.5)

    ax_a.set_xticks(x)
    short_names = []
    for name in all_prompts:
        # Shorten prompt names for display
        short = name.replace("_", " ")
        if len(short) > 14:
            short = short[:12] + "…"
        short_names.append(short)
    ax_a.set_xticklabels(short_names, fontsize=FONT_TICK_DENSE - 1,
                         rotation=45, ha="right")
    ax_a.set_ylabel("Marker Hit Rate", fontsize=FONT_LABEL)
    ax_a.set_title("Marker Recovery by OOD Prompt", fontsize=FONT_TITLE)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Add legend for categories
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=cat_colors[c], label=cat_labels[c], alpha=0.85)
                      for c in categories if any(cc == c for cc in all_categories)]
    ax_a.legend(handles=legend_patches, fontsize=FONT_LEGEND - 1,
                loc="upper right", frameon=False)
    style_axes(ax_a)

    if n_ff_skipped > 0:
        ax_a.text(0.98, 0.95, f"Free-form prompts (n={n_ff_skipped}): no marker ground truth",
                  transform=ax_a.transAxes, ha="right", va="top",
                  fontsize=FONT_ANNOTATION - 1, style="italic", color=COLORS["neutral"])

    # Annotate bars with hit count / markers in vocab
    for i, (bar, n_v, hr) in enumerate(zip(bars, all_n_vocab, all_hit_rates)):
        h = bar.get_height()
        n_hit = round(hr * n_v) if n_v > 0 else 0
        ax_a.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                  f"{n_hit}/{n_v}", ha="center", va="bottom",
                  fontsize=FONT_ANNOTATION - 1, color=COLORS["annotation_dark"])

    # ── Panel (b): Embedding coherence ──
    ax_b = p_b.add_axes(fig)
    add_panel_label(ax_b, "b", x=-0.15, y=1.06)

    dot_colors = [cat_colors.get(c, COLORS["neutral"]) for c in all_categories]
    ax_b.scatter(all_coherence, x, c=dot_colors, s=50, alpha=0.8,
                 edgecolors="white", linewidth=0.5, zorder=3)

    ax_b.set_yticks(x)
    ax_b.set_yticklabels(short_names, fontsize=FONT_TICK_DENSE - 1)
    ax_b.set_xlabel("Cosine Similarity", fontsize=FONT_LABEL)
    ax_b.set_title("Embedding Coherence", fontsize=FONT_TITLE)
    # Zoom x-axis to meaningful range so dots aren't compressed into a sliver
    if all_coherence:
        coh_min = min(all_coherence)
        coh_pad = max(0.05, (1.0 - coh_min) * 0.15)
        ax_b.set_xlim(max(0, coh_min - coh_pad), 1.02)
    else:
        ax_b.set_xlim(0, 1.05)
    ax_b.invert_yaxis()
    style_axes(ax_b)

    # ── Panel (c): Category summary ──
    ax_c = p_c.add_axes(fig)
    add_panel_label(ax_c, "c", x=-0.18, y=1.06)

    cat_means = []
    cat_ns = []
    cat_with_hits = []
    for cat in categories:
        cat_hrs = [h for h, c in zip(all_hit_rates, all_categories) if c == cat]
        if cat_hrs:
            cat_means.append(np.mean(cat_hrs))
            cat_ns.append(len(cat_hrs))
            cat_with_hits.append(sum(1 for h in cat_hrs if h > 0))
        else:
            cat_means.append(0)
            cat_ns.append(0)
            cat_with_hits.append(0)

    y_cat = np.arange(len(categories))
    cat_bar_colors = [cat_colors[c] for c in categories]
    bars_c = ax_c.barh(y_cat, cat_means, height=0.5, color=cat_bar_colors,
                       alpha=0.85, edgecolor="white", linewidth=0.5)

    ax_c.set_yticks(y_cat)
    ax_c.set_yticklabels([cat_labels[c] for c in categories], fontsize=FONT_TICK)
    ax_c.set_xlabel("Mean Hit Rate", fontsize=FONT_LABEL)
    ax_c.set_title("Category Summary", fontsize=FONT_TITLE)
    ax_c.set_xlim(0, 1.0)
    ax_c.invert_yaxis()
    style_axes(ax_c)

    for i, (v, n, nh) in enumerate(zip(cat_means, cat_ns, cat_with_hits)):
        if n == 0:
            ax_c.text(0.02, i, "N/A (no marker ground truth)",
                      ha="left", va="center", fontsize=FONT_ANNOTATION,
                      style="italic", color=COLORS["neutral"])
        else:
            ax_c.text(v + 0.02, i, f"{v:.2f} ({nh}/{n})",
                      ha="left", va="center", fontsize=FONT_ANNOTATION)

    # Save
    if save:
        if save_panel_fn:
            save_panel_fn(fig, "fig27_ood_robustness")
        else:
            out = output_dir / "fig27_ood_robustness.png"
            save_panel(fig, out, dpi=dpi)
        logger.info("Saved Fig 27 → %s", output_dir)

    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_ood_robustness()
