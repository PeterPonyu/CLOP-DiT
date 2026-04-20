"""Fig 23: OOD showcase — generation quality for novel/unseen cell types.

Reads ``results/ood_evaluation/ood_results.json`` and renders a showcase
of the model's ability to generalize to out-of-distribution cell types
through structured and free-form prompts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..direct_layout import bind_figure_region
from ..style import (
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


def plot_ood_showcase(
    ood_path: str | Path = "results/ood_evaluation/ood_results.json",
    output_dir: str | Path = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Render OOD evaluation showcase.

    Panel (a): Novel cell types — a summary table showing which cell types
    were tested and their prompt excerpts.

    Panel (b): Free-form prompts — comparison of informal vs. structured
    prompt handling.

    Parameters
    ----------
    ood_path : path to ood_results.json
    output_dir : figure output directory
    dpi : export resolution
    save : whether to save figure
    save_panel_fn : optional callback for custom saving

    Returns
    -------
    fig : Figure or None on error
    """
    apply_style()
    ood_path = Path(ood_path)
    output_dir = Path(output_dir)

    if not ood_path.exists():
        logger.warning("OOD results not found: %s", ood_path)
        return None

    with open(ood_path) as f:
        ood = json.load(f)

    novel_types = ood.get("novel_types", {})
    free_form = ood.get("free_form", {})

    if not novel_types and not free_form:
        logger.warning("No OOD entries found")
        return None

    # Figure with two panels
    fig = plt.figure(figsize=(16.0, 6.5))
    layout = bind_figure_region(fig, (0.04, 0.08, 0.97, 0.80))
    left, right = layout.split_cols([1, 1], wspace=0.45)

    # ── Panel (a): Novel cell types ──
    ax_a = left.add_axes(fig)
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(-1.2, max(len(novel_types), 1) - 0.5)
    ax_a.invert_yaxis()
    ax_a.axis("off")

    # Header
    ax_a.text(0.0, -1.0, "Cell Type", fontsize=FONT_LABEL, fontweight="bold",
              va="center")
    ax_a.text(4.5, -1.0, "Prompt Excerpt", fontsize=FONT_LABEL, fontweight="bold",
              va="center")

    for i, (type_name, info) in enumerate(novel_types.items()):
        prompt = info.get("prompt", "")
        # Truncate prompt for display (keep short to avoid cross-panel spillover)
        excerpt = prompt[:55] + "\u2026" if len(prompt) > 55 else prompt

        color = COLORS["real"]
        ax_a.text(0.0, i, type_name, fontsize=FONT_LABEL - 1, va="center",
                  color=color, fontweight="medium")
        ax_a.text(4.5, i, excerpt, fontsize=FONT_LEGEND_DENSE, va="center",
                  color=COLORS["annotation_dark"], style="italic")
        # Separator line
        if i < len(novel_types) - 1:
            ax_a.axhline(y=i + 0.5, color=COLORS["border_light"], linewidth=0.5,
                         xmin=0, xmax=1)

    add_panel_label(ax_a, "a", x=-0.06, y=1.12)
    ax_a.set_title("Novel Cell Types (Unseen during Training)", fontsize=FONT_TITLE,
                   pad=20)

    # ── Panel (b): Free-form prompts ──
    ax_b = right.add_axes(fig)
    ax_b.set_xlim(0, 10)
    n_ff = max(len(free_form), 1)
    ax_b.set_ylim(-1.2, n_ff - 0.5)
    ax_b.invert_yaxis()
    ax_b.axis("off")

    # Header
    ax_b.text(0.0, -1.0, "Prompt ID", fontsize=FONT_LABEL, fontweight="bold",
              va="center")
    ax_b.text(3.5, -1.0, "Free-form Description", fontsize=FONT_LABEL, fontweight="bold",
              va="center")

    for i, (prompt_id, info) in enumerate(free_form.items()):
        ff_prompt = info.get("free_form_prompt", "")
        excerpt = ff_prompt[:50] + "…" if len(ff_prompt) > 50 else ff_prompt

        display_id = prompt_id.replace("_", " ")
        ax_b.text(0.0, i, display_id, fontsize=FONT_LABEL - 1, va="center",
                  color=COLORS["generated"], fontweight="medium")
        ax_b.text(3.5, i, excerpt, fontsize=FONT_LEGEND_DENSE, va="center",
                  color=COLORS["annotation_dark"], style="italic")
        if i < len(free_form) - 1:
            ax_b.axhline(y=i + 0.5, color=COLORS["border_light"], linewidth=0.5,
                         xmin=0, xmax=1)

    add_panel_label(ax_b, "b", x=-0.06, y=1.12)
    ax_b.set_title("Free-form Prompt Generalization", fontsize=FONT_TITLE, pad=20)

    set_figure_suptitle(fig, "Out-of-Distribution Generation Showcase", y=0.93)

    if save:
        out = output_dir / "fig23_ood_showcase.png"
        if save_panel_fn:
            save_panel_fn(fig, "fig23_ood_showcase")
        else:
            save_with_vcd(fig, out, dpi=dpi)

    return fig
