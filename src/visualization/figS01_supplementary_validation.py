"""Integrated Supplementary Figure S1 composed directly in Python.

This module replaces the manuscript-side LaTeX stitching of three separate PDF
assets with a single Matplotlib-composed appendix figure. The layout preserves
all panels (a--l) while giving the downstream and decoder panels more usable
space and consistent geometry.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.utils.paths import RESULTS_DIR

from .direct_layout import bind_figure_region
from .figS1_robustness_ablation import (
    _draw_ablation_heatmap,
    _draw_multi_seed,
    _draw_ood_showcase,
)
from .figS2_downstream_validation import (
    _panel_a as _downstream_panel_d,
    _panel_b as _downstream_panel_e,
    _panel_c as _downstream_panel_f,
    _panel_d as _downstream_panel_g,
    _panel_e as _downstream_panel_h,
)
from .figS3_expression_decoder import (
    _panel_a as _expr_panel_i,
    _panel_b as _expr_panel_j,
    _panel_c as _expr_panel_k,
    _panel_d as _expr_panel_l,
    _placeholder as _expr_placeholder,
)
from .style import add_panel_label, apply_style, save_with_vcd

logger = logging.getLogger(__name__)


_LABEL_SIZE = 15
_LABEL_Y = 1.08
_BOTTOM_TITLE_SIZE = 14


def _load_expression_decoder_inputs() -> tuple[np.ndarray | None, np.ndarray | None, dict | None, dict, list[str]]:
    """Load the inputs required by panels i--l."""
    real_var: np.ndarray | None = None
    gen_var: np.ndarray | None = None
    aug_data: dict | None = None
    dec_metrics: dict = {}
    dec_approaches: list[str] = []

    try:
        real_expr = np.load(RESULTS_DIR / "real_expression.npy")
        gen_expr = np.load(RESULTS_DIR / "generated_expression.npy")
        real_var = np.var(real_expr, axis=0)
        gen_var = np.var(gen_expr, axis=0)
    except FileNotFoundError:
        logger.warning("Expression arrays not found; panels i/j will use placeholders")

    aug_path = RESULTS_DIR / "downstream" / "embedding_augmentation.json"
    if aug_path.exists():
        try:
            with open(aug_path) as f:
                aug_data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load augmentation data: %s", exc)

    ablation_dir = RESULTS_DIR / "ablations" / "decoder"
    for name in ["baseline", "lora_light", "mlp"]:
        mpath = ablation_dir / name / "metrics.json"
        if not mpath.exists():
            continue
        try:
            with open(mpath) as f:
                dec_metrics[name] = json.load(f)
            dec_approaches.append(name)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load decoder metrics for %s: %s", name, exc)

    return real_var, gen_var, aug_data, dec_metrics, dec_approaches



def plot_supplementary_validation(
    output_dir: str | Path = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Render the integrated Supplementary Figure S1."""
    apply_style()
    output_dir = Path(output_dir)
    results_dir = Path(RESULTS_DIR)

    ablation_path = Path("results/ablations/all_summaries.json")
    seed_path = Path("results/multi_seed/multi_seed_report.json")
    ood_path = Path("results/ood_evaluation/ood_results.json")
    real_var, gen_var, aug_data, dec_metrics, dec_approaches = _load_expression_decoder_inputs()

    fig = plt.figure(figsize=(14.0, 15.5))
    layout = bind_figure_region(fig, (0.06, 0.02, 0.96, 0.98))

    # 5 explicit rows — panel a gets extra height; last gap wider for xticklabels
    row_a, row_b, row_c, row_4, row_5 = layout.split_rows(
        [1.45, 0.55, 0.82, 0.82, 0.82], gap=[0.04, 0.04, 0.04, 0.09],
    )

    # Row 1: panel a (ablation heatmap — tall for many y-labels)
    _draw_ablation_heatmap(fig, row_a, ablation_path)

    # Row 2: panel b (multi-seed bars)
    _draw_multi_seed(fig, row_b, seed_path)

    # Row 3: panel c (OOD table)
    _draw_ood_showcase(fig, row_c, ood_path)

    # Row 4: d, e, f, i, j — variable gaps: wider before e (heatmap yticklabels)
    #   and before f (barh yticklabels) to prevent masking neighbours
    _lbl_x = -0.12
    d_r, e_r, f_r, i_r, j_r = row_4.split_cols(
        [1.05, 1.00, 1.05, 1.00, 1.00], gap=[0.18, 0.08, 0.08, 0.05],
    )

    _row4_lbl_y = 1.14  # row 4 labels higher to clear titles

    ax_d = d_r.add_axes(fig)
    _downstream_panel_d(ax_d, results_dir)
    add_panel_label(ax_d, "d", x=_lbl_x, y=_row4_lbl_y, fontsize=_LABEL_SIZE)

    ax_e = e_r.add_axes(fig)
    _downstream_panel_e(fig, ax_e, results_dir)
    add_panel_label(ax_e, "e", x=_lbl_x, y=_row4_lbl_y, fontsize=_LABEL_SIZE)

    ax_f = f_r.add_axes(fig)
    _downstream_panel_f(ax_f, results_dir)
    add_panel_label(ax_f, "f", x=_lbl_x, y=_row4_lbl_y, fontsize=_LABEL_SIZE)

    ax_i = i_r.add_axes(fig)
    if real_var is not None and gen_var is not None:
        _expr_panel_i(ax_i, real_var, gen_var)
        add_panel_label(ax_i, "i", x=_lbl_x, y=_row4_lbl_y, fontsize=_LABEL_SIZE)
    else:
        _expr_placeholder(ax_i, "Variance Scatter", "i", label_fontsize=_LABEL_SIZE)

    ax_j = j_r.add_axes(fig)
    if real_var is not None and gen_var is not None:
        _expr_panel_j(ax_j, real_var, gen_var)
        add_panel_label(ax_j, "j", x=_lbl_x, y=_row4_lbl_y, fontsize=_LABEL_SIZE)
    else:
        _expr_placeholder(ax_j, "Variance Ratio", "j", label_fontsize=_LABEL_SIZE)

    # Row 5: g, h, k, l — 4 panels across full width
    g_r, h_r, k_r, l_r = row_5.split_cols(
        [1.00, 1.50, 1.00, 1.00], gap=0.08,
    )

    ax_g = g_r.add_axes(fig)
    _downstream_panel_g(ax_g, results_dir)
    add_panel_label(ax_g, "g", x=_lbl_x, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    # Use full region — no inset shrink — so radar fills the space
    ax_h = _downstream_panel_h(fig, list(h_r.as_tuple()), results_dir)
    add_panel_label(ax_h, "h", x=_lbl_x, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    ax_k = k_r.add_axes(fig)
    if aug_data is not None:
        _expr_panel_k(ax_k, aug_data)
        add_panel_label(ax_k, "k", x=_lbl_x, y=_LABEL_Y, fontsize=_LABEL_SIZE)
    else:
        _expr_placeholder(ax_k, "Embedding Augmentation", "k", label_fontsize=_LABEL_SIZE)

    ax_l = l_r.add_axes(fig)
    if len(dec_approaches) >= 2:
        _expr_panel_l(ax_l, dec_metrics, dec_approaches)
        add_panel_label(ax_l, "l", x=_lbl_x, y=_LABEL_Y, fontsize=_LABEL_SIZE)
    else:
        _expr_placeholder(ax_l, "Decoder Comparison", "l", label_fontsize=_LABEL_SIZE)

    # Harmonize title size / pad for bottom panels
    for ax in [ax_d, ax_e, ax_f, ax_g, ax_i, ax_j, ax_k, ax_l]:
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=_BOTTOM_TITLE_SIZE, fontweight="normal", pad=4)
    if ax_h.get_title():
        ax_h.set_title(ax_h.get_title(), fontsize=_BOTTOM_TITLE_SIZE, fontweight="normal", pad=6)

    if save:
        stem = output_dir / "figS01_supplementary_validation"
        if save_panel_fn is not None:
            save_panel_fn(fig, stem)
        else:
            save_with_vcd(fig, stem, dpi=dpi)
        logger.info("Saved %s", stem)

    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    logging.basicConfig(level=logging.INFO)
    plot_supplementary_validation()
