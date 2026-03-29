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


_LABEL_SIZE = 14
_LABEL_Y = 1.05
_BOTTOM_TITLE_SIZE = 15


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

    fig = plt.figure(figsize=(11.5, 11.8))
    layout = bind_figure_region(fig, (0.05, 0.035, 0.985, 0.975))

    top, bottom = layout.split_rows([1.18, 1.12], gap=0.08)

    # Top block: panels a--c (robustness / ablation / OOD)
    row_a, row_b, row_c = top.split_rows([1.22, 0.80, 1.00], gap=0.10)
    _draw_ablation_heatmap(fig, row_a, ablation_path)
    _draw_multi_seed(fig, row_b, seed_path)
    _draw_ood_showcase(fig, row_c, ood_path)

    # Bottom block: panels d--h on the left, i--l on the right
    left, right = bottom.split_cols([1.25, 1.15], gap=0.08)

    left_top, left_bottom = left.split_rows([1.0, 1.0], gap=0.12)
    d_region, e_region, f_region = left_top.split_cols([1.08, 1.06, 1.10], gap=0.06)
    g_region, h_region = left_bottom.split_cols([1.0, 1.18], gap=0.08)

    ax_d = d_region.add_axes(fig)
    _downstream_panel_d(ax_d, results_dir)
    add_panel_label(ax_d, "d", x=-0.12, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    ax_e = e_region.add_axes(fig)
    _downstream_panel_e(fig, ax_e, results_dir)
    add_panel_label(ax_e, "e", x=-0.12, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    ax_f = f_region.add_axes(fig)
    _downstream_panel_f(ax_f, results_dir)
    add_panel_label(ax_f, "f", x=-0.10, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    ax_g = g_region.add_axes(fig)
    _downstream_panel_g(ax_g, results_dir)
    add_panel_label(ax_g, "g", x=-0.12, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    radar_region = h_region.inset(left=0.02, right=0.00, bottom=0.02, top=0.01)
    ax_h = _downstream_panel_h(fig, list(radar_region.as_tuple()), results_dir)
    add_panel_label(ax_h, "h", x=-0.08, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    right_top, right_bottom = right.split_rows([1.0, 1.0], gap=0.12)
    i_region, j_region = right_top.split_cols([1.0, 1.0], gap=0.08)
    k_region, l_region = right_bottom.split_cols([1.0, 1.0], gap=0.08)

    ax_i = i_region.add_axes(fig)
    if real_var is not None and gen_var is not None:
        _expr_panel_i(ax_i, real_var, gen_var)
        add_panel_label(ax_i, "i", x=-0.10, y=_LABEL_Y, fontsize=_LABEL_SIZE)
    else:
        _expr_placeholder(ax_i, "Variance Scatter", "i", label_fontsize=_LABEL_SIZE)

    ax_j = j_region.add_axes(fig)
    if real_var is not None and gen_var is not None:
        _expr_panel_j(ax_j, real_var, gen_var)
        add_panel_label(ax_j, "j", x=-0.10, y=_LABEL_Y, fontsize=_LABEL_SIZE)
    else:
        _expr_placeholder(ax_j, "Variance Ratio", "j", label_fontsize=_LABEL_SIZE)

    ax_k = k_region.add_axes(fig)
    if aug_data is not None:
        _expr_panel_k(ax_k, aug_data)
        add_panel_label(ax_k, "k", x=-0.10, y=_LABEL_Y, fontsize=_LABEL_SIZE)
    else:
        _expr_placeholder(ax_k, "Embedding Augmentation", "k", label_fontsize=_LABEL_SIZE)

    ax_l = l_region.add_axes(fig)
    if len(dec_approaches) >= 2:
        _expr_panel_l(ax_l, dec_metrics, dec_approaches)
        add_panel_label(ax_l, "l", x=-0.10, y=_LABEL_Y, fontsize=_LABEL_SIZE)
    else:
        _expr_placeholder(ax_l, "Decoder Comparison", "l", label_fontsize=_LABEL_SIZE)

    for ax in [ax_d, ax_e, ax_f, ax_g, ax_i, ax_j, ax_k, ax_l]:
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=_BOTTOM_TITLE_SIZE, fontweight="normal", pad=8)
    if ax_h.get_title():
        ax_h.set_title(ax_h.get_title(), fontsize=_BOTTOM_TITLE_SIZE, fontweight="normal", pad=12)

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
