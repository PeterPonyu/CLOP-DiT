"""
fig12_diversity.py — Article Figure 12: diversity diagnostics.

Extracted from panels_diversity.py (formerly Panel J / plot_diagnostics).
Plotting only; scripts run diagnostics and pass all_results from diversity_diagnostics.json.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .direct_layout import bind_figure_region
from .explicit_positioning import add_shared_legend_axes
from .style import COLORS, FONT_DENSE_YTICK, apply_style, save_with_vcd, add_panel_label, abbreviate_cell_type
from .fig14_expr_diversity import plot_expression_diversity_panel
from src.utils.paths import load_thresholds

logger = logging.getLogger(__name__)

_viz_thresh = load_thresholds().get("visualization", {})
_DIVERSITY_BANDS = _viz_thresh.get("diversity_ratio_bands", [0.5, 0.8, 1.2])
_MEMORIZATION_THRESH = _viz_thresh.get("memorization_threshold", 0.01)


def plot_diagnostics(
    all_results: Dict,
    output_dir: str = "results/figures",
    dpi: int = 300,
    type_names: Optional[Dict[int, str]] = None,
    include_noise_panel: bool = True,
    noise_data: Optional[Dict] = None,
) -> List[Path]:
    """Generate diversity diagnostics figure and expression diversity figure from test results.

    When *include_noise_panel* is True (default), the noise-tradeoff panel from
    fig13 is appended as a 5th panel (e) in the diagnostics figure, producing a
    combined ``fig06_diversity_diagnostics.pdf`` suitable for the merged article
    layout.  The expression diversity figure (fig14) is always generated
    separately with panels starting at label 'a'.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    # Auto-load type names from captions file if not provided
    if type_names is None:
        import json as _json
        _cap_candidates = [
            Path("data/cached_latents/text_captions_deduplicated.json"),
        ]
        for _cp in _cap_candidates:
            if _cp.exists():
                try:
                    with open(_cp) as _cf:
                        _raw = _json.load(_cf)
                    type_names = {}
                    for k, v in _raw.items():
                        name = v.split(" are ")[0] if " are " in v else v[:40]
                        type_names[int(k)] = name
                except Exception:
                    type_names = {}
                break
        if type_names is None:
            type_names = {}

    apply_style()

    # ── Figure 12: Diversity Diagnostics (4 subplots + optional noise panel) ──
    if include_noise_panel:
        fig = plt.figure(figsize=(14.0, 9.8))
        outer_layout = bind_figure_region(fig, (0.12, 0.08, 0.988, 0.93))
        diag_region, noise_region = outer_layout.split_rows([1.0, 0.48], hspace=0.28)
    else:
        fig = plt.figure(figsize=(14.0, 6.8))
        diag_region = bind_figure_region(fig, (0.12, 0.12, 0.988, 0.93))
    top_row, bottom_row = diag_region.split_rows(2, hspace=0.42)
    top_left, top_right = top_row.split_cols([1.00, 1.00], gap=0.048)
    bottom_left, bottom_right = bottom_row.split_cols([0.92, 1.02], gap=0.074)
    top_left = top_left.inset(right=0.006)
    top_right = top_right.inset(left=0.006)
    bottom_left = bottom_left.inset(right=0.030, bottom=-0.04)
    bottom_right = bottom_right.inset(left=0.024)
    axes = np.array([
        [top_left.add_axes(fig), top_right.add_axes(fig)],
        [bottom_left.add_axes(fig), bottom_right.add_axes(fig)],
    ], dtype=object)
    add_panel_label(axes[0, 0], 'a', x=-0.08, y=1.02)
    add_panel_label(axes[0, 1], 'b', x=-0.08, y=1.02)
    add_panel_label(axes[1, 0], 'c', x=-0.08, y=1.02)
    add_panel_label(axes[1, 1], 'd', x=-0.08, y=1.02)

    ax = axes[0, 0]
    t1 = all_results.get("test1_intratype_diversity", {}).get("per_type", {})
    if t1:
        names = list(t1.keys())
        div_ratios = [t1[n]["diversity_ratio"] for n in names]
        sorted_idx = np.argsort(div_ratios)
        sorted_names = [abbreviate_cell_type(names[i], 14) for i in sorted_idx]
        sorted_divs = [div_ratios[i] for i in sorted_idx]

        colors = [COLORS["bad"] if d < _DIVERSITY_BANDS[0] else COLORS["warn"] if d < _DIVERSITY_BANDS[1] else COLORS["good"] if d < _DIVERSITY_BANDS[2] else COLORS["real"]
                  for d in sorted_divs]
        ax.barh(range(len(sorted_divs)), sorted_divs, color=colors, height=0.8)
        ax.set_yticks(range(len(sorted_divs)))
        _step_j1 = max(2, int(np.ceil(len(sorted_divs) / 9)))
        _ytl_j1 = [n if i % _step_j1 == 0 else "" for i, n in enumerate(sorted_names)]
        ax.set_yticklabels(_ytl_j1, fontsize=8)
        ax.axvline(x=1.0, color="black", ls="--", lw=1, alpha=0.5, label="ratio=1 (equal)")
        ax.axvline(x=_DIVERSITY_BANDS[0], color="red", ls=":", lw=1, alpha=0.5, label=f"ratio={_DIVERSITY_BANDS[0]} (collapse)")
        ax.set_xlabel("Diversity Ratio (gen / real)")
        summary = all_results["test1_intratype_diversity"]["summary"]
        ax.set_title("Intra-Type Diversity Ratio")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Intra-Type Diversity Ratio")

    ax = axes[0, 1]
    t2 = all_results.get("test2_memorization", {})
    if t2:
        nn = t2["nn_cosine_distance"]
        labels = ["p5", "p25", "median", "mean", "p75", "p95"]
        vals = [nn["p5"], nn["p25"], nn["median"], nn["mean"], nn["p75"], nn["p95"]]
        ax.bar(labels, vals, color=[COLORS["bad"], COLORS["warn"], COLORS["good"], COLORS["real"], COLORS["warn"], COLORS["bad"]],
               alpha=0.8, edgecolor="white")
        ax.set_ylabel("Cosine Distance to Nearest Real Cell")
        ax.axhline(y=_MEMORIZATION_THRESH, color="red", ls=":", alpha=0.5, label="memorization threshold")
        ax.set_title("Nearest-Neighbour Distance")
    else:
        ax.set_title("Nearest-Neighbour Distance")

    ax = axes[1, 0]
    t3 = all_results.get("test3_cfg_sweep", {})
    if t3:
        cfg_vals = sorted(t3.keys(), key=lambda k: t3[k]["cfg_scale"])
        scales = [t3[k]["cfg_scale"] for k in cfg_vals]
        divs = [t3[k]["mean_intra_diversity"] for k in cfg_vals]
        norms = [t3[k]["mean_norm"] for k in cfg_vals]

        color = COLORS["real"]
        ax.plot(scales, divs, "o-", color=color, lw=2, markersize=8, label="Div.")
        ax.set_xlabel("CFG Scale")
        ax.set_ylabel("Mean Intra-Type Diversity", color=color)
        ax.tick_params(axis="y", labelcolor=color)

        ax2 = ax.twinx()
        color2 = COLORS["generated"]
        ax2.plot(scales, norms, "s--", color=color2, lw=2, markersize=8, label="Norm")
        ax2.set_ylabel("Mean Norm", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax2.yaxis.labelpad = 10
        ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune="upper"))

        ax.set_title("CFG Scale vs Diversity & Norm")
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune="both"))
        ax.text(0.03, 0.05, "● Div.", transform=ax.transAxes,
            fontsize=9, color=COLORS["real"], ha="left", va="bottom")
        ax.text(0.03, 0.13, "■ Norm", transform=ax.transAxes,
            fontsize=9, color=COLORS["generated"], ha="left", va="bottom")
    else:
        ax.set_title("CFG Scale vs Diversity")

    ax = axes[1, 1]
    t5 = all_results.get("test5_condition_sensitivity", {})
    if t5:
        per_type = t5.get("per_type", {})
        type_ids = sorted(per_type.keys(), key=lambda k: per_type[k]["diversity_gain"])
        cent_divs = [per_type[k]["centroid_diversity"] for k in type_ids]
        noise_divs = [per_type[k]["noise_diversity"] for k in type_ids]

        x = np.arange(len(type_ids))
        w = 0.35
        ax.bar(x - w / 2, cent_divs, w, label="Centroid Cond", color=COLORS["real"], alpha=0.8)
        ax.bar(x + w / 2, noise_divs, w, label="Centroid + Noise", color=COLORS["generated"], alpha=0.8)
        ax.set_xticks(x)
        _type_labels_d = [abbreviate_cell_type(type_names.get(int(k), f"Type {k}"), 12)
                          for k in type_ids]
        _displayed_labels = [label if i % 2 == 0 else "" for i, label in enumerate(_type_labels_d)]
        ax.set_xticklabels(_displayed_labels, fontsize=8, rotation=45, ha="right")
        ax.set_xlabel("Cell Type")
        ax.set_ylabel("Intra-Type Div.")
        gain = t5["summary"]["mean_diversity_gain"]
        eps = t5["summary"].get("noise_scale", "?")
        ax.set_title("Centroid vs Noisy Conditioning")
    else:
        ax.set_title("Centroid vs Noisy Conditioning")

    # Collect unique legend handles from all axes (including twinx) into a single figure legend
    _seen_labels = set()
    _handles, _labels = [], []
    for _ax in fig.axes:
        for _h, _l in zip(*_ax.get_legend_handles_labels()):
            if _l not in _seen_labels:
                _seen_labels.add(_l)
                _handles.append(_h)
                _labels.append(_l)
    # Shared legend removed: the remaining labels are self-explanatory and the
    # legend occupied more visual space than the signal it provided.

    # ── Optional noise tradeoff panel (e) ──
    if include_noise_panel and noise_data:
        from .fig13_noise_tradeoff import plot_panel_l
        noise_plot_region, noise_legend_region = noise_region.split_cols([1.0, 0.34], wspace=0.08)
        legend_ax = noise_legend_region.inset(left=0.05, right=0.05, top=0.10, bottom=0.10).add_axes(fig)
        legend_ax.axis("off")
        noise_ax = noise_plot_region.inset(left=0.04, right=0.04, top=0.04, bottom=0.05).add_axes(fig)
        plot_panel_l(
            noise_scales=noise_data["noise_scales"],
            fds=noise_data["fds"],
            centroids=noise_data["centroids"],
            div_ratios=noise_data["div_ratios"],
            output_dir=str(out),
            dpi=dpi,
            cfg_scale=noise_data.get("cfg_scale", 1.5),
            ax_target=noise_ax,
            panel_label='e',
            legend_ax=legend_ax,
            embedded_label_pos=(-0.12, 1.06),
        )

    path = out / "fig06_diversity_diagnostics.png"
    save_with_vcd(fig, path, dpi)
    logger.info(f"Saved Fig 12 \u2192 {path}")
    saved.append(path)
    plt.close(fig)

    # ── Figure 14: Expression Diversity (now standalone Fig 9 with labels a-b) ──
    t6 = all_results.get("test6_expression_diversity", {})
    fig_k = plot_expression_diversity_panel(t6, output_dir=str(out), dpi=dpi, label_offset=0)
    if fig_k:
        saved.append(out / "fig07a_expression_diversity.png")
        plt.close(fig_k)

    return saved
