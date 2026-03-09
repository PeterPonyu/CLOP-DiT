"""
panels_diversity.py — Panels J (diversity diagnostics) and K (expression diversity).

Plotting only; scripts run diagnostics and pass all_results from diversity_diagnostics.json.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, apply_style, save_with_vcd, set_figure_suptitle, add_panel_label, abbreviate_cell_type

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def plot_diagnostics(
    all_results: Dict,
    output_dir: str = "results/figures",
    dpi: int = 300,
    type_names: Optional[Dict[int, str]] = None,
) -> List[Path]:
    """Generate Panel J (diversity diagnostics) and Panel K (expression diversity) from test results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    # Auto-load type names from captions file if not provided
    if type_names is None:
        import json as _json
        _cap_candidates = [
            Path("data/cached_latents_v5.2/text_captions_deduplicated.json"),
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

    # ── Panel J: Diversity Diagnostics (4 subplots) ──
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.5),
                             gridspec_kw={"hspace": 0.60, "wspace": 0.55})
    fig._clop_layout_rect = (0.02, 0.03, 0.82, 0.95)
    add_panel_label(axes[0, 0], 'a', x=-0.10, y=1.05)
    add_panel_label(axes[0, 1], 'b', x=-0.10, y=1.05)
    add_panel_label(axes[1, 0], 'c', x=-0.10, y=1.05)
    add_panel_label(axes[1, 1], 'd', x=-0.10, y=1.05)

    ax = axes[0, 0]
    t1 = all_results.get("test1_intratype_diversity", {}).get("per_type", {})
    if t1:
        names = list(t1.keys())
        div_ratios = [t1[n]["diversity_ratio"] for n in names]
        sorted_idx = np.argsort(div_ratios)
        sorted_names = [abbreviate_cell_type(names[i], 22) for i in sorted_idx]
        sorted_divs = [div_ratios[i] for i in sorted_idx]

        colors = [COLORS["bad"] if d < 0.5 else COLORS["warn"] if d < 0.8 else COLORS["good"] if d < 1.2 else COLORS["real"]
                  for d in sorted_divs]
        ax.barh(range(len(sorted_divs)), sorted_divs, color=colors, height=0.8)
        ax.set_yticks(range(len(sorted_divs)))
        _step_j1 = max(1, len(sorted_divs) // 14)
        _ytl_j1 = [n if i % _step_j1 == 0 else "" for i, n in enumerate(sorted_names)]
        ax.set_yticklabels(_ytl_j1, fontsize=7)
        ax.axvline(x=1.0, color="black", ls="--", lw=1, alpha=0.5, label="ratio=1 (equal)")
        ax.axvline(x=0.5, color="red", ls=":", lw=1, alpha=0.5, label="ratio=0.5 (collapse)")
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
        ax.axhline(y=0.01, color="red", ls=":", alpha=0.5, label="memorization threshold")
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
        ax.plot(scales, divs, "o-", color=color, lw=2, markersize=8, label="Diversity")
        ax.set_xlabel("CFG Scale")
        ax.set_ylabel("Mean Intra-Type Diversity", color=color)
        ax.tick_params(axis="y", labelcolor=color)

        ax2 = ax.twinx()
        color2 = COLORS["generated"]
        ax2.plot(scales, norms, "s--", color=color2, lw=2, markersize=8, label="Norm")
        ax2.set_ylabel("Mean Embedding Norm", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        ax.set_title("CFG Scale vs Diversity & Norm")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Labels will be collected into the figure-level legend below
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
        _type_labels_d = [abbreviate_cell_type(type_names.get(int(k), f"Type {k}"), 18)
                          for k in type_ids]
        ax.set_xticklabels(_type_labels_d, fontsize=7, rotation=30, ha="right")
        ax.set_xlabel("Cell Type")
        ax.set_ylabel("Intra-Type Diversity (1 - mean cosine)")
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
    if _handles:
        fig.legend(_handles, _labels, loc='center right',
                   bbox_to_anchor=(0.99, 0.5), fontsize=8, frameon=False, ncol=1)
        fig.subplots_adjust(right=0.83)

    path = out / "panel_j_diversity_diagnostics.png"
    save_with_vcd(fig, path, dpi)
    logger.info(f"Saved Panel J → {path}")
    saved.append(path)
    plt.close(fig)

    # ── Panel K: Expression Diversity ──
    t6 = all_results.get("test6_expression_diversity", {})
    fig_k = plot_expression_diversity_panel(t6, output_dir=str(out), dpi=dpi)
    if fig_k:
        saved.append(out / "panel_k_expression_diversity.png")
        plt.close(fig_k)

    return saved


def plot_expression_diversity_panel(
    t6_data: Dict,
    output_dir: str = "results/figures",
    dpi: int = 300,
    label_offset: int = 0,
    save: bool = True,
) -> "Optional[plt.Figure]":
    """Panel K: Expression diversity (standalone, supports label_offset for merged figures).

    Parameters
    ----------
    t6_data : dict
        The ``test6_expression_diversity`` section from diversity_diagnostics.json.
    output_dir : str
        Directory for saved figures.
    dpi : int
        Resolution for raster output.
    label_offset : int
        Offset for panel labels (0 -> 'a','b'; 1 -> 'b','c'; etc.)
    save : bool
        Whether to persist the figure to disk.

    Returns
    -------
    fig or None
    """
    from typing import Optional as _Opt  # local to avoid top-level cycle

    if not t6_data or not t6_data.get("overall"):
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()

    o = t6_data["overall"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.8))
    add_panel_label(axes[0], chr(ord('a') + label_offset), x=-0.10, y=1.05)
    add_panel_label(axes[1], chr(ord('a') + label_offset + 1), x=-0.10, y=1.05)

    ax = axes[0]
    labels = ["Cell Std\n(across genes)", "Gene Std\n(across cells)"]
    real_vals = [o["real_mean_cell_std"], o["real_mean_gene_std"]]
    gen_vals = [o["gen_mean_cell_std"], o["gen_mean_gene_std"]]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, real_vals, w, label="Real", color=COLORS["real"], alpha=0.8)
    ax.bar(x + w / 2, gen_vals, w, label="Generated", color=COLORS["generated"], alpha=0.8)
    # Add gen/real ratio annotations above each bar pair
    for _bi in range(len(real_vals)):
        if real_vals[_bi] > 0:
            ratio = gen_vals[_bi] / real_vals[_bi]
            max_h = max(real_vals[_bi], gen_vals[_bi])
            ax.text(_bi, max_h * 1.05, f"ratio={ratio:.2f}",
                    ha="center", fontsize=8, color=COLORS["neutral"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Expression Variability Summary")
    ax.legend(frameon=False)

    ax = axes[1]
    pt_ratio = t6_data.get("per_type_gene_std_ratio", {})
    if pt_ratio:
        vals = [pt_ratio["min"], pt_ratio["mean"], pt_ratio["max"]]
        lbls = ["Min", "Mean", "Max"]
        colors = [COLORS["bad"] if v < 0.5 else COLORS["good"] for v in vals]
        ax.bar(lbls, vals, color=colors, alpha=0.8, edgecolor="white")
        ax.axhline(y=1.0, color="black", ls="--", lw=1, alpha=0.5,
                   label="ratio=1 (equal diversity)")
        ax.set_ylabel("Gene Std Ratio (gen / real)")
        ax.set_title("Per-Type Gene Std Ratio")
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "No per-type data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=COLORS["neutral"])
        ax.set_title("Per-Type Gene Std Ratio")

    if save:
        path = out / "panel_k_expression_diversity.png"
        save_with_vcd(fig, path, dpi)
        logger.info(f"Saved Panel K → {path}")

    return fig
