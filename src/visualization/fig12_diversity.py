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

from .explicit_positioning import add_shared_legend_axes
from .panel_geometry import apply_layout_rect
from .style import COLORS, FONT_DENSE_YTICK, apply_style, save_with_vcd, add_panel_label, abbreviate_cell_type
from .fig14_expr_diversity import plot_expression_diversity_panel

logger = logging.getLogger(__name__)


def plot_diagnostics(
    all_results: Dict,
    output_dir: str = "results/figures",
    dpi: int = 300,
    type_names: Optional[Dict[int, str]] = None,
) -> List[Path]:
    """Generate Fig 12 (diversity diagnostics) and Fig 14 (expression diversity) from test results."""
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

    # ── Figure 12: Diversity Diagnostics (4 subplots) ──
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.2),
                             gridspec_kw={"hspace": 0.65, "wspace": 0.62})
    apply_layout_rect(fig, (0.02, 0.12, 0.98, 0.96))
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
        ax.set_yticklabels(_ytl_j1, fontsize=FONT_DENSE_YTICK)
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
        _type_labels_d = [abbreviate_cell_type(type_names.get(int(k), f"Type {k}"), 22)
                          for k in type_ids]
        ax.set_xticklabels(_type_labels_d, fontsize=FONT_DENSE_YTICK, rotation=45, ha="right")
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
        legend_ax = add_shared_legend_axes(fig, (0.07, 0.02, 0.46, 0.08))
        legend_ax.legend(_handles, _labels, loc='center', fontsize=9, frameon=False,
                         ncol=min(len(_handles), 3))

    path = out / "fig12_diversity_diagnostics.png"
    save_with_vcd(fig, path, dpi)
    logger.info(f"Saved Fig 12 \u2192 {path}")
    saved.append(path)
    plt.close(fig)

    # ── Figure 14: Expression Diversity ──
    t6 = all_results.get("test6_expression_diversity", {})
    fig_k = plot_expression_diversity_panel(t6, output_dir=str(out), dpi=dpi)
    if fig_k:
        saved.append(out / "fig14_expression_diversity.png")
        plt.close(fig_k)

    return saved
