"""
panels_conditioning.py — Panels L (noise-scale tradeoff) and M (conditioning UMAP).

Plotting only; scripts run sweeps/generation and pass pre-computed data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .style import COLORS, SUPTITLE_Y_CLOSE, TYPE_PALETTE, apply_style, save_with_vcd, set_figure_suptitle

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def plot_panel_l(
    noise_scales: List[float],
    fds: List[float],
    centroids: List[float],
    div_ratios: List[float],
    output_dir: str | Path,
    dpi: int = 300,
    cfg_scale: float = 1.5,
) -> Path:
    """Panel L: Noise-scale vs fidelity/diversity tradeoff (plot only)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    fig, ax1 = plt.subplots(figsize=(6.8, 4.8))
    set_figure_suptitle(fig, "Noise-Scale Trade-off (CFG=1.5)", fontsize=11)

    color_fd = COLORS["real"]
    color_cos = COLORS["baseline_gauss"]
    color_div = COLORS["generated"]

    ax1.plot(noise_scales, fds, "o-", color=color_fd, lw=2, markersize=5, label="Fréchet Distance ↓", zorder=3)
    ax1.set_xlabel("Noise Scale (ε)")
    ax1.set_ylabel("Fréchet Distance", color=color_fd)
    ax1.tick_params(axis="y", labelcolor=color_fd)

    ax2 = ax1.twinx()
    ax2.plot(noise_scales, centroids, "s-", color=color_cos, lw=2, markersize=5, label="Centroid Cosine ↑")
    ax2.plot(noise_scales, div_ratios, "D-", color=color_div, lw=2, markersize=5, label="Diversity Ratio ↑")
    ax2.set_ylabel("Cosine / Ratio")
    ax2.set_ylim(0, 1.2)
    ax2.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    xticks = sorted(set(noise_scales))
    ax1.set_xticks(xticks, [f"{x:.2f}" for x in xticks])
    ax1.set_xlim(min(xticks) - 0.004, max(xticks) + 0.004)

    ax1.axvspan(0.02, 0.04, alpha=0.1, color=COLORS["good"], label="Sweet spot")

    chosen_eps = 0.03
    summary_chunks = []
    if chosen_eps in noise_scales:
        idx_chosen = noise_scales.index(chosen_eps)
        ax1.axvline(x=chosen_eps, color=COLORS["baseline_shuffle"], linestyle="-", linewidth=2.5,
                     alpha=0.8, zorder=10, label=f"Production (ε={chosen_eps})")
        ax1.scatter(
            [chosen_eps],
            [fds[idx_chosen]],
            color=COLORS["baseline_shuffle"],
            s=34,
            zorder=12,
        )
        summary_chunks.append(
            f"Production ε={chosen_eps:.2f}: FD {fds[idx_chosen]:.3f}, cos {centroids[idx_chosen]:.3f}, div {div_ratios[idx_chosen]:.3f}"
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=8,
        frameon=False,
    )

    best_idx = np.argmin(fds)
    if noise_scales[best_idx] != chosen_eps:
        ax1.scatter([noise_scales[best_idx]], [fds[best_idx]], color="black", s=28, zorder=12)
        summary_chunks.append(f"Best FD ε={noise_scales[best_idx]:.2f}: {fds[best_idx]:.3f}")

    if summary_chunks:
        fig.text(
            0.5,
            0.015,
            " | ".join(summary_chunks),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    path = output_dir / "panel_l_noise_tradeoff.png"
    save_with_vcd(fig, path, dpi)
    logger.info(f"Saved Panel L → {path}")
    plt.close(fig)
    return path


def plot_panel_m(
    coords: np.ndarray,
    combined_labels: np.ndarray,
    combined_source: np.ndarray,
    selected_types: List[int],
    mode_diversity: Dict[str, float],
    real_diversity: float,
    type_names: Dict[int, str],
    output_dir: str | Path,
    dpi: int = 300,
    cfg_scale: float = 1.5,
    n_real: int = 0,
    mode_counts: Optional[Dict[str, int]] = None,
) -> Path:
    """Panel M: Conditioning mode comparison (plot only). coords are (n, 2) PCA/UMAP."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    n_modes = 1 + len(mode_diversity)  # real + each mode
    _fw = max(10.0, 2.8 * n_modes)
    fig = plt.figure(figsize=(_fw, 8.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.1, 1.2], hspace=0.48)
    gs_top = outer[0].subgridspec(1, n_modes, wspace=0.45)
    axes = [fig.add_subplot(gs_top[0, i]) for i in range(n_modes)]

    # Local spacing policy for Panel M:
    # - suptitle raised to y=1.00 to increase clearance above row-1 titles
    # - row-1 titles kept short and semantic
    # - figure-level legend at bottom with stable anchor, no overlap
    set_figure_suptitle(
        fig,
        f"Conditioning Mode Comparison (CFG={cfg_scale}, {len(selected_types)} types, PCA 2D)",
        fontsize=11,
        y=1.00,
    )

    type_to_color = {tid: TYPE_PALETTE[i % len(TYPE_PALETTE)] for i, tid in enumerate(selected_types)}
    type_to_name = {
        tid: (type_names.get(int(tid), f"Type_{tid}")[:25] if type_names else f"Type_{tid}")
        for tid in selected_types
    }

    def plot_one(ax, mask, title, alpha=0.4, size=8, *, show_ylabel=True):
        for tid in selected_types:
            tmask = mask & (combined_labels == tid)
            ax.scatter(coords[tmask, 0], coords[tmask, 1],
                       c=[type_to_color[tid]], s=size, alpha=alpha,
                       edgecolors="white", linewidths=0.2,
                       label=type_to_name[tid])
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xlabel("PC1", fontsize=9)
        if show_ylabel:
            ax.set_ylabel("PC2", fontsize=9)
        else:
            ax.set_ylabel("", fontsize=9)
            ax.tick_params(axis="y", labelleft=False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    real_mask_bool = combined_source == "Real"
    plot_one(axes[0], real_mask_bool,
             f"Real\n{n_real} cells, div={real_diversity:.3f}",
             alpha=0.2, size=4)

    mode_counts = mode_counts or {}
    for i, mode_name in enumerate(mode_diversity.keys()):
        mode_mask = combined_source == mode_name
        div_val = mode_diversity.get(mode_name, 0.0)
        count = mode_counts.get(mode_name, mode_mask.sum())
        plot_one(
            axes[i + 1],
            mode_mask,
            f"{mode_name}\n{count} cells, div={div_val:.3f}",
            show_ylabel=False,
        )

    # Quantitative second row: mode shift and diversity summaries.
    gs_bottom = outer[1].subgridspec(1, 3, wspace=0.38)
    ax_b1 = fig.add_subplot(gs_bottom[0, 0])
    ax_b2 = fig.add_subplot(gs_bottom[0, 1])
    ax_b3 = fig.add_subplot(gs_bottom[0, 2])

    # Build per-type real centroids in 2D for shift summaries.
    real_centroids = {}
    for tid in selected_types:
        rmask = real_mask_bool & (combined_labels == tid)
        if np.any(rmask):
            real_centroids[tid] = coords[rmask].mean(axis=0)

    shift_means = []
    shift_stds = []
    shift_labels = []
    per_type_shift_distributions = []

    for mode_name in mode_diversity.keys():
        mode_mask = combined_source == mode_name
        shifts = []
        for tid in selected_types:
            if tid not in real_centroids:
                continue
            mmask = mode_mask & (combined_labels == tid)
            if not np.any(mmask):
                continue
            mode_centroid = coords[mmask].mean(axis=0)
            shifts.append(float(np.linalg.norm(mode_centroid - real_centroids[tid])))
        if shifts:
            shift_labels.append(mode_name)
            shift_means.append(float(np.mean(shifts)))
            shift_stds.append(float(np.std(shifts)))
            per_type_shift_distributions.append(shifts)

    if shift_labels:
        xpos = np.arange(len(shift_labels))
        ax_b1.bar(
            xpos,
            shift_means,
            yerr=shift_stds,
            capsize=3,
            color=COLORS["generated"],
            alpha=0.85,
            edgecolor="white",
        )
        ax_b1.set_xticks(xpos)
        ax_b1.set_xticklabels([m.split(" (")[0] for m in shift_labels], rotation=18, ha="right", fontsize=8)
        ax_b1.set_ylabel("Mean centroid shift (PC units)", fontsize=9)
        ax_b1.set_title("Mode -> Real Shift", fontsize=10)
    else:
        ax_b1.text(0.5, 0.5, "No centroid shift data", ha="center", va="center", transform=ax_b1.transAxes)
        ax_b1.set_title("Mode -> Real Shift", fontsize=10)

    div_labels = ["Real"] + list(mode_diversity.keys())
    div_values = [real_diversity] + [mode_diversity[m] for m in mode_diversity.keys()]
    div_colors = [COLORS["real"]] + [COLORS["generated"] for _ in mode_diversity.keys()]
    xdiv = np.arange(len(div_labels))
    ax_b2.bar(xdiv, div_values, color=div_colors, alpha=0.85, edgecolor="white")
    ax_b2.axhline(real_diversity, color=COLORS["real"], linestyle="--", linewidth=1.2, alpha=0.8)
    ax_b2.set_xticks(xdiv)
    ax_b2.set_xticklabels([m.split(" (")[0] for m in div_labels], rotation=18, ha="right", fontsize=8)
    ax_b2.set_ylabel("Within-type diversity", fontsize=9)
    ax_b2.set_title("Diversity by Mode", fontsize=10)

    if per_type_shift_distributions:
        ax_b3.boxplot(
            per_type_shift_distributions,
            labels=[m.split(" (")[0] for m in shift_labels],
            patch_artist=True,
            boxprops=dict(facecolor=COLORS["bg_light"], edgecolor=COLORS["neutral"]),
            medianprops=dict(color=COLORS["bad"], linewidth=1.3),
            whiskerprops=dict(color=COLORS["neutral"], linewidth=1.0),
            capprops=dict(color=COLORS["neutral"], linewidth=1.0),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=COLORS["warn"], markeredgecolor="none", alpha=0.6),
        )
        ax_b3.tick_params(axis="x", labelrotation=18, labelsize=8)
        ax_b3.set_ylabel("Per-type centroid shift", fontsize=9)
        ax_b3.set_title("Shift Distribution", fontsize=10)
    else:
        ax_b3.text(0.5, 0.5, "No shift distribution data", ha="center", va="center", transform=ax_b3.transAxes)
        ax_b3.set_title("Shift Distribution", fontsize=10)

    # Legend: type keys only, anchored at bottom with no overlap
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(handles), 5), fontsize=7,
               markerscale=1.5, frameon=False,
               columnspacing=0.8, handletextpad=0.3,
               bbox_to_anchor=(0.5, -0.01))
    # Layout rect: extra top clearance (0.96) for suptitle-to-row-1 separation
    fig._clop_layout_rect = (0.02, 0.06, 0.98, 0.96)

    path = output_dir / "panel_m_conditioning_umap.png"
    save_with_vcd(fig, path, dpi)
    logger.info(f"Saved Panel M → {path}")
    plt.close(fig)
    return path
