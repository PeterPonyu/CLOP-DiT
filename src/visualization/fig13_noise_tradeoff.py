"""
fig13_noise_tradeoff.py — Article Figure 13: noise-scale tradeoff.

Extracted from panels_conditioning.py (formerly Panel L).
Plotting only; scripts run sweeps and pass pre-computed data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, apply_style, save_with_vcd, add_panel_label

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
    """Panel L / Figure 13: Noise-scale vs fidelity/diversity tradeoff (plot only)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    # Title moved to LaTeX caption
    # set_figure_suptitle(fig, "Noise-Scale Trade-off (CFG=1.5)", fontsize=11)
    add_panel_label(ax1, 'a', x=-0.10, y=1.05)

    color_fd = COLORS["real"]
    color_cos = COLORS["baseline_gauss"]
    color_div = COLORS["generated"]

    ax1.plot(noise_scales, fds, "o-", color=color_fd, lw=2, markersize=5, label="Fréchet Distance \u2193", zorder=3)
    ax1.set_xlabel("Noise Scale (\u03b5)")
    ax1.set_ylabel("Fréchet Distance", color=color_fd)
    ax1.tick_params(axis="y", labelcolor=color_fd)

    ax2 = ax1.twinx()
    ax2.plot(noise_scales, centroids, "s-", color=color_cos, lw=2, markersize=5, label="Centroid Cosine \u2191")
    ax2.plot(noise_scales, div_ratios, "D-", color=color_div, lw=2, markersize=5, label="Diversity Ratio \u2191")
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
                     alpha=0.8, zorder=10, label=f"Production (\u03b5={chosen_eps})")
        ax1.scatter(
            [chosen_eps],
            [fds[idx_chosen]],
            color=COLORS["baseline_shuffle"],
            s=34,
            zorder=12,
        )
        summary_chunks.append(
            f"Production \u03b5={chosen_eps:.2f}: FD {fds[idx_chosen]:.3f}, cos {centroids[idx_chosen]:.3f}, div {div_ratios[idx_chosen]:.3f}"
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=1,
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
    )

    best_idx = np.argmin(fds)
    if noise_scales[best_idx] != chosen_eps:
        ax1.scatter([noise_scales[best_idx]], [fds[best_idx]], color="black", s=28, zorder=12)
        summary_chunks.append(f"Best FD \u03b5={noise_scales[best_idx]:.2f}: {fds[best_idx]:.3f}")

    # Summary footer removed per revision; information moved to LaTeX caption
    # if summary_chunks:
    #     fig.text(0.5, 0.015, " | ".join(summary_chunks), ...)

    path = output_dir / "fig13_noise_tradeoff.png"
    save_with_vcd(fig, path, dpi)
    logger.info(f"Saved Fig 13 \u2192 {path}")
    plt.close(fig)
    return path
