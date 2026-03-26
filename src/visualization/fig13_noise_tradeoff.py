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
from matplotlib.ticker import MaxNLocator

from .direct_layout import bind_figure_region
from .explicit_positioning import add_shared_legend_axes
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
    ax_target: plt.Axes | None = None,
    panel_label: str = 'e',
    legend_ax: plt.Axes | None = None,
    embedded_label_pos: tuple[float, float] | None = None,
) -> Path | None:
    """Panel L / Figure 13: Noise-scale vs fidelity/diversity tradeoff (plot only).

    If *ax_target* is given, draw into it instead of creating a standalone figure.
    Returns the saved path (standalone) or None (embedded).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    standalone = ax_target is None
    if standalone:
        fig = plt.figure(figsize=(8.2, 4.0))
        ax1 = bind_figure_region(fig, (0.10, 0.21, 0.92, 0.92)).add_axes(fig)
    else:
        ax1 = ax_target
        fig = ax1.figure
    # Title moved to LaTeX caption
    # set_figure_suptitle(fig, "Noise-Scale Trade-off (CFG=1.5)", fontsize=11)

    label_x, label_y = (-0.12, 1.05) if standalone else (embedded_label_pos or (-0.14, 1.08))
    add_panel_label(ax1, panel_label, x=label_x, y=label_y)

    color_fd = COLORS["real"]
    color_cos = COLORS["baseline_gauss"]
    color_div = COLORS["generated"]

    ax1.plot(noise_scales, fds, "o-", color=color_fd, lw=2, markersize=5, label="FD \u2193", zorder=3)
    ax1.set_xlabel("Noise Scale (\u03b5)")
    ax1.set_ylabel("Fréchet Distance", color=color_fd)
    ax1.tick_params(axis="y", labelcolor=color_fd)

    ax2 = ax1.twinx()
    ax2.plot(noise_scales, centroids, "s-", color=color_cos, lw=2, markersize=5, label="Centroid cos \u2191")
    ax2.plot(noise_scales, div_ratios, "D-", color=color_div, lw=2, markersize=5, label="Diversity \u2191")
    ax2.set_ylabel("Cosine / ratio", labelpad=8)
    ax2.set_ylim(0, 1.2)
    ax2.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))

    xticks = sorted(set(noise_scales))
    ax1.set_xticks(xticks, [f"{x:.2f}" for x in xticks])
    ax1.set_xlim(min(xticks) - 0.004, max(xticks) + 0.004)

    ax1.axvspan(0.02, 0.04, alpha=0.1, color=COLORS["good"], label="Sweet spot")

    chosen_eps = 0.03
    summary_chunks = []
    if chosen_eps in noise_scales:
        idx_chosen = noise_scales.index(chosen_eps)
        ax1.axvline(x=chosen_eps, color=COLORS["baseline_shuffle"], linestyle="-", linewidth=2.5,
                 alpha=0.8, zorder=10, label=f"Production \u03b5={chosen_eps}")
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
    legend_handles = lines1 + lines2
    legend_labels = labels1 + labels2
    if standalone:
        legend_ax = add_shared_legend_axes(fig, (0.10, 0.02, 0.80, 0.08))
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="center",
            ncol=3,
            fontsize=9,
            frameon=False,
        )
    elif legend_ax is not None:
        legend_ax.set_axis_off()
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            ncol=1,
            fontsize=8.5,
            frameon=False,
            handlelength=1.6,
            handletextpad=0.5,
            columnspacing=0.8,
        )
    else:
        ax1.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            ncol=2,
            fontsize=7.5,
            frameon=False,
        )

    best_idx = np.argmin(fds)
    if noise_scales[best_idx] != chosen_eps:
        ax1.scatter([noise_scales[best_idx]], [fds[best_idx]], color="black", s=28, zorder=12)
        summary_chunks.append(f"Best FD \u03b5={noise_scales[best_idx]:.2f}: {fds[best_idx]:.3f}")

    # Summary footer removed per revision; information moved to LaTeX caption
    # if summary_chunks:
    #     fig.text(0.5, 0.015, " | ".join(summary_chunks), ...)

    if not standalone:
        return None

    path = output_dir / "fig13_noise_tradeoff.png"
    save_with_vcd(fig, path, dpi)
    logger.info(f"Saved Fig 13 \u2192 {path}")
    plt.close(fig)
    return path
