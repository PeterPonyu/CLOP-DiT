"""
fig06_fidelity.py -- Article Figure 6: Per-Type Generation Fidelity.

  G1: Centroid cosine per type (sorted, hardest types highlighted)
  G2: Frechet outlier profile (sorted, mean-anchored)
  G3: Fidelity vs abundance with FD bubble size and diversity-ratio color

Standalone function extracted from panels_heatmaps for composability and
VCD-integrated saving via an optional ``save_panel_fn`` callback.
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
    COLORS, FONT_HEATMAP_CELL, FONT_LEGEND, FONT_SMALL,
    PANEL_OFFSET_STD, PANEL_OFFSET_WIDE,
    abbreviate_cell_type, add_panel_label,
    quality_color, save_with_vcd, set_adaptive_ytick_labels, style_axes,
)
from .explicit_positioning import add_axes_next_to
from src.utils.paths import FIG_DIR, load_thresholds

logger = logging.getLogger(__name__)

_viz_thresh = load_thresholds().get("visualization", {})
_COSINE_QUALITY_BANDS = tuple(_viz_thresh.get("cosine_quality_bands", [0.9, 0.7]))


def plot_per_type_generation(
    metrics_path: str = "results/generation_metrics.json",
    div_metrics_path: str = "results/diversity_diagnostics.json",
    output_dir: Optional[str] = None,
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
    label_offset: int = 0,
) -> Optional[plt.Figure]:
    """Per-type generation quality with heterogeneity emphasis.

    G1: Centroid cosine per type (sorted, hardest types highlighted)
    G2: Frechet outlier profile (sorted, mean-anchored)
    G3: Fidelity vs abundance with FD bubble size and diversity-ratio color
    """
    if output_dir is None:
        output_dir = str(FIG_DIR)
    if not Path(metrics_path).exists():
        logger.info(f"No generation metrics found at {metrics_path} — skipping Panel G")
        return None

    with open(metrics_path) as f:
        data = json.load(f)

    per_type = data.get("per_type", {})
    if not per_type:
        logger.warning("No per-type metrics — skipping Panel G")
        return None

    names = list(per_type.keys())
    cosines = [per_type[n]["centroid_cosine"] for n in names]
    fds = [per_type[n].get("frechet_distance", float("nan")) for n in names]
    n_real = [per_type[n]["n_real"] for n in names]

    diversity_by_type_id = {}
    collapsed_type_ids = set()
    div_path = Path(div_metrics_path)
    if div_path.exists():
        try:
            with open(div_path) as f:
                div_data = json.load(f)
            t1 = div_data.get("test1_intratype_diversity", {})
            per_type_div = t1.get("per_type", {})
            for entry in per_type_div.values():
                t_id = entry.get("type_id")
                ratio = entry.get("diversity_ratio")
                if t_id is None or ratio is None:
                    continue
                diversity_by_type_id[int(t_id)] = float(ratio)
                if ratio < 0.5:
                    collapsed_type_ids.add(int(t_id))
        except Exception:
            diversity_by_type_id = {}
            collapsed_type_ids = set()

    type_ids = [per_type[n].get("type_id") for n in names]
    short_names = [abbreviate_cell_type(n, max_len=26) for n in names]

    div_ratios = []
    for t_id in type_ids:
        if t_id is None:
            div_ratios.append(np.nan)
        else:
            div_ratios.append(diversity_by_type_id.get(int(t_id), np.nan))

    fd_array = np.asarray(fds, dtype=float)
    cos_array = np.asarray(cosines, dtype=float)
    n_real_array = np.asarray(n_real, dtype=float)
    div_array = np.asarray(div_ratios, dtype=float)
    fd_valid = np.isfinite(fd_array)
    fd_mean = float(np.nanmean(fd_array)) if fd_valid.any() else float("nan")

    fig = plt.figure(figsize=(14.0, 5.8))
    layout = bind_figure_region(fig, (0.12, 0.12, 0.985, 0.92))
    g1_slot, g2_slot, g3_slot = layout.split_cols([1.00, 1.18, 0.76], gap=[0.024, 0.038])
    g1_rect = g1_slot.inset(left=0.04, right=0.004)
    g2_rect = g2_slot.inset(left=0.148, right=0.058)
    g3_rect = g3_slot.inset(left=0.020, right=0.030)
    summary = data.get("summary", {})
    # Title moved to LaTeX caption

    # G1: Centroid cosine (sorted)
    from matplotlib.ticker import MaxNLocator
    ax = g1_rect.add_axes(fig)
    add_panel_label(ax, chr(ord('a') + label_offset), x=PANEL_OFFSET_WIDE[0], y=PANEL_OFFSET_WIDE[1])
    sorted_idx = np.argsort(cosines)
    sorted_cos = [cosines[i] for i in sorted_idx]
    sorted_names_cos = [short_names[i] for i in sorted_idx]
    sorted_type_ids = [type_ids[i] for i in sorted_idx]
    # Centroid cosine quality bands (from configs/thresholds.yaml → visualization.cosine_quality_bands)
    colors = []
    for v, t_id in zip(sorted_cos, sorted_type_ids):
        if t_id is not None and int(t_id) in collapsed_type_ids:
            colors.append(COLORS["bad"])
        else:
            colors.append(quality_color(v, _COSINE_QUALITY_BANDS))
    ax.barh(range(len(sorted_cos)), sorted_cos, color=colors, height=0.8)
    set_adaptive_ytick_labels(ax, sorted_names_cos, max_visible=18, fontsize=FONT_HEATMAP_CELL)
    ax.set_xlabel("Centroid Cosine Similarity")
    ax.set_title("Real\u2194Gen Centroid Cosine", fontsize=12)
    ax.axvline(
        x=summary.get("mean_centroid_cosine", 0), color=COLORS["bad"],
        linestyle="--", alpha=0.5,
        label="mean (see caption)",
    )
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))
    ax.legend(fontsize=FONT_LEGEND, frameon=False, loc="lower right")
    style_axes(ax, kind="bar")

    # G2: Frechet outlier profile
    ax = g2_rect.add_axes(fig)
    add_panel_label(ax, chr(ord('a') + label_offset + 1), x=PANEL_OFFSET_WIDE[0], y=PANEL_OFFSET_WIDE[1])
    if fd_valid.any():
        fd_idx = np.where(fd_valid)[0][np.argsort(fd_array[fd_valid])]
        fd_vals = fd_array[fd_idx]
        fd_names = [short_names[i] for i in fd_idx]
        fd_min = float(np.nanmin(fd_vals))
        fd_ptp = float(np.nanmax(fd_vals) - fd_min) or 1.0
        fd_norm = np.clip((fd_vals - fd_min) / fd_ptp, 0, 1)
        colors_fd = plt.cm.PiYG_r(fd_norm)
        y_pos = np.arange(len(fd_vals))
        ax.hlines(y_pos, 0, fd_vals, color=colors_fd, linewidth=2.8, alpha=0.85)
        ax.scatter(fd_vals, y_pos, s=28 + 70 * fd_norm, color=colors_fd, edgecolors="white", linewidths=0.4, zorder=3)
        if np.isfinite(fd_mean):
            ax.axvline(fd_mean, color=COLORS["bad"], linestyle="--", alpha=0.7, linewidth=1.5, label="mean (see caption)")
        set_adaptive_ytick_labels(ax, fd_names, max_visible=18, fontsize=FONT_HEATMAP_CELL)
        ax.set_xlabel("Fr\u00e9chet Distance (lower = better)")
        ax.set_title("Fr\u00e9chet Outlier Profile")
        # Mean reference is described in the caption; omit legend here to keep the panel clear.
        ax.set_xlim(0, float(np.nanmax(fd_vals)) * 1.12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
        style_axes(ax, kind="bar")
    else:
        ax.text(
            0.5, 0.5, "No valid FD values",
            ha="center", va="center", transform=ax.transAxes,
        )

    # G3: Cosine vs abundance with FD bubble size and diversity color
    ax = g3_rect.add_axes(fig)
    add_panel_label(ax, chr(ord('a') + label_offset + 2), x=0.00, y=PANEL_OFFSET_STD[1])
    fd_for_size = np.where(fd_valid, fd_array, np.nanmedian(fd_array[fd_valid]) if fd_valid.any() else 1.0)
    fd_min = float(np.nanmin(fd_for_size)) if np.isfinite(fd_for_size).any() else 0.0
    fd_ptp = float(np.nanmax(fd_for_size) - fd_min) if np.isfinite(fd_for_size).any() else 1.0
    fd_ptp = fd_ptp or 1.0
    bubble_sizes = 50 + 220 * np.clip((fd_for_size - fd_min) / fd_ptp, 0, 1)
    x_vals = np.log10(np.maximum(n_real_array, 1))

    cax = None
    if np.isfinite(div_array).any():
        color_values = np.where(np.isfinite(div_array), div_array, np.nanmedian(div_array[np.isfinite(div_array)]))
        sc = ax.scatter(
            x_vals,
            cos_array,
            c=color_values,
            cmap="viridis",
            vmin=0.5,
            vmax=1.05,
            s=bubble_sizes,
            alpha=0.78,
            edgecolors="white",
            linewidth=0.6,
            clip_on=False,
        )
        cax = add_axes_next_to(
            fig,
            ax,
            side="right",
            width=0.010,
            height=ax.get_position().height * 0.34,
            pad=0.012,
            align="bottom",
            y_offset=0.012,
        )
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("")
        cbar.ax.set_title("Div.\nratio", fontsize=12, pad=4)
        cbar.ax.tick_params(labelsize=12)
    else:
        ax.scatter(
            x_vals,
            cos_array,
            c=COLORS["real"],
            s=bubble_sizes,
            alpha=0.78,
            edgecolors="white",
            linewidth=0.6,
            clip_on=False,
        )

    if len(x_vals) > 1:
        slope, intercept = np.polyfit(x_vals, cos_array, deg=1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color=COLORS["trend_dark"], linestyle="--", linewidth=1.3, label="Trend")

    candidate_idx = np.argsort(cos_array)[:10]
    label_offsets = [(-34, -12), (28, -10), (-30, 14), (28, 12), (16, -20), (-26, 20)]
    annotated = 0
    placed_data_coords: list[tuple[float, float]] = []
    for rank, i in enumerate(candidate_idx):
        if annotated >= 3:
            break
        if cos_array[i] < 0.94:
            if x_vals[i] > np.quantile(x_vals, 0.88) and annotated >= 3:
                continue
            # Skip if too close to an already-placed annotation (avoid overlap)
            too_close = any(
                abs(x_vals[i] - px) < 0.25 and abs(cos_array[i] - py) < 0.04
                for px, py in placed_data_coords
            )
            if too_close:
                continue
            x_offset, y_offset = label_offsets[rank % len(label_offsets)]
            if x_vals[i] > np.median(x_vals):
                x_offset = min(x_offset, -10)
            else:
                x_offset = max(x_offset, 10)
            ax.annotate(
                abbreviate_cell_type(short_names[i], max_len=10),
                (x_vals[i], cos_array[i]),
                fontsize=FONT_SMALL + 2,
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.75),
            )
            placed_data_coords.append((x_vals[i], cos_array[i]))
            annotated += 1
    # Panel G (third sub-panel): user requested larger fonts (+2 pt) for
    # improved legibility; keep panels E/F at their default sizes.
    ax.set_xlabel("log10(Number of Real Cells)", fontsize=13)
    ax.set_ylabel("Centroid Cosine Similarity", fontsize=13)
    ax.set_title("Fidelity vs Abundance", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.axhline(y=0.9, color=COLORS["good"], linestyle=":", alpha=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlim(x_vals.min() - 0.10, x_vals.max() + 0.10)
    style_axes(ax, kind="scatter")

    if save:
        path = Path(output_dir) / "fig03b_per_type_fidelity.png"
        if save_panel_fn is not None:
            save_panel_fn(fig, path, dpi)
        else:
            save_with_vcd(fig, path, dpi)
    return fig
