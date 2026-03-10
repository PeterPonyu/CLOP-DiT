"""
fig07_alignment.py -- Article Figure 7: Text-Cell Alignment Heatmap.

  F1: Clustered heatmap with diagonal highlight, off-diagonal confusions annotated,
      inset zoom on diagonal, and statistical summary text box
  F2: Sorted per-type alignment bars with threshold bands, value annotations,
      quality-tier counts, and median marker
  F3: Distribution of diagonal vs off-diagonal similarities with KDE overlay,
      Mann-Whitney U p-value, Cohen's d effect size, median markers, and
      bootstrap CI annotation

Standalone function extracted from panels_heatmaps for composability and
VCD-integrated saving via an optional ``save_panel_fn`` callback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .style import (
    COLORS, FONT_HEATMAP_CELL, FONT_SMALL, FONT_TITLE,
    abbreviate_cell_type, add_panel_label,
    quality_color, save_with_vcd, set_adaptive_ytick_labels, style_axes,
)
from .explicit_positioning import add_axes_next_to, layout_axes_row
from .panel_geometry import apply_layout_rect
from src.utils.paths import FIG_DIR

logger = logging.getLogger(__name__)


def plot_text_cell_heatmap(
    cache_dir: str = "data/cache",
    type_names: Optional[Dict[int, str]] = None,
    output_dir: Optional[str] = None,
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
    label_offset: int = 0,
) -> Optional[plt.Figure]:
    """Enhanced 69x69 text-cell alignment heatmap with rich annotations.

    F1: Clustered heatmap with diagonal highlight, off-diagonal confusions annotated,
        inset zoom on diagonal, and statistical summary text box
    F2: Sorted per-type alignment bars with threshold bands, value annotations,
        quality-tier counts, and median marker
    F3: Distribution of diagonal vs off-diagonal similarities with KDE overlay,
        Mann-Whitney U p-value, Cohen's d effect size, median markers, and
        bootstrap CI annotation
    """
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.patheffects as mpe

    if output_dir is None:
        output_dir = str(FIG_DIR)
    cache = Path(cache_dir)
    if type_names is None:
        type_names = {}

    proj_text_path = cache / "projected_text.npy"
    proj_cell_path = cache / "projected_cells.npy"
    gid_path = cache / "text_group_ids_dedup.npy"

    for p in [proj_text_path, proj_cell_path, gid_path]:
        if not p.exists():
            logger.warning(f"Missing {p.name} — skipping Panel F")
            return None

    proj_text = np.load(proj_text_path)
    proj_cells = np.load(proj_cell_path)
    group_ids = np.load(gid_path)
    # Use full text group IDs for text centroid computation if sizes differ
    gid_text_path = cache / "text_group_ids.npy"
    if gid_text_path.exists() and proj_text.shape[0] != group_ids.shape[0]:
        text_group_ids = np.load(gid_text_path)
    else:
        text_group_ids = group_ids

    # Auto-load type names from deduplicated captions if not provided
    if not type_names:
        cap_path = cache / "text_captions_deduplicated.json"
        if cap_path.exists():
            with open(cap_path) as _cf:
                _raw_caps = json.load(_cf)
            type_names = {}
            for k, v in _raw_caps.items():
                name = v.split(" are ")[0] if " are " in v else v[:50]
                type_names[int(k)] = name

    unique_types = np.sort(np.unique(group_ids))
    n_types = len(unique_types)

    text_centroids = np.zeros((n_types, proj_text.shape[1]), dtype=np.float32)
    cell_centroids = np.zeros((n_types, proj_cells.shape[1]), dtype=np.float32)
    type_counts = np.zeros(n_types, dtype=int)
    for i, t in enumerate(unique_types):
        mask = group_ids == t
        type_counts[i] = mask.sum()
        tc = proj_text[text_group_ids == t].mean(axis=0)
        text_centroids[i] = tc / (np.linalg.norm(tc) + 1e-8)
        cc = proj_cells[mask].mean(axis=0)
        cell_centroids[i] = cc / (np.linalg.norm(cc) + 1e-8)

    sim_matrix = text_centroids @ cell_centroids.T

    labels = [abbreviate_cell_type(type_names.get(int(t), f"T{t}"), max_len=22) for t in unique_types]
    diag = np.diag(sim_matrix)
    mean_diag = diag.mean()
    std_diag = diag.std()
    median_diag = float(np.median(diag))
    off_diag = sim_matrix[~np.eye(n_types, dtype=bool)]
    mean_off = off_diag.mean()
    std_off = off_diag.std()
    median_off = float(np.median(off_diag))

    # --- Compute statistics for annotations ---
    # Cohen's d: effect size between diagonal and off-diagonal distributions
    pooled_std = np.sqrt((std_diag**2 + std_off**2) / 2.0)
    cohens_d = (mean_diag - mean_off) / pooled_std if pooled_std > 0 else float("inf")
    # Separation ratio: mean_diag / mean_off
    separation_ratio = mean_diag / mean_off if mean_off != 0 else float("inf")
    # Mann-Whitney U test for diagonal vs off-diagonal separation
    try:
        from scipy.stats import mannwhitneyu
        _u_stat, p_value = mannwhitneyu(diag, off_diag, alternative="greater")
    except ImportError:
        p_value = None

    # Quality tier counts
    n_excellent = int(np.sum(diag >= 0.9))
    n_good = int(np.sum((diag >= 0.7) & (diag < 0.9)))
    n_poor = int(np.sum(diag < 0.7))

    # Try to load bootstrap CIs for centroid cosine
    bootstrap_ci = None
    try:
        from src.utils.paths import RESULTS_DIR as _RESULTS_DIR
        ci_path = _RESULTS_DIR / "bootstrap_cis.json"
        if ci_path.exists():
            with open(ci_path) as _f:
                ci_data = json.load(_f)
            cc_ci = ci_data.get("metrics", {}).get("centroid_cosine", {})
            if cc_ci:
                bootstrap_ci = {
                    "lower": cc_ci.get("ci_95_lower"),
                    "upper": cc_ci.get("ci_95_upper"),
                    "point": cc_ci.get("point_estimate"),
                }
    except Exception:
        bootstrap_ci = None

    # Reorder by diagonal similarity for visual clarity
    sort_order = np.argsort(-diag)
    sim_sorted = sim_matrix[sort_order][:, sort_order]
    labels_sorted = [labels[i] for i in sort_order]

    fig = plt.figure(figsize=(15.2, 9.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 0.84, 0.66], wspace=0.46)
    # Title moved to LaTeX caption
    apply_layout_rect(fig, (0.03, 0.10, 0.98, 0.95))

    # ── F1: Clustered heatmap with annotations ──
    ax1 = fig.add_subplot(gs[0])
    add_panel_label(ax1, chr(ord('a') + label_offset), x=-0.10, y=1.05)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_heat",
        [
            "#1a237e", "#283593", "#42a5f5", "#e3f2fd", "#fff9c4",
            "#ffcc80", "#ff7043", "#d32f2f", "#b71c1c",
        ],
        N=256,
    )
    im = ax1.imshow(
        sim_sorted, cmap=cmap, vmin=-0.1, vmax=1.0,
        aspect="auto", interpolation="nearest",
    )
    step = 4
    _xtl = [labels_sorted[i] if i % step == 0 else "" for i in range(n_types)]
    _ytl = [labels_sorted[i] if i % step == 0 else "" for i in range(n_types)]
    ax1.set_xticks(range(n_types))
    ax1.set_yticks(range(n_types))
    ax1.set_xticklabels(_xtl, rotation=75, fontsize=9, ha="right")
    ax1.set_yticklabels(_ytl, fontsize=10, ha="right")
    ax1.set_ylabel("Cell Type (text prototypes)", fontsize=11)
    ax1.set_title("Cosine Similarity (sorted by diagonal)", fontsize=12)

    # Draw diagonal guide line
    ax1.plot(
        [0, n_types - 1], [0, n_types - 1],
        color="white", linewidth=0.8, linestyle=":", alpha=0.6, zorder=3,
    )

    # Off-diagonal confusion details moved to LaTeX caption for cleaner panel

    cax = add_axes_next_to(
        fig,
        ax1,
        side="right",
        width=0.011,
        height=ax1.get_position().height * 0.52,
        pad=0.012,
        align="bottom",
        y_offset=0.015,
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Cosine similarity", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Statistical summary moved to LaTeX caption for cleaner in-panel appearance

    # Inset: zoomed view of top-left diagonal corner (best-aligned types)
    n_inset = min(8, n_types)
    ax_inset = inset_axes(ax1, width="24%", height="24%", loc="upper right",
                          borderpad=1.5)
    ax_inset.imshow(
        sim_sorted[:n_inset, :n_inset], cmap=cmap, vmin=-0.1, vmax=1.0,
        aspect="auto", interpolation="nearest",
    )
    # Annotate diagonal values in inset
    for ii in range(n_inset):
        val_ii = sim_sorted[ii, ii]
        ax_inset.text(
            ii, ii, f"{val_ii:.2f}", ha="center", va="center",
            fontsize=FONT_HEATMAP_CELL, color="white" if val_ii > 0.5 else "black",
            fontweight="normal",
        )
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_title(f"Top {n_inset}", fontsize=FONT_SMALL, pad=2)
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(1.5)
    ax_inset._clop_styled = True  # prevent save_with_vcd from overriding inset styling

    style_axes(ax1, kind="heatmap")

    # ── F2: Per-type alignment bars with threshold bands ──
    ax2 = fig.add_subplot(gs[1])
    add_panel_label(ax2, chr(ord('a') + label_offset + 1), x=-0.10, y=1.05)
    sorted_idx_asc = np.argsort(diag)
    d_asc = diag[sorted_idx_asc]
    labels_asc = [labels[i] for i in sorted_idx_asc]

    # Draw threshold bands (background shading for quality tiers)
    ax2.axvspan(0.9, 1.08, color=COLORS["good"], alpha=0.06, zorder=0)
    ax2.axvspan(0.7, 0.9, color=COLORS["warn"], alpha=0.06, zorder=0)
    ax2.axvspan(0.0, 0.7, color=COLORS["bad"], alpha=0.06, zorder=0)

    # Thresholds (0.9, 0.7): alignment quality bands per FIGURE_PRESENTATION_POLICY
    color_map = [quality_color(v, (0.9, 0.7)) for v in d_asc]
    ax2.barh(range(n_types), d_asc, color=color_map, height=0.8,
             edgecolor="white", linewidth=0.3)
    set_adaptive_ytick_labels(ax2, labels_asc, max_visible=18, fontsize=10)
    ax2.set_xlabel("Cosine Similarity", fontsize=11)

    # Annotate values on the worst 3 and best 3 bars (with collision avoidance)
    _ann_indices = list(range(min(3, n_types))) + list(range(max(0, n_types - 3), n_types))
    _ann_indices = sorted(set(_ann_indices))  # deduplicate if n_types <= 6
    _prev_y = -999
    for idx_bar in _ann_indices:
        val_bar = d_asc[idx_bar]
        # Skip if too close to previous annotation vertically
        if abs(idx_bar - _prev_y) < 1.5 and idx_bar != _ann_indices[0] and idx_bar < n_types - 3:
            continue
        _prev_y = idx_bar
        ax2.text(
            val_bar + 0.01, idx_bar, f"{val_bar:.3f}",
            va="center", ha="left", fontsize=10, color=COLORS["annotation_medium"],
        )

    # Reference lines with annotations
    ax2.axvline(x=mean_diag, color=COLORS["bad"], linestyle="--", alpha=0.7, linewidth=1.5)
    ax2.text(
        mean_diag, n_types + 1.5, f"\u03bc={mean_diag:.3f}",
        ha="center", va="bottom", fontsize=FONT_SMALL, color=COLORS["bad"],
        clip_on=False,
    )
    ax2.axvline(x=0.9, color=COLORS["good"], linestyle=":", alpha=0.5, linewidth=1.0)
    ax2.axvline(x=0.7, color=COLORS["warn"], linestyle=":", alpha=0.5, linewidth=1.0)

    # Title with quality-tier counts
    ax2.set_title(
        f"Per-Type Alignment\n"
        f"[{n_excellent} excellent / {n_good} good / {n_poor} poor]",
        fontsize=FONT_TITLE,
    )
    ax2.set_xlim(0, 1.08)

    # Add median marker (offset from mean label to avoid overlap)
    ax2.axvline(
        x=median_diag, color=COLORS["heatmap_purple"], linestyle="-.", alpha=0.6, linewidth=1.0,
    )
    # Place median label below the bar area to avoid colliding with the mean label
    ax2.text(
        median_diag, -2.0, f"med={median_diag:.3f}",
        ha="center", va="top", fontsize=10, color=COLORS["heatmap_purple"],
        clip_on=False,
    )

    style_axes(ax2, kind="bar")

    # ── F3: Distribution comparison with statistics ──
    ax3 = fig.add_subplot(gs[2])
    add_panel_label(ax3, chr(ord('a') + label_offset + 2), x=-0.10, y=1.05)

    # Histograms with concise legend entries
    ax3.hist(
        diag, bins=12, alpha=0.7, color=COLORS["real"], edgecolor="white",
        label=f"Diagonal (n={len(diag)})",
        density=True,
        orientation="horizontal",
    )
    ax3.hist(
        off_diag, bins=25, alpha=0.45, color=COLORS["generated"], edgecolor="white",
        label=f"Off-diagonal (n={len(off_diag)})",
        density=True,
        orientation="horizontal",
    )

    # KDE overlay for smoother density estimation
    try:
        from scipy.stats import gaussian_kde
        # Diagonal KDE
        kde_diag = gaussian_kde(diag, bw_method=0.3)
        y_kde = np.linspace(diag.min() - 0.05, diag.max() + 0.05, 200)
        ax3.plot(kde_diag(y_kde), y_kde, color=COLORS["real"],
                 linewidth=1.5, linestyle="-", alpha=0.8)
        # Off-diagonal KDE
        kde_off = gaussian_kde(off_diag, bw_method=0.3)
        y_kde_off = np.linspace(
            off_diag.min() - 0.05, min(off_diag.max() + 0.05, 1.0), 300
        )
        ax3.plot(kde_off(y_kde_off), y_kde_off, color=COLORS["generated"],
                 linewidth=1.5, linestyle="-", alpha=0.7)
    except ImportError:
        pass  # scipy not available, skip KDE

    # Mean lines
    ax3.axhline(y=mean_diag, color=COLORS["real"], linestyle="--", linewidth=1.5)
    ax3.axhline(y=mean_off, color=COLORS["generated"], linestyle=":", linewidth=1.5)
    # Median lines
    ax3.axhline(y=median_diag, color=COLORS["real"], linestyle="-.",
                linewidth=1.0, alpha=0.6)
    ax3.axhline(y=median_off, color=COLORS["generated"], linestyle="-.",
                linewidth=1.0, alpha=0.6)

    ax3.set_ylabel("Cosine Similarity", fontsize=11)
    ax3.set_xlabel("Density", fontsize=11)
    ax3.set_title("Diag vs Off-Diag", fontsize=FONT_TITLE)

    # Tighter y-axis: avoid wasting space on empty negative range
    _ylim_lo = max(off_diag.min() - 0.08, -0.15)
    ax3.set_ylim(_ylim_lo, 1.05)
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=8, prune="both"))
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax3.tick_params(axis="both", labelsize=10)

    # Legend — place in empty region without frame
    ax3.legend(fontsize=10, frameon=False, loc="upper left")

    # Add gridlines for readability
    ax3.grid(True, axis="both", alpha=0.2, linewidth=0.4)

    # Statistical annotation — compact format (detailed stats in caption)
    stat_anno = "Mann\u2013Whitney U test\n"
    stat_anno += f"$d$={cohens_d:.1f}"
    if p_value is not None:
        if p_value < 1e-10:
            stat_anno += ",  $p$<1e-10***"
        elif p_value < 0.001:
            stat_anno += f",  $p$={p_value:.1e}***"
        elif p_value < 0.01:
            stat_anno += f",  $p$={p_value:.3f}**"
        elif p_value < 0.05:
            stat_anno += f",  $p$={p_value:.3f}*"
        else:
            stat_anno += f",  $p$={p_value:.3f} n.s."
    ax3.text(
        0.97, 0.03, stat_anno,
        transform=ax3.transAxes, fontsize=10, va="bottom", ha="right",
        fontweight="normal",
        color=COLORS["annotation_dark"],
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.92,
            edgecolor=COLORS["border_light"], linewidth=0.5,
        ),
        zorder=10,
    )

    style_axes(ax3, kind="default")

    layout_axes_row([ax1, ax2, ax3], widths=[1.18, 0.90, 0.72], gaps=[0.040, 0.020])
    pos1 = ax1.get_position()
    cax.set_position((pos1.x1 + 0.012, pos1.y0 + 0.015, 0.011, pos1.height * 0.52))

    if save:
        path = Path(output_dir) / "fig07_text_cell_alignment.png"
        if save_panel_fn is not None:
            save_panel_fn(fig, path, dpi)
        else:
            save_with_vcd(fig, path, dpi)
    return fig
