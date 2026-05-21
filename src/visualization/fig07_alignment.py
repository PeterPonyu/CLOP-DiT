"""
fig07_alignment.py -- Article Figure 7: Text-Cell Alignment Heatmap.

  F1: Clustered text-prototype x cell-centroid heatmap with a diagonal guide,
      inset zoom, and compact color scale
  F2: Sorted per-type matched-lift bars relative to the row-wise off-diagonal
      median baseline
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

from .direct_layout import bind_figure_region
from .style import (
    COLORS, FONT_HEATMAP_CELL, FONT_SMALL, FONT_TITLE,
    PANEL_OFFSET_LEFT, PANEL_OFFSET_WIDE,
    add_colorbar_safe,
    abbreviate_cell_type, add_panel_label,
    save_with_vcd, set_adaptive_ytick_labels, style_axes,
)
from .explicit_positioning import add_axes_next_to
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

    F1: Clustered text-prototype x cell-centroid heatmap with diagonal guide and
        inset zoom.
    F2: Sorted per-type matched-lift bars relative to each row's off-diagonal
        median baseline.
    F3: Distribution of diagonal vs off-diagonal similarities with KDE overlay,
        Mann-Whitney U p-value, Cohen's d effect size, median markers, and
        bootstrap CI annotation
    """
    from matplotlib.ticker import MaxNLocator
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

    x_labels = [abbreviate_cell_type(type_names.get(int(t), f"T{t}"), max_len=13) for t in unique_types]
    y_labels = [abbreviate_cell_type(type_names.get(int(t), f"T{t}"), max_len=22) for t in unique_types]
    diag = np.diag(sim_matrix)
    mean_diag = diag.mean()
    std_diag = diag.std()
    median_diag = float(np.median(diag))
    off_diag = sim_matrix[~np.eye(n_types, dtype=bool)]
    mean_off = off_diag.mean()
    std_off = off_diag.std()
    median_off = float(np.median(off_diag))
    row_off = sim_matrix.copy()
    np.fill_diagonal(row_off, np.nan)
    row_off_median = np.nanmedian(row_off, axis=1)
    matched_lift = diag - row_off_median
    mean_lift = float(np.mean(matched_lift))
    median_lift = float(np.median(matched_lift))
    n_positive_lift = int(np.sum(matched_lift > 0))

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
    x_labels_sorted = [x_labels[i] for i in sort_order]
    y_labels_sorted = [y_labels[i] for i in sort_order]

    fig = plt.figure(figsize=(15.2, 7.4))
    # Title moved to LaTeX caption
    layout = bind_figure_region(fig, (0.12, 0.14, 0.985, 0.92))
    ax1_slot, ax2_slot, ax3_slot = layout.split_cols([1.56, 0.92, 0.68], gap=[0.048, 0.028])
    ax1_rect = ax1_slot.inset(right=0.004)
    ax2_rect = ax2_slot.inset(left=0.110, right=0.010)
    ax3_rect = ax3_slot.inset(left=0.012)

    # ── F1: Clustered heatmap with annotations ──
    ax1 = ax1_rect.add_axes(fig)
    add_panel_label(ax1, chr(ord('a') + label_offset), x=PANEL_OFFSET_WIDE[0], y=PANEL_OFFSET_WIDE[1])
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
    im.set_rasterized(True)
    step_x = max(10, int(np.ceil(n_types / 6)))
    step_y = max(5, int(np.ceil(n_types / 12)))
    _xtl = [x_labels_sorted[i] if i % step_x == 0 else "" for i in range(n_types)]
    _ytl = [y_labels_sorted[i] if i % step_y == 0 else "" for i in range(n_types)]
    ax1.set_xticks(range(n_types))
    ax1.set_yticks(range(n_types))
    ax1.set_xticklabels(_xtl, rotation=55, fontsize=8, ha="right")
    ax1.set_yticklabels(_ytl, fontsize=10.5, ha="right")
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
        width=0.008,
        height=ax1.get_position().height * 0.42,
        pad=0.012,
        align="bottom",
        y_offset=0.012,
    )
    cbar = add_colorbar_safe(im, ax=ax1, cax=cax, shrink=1.0, pad=0.0, aspect=14)
    cbar.set_label("")
    cbar.ax.set_title("Cos.\nsim.", fontsize=10, pad=4)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Statistical summary moved to LaTeX caption for cleaner in-panel appearance

    # Inset: zoomed view of top-left diagonal corner (best-aligned types)
    n_inset = min(6, n_types)
    ax_inset = ax1.inset_axes([0.68, 0.68, 0.24, 0.24])  # native inset (PDF-safe)
    im_inset = ax_inset.imshow(
        sim_sorted[:n_inset, :n_inset], cmap=cmap, vmin=-0.1, vmax=1.0,
        aspect="auto", interpolation="nearest",
    )
    im_inset.set_rasterized(True)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(1.5)
    ax_inset._clop_styled = True  # prevent save_with_vcd from overriding inset styling

    style_axes(ax1, kind="heatmap")

    # ── F2: Per-type matched lift over row-wise off-diagonal baseline ──
    ax2 = ax2_rect.add_axes(fig)
    add_panel_label(ax2, chr(ord('a') + label_offset + 1), x=PANEL_OFFSET_LEFT[0], y=PANEL_OFFSET_LEFT[1])
    sorted_idx_asc = np.argsort(matched_lift)
    d_asc = matched_lift[sorted_idx_asc]
    labels_asc = [y_labels[i] for i in sorted_idx_asc]

    # Raw text-cell cosines are centred near zero after whitening/projection.
    # Plotting a diagonal lift is more faithful than borrowing thresholds from
    # the generated-cell centroid-fidelity panels.
    color_map = [COLORS["real"] if v >= 0 else COLORS["generated"] for v in d_asc]
    ax2.barh(range(n_types), d_asc, color=color_map, height=0.8,
             edgecolor="white", linewidth=0.3)
    set_adaptive_ytick_labels(ax2, labels_asc, max_visible=18, fontsize=10)
    ax2.set_xlabel("Matched Lift (diag - row off-diag median)", fontsize=11)

    ax2.axvline(x=0, color=COLORS["annotation_dark"], linestyle="-", alpha=0.75, linewidth=1.0)
    ax2.axvline(x=mean_lift, color=COLORS["bad"], linestyle="--", alpha=0.7, linewidth=1.3)
    ax2.axvline(x=median_lift, color=COLORS["heatmap_purple"], linestyle="-.", alpha=0.65, linewidth=1.0)
    x_abs = float(np.nanmax(np.abs(d_asc))) if len(d_asc) else 0.05
    x_abs = max(x_abs, 0.035)
    ax2.set_xlim(-x_abs * 1.20, x_abs * 1.20)
    ax2.set_title("Per-Type Matched Lift", fontsize=FONT_TITLE, pad=8, x=0.58)
    ax2.text(
        0.98, 0.98,
        f"\u0394\u03bc={mean_lift:.3f} | med={median_lift:.3f}\n"
        f"{n_positive_lift}/{n_types} positive lifts",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_SMALL - 1,
        color=COLORS["annotation_dark"],
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="none", alpha=0.85),
        zorder=10,
    )

    style_axes(ax2, kind="bar")

    # ── F3: Distribution comparison with statistics ──
    ax3 = ax3_rect.add_axes(fig)
    add_panel_label(
        ax3,
        chr(ord('a') + label_offset + 2),
        x=PANEL_OFFSET_WIDE[0],
        y=PANEL_OFFSET_LEFT[1],
    )

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
    _ylim_lo = min(off_diag.min() - 0.02, diag.min() - 0.02)
    ax3.set_ylim(_ylim_lo, 1.05)
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=8, prune="both"))
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax3.tick_params(axis="both", labelsize=10)

    # Legend — place in empty region without frame
    ax3.legend(fontsize=10, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1)

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

    if save:
        path = Path(output_dir) / "fig03c_text_cell_alignment.png"
        if save_panel_fn is not None:
            save_panel_fn(fig, path, dpi)
        else:
            save_with_vcd(fig, path, dpi)
    return fig
