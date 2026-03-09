"""
panels_heatmaps.py — Panels F (text-cell heatmap) and G (per-type generation) for CLOP-DiT.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from . import io as viz_io
from .style import (
    COLORS, FONT_HEATMAP_CELL, FONT_LEGEND, FONT_SMALL, FONT_TITLE,
    abbreviate_cell_type, add_colorbar_safe, add_panel_label,
    quality_color, set_adaptive_ytick_labels, style_axes,
)
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

    fig = plt.figure(figsize=(15.5, 9.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.80, 0.68], wspace=0.52)
    # Title moved to LaTeX caption

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
    step = 3
    _xtl = [labels_sorted[i] if i % step == 0 else "" for i in range(n_types)]
    _ytl = [labels_sorted[i] if i % step == 0 else "" for i in range(n_types)]
    ax1.set_xticks(range(n_types))
    ax1.set_yticks(range(n_types))
    ax1.set_xticklabels(_xtl, rotation=90, fontsize=8, ha="center")
    ax1.set_yticklabels(_ytl, fontsize=8, ha="right")
    ax1.set_ylabel("Cell Type (text prototypes)", fontsize=10)
    ax1.set_title("Cosine Similarity (sorted by diagonal)", fontsize=11)

    # Draw diagonal guide line
    ax1.plot(
        [0, n_types - 1], [0, n_types - 1],
        color="white", linewidth=0.8, linestyle=":", alpha=0.6, zorder=3,
    )

    # Mark top off-diagonal confusions with value annotations
    off_diag_matrix = sim_sorted.copy()
    np.fill_diagonal(off_diag_matrix, -1)
    confusion_pairs = []
    _confusion_positions = []  # track positions for collision avoidance
    for _ in range(min(3, n_types)):
        idx = np.unravel_index(off_diag_matrix.argmax(), off_diag_matrix.shape)
        val = off_diag_matrix[idx]
        if val < 0.3:
            break
        ax1.plot(
            idx[1], idx[0], "x", color=COLORS["confusion_marker"],
            markersize=7, markeredgewidth=1.8, zorder=4,
        )
        # Compute annotation offset; shift further if close to a previous label
        ann_offset_x, ann_offset_y = 6, -6
        for px, py in _confusion_positions:
            if abs(idx[1] - px) < 3 and abs(idx[0] - py) < 3:
                # Shift up if near bottom edge, down otherwise
                if idx[0] > n_types * 0.7:
                    ann_offset_y += 14
                else:
                    ann_offset_y -= 14
        _confusion_positions.append((idx[1], idx[0]))
        # Annotate with the similarity value
        ax1.annotate(
            f"{val:.2f}",
            xy=(idx[1], idx[0]),
            xytext=(ann_offset_x, ann_offset_y),
            textcoords="offset points",
            fontsize=FONT_SMALL,
            color=COLORS["confusion_marker"],
            fontweight="normal",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6,
                      edgecolor="none"),
            zorder=5,
        )
        confusion_pairs.append(
            (abbreviate_cell_type(labels_sorted[idx[0]], max_len=12),
             abbreviate_cell_type(labels_sorted[idx[1]], max_len=12), val)
        )
        off_diag_matrix[idx] = -1

    try:
        cbar = add_colorbar_safe(im, ax=ax1, label="Cosine similarity", shrink=0.72, pad=0.04)
    except Exception:
        cbar = fig.colorbar(im, ax=ax1, shrink=0.72, pad=0.04)
        cbar.set_label("Cosine similarity", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    cbar.ax.axhline(y=mean_diag, color="white", linewidth=1.5, linestyle="--")
    cbar.ax.axhline(y=mean_off, color="black", linewidth=1, linestyle=":")
    # Add text labels on colorbar reference lines
    cbar.ax.text(
        1.1, mean_diag, f"diag={mean_diag:.3f}",
        transform=cbar.ax.get_yaxis_transform(),
        fontsize=8, color="#333333", va="bottom", ha="left",
        fontweight="normal",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.85,
                  edgecolor="none"),
    )
    cbar.ax.text(
        1.1, mean_off, f"off={mean_off:.3f}",
        transform=cbar.ax.get_yaxis_transform(),
        fontsize=8, color="#333333", va="top", ha="left",
        fontweight="normal",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.85,
                  edgecolor="none"),
    )

    # Statistical summary moved to LaTeX caption for cleaner in-panel appearance

    # Inset: zoomed view of top-left diagonal corner (best-aligned types)
    n_inset = min(12, n_types)
    ax_inset = inset_axes(ax1, width="28%", height="28%", loc="upper right",
                          borderpad=2.5)
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
    ax_inset.set_title(f"Top {n_inset} (zoom)", fontsize=FONT_SMALL, pad=2)
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
    set_adaptive_ytick_labels(ax2, labels_asc, max_visible=22, fontsize=6)
    ax2.set_xlabel("Cosine Similarity", fontsize=10)

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
            va="center", ha="left", fontsize=FONT_HEATMAP_CELL, color=COLORS["annotation_medium"],
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
        ha="center", va="top", fontsize=FONT_HEATMAP_CELL, color=COLORS["heatmap_purple"],
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

    ax3.set_ylabel("Cosine Similarity", fontsize=10)
    ax3.set_xlabel("Density", fontsize=10)
    ax3.set_title("Diag vs Off-Diag", fontsize=FONT_TITLE)

    # Tighter y-axis: avoid wasting space on empty negative range
    _ylim_lo = max(off_diag.min() - 0.08, -0.15)
    ax3.set_ylim(_ylim_lo, 1.05)
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=8, prune="both"))
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax3.tick_params(axis="both", labelsize=8)

    # Legend — place in empty region with background for visibility
    ax3.legend(fontsize=8, frameon=True, loc="center left",
               facecolor="white", edgecolor="#cccccc", framealpha=0.9)

    # Add gridlines for readability
    ax3.grid(True, axis="both", alpha=0.2, linewidth=0.4)

    # Statistical annotation text box (compact single-line format)
    stat_anno = f"$d$={cohens_d:.1f}"
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
    # Bootstrap CI for centroid cosine if available
    if bootstrap_ci and bootstrap_ci.get("lower") is not None:
        stat_anno += (
            f"\n95% CI [{bootstrap_ci['lower']:.3f}, {bootstrap_ci['upper']:.3f}]"
        )
    stat_anno += f"\nSep.={separation_ratio:.1f}x"
    ax3.text(
        0.97, 0.03, stat_anno,
        transform=ax3.transAxes, fontsize=FONT_SMALL, va="bottom", ha="right",
        fontweight="normal",
        color=COLORS["annotation_dark"],
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
            edgecolor="#cccccc", linewidth=0.5,
        ),
        zorder=10,
    )

    # Bracket showing the separation between mean diagonal and mean off-diagonal
    bracket_x_ax = 0.75
    ax3.annotate(
        "",
        xy=(bracket_x_ax, mean_diag),
        xycoords=("axes fraction", "data"),
        xytext=(bracket_x_ax, mean_off),
        textcoords=("axes fraction", "data"),
        arrowprops=dict(
            arrowstyle="<->", color=COLORS["annotation_medium"], lw=1.0,
            connectionstyle="arc3,rad=0",
        ),
    )
    mid_y = (mean_diag + mean_off) / 2
    ax3.text(
        bracket_x_ax - 0.02, mid_y, f"\u0394={mean_diag - mean_off:.3f}",
        transform=ax3.get_yaxis_transform(),
        fontsize=FONT_SMALL, ha="right", va="center",
        fontweight="normal",
        color=COLORS["annotation_medium"],
    )

    style_axes(ax3, kind="default")

    fig._clop_layout_rect = (0.02, 0.06, 0.98, 0.95)

    if save:
        viz_io.save_to_dir(fig, "panel_f_text_cell_heatmap", output_dir, dpi, save_panel_fn)
    return fig


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
    short_names = [abbreviate_cell_type(n, max_len=22) for n in names]

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

    fig = plt.figure(figsize=(14.0, 7.2))
    gs_g = fig.add_gridspec(1, 3, wspace=0.50, width_ratios=[1.2, 1.2, 1.0])
    fig._clop_layout_rect = (0.02, 0.06, 0.98, 0.95)
    summary = data.get("summary", {})
    # Title moved to LaTeX caption

    # G1: Centroid cosine (sorted)
    from matplotlib.ticker import MaxNLocator
    ax = fig.add_subplot(gs_g[0])
    add_panel_label(ax, chr(ord('a') + label_offset), x=-0.10, y=1.05)
    sorted_idx = np.argsort(cosines)
    sorted_cos = [cosines[i] for i in sorted_idx]
    sorted_names_cos = [short_names[i] for i in sorted_idx]
    sorted_type_ids = [type_ids[i] for i in sorted_idx]
    # Thresholds (0.9, 0.7): centroid cosine quality bands per FIGURE_PRESENTATION_POLICY
    colors = []
    for v, t_id in zip(sorted_cos, sorted_type_ids):
        if t_id is not None and int(t_id) in collapsed_type_ids:
            colors.append(COLORS["bad"])
        else:
            colors.append(quality_color(v, (0.9, 0.7)))
    ax.barh(range(len(sorted_cos)), sorted_cos, color=colors, height=0.8)
    set_adaptive_ytick_labels(ax, sorted_names_cos, max_visible=25, fontsize=FONT_HEATMAP_CELL)
    ax.set_xlabel("Centroid Cosine Similarity")
    ax.set_title("Real\u2194Gen Centroid Cosine", fontsize=11)
    ax.axvline(
        x=summary.get("mean_centroid_cosine", 0), color=COLORS["bad"],
        linestyle="--", alpha=0.5,
        label="mean (see caption)",
    )
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))
    ax.legend(fontsize=FONT_LEGEND, frameon=False, loc="lower right")
    style_axes(ax, kind="bar")

    # G2: Fr\u00e9chet outlier profile
    ax = fig.add_subplot(gs_g[1])
    add_panel_label(ax, chr(ord('a') + label_offset + 1), x=-0.10, y=1.05)
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
        set_adaptive_ytick_labels(ax, fd_names, max_visible=25, fontsize=FONT_HEATMAP_CELL)
        ax.set_xlabel("Fr\u00e9chet Distance (lower = better)")
        ax.set_title("Fr\u00e9chet Outlier Profile")
        ax.legend(fontsize=FONT_LEGEND, frameon=False, loc="lower right")
        ax.set_xlim(0, float(np.nanmax(fd_vals)) * 1.12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
        style_axes(ax, kind="bar")
    else:
        ax.text(
            0.5, 0.5, "No valid FD values",
            ha="center", va="center", transform=ax.transAxes,
        )

    # G3: Cosine vs abundance with FD bubble size and diversity color
    ax = fig.add_subplot(gs_g[2])
    add_panel_label(ax, chr(ord('a') + label_offset + 2), x=-0.10, y=1.05)
    fd_for_size = np.where(fd_valid, fd_array, np.nanmedian(fd_array[fd_valid]) if fd_valid.any() else 1.0)
    fd_min = float(np.nanmin(fd_for_size)) if np.isfinite(fd_for_size).any() else 0.0
    fd_ptp = float(np.nanmax(fd_for_size) - fd_min) if np.isfinite(fd_for_size).any() else 1.0
    fd_ptp = fd_ptp or 1.0
    bubble_sizes = 50 + 220 * np.clip((fd_for_size - fd_min) / fd_ptp, 0, 1)
    x_vals = np.log10(np.maximum(n_real_array, 1))

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
        try:
            cbar = add_colorbar_safe(sc, ax=ax, label="Diversity ratio", shrink=0.65, pad=0.10)
        except Exception:
            cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.10)
            cbar.set_label("Diversity ratio", fontsize=10)
        cbar.ax.tick_params(labelsize=8)
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

    worst_idx = np.argsort(cos_array)[:5]
    label_offsets = [(-34, -12), (10, -10), (-30, 10), (12, 10), (16, -18)]
    for rank, i in enumerate(worst_idx):
        if cos_array[i] < 0.94:
            x_offset, y_offset = label_offsets[rank % len(label_offsets)]
            if x_vals[i] > np.median(x_vals):
                x_offset = min(x_offset, -10)
            else:
                x_offset = max(x_offset, 10)
            ax.annotate(
                abbreviate_cell_type(short_names[i], max_len=14),
                (x_vals[i], cos_array[i]),
                fontsize=7,
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.75),
            )
    ax.set_xlabel("log10(Number of Real Cells)")
    ax.set_ylabel("Centroid Cosine Similarity")
    ax.set_title("Fidelity vs Abundance")
    ax.axhline(y=0.9, color=COLORS["good"], linestyle=":", alpha=0.4, label="Target (0.9)")
    ax.legend(fontsize=FONT_LEGEND, frameon=False, loc="upper left")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlim(x_vals.min() - 0.10, x_vals.max() + 0.10)
    style_axes(ax, kind="scatter")

    if save:
        viz_io.save_to_dir(fig, "panel_g_per_type_generation", output_dir, dpi, save_panel_fn)
    return fig
