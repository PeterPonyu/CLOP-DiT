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
from .style import COLORS, FONT_LEGEND, add_colorbar_safe, quality_color, set_dense_tick_labels, set_figure_suptitle
from src.utils.paths import FIG_DIR

logger = logging.getLogger(__name__)


def plot_text_cell_heatmap(
    cache_dir: str = "data/cache",
    type_names: Optional[Dict[int, str]] = None,
    output_dir: Optional[str] = None,
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Enhanced 69x69 text-cell alignment heatmap with rich annotations.

    F1: Clustered heatmap with diagonal highlight and off-diagonal structure
    F2: Sorted per-type alignment bars with threshold bands
    F3: Distribution of diagonal vs off-diagonal similarities
    """
    from matplotlib.patches import Rectangle  # noqa: F401 (kept for parity)

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

    unique_types = np.sort(np.unique(group_ids))
    n_types = len(unique_types)

    text_centroids = np.zeros((n_types, proj_text.shape[1]), dtype=np.float32)
    cell_centroids = np.zeros((n_types, proj_cells.shape[1]), dtype=np.float32)
    type_counts = np.zeros(n_types, dtype=int)
    for i, t in enumerate(unique_types):
        mask = group_ids == t
        type_counts[i] = mask.sum()
        tc = proj_text[mask].mean(axis=0)
        text_centroids[i] = tc / (np.linalg.norm(tc) + 1e-8)
        cc = proj_cells[mask].mean(axis=0)
        cell_centroids[i] = cc / (np.linalg.norm(cc) + 1e-8)

    sim_matrix = text_centroids @ cell_centroids.T

    labels = [type_names.get(int(t), f"T{t}")[:18] for t in unique_types]
    diag = np.diag(sim_matrix)
    mean_diag = diag.mean()
    off_diag = sim_matrix[~np.eye(n_types, dtype=bool)]
    mean_off = off_diag.mean()

    # Reorder by diagonal similarity for visual clarity
    sort_order = np.argsort(-diag)
    sim_sorted = sim_matrix[sort_order][:, sort_order]
    labels_sorted = [labels[i] for i in sort_order]

    fig = plt.figure(figsize=(14.0, 9.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 0.7, 0.5], wspace=0.55)
    set_figure_suptitle(fig, "Text–Cell Alignment", fontsize=11)

    # ── F1: Clustered heatmap with annotations ──
    ax1 = fig.add_subplot(gs[0])
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
    step = 7
    _xtl = [labels_sorted[i] if i % step == 0 else "" for i in range(n_types)]
    _ytl = [labels_sorted[i] if i % step == 0 else "" for i in range(n_types)]
    ax1.set_xticks(range(n_types))
    ax1.set_yticks(range(n_types))
    ax1.set_xticklabels(_xtl, rotation=90, fontsize=8, ha="center")
    ax1.set_yticklabels(_ytl, fontsize=8, ha="right")
    ax1.set_ylabel("Cell Type (text prototypes)", fontsize=10)
    ax1.set_title("Cosine Similarity (sorted)", fontsize=11)

    off_diag_matrix = sim_sorted.copy()
    np.fill_diagonal(off_diag_matrix, -1)
    for _ in range(min(3, n_types)):
        idx = np.unravel_index(off_diag_matrix.argmax(), off_diag_matrix.shape)
        val = off_diag_matrix[idx]
        if val < 0.3:
            break
        ax1.plot(
            idx[1], idx[0], "x", color="lime",
            markersize=6, markeredgewidth=1.5, zorder=4,
        )
        off_diag_matrix[idx] = -1

    cbar = add_colorbar_safe(im, ax=ax1, label="Cosine similarity", shrink=0.52, pad=0.12)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.axhline(y=mean_diag, color="white", linewidth=1.5, linestyle="--")
    cbar.ax.axhline(y=mean_off, color="black", linewidth=1, linestyle=":")

    # ── F2: Per-type alignment bars ──
    ax2 = fig.add_subplot(gs[1])
    sorted_idx_asc = np.argsort(diag)
    d_asc = diag[sorted_idx_asc]
    labels_asc = [labels[i] for i in sorted_idx_asc]

    # Thresholds (0.9, 0.7): alignment quality bands per FIGURE_PRESENTATION_POLICY
    color_map = [quality_color(v, (0.9, 0.7)) for v in d_asc]
    ax2.barh(range(n_types), d_asc, color=color_map, height=0.8,
             edgecolor="white", linewidth=0.3)
    ax2.set_yticks(range(n_types))
    _step2 = 7
    _ytl2 = [labels_asc[i] if i % _step2 == 0 else "" for i in range(n_types)]
    ax2.set_yticklabels(_ytl2, fontsize=8, ha="right")
    set_dense_tick_labels(ax2, axis="y", max_labels=10, fontsize=8, rotation=0)
    ax2.set_title("Per-Type Alignment", fontsize=11)
    ax2.axvline(x=mean_diag, color=COLORS["bad"], linestyle="--", alpha=0.7, linewidth=1.5)
    ax2.set_xlim(0, 1.05)

    # ── F3: Distribution comparison ──
    ax3 = fig.add_subplot(gs[2])
    # Legend keys kept short; μ values are in suptitle/caption
    ax3.hist(
        diag, bins=10, alpha=0.7, color="#1976D2", edgecolor="white",
        label="Diag", density=True,
        orientation="horizontal",
    )
    ax3.hist(
        off_diag, bins=20, alpha=0.5, color="#FF7043", edgecolor="white",
        label="Off-diag", density=True,
        orientation="horizontal",
    )
    ax3.axhline(y=mean_diag, color="#1565C0", linestyle="--", linewidth=1.5)
    ax3.axhline(y=mean_off, color="#E64A19", linestyle=":", linewidth=1.5)
    ax3.set_ylabel("Cosine Similarity", fontsize=10)
    ax3.set_xlabel("Density", fontsize=10)
    ax3.set_title("Distribution", fontsize=11)
    ax3.legend(fontsize=FONT_LEGEND, frameon=False, loc="upper left")
    ax3.set_ylim(min(off_diag.min() * 1.05, -0.1), 1.05)
    from matplotlib.ticker import MaxNLocator
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

    fig.subplots_adjust(bottom=0.18)

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
) -> Optional[plt.Figure]:
    """Per-type generation quality with heterogeneity emphasis.

    G1: Centroid cosine per type (sorted, hardest types highlighted)
    G2: Frechet outlier profile (sorted, mean-anchored)
    G3: Fidelity vs abundance with FD bubble size and diversity-ratio colour
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
    collapsed_summary_text = ""
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
            summary = t1.get("summary", {})
            n_collapsed = summary.get("n_collapsed")
            n_healthy = summary.get("n_healthy")
            if n_collapsed is not None and n_healthy is not None:
                total = n_collapsed + n_healthy
                collapsed_summary_text = f" (collapsed types={n_collapsed}/{total}, DivR<0.5)"
        except Exception:
            diversity_by_type_id = {}
            collapsed_type_ids = set()
            collapsed_summary_text = ""

    type_ids = [per_type[n].get("type_id") for n in names]
    short_names = [n[:20] for n in names]

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

    fig = plt.figure(figsize=(12.2, 6.8))
    gs_g = fig.add_gridspec(1, 3, wspace=0.55)
    summary = data.get("summary", {})
    overall = data.get("overall", {})
    set_figure_suptitle(fig, "Per-Type Generation Fidelity", fontsize=11)

    # G1: Centroid cosine (sorted)
    ax = fig.add_subplot(gs_g[0])
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
    ax.set_yticks(range(len(sorted_cos)))
    _ytlg = [
        sorted_names_cos[i] if i % 7 == 0 else ""
        for i in range(len(sorted_cos))
    ]
    ax.set_yticklabels(_ytlg, fontsize=8, ha="right")
    set_dense_tick_labels(ax, axis="y", max_labels=10, fontsize=8, rotation=0)
    ax.set_xlabel("Centroid Cosine Similarity")
    ax.set_title("Real\u2194Gen Centroid Cosine")
    ax.axvline(
        x=summary.get("mean_centroid_cosine", 0), color=COLORS["bad"],
        linestyle="--", alpha=0.5,
        label="mean (see caption)",
    )
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=FONT_LEGEND, frameon=False, loc="lower right")

    # G2: Fréchet outlier profile
    ax = fig.add_subplot(gs_g[1])
    if fd_valid.any():
        fd_idx = np.where(fd_valid)[0][np.argsort(fd_array[fd_valid])]
        fd_vals = fd_array[fd_idx]
        fd_names = [short_names[i] for i in fd_idx]
        fd_min = float(np.nanmin(fd_vals))
        fd_ptp = float(np.nanmax(fd_vals) - fd_min) or 1.0
        fd_norm = np.clip((fd_vals - fd_min) / fd_ptp, 0, 1)
        colors_fd = plt.cm.RdYlGn_r(fd_norm)
        y_pos = np.arange(len(fd_vals))
        ax.hlines(y_pos, 0, fd_vals, color=colors_fd, linewidth=2.8, alpha=0.85)
        ax.scatter(fd_vals, y_pos, s=28 + 70 * fd_norm, color=colors_fd, edgecolors="white", linewidths=0.4, zorder=3)
        if np.isfinite(fd_mean):
            ax.axvline(fd_mean, color=COLORS["bad"], linestyle="--", alpha=0.7, linewidth=1.5, label="mean (see caption)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([fd_names[i] if i % 7 == 0 else "" for i in range(len(fd_vals))], fontsize=8, ha="right")
        set_dense_tick_labels(ax, axis="y", max_labels=10, fontsize=8, rotation=0)
        ax.set_xlabel("Fr\u00e9chet Distance (lower = better)")
        ax.set_title("Fr\u00e9chet Outlier Profile")
        ax.legend(fontsize=FONT_LEGEND, frameon=False, loc="lower right")
    else:
        ax.text(
            0.5, 0.5, "No valid FD values",
            ha="center", va="center", transform=ax.transAxes,
        )

    # G3: Cosine vs abundance with FD bubble size and diversity colouring
    ax = fig.add_subplot(gs_g[2])
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
        cbar = add_colorbar_safe(sc, ax=ax, label="Diversity ratio", shrink=0.65, pad=0.10)
        cbar.ax.tick_params(labelsize=8)
    else:
        ax.scatter(
            x_vals,
            cos_array,
            c="#3F51B5",
            s=bubble_sizes,
            alpha=0.78,
            edgecolors="white",
            linewidth=0.6,
            clip_on=False,
        )

    if len(x_vals) > 1:
        slope, intercept = np.polyfit(x_vals, cos_array, deg=1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="#263238", linestyle="--", linewidth=1.3, label="Trend")

    worst_idx = np.argsort(cos_array)[:2]
    for i in worst_idx:
        if cos_array[i] < 0.9:
            x_offset = -28 if x_vals[i] > np.median(x_vals) else 5
            ax.annotate(
                short_names[i][:12], (x_vals[i], cos_array[i]),
                fontsize=8, xytext=(x_offset, -5), textcoords="offset points",
            )
    ax.set_xlabel("log10(Number of Real Cells)")
    ax.set_ylabel("Centroid Cosine Similarity")
    ax.set_title("Fidelity vs Abundance")
    ax.axhline(y=0.9, color=COLORS["good"], linestyle=":", alpha=0.4, label="Target (0.9)")
    ax.legend(fontsize=FONT_LEGEND, frameon=False, loc="upper left")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.set_xlim(x_vals.min() - 0.10, x_vals.max() + 0.10)

    fig.subplots_adjust(left=0.18, right=0.95)

    if save:
        viz_io.save_to_dir(fig, "panel_g_per_type_generation", output_dir, dpi, save_panel_fn)
    return fig
