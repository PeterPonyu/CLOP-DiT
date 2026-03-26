"""
fig11_conditioning.py — Article Figure 11: conditioning landscape (UMAP/PCA).

Extracted from panels_conditioning.py (formerly Panel M).
Plotting only; scripts run sweeps/generation and pass pre-computed data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from .direct_layout import bind_figure_region
from .explicit_positioning import add_axes_next_to, add_shared_legend_axes
from .style import (
    COLORS, TYPE_PALETTE, apply_style, save_with_vcd,
    add_panel_label, abbreviate_cell_type,
    FONT_LABEL, FONT_TITLE, FONT_TICK, FONT_TICK_DENSE, FONT_ANNOTATION,
    FONT_HEATMAP_CELL,
)

logger = logging.getLogger(__name__)


def _adjust_axes_rect(ax: plt.Axes, *, dx: float = 0.0, width_scale: float = 1.0) -> None:
    """Apply a small horizontal nudge/resize in figure coordinates."""
    pos = ax.get_position()
    new_x0 = pos.x0 + dx
    new_w = pos.width * width_scale
    ax.set_position([new_x0, pos.y0, new_w, pos.height])


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
    full_dim_data: Optional[np.ndarray] = None,
    full_dim_labels: Optional[np.ndarray] = None,
    full_dim_source: Optional[np.ndarray] = None,
    label_offset: int = 0,
) -> Path:
    """Panel M / Figure 11: Conditioning mode comparison (plot only). coords are (n, 2) PCA/UMAP."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    n_modes = 1 + len(mode_diversity)  # real + each mode
    _fw = 15.0  # Fixed width for reproducible layout
    has_row3 = full_dim_data is not None and len(full_dim_data) > 0
    if has_row3:
        fig = plt.figure(figsize=(_fw, 9.0))
        row_regions = bind_figure_region(fig, (0.05, 0.12, 0.97, 0.95)).split_rows(
            [1.34, 1.02, 0.70],
            gap=[0.108, 0.074],
        )
    else:
        fig = plt.figure(figsize=(_fw, 6.7))
        row_regions = bind_figure_region(fig, (0.05, 0.12, 0.97, 0.95)).split_rows(
            [1.30, 0.96],
            gap=0.104,
        )
    top_widths = [1.0] * n_modes
    if n_modes > 1:
        top_widths[-1] = 1.04
    axes = [region.add_axes(fig) for region in row_regions[0].split_cols(top_widths, wspace=0.36)]

    # Panel labels: placed with enough clearance for single-line titles
    _panel_label_y = 1.09
    add_panel_label(axes[0], chr(ord('a') + label_offset), x=-0.14, y=_panel_label_y)

    type_to_color = {tid: TYPE_PALETTE[i % len(TYPE_PALETTE)] for i, tid in enumerate(selected_types)}
    type_to_name = {
        tid: abbreviate_cell_type(
            type_names.get(int(tid), f"Type_{tid}") if type_names else f"Type_{tid}",
            max_len=15,
        )
        for tid in selected_types
    }

    # ── Helper: clean mode name for display ──
    def _short_mode(name: str) -> str:
        return name.split(" (")[0]

    def plot_one(ax, mask, title, alpha=0.45, size=8, *, show_ylabel=True):
        for tid in selected_types:
            tmask = mask & (combined_labels == tid)
            ax.scatter(coords[tmask, 0], coords[tmask, 1],
                       c=[type_to_color[tid]], s=size, alpha=alpha,
                       edgecolors="white", linewidths=0.2,
                       label=type_to_name[tid])
        ax.set_title(title, fontsize=FONT_TITLE - 1, pad=10)
        ax.set_xlabel("", fontsize=FONT_LABEL)
        if show_ylabel:
            ax.set_ylabel("PC2", fontsize=FONT_LABEL)
        else:
            ax.set_ylabel("", fontsize=FONT_LABEL)
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(labelsize=FONT_TICK)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))

    # ── Row 1: PCA scatter for Real + each conditioning mode ──
    real_mask_bool = combined_source == "Real"
    plot_one(axes[0], real_mask_bool,
             f"Real (n={n_real})",
             alpha=0.35, size=4)

    mode_counts = mode_counts or {}
    for i, mode_name in enumerate(mode_diversity.keys()):
        mode_mask = combined_source == mode_name
        count = mode_counts.get(mode_name, mode_mask.sum())
        plot_one(
            axes[i + 1],
            mode_mask,
            f"{_short_mode(mode_name)} (n={count})",
            show_ylabel=False,
        )

    # ── Row 2: Quantitative summaries ──
    bottom_regions = row_regions[1].split_cols([1.08, 1.02, 0.92], wspace=0.28)
    ax_b1 = bottom_regions[0].add_axes(fig)
    add_panel_label(ax_b1, chr(ord('a') + label_offset + 1), x=-0.18, y=_panel_label_y)
    ax_b2 = bottom_regions[1].add_axes(fig)
    add_panel_label(ax_b2, chr(ord('a') + label_offset + 2), x=-0.18, y=_panel_label_y)
    ax_b3 = bottom_regions[2].add_axes(fig)
    add_panel_label(ax_b3, chr(ord('a') + label_offset + 3), x=-0.18, y=_panel_label_y)
    _adjust_axes_rect(ax_b1, width_scale=0.90)
    _adjust_axes_rect(ax_b3, dx=ax_b3.get_position().width * 0.08, width_scale=0.92)

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
        ax_b1.set_xticklabels([_short_mode(m) for m in shift_labels],
                               rotation=0, ha="center", fontsize=FONT_TICK)
        ax_b1.set_ylabel("Mean centroid shift (PC units)", fontsize=FONT_LABEL)
        ax_b1.set_title("Centroid Shift", fontsize=FONT_TITLE, x=0.58, pad=2)
    else:
        ax_b1.text(0.5, 0.5, "No centroid shift data", ha="center", va="center", transform=ax_b1.transAxes)
        ax_b1.set_title("Centroid Shift", fontsize=FONT_TITLE, x=0.58, pad=2)

    div_labels = ["Real"] + list(mode_diversity.keys())
    div_values = [real_diversity] + [mode_diversity[m] for m in mode_diversity.keys()]
    div_colors = [COLORS["real"]] + [COLORS["generated"] for _ in mode_diversity.keys()]
    xdiv = np.arange(len(div_labels))
    ax_b2.bar(xdiv, div_values, color=div_colors, alpha=0.85, edgecolor="white")
    ax_b2.axhline(real_diversity, color=COLORS["real"], linestyle="--", linewidth=1.2, alpha=0.8)
    ax_b2.set_xticks(xdiv)
    ax_b2.set_xticklabels([_short_mode(m) for m in div_labels],
                           rotation=0, ha="center", fontsize=FONT_TICK)
    ax_b2.set_ylabel("Within-type diversity", fontsize=FONT_LABEL)
    ax_b2.set_title("Diversity by Mode", fontsize=FONT_TITLE, pad=2)

    if per_type_shift_distributions:
        ax_b3.boxplot(
            per_type_shift_distributions,
            labels=[_short_mode(m) for m in shift_labels],
            patch_artist=True,
            boxprops=dict(facecolor=COLORS["bg_light"], edgecolor=COLORS["neutral"]),
            medianprops=dict(color=COLORS["bad"], linewidth=1.3),
            whiskerprops=dict(color=COLORS["neutral"], linewidth=1.0),
            capprops=dict(color=COLORS["neutral"], linewidth=1.0),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=COLORS["warn"], markeredgecolor="none", alpha=0.6),
        )
        ax_b3.tick_params(axis="x", labelrotation=0, labelsize=FONT_TICK)
        ax_b3.set_ylabel("Per-type centroid shift", fontsize=FONT_LABEL)
        ax_b3.set_title("Shift Distribution", fontsize=FONT_TITLE, pad=2)
        ax_b3.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
    else:
        ax_b3.text(0.5, 0.5, "No shift distribution data", ha="center", va="center", transform=ax_b3.transAxes)
        ax_b3.set_title("Shift Distribution", fontsize=FONT_TITLE, pad=2)

    # ── Row 3: KNN accuracy, diversity heatmap, pairwise cosine violin ──
    if has_row3:
        from sklearn.decomposition import PCA as _PCA
        from sklearn.neighbors import KNeighborsClassifier

        row3_regions = row_regions[2].split_cols([0.98, 1.00, 0.94], wspace=0.34)
        ax_c1 = row3_regions[0].add_axes(fig)
        add_panel_label(ax_c1, chr(ord('a') + label_offset + 4), x=-0.18, y=_panel_label_y)
        ax_c2 = row3_regions[1].add_axes(fig)
        add_panel_label(ax_c2, chr(ord('a') + label_offset + 5), x=-0.18, y=_panel_label_y)
        ax_c3 = row3_regions[2].add_axes(fig)
        add_panel_label(ax_c3, chr(ord('a') + label_offset + 6), x=-0.04, y=_panel_label_y)
        _adjust_axes_rect(ax_c1, width_scale=0.90)
        _adjust_axes_rect(ax_c3, dx=ax_c3.get_position().width * 0.08, width_scale=0.92)

        # PCA reduce full-dim data for KNN
        pca_full = _PCA(n_components=30, random_state=42)
        fd_pca = pca_full.fit_transform(full_dim_data)

        real_fd_mask = full_dim_source == "Real"
        real_fd_pca = fd_pca[real_fd_mask]
        real_fd_labels = full_dim_labels[real_fd_mask]

        # Train KNN on real data
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
        knn.fit(real_fd_pca, real_fd_labels)

        # ── C1: Per-mode KNN accuracy ──
        mode_names_list = list(mode_diversity.keys())
        knn_accs = []
        for mode_name in mode_names_list:
            mode_fd_mask = full_dim_source == mode_name
            if not np.any(mode_fd_mask):
                knn_accs.append(0.0)
                continue
            mode_fd_pca = fd_pca[mode_fd_mask]
            mode_fd_lab = full_dim_labels[mode_fd_mask]
            preds = knn.predict(mode_fd_pca)
            knn_accs.append(float(np.mean(preds == mode_fd_lab)))

        xpos_c1 = np.arange(len(mode_names_list))
        max_acc = max(knn_accs) if knn_accs else 1.0
        ylim_top = min(1.08, max_acc * 1.24) if knn_accs else 1.0
        bars = ax_c1.bar(
            xpos_c1, knn_accs,
            color=COLORS["generated"], alpha=0.85, edgecolor="white",
        )
        for bar, acc in zip(bars, knn_accs):
            y_pos = min(bar.get_height() + 0.015, ylim_top - 0.02)
            ax_c1.text(bar.get_x() + bar.get_width() / 2, y_pos,
                       f"{acc:.2f}", ha="center", va="bottom", fontsize=FONT_ANNOTATION,
                       fontweight="normal", color=COLORS["annotation_dark"])
        ax_c1.set_xticks(xpos_c1)
        ax_c1.set_xticklabels([_short_mode(m) for m in mode_names_list],
                               rotation=0, ha="center", fontsize=FONT_TICK)
        ax_c1.set_ylabel("KNN-5 Accuracy", fontsize=FONT_LABEL)
        ax_c1.set_title("Identity Preservation (KNN-5)", fontsize=FONT_TITLE - 1)
        ax_c1.set_ylim(0, ylim_top)

        # ── C2: Per-type diversity heatmap (types x modes) ──
        all_mode_names = ["Real"] + mode_names_list
        n_types_sel = len(selected_types)
        div_matrix = np.full((n_types_sel, len(all_mode_names)), np.nan)

        for j, src_name in enumerate(all_mode_names):
            src_mask = full_dim_source == src_name
            for i, tid in enumerate(selected_types):
                tmask = src_mask & (full_dim_labels == tid)
                pts = full_dim_data[tmask]
                if len(pts) >= 5:
                    norms = np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
                    pts_n = pts / norms
                    sim = pts_n @ pts_n.T
                    idx_tri = np.triu_indices(len(pts), k=1)
                    div_matrix[i, j] = 1.0 - float(sim[idx_tri].mean())

        type_short_names = [
            abbreviate_cell_type(
                type_names.get(int(tid), f"T{tid}") if type_names else f"T{tid}",
                max_len=16,
            )
            for tid in selected_types
        ]
        im = ax_c2.imshow(div_matrix, aspect="auto", cmap="cividis",
                          norm=Normalize(vmin=np.nanmin(div_matrix) * 0.9,
                                         vmax=np.nanmax(div_matrix) * 1.1))
        ax_c2.set_xticks(np.arange(len(all_mode_names)))
        ax_c2.set_xticklabels([_short_mode(m) for m in all_mode_names],
                               rotation=25, ha="right", fontsize=FONT_TICK_DENSE)
        ax_c2.set_yticks(np.arange(n_types_sel))
        ax_c2.set_yticklabels(type_short_names, fontsize=FONT_TICK_DENSE)
        ax_c2.set_title("Per-Type Diversity (1\u2212cos)", fontsize=FONT_TITLE)
        # Annotate cells
        for i in range(n_types_sel):
            for j in range(len(all_mode_names)):
                val = div_matrix[i, j]
                if not np.isnan(val):
                    ax_c2.text(j, i, f"{val:.2f}", ha="center", va="center",
                               fontsize=FONT_HEATMAP_CELL,
                               color="white" if val > np.nanmedian(div_matrix) else "black")
        cax = add_axes_next_to(
            fig,
            ax_c2,
            side="right",
            width=0.010,
            height=ax_c2.get_position().height * 0.55,
            pad=0.012,
            align="bottom",
            y_offset=0.01,
        )
        fig.colorbar(im, cax=cax)

        # ── C3: Pairwise cosine violin per source ──
        violin_data = []
        violin_labels_list = []
        max_pairs = 2000  # subsample for speed
        rng_v = np.random.default_rng(42)

        for src_name in all_mode_names:
            src_mask = full_dim_source == src_name
            pts = full_dim_data[src_mask]
            if len(pts) >= 5:
                norms = np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
                pts_n = pts / norms
                sim = pts_n @ pts_n.T
                idx_tri = np.triu_indices(len(pts), k=1)
                pw_sims = sim[idx_tri]
                if len(pw_sims) > max_pairs:
                    pw_sims = rng_v.choice(pw_sims, max_pairs, replace=False)
                violin_data.append(pw_sims)
                violin_labels_list.append(_short_mode(src_name))

        if violin_data:
            parts = ax_c3.violinplot(violin_data, showmeans=True, showmedians=True)
            for idx_v, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(COLORS["real"] if idx_v == 0 else COLORS["generated"])
                pc.set_alpha(0.6)
            if "cmeans" in parts:
                parts["cmeans"].set_color(COLORS["real"])
            if "cmedians" in parts:
                parts["cmedians"].set_color(COLORS["bad"])
            ax_c3.set_xticks(np.arange(1, len(violin_labels_list) + 1))
            ax_c3.set_xticklabels(violin_labels_list, rotation=0, ha="center",
                                   fontsize=FONT_TICK)
        ax_c3.set_ylabel("Pairwise Cosine Similarity", fontsize=FONT_LABEL)
        ax_c3.set_title("Cluster Tightness", fontsize=FONT_TITLE)

    # ── Legend: cell-type keys — place just below the top-row scatter plots ──
    handles, labels = axes[0].get_legend_handles_labels()
    # Remove any subplot-level legend that may have been auto-added
    for _ax in axes:
        leg = _ax.get_legend()
        if leg is not None:
            leg.remove()
    # Position legend just below the bottom edge of the first row of scatter plots
    row1_bottom = axes[0].get_position().y0
    legend_ax = add_shared_legend_axes(fig, (0.11, row1_bottom - 0.046, 0.78, 0.036))
    legend_ax.legend(
        handles, labels, loc="center",
        ncol=min(len(handles), 5), fontsize=FONT_TICK_DENSE - 1,
        markerscale=2.1, frameon=False,
        columnspacing=0.8, handletextpad=0.36,
    )

    path = output_dir / "fig05b_conditioning_landscape.png"
    save_with_vcd(fig, path, dpi)
    logger.info(f"Saved conditioning landscape \u2192 %s", path)
    plt.close(fig)
    return path
