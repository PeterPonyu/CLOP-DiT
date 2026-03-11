"""
fig04_embedding.py — Article Figure 4: Embedding-space panels.

Consolidated from:
  - panels_embedding.py  (Panel B: CLOP alignment space)
  - panels_umap_quality.py (Panel E: Real vs Generated UMAP)
  - panels_merged.py     (B+E merged figure, G+F merged figure)
"""

from __future__ import annotations

import io as std_io
import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .direct_layout import bind_figure_region
from .style import (
    COLORS,
    FONT_DENSE_YTICK,
    FONT_LEGEND,
    TYPE_PALETTE,
    abbreviate_cell_type,
    add_panel_label,
    apply_style,
    get_export_savefig_kwargs,
    save_with_vcd,
    set_dense_tick_labels,
    set_figure_suptitle,
    style_axes,
)
from src.utils.paths import FIG_DIR

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Primary merged figure: B + E  (Article Fig 4)
# ──────────────────────────────────────────────────────────────────────

def plot_embedding_space_merged(
    cache_dir: str = "data/cache",
    generated_path: Optional[str] = None,
    generated_labels_path: Optional[str] = None,
    type_names: Optional[Dict[int, str]] = None,
    n_cells: int = 5000,
    output_dir: str = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Merged embedding-space figure (former Panels B + E).

    Top row (B): CLOP alignment space
      B1: 69 text prototypes (UMAP, top-12 labelled)
      B2: Cells + prototype overlay (CLOP projection space)
    Bottom row (E): Real vs Generated
      E1: Real cells UMAP (type-coloured)
      E2: Generated cells UMAP (type-coloured)
      E3: Overlay (circles=real, triangles=generated)
    """
    import umap as umap_lib

    cache = Path(cache_dir)
    if type_names is None:
        type_names = {}

    gid_path = cache / "text_group_ids_dedup.npy"
    if not gid_path.exists():
        logger.warning("Missing text_group_ids_dedup.npy — skipping merged B+E")
        return None
    group_ids = np.load(gid_path)
    unique_types = np.unique(group_ids)
    n_types = len(unique_types)
    # Full text group IDs for text prototype computation
    gid_text_path = cache / "text_group_ids.npy"

    proj_text_path = cache / "projected_text.npy"
    proj_cell_path = cache / "projected_cells.npy"
    has_b = proj_text_path.exists()
    if has_b and not proj_cell_path.exists():
        fb = cache / "cell_embeddings_dedup_preprocessed.npy"
        if fb.exists():
            proj_cell_path = fb
        else:
            has_b = False

    if generated_path is None:
        for c in ["results/generated_embeddings.npy",
                   "models/checkpoints/generated_cells.npy"]:
            if Path(c).exists():
                generated_path = c
                break
    if generated_labels_path is None:
        if Path("results/generated_labels.npy").exists():
            generated_labels_path = "results/generated_labels.npy"
    has_e = generated_path is not None and Path(generated_path).exists()

    if not has_b and not has_e:
        logger.info("Neither B nor E data available — skipping merged figure")
        return None

    apply_style()
    n_rows = (1 if has_b else 0) + (1 if has_e else 0)
    fig = plt.figure(figsize=(14.8, 5.2 * n_rows))
    layout = bind_figure_region(fig, (0.04, 0.06, 0.98, 0.96))
    row_regions = layout.split_rows(n_rows, hspace=0.25)
    # suptitle removed per revision; title information moved to LaTeX caption
    row = 0

    if has_b:
        b_rects = row_regions[row].split_cols([0.92, 0.92, 1.04], gap=[0.018, 0.042])
        proj_text = np.load(proj_text_path)
        cell_proj = np.load(proj_cell_path)
        if gid_text_path.exists() and proj_text.shape[0] != group_ids.shape[0]:
            text_group_ids = np.load(gid_text_path)
        else:
            text_group_ids = group_ids
        text_proto = np.zeros((n_types, proj_text.shape[1]), dtype=np.float32)
        for i, t in enumerate(unique_types):
            text_proto[i] = proj_text[text_group_ids == t].mean(axis=0)
        norms = np.linalg.norm(text_proto, axis=1, keepdims=True) + 1e-8
        text_proto = text_proto / norms
        type_counts = np.array([np.sum(group_ids == t) for t in unique_types])

        rng = np.random.default_rng(42)
        n_per = max(10, n_cells // n_types)
        s_idx = []
        for t in unique_types:
            t_idx = np.where(group_ids == t)[0]
            s_idx.extend(rng.choice(t_idx, min(n_per, len(t_idx)), replace=False).tolist())
        s_idx = np.array(s_idx)
        rng.shuffle(s_idx)
        cell_sub = cell_proj[s_idx]
        gids_sub = group_ids[s_idx]

        combined_b = np.vstack([cell_sub, text_proto])
        reducer_b = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                                   metric="cosine", random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coords_b = reducer_b.fit_transform(combined_b)
        cell_coords = coords_b[:len(s_idx)]
        proto_coords = coords_b[len(s_idx):]

        ax_b0 = b_rects[0].add_axes(fig)
        ax = ax_b0
        add_panel_label(ax, 'a', x=-0.10, y=1.05)
        for i, (x, y) in enumerate(proto_coords):
            color = TYPE_PALETTE[i % len(TYPE_PALETTE)]
            ax.scatter(x, y, c=[color], s=120, marker="D", edgecolors="black",
                       linewidths=0.6, zorder=5)
        ax.set_title("Text Prototypes (CLOP space)")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        style_axes(ax, kind="umap")

        ax_b1 = b_rects[1].add_axes(fig)
        ax = ax_b1
        add_panel_label(ax, 'b', x=-0.10, y=1.05)
        for t in unique_types:
            mask = gids_sub == t
            color = TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]
            ax.scatter(cell_coords[mask, 0], cell_coords[mask, 1],
                       c=[color], s=3, alpha=0.4, rasterized=True)
        for i, (x, y) in enumerate(proto_coords):
            color = TYPE_PALETTE[int(unique_types[i]) % len(TYPE_PALETTE)]
            ax.scatter(x, y, c=[color], s=80, marker="*", edgecolors="black",
                       linewidths=0.5, zorder=6)
        ax.set_title(f"Cell + Prototype Overlay ({len(s_idx)} cells)")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        style_axes(ax, kind="umap")

        ax_b2 = b_rects[2].add_axes(fig)
        ax = ax_b2
        add_panel_label(ax, 'c', x=-0.10, y=1.05)
        so = np.argsort(type_counts)[::-1]
        bar_c = [TYPE_PALETTE[t % len(TYPE_PALETTE)] for t in unique_types[so]]
        y_pos = np.arange(n_types)
        ax.barh(y_pos, type_counts[so], color=bar_c, height=0.8)
        ax.set_yticks(y_pos)
        bar_labels = [abbreviate_cell_type(type_names.get(int(t), f"T{t}"), 22) for t in unique_types[so]]
        ax.set_yticklabels(bar_labels, fontsize=FONT_DENSE_YTICK)
        set_dense_tick_labels(ax, axis="y", max_labels=14, fontsize=FONT_DENSE_YTICK, rotation=0)
        ax.invert_yaxis()
        ax.set_xlabel("Cells")
        ax.set_title("Cells per Type")
        style_axes(ax, kind="bar")
        row += 1

    if has_e:
        e_rects = row_regions[row].split_cols([0.92, 0.92, 1.04], gap=[0.018, 0.042])
        cell_path = cache / "cell_embeddings_dedup_preprocessed.npy"
        if not cell_path.exists():
            logger.warning("Missing cell embeddings for E row")
        else:
            real = np.load(cell_path)
            gen = np.load(generated_path)
            real_gids = group_ids
            gen_gids = (np.load(generated_labels_path)
                        if generated_labels_path and Path(generated_labels_path).exists()
                        else None)

            rng = np.random.default_rng(42)
            n_r = min(n_cells, len(real))
            n_g = min(n_cells, len(gen))
            ri = rng.choice(len(real), n_r, replace=False)
            gi = rng.choice(len(gen), n_g, replace=False) if len(gen) > n_cells else np.arange(len(gen))
            r_sub, g_sub = real[ri], gen[gi]
            r_gids = real_gids[ri]
            g_gids = gen_gids[gi] if gen_gids is not None else None

            combined_e = np.vstack([r_sub, g_sub])
            reducer_e = umap_lib.UMAP(n_components=2, random_state=42, metric="cosine")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coords_e = reducer_e.fit_transform(combined_e)
            rc = coords_e[:len(r_sub)]
            gc = coords_e[len(r_sub):]

            ax_e0 = e_rects[0].add_axes(fig)
            ax = ax_e0
            add_panel_label(ax, 'd', x=-0.10, y=1.05)
            for t in np.unique(r_gids):
                m = r_gids == t
                ax.scatter(rc[m, 0], rc[m, 1],
                           c=[TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]],
                           s=4, alpha=0.4, rasterized=True)
            ax.set_title(f"Real ({len(r_sub)} cells)")
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            style_axes(ax, kind="umap")

            ax_e1 = e_rects[1].add_axes(fig)
            ax = ax_e1
            add_panel_label(ax, 'e', x=-0.10, y=1.05)
            if g_gids is not None:
                for t in np.unique(g_gids):
                    m = g_gids == t
                    ax.scatter(gc[m, 0], gc[m, 1],
                               c=[TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]],
                               s=4, alpha=0.4, rasterized=True)
            else:
                ax.scatter(gc[:, 0], gc[:, 1], c=COLORS["generated"], s=4, alpha=0.4,
                           rasterized=True)
            ax.set_title(f"Generated ({len(g_sub)} cells)")
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            style_axes(ax, kind="umap")

            ax_e2 = e_rects[2].add_axes(fig)
            ax = ax_e2
            add_panel_label(ax, 'f', x=-0.10, y=1.05)
            all_types = np.unique(np.concatenate([r_gids, g_gids])) if g_gids is not None else np.unique(r_gids)
            for t in all_types:
                color = TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]
                rm = r_gids == t
                if rm.any():
                    ax.scatter(
                        rc[rm, 0], rc[rm, 1], c=[color], s=5, alpha=0.25,
                        marker="o", edgecolors="white", linewidths=0.2, rasterized=True
                    )
                if g_gids is not None:
                    gm = g_gids == t
                    if gm.any():
                        ax.scatter(
                            gc[gm, 0], gc[gm, 1], c=[color], s=10, alpha=0.40,
                            marker="^", edgecolors="black", linewidths=0.2, rasterized=True
                        )
            ax.scatter([], [], c=COLORS["real"], s=26, marker="o", label="Real")
            ax.scatter([], [], c=COLORS["generated"], s=30, marker="^", label="Generated")
            ax.legend(
                markerscale=2.0,
                fontsize=FONT_LEGEND,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=False,
            )
            ax.set_title("Type-Colored Overlay")
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            style_axes(ax, kind="umap")
    if save:
        path = Path(output_dir) / "fig04_embedding_space.png"
        if save_panel_fn is not None:
            save_panel_fn(fig, path, dpi)
        else:
            save_with_vcd(fig, path, dpi)
    return fig


# ──────────────────────────────────────────────────────────────────────
# Standalone Panel B  (legacy report)
# ──────────────────────────────────────────────────────────────────────

def plot_clop_embedding_space(
    cache_dir: str | Path,
    type_names: Optional[Dict[int, str]] = None,
    output_dir: Optional[str | Path] = None,
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
    n_cells: int = 8000,
) -> Optional[plt.Figure]:
    """3-panel CLOP alignment visualization.

    Both cells and text must be in the same CLOP projection space.
    B0: Cells-per-type histogram (data balance check)
    B1: 69 text prototypes (centroids of projected text per type), labelled
    B2: Subsampled CLOP-projected cells coloured by type + text proto overlay
    """
    if output_dir is None:
        output_dir = FIG_DIR
    cache = Path(cache_dir)
    if type_names is None:
        type_names = {}

    proj_text_path = cache / "projected_text.npy"
    proj_cell_path = cache / "projected_cells.npy"
    gid_path = cache / "text_group_ids_dedup.npy"
    # text_group_ids (full) matches projected_text; dedup version matches cells
    gid_text_path = cache / "text_group_ids.npy"

    for p in [proj_text_path, gid_path]:
        if not p.exists():
            logger.warning(f"Missing {p.name} — skipping Panel B")
            return None

    if not proj_cell_path.exists():
        fallback_path = cache / "cell_embeddings_dedup_preprocessed.npy"
        if not fallback_path.exists():
            logger.warning(
                "Missing projected_cells.npy and fallback — skipping Panel B"
            )
            return None
        cell_proj = np.load(fallback_path)
        space_label = "scGPT latent (WARNING: different space from prototypes)"
    else:
        cell_proj = np.load(proj_cell_path)
        space_label = "CLOP shared projection"

    proj_text = np.load(proj_text_path)
    group_ids = np.load(gid_path)
    # Use full text group IDs for text prototype computation if available
    if gid_text_path.exists() and proj_text.shape[0] != group_ids.shape[0]:
        text_group_ids = np.load(gid_text_path)
    else:
        text_group_ids = group_ids
    unique_types = np.unique(group_ids)
    logger.info(
        f"Panel B: {cell_proj.shape[0]} cells ({space_label}), "
        f"{proj_text.shape[0]} text conditions"
    )

    n_types = len(unique_types)
    proto_dim = proj_text.shape[1]
    text_proto = np.zeros((n_types, proto_dim), dtype=np.float32)
    for i, t in enumerate(unique_types):
        text_proto[i] = proj_text[text_group_ids == t].mean(axis=0)
    norms = np.linalg.norm(text_proto, axis=1, keepdims=True) + 1e-8
    text_proto = text_proto / norms
    logger.info(f"Computed {n_types} text prototype centroids")

    type_counts = np.array([np.sum(group_ids == t) for t in unique_types])

    rng = np.random.default_rng(42)
    n_per_type = max(10, n_cells // n_types)
    sampled_idx = []
    for t in unique_types:
        t_idx = np.where(group_ids == t)[0]
        n = min(n_per_type, len(t_idx))
        sampled_idx.extend(rng.choice(t_idx, n, replace=False).tolist())
    sampled_idx = np.array(sampled_idx)
    rng.shuffle(sampled_idx)

    cell_sub = cell_proj[sampled_idx]
    gids_sub = group_ids[sampled_idx]
    logger.info(f"UMAP: {len(sampled_idx)} cells + {n_types} prototypes (all in CLOP space)")

    import umap as umap_lib

    combined = np.vstack([cell_sub, text_proto])
    reducer = umap_lib.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.3,
        metric="cosine", random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = reducer.fit_transform(combined)

    cell_coords = coords[: len(sampled_idx)]
    proto_coords = coords[len(sampled_idx) :]

    fig = plt.figure(figsize=(11.2, 5.4))
    ax_rect_1, ax_rect_2, ax_rect_3 = bind_figure_region(fig, (0.18, 0.12, 0.98, 0.87)).split_cols(
        [0.90, 0.96, 1.10],
        gap=[0.020, 0.048],
    )
    # REMOVED: set_figure_suptitle(fig, "CLOP Alignment Space...", ...)
    # Title information moved to LaTeX caption per MDPI style guidelines

    ax_b0 = ax_rect_1.add_axes(fig)
    add_panel_label(ax_b0, 'a', x=-0.18, y=1.02)
    sorted_order = np.argsort(type_counts)[::-1]
    bar_colors = [TYPE_PALETTE[t % len(TYPE_PALETTE)] for t in unique_types[sorted_order]]
    bar_labels = [abbreviate_cell_type(type_names.get(int(t), f"Type {t}"), 22) for t in unique_types[sorted_order]]
    y_pos = np.arange(n_types)
    ax_b0.barh(y_pos, type_counts[sorted_order], color=bar_colors, height=0.8)
    ax_b0.set_yticks(y_pos)
    ax_b0.set_yticklabels(bar_labels, fontsize=10)
    set_dense_tick_labels(ax_b0, axis="y", max_labels=10, fontsize=10, rotation=0)
    ax_b0.invert_yaxis()
    ax_b0.set_xlabel("Cells")
    ax_b0.set_title("Cells per Type")
    ax_b0.axvline(
        x=np.median(type_counts),
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"median={int(np.median(type_counts))}",
    )
    ax_b0.legend(fontsize=10, loc="lower right", frameon=False)
    ax_b0.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

    ax_b1 = ax_rect_2.add_axes(fig)
    add_panel_label(ax_b1, 'b', x=-0.18, y=1.02)
    top_label_candidates = np.argsort(type_counts)[::-1].tolist()
    label_offsets = [(0, 4), (0, -10), (8, 4), (-8, 4), (10, -8), (-10, -8)]
    labeled_points: list[np.ndarray] = []
    for i, (x, y) in enumerate(proto_coords):
        color = TYPE_PALETTE[int(unique_types[i]) % len(TYPE_PALETTE)]
        ax_b1.scatter(
            x, y, c=[color], s=120, marker="D", edgecolors="black",
            linewidths=0.6, zorder=5,
        )
        if i in top_label_candidates[:6] and len(labeled_points) < 4:
            point = np.array([x, y])
            min_dist = 0.32
            if any(np.linalg.norm(point - prev) < min_dist for prev in labeled_points):
                continue
            name = type_names.get(int(unique_types[i]), f"Type {i}")
            short = abbreviate_cell_type(name, 22)
            dx, dy = label_offsets[len(labeled_points) % len(label_offsets)]
            ax_b1.annotate(
                short, (x, y), fontsize=10, ha="center", va="bottom",
                xytext=(dx, dy), textcoords="offset points",
            )
            labeled_points.append(point)
    ax_b1.set_title("69 Text Prototypes (CLOP space)")
    ax_b1.set_xlabel("UMAP 1")
    ax_b1.set_ylabel("UMAP 2")

    ax_b2 = ax_rect_3.add_axes(fig)
    add_panel_label(ax_b2, 'c', x=-0.12, y=1.02)
    for t in unique_types:
        mask = gids_sub == t
        color = TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]
        ax_b2.scatter(
            cell_coords[mask, 0], cell_coords[mask, 1],
            c=[color], s=3, alpha=0.4, rasterized=True,
        )
    for i, (x, y) in enumerate(proto_coords):
        color = TYPE_PALETTE[int(unique_types[i]) % len(TYPE_PALETTE)]
        ax_b2.scatter(
            x, y, c=[color], s=80, marker="*", edgecolors="black",
            linewidths=0.5, zorder=6,
        )
    ax_b2.set_title("Cell + Prototype Overlay")
    ax_b2.set_xlabel("UMAP 1")
    ax_b2.set_ylabel("UMAP 2")
    ax_b2.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax_b2.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

    if save:
        path = Path(output_dir) / "fig04_clop_embedding_umap.png"
        if save_panel_fn is not None:
            save_panel_fn(fig, path, dpi)
        else:
            save_with_vcd(fig, path, dpi)
    return fig


# ──────────────────────────────────────────────────────────────────────
# Standalone Panel E  (legacy report)
# ──────────────────────────────────────────────────────────────────────

def plot_real_vs_generated(
    cache_dir: str = "data/cache",
    generated_path: Optional[str] = None,
    generated_labels_path: Optional[str] = None,
    n_cells: int = 5000,
    output_dir: Optional[str] = None,
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """3-panel real vs generated comparison with type colouring.

    E1: Real cells (UMAP, coloured by type)
    E2: Generated cells (UMAP, coloured by type)
    E3: Overlay (real=circles, generated=triangles)
    """
    if output_dir is None:
        output_dir = str(FIG_DIR)
    cache = Path(cache_dir)

    # Auto-detect generated cells
    if generated_path is None:
        candidates = [
            "results/generated_embeddings.npy",
            "models/checkpoints/generated_cells.npy",
        ]
        for c in candidates:
            if Path(c).exists():
                generated_path = c
                break
    if generated_labels_path is None:
        candidates = [
            "results/generated_labels.npy",
        ]
        for c in candidates:
            if Path(c).exists():
                generated_labels_path = c
                break

    if generated_path is None or not Path(generated_path).exists():
        logger.info(
            "No generated cells found — Panel E deferred until after "
            "DiT inference (run scripts/inference/generate_embeddings.py first)"
        )
        return None

    cell_path = cache / "cell_embeddings_dedup_preprocessed.npy"
    gid_path = cache / "text_group_ids_dedup.npy"
    if not cell_path.exists():
        logger.warning(f"Missing {cell_path.name}")
        return None

    real = np.load(cell_path)
    gen = np.load(generated_path)
    real_gids = np.load(gid_path) if gid_path.exists() else None
    gen_gids = (
        np.load(generated_labels_path)
        if generated_labels_path and Path(generated_labels_path).exists()
        else None
    )
    logger.info(f"Real: {real.shape}, Generated: {gen.shape}")

    # Stratified subsample of real cells
    rng = np.random.default_rng(42)
    n_real = min(n_cells, len(real))
    n_gen = min(n_cells, len(gen))
    real_idx = rng.choice(len(real), n_real, replace=False)
    gen_idx = (
        rng.choice(len(gen), n_gen, replace=False)
        if len(gen) > n_cells
        else np.arange(len(gen))
    )

    real_sub = real[real_idx]
    gen_sub = gen[gen_idx]
    real_gids_sub = real_gids[real_idx] if real_gids is not None else None
    gen_gids_sub = gen_gids[gen_idx] if gen_gids is not None else None

    combined = np.vstack([real_sub, gen_sub])

    import umap as umap_lib

    reducer = umap_lib.UMAP(n_components=2, random_state=42, metric="cosine")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = reducer.fit_transform(combined)

    real_c = coords[: len(real_sub)]
    gen_c = coords[len(real_sub) :]

    fig = plt.figure(figsize=(9.0, 5.2))
    ax_rect_1, ax_rect_2, ax_rect_3 = bind_figure_region(fig, (0.06, 0.12, 0.98, 0.92)).split_cols(
        [0.92, 0.92, 1.02],
        gap=[0.020, 0.040],
    )
    # suptitle removed per revision; title information moved to LaTeX caption

    ax_e1 = None
    ax_e2 = None
    ax_e3 = None

    # E1: Real — type-coloured
    ax_e1 = ax_rect_1.add_axes(fig)
    ax = ax_e1
    add_panel_label(ax, 'a')
    if real_gids_sub is not None:
        for t in np.unique(real_gids_sub):
            mask = real_gids_sub == t
            color = TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]
            ax.scatter(
                real_c[mask, 0], real_c[mask, 1],
                c=[color], s=4, alpha=0.4, rasterized=True,
            )
    else:
        ax.scatter(
            real_c[:, 0], real_c[:, 1],
            c=COLORS["real"], s=4, alpha=0.4, rasterized=True,
        )
    n_types = len(np.unique(real_gids_sub)) if real_gids_sub is not None else "?"
    ax.set_title(
        f"Real ({len(real_sub)} cells, "
        f"{n_types} types)"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # E2: Generated — type-coloured
    ax_e2 = ax_rect_2.add_axes(fig)
    ax = ax_e2
    add_panel_label(ax, 'b')
    if gen_gids_sub is not None:
        for t in np.unique(gen_gids_sub):
            mask = gen_gids_sub == t
            color = TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]
            ax.scatter(
                gen_c[mask, 0], gen_c[mask, 1],
                c=[color], s=4, alpha=0.4, rasterized=True,
            )
    else:
        ax.scatter(
            gen_c[:, 0], gen_c[:, 1],
            c=COLORS["generated"], s=4, alpha=0.4, rasterized=True,
        )
    ax.set_title(f"Generated ({len(gen_sub)} cells)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # E3: Overlay — type-coloured, shape-split (circle=real, triangle=gen)
    ax_e3 = ax_rect_3.add_axes(fig)
    ax = ax_e3
    add_panel_label(ax, 'c')
    if real_gids_sub is not None and gen_gids_sub is not None:
        for t in np.unique(np.concatenate([real_gids_sub, gen_gids_sub])):
            color = TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]
            rmask = real_gids_sub == t
            gmask = gen_gids_sub == t
            if rmask.any():
                ax.scatter(
                    real_c[rmask, 0], real_c[rmask, 1],
                    c=[color], s=3, alpha=0.2, marker="o", rasterized=True,
                )
            if gmask.any():
                ax.scatter(
                    gen_c[gmask, 0], gen_c[gmask, 1],
                    c=[color], s=6, alpha=0.35, marker="^", rasterized=True,
                )
        # Dummy handles for legend
        ax.scatter([], [], c=COLORS["real"], s=60, marker="o", linewidths=0.5, label="Real")
        ax.scatter([], [], c=COLORS["generated"], s=60, marker="^", label="Generated")
    else:
        ax.scatter(
            real_c[:, 0], real_c[:, 1],
            c=COLORS["real"], s=3, alpha=0.25, label="Real", rasterized=True,
        )
        ax.scatter(
            gen_c[:, 0], gen_c[:, 1],
            c=COLORS["generated"], s=3, alpha=0.25, marker="^",
            label="Generated", rasterized=True,
        )
    ax.legend(markerscale=3, fontsize=FONT_LEGEND, frameon=False, loc="upper right")
    ax.set_title("Type-Colored Overlay", fontsize=12)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)

    # Reduce tick density on all UMAP axes
    for _ax in fig.get_axes():
        _ax.tick_params(labelsize=10)
        _ax.xaxis.set_major_locator(MaxNLocator(nbins=2, symmetric=True, prune="both"))
        _ax.yaxis.set_major_locator(MaxNLocator(nbins=2, symmetric=True, prune="both"))
        # Widen axis limits slightly to give tick labels breathing room
        xl = _ax.get_xlim()
        _ax.set_xlim(xl[0] - (xl[1] - xl[0]) * 0.10, xl[1] + (xl[1] - xl[0]) * 0.10)

    if save:
        path = Path(output_dir) / "fig04_real_vs_generated.png"
        if save_panel_fn is not None:
            save_panel_fn(fig, path, dpi)
        else:
            save_with_vcd(fig, path, dpi)
    return fig


# ──────────────────────────────────────────────────────────────────────
# Legacy merged figure: G + F  (fidelity & alignment)
# ──────────────────────────────────────────────────────────────────────

def plot_fidelity_and_alignment_merged(
    cache_dir: str = "data/cache",
    metrics_path: str = "results/generation_metrics.json",
    div_metrics_path: str = "results/diversity_diagnostics.json",
    type_names: Optional[Dict[int, str]] = None,
    output_dir: str = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Merged fidelity + alignment figure (former Panels G + F).

    Top row (G): Per-type generation fidelity  (panels a-c)
    Bottom row (F): Text-cell alignment         (panels d-f)

    Uses matplotlib subfigures for native vector (PDF) output.
    Each sub-function renders into its own subfigure, preserving
    the standalone layout while combining into a single figure.
    """
    from .panels_heatmaps import plot_per_type_generation, plot_text_cell_heatmap

    # Build the two sub-figures first (unsaved) to check availability
    fig_g = plot_per_type_generation(
        metrics_path=metrics_path,
        div_metrics_path=div_metrics_path,
        output_dir=output_dir, dpi=dpi, save=False,
    )
    fig_f = plot_text_cell_heatmap(
        cache_dir=cache_dir,
        type_names=type_names,
        output_dir=output_dir, dpi=dpi, save=False,
        label_offset=3,
    )

    if fig_g is None or fig_f is None:
        missing = "G" if fig_g is None else "F"
        logger.info("Both panels G and F required for merged figure; skipping (missing %s)", missing)
        if fig_g is not None:
            plt.close(fig_g)
        if fig_f is not None:
            plt.close(fig_f)
        return None

    from .style import run_vcd_check
    run_vcd_check(fig_g, "panel_g_per_type_generation")
    run_vcd_check(fig_f, "panel_f_text_cell_heatmap")

    # Use PIL composition for the merged figure — the two sub-figures have
    # incompatible GridSpec layouts that cannot share a single figure canvas.
    # Render at full DPI for publication-quality raster; standalone vector
    # PDFs (panel_g*.pdf, panel_f*.pdf) are the primary publication outputs.
    from PIL import Image

    images = []
    for f in [fig_g, fig_f]:
        buf = std_io.BytesIO()
        f.savefig(buf, format="png", **get_export_savefig_kwargs(f, dpi=dpi, pad_inches=0.08))
        buf.seek(0)
        images.append(Image.open(buf))
        plt.close(f)

    max_w = max(im.width for im in images)
    resized = []
    for im in images:
        if im.mode == "RGBA":
            im = im.convert("RGB")
        if im.width != max_w:
            padded = Image.new("RGB", (max_w, im.height), "white")
            x_offset = (max_w - im.width) // 2
            padded.paste(im, (x_offset, 0))
            im = padded
        resized.append(im)

    total_h = sum(im.height for im in resized)
    composite = Image.new("RGB", (max_w, total_h), "white")
    y_off = 0
    for im in resized:
        composite.paste(im, (0, y_off))
        y_off += im.height

    fig_merged = plt.figure(figsize=(max_w / dpi, total_h / dpi), dpi=dpi)
    ax = fig_merged.add_axes([0, 0, 1, 1])
    ax.imshow(np.array(composite))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.axis("off")

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "fig_fidelity_alignment.png"
        if save_panel_fn is not None:
            save_panel_fn(fig_merged, path, dpi)
        else:
            save_with_vcd(fig_merged, path, dpi, run_vcd=False)
        logger.info("Saved merged G+F → %s", path)
    return fig_merged
