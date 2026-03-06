"""
panels_embedding.py — Panel B: CLOP alignment space (UMAP of prototypes + cells).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from . import io as viz_io
from .style import TYPE_PALETTE, set_dense_tick_labels, set_figure_suptitle
from src.utils.paths import FIG_DIR

logger = logging.getLogger(__name__)


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
    unique_types = np.unique(group_ids)
    logger.info(
        f"Panel B: {cell_proj.shape[0]} cells ({space_label}), "
        f"{proj_text.shape[0]} text conditions"
    )

    n_types = len(unique_types)
    proto_dim = proj_text.shape[1]
    text_proto = np.zeros((n_types, proto_dim), dtype=np.float32)
    for i, t in enumerate(unique_types):
        text_proto[i] = proj_text[group_ids == t].mean(axis=0)
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
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 1.2], wspace=0.42)
    set_figure_suptitle(
        fig,
        "CLOP Alignment Space (UMAP — all embeddings in shared CLOP projection)",
        fontsize=11,
    )

    ax = fig.add_subplot(gs[0])
    sorted_order = np.argsort(type_counts)[::-1]
    bar_colors = [TYPE_PALETTE[t % len(TYPE_PALETTE)] for t in unique_types[sorted_order]]
    bar_labels = [type_names.get(int(t), f"Type {t}")[:16] for t in unique_types[sorted_order]]
    y_pos = np.arange(n_types)
    ax.barh(y_pos, type_counts[sorted_order], color=bar_colors, height=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels, fontsize=8)
    set_dense_tick_labels(ax, axis="y", max_labels=10, fontsize=8, rotation=0)
    ax.invert_yaxis()
    ax.set_xlabel("Cells")
    ax.set_title("Cells per Type")
    ax.axvline(
        x=np.median(type_counts),
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"median={int(np.median(type_counts))}",
    )
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

    ax = fig.add_subplot(gs[1])
    top_label_candidates = np.argsort(type_counts)[::-1].tolist()
    label_offsets = [(0, 4), (0, -10), (8, 4), (-8, 4), (10, -8), (-10, -8)]
    labeled_points: list[np.ndarray] = []
    for i, (x, y) in enumerate(proto_coords):
        color = TYPE_PALETTE[i % len(TYPE_PALETTE)]
        ax.scatter(
            x, y, c=[color], s=120, marker="D", edgecolors="black",
            linewidths=0.6, zorder=5,
        )
        if i in top_label_candidates[:6] and len(labeled_points) < 4:
            point = np.array([x, y])
            min_dist = 0.32
            if any(np.linalg.norm(point - prev) < min_dist for prev in labeled_points):
                continue
            name = type_names.get(int(unique_types[i]), f"Type {i}")
            short = name[:16] + "…" if len(name) > 16 else name
            dx, dy = label_offsets[len(labeled_points) % len(label_offsets)]
            ax.annotate(
                short, (x, y), fontsize=8, ha="center", va="bottom",
                xytext=(dx, dy), textcoords="offset points",
            )
            labeled_points.append(point)
    ax.set_title("69 Text Prototypes (CLOP space)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    ax = fig.add_subplot(gs[2])
    for t in unique_types:
        mask = gids_sub == t
        color = TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]
        ax.scatter(
            cell_coords[mask, 0], cell_coords[mask, 1],
            c=[color], s=3, alpha=0.4, rasterized=True,
        )
    for i, (x, y) in enumerate(proto_coords):
        color = TYPE_PALETTE[int(unique_types[i]) % len(TYPE_PALETTE)]
        ax.scatter(
            x, y, c=[color], s=80, marker="*", edgecolors="black",
            linewidths=0.5, zorder=6,
        )
    ax.set_title("Cells + ★ Proto Overlay")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.12, top=0.84)

    if save:
        viz_io.save_to_dir(
            fig, "panel_b_clop_embedding_umap", str(output_dir), dpi, save_panel_fn
        )
    return fig
