"""
panels_umap_quality.py — Panel E: Real vs Generated UMAP (type-coloured) for CLOP-DiT.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import io as viz_io
from .style import TYPE_PALETTE

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def plot_real_vs_generated(
    cache_dir: str = "data/cache",
    generated_path: Optional[str] = None,
    generated_labels_path: Optional[str] = None,
    n_cells: int = 5000,
    output_dir: str = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """3-panel real vs generated comparison with type colouring.

    E1: Real cells (UMAP, coloured by type)
    E2: Generated cells (UMAP, coloured by type)
    E3: Overlay (real=circles, generated=triangles)
    """
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
            "DiT inference (run scripts/generate_embeddings.py first)"
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
    gs_e = fig.add_gridspec(1, 3, wspace=0.40)
    fig.suptitle(
        "Real vs Generated Cell Embeddings (DiT v1)",
        fontsize=11,
    )

    # E1: Real — type-coloured
    ax = fig.add_subplot(gs_e[0])
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
            c="#2196F3", s=4, alpha=0.4, rasterized=True,
        )
    ax.set_title(
        f"Real ({len(real_sub)} cells, "
        f"{69 if real_gids_sub is not None else '?'} types)"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # E2: Generated — type-coloured
    ax = fig.add_subplot(gs_e[1])
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
            c="#FF5722", s=4, alpha=0.4, rasterized=True,
        )
    ax.set_title(f"Generated ({len(gen_sub)} cells)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # E3: Overlay — type-coloured, shape-split (circle=real, triangle=gen)
    ax = fig.add_subplot(gs_e[2])
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
        ax.scatter([], [], c="gray", s=20, marker="o", label="Real")
        ax.scatter([], [], c="gray", s=20, marker="^", label="Generated")
    else:
        ax.scatter(
            real_c[:, 0], real_c[:, 1],
            c="#2196F3", s=3, alpha=0.25, label="Real", rasterized=True,
        )
        ax.scatter(
            gen_c[:, 0], gen_c[:, 1],
            c="#FF5722", s=3, alpha=0.25, marker="^",
            label="Generated", rasterized=True,
        )
    ax.legend(markerscale=4, fontsize=10, loc="upper right")
    ax.set_title("Type-Coloured Overlay", fontsize=11)
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)

    # Reduce tick density on all UMAP axes
    from matplotlib.ticker import MaxNLocator
    for _ax in fig.get_axes():
        _ax.tick_params(labelsize=9)
        _ax.xaxis.set_major_locator(MaxNLocator(nbins=2, symmetric=True, prune="both"))
        _ax.yaxis.set_major_locator(MaxNLocator(nbins=2, symmetric=True, prune="both"))
        # Widen axis limits slightly to give tick labels breathing room
        xl = _ax.get_xlim()
        _ax.set_xlim(xl[0] - (xl[1] - xl[0]) * 0.10, xl[1] + (xl[1] - xl[0]) * 0.10)

    if save:
        viz_io.save_to_dir(fig, "panel_e_real_vs_generated", output_dir, dpi, save_panel_fn)
    return fig
