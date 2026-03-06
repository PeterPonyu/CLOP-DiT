"""
panels_merged.py — Merged multi-panel figures: B+E (embedding space) and G+F (fidelity & alignment).
"""

from __future__ import annotations

import io as std_io
import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import io as viz_io
from .style import COLORS, TYPE_PALETTE, apply_style, set_dense_tick_labels
from .panels_heatmaps import plot_per_type_generation, plot_text_cell_heatmap

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


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
    fig = plt.figure(figsize=(14.2, 5.4 * n_rows))
    gs = fig.add_gridspec(n_rows, 3, wspace=0.46, hspace=0.48,
                          width_ratios=[1.2, 1.2, 1.0])
    fig.suptitle("Embedding Space Analysis", fontsize=12, y=0.98)
    row = 0

    if has_b:
        proj_text = np.load(proj_text_path)
        cell_proj = np.load(proj_cell_path)
        text_proto = np.zeros((n_types, proj_text.shape[1]), dtype=np.float32)
        for i, t in enumerate(unique_types):
            text_proto[i] = proj_text[group_ids == t].mean(axis=0)
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

        ax = fig.add_subplot(gs[row, 0])
        for i, (x, y) in enumerate(proto_coords):
            color = TYPE_PALETTE[i % len(TYPE_PALETTE)]
            ax.scatter(x, y, c=[color], s=120, marker="D", edgecolors="black",
                       linewidths=0.6, zorder=5)
        ax.set_title("Text Prototypes (CLOP space)")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

        ax = fig.add_subplot(gs[row, 1])
        for t in unique_types:
            mask = gids_sub == t
            color = TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]
            ax.scatter(cell_coords[mask, 0], cell_coords[mask, 1],
                       c=[color], s=3, alpha=0.4, rasterized=True)
        for i, (x, y) in enumerate(proto_coords):
            color = TYPE_PALETTE[int(unique_types[i]) % len(TYPE_PALETTE)]
            ax.scatter(x, y, c=[color], s=80, marker="*", edgecolors="black",
                       linewidths=0.5, zorder=6)
        ax.set_title(f"Cells + \u2605 Proto ({len(s_idx)} cells)")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

        ax = fig.add_subplot(gs[row, 2])
        so = np.argsort(type_counts)[::-1]
        bar_c = [TYPE_PALETTE[t % len(TYPE_PALETTE)] for t in unique_types[so]]
        y_pos = np.arange(n_types)
        ax.barh(y_pos, type_counts[so], color=bar_c, height=0.8)
        ax.set_yticks(y_pos)
        bar_labels = [type_names.get(int(t), f"T{t}")[:16] for t in unique_types[so]]
        ax.set_yticklabels(bar_labels, fontsize=6)
        set_dense_tick_labels(ax, axis="y", max_labels=14, fontsize=6, rotation=0)
        ax.invert_yaxis()
        ax.set_xlabel("Cells")
        ax.set_title("Cells per Type")
        row += 1

    if has_e:
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

            ax = fig.add_subplot(gs[row, 0])
            for t in np.unique(r_gids):
                m = r_gids == t
                ax.scatter(rc[m, 0], rc[m, 1],
                           c=[TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]],
                           s=4, alpha=0.4, rasterized=True)
            ax.set_title(f"Real ({len(r_sub)} cells)")
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

            ax = fig.add_subplot(gs[row, 1])
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

            ax = fig.add_subplot(gs[row, 2])
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
                fontsize=9,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=False,
            )
            ax.set_title("Type-Coloured Overlay")
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    if save:
        viz_io.save_to_dir(fig, "fig_embedding_space", output_dir, dpi, save_panel_fn)
    return fig


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

    Top row (G): Per-type generation fidelity
    Bottom row (F): Text-cell alignment
    """
    fig_g = plot_per_type_generation(
        metrics_path=metrics_path,
        div_metrics_path=div_metrics_path,
        output_dir=output_dir, dpi=dpi, save=False,
    )
    fig_f = plot_text_cell_heatmap(
        cache_dir=cache_dir,
        type_names=type_names,
        output_dir=output_dir, dpi=dpi, save=False,
    )

    if fig_g is None and fig_f is None:
        logger.info("No data for G or F — skipping merged figure")
        return None

    from PIL import Image

    images = []
    for f in [fig_g, fig_f]:
        if f is not None:
            buf = std_io.BytesIO()
            f.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.08)
            buf.seek(0)
            images.append(Image.open(buf))
            plt.close(f)

    if not images:
        return None

    max_w = max(im.width for im in images)
    resized = []
    for im in images:
        if im.width != max_w:
            ratio = max_w / im.width
            im = im.resize((max_w, int(im.height * ratio)), Image.LANCZOS)
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
    ax.axis("off")

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "fig_fidelity_alignment.png"
        if save_panel_fn is not None:
            save_panel_fn(fig_merged, path, dpi)
        else:
            from .style import save_with_vcd
            save_with_vcd(fig_merged, path, dpi)
        logger.info(f"Saved merged G+F → {path}")
    return fig_merged
