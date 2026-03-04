"""
results_visualizer.py — Publication-quality visualization for CLOP-DiT pipeline.

Generates multi-panel PDF/PNG reports proving training success:
  Panel A: CLOP training dynamics (loss, temperature, prototype accuracy, embedding quality)
  Panel B: CLOP embedding space (UMAP of 69-type prototypes + cell embeddings)
  Panel C: DiT training dynamics (flow-matching loss, cosine similarity, LR schedule)
  Panel D: Metrics summary table
  Panel E: [Post-inference] Real vs generated cell overlay (type-coloured UMAP)
  Panel F: Text–Cell similarity heatmap (69×69 cosine matrix proving CLOP alignment)
  Panel G: Per-type generation fidelity (centroid cosine + Fréchet distance)

Usage:
    python -m src.visualization.results_visualizer                  # defaults
    python -m src.visualization.results_visualizer --no-umap        # skip slow UMAP
    python -m src.visualization.results_visualizer --dit-history models/checkpoints/dit_history.json
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

matplotlib.use("Agg")  # non-interactive backend for PDF generation

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Publication style — Nature/Cell conventions
# ──────────────────────────────────────────────────────────────
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.edgecolor": "0.8",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "figure.constrained_layout.use": True,
    "figure.facecolor": "white",
}

# Colour palette for 69 cell types — deterministic, colourblind-friendly
def _build_type_palette(n: int = 69) -> np.ndarray:
    """Generate n distinct colours via HSL spacing."""
    cmap = matplotlib.colormaps.get_cmap("gist_ncar").resampled(n + 4)
    colours = cmap(np.linspace(0.02, 0.95, n))
    rng = np.random.default_rng(42)
    order = rng.permutation(n)
    return colours[order]


TYPE_PALETTE = _build_type_palette(69)


# ──────────────────────────────────────────────────────────────
# Main Visualizer
# ──────────────────────────────────────────────────────────────
class ResultsVisualizer:
    """Publication-quality visualization for CLOP-DiT results.

    Parameters
    ----------
    clop_history_path : path to CLOP training history JSON (dict of lists)
    dit_history_path  : path to DiT training history JSON (dict of lists)
    cache_dir         : cached latents directory
    output_dir        : where to save generated figures
    dpi               : figure DPI
    """

    def __init__(
        self,
        clop_history_path: str = "models/checkpoints/clop_history.json",
        dit_history_path: Optional[str] = "models/checkpoints/dit_history.json",
        cache_dir: str = "data/cached_latents_v5.2",
        output_dir: str = "results/figures",
        dpi: int = 300,
    ):
        self.cache = Path(cache_dir)
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        # Load CLOP history
        clop_path = Path(clop_history_path)
        if clop_path.exists():
            with open(clop_path) as f:
                self.clop_hist: Dict = json.load(f)
            logger.info(f"CLOP history: {len(next(iter(self.clop_hist.values())))} epochs")
        else:
            self.clop_hist = {}
            logger.warning(f"CLOP history not found: {clop_path}")

        # Load DiT history
        dit_path = Path(dit_history_path) if dit_history_path else None
        if dit_path and dit_path.exists():
            with open(dit_path) as f:
                self.dit_hist: Dict = json.load(f)
            logger.info(f"DiT history: {len(next(iter(self.dit_hist.values())))} epochs")
        else:
            self.dit_hist = {}
            logger.warning("DiT history not found or not provided")

        # Load caption metadata
        cap_path = self.cache / "text_captions_deduplicated.json"
        if cap_path.exists():
            with open(cap_path) as f:
                raw = json.load(f)
            # Extract short names: text before " are "
            self.type_names = {}
            for k, v in raw.items():
                name = v.split(" are ")[0] if " are " in v else v[:50]
                self.type_names[int(k)] = name
            logger.info(f"Loaded {len(self.type_names)} cell type names")
        else:
            self.type_names = {}

        matplotlib.rcParams.update(STYLE)

    # ──────────────────────────────────────────────────────────
    # PANEL A: CLOP Training Dynamics
    # ──────────────────────────────────────────────────────────
    def plot_clop_training(self, save: bool = True) -> Optional[plt.Figure]:
        """4-panel CLOP training dynamics.

        A1: Train/Val contrastive loss
        A2: Temperature curve (fixed @ 14.0)
        A3: Prototype accuracy (train/val + top-5, top-10)
        A4: Embedding quality (alignment, uniformity, inter-sep, t↔c alignment)
        """
        if not self.clop_hist:
            logger.warning("No CLOP history — skipping Panel A")
            return None

        h = self.clop_hist
        epochs = np.arange(1, len(h["train_loss"]) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("CLOP Contrastive Pre-training (v9.3)", fontsize=14, fontweight="bold")

        # A1: Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, h["train_loss"], label="Train", color="#2196F3")
        ax.plot(epochs, h["val_loss"], label="Val", color="#FF5722", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Contrastive Loss")
        ax.set_title("A1: Loss Convergence")
        ax.legend()
        # Annotate final values
        ax.annotate(f'{h["train_loss"][-1]:.4f}', xy=(epochs[-1], h["train_loss"][-1]),
                    fontsize=8, color="#2196F3", ha="right")
        ax.annotate(f'{h["val_loss"][-1]:.4f}', xy=(epochs[-1], h["val_loss"][-1]),
                    fontsize=8, color="#FF5722", ha="right")

        # A2: Temperature stability
        ax = axes[0, 1]
        ax.plot(epochs, h["temperature"], color="#4CAF50", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Temperature (τ)")
        ax.set_title("A2: Temperature Stability")
        ax.axhline(y=14.0, color="gray", linestyle=":", alpha=0.5, label="τ=14.0 (fixed)")
        ax.set_ylim(13.5, 14.5)
        ax.legend()

        # A3: Prototype accuracy
        ax = axes[1, 0]
        ax.plot(epochs, np.array(h["val_proto_acc"]) * 100, label="Val Proto Acc", color="#9C27B0", linewidth=2)
        ax.plot(epochs, np.array(h["train_proto_acc"]) * 100, label="Train Proto Acc",
                color="#9C27B0", linestyle=":", alpha=0.6)
        if "val_proto_top5" in h:
            ax.plot(epochs, np.array(h["val_proto_top5"]) * 100, label="Val Top-5",
                    color="#00BCD4", linestyle="--")
        if "val_proto_top10" in h:
            ax.plot(epochs, np.array(h["val_proto_top10"]) * 100, label="Val Top-10",
                    color="#8BC34A", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("A3: Cell-Type Classification Accuracy")
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right", fontsize=8)
        # Annotate convergence
        final_acc = h["val_proto_acc"][-1] * 100
        ax.annotate(f'{final_acc:.1f}%', xy=(epochs[-1], final_acc),
                    fontsize=9, fontweight="bold", color="#9C27B0", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#9C27B0", alpha=0.8))

        # A4: Embedding quality metrics
        ax = axes[1, 1]
        quality_metrics = [
            ("val_text_cell_align", "Text↔Cell Alignment", "#E91E63"),
            ("val_inter_sep", "Inter-type Separation", "#FF9800"),
            ("val_mean_cosine_sim", "Mean Cosine Sim", "#3F51B5"),
        ]
        for key, label, color in quality_metrics:
            if key in h:
                ax.plot(epochs, h[key], label=label, color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("A4: Embedding Quality Metrics")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=8)

        if save:
            path = self.output / "panel_a_clop_training.png"
            fig.savefig(path, dpi=self.dpi)
            fig.savefig(path.with_suffix(".pdf"), dpi=self.dpi)
            logger.info(f"Saved Panel A → {path}")
        return fig

    # ──────────────────────────────────────────────────────────
    # PANEL B: CLOP Alignment Space (UMAP) — all in CLOP shared space
    # ──────────────────────────────────────────────────────────
    def plot_clop_embedding_space(
        self, n_cells: int = 8000, save: bool = True
    ) -> Optional[plt.Figure]:
        """3-panel CLOP alignment visualization.

        IMPORTANT: Both cells and text must be in the *same* CLOP projection
        space.  projected_cells.npy = CLOPAligner.project_cell(cell_emb),
        projected_text.npy = CLOPAligner.project_text(text_emb).
        Using raw cell_embeddings_dedup_preprocessed.npy would mix two
        different representation spaces and produce misleading UMAP.

        B0: Cells-per-type histogram (data balance check)
        B1: 69 text prototypes (centroids of projected text per type), labelled
        B2: Subsampled CLOP-projected cells coloured by type + text proto overlay
        """
        # ── Load CLOP-projected data (SAME space) ──
        proj_text_path = self.cache / "projected_text.npy"
        proj_cell_path = self.cache / "projected_cells.npy"
        gid_path = self.cache / "text_group_ids_dedup.npy"

        for p in [proj_text_path, gid_path]:
            if not p.exists():
                logger.warning(f"Missing {p.name} — skipping Panel B")
                return None

        if not proj_cell_path.exists():
            logger.warning(
                f"Missing projected_cells.npy — run CLOP cell projection first.\n"
                f"  Hint: project cells through CLOPAligner.project_cell() and save.\n"
                f"  Falling back to raw cell embeddings (DIFFERENT space — prototypes "
                f"  will NOT align with cells in UMAP)."
            )
            fallback_path = self.cache / "cell_embeddings_dedup_preprocessed.npy"
            if not fallback_path.exists():
                logger.warning(f"Missing {fallback_path.name} too — skipping Panel B")
                return None
            cell_proj = np.load(fallback_path)
            space_label = "scGPT latent (WARNING: different space from prototypes)"
        else:
            cell_proj = np.load(proj_cell_path)
            space_label = "CLOP shared projection"

        proj_text = np.load(proj_text_path)  # (167245, 512) — CLOP-projected, per cell
        group_ids = np.load(gid_path)  # (167245,)
        logger.info(
            f"Panel B: {cell_proj.shape[0]} cells ({space_label}), "
            f"{proj_text.shape[0]} text conditions"
        )

        # Compute per-type text prototype centroids (in CLOP space)
        unique_types = np.unique(group_ids)
        n_types = len(unique_types)
        proto_dim = proj_text.shape[1]
        text_proto = np.zeros((n_types, proto_dim), dtype=np.float32)
        for i, t in enumerate(unique_types):
            text_proto[i] = proj_text[group_ids == t].mean(axis=0)
        # L2-normalize prototypes (they should already be near-unit but ensure)
        norms = np.linalg.norm(text_proto, axis=1, keepdims=True) + 1e-8
        text_proto = text_proto / norms
        logger.info(f"Computed {n_types} text prototype centroids")

        # Cells-per-type counts
        type_counts = np.array([np.sum(group_ids == t) for t in unique_types])

        # ── Stratified subsample for UMAP ──
        rng = np.random.default_rng(42)
        n_per_type = max(10, n_cells // n_types)
        sampled_idx = []
        for t in unique_types:
            t_idx = np.where(group_ids == t)[0]
            n = min(n_per_type, len(t_idx))
            sampled_idx.extend(rng.choice(t_idx, n, replace=False).tolist())
        sampled_idx = np.array(sampled_idx)
        rng.shuffle(sampled_idx)

        cell_sub = cell_proj[sampled_idx]  # CLOP-projected cells
        gids_sub = group_ids[sampled_idx]

        logger.info(f"UMAP: {len(sampled_idx)} cells + {n_types} prototypes (all in CLOP space)")

        # ── Compute UMAP on combined (cells + text prototypes, SAME space) ──
        import umap as umap_lib
        combined = np.vstack([cell_sub, text_proto])
        reducer = umap_lib.UMAP(
            n_components=2, n_neighbors=30, min_dist=0.3,
            metric="cosine", random_state=42
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coords = reducer.fit_transform(combined)

        cell_coords = coords[:len(sampled_idx)]
        proto_coords = coords[len(sampled_idx):]

        # ── Plot (3 panels) ──
        fig = plt.figure(figsize=(22, 8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 1.2], wspace=0.25)
        fig.suptitle("CLOP Alignment Space (UMAP — all embeddings in shared CLOP projection)",
                     fontsize=14, fontweight="bold")

        # B0: Cells-per-type histogram
        ax = fig.add_subplot(gs[0])
        sorted_order = np.argsort(type_counts)[::-1]
        bar_colors = [TYPE_PALETTE[t % len(TYPE_PALETTE)] for t in unique_types[sorted_order]]
        bar_labels = [self.type_names.get(int(t), f"Type {t}")[:20] for t in unique_types[sorted_order]]
        y_pos = np.arange(n_types)
        ax.barh(y_pos, type_counts[sorted_order], color=bar_colors, height=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bar_labels, fontsize=5)
        ax.invert_yaxis()
        ax.set_xlabel("Cells")
        ax.set_title("B0: Cells per Type")
        # Annotate min/max
        ax.axvline(x=np.median(type_counts), color="red", linestyle="--", alpha=0.5, label=f"median={int(np.median(type_counts))}")
        ax.legend(fontsize=7)

        # B1: Prototypes only (labelled)
        ax = fig.add_subplot(gs[1])
        for i, (x, y) in enumerate(proto_coords):
            color = TYPE_PALETTE[i % len(TYPE_PALETTE)]
            ax.scatter(x, y, c=[color], s=120, marker="D", edgecolors="black",
                       linewidths=0.6, zorder=5)
            name = self.type_names.get(int(unique_types[i]), f"Type {i}")
            short = name[:25] + "…" if len(name) > 25 else name
            ax.annotate(short, (x, y), fontsize=5.5, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points")

        ax.set_title("B1: 69 Text Prototypes (CLOP space)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        # B2: Cells coloured by type + prototype centroids
        ax = fig.add_subplot(gs[2])
        for t in unique_types:
            mask = gids_sub == t
            color = TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]
            ax.scatter(cell_coords[mask, 0], cell_coords[mask, 1],
                       c=[color], s=3, alpha=0.4, rasterized=True)
        # Overlay text prototypes as stars
        for i, (x, y) in enumerate(proto_coords):
            color = TYPE_PALETTE[int(unique_types[i]) % len(TYPE_PALETTE)]
            ax.scatter(x, y, c=[color], s=80, marker="*", edgecolors="black",
                       linewidths=0.5, zorder=6)

        ax.set_title(f"B2: {len(sampled_idx)} Cells + ★ Text Proto (CLOP space)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        if save:
            path = self.output / "panel_b_clop_embedding_umap.png"
            fig.savefig(path, dpi=self.dpi)
            fig.savefig(path.with_suffix(".pdf"), dpi=self.dpi)
            logger.info(f"Saved Panel B → {path}")
        return fig

    # ──────────────────────────────────────────────────────────
    # PANEL C: DiT Training Dynamics
    # ──────────────────────────────────────────────────────────
    def plot_dit_training(self, save: bool = True) -> Optional[plt.Figure]:
        """3-panel DiT flow-matching training dynamics.

        C1: Train/Val loss (flow-matching MSE)
        C2: Validation cosine similarity
        C3: Learning rate schedule
        """
        if not self.dit_hist:
            logger.warning("No DiT history — skipping Panel C")
            return None

        h = self.dit_hist
        epochs = np.arange(1, len(h["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        fig.suptitle("DiT Flow-Matching Training", fontsize=14, fontweight="bold")

        # C1: Loss
        ax = axes[0]
        ax.plot(epochs, h["train_loss"], label="Train MSE", color="#2196F3")
        ax.plot(epochs, h["val_loss"], label="Val MSE", color="#FF5722", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Flow-Matching Loss")
        ax.set_title("C1: Loss Convergence")
        ax.set_yscale("log")
        ax.legend()
        # Annotate final
        ax.annotate(f'{h["train_loss"][-1]:.4f}',
                    xy=(epochs[-1], h["train_loss"][-1]),
                    fontsize=8, color="#2196F3", ha="right")
        ax.annotate(f'{h["val_loss"][-1]:.4f}',
                    xy=(epochs[-1], h["val_loss"][-1]),
                    fontsize=8, color="#FF5722", ha="right")

        # C2: Cosine similarity
        ax = axes[1]
        ax.plot(epochs, h["val_cosine_sim"], color="#4CAF50", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("C2: Generation Fidelity (Cosine)")
        ax.set_ylim(0.6, 1.0)
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
        final_cos = h["val_cosine_sim"][-1]
        ax.annotate(f'{final_cos:.4f}',
                    xy=(epochs[-1], final_cos),
                    fontsize=9, fontweight="bold", color="#4CAF50",
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#4CAF50", alpha=0.8))

        # C3: Learning rate
        ax = axes[2]
        ax.plot(epochs, h["lr"], color="#9C27B0", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("C3: LR Schedule (Cosine Decay)")
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(-4, -4))

        if save:
            path = self.output / "panel_c_dit_training.png"
            fig.savefig(path, dpi=self.dpi)
            fig.savefig(path.with_suffix(".pdf"), dpi=self.dpi)
            logger.info(f"Saved Panel C → {path}")
        return fig

    # ──────────────────────────────────────────────────────────
    # PANEL D: Metrics Summary
    # ──────────────────────────────────────────────────────────
    def plot_metrics_summary(self, gen_metrics_path: str = "results/generation_metrics.json",
                             save: bool = True) -> Optional[plt.Figure]:
        """Clean metrics summary table as a figure panel."""
        rows = []

        if self.clop_hist:
            h = self.clop_hist
            rows.extend([
                ["CLOP", "Train Loss", f'{h["train_loss"][-1]:.4f}', f'{h["train_loss"][0]:.4f}'],
                ["CLOP", "Val Loss", f'{h["val_loss"][-1]:.4f}', f'{h["val_loss"][0]:.4f}'],
                ["CLOP", "Val Proto Accuracy", f'{h["val_proto_acc"][-1]*100:.1f}%',
                 f'{h["val_proto_acc"][0]*100:.1f}%'],
                ["CLOP", "Val Top-5 Accuracy", f'{h["val_proto_top5"][-1]*100:.1f}%',
                 f'{h["val_proto_top5"][0]*100:.1f}%'],
                ["CLOP", "Text↔Cell Alignment", f'{h["val_text_cell_align"][-1]:.4f}',
                 f'{h["val_text_cell_align"][0]:.4f}'],
                ["CLOP", "Inter-type Separation", f'{h["val_inter_sep"][-1]:.4f}',
                 f'{h["val_inter_sep"][0]:.4f}'],
                ["CLOP", "Mean Cosine Sim", f'{h["val_mean_cosine_sim"][-1]:.4f}',
                 f'{h["val_mean_cosine_sim"][0]:.4f}'],
                ["CLOP", "Temperature (τ)", f'{h["temperature"][-1]:.1f}',
                 f'{h["temperature"][0]:.1f}'],
            ])

        if self.dit_hist:
            h = self.dit_hist
            rows.extend([
                ["DiT", "Train Loss", f'{h["train_loss"][-1]:.4f}', f'{h["train_loss"][0]:.4f}'],
                ["DiT", "Val Loss", f'{h["val_loss"][-1]:.4f}', f'{h["val_loss"][0]:.4f}'],
                ["DiT", "Val Cosine Sim", f'{h["val_cosine_sim"][-1]:.4f}',
                 f'{h["val_cosine_sim"][0]:.4f}'],
            ])

        # Add generation metrics if available
        gen_path = Path(gen_metrics_path)
        if gen_path.exists():
            with open(gen_path) as f:
                gen_data = json.load(f)
            overall = gen_data.get("overall", {})
            summary = gen_data.get("summary", {})
            rows.extend([
                ["Gen", "Fréchet Distance ↓", f'{overall.get("frechet_distance", 0):.4f}', "—"],
                ["Gen", "MMD-RBF ↓", f'{overall.get("mmd_rbf", 0):.6f}', "—"],
                ["Gen", "Coverage ↑", f'{overall.get("coverage", 0):.4f}', "—"],
                ["Gen", "Density", f'{overall.get("density", 0):.2f}', "—"],
                ["Gen", "Diversity Index", f'{overall.get("diversity_index", 0):.4f}', "—"],
                ["Gen", "Mean Centroid Cosine", f'{summary.get("mean_centroid_cosine", 0):.4f}', "—"],
                ["Gen", "Min Centroid Cosine", f'{summary.get("min_centroid_cosine", 0):.4f}', "—"],
            ])

        if not rows:
            return None

        fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 1.5))
        ax.axis("off")
        ax.set_title("Metrics Summary", fontsize=14, fontweight="bold", pad=20)

        col_labels = ["Stage", "Metric", "Final", "Initial"]
        table = ax.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Style header
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")

        # Alternate row colours
        for i in range(len(rows)):
            stage = rows[i][0]
            for j in range(len(col_labels)):
                cell = table[i + 1, j]
                if stage == "CLOP":
                    cell.set_facecolor("#E3F2FD" if i % 2 == 0 else "#BBDEFB")
                elif stage == "DiT":
                    cell.set_facecolor("#E8F5E9" if i % 2 == 0 else "#C8E6C9")
                else:  # Gen
                    cell.set_facecolor("#FFF3E0" if i % 2 == 0 else "#FFE0B2")

        if save:
            path = self.output / "panel_d_metrics_summary.png"
            fig.savefig(path, dpi=self.dpi)
            fig.savefig(path.with_suffix(".pdf"), dpi=self.dpi)
            logger.info(f"Saved Panel D → {path}")
        return fig

    # ──────────────────────────────────────────────────────────
    # PANEL E: Real vs Generated UMAP (post-inference) — type-coloured
    # ──────────────────────────────────────────────────────────
    def plot_real_vs_generated(
        self,
        generated_path: Optional[str] = None,
        generated_labels_path: Optional[str] = None,
        n_cells: int = 5000,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """3-panel real vs generated comparison with type colouring.

        E1: Real cells (UMAP, coloured by type)
        E2: Generated cells (UMAP, coloured by type)
        E3: Overlay (real=circles, generated=triangles)
        """
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

        cell_path = self.cache / "cell_embeddings_dedup_preprocessed.npy"
        gid_path = self.cache / "text_group_ids_dedup.npy"
        if not cell_path.exists():
            logger.warning(f"Missing {cell_path.name}")
            return None

        real = np.load(cell_path)
        gen = np.load(generated_path)
        real_gids = np.load(gid_path) if gid_path.exists() else None
        gen_gids = np.load(generated_labels_path) if generated_labels_path and Path(generated_labels_path).exists() else None
        logger.info(f"Real: {real.shape}, Generated: {gen.shape}")

        # Stratified subsample of real cells
        rng = np.random.default_rng(42)
        n_real = min(n_cells, len(real))
        n_gen = min(n_cells, len(gen))
        real_idx = rng.choice(len(real), n_real, replace=False)
        gen_idx = rng.choice(len(gen), n_gen, replace=False) if len(gen) > n_cells else np.arange(len(gen))

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

        real_c = coords[:len(real_sub)]
        gen_c = coords[len(real_sub):]

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle("Real vs Generated Cell Embeddings (DiT v1)", fontsize=14, fontweight="bold")

        # E1: Real — type-coloured
        ax = axes[0]
        if real_gids_sub is not None:
            for t in np.unique(real_gids_sub):
                mask = real_gids_sub == t
                color = TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]
                ax.scatter(real_c[mask, 0], real_c[mask, 1], c=[color], s=4,
                           alpha=0.4, rasterized=True)
        else:
            ax.scatter(real_c[:, 0], real_c[:, 1], c="#2196F3", s=4, alpha=0.4, rasterized=True)
        ax.set_title(f"E1: Real ({len(real_sub)} cells, {69 if real_gids_sub is not None else '?'} types)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        # E2: Generated — type-coloured
        ax = axes[1]
        if gen_gids_sub is not None:
            for t in np.unique(gen_gids_sub):
                mask = gen_gids_sub == t
                color = TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]
                ax.scatter(gen_c[mask, 0], gen_c[mask, 1], c=[color], s=4,
                           alpha=0.4, rasterized=True)
        else:
            ax.scatter(gen_c[:, 0], gen_c[:, 1], c="#FF5722", s=4, alpha=0.4, rasterized=True)
        ax.set_title(f"E2: Generated ({len(gen_sub)} cells)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        # E3: Overlay
        ax = axes[2]
        ax.scatter(real_c[:, 0], real_c[:, 1], c="#2196F3", s=3, alpha=0.25,
                   label="Real", rasterized=True)
        ax.scatter(gen_c[:, 0], gen_c[:, 1], c="#FF5722", s=3, alpha=0.25,
                   marker="^", label="Generated", rasterized=True)
        ax.legend(markerscale=5, fontsize=10)
        ax.set_title("E3: Overlay")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        if save:
            path = self.output / "panel_e_real_vs_generated.png"
            fig.savefig(path, dpi=self.dpi)
            fig.savefig(path.with_suffix(".pdf"), dpi=self.dpi)
            logger.info(f"Saved Panel E → {path}")
        return fig

    # ──────────────────────────────────────────────────────────
    # PANEL F: Text–Cell Cosine Similarity Heatmap (69×69)
    # ──────────────────────────────────────────────────────────
    def plot_text_cell_heatmap(self, save: bool = True) -> Optional[plt.Figure]:
        """69×69 cosine similarity matrix between text prototypes and cell centroids.

        Proves CLOP alignment: the diagonal should be bright (text matches its cells).
        Off-diagonal shows inter-type confusion patterns.
        """
        proj_text_path = self.cache / "projected_text.npy"
        proj_cell_path = self.cache / "projected_cells.npy"
        gid_path = self.cache / "text_group_ids_dedup.npy"

        for p in [proj_text_path, proj_cell_path, gid_path]:
            if not p.exists():
                logger.warning(f"Missing {p.name} — skipping Panel F")
                return None

        proj_text = np.load(proj_text_path)
        proj_cells = np.load(proj_cell_path)
        group_ids = np.load(gid_path)

        unique_types = np.sort(np.unique(group_ids))
        n_types = len(unique_types)

        # Compute per-type centroids
        text_centroids = np.zeros((n_types, proj_text.shape[1]), dtype=np.float32)
        cell_centroids = np.zeros((n_types, proj_cells.shape[1]), dtype=np.float32)
        for i, t in enumerate(unique_types):
            mask = group_ids == t
            tc = proj_text[mask].mean(axis=0)
            text_centroids[i] = tc / (np.linalg.norm(tc) + 1e-8)
            cc = proj_cells[mask].mean(axis=0)
            cell_centroids[i] = cc / (np.linalg.norm(cc) + 1e-8)

        # Cosine similarity: text_centroids @ cell_centroids.T → (69, 69)
        sim_matrix = text_centroids @ cell_centroids.T

        # Type labels (short)
        labels = [self.type_names.get(int(t), f"T{t}")[:20] for t in unique_types]

        # Diagonal values
        diag = np.diag(sim_matrix)
        mean_diag = diag.mean()
        off_diag = sim_matrix[~np.eye(n_types, dtype=bool)]
        mean_off = off_diag.mean()

        fig, axes = plt.subplots(1, 2, figsize=(18, 10),
                                 gridspec_kw={"width_ratios": [1.3, 0.7]})
        fig.suptitle(
            f"Text–Cell Alignment Heatmap (CLOP Space) — "
            f"diag={mean_diag:.3f}, off-diag={mean_off:.3f}",
            fontsize=14, fontweight="bold",
        )

        # F1: Full heatmap
        ax = axes[0]
        im = ax.imshow(sim_matrix, cmap="RdYlBu_r", vmin=-0.1, vmax=1.0, aspect="auto")
        ax.set_xticks(range(n_types))
        ax.set_yticks(range(n_types))
        ax.set_xticklabels(labels, rotation=90, fontsize=5, ha="center")
        ax.set_yticklabels(labels, fontsize=5)
        ax.set_xlabel("Cell Type (cell centroids)")
        ax.set_ylabel("Cell Type (text prototypes)")
        ax.set_title("F1: Cosine Similarity Matrix")
        fig.colorbar(im, ax=ax, shrink=0.6, label="Cosine Similarity")

        # F2: Diagonal values bar chart (sorted)
        ax = axes[1]
        sorted_idx = np.argsort(diag)
        sorted_diag = diag[sorted_idx]
        sorted_labels = [labels[i] for i in sorted_idx]
        colors = ["#4CAF50" if v > 0.8 else "#FF9800" if v > 0.5 else "#F44336" for v in sorted_diag]
        ax.barh(range(n_types), sorted_diag, color=colors, height=0.8)
        ax.set_yticks(range(n_types))
        ax.set_yticklabels(sorted_labels, fontsize=5)
        ax.set_xlabel("Diagonal Cosine Similarity")
        ax.set_title("F2: Per-Type Text–Cell Alignment")
        ax.axvline(x=mean_diag, color="red", linestyle="--", alpha=0.5,
                    label=f"mean={mean_diag:.3f}")
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=8)

        if save:
            path = self.output / "panel_f_text_cell_heatmap.png"
            fig.savefig(path, dpi=self.dpi)
            fig.savefig(path.with_suffix(".pdf"), dpi=self.dpi)
            logger.info(f"Saved Panel F → {path}")
        return fig

    # ──────────────────────────────────────────────────────────
    # PANEL G: Per-Type Generation Fidelity
    # ──────────────────────────────────────────────────────────
    def plot_per_type_generation(
        self,
        metrics_path: str = "results/generation_metrics.json",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Per-type generation quality: centroid cosine + Fréchet distance.

        G1: Centroid cosine per type (sorted)
        G2: Fréchet distance per type (sorted, log scale)
        G3: Scatter of centroid cosine vs n_real (type size dependency)
        """
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

        # Short labels
        short_names = [n[:22] for n in names]

        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        summary = data.get("summary", {})
        overall = data.get("overall", {})
        fig.suptitle(
            f"Per-Type Generation Fidelity — "
            f"mean cos={summary.get('mean_centroid_cosine', 0):.3f}, "
            f"overall FD={overall.get('frechet_distance', 0):.3f}",
            fontsize=14, fontweight="bold",
        )

        # G1: Centroid cosine (sorted)
        ax = axes[0]
        sorted_idx = np.argsort(cosines)
        sorted_cos = [cosines[i] for i in sorted_idx]
        sorted_names_cos = [short_names[i] for i in sorted_idx]
        colors = ["#4CAF50" if v > 0.9 else "#FF9800" if v > 0.7 else "#F44336" for v in sorted_cos]
        ax.barh(range(len(sorted_cos)), sorted_cos, color=colors, height=0.8)
        ax.set_yticks(range(len(sorted_cos)))
        ax.set_yticklabels(sorted_names_cos, fontsize=5)
        ax.set_xlabel("Centroid Cosine Similarity")
        ax.set_title("G1: Real↔Gen Centroid Cosine")
        ax.axvline(x=summary.get("mean_centroid_cosine", 0), color="red",
                    linestyle="--", alpha=0.5, label=f"mean={summary.get('mean_centroid_cosine', 0):.3f}")
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=8)

        # G2: Fréchet distance (sorted, larger = worse)
        ax = axes[1]
        valid_fd = [(n, f) for n, f in zip(short_names, fds) if np.isfinite(f)]
        if valid_fd:
            fd_names, fd_vals = zip(*sorted(valid_fd, key=lambda x: x[1]))
            colors_fd = ["#4CAF50" if v < 0.5 else "#FF9800" if v < 2.0 else "#F44336" for v in fd_vals]
            ax.barh(range(len(fd_vals)), fd_vals, color=colors_fd, height=0.8)
            ax.set_yticks(range(len(fd_vals)))
            ax.set_yticklabels(fd_names, fontsize=5)
            ax.set_xlabel("Fréchet Distance (lower = better)")
            ax.set_title("G2: Per-Type Fréchet Distance")
        else:
            ax.text(0.5, 0.5, "No valid FD values", ha="center", va="center",
                    transform=ax.transAxes)

        # G3: Cosine vs dataset size
        ax = axes[2]
        ax.scatter(n_real, cosines, c="#3F51B5", s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
        # Label outliers (low cosine)
        for i, (nr, cos) in enumerate(zip(n_real, cosines)):
            if cos < 0.7:
                ax.annotate(short_names[i][:12], (nr, cos), fontsize=6,
                            xytext=(5, -5), textcoords="offset points")
        ax.set_xlabel("Number of Real Cells")
        ax.set_ylabel("Centroid Cosine Similarity")
        ax.set_title("G3: Fidelity vs Dataset Size")
        ax.axhline(y=0.9, color="green", linestyle=":", alpha=0.4, label="cos=0.9")
        ax.legend(fontsize=8)

        if save:
            path = self.output / "panel_g_per_type_generation.png"
            fig.savefig(path, dpi=self.dpi)
            fig.savefig(path.with_suffix(".pdf"), dpi=self.dpi)
            logger.info(f"Saved Panel G → {path}")
        return fig

    # ──────────────────────────────────────────────────────────
    # COMBINED REPORT
    # ──────────────────────────────────────────────────────────
    def generate_full_report(self, include_umap: bool = True) -> List[Path]:
        """Generate all available panels and a combined multi-page PDF.

        Returns list of saved file paths.
        """
        saved = []

        logger.info("=" * 60)
        logger.info("Generating CLOP-DiT Results Report")
        logger.info("=" * 60)

        # Panel A: CLOP training
        fig_a = self.plot_clop_training()
        if fig_a:
            saved.append(self.output / "panel_a_clop_training.pdf")
            plt.close(fig_a)

        # Panel B: Embedding UMAP
        if include_umap:
            fig_b = self.plot_clop_embedding_space()
            if fig_b:
                saved.append(self.output / "panel_b_clop_embedding_umap.pdf")
                plt.close(fig_b)
        else:
            logger.info("Skipping Panel B (UMAP) — use --include-umap to enable")

        # Panel C: DiT training
        fig_c = self.plot_dit_training()
        if fig_c:
            saved.append(self.output / "panel_c_dit_training.pdf")
            plt.close(fig_c)

        # Panel D: Summary table
        fig_d = self.plot_metrics_summary()
        if fig_d:
            saved.append(self.output / "panel_d_metrics_summary.pdf")
            plt.close(fig_d)

        # Panel E: Real vs Generated (if available)
        fig_e = self.plot_real_vs_generated()
        if fig_e:
            saved.append(self.output / "panel_e_real_vs_generated.pdf")
            plt.close(fig_e)

        # Panel F: Text-Cell similarity heatmap
        fig_f = self.plot_text_cell_heatmap()
        if fig_f:
            saved.append(self.output / "panel_f_text_cell_heatmap.pdf")
            plt.close(fig_f)

        # Panel G: Per-type generation fidelity
        fig_g = self.plot_per_type_generation()
        if fig_g:
            saved.append(self.output / "panel_g_per_type_generation.pdf")
            plt.close(fig_g)

        # ── Combine into multi-page PDF ──
        if saved:
            from matplotlib.backends.backend_pdf import PdfPages
            combined_path = self.output / "clop_dit_full_report.pdf"
            with PdfPages(combined_path) as pdf:
                for panel_path in saved:
                    if panel_path.exists():
                        # Re-read image and add to PDF
                        img = plt.imread(str(panel_path.with_suffix(".png")))
                        fig_tmp, ax_tmp = plt.subplots(
                            figsize=(img.shape[1] / 100, img.shape[0] / 100)
                        )
                        ax_tmp.imshow(img)
                        ax_tmp.axis("off")
                        pdf.savefig(fig_tmp, bbox_inches="tight", pad_inches=0.1)
                        plt.close(fig_tmp)

            saved.append(combined_path)
            logger.info(f"Combined report → {combined_path}")

        logger.info(f"Done — {len(saved)} files generated in {self.output}/")
        return saved


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────
def main():
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Generate CLOP-DiT results report")
    parser.add_argument("--clop-history", default="models/checkpoints/clop_history.json")
    parser.add_argument("--dit-history", default="models/checkpoints/dit_history.json")
    parser.add_argument("--cache-dir", default="data/cached_latents_v5.2")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--no-umap", action="store_true", help="Skip UMAP (faster)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--generated", default=None, help="Path to generated cells .npy")
    args = parser.parse_args()

    viz = ResultsVisualizer(
        clop_history_path=args.clop_history,
        dit_history_path=args.dit_history,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    saved = viz.generate_full_report(include_umap=not args.no_umap)
    print(f"\nGenerated {len(saved)} files:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
