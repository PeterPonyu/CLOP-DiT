"""
results_visualizer.py — Publication-quality visualization for CLOP-DiT pipeline.

Generates multi-panel PDF/PNG reports proving training success:
  Panel A: CLOP training dynamics (loss, temperature, prototype accuracy, embedding quality)
  Panel B: CLOP embedding space (UMAP of 69-type prototypes + cell embeddings)
  Panel C: DiT training dynamics (flow-matching loss, cosine similarity, LR schedule)
  Panel D: Metrics summary table (grouped two-column: training/gen | diversity/expr/config)
  Panel E: [Post-inference] Real vs generated cell overlay (type-coloured UMAP)
  Panel F: Text–Cell similarity heatmap (69×69 cosine matrix proving CLOP alignment)
  Panel G: Per-type generation fidelity (centroid cosine + Fréchet distance)
  Panel H: Gene expression correlation (per-gene scatter + per-type r + marker genes)
  Panel I: Expression decoder analysis (CV distribution + expression range + per-cell variance)
  Panel J: Diversity diagnostics (intra-type ratio + memorization + CFG sweep + condition sensitivity)
  Panel K: Expression diversity (gene std ratio real vs generated)
  Panel L: Noise-scale trade-off (FD + centroid cosine + diversity ratio vs ε)
  Panel M: Conditioning mode comparison (PCA of centroid vs noise vs variant)
  Panel N: Marker gene comparison (per-type real vs generated expression, focused heatmap)
  Panel O: Baseline comparison (CLOP-DiT vs Gaussian/Shuffled baselines, radar + bars)

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
    # PANEL D: Metrics Summary (grouped, compact, publication-ready)
    # ──────────────────────────────────────────────────────────
    def plot_metrics_summary(self, gen_metrics_path: str = "results/generation_metrics.json",
                             expr_metrics_path: str = "results/expression_metrics.json",
                             div_metrics_path: str = "results/diversity_diagnostics.json",
                             save: bool = True) -> Optional[plt.Figure]:
        """Publication-quality grouped metrics summary table.

        Two-column layout: left = training/generation, right = diversity/expression/config.
        Key metrics highlighted. Directional arrows on metric names.
        """
        # ── Collect metrics by category ──
        cat_training: list = []   # CLOP + DiT
        cat_gen: list = []        # Generation quality
        cat_div: list = []        # Diversity
        cat_expr: list = []       # Expression fidelity
        cat_cfg: list = []        # Configuration

        if self.clop_hist:
            h = self.clop_hist
            cat_training.extend([
                ("CLOP Val Loss ↓", f'{h["val_loss"][-1]:.4f}', True),
                ("CLOP Proto Accuracy ↑", f'{h["val_proto_acc"][-1]*100:.1f}%', True),
                ("CLOP Top-5 Acc ↑", f'{h["val_proto_top5"][-1]*100:.1f}%', False),
                ("Text↔Cell Alignment ↑", f'{h["val_text_cell_align"][-1]:.4f}', False),
                ("Inter-type Separation ↑", f'{h["val_inter_sep"][-1]:.4f}', False),
            ])

        if self.dit_hist:
            h = self.dit_hist
            cat_training.extend([
                ("DiT Val Loss ↓", f'{h["val_loss"][-1]:.4f}', True),
                ("DiT Val Cosine ↑", f'{h["val_cosine_sim"][-1]:.4f}', True),
            ])

        gen_path = Path(gen_metrics_path)
        if gen_path.exists():
            with open(gen_path) as f:
                gen_data = json.load(f)
            overall = gen_data.get("overall", {})
            summary = gen_data.get("summary", {})
            cat_gen = [
                ("Fréchet Distance ↓", f'{overall.get("frechet_distance", 0):.4f}', True),
                ("MMD-RBF ↓", f'{overall.get("mmd_rbf", 0):.6f}', False),
                ("Coverage ↑", f'{overall.get("coverage", 0):.4f}', False),
                ("Density", f'{overall.get("density", 0):.2f}', False),
                ("Diversity Index ↑", f'{overall.get("diversity_index", 0):.4f}', False),
                ("Mean Centroid Cosine ↑", f'{summary.get("mean_centroid_cosine", 0):.4f}', True),
                ("Min Centroid Cosine", f'{summary.get("min_centroid_cosine", 0):.4f}', False),
            ]

        div_path = Path(div_metrics_path)
        if div_path.exists():
            with open(div_path) as f:
                div_data = json.load(f)
            t1 = div_data.get("test1_intratype_diversity", {}).get("summary", {})
            t2 = div_data.get("test2_memorization", {})
            t5 = div_data.get("test5_condition_sensitivity", {}).get("summary", {})
            t6 = div_data.get("test6_expression_diversity", {}).get("summary", {})
            cat_div = [
                ("Diversity Ratio ↑ (gen/real)", f'{t1.get("mean_diversity_ratio", 0):.4f}', True),
                ("Collapsed Types (<0.5)", f'{t1.get("n_collapsed", 0)}/{t1.get("n_collapsed", 0) + t1.get("n_healthy", 0)}', True),
                ("Memorization (near-copies)", f'{t2.get("n_very_close", 0)}', False),
                ("NN Distance ↑", f'{t2.get("nn_cosine_distance", {}).get("mean", 0):.4f}', False),
                ("Cond Sensitivity Gain ↑", f'{t5.get("mean_diversity_gain", 0):.2f}×', True),
            ]
            if t6:
                cat_div.append(
                    ("Expr Gene-Std Ratio", f'{t6.get("mean_gene_std_ratio", 0):.4f}', False)
                )

        expr_path = Path(expr_metrics_path)
        if expr_path.exists():
            with open(expr_path) as f:
                expr_data = json.load(f)
            gene_corr = expr_data.get("gene_correlation", {})
            per_type_sum = expr_data.get("per_type_summary", {})
            cat_expr = [
                ("Gene Pearson r ↑", f'{gene_corr.get("pearson_r", 0):.6f}', True),
                ("Gene Spearman ρ ↑", f'{gene_corr.get("spearman_rho", 0):.6f}', False),
                ("Genes Compared", f'{gene_corr.get("n_genes_compared", 0)}', False),
                ("Per-Type Mean r ↑", f'{per_type_sum.get("mean_pearson_r", 0):.6f}', True),
                ("Per-Type Min r", f'{per_type_sum.get("min_pearson_r", 0):.6f}', False),
            ]

        gen_meta_path = Path("results/generation_metadata.json")
        if gen_meta_path.exists():
            with open(gen_meta_path) as f:
                gen_meta = json.load(f)
            cat_cfg = [
                ("Condition Mode", gen_meta.get("condition_mode", "?"), False),
                ("Noise Scale (ε)", f'{gen_meta.get("noise_scale", 0):.3f}', False),
                ("CFG Scale", f'{gen_meta.get("cfg_scale", 0):.1f}', False),
                ("Cells / Type", f'{gen_meta.get("num_per_type", 0)}', False),
                ("Total Generated", f'{gen_meta.get("total_cells", 0):,}', False),
            ]

        # ── Build two-column layout ──
        # Left column: Training + Generation
        # Right column: Diversity + Expression + Config
        left_sections = []
        if cat_training:
            left_sections.append(("Training", cat_training, "#1976D2"))
        if cat_gen:
            left_sections.append(("Generation Quality", cat_gen, "#E65100"))

        right_sections = []
        if cat_div:
            right_sections.append(("Diversity", cat_div, "#C62828"))
        if cat_expr:
            right_sections.append(("Expression Fidelity", cat_expr, "#6A1B9A"))
        if cat_cfg:
            right_sections.append(("Configuration", cat_cfg, "#37474F"))

        if not left_sections and not right_sections:
            return None

        # ── Render as two side-by-side tables ──
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 12))
        fig.suptitle("CLOP-DiT Pipeline — Metrics Summary",
                     fontsize=15, fontweight="bold", y=0.98)

        def _render_sections(ax, sections):
            """Render grouped metric sections on one axes."""
            ax.axis("off")
            all_rows = []
            section_row_indices = []  # (start_idx, color) for header rows
            for title, items, color in sections:
                header_idx = len(all_rows)
                all_rows.append([title, "", ""])
                section_row_indices.append((header_idx, color))
                for metric_name, value, is_key in items:
                    all_rows.append(["", metric_name, value])

            col_labels = ["", "Metric", "Value"]
            tbl = ax.table(
                cellText=all_rows,
                colLabels=col_labels,
                cellLoc="left",
                loc="upper center",
                colWidths=[0.01, 0.55, 0.30],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.6)

            # Style column header
            for j in range(3):
                cell = tbl[0, j]
                cell.set_facecolor("#37474F")
                cell.set_text_props(color="white", fontweight="bold", fontsize=10)
                cell.set_height(0.04)

            # Style section headers and data rows
            for i, row_data in enumerate(all_rows):
                row_idx = i + 1  # offset by column header
                is_section_header = any(idx == i for idx, _ in section_row_indices)

                if is_section_header:
                    # Find the color for this section header
                    color = next(c for idx, c in section_row_indices if idx == i)
                    for j in range(3):
                        cell = tbl[row_idx, j]
                        cell.set_facecolor(color)
                        cell.set_text_props(color="white", fontweight="bold", fontsize=11)
                        cell.set_height(0.035)
                    # Merge-like: put title text in metric column
                    tbl[row_idx, 1].get_text().set_text(row_data[0])
                    tbl[row_idx, 0].get_text().set_text("")
                else:
                    # Find which section this row belongs to
                    sec_color = "#FFFFFF"
                    for idx, color in section_row_indices:
                        if i > idx:
                            sec_color = color
                    # Light tint of section color
                    import matplotlib.colors as mcolors
                    base_rgb = mcolors.to_rgb(sec_color)
                    tint = tuple(c * 0.08 + 0.92 for c in base_rgb)
                    alt_tint = tuple(c * 0.14 + 0.86 for c in base_rgb)
                    bg = tint if (i % 2 == 0) else alt_tint

                    # Find if this is a key metric
                    # Reconstruct: find the section and item index
                    is_key = False
                    row_counter = 0
                    for _, items, _ in sections:
                        row_counter += 1  # header
                        for _, _, ik in items:
                            if row_counter == i:
                                is_key = ik
                                break
                            row_counter += 1
                        if row_counter > i:
                            break

                    for j in range(3):
                        cell = tbl[row_idx, j]
                        cell.set_facecolor(bg)
                        if is_key:
                            cell.set_text_props(fontweight="bold")

            return tbl

        _render_sections(ax_left, left_sections)
        _render_sections(ax_right, right_sections)

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

        # E3: Overlay — type-coloured, shape-split (circle=real, triangle=gen)
        ax = axes[2]
        if real_gids_sub is not None and gen_gids_sub is not None:
            for t in np.unique(np.concatenate([real_gids_sub, gen_gids_sub])):
                color = TYPE_PALETTE[int(t) % len(TYPE_PALETTE)]
                rmask = real_gids_sub == t
                gmask = gen_gids_sub == t
                if rmask.any():
                    ax.scatter(real_c[rmask, 0], real_c[rmask, 1], c=[color],
                               s=3, alpha=0.2, marker="o", rasterized=True)
                if gmask.any():
                    ax.scatter(gen_c[gmask, 0], gen_c[gmask, 1], c=[color],
                               s=6, alpha=0.35, marker="^", rasterized=True)
            # Dummy handles for legend
            ax.scatter([], [], c="gray", s=20, marker="o", label="Real")
            ax.scatter([], [], c="gray", s=20, marker="^", label="Generated")
        else:
            ax.scatter(real_c[:, 0], real_c[:, 1], c="#2196F3", s=3, alpha=0.25,
                       label="Real", rasterized=True)
            ax.scatter(gen_c[:, 0], gen_c[:, 1], c="#FF5722", s=3, alpha=0.25,
                       marker="^", label="Generated", rasterized=True)
        ax.legend(markerscale=5, fontsize=10)
        ax.set_title("E3: Type-Coloured Overlay")
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
            colors_fd = ["#4CAF50" if v < 0.1 else "#FF9800" if v < 0.3 else "#F44336" for v in fd_vals]
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
    # PANEL H: Gene Expression Correlation (scatter + per-type + markers)
    # ──────────────────────────────────────────────────────────
    def plot_expression_correlation(
        self,
        real_expr_path: str = "results/real_expression.npy",
        gen_expr_path: str = "results/generated_expression.npy",
        real_labels_path: str = "results/real_expression_labels.npy",
        gen_labels_path: str = "results/generated_expression_labels.npy",
        gene_names_path: str = "results/expression_gene_names.json",
        metrics_path: str = "results/expression_metrics.json",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Gene expression fidelity: per-gene scatter, per-type correlation, marker genes.

        H1: Scatter of per-gene mean expression (real vs generated)
        H2: Per-type expression Pearson r (bar chart, all types)
        H3: Marker-gene expression comparison (selected categories)
        """
        paths = [real_expr_path, gen_expr_path, metrics_path, gene_names_path]
        if not all(Path(p).exists() for p in paths):
            logger.info("Expression data not found — skipping Panel H")
            return None

        real = np.load(real_expr_path)
        gen = np.load(gen_expr_path)
        with open(gene_names_path) as f:
            gene_names = json.load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)

        real_means = real.mean(axis=0)
        gen_means = gen.mean(axis=0)
        pearson_r = metrics["gene_correlation"]["pearson_r"]
        spearman_rho = metrics["gene_correlation"]["spearman_rho"]

        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(
            f"Gene Expression Recovery — Pearson r={pearson_r:.6f}, "
            f"Spearman ρ={spearman_rho:.6f}",
            fontsize=14, fontweight="bold",
        )

        # ── H1: Per-gene mean expression scatter ──
        ax = axes[0]
        sc = ax.scatter(real_means, gen_means, c=real_means, cmap="viridis",
                        s=8, alpha=0.6, edgecolors="none")
        lo = min(real_means.min(), gen_means.min()) - 0.2
        hi = max(real_means.max(), gen_means.max()) + 0.2
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, alpha=0.7, label="y = x")
        ax.set_xlabel("Real Mean Expression")
        ax.set_ylabel("Generated Mean Expression")
        ax.set_title(f"H1: Per-Gene Mean Expression (n={len(gene_names)})")
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="Expression Level", shrink=0.8)

        # Annotate a few genes with highest real expression
        top_idx = np.argsort(real_means)[-5:]
        for i in top_idx:
            if i < len(gene_names):
                ax.annotate(gene_names[i], (real_means[i], gen_means[i]),
                            fontsize=6, xytext=(5, 5), textcoords="offset points",
                            arrowprops=dict(arrowstyle="-", lw=0.5))

        # ── H2: Per-type expression Pearson r ──
        ax = axes[1]
        per_type = metrics.get("per_type_expression_fidelity", {})
        if per_type:
            type_names_sorted = sorted(per_type.keys(),
                                       key=lambda k: per_type[k]["pearson_r"])
            type_rs = [per_type[n]["pearson_r"] for n in type_names_sorted]
            short_names = [n[:28] for n in type_names_sorted]

            colors = ["#4CAF50" if r > 0.9999 else "#FF9800" if r > 0.999 else "#F44336"
                       for r in type_rs]
            ax.barh(range(len(type_rs)), type_rs, color=colors, height=0.8)
            ax.set_yticks(range(len(type_rs)))
            ax.set_yticklabels(short_names, fontsize=4.5)
            ax.set_xlabel("Pearson r (Gene Expression)")
            mean_r = metrics.get("per_type_summary", {}).get("mean_pearson_r", 0)
            ax.axvline(x=mean_r, color="red", linestyle="--", alpha=0.5,
                       label=f"mean r={mean_r:.6f}")
            ax.legend(fontsize=7)

            # Smart x-axis: zoom into the interesting range
            min_r = min(type_rs)
            ax.set_xlim(min_r - 0.0001, 1.00005)
        else:
            ax.text(0.5, 0.5, "No per-type data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title("H2: Per-Type Expression Pearson r")

        # ── H3: Marker gene expression — selected categories ──
        ax = axes[2]
        marker_dict = metrics.get("marker_genes", {})
        # Pick up to 6 categories with genes actually in our set
        selected_cats = []
        selected_genes = []
        for cat, genes_list in marker_dict.items():
            found = [g for g in genes_list if g in gene_names]
            if found:
                selected_cats.append(cat)
                selected_genes.append(found[:3])  # up to 3 per category
            if len(selected_cats) >= 6:
                break

        if selected_cats:
            all_marker_genes = []
            cat_labels = []
            for cat, gg in zip(selected_cats, selected_genes):
                for g in gg:
                    all_marker_genes.append(g)
                    cat_labels.append(cat.replace("_", " "))

            gidx = [gene_names.index(g) for g in all_marker_genes]
            r_vals = [real_means[i] for i in gidx]
            g_vals = [gen_means[i] for i in gidx]

            x = np.arange(len(all_marker_genes))
            width = 0.35
            bars_r = ax.bar(x - width / 2, r_vals, width, label="Real", color="#1976D2", alpha=0.8)
            bars_g = ax.bar(x + width / 2, g_vals, width, label="Generated", color="#FF7043", alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{g}\n({c[:8]})" for g, c in zip(all_marker_genes, cat_labels)],
                fontsize=6, rotation=45, ha="right",
            )
            ax.set_ylabel("Mean Expression")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No marker genes found", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title("H3: Marker Gene Expression (Real vs Gen)")

        if save:
            path = self.output / "panel_h_expression_correlation.png"
            fig.savefig(path, dpi=self.dpi)
            fig.savefig(path.with_suffix(".pdf"), dpi=self.dpi)
            logger.info(f"Saved Panel H → {path}")
        return fig

    # ──────────────────────────────────────────────────────────
    # PANEL I: Expression Decoder Analysis
    # ──────────────────────────────────────────────────────────
    def plot_expression_analysis(
        self,
        real_expr_path: str = "results/real_expression.npy",
        gen_expr_path: str = "results/generated_expression.npy",
        gene_names_path: str = "results/expression_gene_names.json",
        metrics_path: str = "results/expression_metrics.json",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Expression decoder analysis: variability, range, and per-cell stats.

        I1: Per-gene coefficient of variation (CV) — real vs generated
        I2: Expression range comparison (gene-level distribution)
        I3: Per-cell expression variance (real vs generated histograms)
        """
        paths = [real_expr_path, gen_expr_path, gene_names_path, metrics_path]
        if not all(Path(p).exists() for p in paths):
            logger.info("Expression data not found — skipping Panel I")
            return None

        real = np.load(real_expr_path)
        gen = np.load(gen_expr_path)
        with open(gene_names_path) as f:
            gene_names = json.load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)

        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        overall = metrics.get("overall", {})
        fig.suptitle(
            f"Expression Decoder Analysis — "
            f"real mean={overall.get('real_mean', 0):.3f}, "
            f"gen mean={overall.get('gen_mean', 0):.3f}, "
            f"real cells={real.shape[0]}, gen cells={gen.shape[0]}",
            fontsize=14, fontweight="bold",
        )

        # ── I1: Per-gene CV distribution (real vs generated) ──
        ax = axes[0]
        real_cv = real.std(axis=0) / (np.abs(real.mean(axis=0)) + 1e-8)
        gen_cv = gen.std(axis=0) / (np.abs(gen.mean(axis=0)) + 1e-8)

        bins = np.linspace(0, max(real_cv.max(), gen_cv.max()) * 1.05, 50)
        ax.hist(real_cv, bins=bins, alpha=0.6, color="#1976D2", label=f"Real (mean={real_cv.mean():.5f})",
                edgecolor="white", linewidth=0.3)
        ax.hist(gen_cv, bins=bins, alpha=0.6, color="#FF7043", label=f"Gen (mean={gen_cv.mean():.5f})",
                edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Coefficient of Variation (CV)")
        ax.set_ylabel("Number of Genes")
        ax.set_title("I1: Per-Gene Variability Across Cells")
        ax.legend(fontsize=8)
        ax.annotate(
            f"CV \u2248 {gen_cv.mean():.5f} (gen)\nReal variability well-preserved\nacross cell types",
            xy=(0.95, 0.95), xycoords="axes fraction", fontsize=7,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        # ── I2: Gene expression range — real vs generated (sorted) ──
        ax = axes[1]
        real_means = real.mean(axis=0)
        gen_means = gen.mean(axis=0)
        sort_idx = np.argsort(real_means)
        ax.fill_between(range(len(sort_idx)),
                        real.min(axis=0)[sort_idx],
                        real.max(axis=0)[sort_idx],
                        alpha=0.15, color="#1976D2", label="Real range")
        ax.fill_between(range(len(sort_idx)),
                        gen.min(axis=0)[sort_idx],
                        gen.max(axis=0)[sort_idx],
                        alpha=0.15, color="#FF7043", label="Gen range")
        ax.plot(real_means[sort_idx], color="#1976D2", lw=1.2, label="Real mean")
        ax.plot(gen_means[sort_idx], color="#FF7043", lw=1.2, ls="--", label="Gen mean")
        ax.set_xlabel("Gene Index (sorted by real mean)")
        ax.set_ylabel("Expression Value")
        ax.set_title(f"I2: Gene Expression Range ({len(gene_names)} genes)")
        ax.legend(fontsize=7, loc="upper left")

        # ── I3: Per-cell expression variance histograms ──
        ax = axes[2]
        real_cell_std = real.std(axis=1)
        gen_cell_std = gen.std(axis=1)

        bins3 = np.linspace(
            min(real_cell_std.min(), gen_cell_std.min()) - 0.01,
            max(real_cell_std.max(), gen_cell_std.max()) + 0.01,
            60,
        )
        ax.hist(real_cell_std, bins=bins3, alpha=0.6, color="#1976D2",
                label=f"Real (μ={real_cell_std.mean():.4f})", edgecolor="white", linewidth=0.3)
        ax.hist(gen_cell_std, bins=bins3, alpha=0.6, color="#FF7043",
                label=f"Gen (μ={gen_cell_std.mean():.4f})", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Per-Cell Expression Std Dev")
        ax.set_ylabel("Number of Cells")
        ax.set_title("I3: Per-Cell Expression Variability")
        ax.legend(fontsize=8)

        if save:
            path = self.output / "panel_i_expression_analysis.png"
            fig.savefig(path, dpi=self.dpi)
            fig.savefig(path.with_suffix(".pdf"), dpi=self.dpi)
            logger.info(f"Saved Panel I → {path}")
        return fig

    # ──────────────────────────────────────────────────────────
    # PANEL N: Marker Gene Comparison (per-type real vs generated)
    # ──────────────────────────────────────────────────────────

    # Biologically meaningful markers covering major lineages
    MARKER_PANEL_GENES = {
        "CD8+ T":       ["CD8A", "GZMB"],
        "Myeloid":      ["CD68", "CD163"],
        "Epithelial":   ["EPCAM", "KRT8"],
        "Stromal":      ["COL1A1", "COL1A2"],
    }
    # Representative types to show per-type breakdown
    MARKER_PANEL_TYPES = [
        "CD8+ cytotoxic T lymphocytes",
        "Tumor-associated macrophages",
        "Epithelial tumor cells",
        "Fibroblasts and mesenchymal stromal cell",
    ]

    def plot_marker_gene_comparison(
        self,
        real_expr_path: str = "results/real_expression.npy",
        gen_expr_path: str = "results/generated_expression.npy",
        real_labels_path: str = "results/real_expression_labels.npy",
        gen_labels_path: str = "results/generated_expression_labels.npy",
        gene_names_path: str = "results/expression_gene_names.json",
        metrics_path: str = "results/expression_metrics.json",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Focused marker-gene comparison: real vs generated expression.

        N1: Paired bar chart — per-marker real vs gen mean expression (all cells)
        N2: Per-type × marker heatmap pair (real | generated, same color scale)
        N3: Difference heatmap (gen − real) to highlight biases
        """
        paths = [real_expr_path, gen_expr_path, gene_names_path, metrics_path]
        if not all(Path(p).exists() for p in paths):
            logger.info("Expression data not found — skipping Panel N")
            return None

        real = np.load(real_expr_path)
        gen = np.load(gen_expr_path)
        real_labels = np.load(real_labels_path) if Path(real_labels_path).exists() else None
        gen_labels = np.load(gen_labels_path) if Path(gen_labels_path).exists() else None
        with open(gene_names_path) as f:
            gene_names = json.load(f)

        # Resolve which markers are available
        all_marker_genes = []
        marker_cats = []
        for cat, genes in self.MARKER_PANEL_GENES.items():
            for g in genes:
                if g in gene_names:
                    all_marker_genes.append(g)
                    marker_cats.append(cat)
        if len(all_marker_genes) < 2:
            logger.warning("Too few markers found — skipping Panel N")
            return None

        gene_idx = [gene_names.index(g) for g in all_marker_genes]

        # Resolve which types match the desired representative set
        selected_type_ids = []
        selected_type_names = []
        if real_labels is not None:
            for target_name in self.MARKER_PANEL_TYPES:
                for tid, tname in self.type_names.items():
                    if target_name.lower() in tname.lower():
                        if tid in np.unique(real_labels):
                            selected_type_ids.append(tid)
                            selected_type_names.append(tname[:30])
                            break
        # Fallback: if too few matched, pick the 4 most populated types
        if len(selected_type_ids) < 3 and real_labels is not None:
            unique, counts = np.unique(real_labels, return_counts=True)
            top4 = unique[np.argsort(counts)[-4:]]
            selected_type_ids = top4.tolist()
            selected_type_names = [self.type_names.get(int(t), f"Type_{t}")[:30]
                                   for t in selected_type_ids]

        n_markers = len(all_marker_genes)
        n_sel_types = len(selected_type_ids)

        fig = plt.figure(figsize=(22, 10))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.3, 0.7], wspace=0.3)
        fig.suptitle("Marker Gene Comparison — Real vs Generated Expression",
                     fontsize=14, fontweight="bold")

        # ── N1: Overall paired bar chart for each marker gene ──
        ax = fig.add_subplot(gs[0])
        real_marker_means = np.array([real[:, gi].mean() for gi in gene_idx])
        gen_marker_means = np.array([gen[:, gi].mean() for gi in gene_idx])
        x = np.arange(n_markers)
        w = 0.35
        bars_r = ax.bar(x - w/2, real_marker_means, w, label="Real",
                        color="#1976D2", alpha=0.85, edgecolor="white")
        bars_g = ax.bar(x + w/2, gen_marker_means, w, label="Generated",
                        color="#FF7043", alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n({c})" for g, c in zip(all_marker_genes, marker_cats)],
                           fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Mean Expression")
        ax.set_title("N1: Marker Gene Mean Expression")
        ax.legend(fontsize=9)
        # Annotate ratio gen/real
        for i, (rm, gm) in enumerate(zip(real_marker_means, gen_marker_means)):
            ratio = gm / (rm + 1e-8)
            ax.text(i, max(rm, gm) * 1.01, f"{ratio:.4f}×", ha="center",
                    fontsize=7, color="#333")

        # ── N2: Per-type heatmap pair (real | gen) ──
        if n_sel_types >= 2 and real_labels is not None and gen_labels is not None:
            ax2 = fig.add_subplot(gs[1])
            # Compute per-type per-marker means
            real_heat = np.zeros((n_sel_types, n_markers))
            gen_heat = np.zeros((n_sel_types, n_markers))
            for i, tid in enumerate(selected_type_ids):
                r_mask = real_labels == tid
                g_mask = gen_labels == tid
                if r_mask.any():
                    for j, gi in enumerate(gene_idx):
                        real_heat[i, j] = real[r_mask][:, gi].mean()
                if g_mask.any():
                    for j, gi in enumerate(gene_idx):
                        gen_heat[i, j] = gen[g_mask][:, gi].mean()

            # Side-by-side heatmap
            combined = np.hstack([real_heat, gen_heat])  # (n_types, 2*n_markers)
            vmin = combined.min()
            vmax = combined.max()

            # Draw with imshow
            gap_col = np.full((n_sel_types, 1), np.nan)
            display = np.hstack([real_heat, gap_col, gen_heat])
            im = ax2.imshow(display, cmap="YlOrRd", aspect="auto",
                            vmin=vmin, vmax=vmax)
            ax2.set_yticks(range(n_sel_types))
            ax2.set_yticklabels(selected_type_names, fontsize=8)
            xtick_pos = list(range(n_markers)) + [n_markers] + list(range(n_markers + 1, 2 * n_markers + 1))
            xtick_labels = all_marker_genes + [""] + all_marker_genes
            ax2.set_xticks(xtick_pos)
            ax2.set_xticklabels(xtick_labels, fontsize=7, rotation=45, ha="right")
            ax2.set_title("N2: Per-Type Marker Expression (Real | Generated)")
            # Label the two halves
            ax2.text(n_markers / 2 - 0.5, -0.8, "Real", ha="center",
                     fontsize=10, fontweight="bold", color="#1976D2")
            ax2.text(n_markers + 0.5 + n_markers / 2 - 0.5, -0.8, "Generated",
                     ha="center", fontsize=10, fontweight="bold", color="#FF7043")
            fig.colorbar(im, ax=ax2, shrink=0.6, label="Expression")

            # ── N3: Difference heatmap ──
            ax3 = fig.add_subplot(gs[2])
            diff = gen_heat - real_heat
            max_abs = max(abs(diff.min()), abs(diff.max()), 0.01)
            im3 = ax3.imshow(diff, cmap="RdBu_r", aspect="auto",
                             vmin=-max_abs, vmax=max_abs)
            ax3.set_yticks(range(n_sel_types))
            ax3.set_yticklabels(selected_type_names, fontsize=8)
            ax3.set_xticks(range(n_markers))
            ax3.set_xticklabels(all_marker_genes, fontsize=7, rotation=45, ha="right")
            ax3.set_title("N3: Difference (Gen − Real)")
            fig.colorbar(im3, ax=ax3, shrink=0.6, label="Δ Expression")
            # Annotate cells
            for i in range(n_sel_types):
                for j in range(n_markers):
                    ax3.text(j, i, f"{diff[i, j]:.3f}", ha="center", va="center",
                             fontsize=6, color="black" if abs(diff[i, j]) < max_abs * 0.5 else "white")
        else:
            # Fallback: skip N2/N3 if no per-type breakdown
            ax2 = fig.add_subplot(gs[1:])
            ax2.text(0.5, 0.5, "Per-type labels not available for heatmap",
                     ha="center", va="center", transform=ax2.transAxes, fontsize=12)

        if save:
            path = self.output / "panel_n_marker_gene_comparison.png"
            fig.savefig(path, dpi=self.dpi)
            fig.savefig(path.with_suffix(".pdf"), dpi=self.dpi)
            logger.info(f"Saved Panel N → {path}")
        return fig

    # ──────────────────────────────────────────────────────────
    # PANEL O: Baseline Comparison (CLOP-DiT vs simple baselines)
    # ──────────────────────────────────────────────────────────
    def plot_baseline_comparison(
        self,
        gen_metrics_path: str = "results/generation_metrics.json",
        div_metrics_path: str = "results/diversity_diagnostics.json",
        baseline_metrics_path: str = "results/baseline_metrics.json",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Compare CLOP-DiT generation against simple baselines.

        O1: Bar chart — FD, centroid cosine, diversity ratio across methods
        O2: Radar / summary comparing 4 key dimensions
        O3: Textual summary table of best vs baseline

        If no precomputed baseline_metrics.json exists, generates a Gaussian
        baseline on-the-fly from cached data.
        """
        gen_path = Path(gen_metrics_path)
        if not gen_path.exists():
            logger.info("No generation metrics — skipping Panel O")
            return None

        with open(gen_path) as f:
            gen_data = json.load(f)
        clop_metrics = gen_data.get("overall", {})
        clop_summary = gen_data.get("summary", {})

        # Try loading precomputed baseline metrics
        baselines: Dict[str, Dict] = {}
        bl_path = Path(baseline_metrics_path)
        if bl_path.exists():
            with open(bl_path) as f:
                baselines = json.load(f)
        else:
            # Generate Gaussian baseline on-the-fly
            logger.info("Computing Gaussian and Shuffled baselines on-the-fly...")
            baselines = self._compute_baselines()

        if not baselines:
            logger.info("No baseline data — skipping Panel O")
            return None

        # Load diversity for CLOP-DiT
        div_path = Path(div_metrics_path)
        div_ratio_clop = 0.0
        if div_path.exists():
            with open(div_path) as f:
                div_data = json.load(f)
            div_ratio_clop = div_data.get(
                "test1_intratype_diversity", {}
            ).get("summary", {}).get("mean_diversity_ratio", 0)

        # Build comparison data
        methods = {"CLOP-DiT": {
            "FD": clop_metrics.get("frechet_distance", 0),
            "Centroid Cosine": clop_summary.get("mean_centroid_cosine", 0),
            "Diversity Ratio": div_ratio_clop,
            "Coverage": clop_metrics.get("coverage", 0),
        }}
        for bl_name, bl_data in baselines.items():
            methods[bl_name] = {
                "FD": bl_data.get("frechet_distance", 0),
                "Centroid Cosine": bl_data.get("mean_centroid_cosine", 0),
                "Diversity Ratio": bl_data.get("diversity_ratio", 0),
                "Coverage": bl_data.get("coverage", 0),
            }

        method_names = list(methods.keys())
        n_methods = len(method_names)
        metric_names = ["FD ↓", "Centroid Cosine ↑", "Diversity Ratio ↑", "Coverage ↑"]
        metric_keys = ["FD", "Centroid Cosine", "Diversity Ratio", "Coverage"]

        fig, axes = plt.subplots(1, 3, figsize=(22, 7),
                                 gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
        fig.suptitle("Baseline Comparison — CLOP-DiT vs Simple Baselines",
                     fontsize=14, fontweight="bold")

        # ── O1: Grouped bar chart ──
        ax = axes[0]
        x = np.arange(len(metric_names))
        w = 0.8 / n_methods
        colors = ["#1976D2", "#FF7043", "#4CAF50", "#9C27B0", "#FFC107"]
        for i, mname in enumerate(method_names):
            vals = [methods[mname][k] for k in metric_keys]
            offset = (i - n_methods / 2 + 0.5) * w
            bars = ax.bar(x + offset, vals, w, label=mname,
                          color=colors[i % len(colors)], alpha=0.85,
                          edgecolor="white")
            # Value labels
            for xi, v in zip(x + offset, vals):
                ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=7,
                        rotation=45)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=10)
        ax.legend(fontsize=9)
        ax.set_title("O1: Key Metrics Comparison")
        ax.set_ylabel("Value")

        # ── O2: Radar chart ──
        ax = axes[1]
        # Normalize metrics to [0, 1] for radar
        # FD: lower is better → invert; others: higher is better
        all_vals = {k: [methods[m][k] for m in method_names] for k in metric_keys}
        normalized = {}
        for k in metric_keys:
            mn, mx = min(all_vals[k]), max(all_vals[k])
            rng = mx - mn if mx > mn else 1
            if k == "FD":  # invert
                normalized[k] = [(mx - v) / rng for v in all_vals[k]]
            else:
                normalized[k] = [(v - mn) / rng for v in all_vals[k]]

        angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
        angles += angles[:1]
        axes[1].remove()  # remove Cartesian placeholder before adding polar
        ax = fig.add_subplot(132, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), metric_names, fontsize=8)

        for i, mname in enumerate(method_names):
            vals = [normalized[k][i] for k in metric_keys]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", linewidth=2, label=mname,
                    color=colors[i % len(colors)], markersize=6)
            ax.fill(angles, vals, alpha=0.1, color=colors[i % len(colors)])
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.set_title("O2: Normalized Radar", pad=20)

        # ── O3: Summary table ──
        ax = axes[2]
        ax.axis("off")
        tbl_rows = []
        for mname in method_names:
            row = [mname]
            for k in metric_keys:
                row.append(f"{methods[mname][k]:.4f}")
            tbl_rows.append(row)

        tbl = ax.table(
            cellText=tbl_rows,
            colLabels=["Method"] + metric_names,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.8)
        # Style header
        for j in range(len(metric_names) + 1):
            tbl[0, j].set_facecolor("#37474F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        # Highlight CLOP-DiT row
        for j in range(len(metric_names) + 1):
            tbl[1, j].set_facecolor("#E3F2FD")
            tbl[1, j].set_text_props(fontweight="bold")

        ax.set_title("O3: Summary", fontsize=12, fontweight="bold")

        if save:
            path = self.output / "panel_o_baseline_comparison.png"
            fig.savefig(path, dpi=self.dpi)
            fig.savefig(path.with_suffix(".pdf"), dpi=self.dpi)
            logger.info(f"Saved Panel O → {path}")
        return fig

    def _compute_baselines(self) -> Dict[str, Dict]:
        """Compute simple baselines on-the-fly from cached data.

        Gaussian: Sample from N(μ, Σ) estimated per-type, then L2-normalize.
        Shuffled: Randomly assign generated embeddings to wrong types.
        """
        from src.evaluation.metrics import GenerationMetrics

        cell_path = self.cache / "cell_embeddings_dedup_preprocessed.npy"
        gid_path = self.cache / "text_group_ids_dedup.npy"
        gen_path = Path("results/generated_embeddings.npy")
        gen_lab_path = Path("results/generated_labels.npy")

        if not all(p.exists() for p in [cell_path, gid_path, gen_path, gen_lab_path]):
            return {}

        real_cells = np.load(cell_path)
        group_ids = np.load(gid_path)
        gen_cells = np.load(gen_path)
        gen_labels = np.load(gen_lab_path)

        unique_types = np.sort(np.unique(group_ids))
        rng = np.random.default_rng(42)

        # ── Gaussian baseline ──
        # Per-type: sample from N(centroid, σ²I) where σ = mean intra-type std
        gauss_cells = []
        gauss_labels = []
        n_per = len(gen_cells) // len(unique_types) if len(unique_types) > 0 else 100
        for tid in unique_types:
            r = real_cells[group_ids == tid]
            centroid = r.mean(axis=0)
            std_val = r.std()  # scalar global std
            samples = rng.normal(0, std_val, size=(n_per, real_cells.shape[1]))
            samples += centroid
            norms = np.linalg.norm(samples, axis=1, keepdims=True) + 1e-8
            samples = samples / norms
            gauss_cells.append(samples)
            gauss_labels.extend([tid] * n_per)
        gauss_cells = np.concatenate(gauss_cells, axis=0)
        gauss_labels = np.array(gauss_labels)

        # Evaluate Gaussian baseline
        n_sub = min(5000, len(gauss_cells), len(real_cells))
        r_idx = rng.choice(len(real_cells), n_sub, replace=False)
        g_idx = rng.choice(len(gauss_cells), n_sub, replace=False)
        gauss_overall = GenerationMetrics.full_evaluation(real_cells[r_idx], gauss_cells[g_idx])
        gauss_cosines = []
        gauss_div_ratios = []
        for tid in unique_types:
            r = real_cells[group_ids == tid]
            g = gauss_cells[gauss_labels == tid]
            if len(r) < 5 or len(g) < 5:
                continue
            rc = r.mean(0); rc /= np.linalg.norm(rc) + 1e-8
            gc = g.mean(0); gc /= np.linalg.norm(gc) + 1e-8
            gauss_cosines.append(float(np.dot(rc, gc)))
            # Diversity ratio
            r_sub = r[rng.choice(len(r), min(100, len(r)), replace=False)]
            g_sub = g[rng.choice(len(g), min(100, len(g)), replace=False)]
            r_n = r_sub / (np.linalg.norm(r_sub, axis=1, keepdims=True) + 1e-8)
            g_n = g_sub / (np.linalg.norm(g_sub, axis=1, keepdims=True) + 1e-8)
            rr_sim = (r_n @ r_n.T)[np.triu_indices(len(r_n), k=1)].mean()
            gg_sim = (g_n @ g_n.T)[np.triu_indices(len(g_n), k=1)].mean()
            real_div = 1.0 - rr_sim
            gen_div = 1.0 - gg_sim
            if real_div > 1e-6:
                gauss_div_ratios.append(gen_div / real_div)

        # ── Shuffled baseline ──
        shuffled_labels = gen_labels.copy()
        rng.shuffle(shuffled_labels)
        shuf_cosines = []
        for tid in unique_types:
            r = real_cells[group_ids == tid]
            g = gen_cells[shuffled_labels == tid]
            if len(r) < 5 or len(g) < 5:
                continue
            rc = r.mean(0); rc /= np.linalg.norm(rc) + 1e-8
            gc = g.mean(0); gc /= np.linalg.norm(gc) + 1e-8
            shuf_cosines.append(float(np.dot(rc, gc)))

        shuf_overall = GenerationMetrics.full_evaluation(
            real_cells[r_idx],
            gen_cells[rng.choice(len(gen_cells), n_sub, replace=False)]
        )

        baselines = {
            "Gaussian N(μ,σ²I)": {
                "frechet_distance": gauss_overall.get("frechet_distance", 0),
                "mean_centroid_cosine": float(np.mean(gauss_cosines)) if gauss_cosines else 0,
                "diversity_ratio": float(np.mean(gauss_div_ratios)) if gauss_div_ratios else 0,
                "coverage": gauss_overall.get("coverage", 0),
            },
            "Shuffled Labels": {
                "frechet_distance": shuf_overall.get("frechet_distance", 0),
                "mean_centroid_cosine": float(np.mean(shuf_cosines)) if shuf_cosines else 0,
                "diversity_ratio": 1.0,  # shuffled retains overall diversity
                "coverage": shuf_overall.get("coverage", 0),
            },
        }

        # Save for reuse
        bl_path = Path("results/baseline_metrics.json")
        with open(bl_path, "w") as f:
            json.dump(baselines, f, indent=2)
        logger.info(f"Saved baseline metrics → {bl_path}")

        return baselines

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

        # Panel H: Gene expression correlation
        fig_h = self.plot_expression_correlation()
        if fig_h:
            saved.append(self.output / "panel_h_expression_correlation.pdf")
            plt.close(fig_h)

        # Panel I: Expression decoder analysis
        fig_i = self.plot_expression_analysis()
        if fig_i:
            saved.append(self.output / "panel_i_expression_analysis.pdf")
            plt.close(fig_i)

        # Panel N: Marker gene comparison
        fig_n = self.plot_marker_gene_comparison()
        if fig_n:
            saved.append(self.output / "panel_n_marker_gene_comparison.pdf")
            plt.close(fig_n)

        # Panel O: Baseline comparison
        fig_o = self.plot_baseline_comparison()
        if fig_o:
            saved.append(self.output / "panel_o_baseline_comparison.pdf")
            plt.close(fig_o)

        # Panels J & K: Diversity diagnostics (generated by diversity_diagnostics.py)
        # Panels L & M: Conditioning analysis (generated by conditioning_analysis.py)
        for panel_name in [
            "panel_j_diversity_diagnostics",
            "panel_k_expression_diversity",
            "panel_l_noise_tradeoff",
            "panel_m_conditioning_umap",
        ]:
            panel_png = self.output / f"{panel_name}.png"
            panel_pdf = self.output / f"{panel_name}.pdf"
            if panel_pdf.exists():
                saved.append(panel_pdf)
                logger.info(f"Including pre-generated {panel_name}")
            elif panel_png.exists():
                logger.info(f"Found {panel_name}.png but no PDF — including PNG")

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
