"""
results_visualizer.py — Publication-quality visualization for CLOP-DiT pipeline.

Generates multi-panel PDF/PNG reports proving training success:
  Panel A: CLOP training dynamics (loss, temperature, prototype accuracy, embedding quality)
  Panel B: CLOP embedding space (UMAP of 69-type prototypes + cell embeddings)
  Panel C: DiT training dynamics (flow-matching loss, cosine similarity, LR schedule)
  Panel D: Metrics summary dashboard (grouped bars + radar + gauges)
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
  Panel O: Baseline comparison (CLOP-DiT vs Gaussian/Shuffled baselines, radar + improvement strip)
  Panel P: Clustering alignment (UMAP overlay + kNN mixing score + ARI/NMI gauges)
  Panel Q: Classifier alignment (confusion matrix + per-type accuracy + discriminator ROC)
  Panel R: DE concordance (logFC scatter + concordance heatmap + per-contrast bars)
  Panel S: Model benchmarking (metrics heatmap + composite score + CI comparison)

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

# Shared style infrastructure (centralised in style.py)
from .style import (
    VIS_STYLE as STYLE,
    TYPE_PALETTE,
    COLORS as _COLORS,
    apply_style,
    style_axes,
    save_panel,
    quality_color,
    _build_type_palette,
    GRIDSPEC_TIGHT,
    set_dense_tick_labels,
)

logger = logging.getLogger(__name__)


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

        apply_style()

    # ──────────────────────────────────────────────────────────
    # Style helper — applied automatically to every figure before save
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _polish_figure(fig: plt.Figure) -> None:
        """Walk all axes and apply ``style_axes()`` for consistent spines/grids.

        Automatically detects axis kind from projection (polar, etc.) and the
        presence of images (heatmaps) to choose the right style.
        """
        for ax in fig.get_axes():
            # Detect kind
            if hasattr(ax, "name") and ax.name == "polar":
                style_axes(ax, kind="polar")
            elif ax.images:  # has imshow/heatmap content
                style_axes(ax, kind="heatmap")
            else:
                style_axes(ax, kind="default")

    def _save_panel(self, fig: plt.Figure, basename: str) -> Path:
        """Polish figure, save PNG + PDF, return PNG path."""
        self._polish_figure(fig)
        path = self.output / f"{basename}.png"
        fig.savefig(path, dpi=self.dpi)
        fig.savefig(path.with_suffix(".pdf"), dpi=self.dpi)
        logger.info(f"Saved {basename} → {path}")
        return path

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

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, **GRIDSPEC_TIGHT)
        fig.suptitle("CLOP Contrastive Pre-training (v9.3)", fontsize=14, fontweight="bold")

        # A1: Loss curves
        ax = fig.add_subplot(gs[0, 0])
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
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(epochs, h["temperature"], color="#4CAF50", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Temperature (τ)")
        ax.set_title("A2: Temperature Stability")
        ax.axhline(y=14.0, color="gray", linestyle=":", alpha=0.5, label="τ=14.0 (fixed)")
        ax.set_ylim(13.5, 14.5)
        ax.legend()

        # A3: Prototype accuracy
        ax = fig.add_subplot(gs[1, 0])
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
        ax = fig.add_subplot(gs[1, 1])
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
            self._save_panel(fig, "panel_a_clop_training")
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
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 1.2], **GRIDSPEC_TIGHT)
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
            self._save_panel(fig, "panel_b_clop_embedding_umap")
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

        fig = plt.figure(figsize=(16, 4.5))
        gs_c = fig.add_gridspec(1, 3, **GRIDSPEC_TIGHT)
        fig.suptitle("DiT Flow-Matching Training", fontsize=14, fontweight="bold")

        # C1: Loss
        ax = fig.add_subplot(gs_c[0])
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
        ax = fig.add_subplot(gs_c[1])
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
        ax = fig.add_subplot(gs_c[2])
        ax.plot(epochs, h["lr"], color="#9C27B0", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("C3: LR Schedule (Cosine Decay)")
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(-4, -4))

        if save:
            self._save_panel(fig, "panel_c_dit_training")
        return fig

    # ──────────────────────────────────────────────────────────
    # PANEL D: Metrics Summary (grouped, compact, publication-ready)
    # ──────────────────────────────────────────────────────────
    def plot_metrics_summary(self, gen_metrics_path: str = "results/generation_metrics.json",
                             expr_metrics_path: str = "results/expression_metrics.json",
                             div_metrics_path: str = "results/diversity_diagnostics.json",
                             save: bool = True) -> Optional[plt.Figure]:
        """Visual metrics dashboard — replaces table with bar charts + radar.

        D1: Training convergence (horizontal bars for final key metrics)
        D2: Generation quality radar (FD, coverage, diversity, centroid cos, expr r)
        D3: Diversity gauges (diversity ratio, collapsed types, cond gain)
        D4: Configuration + expression summary (compact annotated bars)
        """
        import matplotlib.colors as mcolors
        from matplotlib.patches import FancyBboxPatch

        # ── Collect all data ──
        train_metrics = {}
        if self.clop_hist:
            h = self.clop_hist
            train_metrics["CLOP Val Loss"] = h["val_loss"][-1]
            train_metrics["Proto Accuracy"] = h["val_proto_acc"][-1]
            train_metrics["Top-5 Accuracy"] = h["val_proto_top5"][-1]
            train_metrics["Text-Cell Align"] = h["val_text_cell_align"][-1]
        if self.dit_hist:
            h = self.dit_hist
            train_metrics["DiT Val Loss"] = h["val_loss"][-1]
            train_metrics["DiT Val Cosine"] = h["val_cosine_sim"][-1]

        gen_metrics = {}
        gen_path = Path(gen_metrics_path)
        if gen_path.exists():
            with open(gen_path) as f:
                gen_data = json.load(f)
            overall = gen_data.get("overall", {})
            summary = gen_data.get("summary", {})
            gen_metrics = {
                "FD": overall.get("frechet_distance", 0),
                "MMD": overall.get("mmd_rbf", 0),
                "Coverage": overall.get("coverage", 0),
                "Density": overall.get("density", 0),
                "Centroid Cos": summary.get("mean_centroid_cosine", 0),
            }

        div_metrics = {}
        div_path = Path(div_metrics_path)
        if div_path.exists():
            with open(div_path) as f:
                div_data = json.load(f)
            t1 = div_data.get("test1_intratype_diversity", {}).get("summary", {})
            t2 = div_data.get("test2_memorization", {})
            t5 = div_data.get("test5_condition_sensitivity", {}).get("summary", {})
            div_metrics = {
                "Diversity Ratio": t1.get("mean_diversity_ratio", 0),
                "Collapsed": t1.get("n_collapsed", 0),
                "Total Types": t1.get("n_collapsed", 0) + t1.get("n_healthy", 0),
                "NN Distance": t2.get("nn_cosine_distance", {}).get("mean", 0),
                "Near-copies": t2.get("n_very_close", 0),
                "Cond Gain": t5.get("mean_diversity_gain", 0),
            }

        expr_metrics = {}
        expr_path = Path(expr_metrics_path)
        if expr_path.exists():
            with open(expr_path) as f:
                expr_data = json.load(f)
            gene_corr = expr_data.get("gene_correlation", {})
            per_type_sum = expr_data.get("per_type_summary", {})
            expr_metrics = {
                "Gene Pearson r": gene_corr.get("pearson_r", 0),
                "Gene Spearman": gene_corr.get("spearman_rho", 0),
                "Per-Type r (mean)": per_type_sum.get("mean_pearson_r", 0),
                "Per-Type r (min)": per_type_sum.get("min_pearson_r", 0),
                "Genes": gene_corr.get("n_genes_compared", 0),
            }

        cfg_meta = {}
        gen_meta_path = Path("results/generation_metadata.json")
        if gen_meta_path.exists():
            with open(gen_meta_path) as f:
                cfg_meta = json.load(f)

        if not train_metrics and not gen_metrics:
            return None

        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(2, 2, **GRIDSPEC_TIGHT)
        fig.suptitle("CLOP-DiT Pipeline — Metrics Dashboard",
                     fontsize=16, fontweight="bold", y=0.98)

        # ── D1: Training convergence bars ──
        ax1 = fig.add_subplot(gs[0, 0])
        if train_metrics:
            names = list(train_metrics.keys())
            vals = list(train_metrics.values())
            display_vals = []
            for n, v in zip(names, vals):
                if "Loss" in n:
                    display_vals.append(v)
                else:
                    display_vals.append(v * 100 if v <= 1.0 else v)
            colors_d1 = []
            for n, v in zip(names, vals):
                if "Loss" in n:
                    colors_d1.append("#F44336" if v > 1.0 else "#FF9800" if v > 0.1 else "#4CAF50")
                else:
                    colors_d1.append("#4CAF50" if v > 0.8 else "#FF9800" if v > 0.5 else "#F44336")

            y_pos = np.arange(len(names))
            bars = ax1.barh(y_pos, display_vals, color=colors_d1, height=0.6,
                            edgecolor="white", linewidth=0.8)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(names, fontsize=10)
            for i, (bar, dv, n) in enumerate(zip(bars, display_vals, names)):
                unit = "" if "Loss" in n else "%"
                fmt = f"{dv:.4f}" if "Loss" in n else f"{dv:.1f}{unit}"
                ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                         fmt, va="center", fontsize=9, fontweight="bold")
            ax1.set_xlabel("Value (accuracy shown as %)")
            ax1.set_title("D1: Training Convergence", fontsize=13, fontweight="bold")
            ax1.invert_yaxis()

        # ── D2: Generation quality radar ──
        ax2_placeholder = fig.add_subplot(gs[0, 1])
        if gen_metrics or expr_metrics:
            ax2_placeholder.remove()
            ax2 = fig.add_subplot(gs[0, 1], polar=True)

            radar_labels = []
            radar_vals = []
            if gen_metrics:
                fd_score = max(0, 1.0 - gen_metrics.get("FD", 1.0))
                radar_labels.append("FD (inverted)")
                radar_vals.append(fd_score)
                radar_labels.append("Coverage")
                radar_vals.append(gen_metrics.get("Coverage", 0))
                radar_labels.append("Centroid Cos")
                radar_vals.append(gen_metrics.get("Centroid Cos", 0))
            if div_metrics:
                radar_labels.append("Diversity")
                radar_vals.append(div_metrics.get("Diversity Ratio", 0))
            if expr_metrics:
                radar_labels.append("Gene Corr")
                radar_vals.append(expr_metrics.get("Gene Pearson r", 0))

            if radar_vals:
                angles = np.linspace(0, 2 * np.pi, len(radar_vals), endpoint=False).tolist()
                radar_vals_plot = radar_vals + radar_vals[:1]
                angles_plot = angles + angles[:1]

                ax2.set_theta_offset(np.pi / 2)
                ax2.set_theta_direction(-1)
                ax2.set_thetagrids(np.degrees(angles), radar_labels, fontsize=9)
                ax2.plot(angles_plot, radar_vals_plot, "o-", linewidth=2.5,
                         color="#1976D2", markersize=8, zorder=5)
                ax2.fill(angles_plot, radar_vals_plot, alpha=0.15, color="#1976D2")
                ax2.set_ylim(0, 1.05)
                ax2.set_title("D2: Generation Quality Profile", pad=25,
                              fontsize=13, fontweight="bold")
                for angle, val, label in zip(angles, radar_vals, radar_labels):
                    ax2.annotate(f"{val:.3f}", xy=(angle, val),
                                 xytext=(5, 5), textcoords="offset points",
                                 fontsize=8, fontweight="bold", color="#1565C0")
        else:
            ax2_placeholder.text(0.5, 0.5, "No generation data", ha="center",
                                 va="center", transform=ax2_placeholder.transAxes)
            ax2_placeholder.set_title("D2: Generation Quality Profile")

        # ── D3: Diversity gauges ──
        ax3 = fig.add_subplot(gs[1, 0])
        if div_metrics:
            gauge_items = [
                ("Diversity\nRatio", div_metrics.get("Diversity Ratio", 0), 1.0,
                 "#4CAF50" if div_metrics.get("Diversity Ratio", 0) >= 0.8 else
                 "#FF9800" if div_metrics.get("Diversity Ratio", 0) >= 0.5 else "#F44336"),
                ("Cond\nGain", div_metrics.get("Cond Gain", 0), 3.0,
                 "#4CAF50" if div_metrics.get("Cond Gain", 0) >= 1.5 else
                 "#FF9800" if div_metrics.get("Cond Gain", 0) >= 1.0 else "#F44336"),
                ("NN\nDistance", div_metrics.get("NN Distance", 0), 1.0,
                 "#4CAF50" if div_metrics.get("NN Distance", 0) >= 0.3 else
                 "#FF9800" if div_metrics.get("NN Distance", 0) >= 0.1 else "#F44336"),
            ]
            x_pos = np.arange(len(gauge_items))
            for i, (label, val, max_val, color) in enumerate(gauge_items):
                bg_bar = ax3.barh(i, max_val, height=0.5, color="#E0E0E0",
                                  edgecolor="none", zorder=1)
                fg_bar = ax3.barh(i, min(val, max_val), height=0.5, color=color,
                                  edgecolor="white", linewidth=0.8, zorder=2)
                ax3.text(min(val, max_val) + 0.02, i, f"{val:.3f}",
                         va="center", fontsize=11, fontweight="bold", zorder=3)

            ax3.set_yticks(range(len(gauge_items)))
            ax3.set_yticklabels([g[0] for g in gauge_items], fontsize=10)
            ax3.invert_yaxis()

            collapsed = div_metrics.get("Collapsed", 0)
            total = div_metrics.get("Total Types", 69)
            copies = div_metrics.get("Near-copies", 0)
            ax3.text(0.95, 0.05,
                     f"Collapsed: {collapsed}/{total}  |  Near-copies: {copies}",
                     transform=ax3.transAxes, ha="right", va="bottom",
                     fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9))
            ax3.set_title("D3: Diversity Health", fontsize=13, fontweight="bold")
            ax3.set_xlabel("Score")
        else:
            ax3.text(0.5, 0.5, "No diversity data", ha="center", va="center",
                     transform=ax3.transAxes)
            ax3.set_title("D3: Diversity Health")

        # ── D4: Expression fidelity + config ──
        ax4 = fig.add_subplot(gs[1, 1])
        if expr_metrics:
            expr_items = [
                ("Gene Pearson r", expr_metrics.get("Gene Pearson r", 0)),
                ("Gene Spearman", expr_metrics.get("Gene Spearman", 0)),
                ("Per-Type r (mean)", expr_metrics.get("Per-Type r (mean)", 0)),
                ("Per-Type r (min)", expr_metrics.get("Per-Type r (min)", 0)),
            ]
            y_pos = np.arange(len(expr_items))
            vals = [v for _, v in expr_items]
            colors_d4 = ["#4CAF50" if v > 0.999 else "#FF9800" if v > 0.99 else "#F44336"
                         for v in vals]
            bars = ax4.barh(y_pos, vals, color=colors_d4, height=0.5,
                            edgecolor="white", linewidth=0.8)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([n for n, _ in expr_items], fontsize=10)
            for i, (bar, v) in enumerate(zip(bars, vals)):
                ax4.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height() / 2,
                         f"{v:.6f}", va="center", fontsize=9, fontweight="bold")
            ax4.invert_yaxis()
            min_val = min(vals) - 0.001
            ax4.set_xlim(min_val, 1.0001)
            ax4.set_title("D4: Expression Fidelity", fontsize=13, fontweight="bold")

            cfg_text_parts = []
            if cfg_meta:
                cfg_text_parts.append(f"Mode: {cfg_meta.get('condition_mode', '?')}")
                cfg_text_parts.append(f"CFG: {cfg_meta.get('cfg_scale', '?')}")
                eps = cfg_meta.get("noise_scale", 0)
                if eps:
                    cfg_text_parts.append(f"ε: {eps}")
                cfg_text_parts.append(f"Cells: {cfg_meta.get('total_cells', '?')}")
            n_genes = expr_metrics.get("Genes", 0)
            if n_genes:
                cfg_text_parts.append(f"Genes: {n_genes}")
            if cfg_text_parts:
                ax4.text(0.95, 0.05, "  |  ".join(cfg_text_parts),
                         transform=ax4.transAxes, ha="right", va="bottom",
                         fontsize=8, color="#555",
                         bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5", ec="#CCC"))
        else:
            ax4.text(0.5, 0.5, "No expression data", ha="center", va="center",
                     transform=ax4.transAxes)
            ax4.set_title("D4: Expression Fidelity")

        if save:
            self._save_panel(fig, "panel_d_metrics_summary")
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

        fig = plt.figure(figsize=(18, 5.5))
        gs_e = fig.add_gridspec(1, 3, **GRIDSPEC_TIGHT)
        fig.suptitle("Real vs Generated Cell Embeddings (DiT v1)", fontsize=14, fontweight="bold")

        # E1: Real — type-coloured
        ax = fig.add_subplot(gs_e[0])
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
        ax = fig.add_subplot(gs_e[1])
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
        ax = fig.add_subplot(gs_e[2])
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
            self._save_panel(fig, "panel_e_real_vs_generated")
        return fig

    # ──────────────────────────────────────────────────────────
    # PANEL F: Text–Cell Cosine Similarity Heatmap (69×69)
    # ──────────────────────────────────────────────────────────
    def plot_text_cell_heatmap(self, save: bool = True) -> Optional[plt.Figure]:
        """Enhanced 69x69 text-cell alignment heatmap with rich annotations.

        F1: Clustered heatmap with diagonal highlight and off-diagonal structure
        F2: Sorted per-type alignment bars with threshold bands
        F3: Distribution of diagonal vs off-diagonal similarities
        """
        import matplotlib.colors as mcolors
        from matplotlib.patches import Rectangle

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

        labels = [self.type_names.get(int(t), f"T{t}")[:25] for t in unique_types]
        diag = np.diag(sim_matrix)
        mean_diag = diag.mean()
        off_diag = sim_matrix[~np.eye(n_types, dtype=bool)]
        mean_off = off_diag.mean()
        std_off = off_diag.std()

        # Reorder by diagonal similarity for visual clarity
        sort_order = np.argsort(-diag)
        sim_sorted = sim_matrix[sort_order][:, sort_order]
        labels_sorted = [labels[i] for i in sort_order]
        diag_sorted = diag[sort_order]
        counts_sorted = type_counts[sort_order]

        fig = plt.figure(figsize=(20, 9))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 0.7, 0.5], **GRIDSPEC_TIGHT)
        fig.suptitle(
            f"Text–Cell Alignment (CLOP Space)  —  "
            f"Diagonal: {mean_diag:.3f} ± {diag.std():.3f}  |  "
            f"Off-diag: {mean_off:.3f} ± {std_off:.3f}  |  "
            f"Separation gap: {mean_diag - mean_off:.3f}",
            fontsize=14, fontweight="bold",
        )

        # ── F1: Clustered heatmap with annotations ──
        ax1 = fig.add_subplot(gs[0])
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_heat",
            ["#1a237e", "#283593", "#42a5f5", "#e3f2fd", "#fff9c4",
             "#ffcc80", "#ff7043", "#d32f2f", "#b71c1c"],
            N=256,
        )
        im = ax1.imshow(sim_sorted, cmap=cmap, vmin=-0.1, vmax=1.0, aspect="auto",
                        interpolation="nearest")
        ax1.set_xticks(range(n_types))
        ax1.set_yticks(range(n_types))
        ax1.set_xticklabels(labels_sorted, rotation=90, fontsize=5, ha="center")
        ax1.set_yticklabels(labels_sorted, fontsize=5, ha="right")
        set_dense_tick_labels(ax1, axis="both", max_labels=24, fontsize=5, rotation=90, ha="center")
        ax1.set_xlabel("Cell Type (cell centroids)", fontsize=10)
        ax1.set_ylabel("Cell Type (text prototypes)", fontsize=10)
        ax1.set_title("F1: Cosine Similarity (sorted by alignment strength)", fontsize=11)

        # Highlight diagonal
        for i in range(n_types):
            rect = Rectangle((i - 0.5, i - 0.5), 1, 1, linewidth=1.5,
                              edgecolor="white", facecolor="none", zorder=3)
            ax1.add_patch(rect)

        # Annotate top-5 off-diagonal confusions
        off_diag_matrix = sim_sorted.copy()
        np.fill_diagonal(off_diag_matrix, -1)
        for _ in range(min(5, n_types)):
            idx = np.unravel_index(off_diag_matrix.argmax(), off_diag_matrix.shape)
            val = off_diag_matrix[idx]
            if val < 0.3:
                break
            ax1.plot(idx[1], idx[0], "x", color="lime", markersize=6,
                     markeredgewidth=1.5, zorder=4)
            off_diag_matrix[idx] = -1

        cbar = fig.colorbar(im, ax=ax1, shrink=0.7, pad=0.02)
        cbar.set_label("Cosine Similarity", fontsize=9)
        cbar.ax.axhline(y=mean_diag, color="white", linewidth=2, linestyle="--")
        cbar.ax.axhline(y=mean_off, color="black", linewidth=1.5, linestyle=":")

        # ── F2: Per-type alignment bars with cell count annotations ──
        ax2 = fig.add_subplot(gs[1])
        sorted_idx_asc = np.argsort(diag)
        d_asc = diag[sorted_idx_asc]
        labels_asc = [labels[i] for i in sorted_idx_asc]
        counts_asc = type_counts[sorted_idx_asc]

        color_map = []
        for v in d_asc:
            if v >= 0.9:
                color_map.append("#2E7D32")
            elif v >= 0.7:
                color_map.append("#4CAF50")
            elif v >= 0.5:
                color_map.append("#FF9800")
            else:
                color_map.append("#D32F2F")
        bars = ax2.barh(range(n_types), d_asc, color=color_map, height=0.8,
                        edgecolor="white", linewidth=0.3)
        ax2.set_yticks(range(n_types))
        ax2.set_yticklabels(labels_asc, fontsize=5, ha="left")
        set_dense_tick_labels(ax2, axis="y", max_labels=24, fontsize=5, rotation=0)
        ax2.set_xlabel("Diagonal Cosine Similarity", fontsize=9)
        ax2.set_title("F2: Per-Type Alignment", fontsize=11)

        ax2.axvline(x=mean_diag, color="#D32F2F", linestyle="--", alpha=0.7, linewidth=1.5)
        ax2.axvspan(0.7, 1.05, alpha=0.05, color="green")
        ax2.axvspan(0.5, 0.7, alpha=0.05, color="orange")
        ax2.axvspan(0.0, 0.5, alpha=0.05, color="red")
        ax2.set_xlim(0, 1.05)

        # Cell count labels on bars
        for i, (bar, cnt) in enumerate(zip(bars, counts_asc)):
            ax2.text(0.02, bar.get_y() + bar.get_height() / 2,
                     f"n={cnt}", va="center", fontsize=3.5, color="white",
                     fontweight="bold", zorder=5)

        n_good = sum(1 for v in diag if v >= 0.7)
        n_ok = sum(1 for v in diag if 0.5 <= v < 0.7)
        n_weak = sum(1 for v in diag if v < 0.5)
        ax2.text(0.95, 0.95,
                 f"Strong (>0.7): {n_good}\nModerate: {n_ok}\nWeak (<0.5): {n_weak}",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9))

        # ── F3: Distribution comparison ──
        ax3 = fig.add_subplot(gs[2])
        ax3.hist(diag, bins=15, alpha=0.7, color="#1976D2", edgecolor="white",
                 label=f"Diagonal (μ={mean_diag:.3f})", density=True, orientation="horizontal")
        ax3.hist(off_diag, bins=30, alpha=0.5, color="#FF7043", edgecolor="white",
                 label=f"Off-diag (μ={mean_off:.3f})", density=True, orientation="horizontal")
        ax3.axhline(y=mean_diag, color="#1565C0", linestyle="--", linewidth=1.5)
        ax3.axhline(y=mean_off, color="#E64A19", linestyle=":", linewidth=1.5)
        ax3.set_ylabel("Cosine Similarity", fontsize=9)
        ax3.set_xlabel("Density", fontsize=9)
        ax3.set_title("F3: Distribution", fontsize=11)
        ax3.legend(fontsize=7, loc="lower right")
        ax3.set_ylim(-0.15, 1.05)

        if save:
            self._save_panel(fig, "panel_f_text_cell_heatmap")
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

        fig = plt.figure(figsize=(18, 7))
        gs_g = fig.add_gridspec(1, 3, **GRIDSPEC_TIGHT)
        summary = data.get("summary", {})
        overall = data.get("overall", {})
        fig.suptitle(
            f"Per-Type Generation Fidelity — "
            f"mean cos={summary.get('mean_centroid_cosine', 0):.3f}, "
            f"overall FD={overall.get('frechet_distance', 0):.3f}",
            fontsize=14, fontweight="bold",
        )

        # G1: Centroid cosine (sorted)
        ax = fig.add_subplot(gs_g[0])
        sorted_idx = np.argsort(cosines)
        sorted_cos = [cosines[i] for i in sorted_idx]
        sorted_names_cos = [short_names[i] for i in sorted_idx]
        colors = ["#4CAF50" if v > 0.9 else "#FF9800" if v > 0.7 else "#F44336" for v in sorted_cos]
        ax.barh(range(len(sorted_cos)), sorted_cos, color=colors, height=0.8)
        ax.set_yticks(range(len(sorted_cos)))
        ax.set_yticklabels(sorted_names_cos, fontsize=5.5, ha="left")
        set_dense_tick_labels(ax, axis="y", max_labels=24, fontsize=5.5, rotation=0)
        ax.set_xlabel("Centroid Cosine Similarity")
        ax.set_title("G1: Real↔Gen Centroid Cosine")
        ax.axvline(x=summary.get("mean_centroid_cosine", 0), color="red",
                    linestyle="--", alpha=0.5, label=f"mean={summary.get('mean_centroid_cosine', 0):.3f}")
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=8)

        # G2: Fréchet distance (sorted, larger = worse)
        ax = fig.add_subplot(gs_g[1])
        valid_fd = [(n, f) for n, f in zip(short_names, fds) if np.isfinite(f)]
        if valid_fd:
            fd_names, fd_vals = zip(*sorted(valid_fd, key=lambda x: x[1]))
            colors_fd = ["#4CAF50" if v < 0.1 else "#FF9800" if v < 0.3 else "#F44336" for v in fd_vals]
            ax.barh(range(len(fd_vals)), fd_vals, color=colors_fd, height=0.8)
            ax.set_yticks(range(len(fd_vals)))
            ax.set_yticklabels(fd_names, fontsize=5.5, ha="left")
            set_dense_tick_labels(ax, axis="y", max_labels=24, fontsize=5.5, rotation=0)
            ax.set_xlabel("Fréchet Distance (lower = better)")
            ax.set_title("G2: Per-Type Fréchet Distance")
        else:
            ax.text(0.5, 0.5, "No valid FD values", ha="center", va="center",
                    transform=ax.transAxes)

        # G3: Cosine vs dataset size
        ax = fig.add_subplot(gs_g[2])
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
            self._save_panel(fig, "panel_g_per_type_generation")
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
        """Enhanced gene expression fidelity with density-aware scatter and richer bars.

        H1: Density scatter of per-gene mean expression with residual coloring
        H2: Per-type Pearson r lollipop chart with tier shading
        H3: Marker gene grouped bars with error bars and fold-change annotation
        H4: Per-gene residual distribution (gen - real)
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
        residuals = gen_means - real_means
        pearson_r = metrics["gene_correlation"]["pearson_r"]
        spearman_rho = metrics["gene_correlation"]["spearman_rho"]

        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 2, **GRIDSPEC_TIGHT)
        fig.suptitle(
            f"Gene Expression Recovery — Pearson r = {pearson_r:.6f}  |  "
            f"Spearman ρ = {spearman_rho:.6f}  |  "
            f"n = {len(gene_names)} genes",
            fontsize=14, fontweight="bold",
        )

        # ── H1: Density scatter with residual coloring ──
        ax1 = fig.add_subplot(gs[0, 0])
        abs_res = np.abs(residuals)
        sc = ax1.scatter(real_means, gen_means, c=abs_res, cmap="magma_r",
                         s=12, alpha=0.7, edgecolors="none",
                         vmin=0, vmax=np.percentile(abs_res, 95))
        lo = min(real_means.min(), gen_means.min()) - 0.2
        hi = max(real_means.max(), gen_means.max()) + 0.2
        ax1.plot([lo, hi], [lo, hi], color="#E53935", linestyle="--", lw=1.5,
                 alpha=0.7, label="y = x", zorder=1)
        ax1.fill_between([lo, hi], [lo - 0.1, hi - 0.1], [lo + 0.1, hi + 0.1],
                         alpha=0.06, color="#4CAF50", zorder=0)
        ax1.set_xlabel("Real Mean Expression", fontsize=10)
        ax1.set_ylabel("Generated Mean Expression", fontsize=10)
        ax1.set_title("H1: Per-Gene Correlation (color = |residual|)", fontsize=11)
        ax1.legend(fontsize=8)
        cbar = plt.colorbar(sc, ax=ax1, shrink=0.8, pad=0.02)
        cbar.set_label("|Gen − Real|", fontsize=8)

        # Annotate outlier genes (top residuals)
        outlier_idx = np.argsort(abs_res)[-8:]
        for i in outlier_idx:
            if i < len(gene_names):
                ax1.annotate(gene_names[i], (real_means[i], gen_means[i]),
                             fontsize=5.5, xytext=(5, 5), textcoords="offset points",
                             arrowprops=dict(arrowstyle="->", lw=0.6, color="#555"),
                             fontweight="bold", color="#333")

        # ── H2: Per-type Pearson r lollipop chart ──
        ax2 = fig.add_subplot(gs[0, 1])
        per_type = metrics.get("per_type_expression_fidelity", {})
        if per_type:
            type_names_sorted = sorted(per_type.keys(),
                                       key=lambda k: per_type[k]["pearson_r"])
            type_rs = [per_type[n]["pearson_r"] for n in type_names_sorted]
            short_names = [n[:30] for n in type_names_sorted]

            mean_r = metrics.get("per_type_summary", {}).get("mean_pearson_r", 0)
            min_r = min(type_rs)
            y_pos = np.arange(len(type_rs))

            # Threshold shading
            ax2.axvspan(0.9999, 1.00005, alpha=0.08, color="#4CAF50", label=">0.9999")
            ax2.axvspan(0.999, 0.9999, alpha=0.08, color="#FF9800", label="0.999–0.9999")
            ax2.axvspan(min_r - 0.001, 0.999, alpha=0.08, color="#F44336", label="<0.999")

            colors_h2 = ["#2E7D32" if r > 0.9999 else "#FF9800" if r > 0.999 else "#D32F2F"
                         for r in type_rs]
            ax2.hlines(y_pos, min_r - 0.0005, type_rs, color="#DDD", linewidth=0.8, zorder=1)
            ax2.scatter(type_rs, y_pos, c=colors_h2, s=30, zorder=3, edgecolors="white",
                        linewidths=0.5)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(short_names, fontsize=5.5, ha="left")
            set_dense_tick_labels(ax2, axis="y", max_labels=24, fontsize=5.5, rotation=0)
            ax2.axvline(x=mean_r, color="#D32F2F", linestyle="--", alpha=0.6, linewidth=1.5,
                        label=f"mean={mean_r:.6f}")
            ax2.set_xlim(min_r - 0.0005, 1.00005)
            ax2.set_xlabel("Pearson r", fontsize=9)
            ax2.legend(fontsize=6, loc="lower right")
        else:
            ax2.text(0.5, 0.5, "No per-type data", ha="center", va="center",
                     transform=ax2.transAxes)
        ax2.set_title("H2: Per-Type Expression Fidelity", fontsize=11)

        # ── H3: Marker gene expression with error bars ──
        ax3 = fig.add_subplot(gs[1, 0])
        marker_dict = metrics.get("marker_genes", {})
        selected_cats = []
        selected_genes = []
        for cat, genes_list in marker_dict.items():
            found = [g for g in genes_list if g in gene_names]
            if found:
                selected_cats.append(cat)
                selected_genes.append(found[:3])
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
            r_means = np.array([real[:, i].mean() for i in gidx])
            g_means = np.array([gen[:, i].mean() for i in gidx])
            r_stds = np.array([real[:, i].std() for i in gidx])
            g_stds = np.array([gen[:, i].std() for i in gidx])

            x = np.arange(len(all_marker_genes))
            width = 0.35
            ax3.bar(x - width / 2, r_means, width, yerr=r_stds * 0.5,
                    label="Real", color="#1976D2", alpha=0.85, edgecolor="white",
                    capsize=3, error_kw=dict(lw=0.8))
            ax3.bar(x + width / 2, g_means, width, yerr=g_stds * 0.5,
                    label="Generated", color="#FF7043", alpha=0.85, edgecolor="white",
                    capsize=3, error_kw=dict(lw=0.8))

            # Fold-change annotations
            for i, (rm, gm) in enumerate(zip(r_means, g_means)):
                fc = gm / (rm + 1e-8)
                color = "#2E7D32" if 0.95 <= fc <= 1.05 else "#D32F2F"
                ax3.text(i, max(rm, gm) + max(r_stds[i], g_stds[i]) * 0.5 + 0.01,
                         f"{fc:.3f}×", ha="center", fontsize=6, color=color,
                         fontweight="bold")

            ax3.set_xticks(x)
            ax3.set_xticklabels(
                [f"{g}\n({c[:10]})" for g, c in zip(all_marker_genes, cat_labels)],
                fontsize=7, rotation=35, ha="right",
            )
            ax3.set_ylabel("Expression (mean ± 0.5×std)", fontsize=9)
            ax3.legend(fontsize=8, loc="upper right")
        else:
            ax3.text(0.5, 0.5, "No marker genes found", ha="center", va="center",
                     transform=ax3.transAxes)
        ax3.set_title("H3: Marker Gene Expression", fontsize=11)

        # ── H4: Residual distribution ──
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(residuals, bins=60, color="#1976D2", alpha=0.7, edgecolor="white",
                 density=True)
        ax4.axvline(x=0, color="#E53935", linestyle="--", linewidth=1.5, label="Zero")
        ax4.axvline(x=residuals.mean(), color="#FF9800", linestyle="-", linewidth=1.5,
                    label=f"Mean={residuals.mean():.4f}")
        ax4.set_xlabel("Residual (Gen − Real)", fontsize=10)
        ax4.set_ylabel("Density", fontsize=10)
        ax4.set_title("H4: Per-Gene Residual Distribution", fontsize=11)
        ax4.legend(fontsize=8)

        pct_within_01 = (np.abs(residuals) < 0.1).mean() * 100
        pct_within_001 = (np.abs(residuals) < 0.01).mean() * 100
        ax4.text(0.95, 0.95,
                 f"|Δ|<0.01: {pct_within_001:.0f}%\n|Δ|<0.10: {pct_within_01:.0f}%",
                 transform=ax4.transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9))

        if save:
            self._save_panel(fig, "panel_h_expression_correlation")
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
        """Enhanced expression decoder analysis with denser information.

        I1: CV scatter (real vs gen per-gene) with identity line and outlier genes
        I2: Expression range ribbon with percentile bands (10-90th, 25-75th)
        I3: Per-cell expression std as overlaid KDE-style histograms
        I4: Top variable genes heatmap (genes with highest CV difference)
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

        overall = metrics.get("overall", {})

        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 2, **GRIDSPEC_TIGHT)
        fig.suptitle(
            f"Expression Decoder Analysis — "
            f"{real.shape[0]} real cells, {gen.shape[0]} gen cells, "
            f"{len(gene_names)} genes  |  "
            f"μ_real={overall.get('real_mean', real.mean()):.3f}  "
            f"μ_gen={overall.get('gen_mean', gen.mean()):.3f}",
            fontsize=14, fontweight="bold",
        )

        # ── I1: CV scatter (real vs gen) with gene labels ──
        ax1 = fig.add_subplot(gs[0, 0])
        real_cv = real.std(axis=0) / (np.abs(real.mean(axis=0)) + 1e-8)
        gen_cv = gen.std(axis=0) / (np.abs(gen.mean(axis=0)) + 1e-8)
        cv_diff = np.abs(gen_cv - real_cv)

        sc = ax1.scatter(real_cv, gen_cv, c=cv_diff, cmap="YlOrRd", s=10,
                         alpha=0.7, edgecolors="none",
                         vmin=0, vmax=np.percentile(cv_diff, 95))
        lo = 0
        hi = max(real_cv.max(), gen_cv.max()) * 1.05
        ax1.plot([lo, hi], [lo, hi], color="#E53935", linestyle="--", lw=1.5,
                 alpha=0.6, label="y = x")
        ax1.set_xlabel("Real CV (std/|mean|)", fontsize=10)
        ax1.set_ylabel("Generated CV", fontsize=10)
        ax1.set_title("I1: Per-Gene Variability (CV)", fontsize=11)
        cbar = plt.colorbar(sc, ax=ax1, shrink=0.8, pad=0.02)
        cbar.set_label("|ΔCV|", fontsize=8)

        # Annotate top divergent genes
        top_cv_idx = np.argsort(cv_diff)[-6:]
        for i in top_cv_idx:
            if i < len(gene_names):
                ax1.annotate(gene_names[i], (real_cv[i], gen_cv[i]),
                             fontsize=5.5, xytext=(4, 4), textcoords="offset points",
                             arrowprops=dict(arrowstyle="->", lw=0.5, color="#555"),
                             fontweight="bold", color="#333")

        cv_corr = np.corrcoef(real_cv, gen_cv)[0, 1]
        ax1.legend(fontsize=8, title=f"CV corr = {cv_corr:.4f}", title_fontsize=8)

        # ── I2: Expression range with percentile bands ──
        ax2 = fig.add_subplot(gs[0, 1])
        real_means = real.mean(axis=0)
        gen_means = gen.mean(axis=0)
        sort_idx = np.argsort(real_means)

        # Percentile bands
        rp10, rp90 = np.percentile(real, [10, 90], axis=0)
        rp25, rp75 = np.percentile(real, [25, 75], axis=0)
        gp10, gp90 = np.percentile(gen, [10, 90], axis=0)
        gp25, gp75 = np.percentile(gen, [25, 75], axis=0)

        x_range = np.arange(len(sort_idx))
        ax2.fill_between(x_range, rp10[sort_idx], rp90[sort_idx],
                         alpha=0.08, color="#1976D2", label="Real 10–90%")
        ax2.fill_between(x_range, rp25[sort_idx], rp75[sort_idx],
                         alpha=0.15, color="#1976D2", label="Real 25–75%")
        ax2.fill_between(x_range, gp10[sort_idx], gp90[sort_idx],
                         alpha=0.08, color="#FF7043", label="Gen 10–90%")
        ax2.fill_between(x_range, gp25[sort_idx], gp75[sort_idx],
                         alpha=0.15, color="#FF7043", label="Gen 25–75%")
        ax2.plot(real_means[sort_idx], color="#1565C0", lw=1.2, label="Real mean", zorder=3)
        ax2.plot(gen_means[sort_idx], color="#E64A19", lw=1.2, ls="--",
                 label="Gen mean", zorder=3)
        ax2.set_xlabel("Gene index (sorted by real mean)", fontsize=9)
        ax2.set_ylabel("Expression", fontsize=9)
        ax2.set_title(f"I2: Expression Range ({len(gene_names)} genes)", fontsize=11)
        ax2.legend(fontsize=6, loc="upper left", ncol=2)

        # ── I3: Per-cell std as overlaid smooth histograms ──
        ax3 = fig.add_subplot(gs[1, 0])
        real_cell_std = real.std(axis=1)
        gen_cell_std = gen.std(axis=1)
        real_cell_mean = real.mean(axis=1)
        gen_cell_mean = gen.mean(axis=1)

        bins3 = np.linspace(
            min(real_cell_std.min(), gen_cell_std.min()) - 0.005,
            max(real_cell_std.max(), gen_cell_std.max()) + 0.005,
            80,
        )
        ax3.hist(real_cell_std, bins=bins3, alpha=0.5, color="#1976D2",
                 label=f"Real (μ={real_cell_std.mean():.4f}, σ={real_cell_std.std():.4f})",
                 edgecolor="white", linewidth=0.3, density=True)
        ax3.hist(gen_cell_std, bins=bins3, alpha=0.5, color="#FF7043",
                 label=f"Gen (μ={gen_cell_std.mean():.4f}, σ={gen_cell_std.std():.4f})",
                 edgecolor="white", linewidth=0.3, density=True)
        ax3.axvline(x=real_cell_std.mean(), color="#1565C0", linestyle="--", lw=1.5)
        ax3.axvline(x=gen_cell_std.mean(), color="#E64A19", linestyle="--", lw=1.5)
        ax3.set_xlabel("Per-Cell Expression Std Dev", fontsize=10)
        ax3.set_ylabel("Density", fontsize=10)
        ax3.set_title("I3: Per-Cell Variability Distribution", fontsize=11)
        ax3.legend(fontsize=7)

        std_ratio = gen_cell_std.mean() / (real_cell_std.mean() + 1e-8)
        ax3.text(0.95, 0.95,
                 f"Std ratio (gen/real): {std_ratio:.3f}\n"
                 f"Real cells: {real.shape[0]:,}\n"
                 f"Gen cells: {gen.shape[0]:,}",
                 transform=ax3.transAxes, ha="right", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9))

        # ── I4: Top variable genes ranked bar chart ──
        ax4 = fig.add_subplot(gs[1, 1])
        real_stds = real.std(axis=0)
        gen_stds = gen.std(axis=0)
        valid_mask = real_stds > 1e-4
        std_ratio_per_gene = np.ones(len(real_stds))
        std_ratio_per_gene[valid_mask] = gen_stds[valid_mask] / real_stds[valid_mask]

        n_show = min(30, len(gene_names))
        clipped_ratio = np.clip(std_ratio_per_gene, 0, 5.0)
        diff_score = np.abs(clipped_ratio - 1.0)
        top_diff_idx = np.argsort(diff_score)[-n_show:]
        top_diff_idx = top_diff_idx[np.argsort(clipped_ratio[top_diff_idx])]

        ratios_show = clipped_ratio[top_diff_idx]
        names_show = [gene_names[i] if i < len(gene_names) else f"G{i}"
                      for i in top_diff_idx]
        colors_i4 = ["#D32F2F" if r < 0.7 else "#FF9800" if r < 0.9 else
                      "#4CAF50" if r <= 1.1 else "#FF9800" if r <= 1.3 else "#D32F2F"
                      for r in ratios_show]
        ax4.barh(range(n_show), ratios_show, color=colors_i4, height=0.7,
                 edgecolor="white", linewidth=0.3)
        ax4.axvline(x=1.0, color="#333", linestyle="-", linewidth=1.5)
        ax4.axvspan(0.9, 1.1, alpha=0.08, color="green")
        ax4.set_yticks(range(n_show))
        ax4.set_yticklabels(names_show, fontsize=5.5, ha="left")
        set_dense_tick_labels(ax4, axis="y", max_labels=18, fontsize=5.5, rotation=0)
        ax4.set_xlabel("Std Ratio (Gen / Real, clipped at 5×)", fontsize=9)
        ax4.set_title("I4: Top Variable Genes (std gen/real)", fontsize=11)
        for i, r in enumerate(ratios_show):
            ax4.text(r + 0.02, i, f"{r:.2f}×", va="center", fontsize=5.5,
                     fontweight="bold" if abs(r - 1.0) > 0.3 else "normal")

        if save:
            self._save_panel(fig, "panel_i_expression_analysis")
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
        """Rich marker-gene comparison with violin plots and annotated heatmaps.

        N1: Violin + strip plot — per-marker expression distribution (real vs gen)
        N2: Dual heatmap with cell-value annotations and row-normalized coloring
        N3: Difference heatmap with statistical significance indicators
        N4: Fold-change waterfall for all markers
        """
        import matplotlib.colors as mcolors

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
        if len(selected_type_ids) < 3 and real_labels is not None:
            unique, counts = np.unique(real_labels, return_counts=True)
            top4 = unique[np.argsort(counts)[-4:]]
            selected_type_ids = top4.tolist()
            selected_type_names = [self.type_names.get(int(t), f"Type_{t}")[:30]
                                   for t in selected_type_ids]

        n_markers = len(all_marker_genes)
        n_sel_types = len(selected_type_ids)

        fig = plt.figure(figsize=(22, 10))
        gs = fig.add_gridspec(2, 2, **GRIDSPEC_TIGHT)
        fig.suptitle("Marker Gene Biological Validation — Real vs Generated Expression",
                     fontsize=14, fontweight="bold")

        # ── N1: Paired bars with error whiskers and category coloring ──
        ax1 = fig.add_subplot(gs[0, 0])
        real_marker_means = np.array([real[:, gi].mean() for gi in gene_idx])
        gen_marker_means = np.array([gen[:, gi].mean() for gi in gene_idx])
        real_marker_stds = np.array([real[:, gi].std() for gi in gene_idx])
        gen_marker_stds = np.array([gen[:, gi].std() for gi in gene_idx])

        cat_colors = {"CD8+ T": "#1565C0", "Myeloid": "#C62828",
                      "Epithelial": "#2E7D32", "Stromal": "#6A1B9A"}
        x = np.arange(n_markers)
        w = 0.35
        for i, (g, c) in enumerate(zip(all_marker_genes, marker_cats)):
            bc = cat_colors.get(c, "#666")
            ax1.bar(i - w / 2, real_marker_means[i], w, yerr=real_marker_stds[i] * 0.3,
                    color=bc, alpha=0.75, edgecolor="white", capsize=3,
                    error_kw=dict(lw=0.8))
            ax1.bar(i + w / 2, gen_marker_means[i], w, yerr=gen_marker_stds[i] * 0.3,
                    color=bc, alpha=0.4, edgecolor=bc, linewidth=1.5,
                    capsize=3, error_kw=dict(lw=0.8), hatch="///")
            fc = gen_marker_means[i] / (real_marker_means[i] + 1e-8)
            color_fc = "#2E7D32" if 0.95 <= fc <= 1.05 else "#D32F2F"
            ax1.text(i, max(real_marker_means[i], gen_marker_means[i]) +
                     max(real_marker_stds[i], gen_marker_stds[i]) * 0.3 + 0.005,
                     f"{fc:.3f}×", ha="center", fontsize=6.5, color=color_fc,
                     fontweight="bold")

        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{g}\n({c})" for g, c in zip(all_marker_genes, marker_cats)],
                            fontsize=7, rotation=30, ha="right")
        ax1.set_ylabel("Expression (mean ± 0.3×std)")
        ax1.set_title("N1: Marker Gene Expression by Lineage", fontsize=11)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#666", alpha=0.75, label="Real"),
                           Patch(facecolor="#666", alpha=0.4, hatch="///", label="Generated")]
        ax1.legend(handles=legend_elements, fontsize=8, loc="upper right")

        # ── N2 & N3: Heatmaps (if per-type labels) ──
        if n_sel_types >= 2 and real_labels is not None and gen_labels is not None:
            real_heat = np.zeros((n_sel_types, n_markers))
            gen_heat = np.zeros((n_sel_types, n_markers))
            real_heat_std = np.zeros((n_sel_types, n_markers))
            gen_heat_std = np.zeros((n_sel_types, n_markers))
            for i, tid in enumerate(selected_type_ids):
                r_mask = real_labels == tid
                g_mask = gen_labels == tid
                for j, gi in enumerate(gene_idx):
                    if r_mask.any():
                        real_heat[i, j] = real[r_mask][:, gi].mean()
                        real_heat_std[i, j] = real[r_mask][:, gi].std()
                    if g_mask.any():
                        gen_heat[i, j] = gen[g_mask][:, gi].mean()
                        gen_heat_std[i, j] = gen[g_mask][:, gi].std()

            # N2: Side-by-side annotated heatmaps
            ax2 = fig.add_subplot(gs[0, 1])
            combined = np.hstack([real_heat, gen_heat])
            vmin, vmax = combined.min(), combined.max()
            gap_col = np.full((n_sel_types, 1), np.nan)
            display = np.hstack([real_heat, gap_col, gen_heat])

            cmap_n2 = mcolors.LinearSegmentedColormap.from_list(
                "expr_heat", ["#fff3e0", "#ffcc80", "#ff9800", "#e65100", "#bf360c"], N=256)
            im = ax2.imshow(display, cmap=cmap_n2, aspect="auto", vmin=vmin, vmax=vmax)
            ax2.set_yticks(range(n_sel_types))
            ax2.set_yticklabels(selected_type_names, fontsize=8)
            xtick_pos = list(range(n_markers)) + [n_markers] + list(range(n_markers + 1, 2 * n_markers + 1))
            xtick_labels = all_marker_genes + ["|"] + all_marker_genes
            ax2.set_xticks(xtick_pos)
            ax2.set_xticklabels(xtick_labels, fontsize=7, rotation=45, ha="right")
            ax2.set_title("N2: Per-Type × Marker (Real | Gen)", fontsize=11)
            ax2.text(n_markers / 2 - 0.5, -0.8, "Real", ha="center",
                     fontsize=10, fontweight="bold", color="#1976D2")
            ax2.text(n_markers + 0.5 + n_markers / 2 - 0.5, -0.8, "Generated",
                     ha="center", fontsize=10, fontweight="bold", color="#FF7043")

            # Cell value annotations
            for i in range(n_sel_types):
                for j in range(n_markers):
                    ax2.text(j, i, f"{real_heat[i, j]:.2f}", ha="center", va="center",
                             fontsize=5.5, color="white" if real_heat[i, j] > vmax * 0.6 else "black")
                    ax2.text(j + n_markers + 1, i, f"{gen_heat[i, j]:.2f}", ha="center",
                             va="center", fontsize=5.5,
                             color="white" if gen_heat[i, j] > vmax * 0.6 else "black")
            fig.colorbar(im, ax=ax2, shrink=0.6, label="Expression", pad=0.02)

            # N3: Difference heatmap with significance
            ax3 = fig.add_subplot(gs[1, 0])
            diff = gen_heat - real_heat
            pct_diff = diff / (np.abs(real_heat) + 1e-8) * 100
            max_abs = max(abs(diff.min()), abs(diff.max()), 0.01)

            im3 = ax3.imshow(diff, cmap="RdBu_r", aspect="auto",
                             vmin=-max_abs, vmax=max_abs)
            ax3.set_yticks(range(n_sel_types))
            ax3.set_yticklabels(selected_type_names, fontsize=8)
            ax3.set_xticks(range(n_markers))
            ax3.set_xticklabels(all_marker_genes, fontsize=7, rotation=45, ha="right")
            ax3.set_title("N3: Δ Expression (Gen − Real) with % change", fontsize=11)
            fig.colorbar(im3, ax=ax3, shrink=0.6, label="Δ Expression", pad=0.02)
            for i in range(n_sel_types):
                for j in range(n_markers):
                    txt_color = "white" if abs(diff[i, j]) > max_abs * 0.5 else "black"
                    ax3.text(j, i, f"{diff[i, j]:+.3f}\n({pct_diff[i, j]:+.1f}%)",
                             ha="center", va="center", fontsize=5, color=txt_color)

            # N4: Fold-change waterfall
            ax4 = fig.add_subplot(gs[1, 1])
            fc_all = gen_marker_means / (real_marker_means + 1e-8)
            sort_fc = np.argsort(fc_all)
            fc_sorted = fc_all[sort_fc]
            names_sorted = [all_marker_genes[i] for i in sort_fc]
            cats_sorted = [marker_cats[i] for i in sort_fc]
            fc_colors = [cat_colors.get(c, "#666") for c in cats_sorted]

            bars = ax4.barh(range(n_markers), fc_sorted - 1.0, left=1.0,
                            color=fc_colors, height=0.6, edgecolor="white")
            ax4.axvline(x=1.0, color="#333", linewidth=1.5, linestyle="-")
            ax4.axvspan(0.95, 1.05, alpha=0.1, color="green")
            ax4.set_yticks(range(n_markers))
            ax4.set_yticklabels([f"{n} ({c})" for n, c in zip(names_sorted, cats_sorted)],
                                fontsize=7)
            ax4.set_xlabel("Fold Change (Gen / Real)", fontsize=9)
            ax4.set_title("N4: Marker Fold Change", fontsize=11)
            for i, fc in enumerate(fc_sorted):
                ax4.text(fc + 0.002 if fc >= 1.0 else fc - 0.002, i,
                         f"{fc:.4f}×", va="center", fontsize=7,
                         ha="left" if fc >= 1.0 else "right",
                         fontweight="bold" if abs(fc - 1.0) > 0.05 else "normal")
        else:
            ax_fallback = fig.add_subplot(gs[0, 1])
            ax_fallback.text(0.5, 0.5, "Per-type labels not available",
                             ha="center", va="center", transform=ax_fallback.transAxes)

        if save:
            self._save_panel(fig, "panel_n_marker_gene_comparison")
        return fig

    # ──────────────────────────────────────────────────────────
    # PANEL O: Baseline Comparison (delegates to baseline_panels module)
    # ──────────────────────────────────────────────────────────
    def plot_baseline_comparison(
        self,
        gen_metrics_path: str = "results/generation_metrics.json",
        div_metrics_path: str = "results/diversity_diagnostics.json",
        baseline_metrics_path: str = "results/baseline_metrics.json",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Panel O: CLOP-DiT vs baselines — graphical O3 (no table)."""
        from .baseline_panels import plot_baseline_comparison as _plot_o
        return _plot_o(
            gen_metrics_path=gen_metrics_path,
            div_metrics_path=div_metrics_path,
            baseline_metrics_path=baseline_metrics_path,
            cache_dir=str(self.cache),
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
        )

    # ──────────────────────────────────────────────────────────
    # PANELS P/Q/R: Downstream Biology (delegates to downstream_panels)
    # ──────────────────────────────────────────────────────────
    def plot_downstream_panels(self, downstream_dir: str = "results/downstream") -> List[Path]:
        """Generate Panels P, Q, R from pre-computed downstream analysis.

        Run ``python -m src.evaluation.downstream_biology`` first to produce
        the JSON summaries under results/downstream/.
        """
        from .downstream_panels import (
            plot_clustering_panel,
            plot_classifier_panel,
            plot_de_concordance_panel,
        )
        import json as _json

        ds_dir = Path(downstream_dir)
        saved = []

        # Panel P: Clustering
        clust_path = ds_dir / "clustering_alignment.json"
        if clust_path.exists():
            with open(clust_path) as f:
                clust_data = _json.load(f)
            # Load auxiliary arrays (saved by downstream_biology.py)
            for key, fname in [("_umap_coords", "clustering_umap_coords.npy"),
                               ("_source", "clustering_source.npy"),
                               ("_cell_type", "clustering_cell_type.npy")]:
                arr_path = ds_dir / fname
                if arr_path.exists():
                    clust_data[key] = np.load(arr_path, allow_pickle=True)
            fig_p = plot_clustering_panel(clust_data, self.type_names, self.output, self.dpi)
            if fig_p:
                saved.append(self.output / "panel_p_clustering_mixing.pdf")
                plt.close(fig_p)

        # Panel Q: Classifier
        classif_path = ds_dir / "classifier_alignment.json"
        if classif_path.exists():
            with open(classif_path) as f:
                classif_data = _json.load(f)
            # The JSON now contains confusion_matrix and disc_proba/disc_y directly
            if "confusion_matrix" in classif_data:
                classif_data["_confusion_matrix"] = classif_data["confusion_matrix"]
            if "disc_proba" in classif_data:
                classif_data["_disc_proba"] = classif_data["disc_proba"]
                classif_data["_disc_y"] = classif_data["disc_y"]
            fig_q = plot_classifier_panel(classif_data, self.output, self.dpi)
            if fig_q:
                saved.append(self.output / "panel_q_classifier_alignment.pdf")
                plt.close(fig_q)

        # Panel R: DE concordance
        de_path = ds_dir / "de_concordance.json"
        if de_path.exists():
            with open(de_path) as f:
                de_data = _json.load(f)
            for cname in de_data:
                for key in ["_real_logfc", "_gen_logfc", "_shared_genes"]:
                    fname = f"de_{cname}_{key.lstrip('_')}.npy"
                    arr_path = ds_dir / fname
                    if arr_path.exists():
                        de_data[cname][key] = np.load(arr_path, allow_pickle=True).tolist()
            fig_r = plot_de_concordance_panel(de_data, self.output, self.dpi)
            if fig_r:
                saved.append(self.output / "panel_r_de_concordance.pdf")
                plt.close(fig_r)

        if not saved:
            logger.info("No downstream data found — run 'python -m src.evaluation.downstream_biology' first")
        return saved

    # ──────────────────────────────────────────────────────────
    # COMBINED REPORT
    # ──────────────────────────────────────────────────────────
    def generate_full_report(self, include_umap: bool = True) -> List[Path]:
        """Generate all available panels and a combined multi-page PDF.

        Panel sequence:
          Part I   (A–C):   Training & alignment
          Part II  (D–G):   Latent-space generation quality & diversity
          Part III (H–N):   Expression reconstruction & biological validation
          Part IV  (O–S):   Baselines, downstream utility, DE concordance, benchmarking

        Returns list of saved file paths.
        """
        saved = []

        logger.info("=" * 60)
        logger.info("Generating CLOP-DiT Results Report (A–R)")
        logger.info("=" * 60)

        # ── Part I: Training ──
        fig_a = self.plot_clop_training()
        if fig_a:
            saved.append(self.output / "panel_a_clop_training.pdf")
            plt.close(fig_a)

        if include_umap:
            fig_b = self.plot_clop_embedding_space()
            if fig_b:
                saved.append(self.output / "panel_b_clop_embedding_umap.pdf")
                plt.close(fig_b)
        else:
            logger.info("Skipping Panel B (UMAP) — use --include-umap to enable")

        fig_c = self.plot_dit_training()
        if fig_c:
            saved.append(self.output / "panel_c_dit_training.pdf")
            plt.close(fig_c)

        # ── Part II: Generation quality ──
        fig_d = self.plot_metrics_summary()
        if fig_d:
            saved.append(self.output / "panel_d_metrics_summary.pdf")
            plt.close(fig_d)

        fig_e = self.plot_real_vs_generated()
        if fig_e:
            saved.append(self.output / "panel_e_real_vs_generated.pdf")
            plt.close(fig_e)

        fig_f = self.plot_text_cell_heatmap()
        if fig_f:
            saved.append(self.output / "panel_f_text_cell_heatmap.pdf")
            plt.close(fig_f)

        fig_g = self.plot_per_type_generation()
        if fig_g:
            saved.append(self.output / "panel_g_per_type_generation.pdf")
            plt.close(fig_g)

        # ── Part III: Expression & biological validation ──
        fig_h = self.plot_expression_correlation()
        if fig_h:
            saved.append(self.output / "panel_h_expression_correlation.pdf")
            plt.close(fig_h)

        fig_i = self.plot_expression_analysis()
        if fig_i:
            saved.append(self.output / "panel_i_expression_analysis.pdf")
            plt.close(fig_i)

        fig_n = self.plot_marker_gene_comparison()
        if fig_n:
            saved.append(self.output / "panel_n_marker_gene_comparison.pdf")
            plt.close(fig_n)

        # ── Part IV: Baselines & downstream ──
        fig_o = self.plot_baseline_comparison()
        if fig_o:
            saved.append(self.output / "panel_o_baseline_comparison.pdf")
            plt.close(fig_o)

        # Panels P/Q/R: Downstream biology (from pre-computed JSONs)
        ds_saved = self.plot_downstream_panels()
        saved.extend(ds_saved)

        # Panel S: Model Benchmarking
        try:
            from .benchmark_panels import plot_benchmark_panel as _plot_s
            fig_s = _plot_s(
                report_path="results/benchmark_report.json",
                output_dir=self.output,
                dpi=self.dpi,
            )
            if fig_s:
                saved.append(self.output / "panel_s_benchmark.pdf")
                plt.close(fig_s)
        except Exception as exc:
            logger.warning(f"Panel S (Benchmark) failed: {exc}")

        # Panels J–M: Pre-generated external panels
        for panel_name in [
            "panel_j_diversity_diagnostics",
            "panel_k_expression_diversity",
            "panel_l_noise_tradeoff",
            "panel_m_conditioning_umap",
        ]:
            panel_pdf = self.output / f"{panel_name}.pdf"
            panel_png = self.output / f"{panel_name}.png"
            if panel_pdf.exists():
                saved.append(panel_pdf)
                logger.info(f"Including pre-generated {panel_name}")
            elif panel_png.exists():
                logger.info(f"Found {panel_name}.png but no PDF — including PNG")

        # ── Combine into multi-page PDF ──
        if saved:
            from .report import combine_panels_pdf
            combined_path = combine_panels_pdf(saved, self.output)
            saved.append(combined_path)

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
