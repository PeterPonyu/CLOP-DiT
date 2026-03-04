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
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend for PDF generation

# Shared style infrastructure (centralised in style.py)
from .style import (
    apply_style,
    style_axes,
    GRIDSPEC_TIGHT,
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
        auto_refine: bool = True,
        max_refine_passes: int = 3,
    ):
        self.cache = Path(cache_dir)
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.auto_refine = auto_refine
        self.max_refine_passes = max_refine_passes
        self._refine_stats: Dict[str, int] = {}  # panel_id -> passes_used

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

    def _save_panel(self, fig: plt.Figure, basename: str) -> Path:
        """Polish figure, run visual conflict detection, save PNG + PDF, return PNG path."""
        from .style import save_with_vcd
        path = self.output / f"{basename}.png"
        return save_with_vcd(fig, path, self.dpi)

    def _save_with_refine(
        self,
        make_fn,
        panel_id: str,
        basename: str,
        *,
        max_passes: int = 0,
    ) -> Optional[plt.Figure]:
        """Optionally auto-refine a panel, then save it.

        Parameters
        ----------
        make_fn : callable(PanelConfig) -> Figure | None
            A function that creates the figure from a PanelConfig.
        panel_id : str
            Short identifier (e.g. "A", "D", "R").
        basename : str
            Output file basename (without extension).
        max_passes : int
            Override max_refine_passes for this panel (0 = use self default).
        """
        if not self.auto_refine:
            return None  # caller should fall back to direct generation

        from .auto_refine import refine_panel
        from .panel_config import PanelConfig

        passes = max_passes or self.max_refine_passes
        default_config = PanelConfig(panel_id=panel_id)

        fig, final_cfg, issues = refine_panel(
            make_fn, default_config,
            max_passes=passes,
            label=panel_id,
            verbose=True,
        )
        n_warn = sum(1 for i in issues if i.get("severity") == "warning")
        self._refine_stats[panel_id] = passes
        logger.info(f"[auto_refine] {panel_id}: {n_warn} warnings after refinement")

        if fig is not None:
            self._save_panel(fig, basename)
        return fig

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
        from .panels_training import plot_clop_training as _plot_a
        return _plot_a(
            hist=self.clop_hist,
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, name, *a, **kw: self._save_panel(fig, name),
        )

    # ──────────────────────────────────────────────────────────
    # PANEL B: CLOP Alignment Space (UMAP) — all in CLOP shared space
    # ──────────────────────────────────────────────────────────
    def plot_clop_embedding_space(
        self, n_cells: int = 8000, save: bool = True
    ) -> Optional[plt.Figure]:
        """3-panel CLOP alignment visualization (B0: histogram, B1: prototypes, B2: cells + overlay)."""
        from .panels_embedding import plot_clop_embedding_space as _plot_b
        return _plot_b(
            cache_dir=self.cache,
            type_names=self.type_names,
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, path, dpi: self._save_panel(fig, "panel_b_clop_embedding_umap"),
            n_cells=n_cells,
        )

    # ──────────────────────────────────────────────────────────
    # PANEL C: DiT Training Dynamics
    # ──────────────────────────────────────────────────────────
    def plot_dit_training(self, save: bool = True) -> Optional[plt.Figure]:
        """2x2 DiT flow-matching training dynamics (matches Panel A layout for side-by-side placement).

        Top-left: Loss; top-right: Cosine similarity; bottom-left: LR schedule; bottom-right: key metrics summary.
        """
        from .panels_training import plot_dit_training as _plot_c
        return _plot_c(
            hist=self.dit_hist,
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, name, *a, **kw: self._save_panel(fig, name),
        )

    # ──────────────────────────────────────────────────────────
    # PANEL D: Metrics Summary (grouped, compact, publication-ready)
    # ──────────────────────────────────────────────────────────
    def plot_metrics_summary(self, gen_metrics_path: str = "results/generation_metrics.json",
                             expr_metrics_path: str = "results/expression_metrics.json",
                             div_metrics_path: str = "results/diversity_diagnostics.json",
                             save: bool = True) -> Optional[plt.Figure]:
        """Visual metrics dashboard -- replaces table with bar charts + radar.

        D1: Training convergence (horizontal bars for final key metrics)
        D2: Generation quality radar (FD, coverage, diversity, centroid cos, expr r)
        D3: Diversity gauges (diversity ratio, collapsed types, cond gain)
        D4: Configuration + expression summary (compact annotated bars)
        """
        from .panels_quality import plot_metrics_summary as _plot_d
        return _plot_d(
            clop_hist=self.clop_hist,
            dit_hist=self.dit_hist,
            gen_metrics_path=gen_metrics_path,
            expr_metrics_path=expr_metrics_path,
            div_metrics_path=div_metrics_path,
            output_dir=str(self.output),
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, path, dpi: self._save_panel(fig, "panel_d_metrics_summary"),
        )

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
        from .panels_quality import plot_real_vs_generated as _plot_e
        return _plot_e(
            cache_dir=str(self.cache),
            generated_path=generated_path,
            generated_labels_path=generated_labels_path,
            n_cells=n_cells,
            output_dir=str(self.output),
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, path, dpi: self._save_panel(fig, "panel_e_real_vs_generated"),
        )

    # ──────────────────────────────────────────────────────────
    # PANEL F: Text–Cell Cosine Similarity Heatmap (69×69)
    # ──────────────────────────────────────────────────────────
    def plot_text_cell_heatmap(self, save: bool = True) -> Optional[plt.Figure]:
        """Enhanced 69x69 text-cell alignment heatmap with rich annotations.

        F1: Clustered heatmap with diagonal highlight and off-diagonal structure
        F2: Sorted per-type alignment bars with threshold bands
        F3: Distribution of diagonal vs off-diagonal similarities
        """
        from .panels_quality import plot_text_cell_heatmap as _plot_f
        return _plot_f(
            cache_dir=str(self.cache),
            type_names=self.type_names,
            output_dir=str(self.output),
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, path, dpi: self._save_panel(fig, "panel_f_text_cell_heatmap"),
        )

    # ──────────────────────────────────────────────────────────
    # PANEL G: Per-Type Generation Fidelity
    # ──────────────────────────────────────────────────────────
    def plot_per_type_generation(
        self,
        metrics_path: str = "results/generation_metrics.json",
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Per-type generation quality: centroid cosine + Frechet distance.

        G1: Centroid cosine per type (sorted)
        G2: Frechet distance per type (sorted, log scale)
        G3: Scatter of centroid cosine vs n_real (type size dependency)
        """
        from .panels_quality import plot_per_type_generation as _plot_g
        return _plot_g(
            metrics_path=metrics_path,
            output_dir=str(self.output),
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, path, dpi: self._save_panel(fig, "panel_g_per_type_generation"),
        )

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
        from .panels_expression import plot_expression_correlation as _plot_h
        return _plot_h(
            real_expr_path=real_expr_path,
            gen_expr_path=gen_expr_path,
            real_labels_path=real_labels_path,
            gen_labels_path=gen_labels_path,
            gene_names_path=gene_names_path,
            metrics_path=metrics_path,
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, name, out, dpi: self._save_panel(fig, name),
        )

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
        from .panels_expression import plot_expression_analysis as _plot_i
        return _plot_i(
            real_expr_path=real_expr_path,
            gen_expr_path=gen_expr_path,
            gene_names_path=gene_names_path,
            metrics_path=metrics_path,
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, name, out, dpi: self._save_panel(fig, name),
        )

    # ──────────────────────────────────────────────────────────
    # PANEL N: Marker Gene Comparison (per-type real vs generated)
    # ──────────────────────────────────────────────────────────

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

        N1: Violin + strip plot -- per-marker expression distribution (real vs gen)
        N2: Dual heatmap with cell-value annotations and row-normalized coloring
        N3: Difference heatmap with statistical significance indicators
        N4: Fold-change waterfall for all markers
        """
        from .panels_expression import plot_marker_gene_comparison as _plot_n
        return _plot_n(
            real_expr_path=real_expr_path,
            gen_expr_path=gen_expr_path,
            real_labels_path=real_labels_path,
            gen_labels_path=gen_labels_path,
            gene_names_path=gene_names_path,
            metrics_path=metrics_path,
            type_names=self.type_names,
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, name, out, dpi: self._save_panel(fig, name),
        )

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
            plot_clustering_and_classifier_merged,
        )
        import json as _json

        ds_dir = Path(downstream_dir)
        saved = []
        clust_data = None
        classif_data = None

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

        # Merged P+Q: Clustering & Classifier
        if clust_data is not None and classif_data is not None:
            try:
                fig_pq = plot_clustering_and_classifier_merged(
                    clustering_data=clust_data,
                    classifier_data=classif_data,
                    type_names=self.type_names,
                    output_dir=self.output,
                    dpi=self.dpi,
                    save=True,
                )
                if fig_pq:
                    saved.append(self.output / "fig_downstream_pq.pdf")
                    plt.close(fig_pq)
            except Exception as exc:
                logger.warning(f"Merged P+Q figure failed: {exc}")

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
    # MERGED L+K: Diversity & Trade-off (PIL composition)
    # ──────────────────────────────────────────────────────────
    def _compose_diversity_tradeoff(self) -> Optional[Path]:
        """Compose Panels L and K with a new diversity-tail violin panel for the article figure."""
        from PIL import Image
        from .panels_quality import plot_diversity_distributions_violin

        k_path = self.output / "panel_k_expression_diversity.png"
        l_path = self.output / "panel_l_noise_tradeoff.png"

        images = []
        for p in [l_path, k_path]:
            if p.exists():
                images.append((p.stem, Image.open(p)))
            else:
                logger.warning(f"Missing {p.name} for merged L+K figure")
        if not images:
            return None

        fig = plt.figure(figsize=(13.2, 8.8), dpi=self.dpi)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.78], wspace=0.03, hspace=0.22)
        fig.suptitle("Diversity Trade-off and Expression Variance", fontsize=12, y=0.985)

        for col, (_, image) in enumerate(images[:2]):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(np.array(image))
            ax.axis("off")

        if len(images) == 1:
            ax_blank = fig.add_subplot(gs[0, 1])
            ax_blank.axis("off")

        ax_bottom = fig.add_subplot(gs[1, :])
        violin_fig = plot_diversity_distributions_violin(
            cache_dir=str(self.cache),
            div_metrics_path="results/diversity_diagnostics.json",
            output_dir=str(self.output),
            dpi=self.dpi,
            save=False,
            ax=ax_bottom,
            top_n=6,
        )
        if violin_fig is None:
            ax_bottom.axis("off")
            ax_bottom.text(
                0.5,
                0.5,
                "Diversity-tail violin enhancement unavailable",
                ha="center",
                va="center",
                transform=ax_bottom.transAxes,
                fontsize=10,
                color="#555",
            )

        merged_path = self.output / "fig_diversity_tradeoff.png"
        from .style import save_with_vcd
        save_with_vcd(fig, merged_path, self.dpi, close=True)
        logger.info(f"Saved merged L+K → {merged_path}")
        return merged_path

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

        # ── Merged: A+C Training Dynamics ──
        try:
            from .panels_training import plot_training_dynamics_combined
            fig_ac = plot_training_dynamics_combined(
                clop_hist=self.clop_hist,
                dit_hist=self.dit_hist,
                output_dir=self.output,
                dpi=self.dpi,
                save=True,
                save_panel_fn=lambda fig, name, *a, **kw: self._save_panel(fig, name),
            )
            if fig_ac:
                saved.append(self.output / "fig_training_dynamics.pdf")
                plt.close(fig_ac)
        except Exception as exc:
            logger.warning(f"Merged A+C figure failed: {exc}")

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

        # ── Merged: B+E Embedding Space ──
        if include_umap:
            try:
                from .panels_quality import plot_embedding_space_merged
                fig_be = plot_embedding_space_merged(
                    cache_dir=str(self.cache),
                    type_names=self.type_names,
                    n_cells=5000,
                    output_dir=str(self.output),
                    dpi=self.dpi,
                    save=True,
                    save_panel_fn=lambda fig, path, dpi: self._save_panel(fig, "fig_embedding_space"),
                )
                if fig_be:
                    saved.append(self.output / "fig_embedding_space.pdf")
                    plt.close(fig_be)
            except Exception as exc:
                logger.warning(f"Merged B+E figure failed: {exc}")

        # ── Merged: G+F Fidelity & Alignment ──
        try:
            from .panels_quality import plot_fidelity_and_alignment_merged
            fig_gf = plot_fidelity_and_alignment_merged(
                cache_dir=str(self.cache),
                metrics_path="results/generation_metrics.json",
                type_names=self.type_names,
                output_dir=str(self.output),
                dpi=self.dpi,
                save=True,
                save_panel_fn=lambda fig, path, dpi: self._save_panel(fig, "fig_fidelity_alignment"),
            )
            if fig_gf:
                saved.append(self.output / "fig_fidelity_alignment.pdf")
                plt.close(fig_gf)
        except Exception as exc:
            logger.warning(f"Merged G+F figure failed: {exc}")

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

        # ── Merged: L+K Diversity & Trade-off ──
        merged_lk = self._compose_diversity_tradeoff()
        if merged_lk:
            saved.append(merged_lk)

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
    parser.add_argument("--no-auto-refine", action="store_true",
                        help="Disable VCD auto-refinement loop")
    parser.add_argument("--max-refine-passes", type=int, default=3,
                        help="Max auto-refinement passes per panel (default: 3)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--generated", default=None, help="Path to generated cells .npy")
    args = parser.parse_args()

    viz = ResultsVisualizer(
        clop_history_path=args.clop_history,
        dit_history_path=args.dit_history,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
        auto_refine=not args.no_auto_refine,
        max_refine_passes=args.max_refine_passes,
    )

    saved = viz.generate_full_report(include_umap=not args.no_umap)
    print(f"\nGenerated {len(saved)} files:")
    for p in saved:
        print(f"  {p}")

    # Print auto-refinement stats
    if viz._refine_stats:
        print(f"\nAuto-refinement passes per panel:")
        for panel_id, passes in sorted(viz._refine_stats.items()):
            print(f"  {panel_id}: {passes} passes")


if __name__ == "__main__":
    main()
