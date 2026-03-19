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

import matplotlib.pyplot as plt
import numpy as np

# Shared style infrastructure (centralised in style.py)
from .direct_layout import bind_figure_region
from .style import (
    apply_style,
    set_figure_suptitle,
    style_axes,
    GRIDSPEC_TIGHT,
)
from src.utils.paths import CACHE_DIR, RESULTS_DIR, FIG_DIR, CHECKPOINT_DIR

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
        clop_history_path: Optional[str] = None,
        dit_history_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        dpi: int = 300,
    ):
        if clop_history_path is None:
            clop_history_path = str(CHECKPOINT_DIR / "clop_history.json")
        if dit_history_path is None:
            dit_history_path = str(CHECKPOINT_DIR / "dit_history.json")
        if cache_dir is None:
            cache_dir = str(CACHE_DIR)
        if output_dir is None:
            output_dir = str(FIG_DIR)
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

    def _save_panel(self, fig: plt.Figure, basename: str) -> Path:
        """Polish figure, run visual conflict detection, save PNG + PDF, return PNG path."""
        from .style import save_with_vcd
        path = self.output / f"{basename}.png"
        return save_with_vcd(fig, path, self.dpi)

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
        from .fig03_training import plot_clop_training as _plot_a
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
        from .fig04_embedding import plot_clop_embedding_space as _plot_b
        return _plot_b(
            cache_dir=self.cache,
            type_names=self.type_names,
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, path, dpi: self._save_panel(fig, "fig04_clop_embedding_umap"),
            n_cells=n_cells,
        )

    # ──────────────────────────────────────────────────────────
    # PANEL C: DiT Training Dynamics
    # ──────────────────────────────────────────────────────────
    def plot_dit_training(self, save: bool = True) -> Optional[plt.Figure]:
        """2x2 DiT flow-matching training dynamics (matches Panel A layout for side-by-side placement).

        Top-left: Loss; top-right: Cosine similarity; bottom-left: LR schedule; bottom-right: key metrics summary.
        """
        from .fig03_training import plot_dit_training as _plot_c
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
    def plot_metrics_summary(self, gen_metrics_path: Optional[str] = None,
                             expr_metrics_path: Optional[str] = None,
                             div_metrics_path: Optional[str] = None,
                             save: bool = True) -> Optional[plt.Figure]:
        """Visual metrics dashboard -- replaces table with bar charts + radar.

        D1: Training convergence (horizontal bars for final key metrics)
        D2: Generation quality radar (FD, coverage, diversity, centroid cos, expr r)
        D3: Diversity gauges (diversity ratio, collapsed types, cond gain)
        D4: Configuration + expression summary (compact annotated bars)
        """
        if gen_metrics_path is None:
            gen_metrics_path = str(RESULTS_DIR / "generation_metrics.json")
        if expr_metrics_path is None:
            expr_metrics_path = str(RESULTS_DIR / "expression_metrics.json")
        if div_metrics_path is None:
            div_metrics_path = str(RESULTS_DIR / "diversity_diagnostics.json")
        from .fig05_metrics import plot_metrics_summary as _plot_d
        return _plot_d(
            clop_hist=self.clop_hist,
            dit_hist=self.dit_hist,
            gen_metrics_path=gen_metrics_path,
            expr_metrics_path=expr_metrics_path,
            div_metrics_path=div_metrics_path,
            output_dir=str(self.output),
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, path, dpi: self._save_panel(fig, "fig05_metrics_summary"),
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
        from .fig04_embedding import plot_real_vs_generated as _plot_e
        return _plot_e(
            cache_dir=str(self.cache),
            generated_path=generated_path,
            generated_labels_path=generated_labels_path,
            n_cells=n_cells,
            output_dir=str(self.output),
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, path, dpi: self._save_panel(fig, "fig04_real_vs_generated"),
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
        from .fig07_alignment import plot_text_cell_heatmap as _plot_f
        return _plot_f(
            cache_dir=str(self.cache),
            type_names=self.type_names,
            output_dir=str(self.output),
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, path, dpi: self._save_panel(fig, "fig07_text_cell_alignment"),
        )

    # ──────────────────────────────────────────────────────────
    # PANEL G: Per-Type Generation Fidelity
    # ──────────────────────────────────────────────────────────
    def plot_per_type_generation(
        self,
        metrics_path: Optional[str] = None,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Per-type generation quality: centroid cosine + Frechet distance.

        G1: Centroid cosine per type (sorted)
        G2: Frechet distance per type (sorted, log scale)
        G3: Scatter of centroid cosine vs n_real (type size dependency)
        """
        if metrics_path is None:
            metrics_path = str(RESULTS_DIR / "generation_metrics.json")
        from .fig06_fidelity import plot_per_type_generation as _plot_g
        return _plot_g(
            metrics_path=metrics_path,
            output_dir=str(self.output),
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, path, dpi: self._save_panel(fig, "fig06_per_type_fidelity"),
        )

    # ──────────────────────────────────────────────────────────
    # PANEL H: Gene Expression Correlation (scatter + per-type + markers)
    # ──────────────────────────────────────────────────────────
    def plot_expression_correlation(
        self,
        real_expr_path: Optional[str] = None,
        gen_expr_path: Optional[str] = None,
        real_labels_path: Optional[str] = None,
        gen_labels_path: Optional[str] = None,
        gene_names_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Enhanced gene expression fidelity with density-aware scatter and richer bars.

        H1: Density scatter of per-gene mean expression with residual coloring
        H2: Per-type Pearson r lollipop chart with tier shading
        H3: Marker gene grouped bars with error bars and fold-change annotation
        H4: Per-gene residual distribution (gen - real)
        """
        real_expr_path = real_expr_path or str(RESULTS_DIR / "real_expression.npy")
        gen_expr_path = gen_expr_path or str(RESULTS_DIR / "generated_expression.npy")
        real_labels_path = real_labels_path or str(RESULTS_DIR / "real_expression_labels.npy")
        gen_labels_path = gen_labels_path or str(RESULTS_DIR / "generated_expression_labels.npy")
        gene_names_path = gene_names_path or str(RESULTS_DIR / "expression_gene_names.json")
        metrics_path = metrics_path or str(RESULTS_DIR / "expression_metrics.json")
        from .fig09_expression_corr import plot_expression_correlation as _plot_h
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
        real_expr_path: Optional[str] = None,
        gen_expr_path: Optional[str] = None,
        gene_names_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Enhanced expression decoder analysis with denser information.

        I1: CV scatter (real vs gen per-gene) with identity line and outlier genes
        I2: Expression range ribbon with percentile bands (10-90th, 25-75th)
        I3: Per-cell expression std as overlaid KDE-style histograms
        I4: Top variable genes heatmap (genes with highest CV difference)
        """
        real_expr_path = real_expr_path or str(RESULTS_DIR / "real_expression.npy")
        gen_expr_path = gen_expr_path or str(RESULTS_DIR / "generated_expression.npy")
        gene_names_path = gene_names_path or str(RESULTS_DIR / "expression_gene_names.json")
        metrics_path = metrics_path or str(RESULTS_DIR / "expression_metrics.json")
        from .fig10_expression_analysis import plot_expression_analysis as _plot_i
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
        real_expr_path: Optional[str] = None,
        gen_expr_path: Optional[str] = None,
        real_labels_path: Optional[str] = None,
        gen_labels_path: Optional[str] = None,
        gene_names_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Rich marker-gene comparison with violin plots and annotated heatmaps.

        N1: Violin + strip plot -- per-marker expression distribution (real vs gen)
        N2: Dual heatmap with cell-value annotations and row-normalized coloring
        N3: Difference heatmap with statistical significance indicators
        N4: Fold-change waterfall for all markers
        """
        real_expr_path = real_expr_path or str(RESULTS_DIR / "real_expression.npy")
        gen_expr_path = gen_expr_path or str(RESULTS_DIR / "generated_expression.npy")
        real_labels_path = real_labels_path or str(RESULTS_DIR / "real_expression_labels.npy")
        gen_labels_path = gen_labels_path or str(RESULTS_DIR / "generated_expression_labels.npy")
        gene_names_path = gene_names_path or str(RESULTS_DIR / "expression_gene_names.json")
        metrics_path = metrics_path or str(RESULTS_DIR / "expression_metrics.json")
        from .fig08_markers import plot_marker_gene_comparison as _plot_n
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
        gen_metrics_path: Optional[str] = None,
        div_metrics_path: Optional[str] = None,
        baseline_metrics_path: Optional[str] = None,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """Panel O: CLOP-DiT vs baselines — graphical O3 (no table)."""
        gen_metrics_path = gen_metrics_path or str(RESULTS_DIR / "generation_metrics.json")
        div_metrics_path = div_metrics_path or str(RESULTS_DIR / "diversity_diagnostics.json")
        baseline_metrics_path = baseline_metrics_path or str(RESULTS_DIR / "baseline_metrics.json")
        from .fig15_baselines import plot_baseline_comparison as _plot_o
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
    def plot_downstream_panels(self, downstream_dir: Optional[str] = None) -> List[Path]:
        """Generate Panels P, Q, R from pre-computed downstream analysis.

        Run ``python -m src.evaluation.downstream_biology`` first to produce
        the JSON summaries under results/downstream/.
        """
        if downstream_dir is None:
            downstream_dir = str(RESULTS_DIR / "downstream")
        from .fig17_downstream import (
            plot_clustering_panel,
            plot_classifier_panel,
            plot_clustering_and_classifier_merged,
        )
        from .fig18_de_concordance import plot_de_concordance_panel
        import json as _json

        ds_dir = Path(downstream_dir)
        saved = []
        clust_data = None
        classif_data = None

        # Panel P: Clustering — load data only (standalone save skipped)
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

        # Panel Q: Classifier — load data only (standalone save skipped)
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
                    saved.append(self.output / "fig17_downstream_pq.pdf")
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
                saved.append(self.output / "fig18_de_concordance.pdf")
                plt.close(fig_r)

        if not saved:
            logger.info("No downstream data found — run 'python -m src.evaluation.downstream_biology' first")
        return saved

    # ──────────────────────────────────────────────────────────
    # MERGED L+K: Diversity & Trade-off (PIL composition)
    # ──────────────────────────────────────────────────────────
    def _compose_diversity_tradeoff(self) -> Optional[Path]:
        """Compose Panels L and K with a new diversity-tail violin panel for the article figure.

        Panel labels are offset to avoid duplication in the merged figure:
          L = (a), K = (b)(c), violin = unlabelled (full-width row).
        """
        import io as _io
        from PIL import Image
        from .fig05_metrics import plot_diversity_distributions_violin
        from .fig14_expr_diversity import plot_expression_diversity_panel

        l_path = self.output / "fig13_noise_tradeoff.png"

        # Regenerate K with label_offset=1 so labels become (b)(c) in merged fig.
        div_metrics_path = str(RESULTS_DIR / "diversity_diagnostics.json")
        div_path = Path(div_metrics_path)
        t6_data: Dict = {}
        if div_path.exists():
            with open(div_path) as f:
                div_all = json.load(f)
            t6_data = div_all.get("test6_expression_diversity", {})

        fig_k = plot_expression_diversity_panel(
            t6_data,
            output_dir=str(self.output),
            dpi=self.dpi,
            label_offset=1,  # (b)(c) to complement L's (a)
            save=False,
        )

        if not l_path.exists() or fig_k is None:
            missing = []
            if not l_path.exists():
                missing.append("L")
            if fig_k is None:
                missing.append("K")
            logger.info("Both panels L and K required for merged figure; skipping (missing %s)", ", ".join(missing))
            if fig_k is not None:
                plt.close(fig_k)
            return None

        # Render K figure to a PIL image
        buf_k = _io.BytesIO()
        from .style import get_export_savefig_kwargs
        fig_k.savefig(buf_k, format="png", **get_export_savefig_kwargs(fig_k, dpi=self.dpi, pad_inches=0.08))
        buf_k.seek(0)
        k_image = Image.open(buf_k)
        plt.close(fig_k)

        def _trim_whitespace(image: Image.Image, threshold: int = 245, pad: int = 6) -> np.ndarray:
            """Crop near-white borders from raster panels before tiling."""
            arr = np.array(image.convert("RGB"))
            mask = np.any(arr < threshold, axis=2)
            if not np.any(mask):
                return arr
            ys, xs = np.where(mask)
            y0 = max(int(ys.min()) - pad, 0)
            y1 = min(int(ys.max()) + pad + 1, arr.shape[0])
            x0 = max(int(xs.min()) - pad, 0)
            x1 = min(int(xs.max()) + pad + 1, arr.shape[1])
            return arr[y0:y1, x0:x1]

        l_image = Image.open(l_path)
        images = [(l_path.stem, l_image), ("fig14_expression_diversity", k_image)]

        fig = plt.figure(figsize=(13.6, 8.6), dpi=self.dpi)
        layout = bind_figure_region(fig, (0.02, 0.04, 0.98, 0.96))
        top_row, bottom_row = layout.split_rows([1.0, 1.0], hspace=0.14)
        top_left, top_right = top_row.split_cols(2, wspace=0.04)
        # suptitle removed per revision; title information moved to LaTeX caption

        for col, (_, image) in enumerate(images):
            ax = [top_left, top_right][col].add_axes(fig)
            ax.imshow(_trim_whitespace(image), aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.axis("off")

        ax_bottom = bottom_row.add_axes(fig)
        violin_fig = plot_diversity_distributions_violin(
            cache_dir=str(self.cache),
            div_metrics_path=div_metrics_path,
            output_dir=str(self.output),
            dpi=self.dpi,
            save=False,
            ax=ax_bottom,
            top_n=6,
        )
        if violin_fig is None:
            plt.close(fig)
            logger.info("Violin panel required for Fig 11; skipping (missing diversity violin data)")
            return None
        ax_bottom.margins(y=0.05)

        merged_path = self.output / "fig_diversity_tradeoff.png"
        from .style import save_with_vcd
        save_with_vcd(
            fig,
            merged_path,
            self.dpi,
            close=True,
            layout_rect=(0.02, 0.04, 0.98, 0.96),
        )
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
        # NOTE: Standalone A/C panels are NOT saved — only the merged A+C is
        # needed for the article (fig03_training_dynamics).

        # ── Merged: A+C Training Dynamics ──
        try:
            from .fig03_training import plot_training_dynamics_combined
            fig_ac = plot_training_dynamics_combined(
                clop_hist=self.clop_hist,
                dit_hist=self.dit_hist,
                output_dir=self.output,
                dpi=self.dpi,
                save=True,
                save_panel_fn=lambda fig, name, *a, **kw: self._save_panel(fig, name),
            )
            if fig_ac:
                saved.append(self.output / "fig03_training_dynamics.pdf")
                plt.close(fig_ac)
        except Exception as exc:
            logger.warning(f"Merged A+C figure failed: {exc}")

        # ── Part II: Generation quality ──
        fig_d = self.plot_metrics_summary()
        if fig_d:
            saved.append(self.output / "fig05_metrics_summary.pdf")
            plt.close(fig_d)

        # NOTE: Standalone B/E panels are NOT saved — only the merged B+E is
        # needed for the article (fig04_embedding_space).

        fig_f = self.plot_text_cell_heatmap()
        if fig_f:
            saved.append(self.output / "fig07_text_cell_alignment.pdf")
            plt.close(fig_f)

        fig_g = self.plot_per_type_generation()
        if fig_g:
            saved.append(self.output / "fig06_per_type_fidelity.pdf")
            plt.close(fig_g)

        # ── Merged: B+E Embedding Space ──
        if include_umap:
            try:
                from .fig04_embedding import plot_embedding_space_merged
                fig_be = plot_embedding_space_merged(
                    cache_dir=str(self.cache),
                    type_names=self.type_names,
                    n_cells=5000,
                    output_dir=str(self.output),
                    dpi=self.dpi,
                    save=True,
                    save_panel_fn=lambda fig, path, dpi: self._save_panel(fig, "fig04_embedding_space"),
                )
                if fig_be:
                    saved.append(self.output / "fig04_embedding_space.pdf")
                    plt.close(fig_be)
            except Exception as exc:
                logger.warning(f"Merged B+E figure failed: {exc}")

        # ── Merged: G+F Fidelity & Alignment ──
        # NOTE: Legacy composite not in article manifest — skipped.

        # ── Part III: Expression & biological validation ──
        fig_h = self.plot_expression_correlation()
        if fig_h:
            saved.append(self.output / "fig09_expression_correlation.pdf")
            plt.close(fig_h)

        fig_i = self.plot_expression_analysis()
        if fig_i:
            saved.append(self.output / "fig10_expression_analysis.pdf")
            plt.close(fig_i)

        fig_n = self.plot_marker_gene_comparison()
        if fig_n:
            saved.append(self.output / "fig08_marker_genes.pdf")
            plt.close(fig_n)

        # ── Part IV: Baselines & downstream ──
        fig_o = self.plot_baseline_comparison()
        if fig_o:
            saved.append(self.output / "fig15_baseline_comparison.pdf")
            plt.close(fig_o)

        # Panels P/Q/R: Downstream biology (from pre-computed JSONs)
        ds_saved = self.plot_downstream_panels()
        saved.extend(ds_saved)

        # Panel S: Model Benchmarking
        try:
            from .fig16_benchmark import plot_benchmark_panel as _plot_s
            fig_s = _plot_s(
                report_path=str(RESULTS_DIR / "benchmark_report.json"),
                output_dir=self.output,
                dpi=self.dpi,
            )
            if fig_s:
                saved.append(self.output / "fig16_benchmark.pdf")
                plt.close(fig_s)
        except Exception as exc:
            logger.warning(f"Panel S (Benchmark) failed: {exc}")

        # Panels J–M: Pre-generated external panels (fig12, fig14, fig13, fig11)
        for panel_name in [
            "fig12_diversity_diagnostics",
            "fig14_expression_diversity",
            "fig13_noise_tradeoff",
            "fig11_conditioning_umap",
        ]:
            panel_pdf = self.output / f"{panel_name}.pdf"
            panel_png = self.output / f"{panel_name}.png"
            if panel_pdf.exists():
                saved.append(panel_pdf)
                logger.info(f"Including pre-generated {panel_name}")
            elif panel_png.exists():
                logger.info(f"Found {panel_name}.png but no PDF — including PNG")

        # ── Merged: L+K Diversity & Trade-off ──
        # NOTE: Legacy composite not in article manifest — skipped.

        # ── Part V: Extended analyses (Figs 21–24) ──
        fig_abl = self.plot_ablation_heatmap()
        if fig_abl:
            saved.append(self.output / "fig21_ablation_heatmap.pdf")
            plt.close(fig_abl)

        fig_ms = self.plot_multi_seed_robustness()
        if fig_ms:
            saved.append(self.output / "fig22_multi_seed_robustness.pdf")
            plt.close(fig_ms)

        fig_ood = self.plot_ood_showcase()
        if fig_ood:
            saved.append(self.output / "fig23_ood_showcase.pdf")
            plt.close(fig_ood)

        fig_var = self.plot_variance_deepdive()
        if fig_var:
            saved.append(self.output / "fig24_variance_deepdive.pdf")
            plt.close(fig_var)

        # ── Part VI: Downstream application experiments (Figs 25–28) ──
        fig_cd = self.plot_cross_dataset_validation()
        if fig_cd:
            saved.append(self.output / "fig25_cross_dataset.pdf")
            plt.close(fig_cd)

        fig_de = self.plot_expanded_de()
        if fig_de:
            saved.append(self.output / "fig26_expanded_de.pdf")
            plt.close(fig_de)

        fig_ood2 = self.plot_ood_robustness()
        if fig_ood2:
            saved.append(self.output / "fig27_ood_robustness.pdf")
            plt.close(fig_ood2)

        fig_mkr = self.plot_marker_completeness()
        if fig_mkr:
            saved.append(self.output / "fig28_marker_completeness.pdf")
            plt.close(fig_mkr)

        # ── Combine into multi-page PDF ──
        if saved:
            from .report import combine_panels_pdf
            combined_path = combine_panels_pdf(saved, self.output)
            saved.append(combined_path)

        logger.info(f"Done — {len(saved)} files generated in {self.output}/")
        return saved

    # ──────────────────────────────────────────────────────────────
    # NEW: Figs 21–24 — Ablation, Multi-seed, OOD, Variance
    # ──────────────────────────────────────────────────────────────

    def plot_ablation_heatmap(self, save: bool = True) -> Optional[plt.Figure]:
        """Fig 21: Ablation comparison heatmap."""
        from .fig21_ablation_heatmap import plot_ablation_heatmap as _plot
        return _plot(
            ablation_path=str(RESULTS_DIR / "ablations" / "all_summaries.json"),
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, name, *a, **kw: self._save_panel(fig, name),
        )

    def plot_multi_seed_robustness(self, save: bool = True) -> Optional[plt.Figure]:
        """Fig 22: Multi-seed robustness."""
        from .fig22_multi_seed import plot_multi_seed_robustness as _plot
        return _plot(
            report_path=str(RESULTS_DIR / "multi_seed" / "multi_seed_report.json"),
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, name, *a, **kw: self._save_panel(fig, name),
        )

    def plot_ood_showcase(self, save: bool = True) -> Optional[plt.Figure]:
        """Fig 23: OOD generation showcase."""
        from .fig23_ood_showcase import plot_ood_showcase as _plot
        return _plot(
            ood_path=str(RESULTS_DIR / "ood_evaluation" / "ood_results.json"),
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, name, *a, **kw: self._save_panel(fig, name),
        )

    def plot_variance_deepdive(self, save: bool = True) -> Optional[plt.Figure]:
        """Fig 24: Per-gene variance deep-dive."""
        from .fig24_variance_deepdive import plot_variance_deepdive as _plot
        return _plot(
            real_expr_path=str(RESULTS_DIR / "real_expression.npy"),
            gen_expr_path=str(RESULTS_DIR / "generated_expression.npy"),
            gene_names_path=str(RESULTS_DIR / "expression_gene_names.json"),
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, name, *a, **kw: self._save_panel(fig, name),
        )

    # ──────────────────────────────────────────────────────────────
    # Figs 25–28 — Downstream application experiments
    # ──────────────────────────────────────────────────────────────

    def plot_cross_dataset_validation(self, save: bool = True) -> Optional[plt.Figure]:
        """Fig 25: Cross-dataset biological validation."""
        from .fig25_cross_dataset import plot_cross_dataset_validation as _plot
        return _plot(
            data_path=str(RESULTS_DIR / "downstream" / "cross_dataset_validation.json"),
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, name, *a, **kw: self._save_panel(fig, name),
        )

    def plot_expanded_de(self, save: bool = True) -> Optional[plt.Figure]:
        """Fig 26: Expanded DE concordance."""
        from .fig26_expanded_de import plot_expanded_de as _plot
        return _plot(
            de_path=str(RESULTS_DIR / "downstream" / "expanded_de_concordance.json"),
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, name, *a, **kw: self._save_panel(fig, name),
        )

    def plot_ood_robustness(self, save: bool = True) -> Optional[plt.Figure]:
        """Fig 27: OOD robustness evaluation."""
        from .fig27_ood_robustness import plot_ood_robustness as _plot
        return _plot(
            data_path=str(RESULTS_DIR / "downstream" / "ood_robustness_combined.json"),
            marker_path=str(RESULTS_DIR / "ood_evaluation" / "ood_marker_analysis.json"),
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, name, *a, **kw: self._save_panel(fig, name),
        )

    def plot_marker_completeness(self, save: bool = True) -> Optional[plt.Figure]:
        """Fig 28: Marker gene program completeness."""
        from .fig28_marker_completeness import plot_marker_completeness as _plot
        return _plot(
            data_path=str(RESULTS_DIR / "downstream" / "marker_completeness.json"),
            output_dir=self.output,
            dpi=self.dpi,
            save=save,
            save_panel_fn=lambda fig, name, *a, **kw: self._save_panel(fig, name),
        )


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
    parser.add_argument("--clop-history", default=str(CHECKPOINT_DIR / "clop_history.json"))
    parser.add_argument("--dit-history", default=str(CHECKPOINT_DIR / "dit_history.json"))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--output-dir", default=str(FIG_DIR))
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
