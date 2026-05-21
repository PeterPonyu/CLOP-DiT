# src/visualization/

Publication-quality figure generation for the CLOP-DiT manuscript. Produces
article-facing component PDFs, display-figure composition policy metadata, and
delivery manifests for the paper-facing assets.

## Panel Modules

Each `panels_*.py` file renders one or more manuscript panels:

| Module | Panel(s) | Content |
|--------|----------|---------|
| `panels_training.py` | A, C | CLOP and DiT training dynamics (loss, temperature, accuracy, LR) |
| `panels_embedding.py` | B | CLOP embedding space UMAP (69 cell-type prototypes) |
| `panels_metrics.py` | D | Metrics summary dashboard (grouped bars, radar, gauges) |
| `panels_umap_quality.py` | E | Real vs generated cell overlay (type-coloured UMAP) |
| `panels_heatmaps.py` | F | Text-cell cosine similarity heatmap (69x69 matrix) |
| `panels_quality.py` | G | Per-type generation fidelity (centroid cosine, Frechet distance) |
| `panels_expression.py` | H, I | Gene expression correlation and decoder analysis |
| `panels_diversity.py` | J, K | Diversity diagnostics and expression diversity |
| `panels_conditioning.py` | L, M | Noise-scale trade-off and conditioning mode PCA |
| `panels_clustering.py` | N | Marker gene comparison (real vs generated heatmap) |
| `baseline_panels.py` | O | Baseline comparison (radar + improvement strip) |
| `panels_clustering.py` | P | Clustering alignment (UMAP overlay, kNN mixing, ARI/NMI) |
| `panels_classifier.py` | Q | Classifier alignment (confusion matrix, per-type accuracy, ROC) |
| `panels_de_concordance.py` | R | DE concordance (logFC scatter, concordance heatmap) |
| `benchmark_panels.py` | S | Model benchmarking (metrics heatmap, composite score, CI) |

## Orchestration

- `results_visualizer.py` -- `ResultsVisualizer`: central orchestrator that sequences all panel renders and produces the full report PDF. Run via `python -m src.visualization.results_visualizer`.
- `panels_merged.py` -- Composes individual panels into multi-panel composite figures for the article.
- `full_pipeline_figures.py` -- Generates all figures from a single pipeline invocation.

## Style System

`style.py` provides a unified visual identity:

- `VIS_STYLE` -- matplotlib rcParams dict following Nature/Cell conventions.
- `TYPE_PALETTE` -- 69 deterministic, colorblind-friendly colours for cell types.
- `COLORS` -- Named semantic colours for real/generated/baseline series.
- `apply_style()` -- Activate rcParams globally.
- `style_axes(ax, kind)` -- Per-subplot typography and spine cleanup.
- `save_panel(fig, path, dpi)` -- Save PNG + PDF in one call.

## Article Delivery

`article_delivery.py` verifies that rendered PDFs exist in `results/figures/`
and symlinks (or copies) them into `articles/figures/` for LaTeX inclusion.
It also defines the canonical display-figure grouping and article-component
manifest that the SciVCD sandbox reuses as its default figure-asset fixture
suite.

## Utilities

- `io.py` -- Safe JSON/numpy loading helpers for panel data.
- `panel_config.py` -- Per-panel configuration and feature flags.
- `report.py` -- Text report generation.
- `auto_refine.py` -- Automated figure refinement passes.
- `_utils.py` -- Internal plotting helpers shared across panels.
