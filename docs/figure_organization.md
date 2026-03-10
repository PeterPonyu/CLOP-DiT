# Figure Organization

*Last revised: 2026-03-10*

All current figures are produced by the report pipeline and live under **`results/figures/`**. Regenerate with:

```bash
bash scripts/pipeline/regenerate_report.sh
```

## Module architecture

Each article figure has its own module named `fig{NN}_{name}.py` in `src/visualization/`. Old module names (`panels_*.py`, `*_panels.py`) are backward-compat shims that re-export from the new modules.

| Module | Article Fig | Description |
|--------|------------|-------------|
| *(external)* `scripts/analysis/generate_architecture_figure.py` | Fig 1 | Architecture overview |
| *(external)* `scripts/analysis/evaluation_pipeline_figure.py` | Fig 2 | Evaluation pipeline schematic |
| `fig03_training.py` | Fig 3 | Training dynamics (CLOP + DiT merged) |
| `fig04_embedding.py` | Fig 4 | Embedding space (CLOP alignment + real vs gen) |
| `fig05_metrics.py` | Fig 5 | Metrics summary dashboard |
| `fig06_fidelity.py` | Fig 6 | Per-type generation fidelity |
| `fig07_alignment.py` | Fig 7 | Text-cell alignment heatmap |
| `fig08_markers.py` | Fig 8 | Marker gene comparison |
| `fig09_expression_corr.py` | Fig 9 | Expression correlation |
| `fig10_expression_analysis.py` | Fig 10 | Expression analysis |
| `fig11_conditioning.py` | Fig 11 | Conditioning landscape |
| `fig12_diversity.py` | Fig 12 | Diversity diagnostics |
| `fig13_noise_tradeoff.py` | Fig 13 | Noise-scale trade-off |
| `fig14_expr_diversity.py` | Fig 14 | Expression diversity |
| `fig15_baselines.py` | Fig 15 | Baseline comparison |
| `fig16_benchmark.py` | Fig 16 | Composite benchmark |
| `fig17_downstream.py` | Fig 17 | Downstream clustering + classifier |
| `fig18_de_concordance.py` | Fig 18 | DE concordance |
| *(external)* `scripts/analysis/variance_matching_pilot.py` | Fig 19 | Variance matching pilot |
| *(external)* `scripts/analysis/gene_gene_correlation.py` | Fig 20 | Gene-gene correlation |

### Shared infrastructure

- `style.py` — centralized style, `save_with_vcd`, palettes, constants
- `_plot_helpers.py` — shared subplot renderers (UMAP overlay, confusion matrix, ROC)
- `_utils.py` — pairwise cosine helper
- `report.py` — PDF combination
- `article_delivery.py` — canonical 20-figure manifest with source→article name mapping

### Backward-compat shims

Old module names still work via re-exports with `DeprecationWarning`:

| Old module | Redirects to |
|-----------|-------------|
| `panels_training.py` | `fig03_training` |
| `panels_embedding.py` | `fig04_embedding` |
| `panels_umap_quality.py` | `fig04_embedding` |
| `panels_merged.py` | `fig04_embedding` |
| `panels_metrics.py` | `fig05_metrics` |
| `panels_heatmaps.py` | `fig06_fidelity` + `fig07_alignment` |
| `panels_expression.py` | `fig08_markers` + `fig09_expression_corr` + `fig10_expression_analysis` |
| `panels_conditioning.py` | `fig11_conditioning` + `fig13_noise_tradeoff` |
| `panels_diversity.py` | `fig12_diversity` + `fig14_expr_diversity` |
| `baseline_panels.py` | `fig15_baselines` |
| `benchmark_panels.py` | `fig16_benchmark` |
| `downstream_panels.py` | `fig17_downstream` + `fig18_de_concordance` |
| `panels_clustering.py` | `fig17_downstream` |
| `panels_classifier.py` | `fig17_downstream` |
| `panels_de_concordance.py` | `fig18_de_concordance` |
| `panels_quality.py` | `fig04_embedding` + `fig05_metrics` + `fig06_fidelity` + `fig07_alignment` |

## MDPI Biology article figure map

All 20 figures are included in `articles/clop_dit_biology.tex`. Each figure is a **standalone full-width** `\includegraphics[width=\textwidth]`. All layout is handled inside Matplotlib.

| Article Figure | Source filename | Article symlink | Producer |
|----------------|----------------|-----------------|----------|
| Fig 1 | `fig_architecture` | `fig01_architecture` | `scripts/analysis/generate_architecture_figure.py` |
| Fig 2 | `fig_evaluation_pipeline` | `fig02_evaluation_pipeline` | `scripts/analysis/evaluation_pipeline_figure.py` |
| Fig 3 | `fig03_training_dynamics` | `fig03_training_dynamics` | `fig03_training.plot_training_dynamics_combined()` |
| Fig 4 | `fig04_embedding_space` | `fig04_embedding_space` | `fig04_embedding.plot_embedding_space_merged()` |
| Fig 5 | `fig05_metrics_summary` | `fig05_metrics_summary` | `fig05_metrics.plot_metrics_summary()` |
| Fig 6 | `fig06_per_type_fidelity` | `fig06_per_type_fidelity` | `fig06_fidelity.plot_per_type_generation()` |
| Fig 7 | `fig07_text_cell_alignment` | `fig07_text_cell_alignment` | `fig07_alignment.plot_text_cell_heatmap()` |
| Fig 8 | `fig08_marker_genes` | `fig08_marker_genes` | `fig08_markers.plot_marker_gene_comparison()` |
| Fig 9 | `fig09_expression_correlation` | `fig09_expression_correlation` | `fig09_expression_corr.plot_expression_correlation()` |
| Fig 10 | `fig10_expression_analysis` | `fig10_expression_analysis` | `fig10_expression_analysis.plot_expression_analysis()` |
| Fig 11 | `fig11_conditioning_umap` | `fig11_conditioning_umap` | `fig11_conditioning.plot_panel_m()` |
| Fig 12 | `fig12_diversity_diagnostics` | `fig12_diversity_diagnostics` | `fig12_diversity.plot_diagnostics()` |
| Fig 13 | `fig13_noise_tradeoff` | `fig13_noise_tradeoff` | `fig13_noise_tradeoff.plot_panel_l()` |
| Fig 14 | `fig14_expression_diversity` | `fig14_expression_diversity` | `fig14_expr_diversity.plot_expression_diversity_panel()` |
| Fig 15 | `fig15_baseline_comparison` | `fig15_baseline_comparison` | `fig15_baselines.plot_baseline_comparison()` |
| Fig 16 | `fig16_benchmark` | `fig16_benchmark` | `fig16_benchmark.plot_benchmark_panel()` |
| Fig 17 | `fig17_downstream_pq` | `fig17_downstream_pq` | `fig17_downstream.plot_clustering_and_classifier_merged()` |
| Fig 18 | `fig18_de_concordance` | `fig18_de_concordance` | `fig18_de_concordance.plot_de_concordance_panel()` |
| Fig 19 | `fig19_variance_matching_pilot` | `fig19_variance_matching_pilot` | `scripts/analysis/variance_matching_pilot.py` |
| Fig 20 | `fig20_gene_gene_correlation` | `fig20_gene_gene_correlation` | `scripts/analysis/gene_gene_correlation.py` |

## Naming convention

- `fig{NN}_{short_name}.png` / `.pdf` — article figures (numbered to match LaTeX)
- `fig_architecture.png` / `.pdf` — architecture diagram (external script)
- `fig_evaluation_pipeline.png` / `.pdf` — evaluation pipeline (external script)
- Article symlinks: `articles/figures/fig{NN}_{name}.pdf` -> `../../results/figures/*.pdf`
- Mapping from article names to source names is defined in `src/visualization/article_delivery.py` (`_SOURCE_MAP`)

## Figure generation policy

Each panel is a **standalone full-width figure** in the LaTeX article. LaTeX does **not** compose multiple panels into a single figure area. All subplot layout is handled entirely within Matplotlib.

**Save and presentation policy:**
- **Single save path:** All figure saves go through `style.save_with_vcd()`. No raw `fig.savefig()` for article figures.
- **Default output:** Figure generators resolve `output_dir=None` to `FIG_DIR`, so direct runs and pipeline runs share the same central path.

**Layout principles:**
1. **Standalone figures**: Every `\begin{figure}` in LaTeX contains exactly one `\includegraphics[width=\textwidth]`.
2. **No panel labels in content**: Subplot titles use descriptive text only. Figure numbers appear only in the LaTeX caption.
3. **Font sizes**: Base font sizes in `style.py` are calibrated for LaTeX full-width figures. Minimum effective size is 5.5pt per VCD policy.
4. **Tighter layout**: `GRIDSPEC_TIGHT` in `style.py` is the compact multi-row spacing preset.

## Article delivery

After figure regeneration, run:

```bash
python -m src.visualization.article_delivery
```

This verifies all 20 PDFs exist in `results/figures/` and creates symlinks in `articles/figures/`. The canonical manifest lives in `src/visualization/article_delivery.py` (`ARTICLE_FIGURE_BASENAMES`). To add a new article figure, update that list and ensure a producer writes the PDF.
