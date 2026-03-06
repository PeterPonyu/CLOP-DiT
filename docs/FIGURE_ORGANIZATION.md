# Figure Organization

*Last revised: 2026-03-06*

All current figures are produced by the report pipeline and live under **`results/figures/`**. Regenerate with:

```bash
bash scripts/regenerate_report.sh
```

Outputs: 19 panels (A-S) as PNG and PDF, plus `fig_architecture.{png,pdf}` and 5 merged figures, plus `clop_dit_full_report.pdf`.

## Panel list (A-S) and merged figures

| Panel | Filename | Description | MDPI Article Figure |
|-------|----------|-------------|---------------------|
| Arch | `fig_architecture` | Pipeline architecture diagram | Fig 1 (architecture overview) |
| A+C | `fig_training_dynamics` | **Merged**: CLOP + DiT training dynamics | Fig 2 (training dynamics) |
| A | `panel_a_clop_training` | CLOP contrastive loss and validation metrics | (standalone, legacy) |
| B+E | `fig_embedding_space` | **Merged**: CLOP UMAP + Real vs Generated | Fig 3 (embedding space) |
| B | `panel_b_clop_embedding_umap` | CLOP embedding UMAP | (standalone, legacy) |
| C | `panel_c_dit_training` | DiT flow matching loss and velocity cosine | (standalone, legacy) |
| D | `panel_d_metrics_summary` | KNN, steering, diversity, metrics dashboard | Fig 4 (metrics dashboard) |
| E | `panel_e_real_vs_generated` | Real vs generated cells (PCA/embedding) | (standalone, legacy) |
| G+F | `fig_fidelity_alignment` | **Merged**: Per-type fidelity + text-cell heatmap | Fig 5 (fidelity & alignment) |
| F | `panel_f_text_cell_heatmap` | Text-cell similarity heatmap | (standalone, legacy) |
| G | `panel_g_per_type_generation` | Per-type generation overview | (standalone, legacy) |
| N | `panel_n_marker_gene_comparison` | Marker gene comparison | Fig 6 (marker genes) |
| H | `panel_h_expression_correlation` | Expression/dimension correlation | Fig 7 (expression correlation) |
| I | `panel_i_expression_analysis` | Expression analysis | Fig 8 (expression analysis) |
| M | `panel_m_conditioning_umap` | Conditioning UMAP | Fig 9 (conditioning landscape) |
| J | `panel_j_diversity_diagnostics` | Diversity diagnostics | Fig 10 (diversity diagnostics) |
| L+K | `fig_diversity_tradeoff` | **Merged**: Noise tradeoff + expression diversity | Fig 11 (diversity & tradeoff) |
| K | `panel_k_expression_diversity` | Expression diversity | (standalone, legacy) |
| L | `panel_l_noise_tradeoff` | CFG/noise trade-off, ODE steps | (standalone, legacy) |
| O | `panel_o_baseline_comparison` | Baseline/decoder comparison | Fig 12 (baselines) |
| S | `panel_s_benchmark` | Composite benchmark (0.668) | Fig 13 (benchmark) |
| P+Q | `fig_downstream_pq` | **Merged**: Clustering + Classifier alignment | Fig 14 (downstream P+Q) |
| P | `panel_p_clustering_mixing` | Clustering/mixing | (standalone, legacy) |
| Q | `panel_q_classifier_alignment` | Classifier alignment | (standalone, legacy) |
| R | `panel_r_de_concordance` | DE concordance | Fig 15 (DE concordance) |

## MDPI Biology article figure map

All 15 figures are included in `articles/clop_dit_biology.tex`. Each figure is a **standalone full-width** `\includegraphics[width=\textwidth]` — no `\subfloat` or multi-panel LaTeX composition. All layout is handled inside Matplotlib. Only the panels used in the article have symlinks in `articles/figures/`; legacy standalone panels are generated for the full report only.

| Article Figure | Panel(s) | Layout | Section |
|----------------|----------|--------|---------|
| Fig 1 | Arch | Full-width | Architecture Overview (Methods) |
| Fig 2 | A+C | Full-width merged | Training Dynamics (CLOP top, DiT bottom) |
| Fig 3 | B+E | Full-width merged | Embedding Space (CLOP alignment top, Real vs Gen bottom) |
| Fig 4 | D | Full-width | Core Evaluation Metrics |
| Fig 5 | G+F | Full-width merged | Per-Type Fidelity & Text-Cell Alignment |
| Fig 6 | N | Full-width | Marker Gene Comparison |
| Fig 7 | H | Full-width | Expression Correlation |
| Fig 8 | I | Full-width | Expression Analysis |
| Fig 9 | M | Full-width | Conditioning Landscape |
| Fig 10 | J | Full-width | Diversity Diagnostics |
| Fig 11 | L+K | Full-width merged | Noise Trade-Off & Expression Diversity |
| Fig 12 | O | Full-width | Baselines |
| Fig 13 | S | Full-width | Benchmark |
| Fig 14 | P+Q | Full-width merged | Downstream: Clustering + Classifier |
| Fig 15 | R | Full-width | Downstream: DE Concordance |
| Table 1 | -- | 7 rows | Core evaluation results |
| Table A1 | -- | Architecture specs | Appendix A |
| Table A2 | -- | CFG sweep (8 scales) | Appendix B |

## JBHI markdown mapping

`docs/CLOP_DiT_JBHI_Article.md` uses `../results/figures/panel_*.png`:

- Fig 2: A, C
- Fig 3: E
- Fig 4: D
- Fig 5: F, N
- Fig 6: J, L
- Fig 7: H
- Fig 8: D (tables in text)
- Fig 9: O
- Fig 10: L

## Folder contents

- **`results/figures/`**: All pipeline outputs — `fig_architecture`, 5 merged figures (`fig_training_dynamics`, etc.), 19 standalone panels (`panel_a_*` … `panel_s_*`), and `clop_dit_full_report.pdf`. Legacy panels (A, B, C, E, F, G, K, L, P, Q) are used only in the full report; the article uses the 15 listed in the table above. **Outdated files** (from old scripts): `fig1_training_dynamics.*`, `fig2_embedding_space.*`, `fig3_metrics_dashboard.*`, `fig4_biological_validation.*`, `fig5_dimension_sampling.*`, `visual_conflict_report.json` — remove with `bash scripts/remove_outdated_figures.sh`.
- **`articles/figures/`**: Only the **15 article figure PDFs** (symlinks to `results/figures/`). No other figures belong here; the LaTeX build includes only these 15.
- **`results/figures/biological_validation/`** (optional): Figure set from `scripts/08_biological_validation.py` (`figure1_text2cell_multi`, `figure2_cell2cell`, `figure3_celltypist`, `figure4_summary`) used for supplementary biological QA and reviewer support.

## Naming convention

- `panel_{letter}_{short_name}.png` / `.pdf` — standalone panels (legacy ones used only in full report)
- `fig_{descriptive_name}.png` / `.pdf` — merged figures used in the article
- `fig_architecture.png` / `.pdf` for the architecture diagram
- Letter order A-S matches the full report layout
- Article symlinks: `articles/figures/{panel,fig}_*.pdf` -> `../../results/figures/*.pdf` (only for the 15 figures used in the article)

## Figure generation policy

Each panel is a **standalone full-width figure** in the LaTeX article. LaTeX does **not** compose multiple panels into a single figure area. All subplot layout is handled entirely within Matplotlib.

**Save and presentation policy:**
- **Single save path:** All figure saves go through `style.save_with_vcd()` (or the caller’s `save_panel_fn` that ultimately uses it). No raw `fig.savefig()` for article figures.
- **Auxiliary figure policy:** Biological validation figures use the same central style and `save_with_vcd()` path as article/report figures.
- **Merged figures:** Composited figures (G+F, B+E, L+K, P+Q) either call `save_panel_fn(fig, path, dpi)` when invoked from the visualizer, or `save_with_vcd(fig, path, dpi)` when run standalone. This keeps VCD and PNG+PDF behaviour consistent.
- **Default output:** Standalone panel and figure helpers resolve `output_dir=None` to `FIG_DIR`, so direct runs and pipeline runs share the same central path configuration.

1. **Standalone figures**: Every `\begin{figure}` in LaTeX contains exactly one `\includegraphics[width=\textwidth]`. No `\subfloat` or side-by-side arrangement.
2. **No panel labels in content**: Subplot titles use descriptive text only (e.g. "Loss Convergence"). **Figure numbers (Figure 1–15) and subpanel labels (\textbf{a}, \textbf{b}) appear only in the LaTeX caption**, not in the figure image. The pipeline uses internal names (panels A–S) for code and the full report; the same 15 PDFs are referenced in the article as Figure 1–15.
3. **Legend vs. caption**: Legends contain only series/keys (what each curve or color means). Statistical summaries (r, p, CIs, sign agreement, etc.) go in the figure caption or in a short in-figure annotation, not in legend titles. See [LEGEND_CAPTION_POLICY.md](LEGEND_CAPTION_POLICY.md). If a legend has many entries (>6–8), use smaller font, `ncol`, or "key types only; full list in caption".
4. **Layout-aware long labels**: When y-labels are long (cell type names > 20 chars), panel code automatically reduces subplots per row (from 3 → 2 or 1×3 → 2×2) for readability.
5. **Font sizes**: Base font sizes in `style.py` are calibrated for LaTeX full-width figures. Minimum effective size is 5.5pt per VCD policy.
6. **VCD + auto-refine**: `save_with_vcd` runs visual-conflict detection before saving. `auto_refine.py` uses weighted scoring and category-specific action appliers to react to issues when enabled.
    - Layer 1 (subplot): per-axes legend/colorbar checks
    - Layer 2 (figure): text overlaps, truncation, artist overlap, spillover, font/label density
    - Layer 3 (perceptual): WCAG contrast, colorblind safety, error-bar visibility, precision
    - Layer 4 (semantic): overplotting, log-scale sanity, scale consistency, significance markers
7. **Tighter layout**: `GRIDSPEC_TIGHT` in `src/visualization/style.py` is the compact multi-row spacing preset. Final margins are applied by `save_with_vcd()`, including extra bottom space for figure-level legends when needed.
8. **Architecture figure**: Generated by `scripts/generate_architecture_figure.py`. Placed as Figure 1 in the article.
9. **Report vs article (presentation)**: Article-facing panels optimize for print clarity and minimal annotation; report-only panels may carry more diagnostics. Do not let report-style density leak into article figures. See [FIGURE_PRESENTATION_POLICY.md](FIGURE_PRESENTATION_POLICY.md) for annotation budget, primary message per panel, and text density ceiling.
10. **Figures 9–12**: For regeneration, presentation, and Fig 11 violin dependencies, see [FIGURES_9-12_POLICY.md](FIGURES_9-12_POLICY.md).
11. **Figures 12–15**: For regeneration, presentation, enhancements, and submission fixes, see [FIGURES_12-15_POLICY.md](FIGURES_12-15_POLICY.md).

## Legacy/outdated figure outputs (do not use; remove if present)

The **canonical pipeline** is `bash scripts/regenerate_report.sh`. It does **not** produce the following. If they exist, they come from the legacy `scripts/10_full_pipeline.py` (Stage 4) or older scripts and should be removed:

| Outdated file(s) | Source | Action |
|------------------|--------|--------|
| `fig1_training_dynamics.png`, `fig2_embedding_space.png`, `fig3_metrics_dashboard.png`, `fig4_biological_validation.png`, `fig5_dimension_sampling.png` | `full_pipeline_figures.py` via `10_full_pipeline.py` | Remove; article uses `fig_training_dynamics`, `fig_embedding_space`, etc. from results_visualizer. |
| `visual_conflict_report.json` | `10_full_pipeline.py` Stage 5 | Remove; VCD runs per-figure in `save_with_vcd()`. |

`src/visualization/full_pipeline_figures.py` is now a no-op compatibility stub by design. It should not be used to produce any figure artifacts.

Run `bash scripts/remove_outdated_figures.sh` to delete these from `results/figures/` without touching current report figures.

## Report vs article

The full report produces all 19 standalone panels + 5 merged figures. The article uses the 5 merged figures plus selected standalone panels (D, H, I, J, M, N, O, R, S) — 15 figures total. Only these 15 files are symlinked in `articles/figures/`. Legacy standalone panels (A, B, C, E, F, G, K, L, P, Q) are still generated for the full report PDF but are not referenced by the LaTeX article.

## Canonical producers

Each article figure has exactly one canonical producer. No duplicate logic writes the same filename from two code paths.

| Article Fig | Output filename | Producer | Pipeline step |
|-------------|----------------|----------|---------------|
| Fig 1 | `fig_architecture` | `scripts/generate_architecture_figure.py` | Step 0 |
| Fig 2 | `fig_training_dynamics` | `training_panels.plot_training_dynamics_combined()` | Step 7 |
| Fig 3 | `fig_embedding_space` | `panels_quality.plot_embedding_space_merged()` | Step 7 |
| Fig 4 | `panel_d_metrics_summary` | `panels_quality.plot_metrics_summary()` | Step 7 |
| Fig 5 | `fig_fidelity_alignment` | `panels_quality.plot_fidelity_and_alignment_merged()` | Step 7 |
| Fig 6 | `panel_n_marker_gene_comparison` | `panels_expression.plot_marker_gene_comparison()` | Step 7 |
| Fig 7 | `panel_h_expression_correlation` | `panels_expression.plot_expression_correlation()` | Step 7 |
| Fig 8 | `panel_i_expression_analysis` | `panels_expression.plot_expression_analysis()` | Step 7 |
| Fig 9 | `panel_m_conditioning_umap` | `scripts/conditioning_analysis.py` | Step 4 |
| Fig 10 | `panel_j_diversity_diagnostics` | `scripts/diversity_diagnostics.py` | Step 3 |
| Fig 11 | `fig_diversity_tradeoff` | `ResultsVisualizer._compose_diversity_tradeoff()` (tiles L+K) | Step 7 |
| Fig 12 | `panel_o_baseline_comparison` | `baseline_panels.plot_baseline_comparison()` | Step 7 |
| Fig 13 | `panel_s_benchmark` | `benchmark_panels.plot_benchmark_panel()` | Step 7 |
| Fig 14 | `fig_downstream_pq` | `downstream_panels.plot_clustering_and_classifier_merged()` | Step 7 |
| Fig 15 | `panel_r_de_concordance` | `downstream_panels.plot_de_concordance_panel()` | Step 7 |

After regeneration, run `bash scripts/verify_article_figures.sh` to confirm all 15 PDFs exist and update symlinks in `articles/figures/`. The script delegates to the Python delivery module; the **canonical list** of 15 article figure basenames lives in `src/visualization/article_delivery.py` (`ARTICLE_FIGURE_BASENAMES`). To add a new article figure, update that list, ensure one producer writes the PDF to `results/figures/`, and re-run the pipeline and delivery.

## Figure logic and limitations

This section states how figures are produced, where they live, and what is fixed vs configurable. Maintainers and reviewers can rely on it for reproducibility and expectations.

### Data flow (logic)

- **Pipeline:** `scripts/regenerate_report.sh` runs steps 0–8. Scripts and `src/visualization/` (e.g. `training_panels`, `panels_quality`, `panels_expression`, `downstream_panels`, `benchmark_panels`, `conditioning_analysis.py`, `diversity_diagnostics.py`) write outputs to `results/figures/`.
- **Single producer per figure:** Each of the 15 article figures has exactly one canonical producer (table above). No duplicate code path writes the same filename.
- **Symlinks:** `scripts/verify_article_figures.sh` calls `python -m src.visualization.article_delivery`, which reads the manifest in `src/visualization/article_delivery.py`, expects the 15 PDFs in `results/figures/` (or `FIG_DIR`), and (re)creates symlinks in `articles/figures/` (or `ARTICLE_FIGURES_DIR`). LaTeX includes only from `articles/figures/`; that directory is for consumption only (no hand-edited figures there).

### Paths

- **Default output:** Panel and figure generators resolve `output_dir=None` to `FIG_DIR` in `src/utils/paths.py`, which respects `CLOPDIT_FIG_DIR` (and `CLOPDIT_RESULTS_DIR`) so the output directory can be overridden at runtime.
- **Article figures:** `articles/figures/` contains only symlinks (or copies) to `results/figures/` for the 15 article figures. Do not edit PDFs in `articles/figures/`; regenerate and re-run `verify_article_figures.sh` instead.

### Explicit limitations

| Limitation | Description |
|------------|-------------|
| **VCD** | Panel S heatmap text uses explicit dark color for WCAG contrast. Run full regeneration and VCD to confirm 0 warnings; info-level issues are documented and accepted. |
| **Pipeline order** | Steps must run in order; some figures depend on earlier steps (e.g. Fig 11 is composed from panels L and K produced in steps 4 and 3, plus a diversity-tail violin that requires `diversity_diagnostics.json`, dedup caches, and generated embeddings; see [FIGURES_9-12_POLICY.md](FIGURES_9-12_POLICY.md)). |
| **Layout** | All multi-panel layout is done in Matplotlib. LaTeX uses a single `\includegraphics[width=\textwidth]` per figure — no `\subfloat` or LaTeX-composed subpanels. |
| **Symlinks** | Article build assumes `articles/figures/*.pdf` resolve (symlinks or copies). `verify_article_figures.sh` is the single place that creates/updates them. |
| **Legacy panels** | Standalone panels A, B, C, E, F, G, K, L, P, Q are still generated for the full report PDF but are not used in the LaTeX article. |

## How figures support the claims

- **Fig 1**: Pipeline overview (CLOP → DiT → Decoder).
- **Fig 2**: Training convergence for both stages (CLOP top row, DiT bottom row) — merged A+C.
- **Fig 3**: Embedding space analysis (CLOP alignment top, real vs generated bottom) — merged B+E.
- **Fig 4**: Metrics dashboard (KNN, steering, diversity, expression fidelity).
- **Fig 5**: Per-type fidelity and text–cell alignment — merged G+F.
- **Fig 6–8**: Biological and expression validation (marker genes, expression correlation, expression analysis).
- **Fig 9**: Conditioning landscape (UMAP of condition modes) — establishes context for the diversity analysis.
- **Fig 10**: Diversity diagnostics (intra-type ratio, memorization, CFG sweep, condition sensitivity).
- **Fig 11**: Noise trade-off and expression diversity — merged L+K.
- **Fig 12–13**: Baselines and composite benchmark (CLOP-DiT vs baselines, composite score 0.668).
- **Fig 14**: Downstream biology: clustering + classifier — merged P+Q.
- **Fig 15**: Downstream biology: DE concordance.

Together they support: (1) two-stage training, (2) text-conditioned generation quality (KNN 37× random, steering 81%), (3) biological fidelity (marker genes, expression, DE), (4) advantage over baselines, (5) utility in downstream workflows.

## Visual Conflict Detection (VCD) architecture

The VCD (`scripts/vcd/`) is a 30-pass, 4-layer modular detection system that runs automatically during figure generation via `save_with_vcd`. It detects layout, perceptual, and semantic issues before they reach the final manuscript.

### Package structure

| Module | Passes | Purpose |
|--------|--------|---------|
| `vcd_core.py` | — | Geometry helpers, `_ArtistInfo`, `_collect_artists` |
| `vcd_config.py` | — | Centralized thresholds (all 30 passes) |
| `vcd_checks_text.py` | 1, 5, 8, 9 | Text overlaps, artist overlap, spillover, panel labels |
| `vcd_checks_artists.py` | 2–4, 6–7 | Truncation, content overlap, axes overflow, scatter clip |
| `vcd_checks_legend.py` | 10–13, 15, 18 | Legend spillover, occlusion, crowding |
| `vcd_checks_colorbar.py` | 14, 17 | Colorbar internal, data overlap |
| `vcd_checks_structure.py` | 16, 19–22 | Significance brackets, font adequacy, tick-spine, font policy, label density |
| `vcd_checks_perceptual.py` | 23–26 | WCAG contrast, CVD safety, error-bar visibility, precision excess |
| `vcd_checks_semantic.py` | 27–30 | Overplotting, log-scale sanity, scale consistency, floating significance |
| `vcd_policy.py` | — | `FigurePolicy` dataclass, helper functions |
| `vcd_actions.py` | — | 24 issue-to-action mappings, `Action` dataclass, `diagnose()` |

### 4-layer detection

| Layer | Passes | Scope | Examples |
|-------|--------|-------|----------|
| 1 — Subplot | 12–13, summary | Per-axes | Legend occluding data within one subplot |
| 2 — Figure | 1–11, 14–22 | Cross-axes | Text overlap, truncation at border, label density |
| 3 — Perceptual | 23–26 | Readability | WCAG contrast < 3.0:1, colorblind-confusable pairs |
| 4 — Semantic | 27–30 | Data correctness | Overplotted scatter, log-scale with non-positive, orphaned significance |

### Weighted scoring

The auto-refine loop (`src/visualization/auto_refine.py`) uses weighted issue scores for optimization:

| Priority | Weight | Issue types |
|----------|--------|-------------|
| Integrity | 8–10 | `text_truncation`, `log_scale_nonpositive`, `floating_significance`, `label_density_excess` |
| Readability | 5–7 | `low_contrast_text`, `fontsize_too_small`, `text_overlap`, `tick_spine_overlap`, `font_family_violation` |
| Clarity | 2–3 | `overplotted_scatter`, `legend_data_occlusion`, `colorblind_confusable`, `precision_excess` |
| Minor | 0.5–1 | `axes_overflow`, `bold_usage`, `log_scale_unlabelled` |

Warning-severity issues use weight × 1.0; info-severity uses weight × 0.3.

### Action categories

Actions generated from issues are applied by category-specific functions in `auto_refine.py`:

| Function | Actions | Examples |
|----------|---------|----------|
| `_apply_figure_actions` | Layout | `increase_figsize`, `increase_margins` |
| `_apply_axis_density_actions` | Tick/label | `reduce_tick_labels`, `rotate_labels` |
| `_apply_legend_actions` | Legend | `move_legend`, `shrink_legend_font` |
| `_apply_structural_actions` | Grid | `reduce_subplots_per_row` |
| `_apply_perceptual_actions` | Semantic | `reduce_alpha`, `fix_cvd_palette`, `use_density_viz` |

### Severity levels

- **warning**: Must be fixed before submission. Indicates text overlaps, truncation, low contrast, or font sizes that would be illegible in print.
- **info**: Acceptable structural artifacts. Includes axes overflow from spines, legend-data occlusion within the same subplot, overplotting hints, and precision suggestions.

### Key thresholds (from `vcd_config.py`)

| Threshold | Value | Pass | Purpose |
|-----------|-------|------|---------|
| `MIN_TEXT_CONTRAST` | 3.0 | 23 | WCAG 2.0 text contrast ratio |
| `MIN_CVD_DISTANCE` | 10.0 | 24 | CIE76 ΔE under simulated deuteranopia |
| `OVERPLOT_OPAQUE_THRESHOLD` | 2000 | 27 | Scatter points with α≥0.5 before flagging |
| `SCALE_RANGE_SPREAD_FACTOR` | 3.0 | 29 | Max range ratio for same-label axes |
| `SIGNIFICANCE_PROXIMITY_PX` | 50.0 | 30 | Max distance from star to nearest data artist |
| `LABEL_DENSITY_THRESHOLD` | 0.92 | 22 | Max label footprint / axis extent ratio |
| `COMPOSED_SCALE` | 0.95 | 19 | Font downscaling factor for full-width LaTeX |
| `MIN_PT` | 5.5 | 19 | Minimum effective font size after scaling |

### Calibration (12-panel test)

After threshold tuning on all 12 generated panels (A, C, D, E, F, G, H, I, N, O, R, S):

- **Warnings**: Panel S heatmap was updated to use explicit dark text (`#1a1a1a`); re-run pipeline to confirm 0 warnings.
- **Info**: ~160 total (mostly `text_artist_overlap`, `legend_data_occlusion` from tightly composed panels)
- **False positives eliminated**: Heatmap annotations excluded from contrast checks; overplot threshold raised from 500→2000 to avoid flagging readable gene scatter plots

### Target: 0 warnings for all panels

After the layout restructuring and Panel S heatmap contrast fix (explicit dark text), run full regeneration in the intended environment to confirm 0 VCD warnings. Info-level detections are structural and acceptable.
