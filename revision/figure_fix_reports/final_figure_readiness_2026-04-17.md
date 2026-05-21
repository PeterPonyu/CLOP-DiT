# Final Figure-Readiness Gate — CLOP-DiT Submission Packet

**Date:** 2026-04-17
**Auditor:** Claude (general-purpose agent)
**Scope:** All 25 article-manifest figures in `/home/zeyufu/Desktop/CLOP-DiT/results/figures/`
**Method:** Opened each PDF as a rendered image (built-in PDF-image support) and inspected layout, labels, legends, and fix-targeted regions.
**Rule:** BLOCKER = reviewer would reject; ADVISORY = cosmetic but ship-acceptable; PASS = ship-ready.

---

## Per-Figure Audit

| # | Figure | Verdict | Notes |
|---|--------|---------|-------|
| 1 | fig01a_architecture.pdf | PASS | Three-stage schematic clean. **Fix verified:** Shared Space dashed box (purple) sits cleanly between panels (a) and (b); arrows flow into DiT; `z_0` box is to its right with clear whitespace — no overlap with Shared Space label. Panel labels (a)(b)(c) present. All stage boxes legible. |
| 2 | fig01b_evaluation_pipeline.pdf | PASS | Five-tier evaluation flow diagram reads left-to-right and top-to-bottom. Legend at bottom (Real Data / Generated Data / Fitted Models / Transforms / Metrics) clear. Panel label (d) present. Arrows do not overlap text. |
| 3 | fig02a_training_dynamics.pdf | PASS | 8-panel (a)–(h) grid. All axes labeled, phase-band annotations (Phase I/II/III) present and non-overlapping. Legends clean. Log-scale Y on (e) rendered correctly. |
| 4 | fig02b_embedding_space.pdf | PASS | 6-panel (i)–(n) UMAP layout. Legend markers (circle=Real, triangle=Generated) in panel (n) clear. Cells-per-type bar chart (k) labels truncated with ellipsis but legible. |
| 5 | fig03a_metrics_summary.pdf | PASS | 4-panel (a)–(d) summary. Bar annotations crisp, log-scale axis on (d) clean. CI overlay in (c) via grey bars clear. |
| 6 | fig03b_per_type_fidelity.pdf | PASS | 3-panel (e)(f)(g). Leader-line annotations in scatter (g) are short and adjacent ("Kidney prox...", "Dorsal horn", "Smooth musc...", "Inhibitory..."). Long type labels truncated with ellipsis. |
| 7 | fig03c_text_cell_alignment.pdf | PASS | 3-panel (h)(i)(j). Heatmap (h) with highlighted sub-block; per-type alignment (i) with [0/0/69 poor] tally visible. Mann–Whitney test stats in (j) legible. |
| 8 | fig04a_marker_genes.pdf | PASS | 4-panel (a)–(d) marker gene comparison. Delta heatmap annotations (+0.06 etc.) visible on colored cells. Bar chart fold-change signs color-coded. |
| 9 | fig04b_expression_correlation.pdf | PASS | 4-panel (e)–(h). Scatter (e) annotations CFH/SHD/EXPH5/FGL1/U91319.1 positioned clearly. Residual histogram (h) summary stats in top-right. |
| 10 | fig05a_expression_analysis.pdf | PASS | 4-panel (a)–(d) variability panels. Annotation leader lines to EXPH5/LYPD6B/FST/ARSI/ZBED2/SHD in panel (a) are adjacent. Panel (d) std-ratio labels aligned right. |
| 11 | fig05b_conditioning_landscape.pdf | PASS | 7-panel (e)–(k) conditioning modes comparison. PC scatter panels clean; error bars in (f) visible; confusion heatmap (j) values legible on both bright and dark cells. |
| 12 | fig06_diversity_diagnostics.pdf | PASS | 5-panel (a)–(e). Dual-axis panel (e) with sweet-spot band legend clear. CFG sweep panel (c) dual-axis labels distinct (blue/red). |
| 13 | fig07a_expression_diversity.pdf | PASS | Simple 2-panel summary (a)(b). Ratio labels inline. Sparse but that's intentional — it's a summary figure. |
| 14 | fig07b_baseline_comparison.pdf | PASS | 3-panel (c)(d)(e) baseline comparison. Normalized scores dot-and-line style in (d) clean; advantage-bars in (e) with signed labels. |
| 15 | fig07c_benchmark.pdf | PASS | 4-panel (f)(g)(h)(i) multi-method benchmark. Heatmap (f) with blue-boxed CLOP-DiT highlights; composite score (g) dual-bar layout (All vs Common) clear; 95% CI dot plot (i) grouped by metric. |
| 16 | fig08a_downstream_validation.pdf | PASS | **Fix verified:** 6-panel (a)–(f). Panel (e) Per-Type P/R/F1 heatmap — zero-valued cells (top 9 rows, black background) show "0.00" in white text (stroke outline rendering); non-zero rows use standard label color. All text legible. Confusion matrix (d) color scale clean. ROC curve (f) AUC=0.656 readable. |
| 17 | fig08b_de_concordance.pdf | PASS | 3-panel (g)(h)(i). Scatter (g) with y=x reference and sign-disagreement marker clear. Concordance heatmap (h) with numeric annotations legible on both light and dark cells. |
| 18 | fig09a_variance_matching.pdf | PASS | **Fix verified:** 4-panel (a)–(d). Panel (d) SWD-vs-training-cells scatter: leader-line annotations for "Inhibitory GABAer...", "Dorsal horn senso...", "Neuroendocrine ar..." are SHORT and attached directly to their data points (upper-right outliers above the OLS fit band). No crossing lines. |
| 19 | fig09b_gene_gene_correlation.pdf | PASS | **Fix verified:** 4-panel (e)–(h). Panel (h) Preservation-vs-Heterogeneity scatter: leader-line annotations for "Alveolar type 2 (...", "Inhibitory GABAer...", "cytes are t..." are SHORT and positioned adjacent to their data points. OLS fit line clean. |
| 20 | figS01_supplementary_validation.pdf | PASS | 10-panel (a)–(j) supplementary validation battery. Pseudobulk r distribution (b) has tight x-range [0.99990, 0.99995] but ticks are clear. Quality-by-family stacked bars (c) with legend. Dimensional concentration (i) with Gini annotation. |
| 21 | figS02_expression_diagnostics.pdf | PASS | 4-panel (a)–(d) latent-space SWD diagnostics. Sample-size annotations (n=1,149 etc.) with leader lines in (a); box plot (b) with IQR band; ECDF (c) with median reference line. |
| 22 | figS_lane_c_zero_shot.pdf | PASS | **Fix verified:** Three vertically stacked sub-panels with clear (a) Kidney (blue), (b) Cerebellum (orange), (c) Fetal gonadal (green) panel labels in color-matched bold. Random-chance reference lines and accuracy values visible. |
| 23 | figS_lane_a1_knn_family_heatmap.pdf | PASS | New supplementary. KNN family confusion matrix with marginal within-family accuracy bar chart. Diagonal highlighted with yellow boxes; counts in white/dark text as needed. Title text "49.2% of cell-type errors are within-family" clear. |
| 24 | figS_lane_a2_organism_stratified.pdf | PASS | New supplementary. 6-panel (a)–(f) organism-stratified quality. Box plots Human/Mouse/Dual with jittered points. Mann–Whitney p-values annotated in upper-right of each panel. Legend at bottom. Panel (d) uses offset y-axis notation (+9.999e−1) which is correct for the narrow near-unity range. |
| 25 | figS_lane_b3_forced_scarcity.pdf | PASS | New supplementary. 2-panel (a)(b) forced-scarcity augmentation curves. Natural vs forced scarcity (solid vs dashed) with ceiling/baseline reference bands. Legend at bottom covers all 5 methods + 2 line styles. Panel titles (a) Megakaryocytes / (b) Ameloblasts clear. |

---

## Tally

- **PASS:** 25 / 25
- **ADVISORY:** 0 / 25
- **BLOCKER:** 0 / 25

## Verdict

**SHIP-READY**

All 25 figures render without presentation blockers. Each of the four just-applied fixes was visually verified:

1. **fig01a** — Shared Space dashed box no longer overlaps `z_0` label. Clean.
2. **fig08a(e)** — P/R/F1 heatmap cells legible (stroke outline working on both zero and non-zero rows).
3. **fig09a(d) and fig09b(h)** — Leader lines are short and adjacent to data points; no long diagonal leaders remain.
4. **figS_lane_c_zero_shot** — (a)(b)(c) panel labels present, color-matched to each dataset.

The three new supplementary figures (lane_a1, lane_a2, lane_b3) render correctly with readable labels, legends, and axis formatting.

**Packet is cleared for tagging and push.**
