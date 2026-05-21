# Final Figure Polish Read — 2026-04-17

**Scope:** Final polish read of all 25 article-manifest figures at `results/figures/`.
**Method:** Visual inspection of rendered PDF for each figure. Priority flags checked:
clipped text, overlapping labels, color-confusable palette without shape redundancy,
missing panel letters, caption/figure number mismatch.

## Top-line verdict

**BLOCKERS: 0.** Pipeline is submission-ready.

- PASS: 21
- ADVISORY: 4
- BLOCKER: 0

All 25 figures carry panel letters where multi-panel, no clipped titles/axes detected,
and legends/colorbars render inside their axes. Advisories listed below are minor
readability polish that do not block the revision but could be addressed in a future pass.

## Per-figure table

| # | Figure | Panel letters | Verdict | Notes |
|---|--------|---------------|---------|-------|
| 1 | fig01a_architecture | (a)(b)(c) | PASS | Three-stage schematic clean; legend row at base; colors match panel backgrounds. |
| 2 | fig01b_evaluation_pipeline | (d) | PASS | Single-panel schematic; flow graph readable; legend footer present. |
| 3 | fig02a_training_dynamics | (a)-(h) | PASS | 8-panel grid; phase shading labeled; no overlaps. |
| 4 | fig02b_embedding_space | (i)-(n) | ADVISORY | Cell-type labels in (k) truncated with ellipses (expected); UMAP scatter in (m)/(n) sparse — acceptable. "Cells per Type" tick labels are heavily truncated ("Kidney proximal tubule", "Conventional dendritic") — readable but tight. |
| 5 | fig03a_metrics_summary | (a)-(d) | PASS | Bars labeled with values; log-scale axis in (d) noted inline; legend present in (b). |
| 6 | fig03b_per_type_fidelity | (e)(f)(g) | ADVISORY | (g) annotation callouts "Smooth musc...", "Dorsal horn...", "Kidney prox...", "Inhibitory..." are truncated with ellipses and one overlaps the "Smooth musc..." label near the trend line; still legible, borderline. |
| 7 | fig03c_text_cell_alignment | (h)(i)(j) | PASS | Heatmap in (h) well-gridded; diagonal/off-diagonal stats present; Mann–Whitney annotation in (j) placed inside plot. |
| 8 | fig04a_marker_genes | (a)-(d) | PASS | Marker gene expression dense but clean; delta values labeled; legend and colorbars present. |
| 9 | fig04b_expression_correlation | (e)-(h) | PASS | Scatter callouts (CFH, SHD, EXPH5, FGL1) non-overlapping; colorbar present; residual histogram annotated. |
| 10 | fig05a_expression_analysis | (a)-(d) | PASS | CV scatter with annotated outliers (EXPH5, ARSI, etc.) with leader lines; divergent-genes barplot values labeled. |
| 11 | fig05b_conditioning_landscape | (e)-(k) | PASS | 7-panel compound; legends outside plot area; heatmap annotations readable. |
| 12 | fig06_diversity_diagnostics | (a)-(e) | PASS | CFG-vs-diversity twin-axis in (c) clearly color-mapped to axis labels; sweet-spot band in (e) shaded with legend. |
| 13 | fig07a_expression_diversity | (a)(b) | PASS | Simple 2-panel; values labeled; ratio=1 reference line present. |
| 14 | fig07b_baseline_comparison | (c)(d)(e) | PASS | Shape redundancy used in (d) (circle/diamond/square/triangle); improvement values annotated. |
| 15 | fig07c_benchmark | (f)-(i) | ADVISORY | Y-axis labels in (i) "95% Bootstrap CI Comparison" panel duplicate the model-name list 3× vertically without an obvious group separator — functional but visually dense. Heatmap (f) uses a diverging green→pink palette with blue outlined cells; blue outline acts as a shape/outline cue so not a color-confusability blocker. |
| 16 | fig08a_downstream_validation | (a)-(f) | PASS | UMAP overlay uses circle/triangle shape redundancy; confusion matrix on full 69×69 grid is dense but intentional; per-type P/R/F1 heatmap includes numeric overlays for accessibility. |
| 17 | fig08b_de_concordance | (g)(h)(i) | PASS | Heatmap cells numerically annotated; legend positioned in (i) not overlapping bars. |
| 18 | fig09a_variance_matching | (a)-(d) | PASS | Four-panel grid; training-cell callout labels in (d) non-overlapping; regression fit with CI band shown. |
| 19 | fig09b_gene_gene_correlation | (e)-(h) | PASS | Permuted null vs observed clearly colored (gray vs blue); best/worst heatmaps share colorbar style. |
| 20 | figS01_supplementary_validation | (a)-(j) | ADVISORY | 10-panel supplementary figure; panel (a) x-axis scale "0.0 0.5 1.0" is compressed because median is 1.000 and bars all saturate — cosmetic but could be rescaled. Panel (b) has a very narrow x-axis range (0.99990–0.99999) that is correct but uses adjacent ticks that look cramped; still readable. |
| 21 | figS02_expression_diagnostics | (a)-(d) | PASS | Per-type variance box + dim-wise correlation ECDF + SWD-vs-training-cells scatter; callout labels non-overlapping. |
| 22 | figS_lane_c_zero_shot | (a)(b)(c) | PASS | Three-subplot zero-shot figure; each subpanel labeled with tissue/atlas; dashed "random =" lines labeled. |
| 23 | figS_lane_a1_knn_family_heatmap | (a) | PASS | 12×12 family confusion matrix with accuracy bar strip; numeric cell values readable; within-family accuracy legend on right. |
| 24 | figS_lane_a2_organism_stratified | (a)-(f) | PASS | 6-panel Human/Mouse/Dual split; MW p-values annotated inline; legend row at base. |
| 25 | figS_lane_b3_forced_scarcity | (a)(b) | PASS | Two-panel rare-cell F1 across augmentation ratios; five methods distinguished by line color + shape (circle/square/triangle/diamond); natural-scarcity ceiling + forced-scarcity baseline bands labeled. |

## Advisory detail

1. **fig02b_embedding_space (Fig 2b, lower panel).** "Cells per Type" horizontal bar
   labels are truncated to ~20 chars with ellipses. Because the same label set appears
   as the colored legend below (l)(m)(n), truncation is tolerable. Not a blocker.

2. **fig03b_per_type_fidelity panel (g).** Three callout labels near the lower-right
   of the Fidelity-vs-Abundance scatter (`Dorsal horn…`, `Smooth musc…`,
   `Kidney prox…`) sit close to each other with leader lines; `Smooth musc…` label
   brushes the trend curve. Borderline readable; not a blocker.

3. **fig07c_benchmark panel (i).** Y-axis in the 95% Bootstrap CI panel repeats the
   model-name list three times (one per metric group). A group header or divider would
   help navigation but does not obscure values.

4. **figS01_supplementary_validation panels (a) and (b).** Pseudobulk correlation
   distribution is (correctly) clustered at r≈1.000, which compresses visual variance.
   Content is accurate; a zoomed subplot would be a nice-to-have but is not required.

## Priority-flag sweep

| Flag | Observed? | Details |
|------|-----------|---------|
| Clipped text | No | Some ellipsis-truncated cell-type labels (expected with long taxonomy names); no axis titles or legends clipped. |
| Overlapping labels | No | Minor near-overlap in fig03b (g) callouts — flagged as advisory. |
| Color-confusable palette w/o shape redundancy | No | Multi-series figures (fig02b, fig07b, fig08a, figS_lane_b3) all use shape markers in addition to color. Diverging heatmaps use monotonic luminance ramps. |
| Missing panel letters | No | Every multi-panel figure carries (a),(b),… Single-schematic figures carry a single panel letter (fig01b=(d), figS_lane_a1=(a), figS_lane_a2=(a)-(f), figS_lane_b3=(a)(b)). |
| Caption/figure number mismatch | No | Filenames match article manifest in `src/visualization/article_delivery.py`. Internal labels (Stage 1/2/3 in fig01a, "Evaluation Pipeline Schematic" in fig01b, etc.) are self-consistent. |

## Final tally

- **PASS: 21 / 25**
- **ADVISORY: 4 / 25** (fig02b, fig03b, fig07c, figS01)
- **BLOCKER: 0 / 25**

No source-script edits required. All figures cleared for submission.
