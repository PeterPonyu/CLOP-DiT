# Visual Figure Audit — 2026-04-17

## Scope and method

This is a direct eyes-on read (not a VCD re-run) of the 22 article-manifest PDFs in
`/home/zeyufu/Desktop/CLOP-DiT/results/figures/`. Every PDF was opened through the
Read tool as a rendered image and reviewed for presentation defects that a
reviewer would flag: clipped text, overlapping labels, low-contrast elements,
legends covering data, color-confusable palettes without redundant coding,
unreadable small font at article width, missing panel labels, unexplained
annotations, axis-tick overflow, and similar. Verdicts are PASS
(reviewer-ready), ADVISORY (visible cosmetic issue, ship-acceptable but worth
noting), or BLOCKER (would prompt a reviewer fix request). The scivcd follow-up
policy from `revision/figure_fix_reports/scivcd_followup_layout_review_2026-04-17.md`
is honoured: figS01/figS02 are not escalated to BLOCKER for supplement-tolerated
density / minimum-font issues alone.

## Per-figure findings

| Figure | Verdict | Defects observed | Suggested fix (source pointer) |
|---|---|---|---|
| fig01a_architecture.pdf | ADVISORY | Shared-Space dashed block (panel b) partially overlaps the `z_0` label and the `~N(0,I)` annotation — the `z_0 / 512-d` text renders on top of the purple dashed box, reducing readability at article width. Legend row at bottom is fine. | Nudge the Shared-Space dashed box up by ~20px or move `z_0 / ~N(0,I)` to a clear gap; source: `scripts/analysis/generate_architecture_figure.py`. |
| fig01b_evaluation_pipeline.pdf | PASS | Panels, arrows, and palette all legible; the `Bootstrap 95% CI (B=1000)` annotation is grey/italic and visually subordinate as intended. No clipping. | None. |
| fig02a_training_dynamics.pdf | PASS | Eight-panel training grid is clean; phase-band callouts sit inside panels (a,e); no tick overflow; legends are compact. | None. |
| fig02b_embedding_space.pdf | ADVISORY | Panel (k) y-axis labels are ellipsis-truncated ("CD8+ cytotoxic T lymp…", "Tissue-resident mac.", etc.) — readable but unresolved without the caption. Panels (l–n) are high-density scatter with many confusable colors; a typical reviewer would accept this as UMAP context but it is not fully decodable from the figure alone. | Widen panel (k) margin or show full names on hover/caption cross-ref; source: `src/visualization/fig02b_*` via `results_visualizer.py`. |
| fig03a_metrics_summary.pdf | PASS | Panels (a–d) render cleanly; numeric callouts sit outside bars and do not collide with axis ticks; CI shadow bars in panel (c) are distinguishable from filled bars. | None. |
| fig03b_per_type_fidelity.pdf | ADVISORY | Long cell-type labels on panels (e) and (f) are truncated with ellipses ("Inhibitory GABAergic neur…", etc.). Panel (g) outlier callouts ("Smooth musc…", "Kidney prox…", "Inhibitory…") are ellipsis-truncated too; one callout line crosses another. | Allow wider y-axis margin or shorten with domain abbreviations; source: `src/visualization/fig03b_*`. |
| fig03c_text_cell_alignment.pdf | ADVISORY | Panel (h) column tick labels are rotated and ellipsis-truncated, which is unavoidable at 69 types but crowds the bottom axis. Panel (i) two numeric labels at the bottom (`-0.095`, `-0.086`) sit almost on top of the axis-bottom annotation "[0 excellent / 0 good / 69 poor]". | Lift the numeric labels into the bars, or move the summary annotation to the panel header; source: `src/visualization/fig03c_*`. |
| fig04a_marker_genes.pdf | PASS | Four panels are clean; delta-expression heatmap (c) shows in-cell numbers with readable contrast; fold-change bar labels are legible. | None. |
| fig04b_expression_correlation.pdf | ADVISORY | Panel (e) leader lines from labels (`CFH`, `SHD`, `EXPH5`, `FGL1`, `U91319.1`) are long and cross the y=x diagonal and one another, making target attribution ambiguous. Panel (f) y-labels are ellipsis-truncated. | Reduce to top-3 gene callouts or place labels using fixed external slots; source: `src/visualization/fig04b_*`. |
| fig05a_expression_analysis.pdf | ADVISORY | Panel (e) legend row for the four clouds ("CD8+ cytotoxic…", "Actively proli…", etc.) is ellipsis-truncated. Panel (j) cell-type row labels are also ellipsis-truncated. Otherwise panels align well. | Widen legend area under panel (e) or show full names; source: `src/visualization/fig05a_*`. |
| fig05b_conditioning_landscape.pdf | ADVISORY | Panel (a) y-labels are ellipsis-truncated ("Dorsal horn s…", "Capillary end…" etc.). Panel (e) uses dual y-axes with three series ("FD↓", "Centroid cos↑", "Diversity↑") and a "Sweet spot" shaded band — the purple "Production ε=0.03" vertical line overlaps with the blue data marker and the shaded band edge, which can read as noise. | Shorten y-labels; shift the purple production-ε line or reduce shaded-band opacity; source: `src/visualization/fig05b_*`. |
| fig06_diversity_diagnostics.pdf | PASS | Very simple two-panel layout; bars, ratio line, and text annotations are all legible and well spaced. | None. |
| fig07a_expression_diversity.pdf | PASS | Standard deviation bars (panel a) and gene-std ratio bars (panel b) render cleanly; ratio=1 dashed reference is labelled. | None. |
| fig07b_baseline_comparison.pdf | PASS | Three-panel comparison (c–e) is clean; numeric deltas in panel (e) are aligned outside bars and do not collide; legends readable. | None. |
| fig07c_benchmark.pdf | ADVISORY | Panel (f) heatmap row labels are ellipsis-truncated ("Gaussian N(μ,…)", "Shuffled Labe…", "Mean-only (co…"). Panel (h) legend has 8 entries and is wide enough to feel crowded. Panel (i) uses repeated identical row labels across three horizontal bands separated only by dashed lines — a reviewer would not immediately read the three-metric grouping without the caption. | Add a left-side metric-group label to panel (i) (FD / Cen / Div bands) instead of repeating model names; source: `src/visualization/fig07c_*`. |
| fig08a_downstream_validation.pdf | BLOCKER | Panel (e) "Per-Type P/R/F1 (worst+best 18)" has 9 rows of 0.00/0.00/0.00 cells rendered on a black (minimum-value) background with black text, making the top nine rows effectively unreadable. The black-on-black values render as a near-empty block and will read to a reviewer as a rendering bug rather than a valid "zero" result. Panel (d) confusion matrix tick labels (0 / 23 / 46) are sparse but acceptable. | Swap the colormap floor to a lighter shade or force text color to contrast (e.g., white on black, or use a non-sequential cmap with visible text); source: `src/visualization/fig08a_*`. |
| fig08b_de_concordance.pdf | ADVISORY | Panel (g) caption-area annotation ("Log FC (scGPT space) max |logFC| 7.1e-04") sits on the scatter and is low-contrast against the blue-shaded background; the `ORM2` label is placed inside the dense cluster and is small. Panel (i) x-tick labels are long ("CD8+ cy vs CD4+ he", "Tissue vs Circula…", "Generic vs Fibrobl…") and rotated — legible but cramped. | Move the annotation out of the scatter; source: `src/visualization/fig08b_*`. |
| fig09a_variance_matching.pdf | ADVISORY | Panel (a) four sample-size callouts ("n=1,149", "n=656", "n=727", "n=853") are stacked along the right margin with leader lines but still read tightly together — consistent with the scivcd-followup "annotation density not chart logic" finding. Panel (d) three outlier leader lines are compact but one still grazes the OLS fit shading. | Cap annotations to top-3 per panel as per the 2026-04-17 layout policy; source: `scripts/analysis/variance_matching_pilot.py`. |
| fig09b_gene_gene_correlation.pdf | ADVISORY | Panel (h) three callout leader lines ("Hepatocytes are t…", "Inhibitory GABAer…", "Alveolar type 2 (…") still cross each other in the upper/lower corners of the scatter despite the scivcd-followup fixed-slot refinement — the lines run through other data points. | Drop to 2 labelled outliers or use right-margin slot list instead of in-panel leaders; source: `scripts/analysis/gene_gene_correlation.py`. |
| figS01_supplementary_validation.pdf | ADVISORY | Supplementary-tolerated density as per the 2026-04-17 policy. Sub-panels (c), (f), (k), (l) carry small fonts and inline text blocks; panels (i)/(j) variance scatter/ratio are readable but tight. Panel (h) radar is small enough that axis labels ("Type Specificity", "Text Steering", "DE Concordance", "Marker Recall", "OOD Handling", "Cross-dataset Corr.") crowd the outer ring. No outright clipping. | Supplement-acceptable per existing policy; only escalate if journal flags. Source: `src/visualization/figS01_*`. |
| figS02_expression_diagnostics.pdf | ADVISORY | Supplementary-tolerated density. Panel (a) cell-type y-labels are abbreviated ("Multiciliated…", "Tissue-reside…", etc.). Panel (c) quality-by-family stacked bars use three-color ramp (green/yellow/red) that conveys Pass/Warn/Fail with redundant ordering — acceptable. Panel (h) single-annotation "Primary driver: Mean shift" reads correctly. | Supplement-acceptable per existing policy; optionally widen panel (a). Source: `src/visualization/figS02_*`. |
| figS_lane_c_zero_shot.pdf | PASS | Three tissue blocks (Kidney / Cerebellum / Fetal gonadal) render cleanly; random-baseline dashed references are labelled; numeric bar-end values do not collide. Matches the scivcd-followup "publication-ready" verdict. | None. |

## Totals

- PASS: 8 (fig01b, fig02a, fig03a, fig04a, fig06, fig07a, fig07b, figS_lane_c_zero_shot)
- ADVISORY: 13 (fig01a, fig02b, fig03b, fig03c, fig04b, fig05a, fig05b, fig07c, fig08b, fig09a, fig09b, figS01, figS02)
- BLOCKER: 1 (fig08a — black-on-black zero cells in the Per-Type P/R/F1 heatmap)

Total audited: 22 / 22. No PDFs failed to load.
