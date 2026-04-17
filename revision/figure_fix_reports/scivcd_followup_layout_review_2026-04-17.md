# scivcd Follow-up Layout Review (2026-04-17)

## Scope

Direct visual read of:

- `results/figures/fig09a_variance_matching.pdf`
- `results/figures/fig09b_gene_gene_correlation.pdf`
- `results/figures/figS01_supplementary_validation.pdf`
- `results/figures/figS02_expression_diagnostics.pdf`
- `results/figures/figS_lane_c_zero_shot.pdf`

## Conclusions

### fig09a_variance_matching

- Main issue: annotation density, not chart logic.
- Panel (a): `n=` sample-size labels crowd the top-right margin and visually merge.
- Panel (d): outlier labels and leader lines cross, making the callouts harder to parse than the points themselves.
- Applied refinement: moved outlier/sample-size annotations into fixed external callout slots with short leader lines.
- Remaining recommendation: if another polish round is needed, reduce panel (a) y-label count further and keep only the top 3 SWD callouts.

### fig09b_gene_gene_correlation

- Main issue: panel (h) outlier labels are valid but placed too close together.
- Applied refinement: replaced offset-point labels with fixed external callout slots so the three highlighted types no longer stack on top of one another.
- Remaining recommendation: keep at most 3 labelled outliers in this panel; any fourth label should go into caption prose, not onto the scatter.

### figS01_supplementary_validation

- Main issue: composed supplementary density remains high.
- Most reviewer-visible pressure points are the text-heavy OOD showcase row and the small lower-right analytical panels.
- Recommendation: acceptable for supplement as-is, but do not add more inline text to panels (c) or (i)–(l). If the journal asks for another readability pass, split S01 into two supplementary figures before shrinking fonts any further.

### figS02_expression_diagnostics

- Overall status: denser than ideal, but materially more readable than S01.
- Main issue: panel (a) still relies on abbreviated cell-type labels and panel widths are tight.
- Recommendation: acceptable for supplement as-is. If another round is needed, widen panel (a) and shorten family labels with curated domain abbreviations rather than ellipsis truncation.

### figS_lane_c_zero_shot

- Status: publication-ready.
- The three tissue blocks read cleanly, random baselines are legible, and numeric bar-end annotations do not collide.
- Recommendation: keep this figure as the rebuttal-embedded preview figure because it carries a complete reviewer-facing story with low visual risk.

## Annotation-style rule distilled from this pass

- Small subplot annotations should not be allowed to self-organize by offset heuristics once there are more than 2 labels in the same local region.
- Prefer one of:
  - fixed external callout slots,
  - top-k label capping,
  - caption-side reporting instead of on-panel text.
- For article-width figures, annotation boxes should generally be readable without zoom at ~8–9 pt effective size; if not, reduce annotation count before reducing font size.
