# A4 — HVG variance deficit numeric summary

**Reviewer comment:** R2.8 — supplementary residual / variance-ratio
figures are not quantified in the main text.
**Retraining:** None.
**Input artifacts (frozen baseline):**
- `revision/prerevision_baseline/metrics_frozen/expression_metrics.json`
- `results/generated_expression.npy` (read-only)

## Question

Turn the existing residual / HVG variance-ratio figures into a
defensible numeric summary for the main text.

## Deliverables

1. **Fraction within tolerance band** — e.g., fraction of genes with
   variance ratio in `[0.5, 2.0]`.
2. **Fraction with >2× variance deficit** — genes where real/generated
   variance ratio exceeds 2.
3. **Median std-ratio for HVGs** and for non-HVGs separately.
4. **Short paragraph** for the main text stating the numbers explicitly
   so reviewers do not have to read them off a supplementary figure.

## Plan

- Load existing gene-level variance arrays.
- Compute the three summary stats above, plus an HVG vs non-HVG split.
- Emit `hvg_variance_summary.json` + a one-line text insert for the
  manuscript.

## Status

- [ ] Tolerance band defined
- [ ] Summary stats computed
- [ ] HVG / non-HVG split computed
- [ ] Paragraph drafted for main text
- [ ] Rebuttal verdict sentence committed

## Results

_(fill in after the analysis runs)_

## Rebuttal-ready sentence

_(2–3 sentences that can be pasted into the R2.8 response)_
