# A3 — Abundance vs fidelity quantitative analysis

**Reviewer comment:** R2.7 — "biologically difficult" is not defined.
**Retraining:** None.
**Input artifacts (frozen baseline):**
- `revision/prerevision_baseline/metrics_frozen/per_type_dashboard.json`
- `revision/prerevision_baseline/metrics_frozen/comprehensive_summary.json`
- Training cell counts from `data/cached_latents/`

## Question

Replace the vague phrase "biologically difficult" with an operational
definition. Is per-type fidelity explained by training-set abundance?

## Deliverables

1. **Operational definition** — underrepresented = bottom quartile by
   training cell count (state the absolute threshold).
2. **Correlation table** (per-type):
   - centroid cosine vs log(train count) — Pearson / Spearman + CI
   - KNN vs log(train count)
   - DivR vs log(train count)
3. **Abundance-stratified box plots** — bottom vs top quartile, with
   Mann-Whitney p-value.
4. **Family-stratified version** (optional, tie-in to A1 taxonomy):
   does abundance explain variance within the same family?

## Plan

- Emit `abundance_fidelity.csv` (one row per cell type).
- Produce a correlation figure + quartile boxplot.
- Compare against the existing `quality ~ training size, r = 0.43`
  result and report the richer decomposition.

## Status

- [ ] Underrepresented threshold chosen + justified
- [ ] Per-type abundance table assembled
- [ ] Correlations computed with CIs
- [ ] Quartile comparison test run
- [ ] Figure drafted
- [ ] Rebuttal verdict sentence committed

## Results

_(fill in after the analysis runs)_

## Rebuttal-ready sentence

_(2–3 sentences that can be pasted into the R2.7 response)_
