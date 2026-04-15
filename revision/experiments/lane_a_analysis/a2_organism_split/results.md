# A2 — Human vs mouse stratified evaluation

**Reviewer comment:** R3.2 — species stratification is not reported.
**Retraining:** None.
**Input artifacts (frozen baseline):**
- `revision/prerevision_baseline/metrics_frozen/comprehensive_summary.json`
- Organism metadata from `data/cached_latents/text_captions_deduplicated.json`

## Question

Does pooled evaluation hide a systematic performance gap between human
(59 datasets) and mouse (21 datasets)? Is the shared CLOP space
dominated by the majority species?

## Deliverables

1. **Stratified metric table** (human-only / mouse-only / overall) for:
   KNN-1, KNN-5, Steering, DivR, centroid cosine, coverage, density.
2. **Gap column**: human value − mouse value per metric.
3. **Organism-stratified field ablation** (optional): does the
   conditioning signal differ in strength by species?

## Plan

- Partition existing generated-vs-real comparisons by organism label.
- Compute each headline metric on the split.
- Emit `metrics_by_organism.csv` + a single summary table for the
  rebuttal.

## Status

- [ ] Organism label attached to every generated sample
- [ ] Metrics recomputed on human subset
- [ ] Metrics recomputed on mouse subset
- [ ] Gap table produced
- [ ] Rebuttal verdict sentence committed

## Results

_(fill in after the analysis runs)_

## Rebuttal-ready sentence

_(2–3 sentences that can be pasted into the R3.2 response)_
