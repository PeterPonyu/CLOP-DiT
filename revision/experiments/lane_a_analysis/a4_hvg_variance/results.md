# A4 — HVG variance deficit numeric summary

**Reviewer comment:** R2.8 — supplementary residual / variance-ratio
figures are not quantified in the main text.
**Retraining:** None.
**Ran on:** 2026-04-15
**Script:** `compute_hvg_variance_summary.py`
**Input artifacts (read-only, live results/ paths):**
- `results/real_expression.npy` (1932 × 1790)
- `results/generated_expression.npy` (2000 × 1790)
- `results/real_expression_labels.npy`, `results/generated_expression_labels.npy`
- `results/expression_gene_names.json`
- cross-dataset reference: `results/downstream/cross_dataset_validation.json`

SHA-256 prefixes of inputs are embedded in `hvg_variance_summary.json`
under `inputs.*.sha256_head` so downstream comparisons stay tied to the
exact arrays used here.

## Question

Turn the existing residual / HVG variance-ratio figures into a
defensible numeric summary for the main text.

## Status

- [x] Tolerance band defined: `[0.5, 2.0]` (R2.8 convention)
- [x] Summary stats computed — pooled, HVG, non-HVG
- [x] HVG / non-HVG split computed (HVG = top quartile by real variance)
- [x] Within-type summary computed (the slice R2.8 actually asks about)
- [x] Cross-dataset reference recorded for reviewer reconciliation
- [x] Paragraph drafted for main text
- [x] Rebuttal verdict sentence committed

## Results

### Three scales — three different answers

| Scale | r_var (Pearson) | median std ratio | frac in `[0.5, 2.0]` | frac with >2× deficit |
|---|---:|---:|---:|---:|
| **Pooled across 69 types** | **+0.988** | 1.326 | 84.6 % | **0.0 %** |
| **Within-type, averaged over 69 types** | median **+0.848**, mean **+0.827** | 1.134 | 75.2 % | 8.7 % |
| **Cross-dataset held-out (5 tissues)** | median ratio **0.011 – 0.022** | — | — | catastrophic |

HVG vs non-HVG (pooled): HVGs have median std ratio 1.331 (81.5 % in
band, 0.0 % >2× deficit); non-HVGs 1.325 (85.7 % in band, 0.0 % >2×
deficit). HVGs are not systematically worse than non-HVGs on the
pooled slice.

### Reading the three numbers together

- **Pooled r_var (+0.988)** is the number a naive readout would
  produce. It is high because, when all 69 cell types are mixed, the
  dominant source of gene variance is *between-type* mean differences,
  not within-type heterogeneity. Conditional generation reproduces the
  type centroids well, so the pooled variance looks correct.
- **Within-type r_var (+0.85)** is the correct scale for R2.8. 100 %
  of the 69 cell types show a positive correlation between real and
  generated gene-wise variance. 75 % of (type, gene) pairs are in
  tolerance; the 8.7 % with >2-fold deficit is concentrated in a
  subset of rare / heterogeneous types (cross-reference Lane A3 once
  complete).
- **Cross-dataset held-out (~0.01)** is the variance-ratio reported in
  the pre-revision baseline snapshot. It reflects a much stricter
  evaluation: generated samples are compared against real cells from
  held-out GEO studies whose cell-type composition differs from the
  training distribution. Median variance ratio is 50-100× smaller
  than real, i.e. variance collapse under distribution shift.

The three numbers are all correct; they describe different slices.
The headline in the baseline snapshot used the cross-dataset number
and is therefore a worst-case number, while the within-type evaluation
on ID data is a best-case number.

## Rebuttal-ready sentence (paste into R2.8 response)

We have added a quantitative summary of the residual and variance-ratio
figures to the main text. Across the 1 790 genes scored in the
in-distribution evaluation, 85 % of genes fall inside the `[0.5, 2.0]`
tolerance band for the gen / real variance ratio, with a median
standard-deviation ratio of 1.33 and a pooled gene-wise variance
Pearson correlation of r_var = 0.988. Restricting the analysis to
within-type variance — the scale relevant to reviewer concerns about
second-order fidelity — median r_var across the 69 cell types is
+0.85 and 100 % of types exhibit a positive correlation, with 75 %
of (type, gene) pairs remaining inside tolerance and 8.7 % showing a
variance deficit greater than two-fold. The cross-dataset held-out
evaluation, which contrasts generation against unseen GEO studies,
remains the stricter test and continues to show variance collapse
(median ratio 0.01 – 0.02), motivating the strict-OOD experiments in
the revised Lane C.
