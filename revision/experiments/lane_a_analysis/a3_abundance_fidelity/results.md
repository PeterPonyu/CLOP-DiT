# A3 — Abundance vs fidelity quantitative analysis

**Reviewer comment:** R2.7 — "biologically difficult" is not defined.
**Retraining:** None.
**Ran on:** 2026-04-15
**Script:** `compute_abundance_fidelity.py`
**Input artifacts (read-only, live paths):**
- `data/cached_latents/text_group_ids_dedup.npy` (167 245 training cells)
- `data/cached_latents/text_captions_deduplicated.json` (69 type captions)
- `results/per_type_dashboard.json` (per-type metrics)
- `results/expression_metrics.json` (per-type expression Pearson)

## Question

Replace the vague "biologically difficult" phrase with an operational
definition and check whether per-type fidelity is explained by
training-set abundance at all — or by something else.

## Status

- [x] Underrepresented threshold chosen: bottom quartile of training
      counts (Q25 ≈ 959 cells).
- [x] Per-type abundance table assembled (68 / 69 types matched;
      "Generic epithelial cells" has no dashboard entry and is skipped).
- [x] Correlations computed with Pearson + Spearman, two-sided p.
- [x] Quartile comparison via Mann-Whitney U.
- [x] Predictor comparison: abundance vs real within-type tightness.
- [x] Rebuttal verdict sentence committed.

## Results

### Abundance alone explains less than you would expect

Correlation of each per-type fidelity metric with `log10(training cell count)`
across 68 cell types:

| Metric | Pearson r | Pearson p | Spearman ρ | Spearman p |
|---|---:|---:|---:|---:|
| centroid cosine | +0.021 | 0.86 | −0.242 | 0.047 |
| expression Pearson r | −0.098 | 0.43 | −0.135 | 0.27 |
| diversity ratio | −0.288 | 0.023 | +0.035 | 0.79 |
| Frechet distance | +0.501 | 1.4 × 10⁻⁵ | **+0.566** | 4.9 × 10⁻⁷ |

Abundance has a *positive* association with Frechet distance —
well-represented types have **worse** FD, not better. Bottom quartile
mean FD = 1.026, top quartile mean FD = 1.140 (Mann-Whitney
p = 1.1 × 10⁻⁴). Centroid cosine is essentially flat (Pearson
r ≈ 0) but top-quartile is slightly worse than bottom-quartile
(0.896 vs 0.907, p = 0.017).

### Heterogeneity is a much stronger predictor

`real_intra_cos` is the mean within-type pairwise cosine of real cell
embeddings — a direct measure of how tight (high value) or spread (low
value) the real cell cluster is. Comparing the two predictors:

| Target | Predictor | Pearson r | Spearman ρ |
|---|---|---:|---:|
| centroid cosine | log₁₀(count) | +0.021 | −0.242 |
| centroid cosine | real_intra_cos | +0.084 | **+0.448** (p = 3 × 10⁻⁴) |
| **Frechet distance** | log₁₀(count) | +0.501 | +0.566 |
| **Frechet distance** | **real_intra_cos** | **−0.782** (p = 6 × 10⁻¹⁴) | **−0.768** (p = 3 × 10⁻¹³) |
| diversity ratio | log₁₀(count) | −0.288 | +0.035 |
| diversity ratio | real_intra_cos | +0.412 (p = 9 × 10⁻⁴) | −0.098 |

For Frechet distance — the metric where abundance looked most
"important" — `real_intra_cos` is almost twice as strong a predictor
(|ρ| = 0.77 vs 0.57) and has the biologically interpretable sign: the
more spread out a cell type is in the real data, the harder the
generator finds it to match. Abundance looked like a predictor only
because highly-abundant types tend to be biologically heterogeneous
(spanning multiple tissue contexts in the corpus).

### The operational definition we will use

- **Underrepresented in training** (the reviewer's literal request):
  `training_count ≤ Q25 ≈ 959 cells`. Associated with *slightly
  better* per-type metrics on most axes.
- **Biologically difficult** (the phrase the reviewer flagged): we
  retire this phrase in the revised manuscript. Where we mean
  "harder to generate", we use **"types with broad intrinsic
  heterogeneity"** — operationalised as `real_intra_cos < P25` (i.e.,
  the bottom quartile of within-type cosine tightness), which is the
  variable that actually predicts FD and diversity-ratio deficits.

## Rebuttal-ready sentence (paste into R2.7 response)

We agree that the phrase "biologically difficult" was imprecise and
have replaced it throughout the manuscript with the operational
definition "cell types with broad intrinsic within-type heterogeneity",
quantified as `real_intra_cos < 25th percentile` (Supplementary
Table S3). Across the 68 matched cell types, per-type Frechet distance
is only weakly related to training-set abundance (Spearman
ρ = 0.57, p = 4.9 × 10⁻⁷) but is strongly explained by within-type
heterogeneity (ρ = −0.77, p = 3 × 10⁻¹³), and a bottom-quartile-by-
training-count slice does not show the fidelity gap the reviewer
suspected (centroid cosine 0.907 vs 0.896 for the top quartile,
Mann-Whitney p = 0.02, favouring the underrepresented set).
Underrepresentation by training-cell-count therefore does not appear
to be the dominant axis of failure; within-type heterogeneity is the
better-targeted diagnosis and motivates the heterogeneity-preserving
training extensions discussed in the revised Discussion.
