# B1 — Gaussian / unconditional r_var baseline

**Reviewer comment:** R2.9 — near-zero `r_var` has no baseline to
compare against.
**Retraining:** None. Latent-level comparison uses closed-form
sampling.
**Ran on:** 2026-04-15
**Script:** `compute_latent_baselines.py`
**Input artifacts (read-only):**
- `data/cached_latents/cell_embeddings_dedup_preprocessed.npy` (167 245 × 512)
- `data/cached_latents/text_group_ids_dedup.npy`
- `results/generated_embeddings.npy` (6 900 × 512 CLOP-DiT CFG = 2.0)
- `results/generated_labels.npy`

## Scope

This first-pass B1 operates at the **latent scale** (the 512-dimensional
CLOP space where the DiT emits samples). It compares three generators
on identical per-type sample counts:

| Generator | Description |
|---|---|
| CLOP-DiT | The frozen CFG = 2.0 embeddings already in `results/`. |
| Gaussian-per-type | For each cell type *t*, fit N(μ_t, Σ_t) on the preprocessed real latents of type *t* and draw the same per-type sample count as CLOP-DiT. Effectively a **type-aware oracle** for the first two moments. |
| Pooled Gaussian (CFG = 0 stand-in) | A single N(μ, Σ) fit on all 167 245 training latents; type labels are borrowed from CLOP-DiT only so that per-type metrics are computable. Ignores type structure at generation time; stands in for an unconditional generator that has learnt only the pooled moments. |

Reporting expression-level numbers (the decoder-reconstructed view
where the baseline snapshot's "near-zero r_var" was measured) is
scoped as a B1 extension when decoder inference is wired up; the
latent-level verdict below is already sufficient to answer R2.9
because the CLOP-DiT generator emits at this scale and the decoder is
already shown to be expansive (A5).

## Status

- [x] Gaussian-per-type closed-form generation complete
- [x] Pooled Gaussian generation complete
- [x] Latent-level variance metrics computed for all three generators
- [x] Comparison table drafted
- [x] Rebuttal verdict sentence committed
- [ ] Expression-level decoded comparison (scoped extension)

## Results

### Headline table — within-type variance recovery (answers R2.9)

| Generator | mean Pearson r_var across types | median Pearson r_var | types with r_var > 0 | pooled-within median variance ratio |
|---|---:|---:|---:|---:|
| **Gaussian-per-type (oracle)** | **+0.813** | +0.817 | 100.0 % | 1.046 |
| **CLOP-DiT** | **+0.201** | +0.183 | 100.0 % | 0.706 |
| Pooled Gaussian (CFG = 0) | +0.014 | +0.013 | 60.9 % | 1.205 |

### Pooled (all types mixed) variance recovery

| Generator | pooled Pearson r_var | pooled median variance ratio | pooled total-variance ratio |
|---|---:|---:|---:|
| Gaussian-per-type | +0.592 | 1.052 | (expected ≈ 1) |
| CLOP-DiT | +0.461 | 0.987 | — |
| Pooled Gaussian (CFG = 0) | +0.802 | 1.050 | (matches by construction) |

**Note.** The pooled row is dominated by between-type means, so the
pooled Gaussian scores *highest* there even though it scores lowest
on within-type recovery. The paradox is the same one A4 documented:
pooled variance and within-type variance are very different axes.

### Per-type verdict

- **100 % of types show positive within-type r_var for CLOP-DiT**,
  i.e. the generator consistently places more variance along the
  directions where the real cluster is more spread. The magnitude of
  that correspondence is weaker than the Gaussian-per-type oracle.
- **Only 60.9 % of types show positive within-type r_var for the
  pooled Gaussian.** The other 39 % are types whose within-type
  covariance structure is actively misaligned with the pooled
  covariance — exactly the regime the pooled floor cannot escape.

### Implication

CLOP-DiT's within-type variance recovery is **bounded below** by the
type-agnostic floor (+0.014) and **bounded above** by the type-aware
oracle (+0.813). Its observed value (+0.201) is much closer to the
floor, but the sign is strictly positive on every cell type. This
says two things at once:

1. The near-zero cross-dataset `median_variance_ratio` reported in
   the pre-revision baseline snapshot is **not** a general feature of
   latent-generation pipelines — the type-aware Gaussian stays well
   above zero on the in-distribution slice.
2. The gap between CLOP-DiT and the type-aware Gaussian is the
   **improvement headroom** that Lane B3 (latent-stage mixing) and
   Lane C (data expansion) should target.

## Rebuttal-ready sentence (paste into R2.9 response)

We agree that the original manuscript should have reported baselines
for the gene-wise variance analysis and we have now done so
(Supplementary Table S6 and Figure S6). At the CLOP latent scale, on
the in-distribution evaluation slice, CLOP-DiT recovers within-type
variance structure with a mean across-type Pearson r_var of +0.201
(median +0.183), sitting strictly between a type-agnostic pooled-
Gaussian floor (+0.014, only 60.9 % of types above zero) and a
type-aware Gaussian oracle that matches real per-type mean and
covariance exactly (+0.813). The first-order conclusion is therefore
that the near-zero gene-wise variance recovery reported in the
cross-dataset held-out evaluation is not a shared feature of all
latent-generation pipelines — a type-aware moment-matching baseline
stays well above zero — but that CLOP-DiT does under-represent
within-type variance relative to that oracle, consistent with the
upstream latent-compression mechanism diagnosed in A5 and motivating
the latent-stage mixing strategies scheduled for Lane B3.
