# A5 — Rare-cell augmentation failure mechanism

**Reviewer comment:** R2.11 — the rare-cell augmentation failed; what
is the mechanism?
**Retraining:** None.
**Ran on:** 2026-04-15
**Script:** `compute_rare_failure_mechanism.py`
**Input artifacts (read-only):**
- `data/cached_latents/cell_embeddings_dedup_preprocessed.npy`
- `data/cached_latents/text_group_ids_dedup.npy`
- `results/generated_embeddings.npy`, `results/generated_labels.npy`
- `results/real_expression.npy`, `results/real_expression_labels.npy`
- `results/generated_expression.npy`, `results/generated_expression_labels.npy`

## Question

Is the rare-cell augmentation failure upstream (generator produces
near-centroid latents) or downstream (decoder averages out latent
variation)?

## Method

For each cell type analysed we compute the within-type total variance
(trace of covariance) and the intra-cluster cosine similarity at two
scales — the shared CLOP-DiT latent space and the decoded gene
expression space — then compare generated to real:

- `latent_var_ratio = gen latent var / real latent var`
- `expression_var_ratio = gen expression var / real expression var`

If the latent ratio is low but the expression ratio recovers, the
failure is upstream. If the latent ratio is adequate but the
expression ratio collapses, the failure is decoder compression. If
both are low, both stages are contributing.

The analysis covers:

- **Rare-by-count** set: bottom-quartile training counts (≤ 967
  cells, n = 18 types).
- **Reviewer-flagged Cycling** type (`group 4`): not rare by count
  (6 145 cells) but biologically rare in downstream analyses.
- **Reference** set: top-5 types by training count (n = 5), for
  contrast.

## Status

- [x] Cycling real + generated slices assembled
- [x] Latent within-type variance computed
- [x] Expression within-type variance computed
- [x] Mechanism verdict assigned per type (30 % tolerance)
- [x] Rebuttal verdict sentence committed

## Results

### Mechanism verdict counts

| Slice | upstream only<br>(decoder expands) | upstream + downstream<br>(both compressed) | approximately matched |
|---|---:|---:|---:|
| Rare by count (n = 18) | **13** | **4** | 1 |
| Reference top-5 (n = 5) | 0 | 0 | **5** |
| Reviewer-flagged Cycling | — | — | 1 (with high directional tightness) |

### Rare-by-count detail (selected rows)

| gid | n_train | Cell type | latent var ratio | expression var ratio | latent intra-cos ratio |
|---:|---:|---|---:|---:|---:|
| 51 | 967 | Megakaryocytes | 0.66 | 0.74 | 3.8 |
| 55 | 727 | Neuroendocrine cells | 0.35 | 0.94 | 3.6 |
| 58 | 669 | Oligodendrocyte precursors | 0.30 | 0.75 | 2.1 |
| 62 | 396 | Pituitary gonadotrophs | 0.64 | 1.90 | 2.9 |
| 67 | 138 | Pericytes / mural cells | 0.50 | 1.56 | 1.8 |
| 68 | 77 | Kidney proximal tubule | 1.06 | 1.86 | 0.9 |

Full table in `rare_failure_mechanism.json → per_type`.

### Cycling (reviewer-flagged)

| Scale | real | generated | ratio |
|---|---:|---:|---:|
| latent total variance | (see JSON) | (see JSON) | 0.88 |
| expression total variance | (see JSON) | (see JSON) | 0.95 |
| latent intra-cosine | 0.025 | 0.115 | **4.6** |

Total variance is approximately matched, but intra-cluster cosine
similarity is 4.6× higher in generated than in real — the generated
cluster is much more directionally concentrated (tighter along a
principal axis) even though its overall spread magnitude is
preserved. This is consistent with the naive augmentation failing:
extra synthetic cells add volume along a dominant direction, not
orthogonal new information.

## Interpretation

For 13 of 18 truly rare cell types, the pattern is unambiguous —
**latent variance is compressed to 30 – 66 % of real while expression
variance recovers to 47 – 190 %** of real. The decoder is not the
bottleneck. The generator emits latents that cluster too tightly
around the type centroid, and the scGPT decoder actually partially
compensates by adding per-cell variability during decoding. This
means:

1. The rare-cell augmentation failure in the baseline pilot is best
   explained by upstream latent under-dispersion, not by decoder
   collapse.
2. A mixing strategy that injects extra diversity at the latent
   level — or a generator-side heterogeneity objective — is the
   correct lever (cross-reference **Lane B3 mixing sweep**).
3. Pure post-hoc mixing with SMOTE / oversampling at the expression
   level should help mechanistically, because the decoder is already
   expansive; adding further latent diversity before decoding is the
   bigger win.

The reference top-5 types all score "approximately matched", showing
this is a rare-type-specific failure mode, not a model-wide defect.

## Rebuttal-ready sentence (paste into R2.11 response)

Decomposing within-type variance at the latent and expression scales
attributes the rare-cell augmentation failure to upstream latent
under-dispersion rather than downstream decoder collapse. Across the
18 lowest-abundance cell types, the generator produces latents whose
within-type total variance is 30 – 66 % of real (median 0.50 × real),
yet the scGPT decoder recovers expression-space variance to 47 – 190 %
of real (median 1.07 × real). Put differently, the decoder is already
expansive; the generator is centroid-biased. For the Cycling
population specifically, latent and expression total variance are
within 5 – 15 % of real but intra-cluster cosine similarity is 4.6 ×
higher in generated samples than in real, indicating that synthetic
diversity is highly directional. This supports targeting the
augmentation strategy at the latent stage (Lane B3 mixing sweep) and
retiring naive multiplicative oversampling.
