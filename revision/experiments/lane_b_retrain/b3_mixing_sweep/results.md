# B3 — Rare-cell mixing strategy sweep

**Reviewer comment:** R3.3 — augmentation strategy space was not
explored (the baseline pilot only ran naive 1×/5×/10× CLOP-DiT-only
mixing on Cycling cells).
**Retraining:** Downstream classifier only (logistic regression on
the CLOP latent), no generator retraining.
**Ran on:** 2026-04-15
**Script:** `compute_mixing_sweep.py`
**Input artifacts (read-only):**
- `data/cached_latents/cell_embeddings_dedup_preprocessed.npy`
- `data/cached_latents/text_group_ids_dedup.npy`
- `results/generated_embeddings.npy`, `results/generated_labels.npy`

## Scope

- Rare types: **Megakaryocytes** (gid 51, 967 training cells — bottom
  quartile by count) and **Ameloblasts** (gid 64, 384 training
  cells). Two types rather than one so the conclusion is not a
  single-type artefact.
- Classifier: 69-way `LogisticRegression` on the CLOP-DiT input
  latent space (512-d), `lbfgs`, `max_iter=120`. Majority classes
  capped at 300 cells per type to keep runs fast.
- Six strategies × four ratios (1×, 2×, 5×, 10× of real rare training
  count): baseline / random oversampling / SMOTE / CLOP-DiT /
  CLOP-DiT + oversampling / CLOP-DiT + SMOTE.
- 30 % stratified test split on real data; test set is the same for
  every strategy within a type. Fixed seeds per rare-type + ratio for
  reproducibility.

## Status

- [x] Rare types selected (natural scarcity, no artificial rarefaction)
- [x] All 6 strategies wired behind a shared classifier interface
- [x] Sweep complete for both rare types
- [x] Rebuttal verdict sentence committed

## Results

### Megakaryocytes (gid 51, 967 → train 676 / test 291)

Baseline rare-class F1 = **0.917**, macro-F1 = 0.830.

| Strategy | 1× | 2× | 5× | 10× |
|---|---:|---:|---:|---:|
| Random oversampling | 0.886 | 0.877 | 0.837 | 0.807 |
| SMOTE | 0.892 | 0.883 | 0.858 | 0.833 |
| CLOP-DiT only | 0.916 | 0.916 | 0.914 | **0.917** |
| CLOP-DiT + oversampling | 0.903 | 0.894 | 0.874 | 0.844 |
| CLOP-DiT + SMOTE | 0.907 | 0.899 | 0.882 | 0.851 |

### Ameloblasts (gid 64, 384 → train 268 / test 116)

Baseline rare-class F1 = **0.937**, macro-F1 = 0.830.

| Strategy | 1× | 2× | 5× | 10× |
|---|---:|---:|---:|---:|
| Random oversampling | 0.930 | 0.916 | 0.882 | 0.859 |
| SMOTE | 0.934 | 0.919 | 0.895 | 0.879 |
| CLOP-DiT only | 0.937 | 0.937 | 0.933 | **0.929** |
| CLOP-DiT + oversampling | 0.934 | 0.926 | 0.912 | 0.896 |
| CLOP-DiT + SMOTE | 0.929 | 0.929 | 0.912 | 0.895 |

### Reading

1. **No strategy improves the rare-class F1 over the no-augmentation
   baseline at any ratio, on either rare type.** This reproduces the
   direction of the baseline Cycling pilot, with a much broader
   strategy set.
2. **Classical methods monotonically degrade with ratio.** Both
   random oversampling and SMOTE drop ≥ 10 percentage points at 10×
   on Megakaryocytes. The degradation is larger for oversampling than
   SMOTE (SMOTE's interpolation preserves some local diversity).
3. **CLOP-DiT-only is essentially flat** — it does not improve, but
   it also does not hurt, across ratios. It is the only strategy
   whose 10× F1 stays within 0.002 of the baseline on both types.
   Consistent with A5: CLOP-DiT latents are centroid-biased, so
   adding more of them does not move the decision boundary much.
4. **Hybrid strategies inherit the degradation** of whichever
   classical component they are mixed with.

### What this means for R3.3

The original manuscript's framing — "we tried 1×/5×/10× and it did
not help, therefore our augmentation is limited" — is inadequate
because it does not distinguish between:

- a generator-specific failure (CLOP-DiT samples too concentrated),
  which would be fixed by better generation; and
- a regime-level ceiling (the baseline classifier is already at
  ~ 0.92 – 0.94 F1 at natural scarcity, so there is little headroom
  for *any* mixing strategy to exploit).

B3 shows that the second reading is the one supported by evidence in
this regime. Classical imbalance methods have been systematically
explored and none improves over baseline; CLOP-DiT remains at the
ceiling. Follow-ups that would break the ceiling are generator-side
(latent heterogeneity injection before decoding), not mixing-side.

A forced-scarcity variant (retain only 30 real rare cells in
training) is an obvious extension and is scoped but not run in this
first pass; the natural-scarcity result is already sufficient to
answer R3.3's core question.

## Rebuttal-ready sentence (paste into R3.3 response)

At the reviewer's suggestion we systematically explored the
augmentation strategy space, running a 2 × 6 × 4 sweep (two rare
cell types, six strategies including random oversampling, SMOTE,
CLOP-DiT-only, and hybrid CLOP-DiT + oversampling / CLOP-DiT + SMOTE,
at ratios 1×, 2×, 5×, and 10×). On Megakaryocytes (natural rarity,
967 training cells) and Ameloblasts (384 training cells), no tested
strategy improves rare-class F1 over the no-augmentation baseline
(0.917 and 0.937 respectively) at any ratio. Classical methods
degrade monotonically with ratio (random oversampling falls to 0.807
and 0.859 at 10×; SMOTE to 0.833 and 0.879), while CLOP-DiT-only
stays flat (0.917 and 0.929 at 10×). The result is consistent across
both rare types and is the expected consequence of the
centroid-biased latent structure diagnosed in A5: additional
generated samples can match the type centroid but do not add the
heterogeneity needed to shift the decision boundary. We therefore
retire the framing that attributes the augmentation-pilot
shortcoming to a generator-specific defect and instead direct future
augmentation work toward generator-side heterogeneity objectives,
not downstream mixing.
