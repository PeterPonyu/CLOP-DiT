# Experiment Comparison Template

Create one copy of this template per experiment or model revision.

Suggested filename pattern:

- `YYYY-MM-DD_<short_experiment_name>.md`

Example:

- `2026-04-15_text_encoder_pubmedbert.md`
- `2026-04-15_species_split_analysis.md`
- `2026-04-16_strict_ood_holdout.md`

---

## Experiment metadata

- **Date**:
- **Branch**:
- **Commit**:
- **Experiment name**:
- **Reviewer issue(s) targeted**:
- **Category**:
  - [ ] Text encoder
  - [ ] Dataset expansion
  - [ ] OOD evaluation
  - [ ] Organism split
  - [ ] Rare-cell augmentation
  - [ ] Whitening / preprocessing
  - [ ] CLOP ablation bridge
  - [ ] Decoder / latent interface
  - [ ] Other

## What changed

Describe exactly what was changed relative to the baseline in `baseline_snapshot_2026-04-15.md`.

## Expected benefit

What specific reviewer criticism is this intended to address?

## Core metric comparison

### Main operating regimes

- **High-fidelity (`CFG=2.0`, Euler-10)**
  - KNN-1: baseline `0.369` → new `__` (delta `__`)
  - KNN-5: baseline `0.553` → new `__` (delta `__`)
  - Steering: baseline `0.810` → new `__` (delta `__`)
  - DivR: baseline `0.513` → new `__` (delta `__`)
  - LinAcc: baseline `0.511` → new `__` (delta `__`)
  - FD: baseline `3.17` → new `__` (delta `__`)

- **High-diversity (`CFG=1.0`, Midpoint-10)**
  - KNN-1: baseline `0.288` → new `__` (delta `__`)
  - KNN-5: baseline `0.461` → new `__` (delta `__`)
  - Steering: baseline `0.807` → new `__` (delta `__`)
  - DivR: baseline `0.929` → new `__` (delta `__`)
  - LinAcc: baseline `0.357` → new `__` (delta `__`)
  - FD: baseline `2.64` → new `__` (delta `__`)

### Secondary reviewer-facing metrics

- `r_var` range / mean: baseline `-0.019 to +0.013` → new `__` (delta `__`)
- Novel OOD marker-hit rate: baseline `0.167` → new `__` (delta `__`)
- Rare-cell Cycling F1: baseline `0.828` → new `__` (delta `__`)
- Rare-cell +1x F1: baseline `0.786` → new `__` (delta `__`)
- Rare-cell +5x F1: baseline `0.741` → new `__` (delta `__`)
- Rare-cell +10x F1: baseline `0.741` → new `__` (delta `__`)

### If species-stratified

- KNN-1: human baseline `__`, human new `__`; mouse baseline `__`, mouse new `__`
- Steering: human baseline `__`, human new `__`; mouse baseline `__`, mouse new `__`
- DivR: human baseline `__`, human new `__`; mouse baseline `__`, mouse new `__`
- Centroid cosine: human baseline `__`, human new `__`; mouse baseline `__`, mouse new `__`

## Interpretation

### What improved?

### What got worse?

### Is the trade-off acceptable?

### Does this actually answer the reviewer’s concern?

## Final decision

- [ ] Keep as candidate revision result
- [ ] Keep only as supplementary / negative result
- [ ] Discard / rollback

## Notes for manuscript or rebuttal

Add 2--5 sentences that could later be pasted into the rebuttal or Discussion.
