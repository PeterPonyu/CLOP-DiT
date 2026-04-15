# B1 — Gaussian / unconditional r_var baseline

**Reviewer comment:** R2.9 — near-zero `r_var` has no baseline to
compare against.
**Retraining:** Inference only (no training).
**Input artifacts (frozen baseline):**
- `models/checkpoints/DiT/` (read-only)
- `data/cached_latents/` (read-only)

## Question

Is near-zero gene-wise variance correlation a CLOP-DiT–specific failure,
or is it a shared limitation of every latent-generation + frozen-decoder
pipeline? Without a baseline the current reported `r_var` is
uninterpretable.

## Deliverables

Per-dataset (5 biological validation datasets) table with three
generators:

| Generator | r_mean | R² | r_var | FD_gene | FD_embed |
|---|---|---|---|---|---|
| CLOP-DiT (CFG=2) | (baseline) | | | | |
| CFG=0 unconditional | — | | | | |
| Gaussian per-type (μ, Σ) | — | | | | |

Plus a verdict paragraph: is CLOP-DiT's r_var worse / comparable /
better than the Gaussian floor?

## Plan

- Generate unconditional samples via CFG=0.
- Generate Gaussian samples by fitting per-type mean and covariance on
  the training latents, sampling from them, then decoding with the same
  scGPT pathway.
- Compute the full metric set on identical held-out datasets.
- Save outputs under `models/revision/b1_gaussian_baseline/` (do **not**
  overwrite the frozen baseline checkpoints).

## Status

- [ ] CFG=0 generation complete
- [ ] Gaussian-per-type generation complete
- [ ] Metrics computed for all three generators
- [ ] Comparison table drafted
- [ ] Rebuttal verdict sentence committed

## Results

_(fill in after the runs complete)_

## Rebuttal-ready sentence

_(2–3 sentences that can be pasted into the R2.9 response)_
