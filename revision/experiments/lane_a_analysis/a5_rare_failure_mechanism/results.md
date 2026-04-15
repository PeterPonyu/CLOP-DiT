# A5 — Rare-cell augmentation failure mechanism

**Reviewer comment:** R2.11 — the rare-cell augmentation failed; what
is the mechanism?
**Retraining:** None.
**Input artifacts (frozen baseline):**
- `revision/prerevision_baseline/metrics_frozen/rare_cell_augmentation/augmentation_results.json`
- Existing generated latents + expressions for the Cycling-cell pilot.

## Question

Is the augmentation failure upstream (latent under-dispersion — the
generator produces near-centroid samples) or downstream (decoder
compression — the decoder averages out latent variation)?

## Deliverables

1. **Within-type variance comparison** (Cycling cells, real vs
   generated):
   - latent-space within-type variance ratio
   - expression-space within-type variance ratio
   - decoder expansion / compression factor
2. **Centroid shift decomposition** — does the generator reproduce the
   correct centroid, the correct spread, both, or neither?
3. **Mechanism verdict** — upstream, downstream, or both. Tie directly
   into the answer to R2.11 and motivate Lane B3 (mixing-strategy
   sweep).

## Plan

- Pull Cycling-cell real and generated samples from the frozen artifacts.
- Compute within-type covariance at both the latent and expression
  level.
- Report the variance ratio and a short mechanism paragraph.

## Status

- [ ] Cycling-cell real + generated slices assembled
- [ ] Latent within-type variance computed
- [ ] Expression within-type variance computed
- [ ] Decoder expansion/compression factor computed
- [ ] Mechanism verdict committed
- [ ] Rebuttal verdict sentence committed

## Results

_(fill in after the analysis runs)_

## Rebuttal-ready sentence

_(2–3 sentences that can be pasted into the R2.11 response)_
