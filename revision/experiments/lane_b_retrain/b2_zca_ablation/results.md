# B2 — ZCA whitening formal ablation

**Reviewer comment:** R3.4 — ZCA claim is anecdotal; formal ablation
required.
**Retraining:** 4× CLOP training + 2–3× DiT training.
**Input artifacts:** `data/cached_latents/` (re-preprocessed per variant).

## Question

Does ZCA whitening actually beat mean-centering, LayerNorm, and raw
input in an end-to-end pipeline, or was the preference based on
preliminary Stage-1 checks only?

## Deliverables

1. **Stage-1 metrics** (all four preprocessing variants):
   - validation prototype accuracy
   - diagonal vs off-diagonal cosine gap
   - centroid separation
2. **Full-pipeline metrics** (select 2–3 variants for the full DiT
   retrain, typically: ZCA / mean-only / LayerNorm):
   - KNN-1 / KNN-5 / Steering / DivR / centroid cosine
3. **Verdict paragraph**: is ZCA's advantage preserved end-to-end, or
   does it disappear once DiT + decoder are added?

## Plan

- Create 4 parallel CLOP preprocessing variants under
  `models/revision/b2_zca/{zca, mean_only, layernorm, raw}/`.
- Train CLOP + DiT per variant (log wall-clock + seed + config hash).
- Compute Stage-1 and full-pipeline metrics on each.
- Keep the frozen baseline's ZCA checkpoint untouched for direct
  comparison.

## Status

- [ ] Variant preprocessing scripts ready
- [ ] 4× CLOP trained
- [ ] Stage-1 metrics collected
- [ ] 2–3× DiT trained
- [ ] Full-pipeline metrics collected
- [ ] Verdict paragraph committed

## Results

_(fill in after the runs complete)_

## Rebuttal-ready sentence

_(2–3 sentences that can be pasted into the R3.4 response)_
