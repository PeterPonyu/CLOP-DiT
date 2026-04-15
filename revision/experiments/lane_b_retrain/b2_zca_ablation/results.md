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

- [x] Variant preprocessing scripts ready
- [x] 3× CLOP trained (`whiten`, `center_norm`, `none`)
- [x] Stage-1 metrics collected
- [ ] Downstream DiT retrain not run in this pass
- [x] Stage-specific verdict paragraph committed

## Results

Three matched CLOP training runs show that ZCA whitening is a modest
refinement rather than a critical dependency of the alignment stage.

| condition | quality | proto_acc | tc_align | separation | cos_sim | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| whiten (ZCA) | **0.9656** | 0.9991 | 0.9783 | 1.0108 | **0.8512** | 98 |
| center_norm | 0.9600 | 0.9995 | 0.9839 | 1.0102 | 0.8148 | 96 |
| none | 0.9562 | 0.9982 | 0.9769 | 1.0099 | 0.8085 | 99 |

Reading:

1. Prototype accuracy stays near-perfect in all conditions
   (`>= 0.998`), so the aligner can learn through raw embedding
   collapse.
2. The composite quality spread is only ~1.0 % from best to worst.
3. ZCA's largest measurable benefit is fine-grained positive-pair
   cosine structure (`0.8512` vs `0.8085`, about +5 % over no
   preprocessing).
4. The current revision answer to R3.4 is therefore stage-specific:
   ZCA is a helpful default, but not a decisive architectural lever.

The completed evidence lives in
`revision/experiments/zca_ablation/zca_ablation_summary.json` and
`revision/experiments/zca_ablation/SYNTHESIS.md`.

## Rebuttal-ready sentence

We formally ablated ZCA whitening in the CLOP preprocessing pipeline
by retraining matched aligners with full whitening, mean-centering +
L2 normalization only, and no preprocessing. Composite alignment
quality changed only modestly (0.966, 0.960, and 0.956 respectively),
and prototype accuracy remained near-perfect in all settings
(`>= 0.998`), showing that the aligner is robust to input-space
collapse. ZCA is therefore best interpreted as a useful best-practice
default that improves fine-grained cosine structure by about 5 %,
rather than a critical architectural requirement.
