# B4 — CLOP ablation → full pipeline bridge

**Reviewer comment:** R2.10 — CLOP ablations run under a different
configuration from the production model, so their conclusions do not
necessarily transfer.
**Retraining:** 2–3× DiT training on selected CLOP variants.
**Input artifacts:** existing CLOP ablation checkpoints + frozen DiT
training pipeline.

## Question

Do the 15 existing CLOP ablations actually transfer to the full
pipeline, or are their conclusions artefacts of the CLOP-only
evaluation?

## Deliverables

1. **Selected variants**: `baseline`, `no_cohesion`, `no_cell_noise`
   (and optionally `fixed_temperature`). Chosen because they have the
   largest claimed Stage-1 effect.
2. **Full-pipeline metrics** per variant: KNN-1, Steering, DivR,
   centroid cosine.
3. **Verdict** per variant: does the ablation's direction match Stage-1
   (confirmed), reverse (failure of Stage-1 as a proxy), or disappear?

## Plan

- For each selected CLOP variant, retrain DiT with identical config
  except for the CLOP checkpoint.
- Generate embeddings + decode expression using the same pathway used
  in the headline metrics.
- Compute metrics and compare to both the frozen baseline and the
  Stage-1 ablation report.

## Status

- [ ] Variants selected and CLOP checkpoints confirmed
- [ ] 3× DiT trained
- [ ] Metrics computed for all variants
- [ ] Transfer verdict committed
- [ ] Rebuttal paragraph drafted

## Results

_(fill in after the runs complete)_

## Rebuttal-ready sentence

_(2–3 sentences that can be pasted into the R2.10 response)_
