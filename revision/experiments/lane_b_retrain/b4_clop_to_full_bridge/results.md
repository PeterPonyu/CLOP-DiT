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

- [x] Variants selected and CLOP checkpoints confirmed
- [x] 3× DiT trained
- [x] Metrics computed for all variants
- [x] Transfer verdict committed
- [x] Rebuttal paragraph drafted

## Results

Three end-to-end DiT retrains were completed on selected CLOP ablation
checkpoints: `abl_baseline`, `no_cohesion`, and `no_cell_noise`.

| variant | Stage-1 proto_acc | Stage-2 FD | coverage | centroid cos |
|---|---:|---:|---:|---:|
| abl_baseline | 0.7617 | 0.1750 | 0.1440 | **0.9291** |
| no_cohesion | **0.8641** | 0.2303 | 0.0790 | 0.8535 |
| no_cell_noise | 0.8623 | **0.1450** | 0.1206 | 0.9232 |

Transfer verdict: **PARTIAL_REVERSAL**.

1. `no_cohesion` looks best at Stage 1 but becomes the worst
   end-to-end generator by every downstream quality metric.
2. `no_cell_noise` partially transfers: FD improves versus the
   ablation baseline, but centroid cosine is slightly worse.
3. Stage-1 CLOP rankings are therefore not a reliable proxy for final
   generation quality; the cohesion loss carries downstream geometric
   structure that DiT uses even when it slightly hurts prototype
   retrieval.

Supporting artefacts:
`revision/experiments/b4_clop_bridge/b4_bridge_comparison.json` and
the Phase 1-4 summary in `revision/REVISION_LOG.md`.

## Rebuttal-ready sentence

We directly bridged the CLOP ablation suite to the full pipeline by
retraining DiT on selected ablation checkpoints and re-evaluating the
generated outputs end-to-end. The strongest Stage-1 variant
(`no_cohesion`, prototype accuracy 0.864 vs 0.762 for the ablation
baseline) became the weakest generator downstream, with worse
Frechet distance, coverage, and centroid cosine, while `no_cell_noise`
showed only a small, mixed transfer. We therefore narrow the original
claim: Stage-1 CLOP ablation rankings are not a reliable proxy for
end-to-end generation quality, and reviewer-facing conclusions should
be drawn from the bridged full-pipeline results instead.
