# Lane D — Alternative text-encoder comparison

**Positioning:** Supplementary robustness evidence. This lane is not the
first priority of the revision; it exists to answer "is the bottleneck
the text encoder?" and to strengthen R2.2 / R2.3 / R2.5.

## Encoders to compare

Keep the set small and representative — reviewers do not reward a 5-way
encoder sweep.

1. **BiomedBERT** (current baseline)
2. **PubMedBERT** — closer biomedical pretraining distribution
3. **BioLinkBERT** — biomedical relation/context-aware

## Metrics to report (per encoder)

1. **Stage-1 alignment** — validation prototype accuracy,
   diagonal / off-diagonal cosine gap, centroid separation.
2. **Main generation** — KNN-1, KNN-5, Steering, DivR,
   centroid cosine.
3. **Semantic sensitivity** — field-ablation drop, external-marker
   substitution, swap-label test.
4. **OOD prompt robustness** — novel-prompt marker hit rate,
   free-form vs structured expression cosine.

## What a positive result looks like

- Alternative encoder reduces the field-ablation drop or boosts OOD
  marker hit without degrading main generation metrics.

## What a negative result looks like (still useful)

- All encoders give within-noise results → strong evidence the bottleneck
  is **not** the text encoder, which directly answers R2.3's
  memorization concern.

## Ground rules

- Do not alter the CLOP architecture or DiT config between encoder runs.
  Only the encoder module swaps.
- Save each encoder variant's run under `experiments/lane_d_encoder/<encoder_name>/`.
- Always report the baseline (BiomedBERT) value pulled from
  `revision/prerevision_baseline/metrics_frozen/` — do **not** recompute
  it from the live `results/` folder.
