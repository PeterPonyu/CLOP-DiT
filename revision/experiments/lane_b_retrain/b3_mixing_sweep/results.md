# B3 — Rare-cell mixing strategy sweep

**Reviewer comment:** R3.3 — augmentation strategy space was not
explored (only naive 1x/5x/10x mixing).
**Retraining:** Downstream classifier only; no generator retraining.
**Input artifacts (frozen baseline):**
- Existing CLOP-DiT synthetic samples for ≥2 rare types.
- `revision/prerevision_baseline/metrics_frozen/rare_cell_augmentation/augmentation_results.json`

## Question

When CLOP-DiT-generated samples are combined with classical imbalance
strategies (random oversampling, SMOTE, hybrid), does rare-class F1
improve?

## Deliverables

Matrix for each chosen rare type (≥ 2 of: Cycling cells,
stress-response cells, progenitor-like, erythroid progenitors):

| Strategy | 1× | 2× | 5× | 10× |
|---|---|---|---|---|
| Baseline (no aug) | | | | |
| Random oversampling | | | | |
| SMOTE | | | | |
| CLOP-DiT only | | | | |
| CLOP-DiT + oversampling | | | | |
| CLOP-DiT + SMOTE | | | | |

Metrics per cell: rare-class F1, macro-F1, precision, recall,
(optional) calibration.

## Plan

- Reuse existing CLOP-DiT synthetic samples where possible; generate
  more if a new rare type is added.
- Implement all strategies behind a common classifier-training
  interface for fair comparison.
- Save per-strategy predictions under
  `experiments/lane_b_retrain/b3_mixing_sweep/<type>/<strategy>_<ratio>/`.

## Status

- [ ] Rare types selected and synthetic samples gathered
- [ ] All 6 strategies wired behind shared interface
- [ ] Sweep complete for type 1
- [ ] Sweep complete for type 2
- [ ] Matrix drafted
- [ ] Rebuttal verdict sentence committed

## Results

_(fill in after the runs complete)_

## Rebuttal-ready sentence

_(2–3 sentences that can be pasted into the R3.3 response)_
