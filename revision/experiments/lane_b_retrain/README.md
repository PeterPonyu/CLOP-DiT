# Lane B — Partial retrain / baseline calibration

Experiments in this lane require new model runs or baseline computations
but do not require new training data or a new text encoder. Each
experiment in this lane must state explicitly which artifact it is
regenerating and save the regenerated outputs to its own subdir (never
overwrite the frozen baseline).

## Experiments

| ID | Question | Reviewer | Retraining cost | Primary output |
|---|---|---|---|---|
| **B1** | Is the near-zero `r_var` a CLOP-DiT-specific failure, or shared by Gaussian / unconditional baselines? | 2.9 | Inference only (hours) | Per-dataset `r_mean`, `r_var`, `FD_gene` for `{CLOP-DiT, CFG=0, Gaussian-per-type}` |
| **B2** | Does ZCA whitening actually beat mean-centering / LayerNorm / raw in an end-to-end comparison? | 3.4 | 4× CLOP train + 2–3× DiT train | Stage-1 val prototype accuracy + full-pipeline KNN / Steering / DivR per preprocessing variant |
| **B3** | When CLOP-DiT-generated samples are combined with classical mixing strategies (oversampling / SMOTE / hybrid), does rare-class F1 improve? | 3.3 | Downstream classifier training only | Rare-class F1 at 1× / 2× / 5× / 10× for ≥2 rare types |
| **B4** | Do the 15 CLOP ablations transfer their conclusions to the full DiT+decoder pipeline, or were they CLOP-internal artefacts? | 2.10 | 2–3× DiT train on selected CLOP variants | KNN / Steering / DivR for `{baseline, no_cohesion, no_cell_noise}` end-to-end |

## Ground rules

- **Every retrain must be logged.** Record seed, config hash, git commit
  at run time, and wall-clock start/end. Save the log as `run_log.txt` in
  the experiment subdir.
- **Baseline comparison is mandatory.** Each experiment's `results.md`
  must include a side-by-side table with the frozen baseline value from
  `revision/prerevision_baseline/metrics_frozen/`.
- **Do not overwrite `models/checkpoints/`.** Put new checkpoints under
  `models/revision/<experiment_id>/` so the SHA256 manifest still
  verifies. (Create that directory when the first retrain starts.)
