# Lane A — Post-hoc analysis (no retraining)

All experiments in this lane are derived from artifacts that already exist
under `data/cached_latents/` and `results/` as of the pre-revision baseline.
No model retraining, no new generation runs. This lane is the lowest-risk,
highest-reviewer-value work and should be completed first.

## Experiments

| ID | Question | Reviewer | Primary output |
|---|---|---|---|
| **A1** | When the KNN classifier misassigns a generated cell, which cell type does it pick? Are errors within related lineages (graceful) or across lineages (catastrophic)? | 2.6 | `a1_knn_confusion/` — family-level confusion table + top-10 confused type pairs |
| **A2** | Does pooled evaluation hide a systematic gap between human (59 datasets) and mouse (21 datasets) performance? | 3.2 | `a2_organism_split/` — organism-stratified KNN / steering / DivR / centroid cosine |
| **A3** | Is "biologically difficult" really "underrepresented"? Quantify per-type fidelity as a function of training cell count and fraction. | 2.7 | `a3_abundance_fidelity/` — correlation table + abundance-stratified metrics |
| **A4** | The supplementary figures show residual and HVG variance ratio but the main text never quantifies them. Add a numeric summary. | 2.8 | `a4_hvg_variance/` — fraction within tolerance, fraction with >2× variance deficit |
| **A5** | The rare-cell augmentation pilot failed. Is the failure upstream (latent under-dispersion) or downstream (decoder compression)? | 2.11 | `a5_rare_failure_mechanism/` — latent vs expression variance ratio, centroid shift decomposition |

## Ground rules

- **Do not mutate `data/cached_latents/` or `models/checkpoints/`.** All
  analyses read artifacts, never write to them.
- **Do not regenerate figures into `results/figures/` or `articles/figures/`.**
  Emit outputs to the experiment's own subdirectory; article delivery
  happens later during manuscript editing.
- **Each experiment produces a `results.md`** filled in from
  `performance_tracking/comparison_template.md`, plus any tables / figure
  PDFs / JSON payloads produced by the analysis.
- **Cross-reference the frozen baseline**
  (`revision/prerevision_baseline/metrics_frozen/…`) rather than the live
  `results/` copy when reporting "baseline value".
