# CLOP-DiT — pre-revision snapshot (2026-04-15)

This release archives the CLOP-DiT codebase immediately before the
major-revision work described on branch `revision/major` begins.

It is the permanent diff base for the revision process: every
reviewer-response experiment is compared against the metrics, artifacts,
and documentation frozen here.

## Contents

The release archive is the repository tree at tag
`pre-revision-2026-04-15`. It contains:

- **Source code**: CLOP aligner, diffusion transformer, preprocessing,
  evaluation, and visualization pipelines.
- **Configuration**: `configs/pipeline.yaml`, `configs/clop.yaml`, and
  the baseline registry.
- **Scripts**: training, inference, figure regeneration, and pipeline
  orchestration.
- **Test suite**: `tests/`.
- **Documentation**: `README.md`, `PIPELINE.md`, `REPRODUCIBILITY.md`,
  `CLAUDE.md`.
- **Revision workspace scaffolding**: `revision/README.md`, the frozen
  pre-revision baseline (`revision/prerevision_baseline/`), and the
  eleven experiment stubs (`revision/experiments/`).

## What is intentionally NOT in the archive

- Trained model weights (`models/checkpoints/*.pth`)
- Preprocessed cached latents (`data/cached_latents/`)
- Figure outputs (`results/figures/`)
- Raw GEO datasets (`data/processed_h5ad/`)
- Venue-specific manuscript sources (local-only under
  `revision/manuscripts/`, gitignored)

Artifact provenance at the time of this release is captured by the
SHA-256 manifest in
`revision/prerevision_baseline/artifact_hashes.txt` (189 files). The
manifest travels with the archive so that any future comparison can
verify whether baseline artifacts have drifted.

## Headline metrics at freeze time

See `revision/prerevision_baseline/baseline_snapshot_2026-04-15.md` for
the full table. Briefly, CLOP-DiT at CFG = 2.0 reaches KNN-1 = 0.369
(25× above random on 69 classes), steering accuracy = 0.810,
diversity ratio = 0.513, and linear-classifier accuracy = 0.511,
against a Gaussian mean-matching baseline of KNN-1 = 0.011.
