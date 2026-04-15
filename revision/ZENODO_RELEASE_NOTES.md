# CLOP-DiT v1.0.0 — PeerJ CS pre-revision snapshot

This release archives the CLOP-DiT codebase as submitted to PeerJ Computer
Science in April 2026, **immediately before** the major-revision work
described on branch `revision/major` begins.

It is the permanent diff base for the revision process: every
reviewer-response experiment is compared against the metrics, artifacts,
and manuscript frozen here.

## Contents

The release archive is the repository tree at commit `3b6f38a`, tag
`pre-revision-2026-04-15`. It contains:

- **Source code**: CLOP aligner, diffusion transformer, preprocessing,
  evaluation, and visualization pipelines.
- **Configuration**: `configs/pipeline.yaml`, `configs/clop.yaml`, and
  the baseline registry.
- **Scripts**: training, inference, figure regeneration, and article
  build.
- **Test suite**: `tests/`.
- **Documentation**: `README.md`, `PIPELINE.md`, `REPRODUCIBILITY.md`,
  `CLAUDE.md`, `ZENODO_WORKFLOW.md`.

## What is intentionally NOT in the archive

- Trained model weights (`models/checkpoints/*.pth`)
- Preprocessed cached latents (`data/cached_latents/`)
- Figure outputs (`results/figures/`)
- Raw GEO datasets (`data/processed_h5ad/`)

These are available from the corresponding author on reasonable request.
Their exact integrity at the time of this release is captured by the
SHA-256 manifest in `revision/prerevision_baseline/artifact_hashes.txt`
(189 files), which travels with the archive.

## Companion manuscript

CLOP-DiT: Text-Guided Single-Cell Gene Expression Generation via
Contrastive Language-Omics Pretraining and Diffusion Transformers.
PeerJ Computer Science, submitted April 2026.
