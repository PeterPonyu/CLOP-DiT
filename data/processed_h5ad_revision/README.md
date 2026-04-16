# processed_h5ad_revision

This directory is reserved for the **revision-only Lane C strict-OOD**
corpus. It must remain separate from `data/processed_h5ad/` so the
pre-revision baseline and all A/B/D analyses stay reproducible.

## Purpose

Lane C may reopen only for the narrowed **Priority-2 strict-OOD**
experiment. If that happens, all newly curated or re-split datasets must
land here rather than mutating the baseline corpus.

## Required files

- `MANIFEST.csv`
  - one row per revision-side dataset artifact
  - used by `scripts/check_strict_ood.py` to detect train/eval leakage
- `label_bridge.csv`
  - vocabulary reconciliation sheet for dataset labels vs the current
    69-type target vocabulary

## MANIFEST.csv contract

Expected columns:

- `dataset_id`
- `source_path`
- `organism`
- `tissue`
- `split`
- `priority`
- `notes`

`split` is expected to use values such as `train`, `val`, `test`,
`eval`, or `heldout`.

For strict-OOD tissues, the same tissue must **never** appear in a row
with a training split.

## label_bridge.csv contract

Expected columns:

- `dataset_id`
- `source_label`
- `target_label`
- `status`
- `reviewer_a`
- `reviewer_b`
- `notes`

Suggested `status` values:

- `proposed`
- `reviewed`
- `approved`
- `novel`

## Baseline comparison source

All Lane C comparisons must stay anchored to the frozen prerevision
baseline:

- tag: `pre-revision-2026-04-15`
- metrics snapshot: `revision/prerevision_baseline/metrics_frozen/`

## Current status

This directory currently contains **scaffolding only**. Its existence
does **not** mean Lane C has started. The staffing/readiness gate must
still be satisfied before any new data is curated or retraining begins.
