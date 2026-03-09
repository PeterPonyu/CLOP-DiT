# CLOP-DiT Version Manifest

> Single source of truth for project versioning

Last Updated: 2026-03-09

---

## Current Stable Versions

| Component | Version | Config File | Status |
|-----------|---------|-------------|--------|
| **CLOP Aligner** | v9.3 | `configs/clop.yaml` | **ACTIVE** |
| **DiT Diffusion** | v2.0 | `configs/dit.yaml` | **ACTIVE** |

### v9.3 Key Changes
- Stratified validation split: all 69 cell types in both train and validation
- Fixed temperature: non-learnable at 14.0 (CLIP standard)
- Deduplicated data: `use_deduplicated: true`

---

## Version History

| Version | Config Location | Key Innovation | Status |
|---------|-----------------|----------------|--------|
| v5.2 | `configs/archive/` | Initial InfoNCE baseline | ARCHIVED |
| v6.0 | `configs/archive/` | ZCA whitening + SigLIP | ARCHIVED |
| v6.1 | `configs/archive/` | Prototype-SigLIP | ARCHIVED |
| v9.0 | `configs/archive/` | Architecture refactor | ARCHIVED |
| v9.1 | `configs/archive/` | Temperature experiments | ARCHIVED |
| v9.2 | `configs/archive/` | Dataset-level split | ARCHIVED |
| **v9.3** | **`configs/clop.yaml`** | **Stratified split** | **ACTIVE** |

---

## Quick Reference

| Task | Command |
|------|---------|
| Train latest CLOP | `python scripts/training/04a_train_clop.py --config configs/clop.yaml` |
| Train DiT | `python scripts/training/04b_train_dit.py --config configs/dit.yaml` |
| Run full pipeline | `python scripts/pipeline/run_pipeline.py --stage all` |
| Generate figures | `bash scripts/pipeline/regenerate_report.sh` |

---

## Naming Conventions

- **Active configs** have no version number: `configs/clop.yaml` = current stable
- **Archived configs** include version: `configs/archive/clop_v9.2.yaml`
- **Source modules** in `src/` evolve in place without version numbers
