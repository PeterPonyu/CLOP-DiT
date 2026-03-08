# CLOP-DiT Version Manifest

> **Single source of truth for project versioning and file locations**

Last Updated: 2025-03-08

---

## Current Stable Version

| Component | Version | Config File | Status |
|-----------|---------|-------------|--------|
| **CLOP Aligner** | v9.3 | `configs/clop.yaml` | **ACTIVE** |
| **DiT Diffusion** | v2.0 | `configs/dit.yaml` | **ACTIVE** |
| **Cell2Cell** | v1.0 | `configs/cell2cell.yaml` | **ACTIVE** |

### v9.3 Key Changes (Current)
- **Stratified validation split**: ALL 69 cell types now appear in both train and validation (previously only 27/69 in v9.2)
- **Fixed temperature**: Non-learnable temperature at 14.0 (CLIP standard)
- **Deduplicated data**: Uses `use_deduplicated: true` for cleaner training

---

## Version History

| Version | Date | Config Location | Key Innovation | Status |
|---------|------|-----------------|----------------|--------|
| v5.2 | 2024 | `configs/archive/clop_dedup_v5.3.yaml` | Initial InfoNCE baseline | ARCHIVED |
| v6.0 | 2024 | `configs/archive/` | ZCA whitening + SigLIP | ARCHIVED |
| v6.1 | 2024 | `configs/archive/` | Prototype-SigLIP (reduced overfitting) | ARCHIVED |
| v6.2 | 2024 | `configs/archive/` | Enriched captions (FAILED - text anchor drift) | ARCHIVED |
| v6.3 | 2024 | `configs/archive/clop_v6.3_legacy.yaml` | Template anchor + temp clamp | ARCHIVED |
| v7.x | 2024 | `configs/archive/` | Various experimental features | ARCHIVED |
| v8.x | 2024 | `configs/archive/` | Sampler experiments, group fixes | ARCHIVED |
| v9.0 | 2025 | `configs/archive/` | Major architecture refactor | ARCHIVED |
| v9.1 | 2025 | `configs/archive/clop_v9.1.yaml` | Temperature experiments | ARCHIVED |
| v9.2 | 2025 | `configs/archive/clop_v9.2.yaml` | Dataset-level split | ARCHIVED |
| **v9.3** | **2025** | **`configs/clop.yaml`** | **Stratified split (CURRENT)** | **ACTIVE** |

---

## Active vs. Archive File Locations

### Active Configuration Files (configs/)
```
configs/
├── clop.yaml              # CURRENT v9.3 (unversioned = latest)
├── dit.yaml               # DiT diffusion model config
├── ablation_siglip.yaml   # SigLIP ablation study
├── ablation_infonce.yaml  # InfoNCE ablation study
├── cell2cell.yaml         # Cell-to-cell model config
├── models.yaml            # Multi-model presets
├── pipeline.yaml          # Pipeline paths
└── baselines/             # Baseline method configs
```

### Archived Configuration Files
All historical configs are in:
- `configs/archive/` - 13+ historical versions (v6.4 through v9.2)

### Pipeline Scripts (scripts/)
```
scripts/
├── 00_prepare_all_data.py          # Data preparation
├── 01_integrate_h5_datasets.py     # 10x h5 integration
├── 01b_integrate_geodh_h5ad.py   # GEO-DataHub integration
├── 02_subcluster_descriptions.py   # Text generation
├── 02b_enrich_descriptions.py    # Evidence-based enrichment
├── 02d_rebuild_v63_cache.py      # Cache rebuilding
├── 03_cache_latents.py             # Embedding caching
├── 03b_preprocess_embeddings.py    # ZCA whitening
├── 03c_build_dedup_cache.py        # Deduplicated cache
├── 04a_train_clop.py               # Train CLOP
├── 04b_train_dit.py                # Train DiT
├── 04c_train_cell2cell.py          # Train Cell2Cell
├── 05_inference.py                 # Text-conditioned generation
├── 06_cell2cell_inference.py       # Cell-to-cell inference
├── 07_marker_gene_analysis.py      # Marker analysis
├── 08_biological_validation.py     # Bio validation
├── 10_full_pipeline.py             # Full pipeline runner
├── regenerate_report.sh            # One-command figure regen
└── baselines/                        # Baseline training scripts
```

**Note:** `scripts/archive/` contains 22 deprecated scripts from v5-v7 era.

---

## Archive Locations

All archived files are consolidated in:

```
archive/
├── 2025-03-08_configs/         # Historical configs (v5.2 - v9.2)
├── 2025-03-08_scripts/         # Deprecated v5-v7 scripts
├── 2025-03-08_docs/            # Old session reports and planning docs
└── README.md                   # Archive index with restoration guide
```

Previous scattered archive folders (`*/archive/`) have been consolidated here for clarity.

---

## Quick Reference: Which Files to Use

| Task | Use This File | Notes |
|------|---------------|-------|
| Train latest CLOP | `configs/clop.yaml` | This is v9.3, unversioned = current |
| Reproduce old result | `configs/archive/clop_v{VERSION}.yaml` | Pin to specific version |
| Run full pipeline | `scripts/10_full_pipeline.py` | Uses current configs |
| Generate figures | `bash scripts/pipeline/regenerate_report.sh` | Uses current results/ |

---

## Naming Conventions

### Active Files (No Version Numbers)
- `configs/clop.yaml` - Always points to current stable version
- `scripts/##_*.py` - Numbered pipeline scripts
- `src/` modules - No version numbers (evolve in place)

### Archived Files (Versioned with Context)
- `configs/archive/clop_v9.2.yaml` - Historical version, full context
- `archive/2025-03-08_scripts/script_v7.py` - Date + version in filename

---

## Updating This Manifest

When releasing a new version:

1. **Archive current**: Move current `configs/clop.yaml` to `configs/archive/clop_v{X.Y}.yaml`
2. **Promote new**: Copy new version config to `configs/clop.yaml` (unversioned)
3. **Update this file**: Add entry to Version History table, update Current Stable
4. **Commit**: `git add VERSIONS.md configs/clop.yaml configs/archive/`

---

## Contact

For questions about version compatibility or restoration from archive, see:
- `archive/README.md` - Restoration guide
- `docs/DIRECTORY_CLEANUP_POLICY.md` - General organization policy
- `README.md` - Main project documentation
