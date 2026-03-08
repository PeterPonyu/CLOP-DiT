# CLOP-DiT Archive Index

> Central index for all archived project artifacts. Archives are distributed by type but indexed here for easy discovery.

Last Updated: 2025-03-08

---

## Archive Philosophy

Archives contain historical artifacts from previous development iterations (v5.2 through v9.2). These are kept for:
- **Reproducibility**: Old experiments can be referenced
- **Debugging**: Comparing against previous versions
- **History**: Understanding the evolution of the codebase

**Important**: Archives are `.gitignore`d and not required for current development or reproduction of published results.

---

## Archive Locations

### 1. Configuration Archives (`configs/archive/`)
**Location**: `../configs/archive/`

| File | Version | Era | Notes |
|------|---------|-----|-------|
| `clop_dedup_v5.3.yaml` | v5.3 | Legacy | Initial InfoNCE baseline |
| `clop_v6.3_legacy.yaml` | v6.3 | Legacy | Template anchor + temp clamp |
| `clop_v6.4.yaml` | v6.4 | Legacy | Post-v6.3 refinements |
| `clop_v641.yaml` | v6.4.1 | Legacy | Minor v6.4 fix |
| `clop_v7.yaml` | v7.0 | Legacy | Major refactor |
| `clop_v71.yaml` | v7.1 | Legacy | v7 refinements |
| `clop_v72.yaml` | v7.2 | Legacy | v7 refinements |
| `clop_v8.1.yaml` | v8.1 | Legacy | Sampler experiments |
| `clop_v8.1_groupfix.yaml` | v8.1 | Legacy | Group fix variant |
| `clop_v8.2.yaml` | v8.2 | Legacy | Continued experiments |
| `clop_v8.3.yaml` | v8.3 | Legacy | Continued experiments |
| `clop_v8_sampler.yaml` | v8.0 | Legacy | Sampler focus |
| `clop_v9.yaml` | v9.0 | Legacy | Major v9 refactor |
| `clop_v9.1.yaml` | v9.1 | Legacy | Temperature experiments |
| `clop_v9.2.yaml` | v9.2 | Legacy | Dataset-level split |
| `clop_v9.3.yaml` | v9.3 | Current | Stratified split (backed up) |

**Total**: 16 configuration files

---

### 2. Script Archives (`scripts/archive/`)
**Location**: `../scripts/archive/`

Contains 22 deprecated Python scripts from v5-v7 era development:

| Script | Era | Purpose | Status |
|--------|-----|---------|--------|
| `02c_build_enriched_cache.py` | v6 | Legacy cache building | Superseded |
| `09_train_and_compare_v7.py` | v7 | Training comparison | Superseded |
| `11_jbhi_enhanced_eval.py` | v5 | JBHI evaluation | Superseded |
| `12_compose_and_check.py` | v5 | Figure composition | Superseded |
| `13_tier_comparison_fixed.py` | v6 | Tier comparison | Superseded |
| `14_biological_evaluation.py` | v6 | Bio evaluation | Superseded |
| `14b_cfg_sweep.py` | v6 | CFG sweep | Superseded |
| `15_final_verified_evaluation.py` | v6 | Final eval | Superseded |
| `16_publication_figures.py` | v6 | Figure generation | Superseded |
| `17_baseline_comparison.py` | v6 | Baseline comparison | Superseded |
| `audit_clop_data_pairing.py` | v6 | Data audit | Superseded |
| `compare_v63_v641.py` | v6 | Version comparison | Historical |
| `evaluate_v63.py` | v6 | Evaluation | Superseded |
| `example_use_dedup_data.py` | v5 | Deduplication example | Superseded |
| `manage_clop_experiments.py` | v6 | Experiment mgmt | Superseded |
| `monitor_v7.py` | v7 | Monitoring | Superseded |
| `polish_texts.py` | v6 | Text polishing | Superseded |
| `prepare_dedup_captions_for_training.py` | v5 | Caption prep | Superseded |
| `rectify_flow.py` | v6 | Flow rectification | Superseded |
| `re_embed_polished_texts.py` | v6 | Re-embedding | Superseded |
| `reembed_v2.py` | v6 | Re-embedding v2 | Superseded |
| `visualize_v63.py` | v6 | Visualization | Superseded |

**Total**: 22 scripts

---

### 3. Documentation Archives (`docs/archive/`)
**Location**: `../docs/archive/`

Contains 6 historical session reports and planning documents:

| Document | Date | Purpose |
|----------|------|---------|
| `CLEANUP_SUMMARY.md` | 2026-03-02 | Historical cleanup notes |
| `CLOP_DIAGNOSIS_AND_PLAN_2026-03-02.md` | 2026-03-02 | Session diagnosis |
| `CLOP_EXPERIMENTS.md` | 2026-03-02 | Experiment notes |
| `ENHANCEMENTS_SUMMARY.md` | 2026-03-02 | Enhancement notes |
| `SESSION_REPORT_2026-03-02.md` | 2026-03-02 | Session report |
| `TEXT_POLISHING_SUMMARY.md` | 2026-03-02 | Text polishing notes |

**Total**: 6 documents

---

## Restoration Guide

### To restore an old configuration:

```bash
# Copy from archive to active
 cp configs/archive/clop_v9.2.yaml configs/clop_v9.2_restored.yaml

# Edit to use restored config in training
 python scripts/04a_train_clop.py --config configs/clop_v9.2_restored.yaml
```

### To reference an old script:

```bash
# View script contents
cat scripts/archive/14_biological_evaluation.py

# Copy and modify if needed
cp scripts/archive/14_biological_evaluation.py scripts/14_bio_eval_restored.py
```

### To check historical documentation:

```bash
# View archived docs
cat docs/archive/CLOP_DIAGNOSIS_AND_PLAN_2026-03-02.md
```

---

## Cleanup Commands

To free space by removing archives:

```bash
# Remove all archives (irreversible - use with caution)
rm -rf configs/archive/*
rm -rf scripts/archive/*
rm -rf docs/archive/*
```

**Note**: Archives are not tracked in git (`.gitignore`d), so removing them only affects your local workspace. They can be regenerated or retrieved from git history if needed.

---

## Version Correspondence

| Version | Config | Scripts | Docs |
|---------|--------|---------|------|
| v5.2 | `configs/archive/clop_dedup_v5.3.yaml` | Various in scripts/archive/ | - |
| v6.x | `configs/archive/clop_v6*.yaml` | `02c_*, 11_*, 12_*, 13_*, 14_*, 15_*, 16_*, 17_*` | See docs/archive/ |
| v7.x | `configs/archive/clop_v7*.yaml` | `09_*, monitor_v7.py` | - |
| v8.x | `configs/archive/clop_v8*.yaml` | - | - |
| v9.x | `configs/archive/clop_v9*.yaml` | - | - |
| **v9.3 (current)** | `configs/clop.yaml` | `scripts/04a_train_clop.py` | See docs/INDEX.md |

---

## Contact

For questions about specific archived artifacts:
1. Check [VERSIONS.md](../VERSIONS.md) for version history
2. Check [DIRECTORY_CLEANUP_POLICY.md](../docs/DIRECTORY_CLEANUP_POLICY.md) for cleanup policies
3. Check [docs/INDEX.md](../docs/INDEX.md) for current documentation
