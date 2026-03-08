# CLOP-DiT Scripts Index

> Master index for all pipeline and utility scripts.

**Location**: `scripts/`
**Total Scripts**: 45
**Last Updated**: 2025-03-08

---

## Directory Structure

| Subdirectory | Phase | Scripts | Purpose |
|--------------|-------|---------|---------|
| [data_prep/](data_prep/INDEX.md) | 0-3 | 9 | Data preparation and caching |
| [training/](training/INDEX.md) | 4 | 4 | Model training |
| [inference/](inference/INDEX.md) | 5-6 | 4 | Generation and validation |
| [analysis/](analysis/INDEX.md) | - | 9 | Analysis and diagnostics |
| [pipeline/](pipeline/INDEX.md) | 10 | 6 | Orchestration and automation |
| [vcd/](vcd/) | - | 12 | Visual Conflict Detection |
| [baselines/](baselines/) | - | 2 | Baseline methods |
| [archive/](archive/) | - | 22 | Deprecated scripts |

---

## Quick Reference

### Most Common Scripts

| Task | Script |
|------|--------|
| Full pipeline | `python scripts/pipeline/10_full_pipeline.py` |
| Train CLOP | `python scripts/training/04a_train_clop.py` |
| Generate cells | `python scripts/inference/05_inference.py` |
| Regenerate figures | `bash scripts/pipeline/regenerate_report.sh` |
| Clean workspace | `bash scripts/pipeline/cleanup_light.sh` |

### Script Organization

All scripts follow phase-based numbering:
- `00_*` - Initial preparation
- `01_*`, `01b_*` - Dataset integration variants
- `02_*`, `02b_*`, `02d_*` - Text generation variants
- `03_*`, `03b_*`, `03c_*` - Caching variants
- `04a_*`, `04b_*`, `04c_*` - Training variants
- `05_*` - Inference
- `06_*` - Evaluation
- `07_*` - Marker analysis
- `08_*` - Biological validation
- `10_*` - Full pipeline

---

## Execution Flow

```
Data Prep (0-3) → Training (4) → Inference (5-6) → Analysis (7-8)
     ↓                  ↓              ↓                 ↓
[data_prep/]      [training/]    [inference/]      [analysis/]
```

---

## Version Compatibility

| Directory | Min Version | Current |
|-----------|-------------|---------|
| data_prep/ | v5.2 | v9.3 |
| training/ | v6.0 | v9.3 |
| inference/ | v6.0 | v9.3 |
| analysis/ | v6.0 | v9.3 |
| pipeline/ | v6.0 | v9.3 |

See [VERSIONS.md](../VERSIONS.md) for detailed version mapping.

---

## See Also

- [Docs: QUICK_START.md](../docs/operational/QUICK_START.md) - Complete workflow
- [Docs: DIRECTORY_CLEANUP_POLICY.md](../docs/operational/DIRECTORY_CLEANUP_POLICY.md) - Cleanup guidelines
- [Configs](../configs/) - Configuration files
