# Pipeline Orchestration Scripts

> Scripts for full pipeline execution, cleanup, and automation.

**Location**: `scripts/pipeline/`
**Version Compatibility**: v9.3+
**Last Updated**: 2025-03-08

---

## Script Index

### Full Pipeline

| Script | Version | Size | Description | Duration |
|--------|---------|------|-------------|----------|
| `10_full_pipeline.py` | v9.3 | 46KB | End-to-end pipeline runner | ~12-16 hours |

### Automation & Build

| Script | Version | Size | Description |
|--------|---------|------|-------------|
| `build_article.sh` | v9.3 | 1.1KB | Build LaTeX article |
| `regenerate_report.sh` | v9.3 | varies | Regenerate all figure panels |
| `setup_data.sh` | v6.0+ | varies | Setup data directories |

### Cleanup Utilities

| Script | Version | Size | Description | Frees |
|--------|---------|------|-------------|-------|
| `cleanup_light.sh` | v6.0+ | 1.4KB | Light cleanup (figures, cache) | ~150MB |
| `cleanup_heavy.sh` | v6.0+ | 2.5KB | Heavy cleanup (checkpoints, archives) | ~5GB+ |

---

## Quick Start

### Full Pipeline Run

```bash
# Run entire pipeline (data → train → inference → figures)
python scripts/pipeline/10_full_pipeline.py \
    --start_phase 0 \
    --end_phase 10 \
    --config configs/pipeline.yaml
```

### Figure Regeneration

```bash
# Regenerate all 19 figure panels (A-S)
bash scripts/pipeline/regenerate_report.sh

# Verify article figures
bash scripts/pipeline/verify_article_figures.sh
```

### Cleanup

```bash
# Light cleanup (safe, reversible)
bash scripts/pipeline/cleanup_light.sh

# Heavy cleanup (removes epoch checkpoints)
bash scripts/pipeline/cleanup_heavy.sh
```

### Article Building

```bash
# Build LaTeX article
bash scripts/pipeline/build_article.sh

# Output: articles/clop_dit_biology.pdf
```

---

## Pipeline Phases

| Phase | Script | Description |
|-------|--------|-------------|
| 0 | `data_prep/00_*` | Data preparation |
| 1-3 | `data_prep/0[1-3]_*` | Integration & caching |
| 4 | `training/04a_*` | Training |
| 5-6 | `inference/0[5-6]_*` | Inference |
| 7-8 | `inference/0[7-8]_*` | Analysis & validation |
| 10 | `pipeline/10_*` | Orchestration |

---

## See Also

- [Data Prep Scripts](../data_prep/INDEX.md) - Phase 0-3
- [Training Scripts](../training/INDEX.md) - Phase 4
- [Inference Scripts](../inference/INDEX.md) - Phase 5-6
- [QUICK_START.md](../../docs/operational/QUICK_START.md) - Complete guide
