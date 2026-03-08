# Directory Cleanup Policy

This document defines the directory organisation, what is tracked in Git, what is generated, and how to clean up the workspace.

## Directory Classification

| Directory | Category | In Git | Size | Notes |
|---|---|---|---|---|
| `src/` | **Source code** | Yes | ~1.5 MB | Core library — always tracked |
| `scripts/` | **Pipeline scripts** | Yes | ~1.2 MB | Excludes `scripts/archive/` |
| `configs/` | **Active configs** | Yes | ~32 KB | Excludes `configs/archive/` |
| `tests/` | **Unit tests** | Yes | ~128 KB | Always tracked |
| `docs/` | **Documentation** | Yes | ~96 KB | Excludes archive/intermediate |
| `articles/` | **Manuscript drafts** | Yes | ~200 KB | MDPI Biology LaTeX template; can be gitignored if preferred |
| `arch-figure/` | **Architecture fig** | Yes | ~432 KB | Nuxt.js diagram source |
| `data/cached_latents_v5.2/` | **Cached latents** | No | ~3.9 GB | Regenerate: `03_cache_latents.py` |
| `data/processed_h5ad/` | **Processed data** | No | ~1.7 GB | Regenerate: `01_integrate_h5_datasets.py` |
| `models/checkpoints/` | **Model weights** | No | ~4.8 GB | Regenerate: train CLOP + DiT |
| `models/scgpt_*/` | **External models** | No | ~396 MB | Download from scGPT repo |
| `results/` | **All results** | No | ~2.4 GB | Regenerate: full pipeline |
| `results/figures/` | **Report panels** | No | ~146 MB | Regenerate: `regenerate_report.sh`; single source for generated figures |
| `results/archive/` | **Old experiments** | No | ~1.8 GB | Safe to delete |
| `figures/` (repo root) | **Deprecated** | No | — | Legacy; not part of pipeline. Use `results/figures/` for output; `articles/figures/` for article symlinks. Ignored in `.gitignore`. |
| `logs/` | **Training logs** | No | ~5.8 MB | TensorBoard / CSV logs |
| `configs/archive/` | **Old configs** | No | ~56 KB | Historical configs |
| `docs/archive/` | **Old docs** | No | ~68 KB | AI session reports |
| `docs/intermediate_analysis/` | **Analysis logs** | No | ~272 KB | Session artifacts |
| `scripts/archive/` | **Old scripts** | No | ~476 KB | Superseded scripts |

## Cleanup Commands

### Light cleanup (~150 MB freed)
Removes only regenerable outputs — figures, benchmark report, caches:
```bash
bash scripts/cleanup_light.sh
```

### Heavy cleanup (~5+ GB freed)
Removes intermediate checkpoints, archives, generated data:
```bash
bash scripts/cleanup_heavy.sh
```

### Data setup / verification
Checks all required data and models are present:
```bash
bash scripts/setup_data.sh
```

## Rebuild Commands

| What | Command |
|---|---|
| Cached latents | `conda run -n dl python scripts/03_cache_latents.py` then `03b_preprocess_embeddings.py` |
| CLOP model | `conda run -n dl python scripts/04a_train_clop.py` |
| DiT model | `conda run -n dl python scripts/04b_train_dit.py` |
| Inference | `conda run -n dl python scripts/05_inference.py` |
| Evaluation | `conda run -n dl python scripts/06_evaluate.py` |
| All 19 panels | `bash scripts/regenerate_report.sh` |
| Full retrain | `bash scripts/full_retrain_pipeline.sh` |

## Archive Policy

- **`*/archive/`** directories contain historical files from previous iterations
- They are `.gitignore`d and safe to delete at any time
- `cleanup_heavy.sh` removes all archives automatically
- When creating new experiments, old results automatically move to `results/archive/`

## What Gets Committed

Only source code, configs, tests, docs, and scripts are tracked. The `.gitignore` ensures all large binary files (data, models, results, figures, logs) stay out of git.

Total tracked size: **~3 MB** (excludes all binary data).
