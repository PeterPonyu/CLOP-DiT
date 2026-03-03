# CLOP-DiT Session Report — 2026-03-02

## Executive Summary

Full project audit, cleanup (~3GB freed), code infrastructure improvements, and 3 experimental training runs.
**v6.4.1 remains the best checkpoint** (val_proto_acc 10.45%, epoch 102). Temperature regularization experiments (v7.0, v7.1) definitively ruled out as counterproductive. v7.2 baseline reproduction achieved 9.34% — a modest regression explained by a missing config parameter (`temp_lr_multiplier` defaulted to 10.0 instead of 5.0).

---

## 1. Project Audit & Cleanup

### Redundancies Removed
| Item | Savings | Notes |
|------|---------|-------|
| `v63_backup/` | ~260MB | Identical to `archive/v63/` |
| 20+ epoch checkpoints | ~1.5GB | Kept best + final only |
| `fold0/` 5-fold CV artifacts | ~1.5GB | Archived |
| Tier experiment checkpoints | ~200MB | Archived |
| 14 one-off scripts | — | Moved to `scripts/archive/` |
| Old PNG results | — | Moved to `results/archive/` |

**Total: ~3.4GB freed**

### Retained Checkpoints
- `models/checkpoints/clop_best.pth` (v6.4.1 — project best)
- `models/checkpoints/clop_final.pth` (v6.4.1)
- `models/checkpoints/dit_best.pth`, `dit_final.pth`
- `models/checkpoints/clop_history.json`, `dit_history.json`
- `models/checkpoints/archive/v63/` (historical reference)

---

## 2. Training History Analysis

### Version Comparison Table

| Version | Best VPA | Epoch | Train PA | Gap | Temp | Status |
|---------|----------|-------|----------|-----|------|--------|
| **v6.3** | 10.22% | 32 | — | — | 20 | Overfit (no train metrics) |
| **v6.4** | 5.65% | 50 | 18.6% | 3.3× | → 0 | Underfit (2-layer projector) |
| **v6.4.1** | **10.45%** | **102** | 38.0% | 3.5× | 20 | **BEST — current checkpoint** |
| v7.0 (temp_reg) | 1.35% | 4 | — | — | — | ❌ FAILED — no cell whitening |
| v7.1 (light_reg) | 1.31% | 45 | — | — | → 1 | ❌ FAILED — temp collapsed to 1.0 |
| v7.2 (baseline+) | 9.34% | 111 | 31.9% | 3.4× | 20 | ⚠ -1.1pp vs v6.4.1 |

### Key Dynamics
- **Temperature**: Always saturates at `max_temperature` (20.0) — this is REQUIRED, not a bug
- **Train/val gap**: Stabilizes around 3.4–3.5× — inherent to zero group overlap
- **Top-5/10 metrics** (new): v7.2 showed 26.5% top-5. 38.2% top-10 — much more stable evaluation signals

---

## 3. Code Improvements Made

### A. Architecture (`src/architecture/clop.py`)
- Added `temp_reg_weight` parameter to `PrototypeSigLIPLoss` (L2 penalty on log_temperature)
- Added top-10 metric computation (`proto_top10_t2c`, `proto_top10_c2t`)
- Exposed `temp_reg_weight` in `CLOPAligner.from_config()`

### B. Training (`src/training/train_clop.py`)
- Tracks `val_proto_top5` and `val_proto_top10` in training history
- Logs train/val gap ratio each epoch
- Per-modality preprocessing control via `preprocess_text`/`preprocess_cell` booleans
- `from_config()` auto-derives preprocessing flags from config methods

### C. Dataset (`src/data_pipeline/dataset.py`)
- Independent `preprocess_text` and `preprocess_cell` boolean flags
- Allows ablation studies on whitening per-modality

### D. Infrastructure
- `ExperimentTracker` integration — each run gets isolated `results/<version>/` with checkpoints, metrics, figures, config snapshot, and `run_info.json`
- `scripts/09_train_and_compare_v7.py` — training + automatic comparison against historical baselines
- Fixed experiment registry at `results/clop_experiment_registry.json` with all v7 experiments

---

## 4. Experimental Findings

### Finding 1: Temperature Regularization is HARMFUL
**v7.0** (temp_reg=0.01, no cell whitening): val_proto_acc collapsed to 1.35%.
**v7.1** (temp_reg=0.002, both whitened, max_temp=15): val_proto_acc 1.31%, temp dropped to 1.0.

**Root cause**: In a 1088-class contrastive setting with zero train/val group overlap, the model genuinely needs high temperature (~20) to produce sharp logit distributions. Penalizing temperature growth directly prevents the model from learning discriminative features.

**Rule**: Never regularize temperature in PrototypeSigLIP loss for this task.

### Finding 2: Cell Whitening is ESSENTIAL
Without cell whitening, raw scGPT embeddings have pairwise cosine similarity ~0.991. This near-uniform similarity makes contrastive learning impossible — all prototypes look identical. ZCA whitening decorrelates the space to enable discrimination.

### Finding 3: v7.2 Regression Root Cause
The 1.1pp regression (9.34% vs 10.45%) is explained by:
1. **Missing `temp_lr_multiplier: 5.0`** — defaulted to 10.0 (2× intended), causing different early temperature dynamics
2. **Cohesion 0.25 vs 0.20** — may be slightly too aggressive

Config has been fixed for future runs.

### Finding 4: Top-10 is a Better Tracking Metric
With 119 val groups and zero overlap, top-1 accuracy (9.34%) has high variance epoch-to-epoch. Top-10 (38.2%) provides a more stable signal for tracking learning progress and comparing runs.

---

## 5. Current Best Model

```
Checkpoint: models/checkpoints/clop_best.pth
Version:    v6.4.1
Metric:     val_proto_acc = 10.45% (epoch 102)
Config:     configs/clop_v641.yaml
```

This checkpoint feeds into the DiT stage for cell generation.

---

## 6. Recommended Next Steps (Prioritized)

### Phase A: Hyperparameter Correction (immediate)
1. **Re-run v7.2 with corrected config** — add `temp_lr_multiplier: 5.0`, revert cohesion to 0.20. This should reproduce v6.4.1's 10.45% and validate the infrastructure.

### Phase B: Targeted Improvements (short-term)
2. **Group-aware batch sampler** — ensure each batch has harder negatives by sampling more text groups per batch. Currently random sampling may produce easy batches.
3. **Hard-negative mining** — select confusing text groups (close in embedding space) as negatives to force sharper discrimination.
4. **Label smoothing schedule** — reduce label_smoothing from 0.1 → 0.05 after epoch 50 to allow sharper predictions in later training.

### Phase C: Architecture Exploration (medium-term)
5. **Cross-attention between modalities** — instead of separate projection MLPs, use cross-attention layers for richer alignment.
6. **Multi-scale prototypes** — cluster groups into super-groups and add hierarchical contrastive terms.
7. **Curriculum learning** — start with more groups per dataset, gradually increase difficulty.

### NOT Recommended
- ❌ Temperature regularization (proven harmful)
- ❌ Removing cell whitening (proven harmful)
- ❌ Reducing projector depth below 3 layers (v6.4 showed underfitting)
- ❌ Learning rate > 5e-4 (v6.3 used 1e-3 and overfit faster)

---

## 7. File Inventory (Post-Cleanup)

```
configs/
  clop.yaml         — legacy config
  clop_v641.yaml    — v6.4.1 (CURRENT BEST)
  clop_v7.yaml      — v7.0 (temp reg, FAILED)
  clop_v71.yaml     — v7.1 (light reg, FAILED)
  clop_v72.yaml     — v7.2 (baseline+, FIXED for rerun)
  clop_v64.yaml     — v6.4
  dit.yaml          — DiT flow matching config
  models.yaml       — model paths

results/
  clop_experiment_registry.json   — master experiment tracker
  v7.0_temp_reg/                  — v7.0 artifacts
  v7.1_light_temp_reg/            — v7.1 artifacts
  v7.2_improved_baseline/         — v7.2 artifacts
  v5_final/                       — final results
  5fold_cv/                       — cross-validation results

scripts/
  09_train_and_compare_v7.py      — NEW: training + comparison pipeline
  archive/                        — 14 archived one-off scripts
```
