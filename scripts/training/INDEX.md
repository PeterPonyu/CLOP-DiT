# Training Scripts (Phase 4)

> Scripts for training CLOP aligner, DiT diffusion model, and Cell2Cell.

**Location**: `scripts/training/`
**Version Compatibility**: v6.0+
**Last Updated**: 2025-03-08

---

## Script Index

| Script | Version | Size | Description | Duration |
|--------|---------|------|-------------|----------|
| `04a_train_clop.py` | v9.3 | 3.7KB | Train CLOP aligner (contrastive learning) | ~2-4 hours |
| `04b_train_dit.py` | v2.0 | 5.1KB | Train DiT diffusion model (flow matching) | ~6-8 hours |
| `04c_train_cell2cell.py` | v1.0 | 6.0KB | Train Cell2Cell model | ~1-2 hours |
| `full_retrain_pipeline.sh` | v9.3 | 12KB | Full retrain orchestration | ~12-16 hours |

---

## Quick Start

```bash
# Train CLOP (current config v9.3)
python scripts/training/04a_train_clop.py --config configs/clop.yaml

# Train DiT
python scripts/training/04b_train_dit.py --config configs/dit.yaml

# Train Cell2Cell
python scripts/training/04c_train_cell2cell.py

# Full retrain (all models)
bash scripts/training/full_retrain_pipeline.sh
```

---

## Checkpoints

Trained models are saved to:
- CLOP: `models/checkpoints/CLOP/versions/v9.3/`
- DiT: `models/checkpoints/DiT/versions/v2.0/`

See [models/checkpoints/README.md](../../models/checkpoints/README.md) for structure.

---

## GPU Requirements

| Script | Min VRAM | Recommended |
|--------|----------|-------------|
| `04a_train_clop.py` | 8GB | 12GB |
| `04b_train_dit.py` | 16GB | 24GB |
| `04c_train_cell2cell.py` | 8GB | 12GB |

---

## See Also

- [Inference Scripts](../inference/INDEX.md) - Use trained models
- [Checkpoint Index](../../models/checkpoints/INDEX.md) - Model storage
- [VERSIONS.md](../../VERSIONS.md) - Version compatibility
