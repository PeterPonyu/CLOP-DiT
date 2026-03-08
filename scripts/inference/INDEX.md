# Inference Scripts (Phase 5-6)

> Scripts for text-conditioned generation and biological validation.

**Location**: `scripts/inference/`
**Version Compatibility**: v9.3+
**Last Updated**: 2025-03-08

---

## Script Index

| Script | Version | Size | Description | Output |
|--------|---------|------|-------------|--------|
| `05_inference.py` | v9.3 | 16KB | Text-conditioned cell generation | `results/generated_*.h5ad` |
| `06_cell2cell_inference.py` | v1.0 | 18KB | Cell-to-cell transformation | `results/cell2cell_*.h5ad` |
| `06_evaluate.py` | v6.0+ | 13KB | General evaluation metrics | `results/eval_*.json` |
| `08_biological_validation.py` | v9.3 | 51KB | Biological downstream validation | `results/downstream/` |

---

## Quick Start

```bash
# Generate cells from text prompt
python scripts/inference/05_inference.py \
    --prompt "human CD4 T cell from blood" \
    --n_cells 1000 \
    --output results/generated_cd4_t.h5ad

# Cell2Cell inference
python scripts/inference/06_cell2cell_inference.py \
    --source_cell_type "naive_CD4" \
    --target_cell_type "activated_CD4"

# Run biological validation suite
python scripts/inference/08_biological_validation.py \
    --config configs/biovalidation_datasets.json
```

---

## Required Checkpoints

| Script | Required Checkpoint | Path |
|--------|---------------------|------|
| `05_inference.py` | CLOP + DiT best | `models/checkpoints/CLOP/best/`, `models/checkpoints/DiT/best/` |
| `06_cell2cell_inference.py` | Cell2Cell | `models/checkpoints/Cell2Cell/best/` |
| `08_biological_validation.py` | Full model suite | All above |

---

## See Also

- [Training Scripts](../training/INDEX.md) - Train models for inference
- [Analysis Scripts](../analysis/INDEX.md) - Post-inference analysis
- [Results Structure](../../results/INDEX.md) - Output organization
