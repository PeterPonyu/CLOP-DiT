# Ablation Study Checkpoints

> Model checkpoints from ablation studies.

**Location**: `models/checkpoints/ablations/`
**Ablations**: 18
**Last Updated**: 2025-03-08

---

## Ablation List

| Ablation | Purpose | Checkpoint Location |
|----------|---------|---------------------|
| `ablation_infonce` | InfoNCE loss vs SigLIP | `ablation_infonce/best/` |
| `ablation_siglip` | SigLIP loss ablation | `ablation_siglip/best/` |
| `baseline` | Baseline configuration | `baseline/best/` |
| `fixed_temperature` | Fixed vs learnable temp | `fixed_temperature/best/` |
| `high_variant_prob` | High variant sampling | `high_variant_prob/best/` |
| `lighter_dropout` | Reduced dropout | `lighter_dropout/best/` |
| `low_temperature_cap` | Low temp maximum | `low_temperature_cap/best/` |
| `no_cell_noise` | No cell embedding noise | `no_cell_noise/best/` |
| `no_cohesion` | No prototype cohesion | `no_cohesion/best/` |
| `no_duplicate_mask` | No duplicate masking | `no_duplicate_mask/best/` |
| `no_label_smoothing` | No label smoothing | `no_label_smoothing/best/` |
| `no_mixup` | No mixup augmentation | `no_mixup/best/` |
| `no_rdrop` | No R-Drop regularization | `no_rdrop/best/` |
| `no_regularization` | No regularization | `no_regularization/best/` |
| `no_variants` | No text variants | `no_variants/best/` |
| `smaller_proj` | Smaller projection dim | `smaller_proj/best/` |
| `wider_proj` | Wider projection dim | `wider_proj/best/` |

---

## Structure

Each ablation directory contains:

```
ablations/{name}/
├── best/              # Best checkpoint symlink or file
├── epochs/            # All epoch checkpoints
├── results/           # Metrics and summaries (from results/ablations/)
├── clop_final.pth     # Final checkpoint
└── clop_history.json  # Training history
```

---

## Loading Ablation Checkpoints

```python
import torch

# Load specific ablation
checkpoint = torch.load('models/checkpoints/ablations/no_cohesion/best/clop_best.pth')

# Compare with main model
main_checkpoint = torch.load('models/checkpoints/CLOP/best/clop_best.pth')
```

---

## Results

Ablation results and metrics are in:
- `results/ablations/{name}/`
- See [results/ablations/INDEX.md](../../../results/ablations/INDEX.md)

---

## See Also

- [Ablations Config](../../configs/ablation_*.yaml)
- [Ablations Results](../../../results/ablations/)
- [Main Checkpoints](../INDEX.md)
