# DiT Model Checkpoints

> Diffusion Transformer (DiT) for flow matching.

**Location**: `models/checkpoints/DiT/`
**Current Version**: v2.0
**Last Updated**: 2025-03-08

---

## Quick Access

| Checkpoint | Path | Size |
|------------|------|------|
| **Best (recommended)** | `best/dit_best.pth` | ~600MB |
| **Final** | `final/dit_final.pth` | ~600MB |
| **History** | `versions/v2.0/dit_history.json` | ~20KB |

---

## Version: v2.0 (Current)

**Key Features**:
- Flow matching diffusion
- 22M parameters
- Conditioned on CLOP text embeddings

**Files**:
- `versions/v2.0/dit_best.pth` - Best checkpoint (EMA)
- `versions/v2.0/dit_final.pth` - Final epoch
- `versions/v2.0/dit_history.json` - Training metrics
- `versions/v2.0/epochs/` - Epoch checkpoints (100, 150, 200)

---

## Loading Checkpoints

```python
import torch
from src.architecture.dit import DiT1D

# Use best symlink
checkpoint = torch.load('models/checkpoints/DiT/best/dit_best.pth')

# Or specific version
checkpoint = torch.load('models/checkpoints/DiT/versions/v2.0/dit_best.pth')

# Load model
model = DiT1D(...)  # Initialize with correct params
model.load_state_dict(checkpoint['ema_model_state_dict'])  # Note: use EMA weights
```

---

## GPU Memory Requirements

| Checkpoint | Load VRAM | Inference VRAM |
|------------|-----------|----------------|
| `dit_best.pth` | ~600MB | ~2-4GB |

---

## See Also

- [DiT Training](../../scripts/training/04b_train_dit.py)
- [Main Checkpoint Index](../INDEX.md)
