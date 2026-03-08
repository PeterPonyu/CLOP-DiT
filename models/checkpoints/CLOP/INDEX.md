# CLOP Model Checkpoints

> Contrastive Language-Omics Pre-training (CLOP) aligner checkpoints.

**Location**: `models/checkpoints/CLOP/`
**Current Version**: v9.3
**Last Updated**: 2025-03-08

---

## Quick Access

| Checkpoint | Path | Size |
|------------|------|------|
| **Best (recommended)** | `best/clop_best.pth` | ~41MB |
| **Final** | `final/clop_final.pth` | ~41MB |
| **History** | `versions/v9.3/clop_history.json` | ~24KB |

---

## Version: v9.3 (Current)

**Key Changes**:
- Stratified validation split (all 69 types in val)
- Fixed temperature (14.0, non-learnable)
- Deduplicated training data

**Files**:
- `versions/v9.3/clop_best.pth` - Best validation checkpoint
- `versions/v9.3/clop_final.pth` - Final epoch checkpoint
- `versions/v9.3/clop_history.json` - Training history (loss curves)
- `versions/v9.3/epochs/` - All epoch checkpoints (10, 20, 30, ..., 120)

---

## Loading Checkpoints

```python
import torch
from src.architecture.clop import CLOPAligner

# Method 1: Use best symlink (always current)
checkpoint = torch.load('models/checkpoints/CLOP/best/clop_best.pth')

# Method 2: Use specific version
checkpoint = torch.load('models/checkpoints/CLOP/versions/v9.3/clop_best.pth')

# Load into model
model = CLOPAligner(text_dim=1024, cell_dim=512, proj_dim=512)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Training History

```python
import json

with open('models/checkpoints/CLOP/versions/v9.3/clop_history.json') as f:
    history = json.load(f)

# Available keys: epoch, train_loss, val_loss, train_acc, val_acc, etc.
epochs = history['epoch']
train_acc = history['train_acc']
val_acc = history['val_acc']
```

---

## Epoch Checkpoints

14 epoch checkpoints available:
- Epoch 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120
- Each ~41MB
- Total: ~560MB

**Location**: `versions/v9.3/epochs/`

To free space:
```bash
rm models/checkpoints/CLOP/versions/v9.3/epochs/*.pth
```

---

## Previous Versions

| Version | Location | Status |
|---------|----------|--------|
| v6.3 | `versions/archive/v6.3/` | Archived |

See `versions/archive/` for older versions.

---

## See Also

- [CLOP Training](../../scripts/training/04a_train_clop.py)
- [Main Checkpoint Index](../INDEX.md)
- [VERSIONS.md](../../../VERSIONS.md)
