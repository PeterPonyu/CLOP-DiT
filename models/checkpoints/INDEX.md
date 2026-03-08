# Model Checkpoints

> Organized storage for trained model weights and training artifacts.

**Location**: `models/checkpoints/`
**Current Version**: v9.3 (CLOP), v2.0 (DiT)
**Last Updated**: 2025-03-08

---

## Quick Access

| Model | Best Checkpoint | Final Checkpoint |
|-------|---------------|------------------|
| **CLOP** | `CLOP/best/clop_best.pth` | `CLOP/final/clop_final.pth` |
| **DiT** | `DiT/best/dit_best.pth` | `DiT/final/dit_final.pth` |

---

## Directory Structure

```
models/checkpoints/
├── INDEX.md                    # This file
├── CURRENT_VERSION.txt         # Current version: "v9.3"
├── CLOP/
│   ├── INDEX.md
│   ├── best/                   # Symlink to current best
│   ├── final/                  # Symlink to current final
│   └── versions/
│       ├── v9.3/              # Current version
│       │   ├── clop_best.pth
│       │   ├── clop_final.pth
│       │   ├── clop_history.json
│       │   └── epochs/         # All epoch checkpoints
│       │       ├── clop_epoch_10.pth
│       │       └── ...
│       └── archive/           # Old versions
├── DiT/
│   ├── INDEX.md
│   ├── best/
│   ├── final/
│   └── versions/
│       └── v2.0/
│           ├── dit_best.pth
│           ├── dit_final.pth
│           ├── dit_history.json
│           └── epochs/
└── ablations/                  # Ablation study checkpoints
    └── INDEX.md
```

---

## Checkpoint Sizes

| Checkpoint | Size | Purpose |
|------------|------|---------|
| `clop_best.pth` | ~41MB | Best validation performance |
| `clop_final.pth` | ~41MB | Final epoch |
| `clop_epoch_*.pth` | ~41MB each | Intermediate checkpoints (14 files) |
| `dit_best.pth` | ~600MB | Best DiT checkpoint |
| `dit_final.pth` | ~600MB | Final DiT checkpoint |
| `dit_epoch_*.pth` | ~300MB each | Intermediate DiT checkpoints |

---

## Usage

### Load Best CLOP Model

```python
import torch

# Via symlink (always points to current)
checkpoint = torch.load('models/checkpoints/CLOP/best/clop_best.pth')

# Or via specific version
checkpoint = torch.load('models/checkpoints/CLOP/versions/v9.3/clop_best.pth')
```

### Training History

```python
import json

with open('models/checkpoints/CLOP/versions/v9.3/clop_history.json') as f:
    history = json.load(f)
```

---

## Version History

| Version | CLOP | DiT | Status |
|---------|------|-----|--------|
| v9.3 | Stratified split, fixed temp | - | **Current** |
| v6.3 | Template anchor | - | Archived |

See [CLOP/versions/](CLOP/versions/) and [DiT/versions/](DiT/versions/) for all versions.

---

## Cleanup

To free space, remove epoch checkpoints:

```bash
# Remove all CLOP epoch checkpoints (frees ~560MB)
rm models/checkpoints/CLOP/versions/v9.3/epochs/*.pth

# Or use the cleanup script
bash scripts/pipeline/cleanup_heavy.sh
```

---

## See Also

- [Training Scripts](../../scripts/training/INDEX.md)
- [Ablations](ablations/INDEX.md)
- [VERSIONS.md](../../VERSIONS.md)
