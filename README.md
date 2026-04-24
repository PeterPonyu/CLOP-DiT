# CLOP-DiT

CLOP-DiT is a codebase for text-conditioned single-cell latent generation.
It combines a contrastive text-cell alignment model with a 1-D diffusion
transformer that samples scGPT-compatible cell embeddings.

This repository contains source code, configuration files, and runnable scripts.
It does not include trained checkpoints, processed single-cell matrices,
manuscript files, or submission archives.

## Repository Layout

- `src/architecture/` — CLOP aligner, DiT generator, decoder, and scGPT wrapper code.
- `src/data_pipeline/` — dataset loading, embedding cache construction, and preprocessing helpers.
- `src/training/` — training utilities for CLOP, DiT, and decoder-side modules.
- `src/evaluation/` — metric and benchmark utilities.
- `scripts/data_prep/` — data preparation and CellxGene/GEO preprocessing entry points.
- `scripts/training/` — command-line training scripts.
- `scripts/inference/` — generation and evaluation entry points.
- `configs/` — YAML and JSON configuration files.

## Installation

The code targets Python 3.10 or newer.

```bash
conda create -n clopdit python=3.10
conda activate clopdit
pip install -e .
```

Install optional packages such as scGPT, CellxGene Census, or experiment tracking
tools only when the corresponding scripts require them.

## Basic Usage

Train the contrastive alignment model:

```bash
python scripts/training/04a_train_clop.py --config configs/clop.yaml
```

Train the diffusion transformer:

```bash
python scripts/training/04b_train_dit.py --config configs/dit.yaml
```

Run generation from an existing checkpoint:

```bash
python scripts/inference/05_inference.py \
  --prompt "CD8+ T cell; tissue: lung; organism: human" \
  --num_cells 500 \
  --cfg_scale 2.0 \
  --output generated_cells.h5ad
```

## Data and Checkpoints

The scripts expect local data and model paths configured through `configs/` or
command-line arguments. Large files are intentionally not tracked:

- raw and processed `.h5ad` matrices
- scGPT model files
- CLOP and DiT checkpoints
- generated embeddings, figures, logs, and benchmark outputs

Use the preprocessing scripts under `scripts/data_prep/` to build local caches
from public datasets before training.

## Notes

The code is research-oriented and assumes familiarity with single-cell data
processing, PyTorch training, and scGPT-style embedding workflows. Configuration
files may need local path edits before running.

## License

Released under the MIT License. See `LICENSE`.
