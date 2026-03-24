# CLOP-DiT: Contrastive Language-Omics Pre-training + Diffusion Transformer

> Text-conditioned generation of single-cell gene expression profiles via flow matching

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Overview

CLOP-DiT is a three-stage generative framework that produces realistic single-cell gene expression profiles from natural language descriptions of biological conditions. Given a text prompt such as _"CD8+ cytotoxic T cells from human lung adenocarcinoma"_, the model generates synthetic transcriptomic profiles that recapitulate expected marker gene expression patterns, cell-type-specific signatures, and inter-cellular diversity.

The pipeline chains three pre-trained components:

1. **CLOP** (Contrastive Language-Omics Pre-training) aligns BiomedBERT text embeddings with scGPT cell embeddings into a shared 512-d space using Prototype-SigLIP loss with ZCA-whitened inputs.
2. **DiT** (Diffusion Transformer) learns the conditional distribution of cell embeddings via 1D flow matching with classifier-free guidance.
3. **scGPT Decoder** maps generated 512-d embeddings back to per-gene expression values through its native `generate()` pathway.

Training data comprises 220,304 cells from 80 GEO datasets spanning cancer, developmental, and normal tissue contexts.

## Architecture

```
User Text --> BiomedBERT-large (1024-d) --> ZCA Whitening --> CLOP Projector --> Condition c (512-d)
                                                                                       |
                              z0 ~ N(0,I) --> DiT(z_t, t, c) --> ODE Integrate --> z1 (512-d)
                                                                                       |
                                                                        scGPT Decoder --> Gene Expression (G genes)
```

## Key Results

| Method | KNN-1 | Steering | DivR | LinAcc | KNN/Rand |
|--------|-------|----------|------|--------|----------|
| **Real Data** | 0.890 | -- | 1.000 | 0.942 | 89x |
| **CLOP-DiT** (CFG=2.0) | 0.369 | 0.810 | 0.513 | 0.511 | 37x |
| CLOP-DiT (CFG=1.0) | 0.288 | 0.807 | 0.929 | 0.357 | 29x |
| Embedding-VAE | 0.112 | 0.547 | 0.744 | 0.189 | 11x |
| Gaussian baseline | 0.011 | 0.466 | 2.277 | 0.009 | 1x |

## Quick Start

### Installation

```bash
conda create -n clopdit python=3.10
conda activate clopdit
pip install -e .
```

### Generate Cells from Text

```bash
python scripts/inference/05_inference.py \
    --prompt "CD8+ cytotoxic T cells from human lung adenocarcinoma" \
    --num_cells 500 --cfg_scale 2.0 --decode_expression \
    --output generated_cells.h5ad
```

### Full Pipeline

```bash
# Train CLOP + DiT from cached embeddings
python scripts/training/04a_train_clop.py --config configs/clop.yaml
python scripts/training/04b_train_dit.py --config configs/dit.yaml

# Regenerate all 30 article figures
python scripts/pipeline/run_regeneration.py

# Or run the full orchestrated pipeline
python scripts/pipeline/run_pipeline.py --stage all
```

See [PIPELINE.md](PIPELINE.md) for stage-by-stage details and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for numeric reproduction.

## Project Structure

```
CLOP-DiT/
├── src/                        # Core Python library (pip install -e .)
│   ├── architecture/           #   DiT, CLOP aligner, scGPT decoder
│   ├── data_pipeline/          #   Dataset loading, caching, ZCA whitening
│   ├── training/               #   CLOPTrainer, DiTTrainer, schedulers
│   ├── evaluation/             #   Metrics, benchmarking, biological validation
│   ├── experiments/            #   OOD evaluation, rare cell augmentation
│   ├── visualization/          #   Publication figure generation (Figs 1-30)
│   └── utils/                  #   Path resolution, logging, helpers
├── scripts/                    # Pipeline entry points
│   ├── data_prep/              #   Steps 00-03: data preparation & caching
│   ├── training/               #   Steps 04a-c: model training & experiments
│   ├── inference/              #   Steps 05-08: generation & evaluation
│   ├── analysis/               #   Post-hoc analysis & figure scripts
│   ├── pipeline/               #   Orchestrators (run_pipeline.py)
│   ├── baselines/              #   Baseline method training
│   └── vcd/                    #   Visual Conflict Detector
├── configs/                    # YAML/JSON configuration
│   ├── clop.yaml               #   CLOP training config
│   ├── dit.yaml                #   DiT training config
│   ├── pipeline.yaml           #   Centralized path configuration
│   └── baselines/              #   Baseline method configs
├── articles/                   # LaTeX manuscript
│   ├── clop_dit_biology.tex    #   Main article (MDPI Biology)
│   └── figures/                #   Symlinks to results/figures/
├── tests/                      # Test suite (pytest)
├── docs/                       # Documentation
├── data/                       # Training data (not in repo)
├── models/                     # Model checkpoints (not in repo)
├── results/                    # Generated outputs (not in repo)
├── .gitignore
├── .gitattributes
├── LICENSE
├── CONTRIBUTING.md
├── PIPELINE.md                 # Pipeline stage documentation
├── REPRODUCIBILITY.md          # Reproduction guide
├── VERSIONS.md                 # Version history
├── requirements.txt
├── setup.py
└── README.md
```

## Reproducibility

Model checkpoints and pre-processed embeddings are available upon request from the corresponding author. Once placed in `models/` and `data/`, all 30 article figures can be regenerated with:

```bash
python scripts/pipeline/run_regeneration.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environment setup, data layout, and expected numeric results.

## Citation

```bibtex
@article{fu2026clopdit,
  author  = {Fu, Zeyu},
  title   = {CLOP-DiT: Text-Guided Single-Cell Gene Expression Generation
             via Contrastive Language-Omics Pretraining and Diffusion Transformers},
  journal = {Biology},
  year    = {2026},
  url     = {https://github.com/PeterPonyu/CLOP-DiT}
}
```

## License

MIT License -- see [LICENSE](LICENSE) for details.
