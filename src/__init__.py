"""CLOP-DiT — text-conditioned single-cell latent generation.

A three-stage pipeline that generates single-cell expression embeddings from
a structured five-field text prompt (cell type, tissue, organism, marker
genes, disease context):

    1. Data preparation  — curate GEO datasets, build per-type captions,
                            and cache frozen scGPT cell embeddings and frozen
                            BiomedBERT text embeddings with ZCA whitening.
    2. CLOP alignment    — prototype-aware SigLIP contrastive alignment
                            between text and cell embeddings in a shared
                            512-dimensional latent space.
    3. DiT generation    — 1-D Diffusion Transformer trained by conditional
                            flow matching with classifier-free guidance;
                            decoded to gene expression through the frozen
                            scGPT decoder for downstream inspection.

See the top-level README for installation, usage, and citation.
"""

__version__ = "1.0.0"
__author__ = "Zeyu Fu, JianXu Zheng, Jiawei Fu"
