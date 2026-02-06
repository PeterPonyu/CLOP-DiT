# CLOP-DiT: Contrastive Language-Omics Pre-training + Diffusion Transformer
# For conditional single-cell gene expression generation
# Author: Zeyu Fu
"""
CLOP-DiT — A text-conditioned generative framework for single-cell transcriptomics.

Pipeline:
    1. Data Ingestion  → GEO metadata + expression matrices
    2. SFT Cleaning    → Structured JSON labels via Llama-3
    3. Latent Caching  → scGPT cell embeddings + PubMedBERT text embeddings
    4. CLOP Alignment  → Contrastive text-cell alignment (InfoNCE)
    5. DiT Training    → Flow-Matching 1D-DiT with AdaLN-Zero conditioning
    6. Inference        → Text → ODE sampling → scGPT decode → expression matrix
"""

__version__ = "0.1.0"
__author__ = "Zeyu Fu"
