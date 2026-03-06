# clop — Contrastive Language-Omics Pre-training (CLOP) Alignment
"""
CLOP Aligner and loss/transform components.
Re-export so "from src.architecture.clop import CLOPAligner" etc. still work.
"""
from .losses import (
    TextWhiteningTransform,
    ProjectionHead,
    SigLIPLoss,
    PrototypeSigLIPLoss,
    InfoNCELoss,
)
from .aligner import CLOPAligner

__all__ = [
    "CLOPAligner",
    "TextWhiteningTransform",
    "ProjectionHead",
    "SigLIPLoss",
    "PrototypeSigLIPLoss",
    "InfoNCELoss",
]
