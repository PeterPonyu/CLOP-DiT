# decoder.py — scGPT Decoder Wrapper for cell embedding → gene expression
"""
ScGPT Decoder wrapper for converting generated cell embeddings
back to gene expression matrices.

v0.3: Uses scGPT's generate() method which properly:
    1. Encodes gene tokens via GeneEncoder
    2. Injects cell_emb at [CLS] position (position 0)
    3. Runs through full transformer_encoder
    4. Applies ExprDecoder to get per-gene expression values

This module handles:
    1. Loading pretrained scGPT weights (via standalone scgpt_embed)
    2. Encoding real cells to embeddings (for training data preparation)
    3. Decoding generated embeddings back to expression vectors via generate()

The pretrained weights (scGPT pan-cancer / whole-human) should be placed in:
    models/scgpt_pancancer/ or models/scgpt_human/
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Union, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ScGPTDecoder(nn.Module):
    """Wrapper around scGPT for encoding/decoding single-cell data.

    Uses the standalone scgpt_embed module (no torchtext dependency).
    v0.3: decode() now uses TransformerModel.generate() for proper reconstruction.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing scGPT pretrained weights
        (best_model.pt, vocab.json, args.json).
    device : torch.device
        Target device.
    max_seq_len : int
        Maximum number of genes to process.
    batch_size : int
        Batch size for encoding/decoding.
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = "models/scgpt_pancancer",
        device: torch.device = torch.device("cuda"),
        max_seq_len: int = 1200,
        batch_size: int = 64,
    ):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.device = device
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self._encoder = None

    def _load_encoder(self):
        """Lazy-load the standalone ScGPTCellEncoder."""
        if self._encoder is not None:
            return

        from .scgpt_embed import ScGPTCellEncoder

        logger.info(f"Loading scGPT from {self.model_dir}")
        self._encoder = ScGPTCellEncoder(
            model_dir=self.model_dir,
            device=self.device,
            max_length=self.max_seq_len,
            batch_size=self.batch_size,
        )
        # Trigger the actual model load
        self._encoder._load()
        logger.info("scGPT encoder ready")

    @torch.no_grad()
    def encode(self, adata, gene_col: str = "feature_name") -> np.ndarray:
        """Encode AnnData cells to cell embeddings via scGPT.

        Parameters
        ----------
        adata : AnnData
            Single-cell data (cells × genes).
        gene_col : str
            Column in adata.var with gene names, or uses var_names index.

        Returns
        -------
        cell_embeddings : (N, 512) numpy array
        """
        self._load_encoder()
        return self._encoder.encode(adata, gene_col=gene_col)

    @torch.no_grad()
    def decode(
        self,
        cell_embeddings: Union[torch.Tensor, np.ndarray],
        gene_ids: Optional[np.ndarray] = None,
        gene_names: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
    ) -> Dict:
        """Decode cell embeddings back to gene expression via scGPT generate().

        v0.3: This properly uses the TransformerModel.generate() method:
            1. Encodes reference gene tokens via GeneEncoder
            2. Injects cell_emb at position 0 ([CLS])
            3. Runs full transformer_encoder
            4. Applies ExprDecoder per gene → scalar expression values

        Parameters
        ----------
        cell_embeddings : (N, 512) cell embeddings from DiT
        gene_ids : (G,) optional gene token IDs. If None, uses last encode() refs.
        gene_names : list of str, optional. Gene names for output.
        batch_size : int, optional.

        Returns
        -------
        result : dict with keys:
            "expression" : (N, G) predicted gene expression matrix (numpy)
            "gene_names" : list of G gene name strings
        """
        self._load_encoder()

        if isinstance(cell_embeddings, torch.Tensor):
            cell_embeddings = cell_embeddings.cpu().numpy()

        return self._encoder.decode(
            cell_embeddings=cell_embeddings,
            gene_ids=gene_ids,
            gene_names=gene_names,
            batch_size=batch_size or self.batch_size,
        )

    def get_reference_genes(self) -> Optional[Dict]:
        """Get reference gene set from last encode() call.

        Returns
        -------
        dict with "gene_ids" and "gene_names", or None.
        """
        self._load_encoder()
        return self._encoder.get_reference_genes()

    @property
    def embed_dim(self) -> int:
        """Return the scGPT embedding dimension."""
        self._load_encoder()
        return self._encoder.embed_dim


class LinearDecoder(nn.Module):
    """Lightweight fallback decoder: cell embedding → gene expression via MLP.

    For use when scGPT decoder is unavailable or when working with
    pre-defined gene sets (e.g., highly variable genes only).

    Parameters
    ----------
    embed_dim : int
        Cell embedding dimension (from DiT output).
    num_genes : int
        Number of output genes.
    hidden_dims : list of int, optional
        Hidden layer dimensions.
    output_activation : str
        'softplus' for count data, 'sigmoid' for normalized, 'none' for raw.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_genes: int = 2000,
        hidden_dims: Optional[List[int]] = None,
        output_activation: str = "softplus",
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 2048]

        layers = []
        dims = [embed_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])

        layers.append(nn.Linear(dims[-1], num_genes))
        self.net = nn.Sequential(*layers)

        # Output activation
        if output_activation == "softplus":
            self.output_act = nn.Softplus()
        elif output_activation == "sigmoid":
            self.output_act = nn.Sigmoid()
        elif output_activation == "relu":
            self.output_act = nn.ReLU()
        else:
            self.output_act = nn.Identity()

    def forward(self, cell_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cell_emb : (B, embed_dim)

        Returns
        -------
        expression : (B, num_genes)
        """
        return self.output_act(self.net(cell_emb))
