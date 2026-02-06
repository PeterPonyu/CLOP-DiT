# decoder.py — scGPT Decoder Wrapper for cell embedding → gene expression
"""
ScGPT Decoder wrapper for converting generated cell embeddings
back to gene expression matrices.

This module handles:
    1. Loading pretrained scGPT weights (via standalone scgpt_embed)
    2. Encoding real cells to embeddings (for training data preparation)
    3. Decoding generated embeddings back to expression vectors

The pretrained weights (scGPT_human) should be placed in:
    models/scgpt_human/
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ScGPTDecoder(nn.Module):
    """Wrapper around scGPT for encoding/decoding single-cell data.

    Uses the standalone scgpt_embed module (no torchtext dependency).

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
        Batch size for encoding.
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = "models/scgpt_human",
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
        cell_embeddings: torch.Tensor,
        gene_ids: Optional[torch.Tensor] = None,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Decode cell embeddings back to gene expression vectors.

        Uses the scGPT ExprDecoder head on the transformer output.
        Note: this requires feeding the embedding back through the model
        which is an approximation; for high-quality decoding, use
        LinearDecoder trained on real (embedding, expression) pairs.

        Parameters
        ----------
        cell_embeddings : (N, embed_dim) cell embeddings
        gene_ids : (G,), optional
            Gene token IDs.
        batch_size : int

        Returns
        -------
        expression_matrix : (N, G) reconstructed gene expression
        """
        self._load_encoder()

        model = self._encoder.model
        N = cell_embeddings.shape[0]
        all_preds = []

        for i in range(0, N, batch_size):
            batch = cell_embeddings[i:i + batch_size]
            if isinstance(batch, np.ndarray):
                batch = torch.from_numpy(batch).float()
            batch = batch.to(self.device)

            # Use the ExprDecoder head to predict per-gene expression
            # This treats each embedding as a single-token representation
            result = model.decoder(batch)
            all_preds.append(result["pred"].cpu())

        return torch.cat(all_preds, dim=0)

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
