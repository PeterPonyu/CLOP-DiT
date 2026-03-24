# decoder.py — scGPT Decoder Wrapper for cell embedding → gene expression
"""
ScGPT Decoder wrapper for converting generated cell embeddings
back to gene expression matrices.

v0.3: Uses scGPT's generate() method which properly:
    1. Encodes gene tokens via GeneEncoder
    2. Injects cell_emb at [CLS] position (position 0)
    3. Runs through full transformer_encoder
    4. Applies ExprDecoder to get per-gene expression values

v0.4: Adds LoRA adapters for fine-tuning frozen scGPT layers.
    Enables targeted fine-tuning of the decoder's last transformer layers
    without full retraining, improving reconstruction of generated embeddings.

This module handles:
    1. Loading pretrained scGPT weights (via standalone scgpt_embed)
    2. Encoding real cells to embeddings (for training data preparation)
    3. Decoding generated embeddings back to expression vectors via generate()
    4. LoRA-based fine-tuning of decoder layers (v0.4)

The pretrained weights (scGPT pan-cancer / whole-human) should be placed in:
    models/scgpt_pancancer/ or models/scgpt_human/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def load_lora_weights(self, checkpoint_path: Union[str, Path]) -> None:
        """Apply LoRA adapters and load trained weights from checkpoint.

        LoRA hyperparameters are read from the checkpoint's saved config.

        Parameters
        ----------
        checkpoint_path : path to scgpt_lora_best.pth

        Raises
        ------
        FileNotFoundError
            If checkpoint_path does not exist.
        ValueError
            If checkpoint has no lora_state_dict.
        """
        self._load_encoder()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        lora_sd = ckpt.get("lora_state_dict", {})
        if not lora_sd:
            raise ValueError(f"No lora_state_dict in checkpoint: {checkpoint_path}")

        cfg = ckpt.get("config", {})
        target_modules = cfg.get("target_modules", ["out_proj", "linear1", "linear2"])
        rank = cfg.get("lora_rank", 8)
        alpha = cfg.get("lora_alpha", 16.0)
        num_last_layers = cfg.get("num_last_layers", 2)
        val_loss = ckpt.get("val_loss", "N/A")
        del ckpt

        model = self._encoder.model
        model = apply_lora_to_model(
            model, target_modules=target_modules,
            rank=rank, alpha=alpha, num_last_layers=num_last_layers,
        )
        # Load the trained LoRA parameters
        missing, unexpected = model.load_state_dict(lora_sd, strict=False)
        if unexpected:
            logger.warning(f"Unexpected keys in LoRA checkpoint: {unexpected}")
        loaded = len(lora_sd) - len(unexpected)
        logger.info(f"Loaded LoRA weights: {loaded} tensors from {checkpoint_path.name} "
                     f"(val_loss={val_loss})")
        # Move only new LoRA params to device (base model already on device)
        for name, param in model.named_parameters():
            if "lora_" in name and param.device.type == "cpu":
                param.data = param.data.to(self.device)

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


# ============================================================================
#  LoRA Adapter for scGPT Fine-Tuning
# ============================================================================

class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) wrapper for a frozen linear layer.

    Adds a trainable low-rank decomposition: W' = W + BA where
    B ∈ R^{out×r}, A ∈ R^{r×in} with rank r << min(in, out).

    Parameters
    ----------
    original_linear : nn.Linear
        The frozen pretrained linear layer.
    rank : int
        LoRA rank.
    alpha : float
        LoRA scaling factor. Effective scale = alpha / rank.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

        # Low-rank adapters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_out


def apply_lora_to_model(model: nn.Module, target_modules: List[str],
                         rank: int = 8, alpha: float = 16.0,
                         num_last_layers: int = 2) -> nn.Module:
    """Apply LoRA adapters to specific linear layers in the last N transformer layers.

    Parameters
    ----------
    model : nn.Module
        The scGPT TransformerModel.
    target_modules : list of str
        Names of submodules to adapt (e.g., ['out_proj', 'linear1', 'linear2']).
    rank : int
        LoRA rank.
    alpha : float
        LoRA alpha scaling.
    num_last_layers : int
        Number of last transformer layers to apply LoRA to.

    Returns
    -------
    model : nn.Module with LoRA adapters applied.
    """
    # Find transformer encoder layers
    encoder_layers = None
    if hasattr(model, 'transformer_encoder') and hasattr(model.transformer_encoder, 'layers'):
        encoder_layers = list(model.transformer_encoder.layers)
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        encoder_layers = list(model.encoder.layers)

    if encoder_layers is None:
        logger.warning("Could not find transformer encoder layers for LoRA injection")
        return model

    # Apply to last N layers
    target_layers = encoder_layers[-num_last_layers:]
    n_adapted = 0

    for layer in target_layers:
        for name in target_modules:
            # Navigate to the target submodule
            parts = name.split('.')
            parent = layer
            for part in parts[:-1]:
                if hasattr(parent, part):
                    parent = getattr(parent, part)
                else:
                    parent = None
                    break

            if parent is None:
                continue

            attr_name = parts[-1]
            if hasattr(parent, attr_name):
                original = getattr(parent, attr_name)
                if isinstance(original, nn.Linear):
                    lora_layer = LoRALinear(original, rank=rank, alpha=alpha)
                    setattr(parent, attr_name, lora_layer)
                    n_adapted += 1

    logger.info(f"Applied LoRA (rank={rank}, alpha={alpha}) to {n_adapted} layers "
                f"in last {num_last_layers} transformer blocks")
    return model
