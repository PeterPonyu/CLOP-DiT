# clop.py — Contrastive Language-Omics Pre-training (CLOP) Alignment Module
"""
CLOP Aligner: Bridges text embeddings (PubMedBERT) and cell embeddings (scGPT)
into a shared contrastive space.

Architecture:
    Text  Embedding (768) → TextProjector  → Shared Space (proj_dim)
    Cell  Embedding (512) → CellProjector  → Shared Space (proj_dim)
    Loss: InfoNCE with learnable temperature

This follows the CLIP paradigm but operates on (text_description, cell_profile)
pairs instead of (text, image) pairs.

Key improvements over basic CLIP:
    1. Asymmetric projectors (different depths for text vs cell)
    2. Learnable temperature with clamping for stability
    3. Hard negative mining via similarity masking
    4. EMA momentum targets for stable training (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


# ============================================================================
#  Projection Heads
# ============================================================================

class ProjectionHead(nn.Module):
    """MLP projection head with optional normalization and residual.

    Projects from an encoder's embedding space to the shared contrastive space.

    Parameters
    ----------
    input_dim : int
        Input embedding dimension (e.g., 768 for BERT, 512 for scGPT).
    proj_dim : int
        Shared projection space dimension.
    hidden_dim : int, optional
        Hidden layer dimension. Defaults to input_dim.
    num_layers : int
        Number of linear layers (1 = linear projection, 2+ = MLP).
    dropout : float
        Dropout rate between layers.
    use_batch_norm : bool
        Whether to use BatchNorm (more stable for contrastive learning).
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 256,
        hidden_dim: Optional[int] = None,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [proj_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation/norm on final layer
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, input_dim)

        Returns
        -------
        (B, proj_dim) L2-normalized projection
        """
        h = self.net(x)
        return F.normalize(h, dim=-1)


# ============================================================================
#  InfoNCE Loss with Learnable Temperature
# ============================================================================

class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE contrastive loss with learnable temperature.

    L = 0.5 * (CE(sim_t2c, labels) + CE(sim_c2t, labels))

    where sim = (text_proj @ cell_proj.T) / temperature

    Parameters
    ----------
    init_temperature : float
        Initial temperature value (log-space).
    min_temperature : float
        Minimum temperature for clamping.
    max_temperature : float
        Maximum temperature for clamping.
    label_smoothing : float
        Label smoothing for cross-entropy.
    """

    def __init__(
        self,
        init_temperature: float = 0.07,
        min_temperature: float = 0.01,
        max_temperature: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temperature)))
        self.min_temp = min_temperature
        self.max_temp = max_temperature
        self.label_smoothing = label_smoothing

    @property
    def temperature(self) -> torch.Tensor:
        """Clamped temperature value."""
        return torch.clamp(
            self.log_temperature.exp(),
            min=self.min_temp,
            max=self.max_temp,
        )

    def forward(
        self,
        text_proj: torch.Tensor,
        cell_proj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        text_proj : (B, proj_dim) L2-normalized text projections
        cell_proj : (B, proj_dim) L2-normalized cell projections
        mask : (B, B), optional
            Binary mask where 1 = valid negative pair. Used for same-batch
            duplicate removal (e.g., cells from the same sample).

        Returns
        -------
        loss : scalar
        metrics : dict with accuracy, temperature, individual losses
        """
        B = text_proj.shape[0]
        device = text_proj.device

        # Cosine similarity matrix / temperature
        logits = (text_proj @ cell_proj.T) / self.temperature  # (B, B)

        # Apply mask if provided (set invalid pairs to large negative)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), float('-inf'))

        # Labels: diagonal is the positive pair
        labels = torch.arange(B, device=device)

        # Symmetric cross-entropy
        loss_t2c = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        loss_c2t = F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)
        loss = 0.5 * (loss_t2c + loss_c2t)

        # Metrics
        with torch.no_grad():
            acc_t2c = (logits.argmax(dim=-1) == labels).float().mean()
            acc_c2t = (logits.T.argmax(dim=-1) == labels).float().mean()

        metrics = {
            "loss_t2c": loss_t2c.item(),
            "loss_c2t": loss_c2t.item(),
            "acc_t2c": acc_t2c.item(),
            "acc_c2t": acc_c2t.item(),
            "temperature": self.temperature.item(),
        }

        return loss, metrics


# ============================================================================
#  CLOP Aligner — Full Module
# ============================================================================

class CLOPAligner(nn.Module):
    """Contrastive Language-Omics Pre-training Aligner.

    Aligns text descriptions and cell profiles in a shared embedding space
    using InfoNCE contrastive learning.

    Parameters
    ----------
    text_dim : int
        Text encoder output dimension (768 for PubMedBERT).
    cell_dim : int
        Cell encoder output dimension (512 for scGPT).
    proj_dim : int
        Shared projection space dimension.
    text_hidden_dim : int, optional
        Hidden dim for text projector.
    cell_hidden_dim : int, optional
        Hidden dim for cell projector.
    text_layers : int
        Number of layers in text projector.
    cell_layers : int
        Number of layers in cell projector.
    dropout : float
        Projector dropout rate.
    temperature : float
        Initial contrastive temperature.
    label_smoothing : float
        Label smoothing for InfoNCE.
    use_ema : bool
        Whether to maintain EMA copies of projectors for stability.
    ema_decay : float
        EMA decay coefficient.
    """

    def __init__(
        self,
        text_dim: int = 768,
        cell_dim: int = 512,
        proj_dim: int = 256,
        text_hidden_dim: Optional[int] = None,
        cell_hidden_dim: Optional[int] = None,
        text_layers: int = 3,
        cell_layers: int = 3,
        dropout: float = 0.1,
        temperature: float = 0.07,
        label_smoothing: float = 0.1,
        use_ema: bool = False,
        ema_decay: float = 0.999,
    ):
        super().__init__()

        self.proj_dim = proj_dim
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Projection heads
        self.text_projector = ProjectionHead(
            input_dim=text_dim,
            proj_dim=proj_dim,
            hidden_dim=text_hidden_dim or text_dim,
            num_layers=text_layers,
            dropout=dropout,
        )
        self.cell_projector = ProjectionHead(
            input_dim=cell_dim,
            proj_dim=proj_dim,
            hidden_dim=cell_hidden_dim or cell_dim,
            num_layers=cell_layers,
            dropout=dropout,
        )

        # Contrastive loss
        self.criterion = InfoNCELoss(
            init_temperature=temperature,
            label_smoothing=label_smoothing,
        )

        # Optional EMA targets
        if use_ema:
            self.text_projector_ema = ProjectionHead(
                input_dim=text_dim, proj_dim=proj_dim,
                hidden_dim=text_hidden_dim or text_dim,
                num_layers=text_layers, dropout=0.0,
            )
            self.cell_projector_ema = ProjectionHead(
                input_dim=cell_dim, proj_dim=proj_dim,
                hidden_dim=cell_hidden_dim or cell_dim,
                num_layers=cell_layers, dropout=0.0,
            )
            # Copy initial weights
            self._ema_copy(self.text_projector, self.text_projector_ema)
            self._ema_copy(self.cell_projector, self.cell_projector_ema)
            # Freeze EMA
            for p in self.text_projector_ema.parameters():
                p.requires_grad = False
            for p in self.cell_projector_ema.parameters():
                p.requires_grad = False

    @staticmethod
    def _ema_copy(source: nn.Module, target: nn.Module):
        """Copy parameters from source to target."""
        for s_param, t_param in zip(source.parameters(), target.parameters()):
            t_param.data.copy_(s_param.data)

    @torch.no_grad()
    def _ema_update(self):
        """Update EMA parameters."""
        if not self.use_ema:
            return
        for s, t in zip(self.text_projector.parameters(), self.text_projector_ema.parameters()):
            t.data.mul_(self.ema_decay).add_(s.data, alpha=1 - self.ema_decay)
        for s, t in zip(self.cell_projector.parameters(), self.cell_projector_ema.parameters()):
            t.data.mul_(self.ema_decay).add_(s.data, alpha=1 - self.ema_decay)

    def forward(
        self,
        text_emb: torch.Tensor,
        cell_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        text_emb : (B, text_dim) frozen text encoder embeddings
        cell_emb : (B, cell_dim) frozen cell encoder embeddings
        mask : (B, B), optional
            Negative pair validity mask.

        Returns
        -------
        loss : scalar
        metrics : dict
        """
        text_proj = self.text_projector(text_emb)
        cell_proj = self.cell_projector(cell_emb)

        loss, metrics = self.criterion(text_proj, cell_proj, mask=mask)

        # EMA update
        if self.training and self.use_ema:
            self._ema_update()

        return loss, metrics

    def project_text(self, text_emb: torch.Tensor, use_ema: bool = False) -> torch.Tensor:
        """Project text embedding to shared space (for DiT conditioning).

        Parameters
        ----------
        text_emb : (B, text_dim)
        use_ema : bool
            Use EMA projector (more stable for inference).

        Returns
        -------
        (B, proj_dim) L2-normalized text projection
        """
        projector = self.text_projector_ema if (use_ema and self.use_ema) else self.text_projector
        return projector(text_emb)

    def project_cell(self, cell_emb: torch.Tensor, use_ema: bool = False) -> torch.Tensor:
        """Project cell embedding to shared space.

        Parameters
        ----------
        cell_emb : (B, cell_dim)
        use_ema : bool

        Returns
        -------
        (B, proj_dim) L2-normalized cell projection
        """
        projector = self.cell_projector_ema if (use_ema and self.use_ema) else self.cell_projector
        return projector(cell_emb)

    def compute_similarity(
        self,
        text_emb: torch.Tensor,
        cell_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between text and cell embeddings.

        Useful for retrieval and zero-shot classification.

        Parameters
        ----------
        text_emb : (N, text_dim)
        cell_emb : (M, cell_dim)

        Returns
        -------
        (N, M) cosine similarity matrix
        """
        text_proj = self.project_text(text_emb)
        cell_proj = self.project_cell(cell_emb)
        return text_proj @ cell_proj.T
