# aligner.py — CLOP Aligner full module
"""CLOP Aligner: contrastive alignment of text and cell embeddings."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ...utils.constants import (
    CLOP_COHESION_WEIGHT,
    CLOP_DROPOUT,
    CLOP_LABEL_SMOOTHING,
    CLOP_MAX_TEMPERATURE,
    CLOP_MIN_TEMPERATURE,
    CLOP_NUM_LAYERS,
    CLOP_SEPARATION_THRESHOLD,
    CLOP_SOFT_LABEL_ALPHA,
    CLOP_SOFT_LABEL_BIAS,
    CLOP_TEMPERATURE,
    CLOP_WHITENING_EPS,
    LATENT_DIM,
    PROJ_DIM,
    TEXT_DIM_BASE,
)
from .losses import (
    TextWhiteningTransform,
    ProjectionHead,
    SigLIPLoss,
    PrototypeSigLIPLoss,
    InfoNCELoss,
)


class CLOPAligner(nn.Module):
    """Contrastive Language-Omics Pre-training Aligner.

    Aligns text descriptions and cell profiles in a shared embedding space
    using InfoNCE contrastive learning with optional soft labels and whitening.

    Parameters
    ----------
    text_dim : int
        Text encoder output dimension (1024 for BiomedBERT-large).
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
    use_batch_norm : bool
        If True, use BatchNorm. If False, use LayerNorm
        (better cross-dataset generalization).
    temperature : float
        Initial contrastive temperature.
    min_temperature : float
        Minimum temperature for clamping.
    max_temperature : float
        Maximum temperature for clamping.
    label_smoothing : float
        Label smoothing for InfoNCE (hard-label mode only).
    use_ema : bool
        Whether to maintain EMA copies of projectors for stability.
    ema_decay : float
        EMA decay coefficient.
    use_soft_labels : bool
        Use text-similarity soft targets instead of hard diagonal labels.
    soft_label_alpha : float
        Sharpening exponent for soft targets (higher = sharper).
    soft_label_bias : float
        Additive bias for diagonal (true positive) in soft targets.
    cell_noise_std : float
        Gaussian noise std for cell embedding augmentation during training.
    use_whitening : bool
        Apply PCA whitening to text embeddings before projection.
        DEPRECATED in v6: use upstream embedding_preprocessor instead.
    whitening_eps : float
        Regularization for whitening (prevents noise amplification).
    loss_type : str
        Loss function: 'prototype_siglip' (v6.1, recommended), 'siglip', or 'infonce'.
        prototype_siglip addresses the text-cell granularity mismatch by aligning
        text embeddings to group centroids instead of individual cells.
    auto_duplicate_mask : bool
        If True, automatically detect and mask cells sharing identical text
        embeddings within a batch so they are not penalized as false negatives.
    cohesion_weight : float
        Weight for prototype cohesion regularization (only for prototype_siglip).
        Pulls individual cells toward their text group centroid.
    max_temperature : float
        Temperature cap for SigLIP/PrototypeSigLIP to prevent overfitting.
    """

    def __init__(
        self,
        text_dim: int = TEXT_DIM_BASE,
        cell_dim: int = LATENT_DIM,
        proj_dim: int = PROJ_DIM,
        text_hidden_dim: Optional[int] = None,
        cell_hidden_dim: Optional[int] = None,
        text_layers: int = CLOP_NUM_LAYERS,
        cell_layers: int = CLOP_NUM_LAYERS,
        dropout: float = CLOP_DROPOUT,
        use_batch_norm: bool = True,
        temperature: float = CLOP_TEMPERATURE,
        min_temperature: float = CLOP_MIN_TEMPERATURE,
        max_temperature: float = CLOP_MAX_TEMPERATURE,
        label_smoothing: float = CLOP_LABEL_SMOOTHING,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        use_soft_labels: bool = False,
        soft_label_alpha: float = CLOP_SOFT_LABEL_ALPHA,
        soft_label_bias: float = CLOP_SOFT_LABEL_BIAS,
        cell_noise_std: float = 0.0,
        use_whitening: bool = False,
        whitening_eps: float = CLOP_WHITENING_EPS,
        loss_type: str = "infonce",
        auto_duplicate_mask: bool = False,
        cohesion_weight: float = CLOP_COHESION_WEIGHT,
        temp_reg_weight: float = 0.0,
        separation_margin: float = 0.0,
        separation_threshold: float = CLOP_SEPARATION_THRESHOLD,
    ):
        super().__init__()

        self.proj_dim = proj_dim
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.cell_noise_std = cell_noise_std
        self.use_whitening = use_whitening
        self.use_batch_norm = use_batch_norm
        self.loss_type = loss_type
        self.auto_duplicate_mask = auto_duplicate_mask

        # Text embedding whitening (v4: addresses BiomedBERT embedding collapse)
        if use_whitening:
            self.text_whitening = TextWhiteningTransform(
                dim=text_dim, eps=whitening_eps
            )
        else:
            self.text_whitening = None

        # Projection heads
        self.text_projector = ProjectionHead(
            input_dim=text_dim,
            proj_dim=proj_dim,
            hidden_dim=text_hidden_dim or text_dim,
            num_layers=text_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        self.cell_projector = ProjectionHead(
            input_dim=cell_dim,
            proj_dim=proj_dim,
            hidden_dim=cell_hidden_dim or cell_dim,
            num_layers=cell_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

        # Contrastive loss
        if loss_type == "prototype_siglip":
            self.criterion = PrototypeSigLIPLoss(
                init_temperature=temperature if temperature > 1.0 else 10.0,
                init_bias=-10.0,
                cohesion_weight=cohesion_weight,
                max_temperature=max_temperature if max_temperature > 1.0 else 100.0,
                temp_reg_weight=temp_reg_weight,
                separation_margin=separation_margin,
                separation_threshold=separation_threshold,
            )
        elif loss_type == "siglip":
            self.criterion = SigLIPLoss(
                init_temperature=temperature if temperature > 1.0 else 10.0,
                init_bias=-10.0,
            )
        else:
            self.criterion = InfoNCELoss(
                init_temperature=temperature,
                min_temperature=min_temperature,
                max_temperature=max_temperature,
                label_smoothing=label_smoothing,
                use_soft_labels=use_soft_labels,
                soft_label_alpha=soft_label_alpha,
                soft_label_bias=soft_label_bias,
            )

        # Optional EMA targets
        if use_ema:
            self.text_projector_ema = ProjectionHead(
                input_dim=text_dim, proj_dim=proj_dim,
                hidden_dim=text_hidden_dim or text_dim,
                num_layers=text_layers, dropout=0.0,
                use_batch_norm=use_batch_norm,
            )
            self.cell_projector_ema = ProjectionHead(
                input_dim=cell_dim, proj_dim=proj_dim,
                hidden_dim=cell_hidden_dim or cell_dim,
                num_layers=cell_layers, dropout=0.0,
                use_batch_norm=use_batch_norm,
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

    def _whiten_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        """Apply whitening transform to text embeddings if enabled."""
        if self.use_whitening and self.text_whitening is not None:
            return self.text_whitening(text_emb)
        return text_emb

    def forward(
        self,
        text_emb: torch.Tensor,
        cell_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        text_sim: Optional[torch.Tensor] = None,
        mixup_alpha: float = 0.0,
        group_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        text_emb : (B, text_dim) frozen text encoder embeddings
        cell_emb : (B, cell_dim) frozen cell encoder embeddings
        mask : (B, B), optional
            Negative pair validity mask.
        text_sim : (B, B), optional
            Cosine similarity of raw text embeddings (for soft labels).
            Computed from RAW embeddings before whitening.
        mixup_alpha : float
            If > 0, apply embedding-level MixUp with Beta(alpha, alpha).
            Mixes embeddings within the same text group to regularize.
        group_ids : (B,) long tensor, optional
            Explicit text group IDs from dataset. When provided, these are
            used directly instead of the cosine-based detection. This is
            essential when caption variant augmentation changes the text
            embedding — cosine detection would fragment groups.
            Values of -1 signal "unknown"; fall back to cosine detection.

        Returns
        -------
        loss : scalar
        metrics : dict
        """
        # Resolve group IDs: prefer explicit, fall back to cosine detection
        _group_ids = None
        if group_ids is not None and (group_ids >= 0).all():
            # Remap to contiguous 0..N-1 (same as _compute_group_ids output)
            _, _group_ids = group_ids.unique(return_inverse=True)
        elif self.loss_type == "prototype_siglip" or (self.training and mixup_alpha > 0):
            _group_ids = self._compute_group_ids(text_emb)

        # Cell embedding augmentation during training
        if self.training and self.cell_noise_std > 0:
            cell_emb = cell_emb + torch.randn_like(cell_emb) * self.cell_noise_std

        # Embedding-level MixUp: interpolate cell embeddings within groups
        if self.training and mixup_alpha > 0:
            cell_emb = self._embedding_mixup(cell_emb, text_emb, mixup_alpha,
                                             group_ids=_group_ids)

        # Apply text whitening before projection
        text_emb_proj = self._whiten_text(text_emb)

        text_proj = self.text_projector(text_emb_proj)
        cell_proj = self.cell_projector(cell_emb)

        # Auto-detect duplicate text embeddings and build mask
        if self.auto_duplicate_mask and mask is None:
            mask = self._build_duplicate_mask(text_emb)

        # For prototype loss, use resolved group IDs
        if self.loss_type == "prototype_siglip":
            loss, metrics = self.criterion(
                text_proj, cell_proj, mask=mask, text_sim=text_sim,
                group_ids=_group_ids,
            )
        else:
            loss, metrics = self.criterion(text_proj, cell_proj, mask=mask, text_sim=text_sim)

        # EMA update
        if self.training and self.use_ema:
            self._ema_update()

        return loss, metrics

    @torch.no_grad()
    def _embedding_mixup(
        self,
        cell_emb: torch.Tensor,
        text_emb: torch.Tensor,
        alpha: float,
        group_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply within-group embedding MixUp for regularization.

        For each cell, interpolate with a random cell from the same text group.
        This smooths the cell embedding space and reduces overfitting.

        Parameters
        ----------
        cell_emb : (B, cell_dim)
        text_emb : (B, text_dim) used for group detection (fallback)
        alpha : float, Beta distribution parameter
        group_ids : (B,) long tensor, optional
            Pre-computed group IDs. If None, computed from text_emb.

        Returns
        -------
        mixed_cell_emb : (B, cell_dim)
        """
        B = cell_emb.shape[0]
        device = cell_emb.device

        if group_ids is None:
            group_ids = self._compute_group_ids(text_emb)
        lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(device)
        lam = lam.unsqueeze(-1)  # (B, 1)

        # For each cell, find a random partner from the same group
        partners = torch.randperm(B, device=device)
        # Ensure partners share the same group; otherwise use self
        same_group = group_ids[partners] == group_ids
        partners = torch.where(same_group, partners, torch.arange(B, device=device))

        mixed = lam * cell_emb + (1 - lam) * cell_emb[partners]
        return mixed

    @staticmethod
    @torch.no_grad()
    def _compute_group_ids(text_emb: torch.Tensor) -> torch.Tensor:
        """Assign text group IDs based on raw embedding identity.

        Cells with identical text embeddings (cosine > 0.9999) get the same
        group ID. Uses raw embeddings so dropout in projection heads doesn't
        break group detection.

        IMPORTANT: Runs in float32 even under AMP autocast. Float16 matmul
        accumulation across 1024 dims loses enough precision to break the
        0.9999 cosine threshold for truly identical embeddings.

        Parameters
        ----------
        text_emb : (B, D) raw text embeddings

        Returns
        -------
        group_ids : (B,) long tensor with contiguous group IDs
        """
        B = text_emb.shape[0]
        device = text_emb.device

        # Force float32 — AMP autocast would cast matmul to float16,
        # causing identical 1024-d vectors to appear different (cosine < 0.9999)
        with torch.amp.autocast("cuda", enabled=False):
            text_f32 = text_emb.float()
            text_normed = F.normalize(text_f32, dim=-1)
            sim = text_normed @ text_normed.T  # (B, B) in float32

        is_same = sim > 0.9999

        # Assign each cell to the earliest index in its equivalence class
        group_ids = torch.arange(B, device=device)
        for i in range(1, B):
            matches = is_same[i, :i].nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                group_ids[i] = group_ids[matches[0]]

        # Remap to contiguous 0..N-1
        _, group_ids = group_ids.unique(return_inverse=True)
        return group_ids

    @staticmethod
    def _build_duplicate_mask(text_emb: torch.Tensor) -> torch.Tensor:
        """Build mask that identifies cells with identical text embeddings.

        For SigLIP: mask=1 means "valid pair" (true positive or true negative),
        mask=0 means "ambiguous/duplicate" (same text, exclude from loss).

        For each pair (i, j): if text_i == text_j and i != j, mask=0.

        IMPORTANT: Runs in float32 even under AMP autocast. Float16 matmul
        loses precision for the 0.9999 cosine threshold.

        Parameters
        ----------
        text_emb : (B, D) text embeddings

        Returns
        -------
        mask : (B, B) float tensor
        """
        B = text_emb.shape[0]

        # Force float32 for precision
        with torch.amp.autocast("cuda", enabled=False):
            text_normed = F.normalize(text_emb.float(), dim=-1)
            sim = text_normed @ text_normed.T  # (B, B)

        # Identical texts have cosine sim > 0.9999
        duplicates = (sim > 0.9999).float()

        # Keep diagonal (true positive), mask off-diagonal duplicates
        mask = 1.0 - duplicates + torch.eye(B, device=text_emb.device)
        mask = mask.clamp(0.0, 1.0)

        return mask

    def project_text(self, text_emb: torch.Tensor, use_ema: bool = False) -> torch.Tensor:
        """Project text embedding to shared space (for DiT conditioning).

        Parameters
        ----------
        text_emb : (B, text_dim) raw text embeddings (whitening applied internally)
        use_ema : bool
            Use EMA projector (more stable for inference).

        Returns
        -------
        (B, proj_dim) L2-normalized text projection
        """
        text_emb = self._whiten_text(text_emb)
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
