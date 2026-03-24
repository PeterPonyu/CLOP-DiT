# losses.py — CLOP loss classes, projection heads, and text whitening.
"""Text whitening, projection heads, and contrastive losses for CLOP."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class TextWhiteningTransform(nn.Module):
    """Fixed (non-learnable) PCA whitening transform for text embeddings.

    Decorrelates and equalizes variance of BiomedBERT embeddings to counter
    embedding space collapse (pairwise cosine sim > 0.88). Computed from
    training data statistics at initialization time, then frozen.

    The transform is:
        x_whitened = (x - mean) @ whiten_matrix.T

    Parameters
    ----------
    dim : int
        Input embedding dimension (1024 for BiomedBERT-large).
    eps : float
        Regularization to prevent amplifying noise dimensions.
    """

    def __init__(self, dim: int, eps: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('whiten_matrix', torch.eye(dim))
        self.initialized = False

    def compute_stats(self, X: np.ndarray):
        """Compute whitening statistics from training text embeddings.

        Parameters
        ----------
        X : (N, dim) numpy array of training text embeddings
        """
        X_t = torch.from_numpy(X).float()
        mean = X_t.mean(dim=0)
        X_centered = X_t - mean

        # Covariance matrix
        cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)

        # Eigendecomposition (symmetric → eigenvalues are real, sorted ascending)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # PCA whitening: W = diag(1/sqrt(lambda + eps)) @ V^T
        scale = 1.0 / torch.sqrt(eigenvalues.clamp(min=self.eps) + self.eps)
        whiten_matrix = torch.diag(scale) @ eigenvectors.T

        self.mean.copy_(mean)
        self.whiten_matrix.copy_(whiten_matrix)
        self.initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply whitening: (B, dim) → (B, dim)."""
        return (x - self.mean) @ self.whiten_matrix.T


# ============================================================================
#  Projection Heads
# ============================================================================

class ProjectionHead(nn.Module):
    """MLP projection head with optional normalization and residual.

    Projects from an encoder's embedding space to the shared contrastive space.

    Parameters
    ----------
    input_dim : int
        Input embedding dimension (e.g., 1024 for BERT, 512 for scGPT).
    proj_dim : int
        Shared projection space dimension.
    hidden_dim : int, optional
        Hidden layer dimension. Defaults to input_dim.
    num_layers : int
        Number of linear layers (1 = linear projection, 2+ = MLP).
    dropout : float
        Dropout rate between layers.
    use_batch_norm : bool
        If True, use BatchNorm1d. If False, use LayerNorm.
        LayerNorm generalizes better across dataset-level splits because
        it normalizes per-sample rather than using running statistics.
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
                else:
                    layers.append(nn.LayerNorm(dims[i + 1]))
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

class SigLIPLoss(nn.Module):
    """SigLIP Sigmoid Loss for contrastive alignment (Google, 2023).

    Unlike InfoNCE which uses softmax normalization (requiring good global
    separation), SigLIP treats each (text_i, cell_j) pair independently with
    binary cross-entropy. The target is +1 for matched pairs, -1 for unmatched.

    Loss = -1/B * sum_i sum_j [
        y_ij * log(sigmoid(logit_ij)) + (1-y_ij) * log(1-sigmoid(logit_ij))
    ]
    where logit_ij = text_i · cell_j * exp(log_temp) + bias

    Key advantage: Works with COLLAPSED embedding spaces where softmax InfoNCE
    fails because the denominator is dominated by near-identical negatives.

    Parameters
    ----------
    init_temperature : float
        Initial temperature (log-space). SigLIP learns this jointly.
    init_bias : float
        Initial bias term. Shifted to compensate for imbalanced pos/neg ratios.
    """

    def __init__(
        self,
        init_temperature: float = 10.0,
        init_bias: float = -10.0,
    ):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temperature)))
        self.bias = nn.Parameter(torch.tensor(init_bias))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(
        self,
        text_proj: torch.Tensor,
        cell_proj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        text_sim: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        text_proj : (B, proj_dim) L2-normalized text projections
        cell_proj : (B, proj_dim) L2-normalized cell projections
        mask : (B, B), optional
            Duplicate mask: 1 = true pair or valid negative, 0 = duplicate
            (same-text cell) that should be excluded from loss.
        text_sim : unused, kept for API compatibility

        Returns
        -------
        loss : scalar
        metrics : dict
        """
        B = text_proj.shape[0]
        device = text_proj.device

        # Pairwise logits: (B, B)
        logits = text_proj @ cell_proj.T * self.temperature + self.bias

        # Target: +1 on diagonal (matched), -1 off-diagonal (unmatched)
        labels = 2 * torch.eye(B, device=device) - 1  # (B, B)

        # If mask provided, exclude duplicate cells from loss
        if mask is not None:
            # mask=1 means valid pair, mask=0 means duplicate to ignore
            valid = mask.float()
        else:
            valid = torch.ones(B, B, device=device)

        # Binary cross-entropy with logits: -log(sigmoid(y * logit))
        loss_matrix = -F.logsigmoid(labels * logits) * valid

        # Average over valid pairs
        n_valid = valid.sum().clamp(min=1.0)
        loss = loss_matrix.sum() / n_valid

        # Metrics
        with torch.no_grad():
            preds_t2c = logits.argmax(dim=-1)
            preds_c2t = logits.T.argmax(dim=-1)
            targets = torch.arange(B, device=device)
            acc_t2c = (preds_t2c == targets).float().mean()
            acc_c2t = (preds_c2t == targets).float().mean()

        metrics = {
            "loss_t2c": loss.item(),
            "loss_c2t": loss.item(),
            "acc_t2c": acc_t2c.item(),
            "acc_c2t": acc_c2t.item(),
            "temperature": self.temperature.item(),
            "bias": self.bias.item(),
        }

        return loss, metrics


class PrototypeSigLIPLoss(nn.Module):
    """Prototype-aware SigLIP loss for cluster-level text × sub-cluster cell alignment.

    Addresses the fundamental granularity mismatch in CLOP:
        - Text descriptions are at Leiden cluster level (~2,300 unique)
        - Cell embeddings capture per-cell variation (220K unique)
        - 67% of cell variance is WITHIN text groups (sub-cluster noise)
        - Only 33% is BETWEEN groups (text-discriminable signal)

    Standard SigLIP aligns individual (text_i, cell_i) pairs, which lets the
    model memorize per-cell noise → train_acc 80% but val_acc 1.5%.

    This loss instead:
    1. Groups cells sharing the same text embedding within each batch
    2. Computes group centroids (prototypes) in projected cell space
    3. SigLIP alignment between unique text projections and prototypes
    4. Cohesion regularization pulls cells toward their group centroid

    For singleton groups (most cells in a batch), degenerates to standard SigLIP.
    For multi-cell groups, averages out within-group noise for cleaner gradients.

    Parameters
    ----------
    init_temperature : float
        Initial temperature (log-space). Higher = sharper similarity.
    init_bias : float
        Initial bias for SigLIP. Compensates for pos/neg imbalance.
    cohesion_weight : float
        Weight for intra-group cohesion loss. Pulls cells toward their group
        centroid, explicitly regularizing within-group variation.
    max_temperature : float
        Cap on learned temperature to prevent overfitting through sharpening.
    """

    def __init__(
        self,
        init_temperature: float = 10.0,
        init_bias: float = -10.0,
        cohesion_weight: float = 0.1,
        max_temperature: float = 100.0,
        temp_reg_weight: float = 0.0,
        separation_margin: float = 0.0,
        separation_threshold: float = 0.3,
    ):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temperature)))
        self.bias = nn.Parameter(torch.tensor(init_bias))
        self.cohesion_weight = cohesion_weight
        self.max_temperature = max_temperature
        self.temp_reg_weight = temp_reg_weight  # L2 penalty on log_temperature
        # Separation margin: penalizes prototype pairs with cosine > threshold.
        # Pushes confusable cell-type prototypes apart (e.g., beta vs alpha pancreatic).
        self.separation_margin = separation_margin
        self.separation_threshold = separation_threshold

    @property
    def temperature(self) -> torch.Tensor:
        return torch.clamp(self.log_temperature.exp(), max=self.max_temperature)

    def forward(
        self,
        text_proj: torch.Tensor,
        cell_proj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        text_sim: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        variant_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        text_proj : (B, proj_dim) L2-normalized text projections
        cell_proj : (B, proj_dim) L2-normalized cell projections
        mask : (B, B), optional — duplicate mask (for fallback group detection)
        text_sim : unused, API compatibility
        group_ids : (B,) int tensor — text group assignment per cell.
            Cells sharing the same text have the same group_id.
            Computed from RAW text embeddings (before projection/dropout).
        variant_mask : (B, T) bool tensor, optional.
            Positive-bag mask for cell→text candidates. True entries indicate
            valid positive text variants for each cell. If None, uses the
            default prototype SigLIP behavior.

        Returns
        -------
        loss : scalar
        metrics : dict
        """
        B = text_proj.shape[0]
        device = text_proj.device

        # ── Step 1: Determine group IDs ──
        if group_ids is None:
            # Fallback: use mask or text_proj cosine (less reliable with dropout)
            if mask is not None:
                group_ids = self._groups_from_mask(mask, B, device)
            else:
                group_ids = torch.arange(B, device=device)

        unique_gids, inverse_ids = group_ids.unique(return_inverse=True)
        n_groups = unique_gids.shape[0]

        # ── Step 2: Compute prototypes (group centroids) via scatter ──
        # Accumulate cell projections per group
        prototypes = torch.zeros(n_groups, cell_proj.shape[-1], device=device)
        prototypes.scatter_add_(
            0, inverse_ids.unsqueeze(-1).expand_as(cell_proj), cell_proj
        )
        counts = torch.zeros(n_groups, 1, device=device)
        counts.scatter_add_(
            0, inverse_ids.unsqueeze(-1),
            torch.ones(B, 1, device=device)
        )
        prototypes = prototypes / counts.clamp(min=1)
        prototypes = F.normalize(prototypes, dim=-1)

        # Representative text for each group (first cell in each group)
        # NOTE: Using first-cell rather than mean-text is critical for gradient
        # flow. Mean-text dilutes the signal (tested: temp stuck at 4.2).
        # When caption variants are active, this cell may have a variant
        # embedding; see CLOPAligner.forward() for how original embeddings
        # are preserved for text_rep when needed.
        first_indices = torch.zeros(n_groups, dtype=torch.long, device=device)
        seen = torch.zeros(n_groups, dtype=torch.bool, device=device)
        for i in range(B):
            g = inverse_ids[i].item()
            if not seen[g]:
                first_indices[g] = i
                seen[g] = True
        text_reps = text_proj[first_indices]
        text_reps = F.normalize(text_reps, dim=-1)

        # ── Step 3: SigLIP loss on prototypes (N × N) ──
        logits = (text_reps @ prototypes.T) * self.temperature + self.bias
        labels = 2 * torch.eye(n_groups, device=device) - 1
        alignment_loss = -F.logsigmoid(labels * logits).mean()

        bag_loss = torch.tensor(0.0, device=device)
        if variant_mask is not None:
            logits_bt = cell_proj @ text_proj.T * self.temperature + self.bias
            bag_loss = self._bag_positive_nce_loss(logits_bt, variant_mask.to(device))
            alignment_loss = bag_loss

        # ── Step 4: Cohesion regularization ──
        cohesion_loss = torch.tensor(0.0, device=device)
        if self.cohesion_weight > 0:
            # For each cell, compute cosine to its group prototype (detached)
            proto_for_cells = prototypes[inverse_ids].detach()  # (B, D)
            cos_to_proto = (cell_proj * proto_for_cells).sum(dim=-1)  # (B,)
            cohesion_loss = (1 - cos_to_proto).mean()

        total_loss = alignment_loss + self.cohesion_weight * cohesion_loss

        # ── Step 5: Separation margin for confusable prototypes ──
        separation_loss = torch.tensor(0.0, device=device)
        if self.separation_margin > 0 and n_groups > 1:
            # Cell-side: push apart confusable cell prototypes
            proto_sim = prototypes @ prototypes.T  # (n_groups, n_groups)
            diag_mask = torch.eye(n_groups, device=device, dtype=torch.bool)
            proto_sim = proto_sim.masked_fill(diag_mask, -1.0)
            violations = torch.clamp(proto_sim - self.separation_threshold, min=0.0)
            if violations.sum() > 0:
                separation_loss = violations.sum() / max(1, (violations > 0).sum())

            # Text-side: push apart confusable text projections
            text_sim = text_reps @ text_reps.T  # (n_groups, n_groups)
            text_sim = text_sim.masked_fill(diag_mask, -1.0)
            text_violations = torch.clamp(text_sim - self.separation_threshold, min=0.0)
            if text_violations.sum() > 0:
                text_sep = text_violations.sum() / max(1, (text_violations > 0).sum())
                separation_loss = separation_loss + text_sep

            total_loss = total_loss + self.separation_margin * separation_loss

        # Temperature regularization: prevent saturation
        # 1. L2 penalty on log_temperature (prevents unbounded growth)
        # 2. Soft penalty if temperature approaches max (encourages staying below max)
        if self.temp_reg_weight > 0:
            # Base L2 penalty
            temp_reg = self.temp_reg_weight * (self.log_temperature ** 2)

            # Additional soft penalty if temperature is close to max
            # This creates a "soft wall" that discourages but doesn't hard-clamp
            current_temp = self.log_temperature.exp()
            if self.max_temperature > 0:
                excess = torch.clamp(current_temp - self.max_temperature * 0.85, min=0)
                temp_reg = temp_reg + self.temp_reg_weight * 10 * (excess ** 2)

            total_loss = total_loss + temp_reg

        # ── Metrics ──
        with torch.no_grad():
            # Prototype-level accuracy
            proto_targets = torch.arange(n_groups, device=device)
            proto_acc_t2c = (logits.argmax(dim=-1) == proto_targets).float().mean()
            proto_acc_c2t = (logits.T.argmax(dim=-1) == proto_targets).float().mean()

            # Top-k prototype accuracy (k=5)
            if n_groups >= 5:
                top5_t2c = (logits.topk(5, dim=-1).indices == proto_targets.unsqueeze(-1)).any(-1).float().mean()
                top5_c2t = (logits.T.topk(5, dim=-1).indices == proto_targets.unsqueeze(-1)).any(-1).float().mean()
            else:
                top5_t2c = proto_acc_t2c
                top5_c2t = proto_acc_c2t

            # Top-k prototype accuracy (k=10)
            if n_groups >= 10:
                top10_t2c = (logits.topk(10, dim=-1).indices == proto_targets.unsqueeze(-1)).any(-1).float().mean()
                top10_c2t = (logits.T.topk(10, dim=-1).indices == proto_targets.unsqueeze(-1)).any(-1).float().mean()
            else:
                top10_t2c = top5_t2c
                top10_c2t = top5_c2t

            # Individual-level accuracy (for comparison with standard SigLIP)
            ind_logits = text_proj @ cell_proj.T * self.temperature + self.bias
            ind_targets = torch.arange(B, device=device)
            acc_t2c = (ind_logits.argmax(dim=-1) == ind_targets).float().mean()
            acc_c2t = (ind_logits.T.argmax(dim=-1) == ind_targets).float().mean()

        metrics = {
            "loss_t2c": alignment_loss.item(),
            "loss_c2t": alignment_loss.item(),
            "bag_loss": bag_loss.item() if variant_mask is not None else 0.0,
            "acc_t2c": acc_t2c.item(),
            "acc_c2t": acc_c2t.item(),
            "proto_acc_t2c": proto_acc_t2c.item(),
            "proto_acc_c2t": proto_acc_c2t.item(),
            "proto_top5_t2c": top5_t2c.item(),
            "proto_top5_c2t": top5_c2t.item(),
            "proto_top10_t2c": top10_t2c.item(),
            "proto_top10_c2t": top10_c2t.item(),
            "temperature": self.temperature.item(),
            "bias": self.bias.item(),
            "n_groups": n_groups,
            "cohesion_loss": cohesion_loss.item(),
            "separation_loss": separation_loss.item(),
        }

        return total_loss, metrics

    @staticmethod
    def _bag_positive_nce_loss(logits: torch.Tensor, variant_mask: torch.Tensor) -> torch.Tensor:
        """Compute positive-bag NCE loss from logits and boolean positive mask.

        Parameters
        ----------
        logits : (B, T)
            Cell→text similarity logits.
        variant_mask : (B, T)
            True entries mark valid positives for each row.

        Returns
        -------
        loss : scalar tensor
        """
        if variant_mask.dtype != torch.bool:
            variant_mask = variant_mask.bool()

        if variant_mask.shape != logits.shape:
            raise ValueError(
                f"variant_mask shape {tuple(variant_mask.shape)} must match logits shape {tuple(logits.shape)}"
            )

        # Guarantee at least one positive per row (fallback to diagonal for square logits)
        row_has_pos = variant_mask.any(dim=1)
        if not row_has_pos.all():
            if logits.shape[0] == logits.shape[1]:
                diag = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
                variant_mask = variant_mask | diag
            else:
                raise ValueError("variant_mask has rows without positives")

        positive_logits = logits.masked_fill(~variant_mask, float("-inf"))
        per_cell_pos = positive_logits.logsumexp(dim=1)
        per_cell_all = logits.logsumexp(dim=1)
        return -(per_cell_pos - per_cell_all).mean()

    @staticmethod
    def _groups_from_mask(mask: torch.Tensor, B: int, device: torch.device) -> torch.Tensor:
        """Convert duplicate mask to group IDs. Fallback when group_ids not provided."""
        # mask[i,j]=0 and i!=j means text_i == text_j
        group_ids = torch.arange(B, device=device)
        for i in range(B):
            if group_ids[i] == i:  # not yet merged
                for j in range(i + 1, B):
                    if mask[i, j] < 0.5 and group_ids[j] == j:
                        group_ids[j] = group_ids[i]
        _, group_ids = group_ids.unique(return_inverse=True)
        return group_ids


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE contrastive loss with learnable temperature and soft labels.

    Kept for backward compatibility and ablation. For new training runs with
    collapsed embedding spaces, prefer SigLIPLoss or PrototypeSigLIPLoss.

    Supports two modes:
    - Hard labels (standard): L = 0.5 * (CE(logits, arange(B)) + CE(logits.T, arange(B)))
    - Soft labels: Uses text embedding similarity to build soft target distributions.
      Semantically similar texts get partial positive credit instead of being treated
      as full negatives.

    Parameters
    ----------
    init_temperature : float
        Initial temperature value (log-space).
    min_temperature : float
        Minimum temperature for clamping.
    max_temperature : float
        Maximum temperature for clamping.
    label_smoothing : float
        Label smoothing for cross-entropy (only used in hard-label mode).
    use_soft_labels : bool
        If True, use text-similarity-based soft labels instead of hard labels.
    soft_label_alpha : float
        Exponent for sharpening text similarity when building soft targets.
    soft_label_bias : float
        Additive bias for the diagonal (true positive) in soft targets.
    """

    def __init__(
        self,
        init_temperature: float = 0.07,
        min_temperature: float = 0.01,
        max_temperature: float = 0.5,
        label_smoothing: float = 0.1,
        use_soft_labels: bool = False,
        soft_label_alpha: float = 2.0,
        soft_label_bias: float = 5.0,
    ):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temperature)))
        self.min_temp = min_temperature
        self.max_temp = max_temperature
        self.label_smoothing = label_smoothing
        self.use_soft_labels = use_soft_labels
        self.soft_label_alpha = soft_label_alpha
        self.soft_label_bias = soft_label_bias

    @property
    def temperature(self) -> torch.Tensor:
        """Clamped temperature value."""
        return torch.clamp(
            self.log_temperature.exp(),
            min=self.min_temp,
            max=self.max_temp,
        )

    def _build_soft_targets(
        self,
        text_sim: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build soft target distribution from text similarity matrix.

        Parameters
        ----------
        text_sim : (B, B) cosine similarity between input text embeddings
        mask : (B, B) optional validity mask (1 = valid pair)

        Returns
        -------
        targets : (B, B) row-normalized soft target distribution
        """
        B = text_sim.shape[0]

        # Clamp similarity to [0, 1]
        sim_clamped = text_sim.clamp(min=0.0)

        # Sharpen: raise to power alpha to separate similar from dissimilar
        raw_targets = sim_clamped.pow(self.soft_label_alpha)

        # Boost diagonal (true positive pair) with additive bias
        raw_targets = raw_targets + self.soft_label_bias * torch.eye(
            B, device=text_sim.device, dtype=text_sim.dtype
        )

        # Zero out masked (invalid) positions
        if mask is not None:
            raw_targets = raw_targets.masked_fill(~mask.bool(), 0.0)

        # Row-normalize to form valid probability distribution
        targets = raw_targets / raw_targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return targets

    def forward(
        self,
        text_proj: torch.Tensor,
        cell_proj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        text_sim: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        text_proj : (B, proj_dim) L2-normalized text projections
        cell_proj : (B, proj_dim) L2-normalized cell projections
        mask : (B, B), optional
            Binary mask where 1 = valid negative pair.
        text_sim : (B, B), optional
            Cosine similarity between raw text embeddings in the batch.
            Required when use_soft_labels=True.

        Returns
        -------
        loss : scalar
        metrics : dict with accuracy, temperature, individual losses
        """
        B = text_proj.shape[0]
        device = text_proj.device

        # Cosine similarity matrix / temperature
        logits = (text_proj @ cell_proj.T) / self.temperature  # (B, B)

        if self.use_soft_labels and text_sim is not None:
            # ── Soft-label mode ──
            targets = self._build_soft_targets(text_sim, mask=mask)

            if mask is not None:
                logits_masked = logits.masked_fill(~mask.bool(), -65000.0)
            else:
                logits_masked = logits

            # Soft cross-entropy: L = -sum(targets * log_softmax(logits))
            log_probs_t2c = F.log_softmax(logits_masked, dim=-1)
            log_probs_c2t = F.log_softmax(logits_masked.T, dim=-1)

            loss_t2c = -(targets * log_probs_t2c).sum(dim=-1).mean()
            loss_c2t = -(targets * log_probs_c2t).sum(dim=-1).mean()
            loss = 0.5 * (loss_t2c + loss_c2t)

        else:
            # ── Hard-label mode (original) ──
            if mask is not None and self.label_smoothing > 0:
                # Custom label-smoothed CE that only distributes smoothing
                # among VALID (unmasked) classes. Using -65000 fill with
                # standard F.cross_entropy would cause label_smoothing to
                # penalize masked entries: (ls/B) * 65000 ≈ 12.7 per masked class.
                logits_masked = logits.masked_fill(~mask.bool(), float('-inf'))
                labels = torch.arange(B, device=device)

                # Valid class count per row
                valid_count = mask.bool().sum(dim=-1, keepdim=True).float()  # (B, 1)

                # Build smoothed targets: diagonal gets (1-ls), rest of valid
                # classes share ls equally, masked classes get 0
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
                # Distribute smoothing only among valid off-diagonal entries
                mask_no_diag = mask.bool().clone()
                mask_no_diag.fill_diagonal_(False)
                valid_off_diag = mask_no_diag.sum(dim=-1, keepdim=True).float().clamp(min=1)
                smooth_targets += (self.label_smoothing / valid_off_diag) * mask_no_diag.float()

                # Cross-entropy with custom targets
                log_probs_t2c = F.log_softmax(logits_masked, dim=-1)
                log_probs_c2t = F.log_softmax(logits_masked.T, dim=-1)
                loss_t2c = -(smooth_targets * log_probs_t2c).sum(dim=-1).mean()
                loss_c2t = -(smooth_targets * log_probs_c2t).sum(dim=-1).mean()
                loss = 0.5 * (loss_t2c + loss_c2t)
            else:
                # Standard CE: no label smoothing, or no mask
                if mask is not None:
                    logits = logits.masked_fill(~mask.bool(), float('-inf'))
                labels = torch.arange(B, device=device)
                loss_t2c = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
                loss_c2t = F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)
                loss = 0.5 * (loss_t2c + loss_c2t)

        # Metrics (always use hard accuracy for comparability)
        with torch.no_grad():
            labels = torch.arange(B, device=device)
            if mask is not None:
                logits_for_acc = logits.masked_fill(~mask.bool(), -65000.0) if self.use_soft_labels else logits
            else:
                logits_for_acc = logits
            acc_t2c = (logits_for_acc.argmax(dim=-1) == labels).float().mean()
            acc_c2t = (logits_for_acc.T.argmax(dim=-1) == labels).float().mean()

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
