# dit.py — 1D Diffusion Transformer with AdaLN-Zero conditioning (Flux-style)
"""
1D Diffusion Transformer for single-cell latent space generation.

Architecture:
    - Sinusoidal timestep embedding → MLP → t_emb
    - Condition vector (from CLOP aligner) → MLP → c_emb
    - Combined conditioning: t_emb + c_emb → modulation signal
    - N x DiTBlock with AdaLN-Zero (adaptive LayerNorm + zero-init gating)
    - Final linear → velocity prediction v(z_t, t, c)

Flow Matching formulation:
    z_t = (1 - t) * z_0 + t * z_1,  where z_0 ~ N(0,I), z_1 = real embedding
    v_target = z_1 - z_0
    Loss = MSE(v_pred, v_target)

References:
    - Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)
    - Flow Matching for Generative Modeling (Lipman et al., 2023)
    - Flux (Black Forest Labs, 2024)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
#  Timestep & Condition Embedders
# ============================================================================

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding → MLP projection.

    Parameters
    ----------
    hidden_dim : int
        Output embedding dimension.
    frequency_dim : int
        Dimension of sinusoidal frequency basis (half of raw embedding dim).
    """

    def __init__(self, hidden_dim: int, frequency_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_dim = frequency_dim

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal positional embeddings for timestep t ∈ [0, 1]."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (B,) tensor of timesteps in [0, 1]

        Returns
        -------
        (B, hidden_dim) timestep embeddings
        """
        t_freq = self.sinusoidal_embedding(t, self.frequency_dim)
        return self.mlp(t_freq)


class ConditionEmbedder(nn.Module):
    """Projects CLOP-aligned condition vector into DiT modulation space.

    Supports classifier-free guidance via random dropout of condition.

    Parameters
    ----------
    cond_dim : int
        Dimension of the input condition vector (from CLOP projector).
    hidden_dim : int
        Output dimension matching DiT hidden size.
    dropout_prob : float
        Probability of dropping condition (replacing with learned null token).
    """

    def __init__(self, cond_dim: int, hidden_dim: int, dropout_prob: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.null_cond = nn.Parameter(torch.randn(1, cond_dim) * 0.02)
        self.dropout_prob = dropout_prob

    def forward(self, cond: torch.Tensor, force_drop: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        cond : (B, cond_dim) condition vectors
        force_drop : bool
            If True, always use null condition (for CFG inference).

        Returns
        -------
        (B, hidden_dim) condition embeddings
        """
        if self.training and self.dropout_prob > 0:
            mask = torch.bernoulli(
                torch.ones(cond.shape[0], 1, device=cond.device) * self.dropout_prob
            )
            cond = cond * (1 - mask) + self.null_cond.expand_as(cond) * mask
        elif force_drop:
            cond = self.null_cond.expand(cond.shape[0], -1)
        return self.mlp(cond)


# ============================================================================
#  AdaLN-Zero Modulation
# ============================================================================

class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with Zero-initialized gating.

    Computes 6 modulation parameters (γ1, β1, α1, γ2, β2, α2)
    from the combined timestep + condition embedding.
    α (gate) is zero-initialized so blocks start as identity.

    Parameters
    ----------
    hidden_dim : int
        DiT hidden dimension.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        # Zero-initialize the gate projections
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """
        Parameters
        ----------
        x : (B, L, D) input features
        cond : (B, D) combined conditioning signal

        Returns
        -------
        Tuple of 6 modulation tensors, each (B, 1, D)
        """
        cond = self.silu(cond)
        params = self.linear(cond).unsqueeze(1)  # (B, 1, 6*D)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)
        return gamma1, beta1, alpha1, gamma2, beta2, alpha2


# ============================================================================
#  DiT Block (Single Transformer Layer with AdaLN-Zero)
# ============================================================================

class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning.

    Structure:
        x → LN → modulate(γ1,β1) → Self-Attention → gate(α1) → residual
        x → LN → modulate(γ2,β2) → FFN → gate(α2) → residual

    Parameters
    ----------
    hidden_dim : int
        Model dimension.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        FFN hidden dim = hidden_dim * mlp_ratio.
    attn_drop : float
        Attention dropout rate.
    proj_drop : float
        Projection dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.attn_proj_drop = nn.Dropout(proj_drop)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(proj_drop),
        )

        self.adaln = AdaLNZero(hidden_dim)

    def _modulate(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Apply affine modulation: x * (1 + γ) + β"""
        return x * (1 + gamma) + beta

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D) token sequence
        cond : (B, D) conditioning signal (t_emb + c_emb)

        Returns
        -------
        (B, L, D) output features
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(x, cond)

        # --- Self-Attention with AdaLN ---
        h = self.norm1(x)
        h = self._modulate(h, gamma1, beta1)
        h, _ = self.attn(h, h, h)
        h = self.attn_proj_drop(h)
        x = x + alpha1 * h

        # --- FFN with AdaLN ---
        h = self.norm2(x)
        h = self._modulate(h, gamma2, beta2)
        h = self.ffn(h)
        x = x + alpha2 * h

        return x


# ============================================================================
#  Final Layer (AdaLN → Linear → velocity)
# ============================================================================

class DiTFinalLayer(nn.Module):
    """Final layer: AdaLN + Linear projection to output space.

    Parameters
    ----------
    hidden_dim : int
        Input dimension.
    out_dim : int
        Output dimension (= latent_dim, i.e., cell embedding dim).
    """

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        # Zero-init
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D)
        cond : (B, D) combined conditioning

        Returns
        -------
        (B, L, out_dim) velocity prediction
        """
        gamma, beta = self.adaln_modulation(cond).unsqueeze(1).chunk(2, dim=-1)
        h = self.norm(x)
        h = h * (1 + gamma) + beta
        return self.linear(h)


# ============================================================================
#  DiT1D — Complete 1D Diffusion Transformer
# ============================================================================

class DiT1D(nn.Module):
    """1D Diffusion Transformer for single-cell latent generation.

    Takes a noisy latent z_t, timestep t, and CLOP condition c,
    and predicts the velocity field v = z_1 - z_0 for flow matching.

    The cell embedding is reshaped into a short "pseudo-sequence" of tokens
    for the transformer to process. This allows attention to model inter-dimensional
    correlations within the embedding.

    Parameters
    ----------
    latent_dim : int
        Dimension of cell embedding (e.g., 512 from scGPT).
    hidden_dim : int
        Transformer hidden dimension.
    cond_dim : int
        Condition vector dimension (from CLOP projector).
    num_blocks : int
        Number of DiT transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        FFN expansion ratio.
    num_tokens : int
        Number of pseudo-tokens to split the latent into.
        latent_dim must be divisible by num_tokens.
    cond_drop_prob : float
        Classifier-free guidance dropout probability.
    attn_drop : float
        Attention dropout.
    proj_drop : float
        Projection dropout.

    Notes
    -----
    For a 512-dim embedding with num_tokens=16, each token is 32-dim,
    projected up to hidden_dim via the input projection.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 384,
        cond_dim: int = 256,
        num_blocks: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_tokens: int = 16,
        cond_drop_prob: float = 0.1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.1,
    ):
        super().__init__()
        assert latent_dim % num_tokens == 0, \
            f"latent_dim ({latent_dim}) must be divisible by num_tokens ({num_tokens})"

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.token_dim = latent_dim // num_tokens

        # --- Input projection: token_dim → hidden_dim ---
        self.input_proj = nn.Linear(self.token_dim, hidden_dim)

        # --- Positional embedding for pseudo-tokens ---
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_tokens, hidden_dim) * 0.02
        )

        # --- Timestep & Condition embedders ---
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.c_embedder = ConditionEmbedder(cond_dim, hidden_dim, dropout_prob=cond_drop_prob)

        # --- Transformer backbone ---
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, attn_drop, proj_drop)
            for _ in range(num_blocks)
        ])

        # --- Final output ---
        self.final_layer = DiTFinalLayer(hidden_dim, self.token_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for linear layers."""
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Only init input_proj and pos_embed; AdaLN layers are already zero-init
        self.input_proj.apply(_init)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        force_drop_cond: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_t : (B, latent_dim) noisy latent embedding
        t : (B,) timestep in [0, 1]
        cond : (B, cond_dim) CLOP condition vector
        force_drop_cond : bool
            Force null conditioning (for CFG unconditional pass).

        Returns
        -------
        v_pred : (B, latent_dim) predicted velocity field
        """
        B = z_t.shape[0]

        # Reshape to pseudo-token sequence: (B, latent_dim) → (B, num_tokens, token_dim)
        x = z_t.view(B, self.num_tokens, self.token_dim)

        # Input projection + positional embedding
        x = self.input_proj(x) + self.pos_embed

        # Conditioning
        t_emb = self.t_embedder(t)                              # (B, hidden_dim)
        c_emb = self.c_embedder(cond, force_drop=force_drop_cond)  # (B, hidden_dim)
        combined_cond = t_emb + c_emb                           # (B, hidden_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, combined_cond)

        # Final projection: (B, num_tokens, token_dim)
        x = self.final_layer(x, combined_cond)

        # Reshape back: (B, num_tokens, token_dim) → (B, latent_dim)
        v_pred = x.reshape(B, self.latent_dim)

        return v_pred

    def forward_with_cfg(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        cfg_scale: float = 3.0,
    ) -> torch.Tensor:
        """Classifier-Free Guidance inference.

        v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)

        Parameters
        ----------
        z_t : (B, latent_dim)
        t : (B,)
        cond : (B, cond_dim)
        cfg_scale : float
            Guidance strength. 1.0 = no guidance; >1.0 = stronger conditioning.

        Returns
        -------
        v_guided : (B, latent_dim)
        """
        # Conditional prediction
        v_cond = self.forward(z_t, t, cond, force_drop_cond=False)
        # Unconditional prediction
        v_uncond = self.forward(z_t, t, cond, force_drop_cond=True)
        # Guided velocity
        v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
        return v_guided

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        num_steps: int = 4,
        cfg_scale: float = 3.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate cell embeddings via Euler ODE integration.

        Integrates the learned velocity field from t=0 (noise) to t=1 (data).

        Parameters
        ----------
        cond : (B, cond_dim) condition vectors
        num_steps : int
            Number of Euler integration steps (3-8 is typically sufficient).
        cfg_scale : float
            Classifier-free guidance scale.
        device : torch.device, optional

        Returns
        -------
        z_1 : (B, latent_dim) generated cell embeddings
        """
        if device is None:
            device = next(self.parameters()).device

        B = cond.shape[0]
        z = torch.randn(B, self.latent_dim, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((B,), t_val, device=device)
            v = self.forward_with_cfg(z, t, cond, cfg_scale=cfg_scale)
            z = z + v * dt

        return z

    @torch.no_grad()
    def sample_midpoint(
        self,
        cond: torch.Tensor,
        num_steps: int = 4,
        cfg_scale: float = 3.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate cell embeddings via Midpoint method (2nd-order ODE solver).

        More accurate than Euler for the same number of function evaluations.

        Parameters
        ----------
        cond : (B, cond_dim) condition vectors
        num_steps : int
            Number of midpoint integration steps.
        cfg_scale : float
            Classifier-free guidance scale.
        device : torch.device, optional

        Returns
        -------
        z_1 : (B, latent_dim) generated cell embeddings
        """
        if device is None:
            device = next(self.parameters()).device

        B = cond.shape[0]
        z = torch.randn(B, self.latent_dim, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((B,), t_val, device=device)

            # Half step
            v1 = self.forward_with_cfg(z, t, cond, cfg_scale=cfg_scale)
            z_mid = z + v1 * (dt / 2)

            # Full step using midpoint velocity
            t_mid = torch.full((B,), t_val + dt / 2, device=device)
            v2 = self.forward_with_cfg(z_mid, t_mid, cond, cfg_scale=cfg_scale)
            z = z + v2 * dt

        return z

    def count_parameters(self) -> dict:
        """Return parameter counts by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_M": f"{total / 1e6:.2f}M",
            "trainable_M": f"{trainable / 1e6:.2f}M",
        }
