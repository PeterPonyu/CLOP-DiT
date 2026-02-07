# cell2cell.py — Cell→Cell Latent Editing via Conditional Flow Matching
"""
Cell2Cell: Intra-modal conditional generation in scGPT latent space.

Analogy to Stable Diffusion img2img:
    Image img2img:   image → VAE encoder → latent → add noise → denoise with prompt → VAE decoder → edited image
    Cell2Cell:       cell  → scGPT encoder → z_src → add noise → flow match with condition → scGPT decoder → edited cell

Architecture:
    The DiT backbone is reused but with an additional source latent input.
    The source cell embedding z_src is injected via:
      1. Cross-attention (source as key/value, noisy z_t as query)
      2. AdaLN modulation (source features modulate the generation)
      3. Concatenation along token dimension (source + noisy tokens → self-attention)

    We use option 3 (concatenation) as the primary method — simplest, most SD-like,
    and allows the transformer to naturally attend between source and target tokens.

Use cases:
    - Perturbation prediction: control cell → drug-treated cell
    - Batch/domain translation: dataset A → dataset B
    - Trajectory editing: cell at time t₀ → cell at time t₁
    - Cell state transition: naive → activated, healthy → diseased

Flow Matching formulation for cell2cell:
    z_t = (1-t) * z_noise + t * z_tgt    (t ∈ [0, 1])
    But we start from z_src (noised) instead of pure noise:
    z_start = (1-s) * z_src + s * ε       (s = edit_strength ∈ [0, 1])
    Then flow from z_start towards z_tgt.

    Small s → gentle edit (preserve cell identity)
    Large s → strong edit (more freedom, more drift)

References:
    - SDEdit (Meng et al., 2021)
    - Flux img2img (Black Forest Labs, 2024)
    - CellOT (Bunne et al., 2023) — optimal transport for single-cell perturbation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .dit import (
    TimestepEmbedder,
    ConditionEmbedder,
    AdaLNZero,
    DiTBlock,
    DiTFinalLayer,
)


# ============================================================================
#  Source Cell Encoder — embeds z_src for cross-referencing
# ============================================================================

class SourceCellEncoder(nn.Module):
    """Encodes the source cell embedding into tokens for cross-attention.

    The source cell is tokenized identically to the noisy target, then projected
    into the same hidden space. This creates "source tokens" that the DiT blocks
    can attend to alongside the target tokens.

    Parameters
    ----------
    latent_dim : int
        Cell embedding dimension (512 from scGPT).
    hidden_dim : int
        Transformer hidden dimension.
    num_tokens : int
        Number of pseudo-tokens (matches DiT1D).
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 384, num_tokens: int = 16):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = latent_dim // num_tokens

        self.proj = nn.Linear(self.token_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)

        # Learnable source indicator embedding (distinguishes source from target tokens)
        self.source_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(self, z_src: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_src : (B, latent_dim) source cell embedding

        Returns
        -------
        src_tokens : (B, num_tokens, hidden_dim) source tokens
        """
        B = z_src.shape[0]
        x = z_src.view(B, self.num_tokens, self.token_dim)
        x = self.proj(x) + self.pos_embed + self.source_type_embed
        return x


# ============================================================================
#  Edit Strength Embedder
# ============================================================================

class EditStrengthEmbedder(nn.Module):
    """Embeds the edit strength parameter s ∈ [0, 1].

    Small s → gentle edit; large s → aggressive edit.
    This is analogous to "denoising strength" in SD img2img.

    Parameters
    ----------
    hidden_dim : int
        Output embedding dimension.
    """

    def __init__(self, hidden_dim: int, frequency_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_dim = frequency_dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        s : (B,) edit strength in [0, 1]

        Returns
        -------
        (B, hidden_dim) strength embedding
        """
        s_freq = TimestepEmbedder.sinusoidal_embedding(s, self.frequency_dim)
        return self.mlp(s_freq)


# ============================================================================
#  Cell2Cell DiT — Main Architecture
# ============================================================================

class Cell2CellDiT(nn.Module):
    """Cell-to-Cell Latent Editing via Conditional Flow Matching.

    Extends DiT1D with source cell input for intra-modal generation.
    Uses token concatenation: [source_tokens; noisy_target_tokens] → self-attention.

    The model learns: v(z_t, t, z_src, y) where:
        z_t   = noisy interpolation between source and target
        t     = flow matching timestep
        z_src = source cell embedding (anchor)
        y     = edit condition (labels, text, or both)

    Parameters
    ----------
    latent_dim : int
        Cell embedding dimension (512).
    hidden_dim : int
        Transformer hidden dimension.
    cond_dim : int
        Condition vector dimension.
    num_blocks : int
        Number of transformer blocks.
    num_heads : int
        Attention heads.
    mlp_ratio : float
        FFN expansion.
    num_tokens : int
        Pseudo-tokens per cell.
    cond_drop_prob : float
        CFG dropout for condition.
    src_drop_prob : float
        Dropout probability for source cell (enables unconditional generation too).
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
        src_drop_prob: float = 0.1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.1,
    ):
        super().__init__()
        assert latent_dim % num_tokens == 0

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.token_dim = latent_dim // num_tokens
        self.src_drop_prob = src_drop_prob

        # --- Target input projection (same as DiT1D) ---
        self.input_proj = nn.Linear(self.token_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)

        # Learnable target type embedding
        self.target_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # --- Source cell encoder ---
        self.source_encoder = SourceCellEncoder(latent_dim, hidden_dim, num_tokens)
        self.null_source = nn.Parameter(torch.randn(1, latent_dim) * 0.02)

        # --- Timestep, condition, and edit strength embedders ---
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.c_embedder = ConditionEmbedder(cond_dim, hidden_dim, dropout_prob=cond_drop_prob)
        self.s_embedder = EditStrengthEmbedder(hidden_dim)

        # --- Transformer backbone ---
        # Processes concatenated [src_tokens; tgt_tokens] sequence
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, attn_drop, proj_drop)
            for _ in range(num_blocks)
        ])

        # --- Output projection (only for target tokens) ---
        self.final_layer = DiTFinalLayer(hidden_dim, self.token_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.input_proj.apply(_init)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_src: torch.Tensor,
        cond: torch.Tensor,
        edit_strength: Optional[torch.Tensor] = None,
        force_drop_cond: bool = False,
        force_drop_src: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_t : (B, latent_dim)
            Noisy target latent.
        t : (B,)
            Flow matching timestep ∈ [0, 1].
        z_src : (B, latent_dim)
            Source cell embedding (anchor).
        cond : (B, cond_dim)
            Edit condition vector (from CLOP or label embeddings).
        edit_strength : (B,) optional
            Edit strength ∈ [0, 1]. If None, defaults to 0.5.
        force_drop_cond : bool
            Drop condition (for CFG).
        force_drop_src : bool
            Drop source (for CFG / unconditional mode).

        Returns
        -------
        v_pred : (B, latent_dim) predicted velocity field
        """
        B = z_t.shape[0]
        device = z_t.device

        # --- Source cell dropout ---
        if force_drop_src:
            z_src = self.null_source.expand(B, -1)
        elif self.training and self.src_drop_prob > 0:
            mask = torch.bernoulli(
                torch.ones(B, 1, device=device) * self.src_drop_prob
            )
            z_src = z_src * (1 - mask) + self.null_source.expand(B, -1) * mask

        # --- Tokenize target and source ---
        tgt_tokens = z_t.view(B, self.num_tokens, self.token_dim)
        tgt_tokens = self.input_proj(tgt_tokens) + self.pos_embed + self.target_type_embed
        # (B, num_tokens, hidden_dim)

        src_tokens = self.source_encoder(z_src)
        # (B, num_tokens, hidden_dim)

        # --- Concatenate: [source; target] ---
        x = torch.cat([src_tokens, tgt_tokens], dim=1)
        # (B, 2*num_tokens, hidden_dim)

        # --- Combined conditioning ---
        t_emb = self.t_embedder(t)
        c_emb = self.c_embedder(cond, force_drop=force_drop_cond)

        if edit_strength is None:
            edit_strength = torch.full((B,), 0.5, device=device)
        s_emb = self.s_embedder(edit_strength)

        combined_cond = t_emb + c_emb + s_emb
        # (B, hidden_dim)

        # --- Transformer blocks ---
        for block in self.blocks:
            x = block(x, combined_cond)

        # --- Extract target tokens only ---
        tgt_out = x[:, self.num_tokens:, :]  # (B, num_tokens, hidden_dim)

        # --- Final projection ---
        v_tokens = self.final_layer(tgt_out, combined_cond)
        # (B, num_tokens, token_dim)

        v_pred = v_tokens.reshape(B, self.latent_dim)
        return v_pred

    def forward_with_cfg(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_src: torch.Tensor,
        cond: torch.Tensor,
        edit_strength: Optional[torch.Tensor] = None,
        cfg_scale: float = 3.0,
        src_cfg_scale: float = 1.5,
    ) -> torch.Tensor:
        """Classifier-Free Guidance with dual conditioning (condition + source).

        v_guided = v_uncond
                   + cfg_scale * (v_cond_src - v_uncond)
                   + src_cfg_scale * (v_cond_src - v_nosrc)

        This provides independent control over:
          - How strongly the edit condition is followed (cfg_scale)
          - How strongly the source cell identity is preserved (src_cfg_scale)

        Parameters
        ----------
        z_t, t, z_src, cond : as in forward()
        cfg_scale : float
            Condition guidance strength.
        src_cfg_scale : float
            Source cell guidance strength.

        Returns
        -------
        v_guided : (B, latent_dim)
        """
        # Fully conditioned
        v_cond_src = self.forward(z_t, t, z_src, cond, edit_strength,
                                  force_drop_cond=False, force_drop_src=False)

        # No condition, with source
        v_nocond_src = self.forward(z_t, t, z_src, cond, edit_strength,
                                    force_drop_cond=True, force_drop_src=False)

        # With condition, no source (pure text2cell mode)
        v_cond_nosrc = self.forward(z_t, t, z_src, cond, edit_strength,
                                    force_drop_cond=False, force_drop_src=True)

        # Dual CFG
        v_guided = (v_nocond_src
                    + cfg_scale * (v_cond_src - v_nocond_src)
                    + src_cfg_scale * (v_cond_src - v_cond_nosrc))

        return v_guided

    @torch.no_grad()
    def edit(
        self,
        z_src: torch.Tensor,
        cond: torch.Tensor,
        edit_strength: float = 0.5,
        num_steps: int = 4,
        cfg_scale: float = 3.0,
        src_cfg_scale: float = 1.5,
    ) -> torch.Tensor:
        """Edit cells: source cell + condition → edited cell.

        Analogous to SD img2img:
        1. Start from noised source: z_start = (1-s)*z_src + s*ε
        2. Flow towards target condition in remaining steps

        Parameters
        ----------
        z_src : (B, latent_dim) source cell embeddings
        cond : (B, cond_dim) edit condition vectors
        edit_strength : float
            How much to deviate from source (0=no edit, 1=full generation).
        num_steps : int
            ODE integration steps.
        cfg_scale : float
            Condition guidance scale.
        src_cfg_scale : float
            Source identity preservation scale.

        Returns
        -------
        z_edited : (B, latent_dim) edited cell embeddings
        """
        B = z_src.shape[0]
        device = z_src.device

        # Noise the source according to edit strength
        noise = torch.randn_like(z_src)
        z = (1 - edit_strength) * z_src + edit_strength * noise

        # Determine starting timestep
        t_start = 1.0 - edit_strength  # More noise → start earlier in flow
        dt = edit_strength / max(num_steps, 1)

        s_tensor = torch.full((B,), edit_strength, device=device)

        for i in range(num_steps):
            t_val = t_start + i * dt
            t = torch.full((B,), t_val, device=device)

            v = self.forward_with_cfg(
                z, t, z_src, cond,
                edit_strength=s_tensor,
                cfg_scale=cfg_scale,
                src_cfg_scale=src_cfg_scale,
            )
            z = z + v * dt

        return z

    @torch.no_grad()
    def sample_text2cell(
        self,
        cond: torch.Tensor,
        num_steps: int = 4,
        cfg_scale: float = 3.0,
    ) -> torch.Tensor:
        """Fallback: pure text-to-cell generation (no source cell).

        Equivalent to DiT1D.sample() but through the Cell2Cell architecture.
        Source is dropped entirely (null source).

        Parameters
        ----------
        cond : (B, cond_dim)
        num_steps : int
        cfg_scale : float

        Returns
        -------
        z_gen : (B, latent_dim)
        """
        B = cond.shape[0]
        device = next(self.parameters()).device

        z = torch.randn(B, self.latent_dim, device=device)
        null_src = self.null_source.expand(B, -1)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((B,), t_val, device=device)

            v_cond = self.forward(z, t, null_src, cond,
                                  force_drop_cond=False, force_drop_src=True)
            v_uncond = self.forward(z, t, null_src, cond,
                                   force_drop_cond=True, force_drop_src=True)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            z = z + v * dt

        return z

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_M": f"{total / 1e6:.2f}M",
            "trainable_M": f"{trainable / 1e6:.2f}M",
        }
