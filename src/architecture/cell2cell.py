# cell2cell.py — Cell→Cell Latent Editing via Conditional Flow Matching (v2: AdaLN injection)
"""
Cell2Cell: Intra-modal conditional generation in scGPT latent space.

Analogy to Stable Diffusion img2img:
    Image img2img:   image → VAE encoder → latent → add noise → denoise with prompt → VAE decoder → edited image
    Cell2Cell:       cell  → scGPT encoder → z_src → add noise → flow match with condition → scGPT decoder → edited cell

Architecture (v2 — AdaLN injection, NOT token concatenation):
    v1 used [src_tokens; tgt_tokens] concatenation (32 tokens), causing 4× attention cost.
    v2 injects z_src via AdaLN conditioning:  combined_cond = t_emb + c_emb + s_emb + src_emb
    This keeps the token count at 16 (same as DiT1D) and adds <5% compute overhead.

    Source cell signal flows through the SAME AdaLN modulation pathway as timestep, text condition,
    and edit strength — a single vector that modulates every transformer block.

Use cases:
    - Perturbation prediction: control cell → drug-treated cell
    - Batch/domain translation: dataset A → dataset B
    - Trajectory editing: cell at time t₀ → cell at time t₁
    - Cell state transition: naive → activated, healthy → diseased

Flow Matching formulation for cell2cell:
    z_t = (1-t) * z_noise + t * z_tgt    (t ∈ [0, 1])
    Starting from z_src (noised) instead of pure noise:
    z_start = (1-s) * z_src + s * ε       (s = edit_strength ∈ [0, 1])

    Small s → gentle edit (preserve cell identity)
    Large s → strong edit (more freedom, more drift)

References:
    - SDEdit (Meng et al., 2021)
    - Flux img2img (Black Forest Labs, 2024)
    - CellOT (Bunne et al., 2023)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from ..utils.constants import (
    CFG_SCALE,
    COND_DROP_PROB,
    DEFAULT_ATTN_DROP,
    DEFAULT_PROJ_DROP,
    DIT_HIDDEN_DIM,
    DIT_MLP_RATIO,
    DIT_NUM_BLOCKS,
    DIT_NUM_HEADS,
    DIT_NUM_TOKENS,
    LATENT_DIM,
    PROJ_DIM,
)
from .dit import (
    TimestepEmbedder,
    ConditionEmbedder,
    AdaLNZero,
    DiTBlock,
    DiTFinalLayer,
)


# ============================================================================
#  Source Cell Embedder — compresses z_src to a conditioning vector
# ============================================================================

class SourceCellEmbedder(nn.Module):
    """Compresses z_src (512-d) into a conditioning vector (hidden_dim).

    Unlike v1's SourceCellEncoder which tokenized z_src into 16 tokens
    (doubling the sequence length to 32 and causing 4× attention cost),
    this simply projects z_src into the same hidden_dim space as
    timestep/text embeddings for AdaLN injection.

    Compute overhead: O(latent_dim × hidden_dim) — negligible.
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, z_src: torch.Tensor) -> torch.Tensor:
        """(B, latent_dim) → (B, hidden_dim)"""
        return self.mlp(z_src)


# ============================================================================
#  Edit Strength Embedder
# ============================================================================

class EditStrengthEmbedder(nn.Module):
    """Embeds the edit strength parameter s ∈ [0, 1].

    Small s → gentle edit; large s → aggressive edit.
    Uses sinusoidal encoding (same as TimestepEmbedder).
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
        s_freq = TimestepEmbedder.sinusoidal_embedding(s, self.frequency_dim)
        return self.mlp(s_freq)


# ============================================================================
#  Cell2Cell DiT — v2: AdaLN Source Injection
# ============================================================================

class Cell2CellDiT(nn.Module):
    """Cell-to-Cell Latent Editing via Conditional Flow Matching.

    v2: Source cell injected via AdaLN conditioning (NOT token concatenation).

    combined_cond = t_emb + c_emb + s_emb + src_emb
                    ^time   ^text   ^strength ^source

    This maintains the SAME 16-token sequence as DiT1D → same attention cost.
    Only overhead vs DiT1D: 3 extra small MLP embedders (~1% compute).
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = DIT_HIDDEN_DIM,
        cond_dim: int = PROJ_DIM,
        num_blocks: int = DIT_NUM_BLOCKS,
        num_heads: int = DIT_NUM_HEADS,
        mlp_ratio: float = DIT_MLP_RATIO,
        num_tokens: int = DIT_NUM_TOKENS,
        cond_drop_prob: float = COND_DROP_PROB,
        src_drop_prob: float = COND_DROP_PROB,
        attn_drop: float = DEFAULT_ATTN_DROP,
        proj_drop: float = DEFAULT_PROJ_DROP,
    ):
        super().__init__()
        assert latent_dim % num_tokens == 0

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.token_dim = latent_dim // num_tokens
        self.src_drop_prob = src_drop_prob

        # --- Target input projection (IDENTICAL to DiT1D) ---
        self.input_proj = nn.Linear(self.token_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)

        # --- Source cell embedder (lightweight MLP, NOT tokenizer) ---
        self.src_embedder = SourceCellEmbedder(latent_dim, hidden_dim)
        self.null_source = nn.Parameter(torch.zeros(1, latent_dim))

        # --- Timestep, condition, and edit strength embedders ---
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.c_embedder = ConditionEmbedder(cond_dim, hidden_dim, dropout_prob=cond_drop_prob)
        self.s_embedder = EditStrengthEmbedder(hidden_dim)

        # --- Transformer backbone (SAME blocks as DiT1D, SAME token count) ---
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, attn_drop, proj_drop)
            for _ in range(num_blocks)
        ])

        # --- Output projection ---
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
        z_t : (B, latent_dim)    Noisy target latent.
        t : (B,)                 Flow matching timestep ∈ [0, 1].
        z_src : (B, latent_dim)  Source cell embedding (anchor).
        cond : (B, cond_dim)     Edit condition vector.
        edit_strength : (B,)     Edit strength ∈ [0, 1]. Default 0.5.
        force_drop_cond : bool   Drop condition (for CFG).
        force_drop_src : bool    Drop source (for CFG).

        Returns
        -------
        v_pred : (B, latent_dim) predicted velocity field
        """
        B = z_t.shape[0]
        device = z_t.device

        # --- Source cell dropout ---
        if force_drop_src:
            z_src_input = self.null_source.expand(B, -1)
        elif self.training and self.src_drop_prob > 0:
            mask = torch.bernoulli(
                torch.ones(B, 1, device=device) * self.src_drop_prob
            )
            z_src_input = z_src * (1 - mask) + self.null_source.expand(B, -1) * mask
        else:
            z_src_input = z_src

        # --- Tokenize target ONLY (16 tokens, same as DiT1D) ---
        x = z_t.view(B, self.num_tokens, self.token_dim)
        x = self.input_proj(x) + self.pos_embed
        # (B, num_tokens, hidden_dim) ← 16 tokens, NOT 32!

        # --- Combined AdaLN conditioning (ALL signals as one vector) ---
        t_emb = self.t_embedder(t)                                   # (B, hidden_dim)
        c_emb = self.c_embedder(cond, force_drop=force_drop_cond)    # (B, hidden_dim)
        src_emb = self.src_embedder(z_src_input)                     # (B, hidden_dim)

        if edit_strength is None:
            edit_strength = torch.full((B,), 0.5, device=device)
        s_emb = self.s_embedder(edit_strength)                       # (B, hidden_dim)

        combined_cond = t_emb + c_emb + s_emb + src_emb
        # (B, hidden_dim)

        # --- Transformer blocks (16 tokens, same cost as DiT1D) ---
        for block in self.blocks:
            x = block(x, combined_cond)

        # --- Final projection ---
        v_tokens = self.final_layer(x, combined_cond)
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
        cfg_scale: float = CFG_SCALE,
        src_cfg_scale: float = 1.5,
    ) -> torch.Tensor:
        """Classifier-Free Guidance with dual conditioning.

        v_guided = v_uncond
                   + cfg_scale * (v_cond_src - v_uncond)
                   + src_cfg_scale * (v_cond_src - v_nosrc)
        """
        v_cond_src = self.forward(z_t, t, z_src, cond, edit_strength,
                                  force_drop_cond=False, force_drop_src=False)
        v_nocond_src = self.forward(z_t, t, z_src, cond, edit_strength,
                                    force_drop_cond=True, force_drop_src=False)
        v_cond_nosrc = self.forward(z_t, t, z_src, cond, edit_strength,
                                    force_drop_cond=False, force_drop_src=True)
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
        cfg_scale: float = CFG_SCALE,
        src_cfg_scale: float = 1.5,
    ) -> torch.Tensor:
        """Edit cells: source cell + condition → edited cell.

        1. Start from noised source: z_start = (1-s)*z_src + s*ε
        2. Flow towards target condition in remaining steps
        """
        B = z_src.shape[0]
        device = z_src.device

        noise = torch.randn_like(z_src)
        z = (1 - edit_strength) * z_src + edit_strength * noise

        t_start = 1.0 - edit_strength
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
        cfg_scale: float = CFG_SCALE,
    ) -> torch.Tensor:
        """Fallback: pure text-to-cell generation (no source cell)."""
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
