# embedding_quality.py — Embedding Space Quality Metrics for CLOP Training
"""
Metrics that evaluate the STRUCTURAL QUALITY of the CLOP alignment space,
rather than retrieval accuracy.  These are better proxies for downstream
cell-generation conditioning than prototype accuracy.

Rationale (from CLIP → Stable Diffusion analogy):
    In image generation, nobody evaluates CLIP by retrieval accuracy.  What
    matters is:
    1. Alignment — matched pairs are close  (Wang & Isola, ICML 2020)
    2. Uniformity — embeddings spread on the hypersphere  (ibid.)
    3. Mean cosine similarity between matched pairs  ("CLIP score")
    4. Inter-type separation / intra-type cohesion  (clustering quality)

All functions are designed to run inside `@torch.no_grad()` on the
validation loop and add negligible overhead.

Reference:
    Wang & Isola, "Understanding Contrastive Representation Learning
    through Alignment and Uniformity on the Hypersphere", ICML 2020.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


@torch.no_grad()
def compute_alignment(
    text_proj: torch.Tensor,
    cell_proj: torch.Tensor,
    alpha: float = 2.0,
) -> float:
    """Alignment loss: mean ℓ_p distance between matched (text, cell) pairs.

    Lower values indicate better pairing — each text embedding is close to its
    corresponding cell embedding in the shared space.

    ℓ_align = E[ ||f(t) − g(c)||^alpha ]   for matched (t, c)

    Parameters
    ----------
    text_proj : (B, D) L2-normalized text projections.
    cell_proj : (B, D) L2-normalized cell projections.
    alpha : float
        Exponent (default 2 = squared L2 distance).

    Returns
    -------
    float  (lower is better; range [0, 4] for L2-normalised vectors with α=2)
    """
    diff = text_proj - cell_proj  # (B, D)
    dist = (diff * diff).sum(dim=-1)  # ||.||^2 per pair
    if alpha != 2.0:
        dist = dist.pow(alpha / 2.0)
    return dist.mean().item()


@torch.no_grad()
def compute_uniformity(
    embeddings: torch.Tensor,
    t: float = 2.0,
    max_pairs: int = 10_000,
) -> float:
    """Uniformity loss: log of average Gaussian kernel between all pairs.

    Lower values indicate embeddings are more spread out on the hypersphere,
    with the optimum being a uniform distribution.

    ℓ_uniform = log E[ exp( −t ||z_i − z_j||^2 ) ]

    Parameters
    ----------
    embeddings : (N, D) L2-normalized embeddings.
    t : float
        Gaussian kernel temperature (default 2.0 per Wang & Isola).
    max_pairs : int
        Subsample if N > max_pairs to keep O(N²) tractable.

    Returns
    -------
    float  (lower is better; min ≈ −D*ln(2)/t for uniform on S^{D-1})
    """
    N = embeddings.shape[0]
    if N > max_pairs:
        idx = torch.randperm(N, device=embeddings.device)[:max_pairs]
        embeddings = embeddings[idx]
        N = max_pairs

    # Pairwise squared distances (avoid diagonal self-pairs)
    sq_dists = torch.cdist(embeddings, embeddings, p=2).pow(2)  # (N, N)
    # Mask diagonal
    mask = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
    sq_dists_off = sq_dists[mask]
    return torch.log(torch.exp(-t * sq_dists_off).mean() + 1e-10).item()


@torch.no_grad()
def compute_mean_cosine_sim(
    text_proj: torch.Tensor,
    cell_proj: torch.Tensor,
) -> float:
    """Mean cosine similarity between matched text–cell pairs (CLOP score).

    Analogous to the "CLIP score" used in image generation evaluation.
    Higher is better — means the conditioning signal is well-aligned with
    the cell representation.

    Parameters
    ----------
    text_proj : (B, D) L2-normalized text projections.
    cell_proj : (B, D) L2-normalized cell projections.

    Returns
    -------
    float  (range [−1, 1]; higher is better)
    """
    cos_sim = (text_proj * cell_proj).sum(dim=-1)  # (B,)
    return cos_sim.mean().item()


@torch.no_grad()
def compute_group_quality(
    text_proj: torch.Tensor,
    cell_proj: torch.Tensor,
    group_ids: torch.Tensor,
) -> Dict[str, float]:
    """Inter-group separation and intra-group cohesion metrics.

    Parameters
    ----------
    text_proj : (B, D) L2-normalized text projections.
    cell_proj : (B, D) L2-normalized cell projections.
    group_ids : (B,) integer group IDs.

    Returns
    -------
    dict with:
        inter_sep       : mean pairwise cosine distance between group centroids
                          (higher = better separation)
        intra_cohesion  : mean cosine similarity of cells to their group centroid
                          (higher = better cohesion)
        text_cell_align : mean cosine sim of each group's text centroid to its
                          cell centroid (conditioning fidelity; higher = better)
    """
    unique_gids, inverse = group_ids.unique(return_inverse=True)
    n_groups = unique_gids.shape[0]
    D = cell_proj.shape[1]
    device = cell_proj.device

    if n_groups < 2:
        return {"inter_sep": 0.0, "intra_cohesion": 1.0, "text_cell_align": 1.0}

    # ── Centroids ──
    cell_centroids = torch.zeros(n_groups, D, device=device)
    text_centroids = torch.zeros(n_groups, D, device=device)
    counts = torch.zeros(n_groups, 1, device=device)

    cell_centroids.scatter_add_(0, inverse.unsqueeze(-1).expand_as(cell_proj), cell_proj)
    text_centroids.scatter_add_(0, inverse.unsqueeze(-1).expand_as(text_proj), text_proj)
    counts.scatter_add_(0, inverse.unsqueeze(-1), torch.ones(cell_proj.shape[0], 1, device=device))

    cell_centroids = F.normalize(cell_centroids / counts.clamp(min=1), dim=-1)
    text_centroids = F.normalize(text_centroids / counts.clamp(min=1), dim=-1)

    # ── Inter-group separation: mean pairwise cosine DISTANCE between cell centroids ──
    if n_groups > 500:
        # Subsample for speed
        idx = torch.randperm(n_groups, device=device)[:500]
        cc_sub = cell_centroids[idx]
    else:
        cc_sub = cell_centroids
    n = cc_sub.shape[0]
    pairwise_cos = cc_sub @ cc_sub.T  # (n, n)
    mask = ~torch.eye(n, dtype=torch.bool, device=device)
    inter_sep = (1.0 - pairwise_cos[mask]).mean().item()  # cosine distance

    # ── Intra-group cohesion: mean cosine sim of cells to their group centroid ──
    cell_proto = cell_centroids[inverse]  # (B, D) centroid for each cell
    cos_to_proto = (cell_proj * cell_proto).sum(dim=-1)  # (B,)
    intra_cohesion = cos_to_proto.mean().item()

    # ── Text-cell centroid alignment (conditioning fidelity) ──
    tc_align = (text_centroids * cell_centroids).sum(dim=-1)  # (n_groups,)
    text_cell_align = tc_align.mean().item()

    return {
        "inter_sep": inter_sep,
        "intra_cohesion": intra_cohesion,
        "text_cell_align": text_cell_align,
    }


@torch.no_grad()
def compute_all_quality_metrics(
    text_proj: torch.Tensor,
    cell_proj: torch.Tensor,
    group_ids: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute all embedding quality metrics in one call.

    Designed to be called from the training validation loop.

    Parameters
    ----------
    text_proj : (B, D) L2-normalized text projections.
    cell_proj : (B, D) L2-normalized cell projections.
    group_ids : (B,) integer group IDs, optional.

    Returns
    -------
    dict with all quality metrics.
    """
    metrics = {}

    # 1. Alignment (lower is better)
    metrics["alignment"] = compute_alignment(text_proj, cell_proj)

    # 2. Uniformity of each space (lower is better)
    metrics["uniformity_text"] = compute_uniformity(text_proj)
    metrics["uniformity_cell"] = compute_uniformity(cell_proj)

    # 3. Mean cosine similarity (higher is better — "CLOP score")
    metrics["mean_cosine_sim"] = compute_mean_cosine_sim(text_proj, cell_proj)

    # 4. Group-level quality (if group_ids available)
    if group_ids is not None:
        group_metrics = compute_group_quality(text_proj, cell_proj, group_ids)
        metrics.update(group_metrics)

    return metrics
