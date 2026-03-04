"""
Internal visualization utilities (embedding/similarity helpers).

Used by panel modules that need pairwise cosine sampling without
materializing full similarity matrices.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def sample_pairwise_cosines(
    emb_a: np.ndarray,
    emb_b: Optional[np.ndarray] = None,
    *,
    n_pairs: int = 768,
    seed: int = 42,
) -> np.ndarray:
    """Sample pairwise cosine similarities without materializing full matrices."""
    rng = np.random.default_rng(seed)
    emb_a = np.asarray(emb_a, dtype=np.float32)
    if emb_a.ndim != 2 or len(emb_a) == 0:
        return np.array([], dtype=np.float32)
    emb_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-8)

    if emb_b is None:
        if len(emb_a) < 2:
            return np.array([], dtype=np.float32)
        idx_a = rng.integers(0, len(emb_a), size=n_pairs)
        idx_b = rng.integers(0, len(emb_a) - 1, size=n_pairs)
        idx_b = np.where(idx_b >= idx_a, idx_b + 1, idx_b)
        return np.sum(emb_a[idx_a] * emb_a[idx_b], axis=1)

    emb_b = np.asarray(emb_b, dtype=np.float32)
    if emb_b.ndim != 2 or len(emb_b) == 0:
        return np.array([], dtype=np.float32)
    emb_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-8)
    idx_a = rng.integers(0, len(emb_a), size=n_pairs)
    idx_b = rng.integers(0, len(emb_b), size=n_pairs)
    return np.sum(emb_a[idx_a] * emb_b[idx_b], axis=1)
