# embedding_preprocessor.py — Industrial-grade embedding space preprocessing
"""
Embedding Preprocessor: Fixes the fundamental space collapse problem in both
BiomedBERT text embeddings and scGPT cell embeddings.

Root Cause Analysis (v5.2 failure):
    - BiomedBERT [CLS] embeddings: pairwise cosine sim = 0.954 (near-uniform)
    - scGPT cell embeddings: pairwise cosine sim = 0.991 (extreme collapse)
    - The contrastive alignment task is nearly impossible when BOTH spaces
      are collapsed. InfoNCE needs discriminable features to learn from.

Industrial Solutions Applied:
    1. Mean-centering: Remove the dominant mode (shifts distribution to origin)
    2. ZCA Whitening: Decorrelate dimensions AND equalize variances
       (preserves alignment with original space better than PCA whitening)
    3. L2 re-normalization: Project back to unit hypersphere for cosine-based loss
    4. Dimensionality reduction (optional): Remove low-variance noise dimensions

This is the SAME approach used by:
    - OpenCLIP (whitening of CLIP embeddings)
    - SigLIP (pre-normalizing encoder outputs)
    - Meta's ImageBind (global mean subtraction)

Usage:
    preprocessor = EmbeddingPreprocessor()
    preprocessor.fit(train_embeddings)
    clean_embeddings = preprocessor.transform(embeddings)
"""

import numpy as np
import logging
from typing import Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class EmbeddingPreprocessor:
    """Fix embedding space collapse via whitening and normalization.

    Parameters
    ----------
    method : str
        Preprocessing method:
        - 'whiten': Mean-center + ZCA whitening + L2 normalize
        - 'whiten_pca': Mean-center + PCA whitening + L2 normalize
        - 'center_norm': Mean-center + L2 normalize (lightest)
        - 'none': No preprocessing
    n_components : int or None
        If set, reduce dimensionality after whitening.
        Keeps top-N components by variance.
    eps : float
        Regularization for whitening (prevents noise amplification).
    """

    def __init__(
        self,
        method: str = "whiten",
        n_components: Optional[int] = None,
        eps: float = 1e-4,
    ):
        self.method = method
        self.n_components = n_components
        self.eps = eps

        self.mean_ = None
        self.whiten_matrix_ = None
        self.components_ = None  # For PCA-based dim reduction
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> "EmbeddingPreprocessor":
        """Compute preprocessing statistics from training data.

        Parameters
        ----------
        X : (N, D) embedding matrix

        Returns
        -------
        self
        """
        if self.method == "none":
            self.fitted_ = True
            return self

        X = X.astype(np.float64)  # Higher precision for eigendecomp
        N, D = X.shape

        # Step 1: Mean-center
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        if self.method in ("whiten", "whiten_pca"):
            # Covariance matrix
            cov = (X_centered.T @ X_centered) / (N - 1)

            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort descending (eigh returns ascending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Log variance distribution
            total_var = eigenvalues.sum()
            cumvar = np.cumsum(eigenvalues) / total_var
            logger.info(
                f"Variance explained: "
                f"top-10={cumvar[9]:.3f}, top-50={cumvar[49]:.3f}, "
                f"top-100={cumvar[min(99,D-1)]:.3f}, top-256={cumvar[min(255,D-1)]:.3f}"
            )

            # Determine effective components
            n_comp = self.n_components or D
            n_comp = min(n_comp, D)

            eigenvalues = eigenvalues[:n_comp]
            eigenvectors = eigenvectors[:, :n_comp]

            # Whitening matrix
            scale = 1.0 / np.sqrt(eigenvalues + self.eps)

            if self.method == "whiten":
                # ZCA whitening: W = V @ diag(1/sqrt(lambda)) @ V^T
                # Preserves the original coordinate alignment
                self.whiten_matrix_ = (
                    eigenvectors @ np.diag(scale) @ eigenvectors.T
                ).astype(np.float32)
            else:
                # PCA whitening: W = diag(1/sqrt(lambda)) @ V^T
                self.whiten_matrix_ = (
                    np.diag(scale) @ eigenvectors.T
                ).astype(np.float32)

            self.components_ = eigenvectors.astype(np.float32)

        self.mean_ = self.mean_.astype(np.float32)
        self.fitted_ = True

        # Verify: compute post-transform statistics
        X_out = self.transform(X.astype(np.float32)[:min(500, N)])
        post_cos = self._pairwise_cosine_mean(X_out)
        logger.info(
            f"Post-preprocessing ({self.method}): "
            f"shape={X_out.shape}, "
            f"mean_pairwise_cosine={post_cos:.4f} "
            f"(lower=better spread)"
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted preprocessing to embeddings.

        Parameters
        ----------
        X : (N, D) embedding matrix

        Returns
        -------
        X_out : (N, D') preprocessed embeddings (L2-normalized)
        """
        if self.method == "none":
            return X

        assert self.fitted_, "Must call fit() before transform()"

        X = X.astype(np.float32)

        # Mean-center
        X = X - self.mean_

        if self.whiten_matrix_ is not None:
            X = X @ self.whiten_matrix_.T

        # L2 normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        X = X / norms

        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(
        self,
        X: np.ndarray,
        target_norm: Optional[float] = None,
    ) -> np.ndarray:
        """Approximately invert the preprocessing transform.

        Reverses: L2-normalize → ZCA-whiten → mean-center.

        The L2 normalization step is lossy (original norms are discarded).
        We use ``target_norm`` to set the scale of the recovered embeddings.
        If not provided, we estimate it by applying the forward transform to
        the mean vector (which gives a representative scale).

        Parameters
        ----------
        X : (N, D) preprocessed embeddings (L2-normalized, whitened)
        target_norm : float, optional
            Target L2 norm for the reconstructed raw embeddings.
            If None, uses the mean norm of the original training data
            (estimated from the whitening matrix).

        Returns
        -------
        X_raw : (N, D) approximately reconstructed raw embeddings
        """
        if self.method == "none":
            return X

        assert self.fitted_, "Must call fit() before inverse_transform()"

        X = X.astype(np.float32)

        if self.whiten_matrix_ is not None:
            # Compute inverse whitening matrix
            W_inv = np.linalg.inv(self.whiten_matrix_).astype(np.float32)

            # Estimate pre-L2-norm scale if not provided
            if target_norm is None:
                # Use a probe: transform a small identity-like set through
                # the whitening matrix to estimate the typical output norm
                # before L2 normalization
                probe = np.eye(min(50, X.shape[1]), X.shape[1], dtype=np.float32)
                whitened_probe = probe @ self.whiten_matrix_.T
                target_norm = float(np.linalg.norm(whitened_probe, axis=1).mean())
                logger.info(f"Estimated pre-norm scale: {target_norm:.2f}")

            # Scale back from unit sphere to whitened space
            X = X * target_norm

            # Inverse whiten
            X = X @ W_inv.T
        elif target_norm is not None:
            X = X * target_norm

        # Add mean back
        if self.mean_ is not None:
            X = X + self.mean_

        return X

    @staticmethod
    def _pairwise_cosine_mean(X: np.ndarray, max_samples: int = 200) -> float:
        """Compute mean pairwise cosine similarity (for diagnostics)."""
        n = min(max_samples, len(X))
        X_sub = X[:n]
        norms = np.linalg.norm(X_sub, axis=1, keepdims=True)
        X_normed = X_sub / np.maximum(norms, 1e-8)
        sim = X_normed @ X_normed.T
        upper = sim[np.triu_indices(n, k=1)]
        return float(upper.mean())

    def save(self, path: str):
        """Save preprocessor state."""
        path = Path(path)
        state = {
            "method": self.method,
            "n_components": self.n_components,
            "eps": self.eps,
            "fitted": self.fitted_,
        }
        np.savez(
            path,
            mean=self.mean_ if self.mean_ is not None else np.array([]),
            whiten_matrix=self.whiten_matrix_ if self.whiten_matrix_ is not None else np.array([]),
            components=self.components_ if self.components_ is not None else np.array([]),
        )
        with open(str(path).replace('.npz', '_config.json'), 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EmbeddingPreprocessor":
        """Load preprocessor state."""
        path = Path(path)
        with open(str(path).replace('.npz', '_config.json')) as f:
            state = json.load(f)

        obj = cls(
            method=state["method"],
            n_components=state["n_components"],
            eps=state["eps"],
        )

        data = np.load(path, allow_pickle=True)
        obj.mean_ = data["mean"] if data["mean"].size > 0 else None
        obj.whiten_matrix_ = data["whiten_matrix"] if data["whiten_matrix"].size > 0 else None
        obj.components_ = data["components"] if data["components"].size > 0 else None
        obj.fitted_ = state["fitted"]

        return obj


def preprocess_cached_embeddings(
    cache_dir: str,
    text_method: str = "whiten",
    cell_method: str = "whiten",
    text_n_components: Optional[int] = None,
    cell_n_components: Optional[int] = None,
    output_suffix: str = "_preprocessed",
) -> dict:
    """Preprocess cached text and cell embeddings with whitening.

    This is the KEY step that fixes the alignment training failure.
    Must be run BEFORE CLOP training when working with collapsed encoders.

    Parameters
    ----------
    cache_dir : str
        Directory with cached embeddings.
    text_method : str
        Preprocessing for text embeddings ('whiten', 'center_norm', 'none').
    cell_method : str
        Preprocessing for cell embeddings ('whiten', 'center_norm', 'none').
    text_n_components : int or None
        Optional dimensionality reduction for text.
    cell_n_components : int or None
        Optional dimensionality reduction for cells.
    output_suffix : str
        Suffix for output filenames.

    Returns
    -------
    stats : dict with preprocessing statistics
    """
    cache_dir = Path(cache_dir)

    logger.info("=" * 60)
    logger.info("Embedding Preprocessing Pipeline")
    logger.info("=" * 60)

    stats = {}

    # --- Text embeddings ---
    logger.info(f"\n--- Text Embeddings (method={text_method}) ---")
    text_emb = np.load(cache_dir / "text_embeddings.npy")
    logger.info(f"Raw shape: {text_emb.shape}")

    # Only fit on unique text embeddings (avoid bias from duplicates)
    unique_texts = np.unique(text_emb, axis=0)
    logger.info(f"Unique texts for fitting: {unique_texts.shape[0]}")

    text_pre = EmbeddingPreprocessor(
        method=text_method,
        n_components=text_n_components,
    )
    text_pre.fit(unique_texts)
    text_processed = text_pre.transform(text_emb)

    text_pre.save(str(cache_dir / f"text_preprocessor{output_suffix}.npz"))
    np.save(cache_dir / f"text_embeddings{output_suffix}.npy", text_processed)

    stats["text"] = {
        "raw_shape": list(text_emb.shape),
        "processed_shape": list(text_processed.shape),
        "raw_cosine_mean": float(EmbeddingPreprocessor._pairwise_cosine_mean(unique_texts)),
        "processed_cosine_mean": float(EmbeddingPreprocessor._pairwise_cosine_mean(
            text_pre.transform(unique_texts)
        )),
    }
    logger.info(f"Text: raw_cos={stats['text']['raw_cosine_mean']:.4f} → "
                f"processed_cos={stats['text']['processed_cosine_mean']:.4f}")

    # --- Cell embeddings ---
    logger.info(f"\n--- Cell Embeddings (method={cell_method}) ---")
    cell_emb = np.load(cache_dir / "cell_embeddings.npy")
    logger.info(f"Raw shape: {cell_emb.shape}")

    cell_pre = EmbeddingPreprocessor(
        method=cell_method,
        n_components=cell_n_components,
    )
    # Fit on a subsample to save memory (220K × 512)
    fit_indices = np.random.RandomState(42).choice(
        len(cell_emb), min(50000, len(cell_emb)), replace=False
    )
    cell_pre.fit(cell_emb[fit_indices])
    cell_processed = cell_pre.transform(cell_emb)

    cell_pre.save(str(cache_dir / f"cell_preprocessor{output_suffix}.npz"))
    np.save(cache_dir / f"cell_embeddings{output_suffix}.npy", cell_processed)

    stats["cell"] = {
        "raw_shape": list(cell_emb.shape),
        "processed_shape": list(cell_processed.shape),
        "raw_cosine_mean": float(EmbeddingPreprocessor._pairwise_cosine_mean(cell_emb[:500])),
        "processed_cosine_mean": float(EmbeddingPreprocessor._pairwise_cosine_mean(cell_processed[:500])),
    }
    logger.info(f"Cell: raw_cos={stats['cell']['raw_cosine_mean']:.4f} → "
                f"processed_cos={stats['cell']['processed_cosine_mean']:.4f}")

    # Save stats
    with open(cache_dir / f"preprocessing_stats{output_suffix}.json", "w") as f:
        json.dump(stats, f, indent=2)

    # --- Dedup variant: if dedup arrays exist, transform with same preprocessors ---
    cell_dedup_path = cache_dir / "cell_embeddings_dedup.npy"
    text_dedup_path = cache_dir / "text_embeddings_dedup.npy"
    if cell_dedup_path.is_file():
        logger.info("\n--- Cell dedup: applying same preprocessor ---")
        cell_dedup = np.load(cell_dedup_path)
        cell_dedup_processed = cell_pre.transform(cell_dedup)
        np.save(cache_dir / "cell_embeddings_dedup_preprocessed.npy", cell_dedup_processed)
        logger.info(f"Saved cell_embeddings_dedup_preprocessed.npy {cell_dedup_processed.shape}")
    if text_dedup_path.is_file():
        logger.info("\n--- Text dedup: applying same preprocessor ---")
        text_dedup = np.load(text_dedup_path)
        text_dedup_processed = text_pre.transform(text_dedup)
        np.save(cache_dir / "text_embeddings_dedup_preprocessed.npy", text_dedup_processed)
        logger.info(f"Saved text_embeddings_dedup_preprocessed.npy {text_dedup_processed.shape}")

    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(f"Text cosine: {stats['text']['raw_cosine_mean']:.4f} → {stats['text']['processed_cosine_mean']:.4f}")
    logger.info(f"Cell cosine: {stats['cell']['raw_cosine_mean']:.4f} → {stats['cell']['processed_cosine_mean']:.4f}")
    logger.info("=" * 60)

    return stats
