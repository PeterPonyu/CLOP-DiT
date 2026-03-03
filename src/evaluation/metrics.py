# metrics.py — Evaluation metrics for CLOP-DiT
"""
Comprehensive evaluation metrics for single-cell generation quality.

Covers:
    1. Embedding space metrics (distribution matching)
    2. Biological plausibility metrics (marker gene expression, cell type purity)
    3. Diversity & coverage metrics (mode collapse detection)
    4. CLOP alignment quality (retrieval metrics)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from scipy import stats
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


class GenerationMetrics:
    """Comprehensive metrics for evaluating generated single-cell data.

    All metrics operate on numpy arrays for compatibility with scanpy.
    """

    @staticmethod
    def frechet_distance(
        real: np.ndarray,
        generated: np.ndarray,
    ) -> float:
        """Fréchet Distance (FD) between real and generated embeddings.

        Analogous to FID in image generation.
        FD = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})

        Parameters
        ----------
        real : (N, D) real cell embeddings
        generated : (M, D) generated cell embeddings

        Returns
        -------
        fd : float
            Lower is better. 0 = identical distributions.
        """
        mu_r = real.mean(axis=0)
        mu_g = generated.mean(axis=0)
        sigma_r = np.cov(real, rowvar=False)
        sigma_g = np.cov(generated, rowvar=False)

        diff = mu_r - mu_g
        diff_sq = np.dot(diff, diff)

        # Matrix square root via eigendecomposition
        from scipy.linalg import sqrtm
        covmean = sqrtm(sigma_r @ sigma_g)

        # Numerical stability
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fd = diff_sq + np.trace(sigma_r + sigma_g - 2 * covmean)
        return float(fd)

    @staticmethod
    def mmd(
        real: np.ndarray,
        generated: np.ndarray,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
    ) -> float:
        """Maximum Mean Discrepancy between distributions.

        Parameters
        ----------
        real : (N, D)
        generated : (M, D)
        kernel : str
            Kernel type ('rbf', 'linear').
        gamma : float, optional
            RBF kernel bandwidth. Auto if None.

        Returns
        -------
        mmd : float
            Lower is better.
        """
        from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

        if kernel == "rbf":
            if gamma is None:
                gamma = 1.0 / real.shape[1]
            K_rr = rbf_kernel(real, real, gamma=gamma)
            K_gg = rbf_kernel(generated, generated, gamma=gamma)
            K_rg = rbf_kernel(real, generated, gamma=gamma)
        elif kernel == "linear":
            K_rr = linear_kernel(real, real)
            K_gg = linear_kernel(generated, generated)
            K_rg = linear_kernel(real, generated)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        mmd_val = K_rr.mean() + K_gg.mean() - 2 * K_rg.mean()
        return float(mmd_val)

    @staticmethod
    def coverage_and_density(
        real: np.ndarray,
        generated: np.ndarray,
        k: int = 5,
    ) -> Dict[str, float]:
        """Coverage and Density metrics for mode collapse detection.

        Coverage: fraction of real samples with a generated neighbor within
                  the k-NN radius of real data.
        Density:  average number of generated samples falling within
                  the k-NN radius of each real sample.

        Parameters
        ----------
        real : (N, D)
        generated : (M, D)
        k : int
            Number of nearest neighbors.

        Returns
        -------
        metrics : dict with 'coverage' and 'density'
        """
        # Compute k-NN radius for real data
        nn_real = NearestNeighbors(n_neighbors=k + 1).fit(real)
        distances_real, _ = nn_real.kneighbors(real)
        radii = distances_real[:, -1]  # k-th neighbor distance

        # For each real sample, count generated samples within radius
        nn_gen = NearestNeighbors(n_neighbors=1).fit(generated)
        gen_dists, _ = nn_gen.kneighbors(real)
        gen_nearest = gen_dists[:, 0]

        coverage = (gen_nearest <= radii).mean()

        # Density: for each real, count generated within radius
        dists_rg = pairwise_distances(real, generated)
        density = (dists_rg <= radii[:, None]).sum(axis=1).mean() / k

        return {
            "coverage": float(coverage),
            "density": float(density),
        }

    @staticmethod
    def kl_per_dimension(
        real: np.ndarray,
        generated: np.ndarray,
        n_bins: int = 50,
    ) -> Dict[str, float]:
        """Per-dimension KL divergence between real and generated distributions.

        Parameters
        ----------
        real : (N, D)
        generated : (M, D)
        n_bins : int

        Returns
        -------
        metrics : dict with 'mean_kl', 'max_kl', 'kl_per_dim'
        """
        D = real.shape[1]
        kl_values = []

        for d in range(D):
            # Histogram-based KL
            r_min = min(real[:, d].min(), generated[:, d].min())
            r_max = max(real[:, d].max(), generated[:, d].max())
            bins = np.linspace(r_min - 1e-6, r_max + 1e-6, n_bins + 1)

            p, _ = np.histogram(real[:, d], bins=bins, density=True)
            q, _ = np.histogram(generated[:, d], bins=bins, density=True)

            # Smoothing
            p = p + 1e-8
            q = q + 1e-8
            p = p / p.sum()
            q = q / q.sum()

            kl = stats.entropy(p, q)
            kl_values.append(kl)

        return {
            "mean_kl": float(np.mean(kl_values)),
            "max_kl": float(np.max(kl_values)),
            "median_kl": float(np.median(kl_values)),
        }

    @staticmethod
    def clop_retrieval(
        text_proj: np.ndarray,
        cell_proj: np.ndarray,
        k_values: Tuple[int, ...] = (1, 5, 10),
    ) -> Dict[str, float]:
        """Text-to-cell and cell-to-text retrieval accuracy.

        Parameters
        ----------
        text_proj : (N, D) text projections
        cell_proj : (N, D) cell projections (matched pairs)
        k_values : tuple of int
            Recall@K values to compute.

        Returns
        -------
        metrics : dict with R@K for both directions
        """
        # Cosine similarity
        sim = text_proj @ cell_proj.T  # (N, N)
        N = sim.shape[0]

        metrics = {}
        for k in k_values:
            # Text → Cell
            t2c_topk = np.argsort(-sim, axis=1)[:, :k]
            t2c_recall = np.mean([i in t2c_topk[i] for i in range(N)])
            metrics[f"t2c_R@{k}"] = float(t2c_recall)

            # Cell → Text
            c2t_topk = np.argsort(-sim.T, axis=1)[:, :k]
            c2t_recall = np.mean([i in c2t_topk[i] for i in range(N)])
            metrics[f"c2t_R@{k}"] = float(c2t_recall)

        return metrics

    # ── Biological Validation Metrics (Reviewer Concern: bio plausibility) ──

    @staticmethod
    def marker_specificity_index(
        generated: np.ndarray,
        expected_high_dims: list,
        expected_low_dims: list = None,
    ) -> Dict[str, float]:
        """Marker specificity index: do generated cells up-regulate the
        expected marker gene dimensions while keeping others baseline?

        Parameters
        ----------
        generated : (M, D) generated cell embeddings
        expected_high_dims : list of int – embedding dimensions that should
            be elevated for this cell type
        expected_low_dims : list of int – embedding dimensions that should
            be low (optional)

        Returns
        -------
        metrics : dict with 'marker_specificity', 'marker_mean', 'background_mean'
        """
        gen_mean = generated.mean(axis=0)
        all_dims = set(range(generated.shape[1]))
        high_set = set(expected_high_dims)
        bg_dims = list(all_dims - high_set)

        marker_mean = gen_mean[expected_high_dims].mean()
        bg_mean = gen_mean[bg_dims].mean()
        bg_std = gen_mean[bg_dims].std() + 1e-8

        specificity = (marker_mean - bg_mean) / bg_std

        result = {
            "marker_specificity": float(specificity),
            "marker_mean": float(marker_mean),
            "background_mean": float(bg_mean),
        }
        if expected_low_dims is not None:
            low_mean = gen_mean[expected_low_dims].mean()
            result["low_dim_mean"] = float(low_mean)

        return result

    @staticmethod
    def inter_type_separation(
        embeddings_dict: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Measure how well-separated different cell types are in embedding space.

        Parameters
        ----------
        embeddings_dict : dict of cell_type → (M, D) embeddings

        Returns
        -------
        metrics : dict with 'mean_inter_cosine', 'min_inter_cosine',
                  'mean_intra_variance', silhouette-like score
        """
        names = list(embeddings_dict.keys())
        means = {n: embeddings_dict[n].mean(axis=0) for n in names}

        # Inter-type cosine distances
        inter_sims = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                v1, v2 = means[names[i]], means[names[j]]
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                inter_sims.append(sim)

        # Intra-type variances
        intra_vars = {n: np.var(embeddings_dict[n], axis=0).mean() for n in names}

        # Pseudo-silhouette: (inter_distance - intra_distance) / max
        mean_inter = float(np.mean(inter_sims))
        mean_intra = float(np.mean(list(intra_vars.values())))

        return {
            "mean_inter_cosine_sim": mean_inter,
            "min_inter_cosine_sim": float(np.min(inter_sims)) if inter_sims else 0.0,
            "max_inter_cosine_sim": float(np.max(inter_sims)) if inter_sims else 0.0,
            "mean_intra_variance": mean_intra,
            "separation_score": float(1.0 - mean_inter),  # higher = better separated
        }

    @staticmethod
    def diversity_score(generated: np.ndarray, k: int = 5) -> Dict[str, float]:
        """Measure generation diversity using average pairwise distance.

        Detects mode collapse: all generated samples landing in one point.

        Parameters
        ----------
        generated : (M, D) generated cell embeddings
        k : int – number of nearest neighbors for local diversity

        Returns
        -------
        metrics : dict with 'mean_pairwise_dist', 'knn_mean_dist', 'variance_ratio'
        """
        n = min(500, len(generated))  # cap for speed
        sub = generated[:n]

        # Global diversity
        norms = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8
        sub_normed = sub / norms
        sim_matrix = sub_normed @ sub_normed.T
        # Exclude diagonal
        np.fill_diagonal(sim_matrix, 0)
        mean_sim = sim_matrix.sum() / (n * (n - 1))

        # k-NN local diversity
        nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine")
        nn.fit(sub)
        dists, _ = nn.kneighbors(sub)
        knn_dist = dists[:, 1:].mean()

        # Variance ratio (variance of generated / ideal gaussian variance)
        gen_var = np.var(sub, axis=0).mean()

        return {
            "mean_pairwise_cosine_sim": float(mean_sim),
            "knn_mean_cosine_dist": float(knn_dist),
            "mean_variance": float(gen_var),
            "diversity_index": float(1.0 - mean_sim),  # higher = more diverse
        }

    @staticmethod
    def conditioning_fidelity(
        generated: np.ndarray,
        condition: np.ndarray,
        real: np.ndarray = None,
    ) -> Dict[str, float]:
        """Measure how well generated cells respond to the conditioning signal.

        Parameters
        ----------
        generated : (M, D) generated cell embeddings
        condition : (1, D_c) or (D_c,) condition vector
        real : (N, D) optional real embeddings for reference

        Returns
        -------
        metrics : dict with cosine similarities
        """
        cond = condition.flatten()
        gen_mean = generated.mean(axis=0)

        # If dims don't match, project to common space via truncation
        d = min(len(cond), len(gen_mean))
        cond_trunc = cond[:d]
        gen_trunc = gen_mean[:d]

        cos_sim = np.dot(cond_trunc, gen_trunc) / (
            np.linalg.norm(cond_trunc) * np.linalg.norm(gen_trunc) + 1e-8
        )

        result = {
            "cond_gen_cosine_sim": float(cos_sim),
        }

        if real is not None:
            real_mean = real.mean(axis=0)[:d]
            real_sim = np.dot(cond_trunc, real_mean) / (
                np.linalg.norm(cond_trunc) * np.linalg.norm(real_mean) + 1e-8
            )
            result["cond_real_cosine_sim"] = float(real_sim)
            result["fidelity_gap"] = float(cos_sim - real_sim)

        return result

    @classmethod
    def full_evaluation(
        cls,
        real: np.ndarray,
        generated: np.ndarray,
    ) -> Dict[str, float]:
        """Run all embedding-space evaluation metrics.

        Parameters
        ----------
        real : (N, D) real cell embeddings
        generated : (M, D) generated cell embeddings

        Returns
        -------
        metrics : dict
        """
        metrics = {}

        # Fréchet Distance
        metrics["frechet_distance"] = cls.frechet_distance(real, generated)

        # MMD
        metrics["mmd_rbf"] = cls.mmd(real, generated, kernel="rbf")

        # Coverage & Density
        cd = cls.coverage_and_density(real, generated)
        metrics.update(cd)

        # Per-dimension KL
        kl = cls.kl_per_dimension(real, generated)
        metrics.update(kl)

        # Diversity (mode collapse detection)
        div = cls.diversity_score(generated)
        metrics.update(div)

        return metrics
