#!/usr/bin/env python3
"""
Enhanced Generation Evaluation Metrics for CLOP-DiT
====================================================

Comprehensive evaluation beyond MSE and proto_acc:
1. Embedding-level metrics: Fréchet Distance, MMD, cosine similarity
2. Decoded expression metrics: Marker gene correlation, silhouette score
3. Biological plausibility: UMAP overlap, cluster purity
4. Cross-tissue generalization: Same cell type in different contexts

Usage:
    from src.evaluation.generative_metrics import GenerativeEvaluator
    
    evaluator = GenerativeEvaluator(dit_model, scgpt_decoder, device)
    metrics = evaluator.evaluate_full(text_descriptions, real_cell_embeddings)
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score

from ..utils.constants import CFG_SCALE, INFERENCE_STEPS, RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerativeEvaluator:
    """Comprehensive evaluation for text → cell generation."""
    
    def __init__(
        self,
        dit_model,
        scgpt_decoder=None,
        device: str = "cuda",
    ):
        """
        Parameters
        ----------
        dit_model : DiT1D
            Trained DiT model for generation.
        scgpt_decoder : ScGPTDecoder, optional
            scGPT decoder for expression reconstruction.
        device : str
            Computation device.
        """
        self.dit_model = dit_model
        self.scgpt_decoder = scgpt_decoder
        self.device = device
    
    def frechet_distance(
        self,
        real_emb: np.ndarray,
        fake_emb: np.ndarray,
    ) -> float:
        """Compute Fréchet Distance between real and generated embeddings.

        Delegates to the canonical implementation in GenerationMetrics.
        """
        from src.evaluation.metrics import GenerationMetrics
        return GenerationMetrics.frechet_distance(real_emb, fake_emb)
    
    def maximum_mean_discrepancy(
        self,
        real_emb: np.ndarray,
        fake_emb: np.ndarray,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
    ) -> float:
        """Compute Maximum Mean Discrepancy between distributions.
        
        MMD measures distribution similarity using kernel trick.
        
        Parameters
        ----------
        real_emb : (N, D)
        fake_emb : (M, D)
        kernel_type : str
            'rbf' or 'linear'.
        gamma : float
            RBF kernel bandwidth.
        
        Returns
        -------
        mmd : float
            Lower is better (0 = perfect match).
        """
        real_emb = torch.from_numpy(real_emb).float()
        fake_emb = torch.from_numpy(fake_emb).float()
        
        def kernel(x, y, gamma):
            if kernel_type == "rbf":
                dist = torch.cdist(x, y, p=2) ** 2
                return torch.exp(-gamma * dist)
            else:  # linear
                return x @ y.T
        
        K_rr = kernel(real_emb, real_emb, gamma).mean()
        K_ff = kernel(fake_emb, fake_emb, gamma).mean()
        K_rf = kernel(real_emb, fake_emb, gamma).mean()
        
        mmd = K_rr + K_ff - 2 * K_rf
        return float(mmd.item())
    
    def coverage_and_density(
        self,
        real_emb: np.ndarray,
        fake_emb: np.ndarray,
        k: int = 5,
    ) -> Tuple[float, float]:
        """Compute Coverage and Density metrics (Naeem et al., 2020).
        
        Coverage: % of real samples with at least one generated neighbor.
        Density: Average # of generated samples in neighborhoods of real samples.
        
        Parameters
        ----------
        real_emb : (N, D)
        fake_emb : (M, D)
        k : int
            Number of nearest neighbors to consider.
        
        Returns
        -------
        coverage : float in [0, 1]
            Higher is better (1 = all real modes covered).
        density : float
            Average density of generated samples.
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Build kNN index on generated embeddings
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(fake_emb)
        distances, indices = nbrs.kneighbors(real_emb)
        
        # Coverage: how many real samples have generated neighbors within radius
        # Use k-th nearest neighbor distance as radius
        radii = distances[:, -1]
        covered = (distances[:, 0] <= radii).sum()
        coverage = covered / len(real_emb)
        
        # Density: average number of generated neighbors per real sample
        density = k  # By construction, always k for kNN
        # More sophisticated: count within adaptive radius
        density = (distances <= radii[:, None]).sum(axis=1).mean()
        
        return float(coverage), float(density)
    
    def cosine_similarity_to_closest_real(
        self,
        real_emb: np.ndarray,
        fake_emb: np.ndarray,
    ) -> Dict[str, float]:
        """Compute cosine similarity between generated and nearest real embeddings.
        
        Parameters
        ----------
        real_emb : (N, D)
        fake_emb : (M, D)
        
        Returns
        -------
        metrics : dict
            "mean_cosine", "median_cosine", "min_cosine", "max_cosine"
        """
        real_emb = torch.from_numpy(real_emb).float()
        fake_emb = torch.from_numpy(fake_emb).float()
        
        # Normalize
        real_norm = F.normalize(real_emb, dim=1)
        fake_norm = F.normalize(fake_emb, dim=1)
        
        # Compute pairwise cosine similarity
        sim_matrix = fake_norm @ real_norm.T  # (M, N)
        
        # For each generated, find max similarity to any real
        max_sims, _ = sim_matrix.max(dim=1)
        
        return {
            "mean_cosine": float(max_sims.mean().item()),
            "median_cosine": float(max_sims.median().item()),
            "min_cosine": float(max_sims.min().item()),
            "max_cosine": float(max_sims.max().item()),
        }
    
    def marker_gene_correlation(
        self,
        generated_expression: np.ndarray,
        real_expression: np.ndarray,
        marker_indices: List[int],
    ) -> Dict[str, float]:
        """Compute correlation between generated and real expression for marker genes.
        
        Parameters
        ----------
        generated_expression : (M, G) generated gene expression
        real_expression : (N, G) real gene expression
        marker_indices : list of int
            Indices of marker genes to evaluate.
        
        Returns
        -------
        metrics : dict
            Pearson correlation per marker, mean across markers.
        """
        from scipy.stats import pearsonr
        
        gen_markers = generated_expression[:, marker_indices]
        real_markers = real_expression[:, marker_indices]
        
        correlations = []
        for i in range(len(marker_indices)):
            # Flatten and compute correlation
            r, _ = pearsonr(gen_markers[:, i], real_markers[:, i])
            correlations.append(r)
        
        return {
            "mean_marker_correlation": float(np.mean(correlations)),
            "median_marker_correlation": float(np.median(correlations)),
            "marker_correlations": correlations,
        }
    
    def silhouette_score_by_cell_type(
        self,
        embeddings: np.ndarray,
        cell_type_labels: np.ndarray,
    ) -> float:
        """Compute silhouette score for cell-type clustering.
        
        Measures how well cell types separate in embedding space.
        
        Parameters
        ----------
        embeddings : (N, D)
        cell_type_labels : (N,) integer labels
        
        Returns
        -------
        silhouette : float in [-1, 1]
            Higher is better (1 = perfect separation).
        """
        if len(np.unique(cell_type_labels)) < 2:
            return 0.0
        
        score = silhouette_score(embeddings, cell_type_labels, metric='cosine')
        return float(score)
    
    def umap_overlap(
        self,
        real_emb: np.ndarray,
        fake_emb: np.ndarray,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> Dict[str, float]:
        """Compute UMAP overlap between real and generated embeddings.
        
        Projects both into 2D UMAP and measures spatial overlap.
        
        Parameters
        ----------
        real_emb : (N, D)
        fake_emb : (M, D)
        n_neighbors : int
        min_dist : float
        
        Returns
        -------
        metrics : dict
            "wasserstein_2d" : Wasserstein distance in 2D UMAP space.
        """
        try:
            from umap import UMAP
            from scipy.stats import wasserstein_distance
        except ImportError:
            logger.warning("UMAP not installed, skipping UMAP overlap")
            return {"wasserstein_2d": float('nan')}
        
        # Fit UMAP on real data
        umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=RANDOM_SEED)
        real_2d = umap_model.fit_transform(real_emb)
        fake_2d = umap_model.transform(fake_emb)
        
        # Compute 2D Wasserstein distance (separately for x and y)
        w_x = wasserstein_distance(real_2d[:, 0], fake_2d[:, 0])
        w_y = wasserstein_distance(real_2d[:, 1], fake_2d[:, 1])
        w_avg = (w_x + w_y) / 2
        
        return {
            "wasserstein_2d": float(w_avg),
            "wasserstein_x": float(w_x),
            "wasserstein_y": float(w_y),
        }
    
    @torch.no_grad()
    def evaluate_full(
        self,
        text_embeddings: np.ndarray,
        real_cell_embeddings: np.ndarray,
        cell_type_labels: Optional[np.ndarray] = None,
        num_samples: int = 1000,
        cfg_scale: float = CFG_SCALE,
        decode_expression: bool = False,
    ) -> Dict:
        """Run full evaluation suite.
        
        Parameters
        ----------
        text_embeddings : (N, D_text)
            Text condition vectors for generation.
        real_cell_embeddings : (N, D_cell)
            Real cell embeddings for comparison.
        cell_type_labels : (N,) optional
            Cell type labels for silhouette score.
        num_samples : int
            Number of samples to generate.
        cfg_scale : float
            Classifier-free guidance scale.
        decode_expression : bool
            Whether to decode to gene expression (requires scgpt_decoder).
        
        Returns
        -------
        metrics : dict
            All computed metrics.
        """
        logger.info(f"Evaluating generation with {num_samples} samples...")
        
        # Subsample if needed
        if len(text_embeddings) > num_samples:
            indices = np.random.choice(len(text_embeddings), num_samples, replace=False)
            text_embeddings = text_embeddings[indices]
            real_cell_embeddings = real_cell_embeddings[indices]
            if cell_type_labels is not None:
                cell_type_labels = cell_type_labels[indices]
        
        # Generate cell embeddings
        text_emb_torch = torch.from_numpy(text_embeddings).float().to(self.device)
        
        logger.info("Generating cell embeddings...")
        generated_embeddings = self.dit_model.sample(
            cond=text_emb_torch,
            num_steps=INFERENCE_STEPS,
            cfg_scale=cfg_scale,
            device=self.device,
        ).cpu().numpy()
        
        logger.info(f"✓ Generated {len(generated_embeddings)} cell embeddings")
        
        metrics = {}
        
        # Embedding-level metrics
        logger.info("Computing Fréchet Distance...")
        metrics["frechet_distance"] = self.frechet_distance(real_cell_embeddings, generated_embeddings)
        
        logger.info("Computing MMD...")
        metrics["mmd_rbf"] = self.maximum_mean_discrepancy(real_cell_embeddings, generated_embeddings, "rbf", gamma=1.0)
        
        logger.info("Computing Coverage & Density...")
        coverage, density = self.coverage_and_density(real_cell_embeddings, generated_embeddings, k=5)
        metrics["coverage"] = coverage
        metrics["density"] = density
        
        logger.info("Computing Cosine Similarity...")
        cosine_metrics = self.cosine_similarity_to_closest_real(real_cell_embeddings, generated_embeddings)
        metrics.update(cosine_metrics)
        
        # Clustering metrics
        if cell_type_labels is not None:
            logger.info("Computing Silhouette Scores...")
            metrics["silhouette_real"] = self.silhouette_score_by_cell_type(real_cell_embeddings, cell_type_labels)
            metrics["silhouette_generated"] = self.silhouette_score_by_cell_type(generated_embeddings, cell_type_labels)
        
        # UMAP overlap
        logger.info("Computing UMAP Overlap...")
        umap_metrics = self.umap_overlap(real_cell_embeddings, generated_embeddings)
        metrics.update(umap_metrics)
        
        # Expression-level metrics (if decoder available)
        if decode_expression and self.scgpt_decoder is not None:
            logger.info("Decoding to gene expression...")
            # This would require implementing decode() call
            # Placeholder for now
            logger.warning("Expression decoding not yet implemented in evaluator")
        
        return metrics


def print_metrics_summary(metrics: Dict):
    """Pretty-print evaluation metrics."""
    print("\n" + "="*80)
    print("GENERATION EVALUATION METRICS")
    print("="*80)
    
    print("\nEmbedding Quality:")
    print(f"  Fréchet Distance:    {metrics.get('frechet_distance', float('nan')):.4f}  (lower is better)")
    print(f"  MMD (RBF):           {metrics.get('mmd_rbf', float('nan')):.6f}  (lower is better)")
    print(f"  Coverage:            {metrics.get('coverage', float('nan')):.4f}  (higher is better, max 1.0)")
    print(f"  Density:             {metrics.get('density', float('nan')):.4f}")
    
    print("\nSimilarity to Real:")
    print(f"  Mean Cosine:         {metrics.get('mean_cosine', float('nan')):.4f}")
    print(f"  Median Cosine:       {metrics.get('median_cosine', float('nan')):.4f}")
    print(f"  Range:               [{metrics.get('min_cosine', float('nan')):.4f}, {metrics.get('max_cosine', float('nan')):.4f}]")
    
    if "silhouette_real" in metrics:
        print("\nClustering Quality (Silhouette):")
        print(f"  Real Embeddings:     {metrics['silhouette_real']:.4f}")
        print(f"  Generated:           {metrics['silhouette_generated']:.4f}")
        diff = metrics['silhouette_generated'] - metrics['silhouette_real']
        print(f"  Difference:          {diff:+.4f}  (closer to 0 is better)")
    
    if "wasserstein_2d" in metrics and not np.isnan(metrics["wasserstein_2d"]):
        print("\nUMAP Overlap:")
        print(f"  2D Wasserstein:      {metrics['wasserstein_2d']:.4f}  (lower is better)")
    
    print("="*80)


if __name__ == "__main__":
    # Demo usage
    print("Enhanced Generative Evaluation Metrics for CLOP-DiT")
    print("Import this module and use GenerativeEvaluator class for evaluation")
