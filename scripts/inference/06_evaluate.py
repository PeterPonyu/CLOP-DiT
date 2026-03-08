#!/usr/bin/env python3
# 06_evaluate.py — Comprehensive evaluation for CLOP-DiT v0.3
"""
Post-training evaluation: computes all quantitative metrics and generates
publication-quality visualizations.

v0.3 additions:
  - Gene expression decoding evaluation via scGPT generate()
  - Marker gene correlation analysis
  - Real vs generated gene expression comparison

Metrics computed:
  - Fréchet Distance (FD)
  - Maximum Mean Discrepancy (MMD)
  - Coverage & Density
  - Per-dimension KL divergence
  - CLOP retrieval R@K
  - Cosine similarity distributions
  - Gene expression correlation (v0.3)

Visualizations generated:
  - CLOP alignment space (UMAP/tSNE)
  - Real vs generated embedding comparison
  - Training curves (CLOP + DiT)
  - Per-dimension distribution comparison
  - Gene expression heatmap (v0.3)

Usage:
    python scripts/06_evaluate.py \
        --cache_dir data/cached_latents_v5.2 \
        --clop_checkpoint models/checkpoints/clop_best.pth \
        --dit_checkpoint models/checkpoints/dit_best.pth \
        --output_dir figures \
        --decode_expression --reference_h5ad data/processed_h5ad/GSE123902_LungAdreHmCancer_processed.h5ad
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.run_metrics import run_embedding_metrics
from src.evaluation.visualizer import EmbeddingVisualizer
from src.architecture.decoder import ScGPTDecoder
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def generate_visualizations(clop_results, gen_results, history_dir, output_dir,
                            expr_results=None):
    """Generate all visualizations."""
    logger.info("\n" + "="*60)
    logger.info("Generating Visualizations")
    logger.info("="*60)

    viz = EmbeddingVisualizer(save_dir=output_dir, dpi=300)

    # 1. CLOP alignment space
    if clop_results is not None:
        logger.info("  Plotting CLOP alignment space...")
        # Subsample for UMAP speed
        n = min(2000, len(clop_results["cell_proj"]))
        idx = np.random.choice(len(clop_results["cell_proj"]), n, replace=False)
        viz.plot_alignment_space(
            text_proj=clop_results["text_proj"][idx],
            cell_proj=clop_results["cell_proj"][idx],
            labels=[str(clop_results["sample_ids"][i]) for i in idx],
            filename="clop_alignment_umap.png",
            method="umap",
        )

    # 2. Generation comparison
    if gen_results is not None:
        logger.info("  Plotting generation comparison...")
        viz.plot_generation_comparison(
            real=gen_results["real"],
            generated=gen_results["generated"],
            filename="generation_comparison_umap.png",
            method="umap",
        )

        # 3. Per-dimension distributions
        logger.info("  Plotting dimension distributions...")
        viz.plot_dimension_distributions(
            real=gen_results["real"],
            generated=gen_results["generated"],
            filename="dim_distributions.png",
        )

    # 4. Training curves
    history_path = Path(history_dir)
    for hist_file, title, fname in [
        ("clop_history.json", "CLOP Training", "clop_training_curves.png"),
        ("dit_history.json", "DiT Training", "dit_training_curves.png"),
    ]:
        hist_path = history_path / hist_file
        if hist_path.exists():
            logger.info(f"  Plotting {title} curves...")
            with open(hist_path) as f:
                history = json.load(f)
            viz.plot_training_curves(history, filename=fname, title=title)

    # 5. Gene expression heatmap (v0.3)
    if expr_results is not None:
        logger.info("  Plotting gene expression comparison...")
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Real expression (subsample for vis)
            real_expr = expr_results.get("real_expression")
            gen_expr = expr_results.get("gen_expression")
            gene_names = expr_results.get("gene_names", [])

            if real_expr is not None and gen_expr is not None:
                n_show = min(50, real_expr.shape[0])
                g_show = min(50, real_expr.shape[1])

                # Top variable genes
                var = np.var(real_expr, axis=0)
                top_genes = np.argsort(var)[-g_show:]

                im0 = axes[0].imshow(
                    real_expr[:n_show, top_genes], aspect='auto', cmap='viridis'
                )
                axes[0].set_title("Real Expression (top HVGs)")
                axes[0].set_xlabel("Genes")
                axes[0].set_ylabel("Cells")
                plt.colorbar(im0, ax=axes[0], shrink=0.8)

                im1 = axes[1].imshow(
                    gen_expr[:n_show, top_genes], aspect='auto', cmap='viridis'
                )
                axes[1].set_title("Generated Expression (scGPT decoded)")
                axes[1].set_xlabel("Genes")
                axes[1].set_ylabel("Cells")
                plt.colorbar(im1, ax=axes[1], shrink=0.8)

                plt.tight_layout()
                plt.savefig(Path(output_dir) / "expression_comparison.png", dpi=300, bbox_inches="tight")
                plt.close()

                # Per-gene correlation scatter
                fig2, ax2 = plt.subplots(figsize=(8, 8))
                real_mean = real_expr.mean(axis=0)
                gen_mean = gen_expr.mean(axis=0)
                ax2.scatter(real_mean, gen_mean, alpha=0.3, s=5)
                ax2.set_xlabel("Real Mean Expression")
                ax2.set_ylabel("Generated Mean Expression")
                ax2.set_title("Per-Gene Mean Expression Correlation")

                # Add correlation
                from scipy import stats
                r, p = stats.pearsonr(real_mean, gen_mean)
                ax2.text(0.05, 0.95, f"Pearson r = {r:.4f}\np = {p:.2e}",
                         transform=ax2.transAxes, fontsize=12, va='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                # Diagonal reference
                lims = [min(real_mean.min(), gen_mean.min()),
                        max(real_mean.max(), gen_mean.max())]
                ax2.plot(lims, lims, 'r--', alpha=0.5)

                plt.tight_layout()
                plt.savefig(Path(output_dir) / "gene_correlation.png", dpi=300, bbox_inches="tight")
                plt.close()

                logger.info(f"  Gene expression figures saved")
        except Exception as e:
            logger.warning(f"  Failed to plot gene expression: {e}")

    logger.info(f"  All figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="CLOP-DiT v0.3 Evaluation")
    parser.add_argument("--cache_dir", type=str, default="data/cached_latents_v5.2")
    parser.add_argument("--clop_checkpoint", type=str, default="models/checkpoints/clop_best.pth")
    parser.add_argument("--dit_checkpoint", type=str, default="models/checkpoints/dit_best.pth")
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_clop", action="store_true")
    parser.add_argument("--skip_gen", action="store_true")
    parser.add_argument("--skip_viz", action="store_true")
    # v0.3 additions
    parser.add_argument("--decode_expression", action="store_true",
                        help="Evaluate gene expression decoding via scGPT generate()")
    parser.add_argument("--scgpt_model_dir", type=str, default="models/scgpt_pancancer",
                        help="scGPT model directory for decoding")
    parser.add_argument("--reference_h5ad", type=str, default=None,
                        help="Reference h5ad for scGPT gene vocabulary")
    args = parser.parse_args()

    setup_logging()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # CLOP + generation evaluation via shared API
    result = run_embedding_metrics(
        cache_dir=args.cache_dir,
        clop_checkpoint=args.clop_checkpoint,
        dit_checkpoint=args.dit_checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        device=args.device,
        skip_clop=args.skip_clop,
        skip_gen=args.skip_gen,
    )
    all_metrics = {"clop": result["clop"], "generation": result["generation"]}
    all_metrics = {k: v for k, v in all_metrics.items() if v is not None}
    clop_results = result["clop_results"]
    gen_results = result["gen_results"]

    # v0.3: Gene expression decoding evaluation
    expr_results = None
    if args.decode_expression and gen_results is not None:
        logger.info("\n" + "="*60)
        logger.info("Gene Expression Decoding Evaluation (v0.3)")
        logger.info("="*60)

        try:
            import scanpy as sc

            scgpt_decoder = ScGPTDecoder(
                model_dir=args.scgpt_model_dir,
                device=torch.device(args.device),
            )

            # Load reference adata for gene vocabulary
            if args.reference_h5ad:
                ref_adata = sc.read_h5ad(args.reference_h5ad)
                logger.info(f"Reference adata: {ref_adata.shape}")
                # Encode reference to set up gene vocabulary
                scgpt_decoder.encode(ref_adata)

            # Decode generated embeddings
            gen_decoded = scgpt_decoder.decode(gen_results["generated"])
            logger.info(f"  Generated expression shape: {gen_decoded['expression'].shape}")

            # Decode real embeddings for comparison
            real_decoded = scgpt_decoder.decode(gen_results["real"])
            logger.info(f"  Real expression shape: {real_decoded['expression'].shape}")

            # Compute gene-level correlation
            from scipy import stats
            real_mean = real_decoded["expression"].mean(axis=0)
            gen_mean = gen_decoded["expression"].mean(axis=0)
            pearson_r, pearson_p = stats.pearsonr(real_mean, gen_mean)
            spearman_r, spearman_p = stats.spearmanr(real_mean, gen_mean)

            logger.info(f"  Per-gene mean expression Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})")
            logger.info(f"  Per-gene mean expression Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")

            # Per-cell correlation
            cell_corrs = []
            for j in range(min(100, len(real_decoded["expression"]))):
                r, _ = stats.pearsonr(real_decoded["expression"][j], gen_decoded["expression"][j])
                if np.isfinite(r):
                    cell_corrs.append(r)
            mean_cell_corr = np.mean(cell_corrs) if cell_corrs else 0.0

            logger.info(f"  Per-cell expression Pearson r: {mean_cell_corr:.4f}")

            expr_metrics = {
                "gene_pearson_r": float(pearson_r),
                "gene_spearman_r": float(spearman_r),
                "cell_pearson_r_mean": float(mean_cell_corr),
                "num_genes": int(gen_decoded["expression"].shape[1]),
            }
            all_metrics["expression"] = expr_metrics

            expr_results = {
                "real_expression": real_decoded["expression"],
                "gen_expression": gen_decoded["expression"],
                "gene_names": gen_decoded["gene_names"],
                "metrics": expr_metrics,
            }

        except Exception as e:
            logger.error(f"Gene expression evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    # Visualizations
    if not args.skip_viz:
        generate_visualizations(
            clop_results, gen_results,
            history_dir="models/checkpoints",
            output_dir=args.output_dir,
            expr_results=expr_results,
        )

    # Save all metrics
    metrics_path = Path(args.output_dir) / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nAll metrics saved to {metrics_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"CLOP-DiT v0.3 Evaluation Summary")
    print(f"{'='*70}")
    if "clop" in all_metrics:
        print(f"\n  CLOP Alignment:")
        for k, v in all_metrics["clop"]["retrieval"].items():
            print(f"    {k}: {v:.4f}")
        print(f"    Cosine sim: {all_metrics['clop']['cosine_sim_mean']:.4f}")
    if "generation" in all_metrics:
        print(f"\n  Generation Quality:")
        for k, v in all_metrics["generation"].items():
            print(f"    {k}: {v:.6f}")
    if "expression" in all_metrics:
        print(f"\n  Gene Expression Decoding (v0.3):")
        for k, v in all_metrics["expression"].items():
            print(f"    {k}: {v}")
    print(f"\n  Figures: {args.output_dir}/")
    print(f"  Metrics: {metrics_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
