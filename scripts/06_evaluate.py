#!/usr/bin/env python3
# 06_evaluate.py — Comprehensive evaluation for CLOP-DiT v0.2
"""
Post-training evaluation: computes all quantitative metrics and generates
publication-quality visualizations.

Metrics computed:
  - Fréchet Distance (FD)
  - Maximum Mean Discrepancy (MMD)
  - Coverage & Density
  - Per-dimension KL divergence
  - CLOP retrieval R@K
  - Cosine similarity distributions

Visualizations generated:
  - CLOP alignment space (UMAP/tSNE)
  - Real vs generated embedding comparison
  - Training curves (CLOP + DiT)
  - Per-dimension distribution comparison

Usage:
    python scripts/06_evaluate.py \
        --cache_dir data/cached_latents \
        --clop_checkpoint models/checkpoints/clop_best.pth \
        --dit_checkpoint models/checkpoints/dit_best.pth \
        --output_dir figures
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import GenerationMetrics
from src.evaluation.visualizer import EmbeddingVisualizer
from src.architecture.clop import CLOPAligner
from src.architecture.dit import DiT1D
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def load_clop(checkpoint_path, device="cuda"):
    """Load trained CLOP model."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    model = CLOPAligner(proj_dim=config.get("proj_dim", 256))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, config


def load_dit(checkpoint_path, cond_dim=256, device="cuda"):
    """Load trained DiT model."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    model = DiT1D(
        latent_dim=config.get("latent_dim", 512),
        hidden_dim=config.get("hidden_dim", 384),
        cond_dim=cond_dim,
        num_tokens=config.get("num_tokens", 16),
    )
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        logger.info("  Using EMA weights for DiT")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, config


@torch.no_grad()
def evaluate_clop(clop_model, cache_dir, device="cuda"):
    """Evaluate CLOP alignment quality."""
    logger.info("\n" + "="*60)
    logger.info("CLOP Alignment Evaluation")
    logger.info("="*60)

    cell_emb = np.load(Path(cache_dir) / "cell_embeddings.npy")
    text_emb = np.load(Path(cache_dir) / "text_embeddings.npy")
    sample_ids = np.load(Path(cache_dir) / "sample_ids.npy")

    # Project through CLOP
    cell_proj_list = []
    text_proj_list = []
    for i in range(0, len(cell_emb), 512):
        c_batch = torch.from_numpy(cell_emb[i:i+512]).float().to(device)
        t_batch = torch.from_numpy(text_emb[i:i+512]).float().to(device)
        c_proj = clop_model.project_cell(c_batch).cpu().numpy()
        t_proj = clop_model.project_text(t_batch).cpu().numpy()
        cell_proj_list.append(c_proj)
        text_proj_list.append(t_proj)

    cell_proj = np.concatenate(cell_proj_list, axis=0)
    text_proj = np.concatenate(text_proj_list, axis=0)

    # Sample-level retrieval (one per dataset)
    unique_ids = np.unique(sample_ids)
    sample_cell_proj = np.array([cell_proj[sample_ids == sid].mean(axis=0) for sid in unique_ids])
    sample_text_proj = np.array([text_proj[sample_ids == sid].mean(axis=0) for sid in unique_ids])

    # Normalize for cosine similarity
    sample_cell_proj = sample_cell_proj / (np.linalg.norm(sample_cell_proj, axis=1, keepdims=True) + 1e-8)
    sample_text_proj = sample_text_proj / (np.linalg.norm(sample_text_proj, axis=1, keepdims=True) + 1e-8)

    retrieval_metrics = GenerationMetrics.clop_retrieval(
        sample_text_proj, sample_cell_proj, k_values=(1, 3, 5, 10)
    )

    logger.info(f"  Number of datasets: {len(unique_ids)}")
    for k, v in retrieval_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Cosine similarity stats
    cosine_sims = np.sum(cell_proj * text_proj, axis=1) / (
        np.linalg.norm(cell_proj, axis=1) * np.linalg.norm(text_proj, axis=1) + 1e-8
    )
    logger.info(f"  Mean cell-text cosine sim: {cosine_sims.mean():.4f} ± {cosine_sims.std():.4f}")

    return {
        "retrieval": retrieval_metrics,
        "cosine_sim_mean": float(cosine_sims.mean()),
        "cosine_sim_std": float(cosine_sims.std()),
        "cell_proj": cell_proj,
        "text_proj": text_proj,
        "sample_ids": sample_ids,
    }


@torch.no_grad()
def evaluate_generation(dit_model, clop_model, cache_dir, num_samples=500,
                        num_steps=4, cfg_scale=3.0, device="cuda"):
    """Evaluate DiT generation quality."""
    logger.info("\n" + "="*60)
    logger.info("DiT Generation Evaluation")
    logger.info("="*60)

    cell_emb = np.load(Path(cache_dir) / "cell_embeddings.npy")
    projected_path = Path(cache_dir) / "projected_text.npy"
    if not projected_path.exists():
        logger.error("projected_text.npy not found. Run CLOP training first.")
        return None

    projected_text = np.load(projected_path)
    sample_ids = np.load(Path(cache_dir) / "sample_ids.npy")

    # Sample conditions from different datasets
    unique_ids = np.unique(sample_ids)
    conditions_list = []
    real_list = []

    for sid in unique_ids:
        mask = sample_ids == sid
        n_per = min(num_samples // len(unique_ids) + 1, mask.sum())
        idx = np.where(mask)[0][:n_per]
        conditions_list.append(projected_text[idx])
        real_list.append(cell_emb[idx])

    conditions = np.concatenate(conditions_list, axis=0)[:num_samples]
    real = np.concatenate(real_list, axis=0)[:num_samples]

    # Generate
    cond_tensor = torch.from_numpy(conditions).float().to(device)
    generated_list = []
    batch_size = 256
    for i in range(0, len(cond_tensor), batch_size):
        batch = cond_tensor[i:i+batch_size]
        gen = dit_model.sample(batch, num_steps=num_steps, cfg_scale=cfg_scale)
        generated_list.append(gen.cpu().numpy())

    generated = np.concatenate(generated_list, axis=0)

    logger.info(f"  Real: {real.shape}, Generated: {generated.shape}")

    # Full evaluation
    metrics = GenerationMetrics.full_evaluation(real, generated)

    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.6f}")

    # Additional generation stats
    real_norms = np.linalg.norm(real, axis=1)
    gen_norms = np.linalg.norm(generated, axis=1)
    cosine_sims = np.sum(generated * real, axis=1) / (gen_norms * real_norms + 1e-8)

    metrics["real_norm_mean"] = float(real_norms.mean())
    metrics["gen_norm_mean"] = float(gen_norms.mean())
    metrics["paired_cosine_mean"] = float(cosine_sims.mean())

    logger.info(f"  Real norm: {real_norms.mean():.4f} ± {real_norms.std():.4f}")
    logger.info(f"  Gen norm:  {gen_norms.mean():.4f} ± {gen_norms.std():.4f}")
    logger.info(f"  Paired cosine sim: {cosine_sims.mean():.4f}")

    return {
        "metrics": metrics,
        "real": real,
        "generated": generated,
    }


def generate_visualizations(clop_results, gen_results, history_dir, output_dir):
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

    logger.info(f"  All figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="CLOP-DiT v0.2 Evaluation")
    parser.add_argument("--cache_dir", type=str, default="data/cached_latents")
    parser.add_argument("--clop_checkpoint", type=str, default="models/checkpoints/clop_best.pth")
    parser.add_argument("--dit_checkpoint", type=str, default="models/checkpoints/dit_best.pth")
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_clop", action="store_true")
    parser.add_argument("--skip_gen", action="store_true")
    parser.add_argument("--skip_viz", action="store_true")
    args = parser.parse_args()

    setup_logging()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    # CLOP evaluation
    clop_results = None
    if not args.skip_clop and Path(args.clop_checkpoint).exists():
        clop_model, clop_config = load_clop(args.clop_checkpoint, args.device)
        clop_results = evaluate_clop(clop_model, args.cache_dir, args.device)
        all_metrics["clop"] = {
            "retrieval": clop_results["retrieval"],
            "cosine_sim_mean": clop_results["cosine_sim_mean"],
            "cosine_sim_std": clop_results["cosine_sim_std"],
        }

    # Generation evaluation
    gen_results = None
    if not args.skip_gen and Path(args.dit_checkpoint).exists():
        clop_config_dim = 256
        if clop_results is None and Path(args.clop_checkpoint).exists():
            _, cc = load_clop(args.clop_checkpoint, args.device)
            clop_config_dim = cc.get("proj_dim", 256)
        dit_model, _ = load_dit(args.dit_checkpoint, cond_dim=clop_config_dim, device=args.device)
        clop_model_for_gen, _ = load_clop(args.clop_checkpoint, args.device) if clop_results is None else (None, None)

        gen_results = evaluate_generation(
            dit_model, clop_model_for_gen or clop_model,
            args.cache_dir, num_samples=args.num_samples,
            num_steps=args.num_steps, cfg_scale=args.cfg_scale,
            device=args.device,
        )
        if gen_results:
            all_metrics["generation"] = gen_results["metrics"]

    # Visualizations
    if not args.skip_viz:
        generate_visualizations(
            clop_results, gen_results,
            history_dir="models/checkpoints",
            output_dir=args.output_dir,
        )

    # Save all metrics
    metrics_path = Path(args.output_dir) / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nAll metrics saved to {metrics_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"CLOP-DiT v0.2 Evaluation Summary")
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
    print(f"\n  Figures: {args.output_dir}/")
    print(f"  Metrics: {metrics_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
