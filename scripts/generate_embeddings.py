#!/usr/bin/env python3
"""generate_embeddings.py — Generate cell embeddings from trained DiT for evaluation.

Uses cached projected_text conditions (no BiomedBERT needed).
Generates cells for every type using the best DiT checkpoint.

Supports two conditioning modes:
  centroid  — one mean condition per type (original approach, less diverse)
  per_cell  — sample real per-cell conditions (recommended for diversity)

Usage:
    python scripts/generate_embeddings.py                                  # per_cell + CFG=1.5
    python scripts/generate_embeddings.py --condition-mode centroid         # original centroid mode
    python scripts/generate_embeddings.py --cfg-scale 1.0 --num-per-type 200
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.architecture.dit import DiT1D
from src.utils.helpers import seed_everything, get_device

logger = logging.getLogger(__name__)


def load_dit(checkpoint_path: str, device: torch.device) -> DiT1D:
    """Load trained DiT from checkpoint, resolving cond_dim from saved config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    # Resolve cond_dim: prefer checkpoint config, then probe null_cond shape, fallback 512
    cond_dim = cfg.get("cond_dim", None)
    if cond_dim is None:
        sd = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", {}))
        null_cond = sd.get("c_embedder.null_cond")
        cond_dim = null_cond.shape[-1] if null_cond is not None else 512
    logger.info(f"Resolved cond_dim={cond_dim} from checkpoint")

    model = DiT1D(
        latent_dim=cfg.get("latent_dim", 512),
        hidden_dim=cfg.get("hidden_dim", 512),
        cond_dim=cond_dim,
        num_tokens=cfg.get("num_tokens", 16),
        num_blocks=8,
        num_heads=8,
        cond_drop_prob=0.0,  # No dropout at inference
    )

    # Prefer EMA weights
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        logger.info("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded model weights (no EMA found)")

    model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})
    logger.info(f"DiT checkpoint epoch={epoch}, metrics={metrics}")
    return model


def generate_all_types(
    model: DiT1D,
    projected_text: np.ndarray,
    group_ids: np.ndarray,
    num_per_type: int = 100,
    num_steps: int = 20,
    cfg_scale: float = 3.0,
    normalize: bool = True,
    condition_mode: str = "per_cell",
    device: torch.device = None,
) -> tuple:
    """Generate cells for every type.

    Parameters
    ----------
    model : DiT1D
    projected_text : (N, cond_dim) per-cell text conditions
    group_ids : (N,) type IDs
    num_per_type : cells to generate per type
    num_steps : ODE integration steps
    cfg_scale : classifier-free guidance strength
    normalize : L2-normalize generated embeddings (recommended)
    condition_mode : str
        'centroid' — one mean condition per type (original, less diverse)
        'per_cell' — sample real per-cell conditions for each generated cell
                     (recommended for realistic within-type diversity)
    device : torch device

    Returns
    -------
    generated : (num_types * num_per_type, cond_dim) generated embeddings
    gen_labels : (num_types * num_per_type,) type labels
    type_ids : sorted unique type IDs
    """
    unique_types = np.sort(np.unique(group_ids))
    logger.info(f"Generating {num_per_type} cells × {len(unique_types)} types "
                f"= {num_per_type * len(unique_types)} total  "
                f"[mode={condition_mode}, cfg={cfg_scale}, steps={num_steps}]")

    rng = np.random.default_rng(42)
    all_gen = []
    all_labels = []

    for t_id in unique_types:
        mask = group_ids == t_id
        type_conds = projected_text[mask]  # (n_type, cond_dim)

        if condition_mode == "centroid":
            # Original approach: all cells share one centroid condition
            proto = type_conds.mean(axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            cond = torch.from_numpy(proto).float().unsqueeze(0).repeat(num_per_type, 1).to(device)
        elif condition_mode == "per_cell":
            # Diverse approach: each cell gets a different real condition
            idx = rng.choice(len(type_conds), size=num_per_type, replace=True)
            sampled = type_conds[idx]
            # L2-normalize each condition individually
            norms = np.linalg.norm(sampled, axis=1, keepdims=True) + 1e-8
            sampled = sampled / norms
            cond = torch.from_numpy(sampled).float().to(device)
        else:
            raise ValueError(f"Unknown condition_mode: {condition_mode}")

        gen = model.sample(cond, num_steps=num_steps, cfg_scale=cfg_scale)

        if normalize:
            gen = torch.nn.functional.normalize(gen, dim=-1)

        all_gen.append(gen.cpu().numpy())
        all_labels.extend([t_id] * num_per_type)

    generated = np.concatenate(all_gen, axis=0)
    gen_labels = np.array(all_labels)

    norms = np.linalg.norm(generated, axis=1)
    logger.info(f"Generated: shape={generated.shape}, "
                f"norm_mean={norms.mean():.4f}, norm_std={norms.std():.6f}")

    return generated, gen_labels, unique_types


def compute_per_type_metrics(
    real_cells: np.ndarray,
    real_labels: np.ndarray,
    gen_cells: np.ndarray,
    gen_labels: np.ndarray,
    type_ids: np.ndarray,
    type_names: dict = None,
) -> dict:
    """Compute per-type generation quality metrics.

    Returns dict with overall + per-type Frechet distances, cosine sims, etc.
    """
    from src.evaluation.metrics import GenerationMetrics

    results = {"overall": {}, "per_type": {}}

    # Subsample real cells to avoid OOM on pairwise distance matrices
    rng = np.random.default_rng(42)
    max_overall = min(len(real_cells), len(gen_cells), 5000)
    real_sub_idx = rng.choice(len(real_cells), max_overall, replace=False)
    gen_sub_idx = rng.choice(len(gen_cells), max_overall, replace=False) if len(gen_cells) > max_overall else np.arange(len(gen_cells))
    
    # Overall
    overall = GenerationMetrics.full_evaluation(real_cells[real_sub_idx], gen_cells[gen_sub_idx])
    overall["real_norm_mean"] = float(np.linalg.norm(real_cells, axis=1).mean())
    overall["gen_norm_mean"] = float(np.linalg.norm(gen_cells, axis=1).mean())
    results["overall"] = overall
    logger.info(f"Overall metrics: FD={overall.get('frechet_distance', 'N/A'):.4f}, "
                f"Coverage={overall.get('coverage', 'N/A'):.4f}, "
                f"MMD={overall.get('mmd_rbf', 'N/A'):.6f}")

    # Per-type
    for t_id in type_ids:
        real_mask = real_labels == t_id
        gen_mask = gen_labels == t_id
        real_t = real_cells[real_mask]
        gen_t = gen_cells[gen_mask]

        if len(real_t) < 5 or len(gen_t) < 5:
            continue

        name = type_names.get(int(t_id), f"Type_{t_id}") if type_names else f"Type_{t_id}"

        # Cosine similarity between generated and real centroids
        real_centroid = real_t.mean(axis=0)
        gen_centroid = gen_t.mean(axis=0)
        cos_sim = np.dot(real_centroid, gen_centroid) / (
            np.linalg.norm(real_centroid) * np.linalg.norm(gen_centroid) + 1e-8
        )

        # Per-type Frechet distance (if enough samples)
        try:
            type_metrics = GenerationMetrics.full_evaluation(real_t, gen_t)
            fd = type_metrics.get("frechet_distance", float("nan"))
        except Exception:
            fd = float("nan")

        results["per_type"][name] = {
            "type_id": int(t_id),
            "n_real": int(len(real_t)),
            "n_gen": int(len(gen_t)),
            "centroid_cosine": float(cos_sim),
            "frechet_distance": float(fd),
        }

    # Summary stats
    centroids = [v["centroid_cosine"] for v in results["per_type"].values()]
    if centroids:
        results["summary"] = {
            "mean_centroid_cosine": float(np.mean(centroids)),
            "min_centroid_cosine": float(np.min(centroids)),
            "max_centroid_cosine": float(np.max(centroids)),
            "std_centroid_cosine": float(np.std(centroids)),
            "num_types_evaluated": len(centroids),
        }
        logger.info(f"Per-type centroid cosine: mean={np.mean(centroids):.4f}, "
                    f"min={np.min(centroids):.4f}, max={np.max(centroids):.4f}")

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dit-checkpoint", default="models/checkpoints/dit_best.pth")
    parser.add_argument("--cache-dir", default="data/cached_latents_v5.2")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--num-per-type", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                        help="CFG scale (default 1.5; lower = more diverse)")
    parser.add_argument("--condition-mode", choices=["centroid", "per_cell"],
                        default="per_cell",
                        help="Condition mode: centroid (one per type) or per_cell (diverse)")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()

    cache = Path(args.cache_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    projected_text = np.load(cache / "projected_text.npy")
    group_ids = np.load(cache / "text_group_ids_dedup.npy")
    real_cells = np.load(cache / "cell_embeddings_dedup_preprocessed.npy")

    # Load type names
    cap_path = cache / "text_captions_deduplicated.json"
    type_names = {}
    if cap_path.exists():
        with open(cap_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            name = v.split(" are ")[0] if " are " in v else v[:50]
            type_names[int(k)] = name

    # Load DiT
    model = load_dit(args.dit_checkpoint, device)

    # Generate
    generated, gen_labels, type_ids = generate_all_types(
        model=model,
        projected_text=projected_text,
        group_ids=group_ids,
        num_per_type=args.num_per_type,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        normalize=not args.no_normalize,
        condition_mode=args.condition_mode,
        device=device,
    )

    # Save generated embeddings
    np.save(out / "generated_embeddings.npy", generated)
    np.save(out / "generated_labels.npy", gen_labels)
    logger.info(f"Saved generated embeddings → {out / 'generated_embeddings.npy'}")

    # Compute metrics
    metrics = compute_per_type_metrics(
        real_cells=real_cells,
        real_labels=group_ids,
        gen_cells=generated,
        gen_labels=gen_labels,
        type_ids=type_ids,
        type_names=type_names,
    )

    with open(out / "generation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics → {out / 'generation_metrics.json'}")

    print(f"\n{'='*60}")
    print(f"Generation Summary")
    print(f"{'='*60}")
    print(f"Generated: {generated.shape[0]} cells ({args.num_per_type}/type × {len(type_ids)} types)")
    print(f"Norm: mean={np.linalg.norm(generated, axis=1).mean():.4f}")
    print(f"Overall FD: {metrics['overall'].get('frechet_distance', 'N/A')}")
    print(f"Overall Coverage: {metrics['overall'].get('coverage', 'N/A')}")
    if "summary" in metrics:
        s = metrics["summary"]
        print(f"Per-type centroid cosine: {s['mean_centroid_cosine']:.4f} "
              f"(min={s['min_centroid_cosine']:.4f}, max={s['max_centroid_cosine']:.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
