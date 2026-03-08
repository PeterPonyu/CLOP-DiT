#!/usr/bin/env python3
"""generate_embeddings.py — Generate cell embeddings from trained DiT for evaluation.

Uses cached projected_text conditions (no BiomedBERT needed).
Generates cells for every type using the best DiT checkpoint.

Supports four conditioning modes:
  centroid        — one mean condition per type (original, least diverse)
  per_cell        — sample real per-cell conditions (no-op when captions are
                    deduplicated to 1 per type — all rows identical)
  condition_noise — centroid + Gaussian noise ε, L2-normalized
                    (recommended: simple, effective diversity lever)
  variant         — project caption variants through CLOP to get genuinely
                    different text conditions per type (highest diversity)

Usage:
    python scripts/inference/generate_embeddings.py                                         # condition_noise + CFG=1.5
    python scripts/inference/generate_embeddings.py --condition-mode variant                # real prompt diversity
    python scripts/inference/generate_embeddings.py --condition-mode centroid --cfg-scale 3 # original
    python scripts/inference/generate_embeddings.py --noise-scale 0.1                       # more noise
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.architecture.dit import DiT1D
from src.utils.paths import CACHE_DIR, RESULTS_DIR, CHECKPOINT_DIR
from src.architecture.clop import CLOPAligner
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


def load_clop(checkpoint_path: str, device: torch.device) -> CLOPAligner:
    """Load trained CLOP model for text projection."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = CLOPAligner(
        text_dim=cfg.get("text_dim", 1024),
        cell_dim=cfg.get("cell_dim", 512),
        proj_dim=cfg.get("proj_dim", 512),
        text_layers=cfg.get("text_layers", 3),
        cell_layers=cfg.get("cell_layers", 3),
        dropout=cfg.get("dropout", 0.2),
        use_batch_norm=cfg.get("use_batch_norm", False),
        use_ema=cfg.get("use_ema", False),
        use_whitening=cfg.get("use_whitening", False),
        loss_type=cfg.get("loss_type", "prototype_siglip"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    logger.info(f"Loaded CLOP (text_dim={cfg.get('text_dim')}, proj_dim={cfg.get('proj_dim')})")
    return model


def load_variant_conditions(
    cache_dir: Path,
    clop: CLOPAligner,
    device: torch.device,
) -> dict:
    """Load caption variant embeddings, project through CLOP, map to dedup types.

    Returns dict: dedup_type_id → (n_variants, proj_dim) array of projected conditions.
    """
    var_emb_path = cache_dir / "text_variant_embeddings.npy"
    var_map_path = cache_dir / "text_variant_map.json"
    grp_map_path = cache_dir / "text_group_mapping.json"

    if not all(p.exists() for p in [var_emb_path, var_map_path, grp_map_path]):
        logger.warning("Variant files not found — cannot use variant mode")
        return {}

    # Load raw BiomedBERT variant embeddings
    var_emb_raw = np.load(var_emb_path)  # (N_var, 1024)
    with open(var_map_path) as f:
        var_map = json.load(f)  # [[sub_cluster_id, variant_idx], ...]
    with open(grp_map_path) as f:
        grp_map = json.load(f)  # {sub_cluster_id: dedup_type_id}

    # Project ALL variants through CLOP in one batch
    logger.info(f"Projecting {len(var_emb_raw)} variant embeddings through CLOP...")
    with torch.no_grad():
        var_t = torch.from_numpy(var_emb_raw).float().to(device)
        # Process in batches to avoid OOM
        proj_parts = []
        bs = 512
        for i in range(0, len(var_t), bs):
            proj = clop.project_text(var_t[i:i + bs])
            proj_parts.append(proj.cpu().numpy())
        projected = np.concatenate(proj_parts, axis=0)  # (N_var, proj_dim)

    # Map to dedup types
    dedup_variants: dict = defaultdict(list)
    for emb_idx, (sub_id, _) in enumerate(var_map):
        dedup_id = grp_map.get(str(sub_id))
        if dedup_id is not None:
            dedup_variants[int(dedup_id)].append(projected[emb_idx])

    # Convert to arrays
    result = {}
    for tid, vecs in dedup_variants.items():
        arr = np.array(vecs)  # (n_var, proj_dim)
        # L2-normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        result[tid] = arr / norms

    logger.info(f"Variant conditions: {len(result)} types, "
                f"mean {np.mean([len(v) for v in result.values()]):.0f} variants/type")
    return result


def generate_all_types(
    model: DiT1D,
    projected_text: np.ndarray,
    group_ids: np.ndarray,
    num_per_type: int = 100,
    num_steps: int = 20,
    cfg_scale: float = 1.5,
    normalize: bool = True,
    condition_mode: str = "condition_noise",
    noise_scale: float = 0.03,
    variant_conditions: dict = None,
    variant_blend: float = 0.7,
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
        'centroid'        — one mean condition per type (original, least diverse)
        'per_cell'        — sample real per-cell conditions (note: no-op when
                            captions are deduplicated to 1 per type)
        'condition_noise'  — centroid + Gaussian noise, L2-normalized
                            (recommended: simple, effective diversity lever)
        'variant'         — blend CLOP-projected caption variants with canonical
                            centroid (controlled by variant_blend)
    noise_scale : float
        Std of Gaussian noise for condition_noise mode (default 0.03).
    variant_conditions : dict
        {type_id: (n_var, proj_dim)} from load_variant_conditions().
        Required only for variant mode.
    variant_blend : float
        Blend weight: cond = blend*canonical + (1-blend)*variant, then L2-norm.
        0.7 = stay close to training distribution while adding real prompt diversity.
    device : torch device

    Returns
    -------
    generated : (num_types * num_per_type, latent_dim) generated embeddings
    gen_labels : (num_types * num_per_type,) type labels
    type_ids : sorted unique type IDs
    """
    unique_types = np.sort(np.unique(group_ids))
    logger.info(f"Generating {num_per_type} cells × {len(unique_types)} types "
                f"= {num_per_type * len(unique_types)} total  "
                f"[mode={condition_mode}, cfg={cfg_scale}, steps={num_steps}")
    if condition_mode == "condition_noise":
        logger.info(f"  noise_scale={noise_scale}")

    rng = np.random.default_rng(42)
    all_gen = []
    all_labels = []

    for t_id in unique_types:
        mask = group_ids == t_id
        type_conds = projected_text[mask]  # (n_type, cond_dim)

        # Compute centroid (used by centroid, condition_noise, and fallback)
        proto = type_conds.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)

        if condition_mode == "centroid":
            cond = torch.from_numpy(proto).float().unsqueeze(0).repeat(num_per_type, 1)

        elif condition_mode == "per_cell":
            idx = rng.choice(len(type_conds), size=num_per_type, replace=True)
            sampled = type_conds[idx]
            norms = np.linalg.norm(sampled, axis=1, keepdims=True) + 1e-8
            sampled = sampled / norms
            cond = torch.from_numpy(sampled).float()

        elif condition_mode == "condition_noise":
            # Each cell gets centroid + small Gaussian noise, then L2-normalize
            # This simulates "slightly different text descriptions"
            protos = np.tile(proto, (num_per_type, 1))  # (n, cond_dim)
            noise = rng.normal(0, noise_scale, size=protos.shape)
            noisy = protos + noise
            norms = np.linalg.norm(noisy, axis=1, keepdims=True) + 1e-8
            noisy = noisy / norms
            cond = torch.from_numpy(noisy).float()

        elif condition_mode == "variant":
            if variant_conditions and t_id in variant_conditions:
                var_pool = variant_conditions[t_id]  # (n_var, cond_dim)
                idx = rng.choice(len(var_pool), size=num_per_type, replace=True)
                variants_raw = var_pool[idx]  # (n, cond_dim)
                # Blend with canonical centroid to stay in DiT's training distribution
                # cond = blend * canonical + (1 - blend) * variant
                protos = np.tile(proto, (num_per_type, 1))
                blended = variant_blend * protos + (1 - variant_blend) * variants_raw
                norms = np.linalg.norm(blended, axis=1, keepdims=True) + 1e-8
                blended = blended / norms
                cond = torch.from_numpy(blended).float()
            else:
                # Fallback to condition_noise if no variants for this type
                protos = np.tile(proto, (num_per_type, 1))
                noise = rng.normal(0, noise_scale, size=protos.shape)
                noisy = protos + noise
                norms = np.linalg.norm(noisy, axis=1, keepdims=True) + 1e-8
                noisy = noisy / norms
                cond = torch.from_numpy(noisy).float()
        else:
            raise ValueError(f"Unknown condition_mode: {condition_mode}")

        gen = model.sample(cond.to(device), num_steps=num_steps, cfg_scale=cfg_scale)

        if normalize:
            gen = F.normalize(gen, dim=-1)

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
    parser.add_argument("--dit-checkpoint", default=str(CHECKPOINT_DIR / "dit_best.pth"))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--num-per-type", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                        help="CFG scale (default 1.5; lower = more diverse)")
    parser.add_argument("--condition-mode",
                        choices=["centroid", "per_cell", "condition_noise", "variant"],
                        default="condition_noise",
                        help="Condition mode (default: condition_noise)")
    parser.add_argument("--noise-scale", type=float, default=0.03,
                        help="Noise std for condition_noise mode (default 0.03)")
    parser.add_argument("--clop-checkpoint", default=str(CHECKPOINT_DIR / "clop_best.pth"),
                        help="CLOP checkpoint for variant mode")
    parser.add_argument("--variant-blend", type=float, default=0.7,
                        help="Blend weight for variant mode (0.7=70%% canonical, 30%% variant)")
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

    # Load CLOP + variant conditions if variant mode requested
    variant_conditions = None
    if args.condition_mode == "variant":
        clop = load_clop(args.clop_checkpoint, device)
        variant_conditions = load_variant_conditions(cache, clop, device)
        if not variant_conditions:
            logger.warning("No variant conditions loaded — falling back to condition_noise")
            args.condition_mode = "condition_noise"
        del clop  # free VRAM
        torch.cuda.empty_cache()

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
        noise_scale=args.noise_scale,
        variant_conditions=variant_conditions,
        variant_blend=args.variant_blend,
        device=device,
    )

    # Save generated embeddings
    np.save(out / "generated_embeddings.npy", generated)
    np.save(out / "generated_labels.npy", gen_labels)
    logger.info(f"Saved generated embeddings → {out / 'generated_embeddings.npy'}")

    # Save generation metadata (reproducibility + consumed by Panel D)
    gen_metadata = {
        "condition_mode": args.condition_mode,
        "noise_scale": args.noise_scale,
        "cfg_scale": args.cfg_scale,
        "num_per_type": args.num_per_type,
        "num_steps": args.num_steps,
        "variant_blend": args.variant_blend if args.condition_mode == "variant" else None,
        "normalize": not args.no_normalize,
        "seed": args.seed,
        "dit_checkpoint": args.dit_checkpoint,
        "total_cells": int(generated.shape[0]),
        "num_types": int(len(type_ids)),
        "latent_dim": int(generated.shape[1]),
    }
    with open(out / "generation_metadata.json", "w") as f:
        json.dump(gen_metadata, f, indent=2)
    logger.info(f"Saved generation metadata → {out / 'generation_metadata.json'}")

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
