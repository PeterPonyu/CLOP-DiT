#!/usr/bin/env python3
"""diversity_diagnostics.py — Comprehensive generation diversity analysis.

Tests whether the DiT is learning diverse within-type distributions
or collapsing to per-type centroids. Generates diagnostic metrics
and figures for the full pipeline:

  Test 1: Intra-type diversity (pairwise distances: real–real vs gen–gen)
  Test 2: Memorization check (nearest-neighbour distance to training data)
  Test 3: CFG sweep (how guidance scale affects diversity)
  Test 4: Noise sensitivity (how much starting noise affects output)
  Test 5: Condition sensitivity (centroid vs per-cell conditions)
  Test 6: Expression-level diversity (decoded gene expression variance)

Usage:
    python scripts/diversity_diagnostics.py
    python scripts/diversity_diagnostics.py --num-per-type 200 --cfg-scales 1.0 2.0 3.0 5.0 7.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.architecture.dit import DiT1D
from src.utils.helpers import seed_everything, get_device
from src.utils.paths import CACHE_DIR, RESULTS_DIR, CHECKPOINT_DIR, FIG_DIR
from src.visualization.panels_diversity import plot_diagnostics

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def load_dit(checkpoint_path: str, device: torch.device) -> DiT1D:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    # Resolve cond_dim from checkpoint
    cond_dim = cfg.get("cond_dim", None)
    if cond_dim is None:
        sd = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", {}))
        null_cond = sd.get("c_embedder.null_cond")
        cond_dim = null_cond.shape[-1] if null_cond is not None else 512

    model = DiT1D(
        latent_dim=cfg.get("latent_dim", 512),
        hidden_dim=cfg.get("hidden_dim", 512),
        cond_dim=cond_dim,
        num_tokens=cfg.get("num_tokens", 16),
        num_blocks=8,
        num_heads=8,
        cond_drop_prob=0.0,
    )
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def pairwise_cosine(X: np.ndarray) -> np.ndarray:
    """Upper-triangle pairwise cosine similarities."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X_n = X / norms
    sim = X_n @ X_n.T
    idx = np.triu_indices(len(X), k=1)
    return sim[idx]


def pairwise_l2(X: np.ndarray, max_pairs: int = 50000) -> np.ndarray:
    """Upper-triangle pairwise L2 distances (subsampled for large N)."""
    n = len(X)
    if n * (n - 1) // 2 > max_pairs:
        # Subsample
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=min(n, int(np.sqrt(max_pairs * 2))), replace=False)
        X = X[idx]
    from scipy.spatial.distance import pdist
    return pdist(X, metric="euclidean")


@torch.no_grad()
def generate_with_conditions(
    model: DiT1D,
    conds: torch.Tensor,
    num_steps: int = 20,
    cfg_scale: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate cells given explicit condition vectors."""
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    conds = conds.to(device)
    gen = model.sample(conds, num_steps=num_steps, cfg_scale=cfg_scale)
    gen = F.normalize(gen, dim=-1)
    return gen.cpu().numpy()


# ──────────────────────────────────────────────────────────────
# Test 1: Intra-type Diversity
# ──────────────────────────────────────────────────────────────

def test_intratype_diversity(
    real_cells: np.ndarray,
    real_labels: np.ndarray,
    gen_cells: np.ndarray,
    gen_labels: np.ndarray,
    type_names: Dict[int, str],
) -> Dict:
    """Compare pairwise distance distributions: real-real vs gen-gen vs real-gen."""
    results = {}
    unique_types = np.sort(np.unique(real_labels))

    for t_id in unique_types:
        r_mask = real_labels == t_id
        g_mask = gen_labels == t_id
        r = real_cells[r_mask]
        g = gen_cells[g_mask]

        if len(r) < 5 or len(g) < 5:
            continue

        # Subsample for speed
        rng = np.random.default_rng(42)
        r_sub = r[rng.choice(len(r), size=min(len(r), 200), replace=False)]
        g_sub = g[rng.choice(len(g), size=min(len(g), 200), replace=False)]

        # Pairwise cosine within real
        rr_cos = pairwise_cosine(r_sub)
        # Pairwise cosine within generated
        gg_cos = pairwise_cosine(g_sub)
        # Cross: real-gen
        r_n = r_sub / (np.linalg.norm(r_sub, axis=1, keepdims=True) + 1e-8)
        g_n = g_sub / (np.linalg.norm(g_sub, axis=1, keepdims=True) + 1e-8)
        rg_cos = (r_n @ g_n.T).flatten()

        name = type_names.get(t_id, f"type_{t_id}")[:40]
        results[name] = {
            "type_id": int(t_id),
            "n_real": int(len(r)),
            "n_gen": int(len(g)),
            "real_real_cos": {
                "mean": float(rr_cos.mean()),
                "std": float(rr_cos.std()),
                "median": float(np.median(rr_cos)),
            },
            "gen_gen_cos": {
                "mean": float(gg_cos.mean()),
                "std": float(gg_cos.std()),
                "median": float(np.median(gg_cos)),
            },
            "real_gen_cos": {
                "mean": float(rg_cos.mean()),
                "std": float(rg_cos.std()),
                "median": float(np.median(rg_cos)),
            },
            # Diversity ratio: gen diversity / real diversity
            # <1 means gen is LESS diverse (more collapsed)
            "diversity_ratio": float(
                (1.0 - gg_cos.mean()) / (1.0 - rr_cos.mean() + 1e-8)
            ),
        }

    # Summary
    divs = [v["diversity_ratio"] for v in results.values()]
    summary = {
        "mean_diversity_ratio": float(np.mean(divs)),
        "min_diversity_ratio": float(np.min(divs)),
        "max_diversity_ratio": float(np.max(divs)),
        "median_diversity_ratio": float(np.median(divs)),
        "n_collapsed": int(sum(1 for d in divs if d < 0.5)),
        "n_healthy": int(sum(1 for d in divs if d >= 0.5)),
        "interpretation": (
            "ratio < 0.5 = severe collapse; 0.5-0.8 = moderate collapse; "
            "0.8-1.2 = healthy; > 1.2 = over-diverse"
        ),
    }

    return {"per_type": results, "summary": summary}


# ──────────────────────────────────────────────────────────────
# Test 2: Memorization Check
# ──────────────────────────────────────────────────────────────

def test_memorization(
    real_cells: np.ndarray,
    gen_cells: np.ndarray,
    gen_labels: np.ndarray,
    real_labels: np.ndarray,
    top_k: int = 5,
) -> Dict:
    """For each generated cell, find distance to nearest real neighbour.

    If many gen cells are extremely close to specific real cells,
    that indicates memorization rather than generalization.
    """
    from scipy.spatial.distance import cdist

    # Cosine distance: 1 - cos_sim
    # Sample for speed
    rng = np.random.default_rng(42)
    n_gen_sample = min(len(gen_cells), 2000)
    g_idx = rng.choice(len(gen_cells), n_gen_sample, replace=False)
    g_sub = gen_cells[g_idx]
    g_lab = gen_labels[g_idx]

    # Normalize
    g_n = g_sub / (np.linalg.norm(g_sub, axis=1, keepdims=True) + 1e-8)
    r_n = real_cells / (np.linalg.norm(real_cells, axis=1, keepdims=True) + 1e-8)

    # Process in chunks to avoid OOM
    chunk_size = 500
    nn_dists = []
    nn_same_type = []

    for start in range(0, len(g_n), chunk_size):
        end = min(start + chunk_size, len(g_n))
        chunk = g_n[start:end]
        sims = chunk @ r_n.T  # (chunk, n_real)
        for i in range(len(chunk)):
            top_ids = np.argsort(sims[i])[-top_k:][::-1]
            nn_cos = sims[i, top_ids[0]]
            nn_dists.append(1.0 - nn_cos)
            # Is nearest neighbour same type?
            nn_type = real_labels[top_ids[0]]
            gen_type = g_lab[start + i]
            nn_same_type.append(int(nn_type == gen_type))

    nn_dists = np.array(nn_dists)
    nn_same_type = np.array(nn_same_type)

    return {
        "nn_cosine_distance": {
            "mean": float(nn_dists.mean()),
            "std": float(nn_dists.std()),
            "min": float(nn_dists.min()),
            "median": float(np.median(nn_dists)),
            "p5": float(np.percentile(nn_dists, 5)),
            "p25": float(np.percentile(nn_dists, 25)),
            "p75": float(np.percentile(nn_dists, 75)),
            "p95": float(np.percentile(nn_dists, 95)),
        },
        "nn_same_type_frac": float(nn_same_type.mean()),
        "n_very_close": int((nn_dists < 0.001).sum()),
        "n_close": int((nn_dists < 0.01).sum()),
        "n_moderate": int((nn_dists < 0.05).sum()),
        "n_sampled": int(n_gen_sample),
        "interpretation": (
            "nn_dist < 0.001 = near-copy; < 0.01 = very close; "
            "< 0.05 = moderate; > 0.05 = novel"
        ),
    }


# ──────────────────────────────────────────────────────────────
# Test 3: CFG Scale Sweep
# ──────────────────────────────────────────────────────────────

def test_cfg_sweep(
    model: DiT1D,
    projected_text: np.ndarray,
    group_ids: np.ndarray,
    cfg_scales: List[float],
    num_per_type: int = 50,
    num_steps: int = 20,
    sample_types: int = 10,
) -> Dict:
    """Generate at different CFG scales and measure diversity."""
    device = next(model.parameters()).device
    unique = np.sort(np.unique(group_ids))
    rng = np.random.default_rng(42)
    sample_type_ids = rng.choice(unique, size=min(sample_types, len(unique)), replace=False)

    results = {}
    for cfg in cfg_scales:
        type_diversities = []
        type_norms = []

        for t_id in sample_type_ids:
            mask = group_ids == t_id
            proto = projected_text[mask].mean(axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            cond = torch.from_numpy(proto).float().unsqueeze(0).repeat(num_per_type, 1)

            seed_everything(42)
            gen = generate_with_conditions(model, cond, num_steps=num_steps, cfg_scale=cfg)

            # Measure diversity
            cos_sims = pairwise_cosine(gen)
            intra_dist = 1.0 - cos_sims.mean()
            type_diversities.append(intra_dist)
            type_norms.append(np.linalg.norm(gen, axis=1).mean())

        results[f"cfg_{cfg:.1f}"] = {
            "cfg_scale": float(cfg),
            "mean_intra_diversity": float(np.mean(type_diversities)),
            "std_intra_diversity": float(np.std(type_diversities)),
            "mean_norm": float(np.mean(type_norms)),
        }
        logger.info(
            f"  CFG={cfg:.1f}: intra_div={np.mean(type_diversities):.6f} "
            f"norm={np.mean(type_norms):.4f}"
        )

    return results


# ──────────────────────────────────────────────────────────────
# Test 4: Noise Sensitivity — same condition, different seeds
# ──────────────────────────────────────────────────────────────

def test_noise_sensitivity(
    model: DiT1D,
    projected_text: np.ndarray,
    group_ids: np.ndarray,
    seeds: List[int] = [0, 1, 2, 3, 4],
    num_per_type: int = 50,
    num_steps: int = 20,
    cfg_scale: float = 3.0,
    sample_types: int = 10,
) -> Dict:
    """Generate with different seeds and measure output variability."""
    device = next(model.parameters()).device
    unique = np.sort(np.unique(group_ids))
    rng = np.random.default_rng(42)
    sample_type_ids = rng.choice(unique, size=min(sample_types, len(unique)), replace=False)

    all_cross_dists = []
    per_type_results = {}

    for t_id in sample_type_ids:
        mask = group_ids == t_id
        proto = projected_text[mask].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        cond = torch.from_numpy(proto).float().unsqueeze(0).repeat(num_per_type, 1)

        seed_gens = []
        for seed in seeds:
            gen = generate_with_conditions(model, cond, num_steps=num_steps,
                                           cfg_scale=cfg_scale, seed=seed)
            seed_gens.append(gen)

        # Cross-seed diversity: pairwise distance between centroids of different seeds
        centroids = np.array([g.mean(axis=0) for g in seed_gens])
        centroid_cos = pairwise_cosine(centroids)
        cross_dist = 1.0 - centroid_cos.mean()

        # Within-seed diversity (average)
        within_divs = []
        for g in seed_gens:
            cos = pairwise_cosine(g)
            within_divs.append(1.0 - cos.mean())

        per_type_results[int(t_id)] = {
            "cross_seed_distance": float(cross_dist),
            "mean_within_seed_diversity": float(np.mean(within_divs)),
        }
        all_cross_dists.append(cross_dist)

    return {
        "per_type": per_type_results,
        "summary": {
            "mean_cross_seed_dist": float(np.mean(all_cross_dists)),
            "interpretation": (
                "cross_seed_dist ≈ 0 means different noise → near-identical output "
                "(centroid collapse). Higher = more noise-sensitive (more diverse)."
            ),
        },
    }


# ──────────────────────────────────────────────────────────────
# Test 5: Condition Sensitivity — centroid vs per-cell prompts
# ──────────────────────────────────────────────────────────────

def test_condition_sensitivity(
    model: DiT1D,
    projected_text: np.ndarray,
    group_ids: np.ndarray,
    num_per_type: int = 100,
    num_steps: int = 20,
    cfg_scale: float = 3.0,
    sample_types: int = 10,
    noise_scale: float = 0.03,
) -> Dict:
    """Compare generation quality: centroid vs condition_noise vs per_cell.

    - Centroid mode: same condition for all cells in a type
    - Condition noise: centroid + ε·N(0,I), L2-normalized (the real diversity lever)
    - Per-cell mode: sample from per-cell projected_text (no-op when deduped)
    """
    device = next(model.parameters()).device
    unique = np.sort(np.unique(group_ids))
    rng = np.random.default_rng(42)
    sample_type_ids = rng.choice(unique, size=min(sample_types, len(unique)), replace=False)

    results = {}

    for t_id in sample_type_ids:
        mask = group_ids == t_id
        type_pts = projected_text[mask]

        # Mode A: Centroid condition
        proto = type_pts.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        cond_centroid = torch.from_numpy(proto).float().unsqueeze(0).repeat(num_per_type, 1)

        seed_everything(42)
        gen_centroid = generate_with_conditions(model, cond_centroid,
                                                num_steps=num_steps, cfg_scale=cfg_scale)

        # Mode B: Condition noise (centroid + Gaussian noise, L2-normed)
        protos = np.tile(proto, (num_per_type, 1))
        noise = rng.normal(0, noise_scale, size=protos.shape)
        noisy = protos + noise
        norms = np.linalg.norm(noisy, axis=1, keepdims=True) + 1e-8
        noisy = noisy / norms
        cond_noise = torch.from_numpy(noisy).float()

        seed_everything(42)  # Same seeds for fair comparison
        gen_noise = generate_with_conditions(model, cond_noise,
                                             num_steps=num_steps, cfg_scale=cfg_scale)

        # Diversity: centroid vs condition_noise
        cos_centroid = pairwise_cosine(gen_centroid)
        cos_noise = pairwise_cosine(gen_noise)

        div_centroid = 1.0 - cos_centroid.mean()
        div_noise = 1.0 - cos_noise.mean()

        results[int(t_id)] = {
            "centroid_diversity": float(div_centroid),
            "noise_diversity": float(div_noise),
            "diversity_gain": float(div_noise / (div_centroid + 1e-8)),
            "n_type_cells": int(mask.sum()),
        }

    # Summary
    gains = [v["diversity_gain"] for v in results.values()]
    return {
        "per_type": results,
        "summary": {
            "mean_diversity_gain": float(np.mean(gains)),
            "max_diversity_gain": float(np.max(gains)),
            "noise_scale": noise_scale,
            "interpretation": (
                "gain > 1.0 means condition noise produces MORE diverse outputs. "
                "gain ≈ 1.0 means the model ignores condition noise. "
                "The gap reveals how much diversity is added by noisy conditions."
            ),
        },
    }


# ──────────────────────────────────────────────────────────────
# Test 6: Expression-level diversity
# ──────────────────────────────────────────────────────────────

def test_expression_diversity(
    real_expr_path: str = "results/real_expression.npy",
    gen_expr_path: str = "results/generated_expression.npy",
    real_labels_path: str = "results/real_expression_labels.npy",
    gen_labels_path: str = "results/generated_expression_labels.npy",
) -> Dict:
    """Compare expression-level variability between real and generated."""
    if not all(Path(p).exists() for p in [real_expr_path, gen_expr_path,
                                           real_labels_path, gen_labels_path]):
        logger.info("Expression data not found — skipping expression diversity test")
        return {}

    real = np.load(real_expr_path)
    gen = np.load(gen_expr_path)
    r_lab = np.load(real_labels_path)
    g_lab = np.load(gen_labels_path)

    # Per-cell expression std
    real_cell_std = real.std(axis=1)
    gen_cell_std = gen.std(axis=1)

    # Per-gene expression std (across cells)
    real_gene_std = real.std(axis=0)
    gen_gene_std = gen.std(axis=0)

    # Per-type analysis
    per_type = {}
    for t_id in np.unique(r_lab):
        r_mask = r_lab == t_id
        g_mask = g_lab == t_id
        if r_mask.sum() < 5 or g_mask.sum() < 5:
            continue
        r = real[r_mask]
        g = gen[g_mask]

        per_type[int(t_id)] = {
            "real_mean_gene_std": float(r.std(axis=0).mean()),
            "gen_mean_gene_std": float(g.std(axis=0).mean()),
            "gene_std_ratio": float(g.std(axis=0).mean() / (r.std(axis=0).mean() + 1e-8)),
            "real_mean_cell_std": float(r.std(axis=1).mean()),
            "gen_mean_cell_std": float(g.std(axis=1).mean()),
        }

    gene_std_ratios = [v["gene_std_ratio"] for v in per_type.values()]

    return {
        "overall": {
            "real_mean_cell_std": float(real_cell_std.mean()),
            "gen_mean_cell_std": float(gen_cell_std.mean()),
            "real_mean_gene_std": float(real_gene_std.mean()),
            "gen_mean_gene_std": float(gen_gene_std.mean()),
            "gene_std_ratio": float(gen_gene_std.mean() / (real_gene_std.mean() + 1e-8)),
        },
        "per_type_gene_std_ratio": {
            "mean": float(np.mean(gene_std_ratios)),
            "min": float(np.min(gene_std_ratios)),
            "max": float(np.max(gene_std_ratios)),
        },
        "interpretation": (
            "gene_std_ratio < 1 means generated has LESS per-gene variability "
            "(=centroid collapse). ratio ≈ 1 = healthy. > 1 = more variable."
        ),
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="CLOP-DiT Diversity Diagnostics")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dit-checkpoint", default=None)
    parser.add_argument("--generated", default=None)
    parser.add_argument("--generated-labels", default=None)
    parser.add_argument("--num-per-type", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--cfg-scales", nargs="+", type=float,
                        default=[1.0, 1.5, 2.0, 3.0, 5.0, 7.0])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.cache_dir = args.cache_dir or str(CACHE_DIR)
    args.dit_checkpoint = args.dit_checkpoint or str(CHECKPOINT_DIR / "dit_best.pth")
    args.generated = args.generated or str(RESULTS_DIR / "generated_embeddings.npy")
    args.generated_labels = args.generated_labels or str(RESULTS_DIR / "generated_labels.npy")
    args.output_dir = args.output_dir or str(RESULTS_DIR)

    device = get_device()
    cache = Path(args.cache_dir)

    # Load data — use the SAME preprocessed embeddings that generate_embeddings.py used
    # cell_embeddings_dedup_preprocessed.npy = ZCA+L2 preprocessed, dedup (167,245)
    # text_group_ids_dedup.npy = 69-type group IDs matching the dedup preprocessed cells
    # These match the space the DiT was trained in and generates in
    preprocessed_path = cache / "cell_embeddings_dedup_preprocessed.npy"
    group_id_path = cache / "text_group_ids_dedup.npy"
    if not preprocessed_path.exists() or not group_id_path.exists():
        logger.error("Required files not found: cell_embeddings_dedup_preprocessed.npy "
                      "and text_group_ids_dedup.npy")
        return
    real_cells = np.load(preprocessed_path)
    group_ids = np.load(group_id_path)
    projected_text = np.load(cache / "projected_text.npy")
    logger.info(f"Using cell_embeddings_dedup_preprocessed.npy + text_group_ids_dedup.npy")

    gen_cells = np.load(args.generated)
    gen_labels = np.load(args.generated_labels)

    # Load type names
    cap_path = cache / "text_captions_deduplicated.json"
    type_names = {}
    if cap_path.exists():
        with open(cap_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            name = v.split(" are ")[0] if " are " in v else v[:50]
            type_names[int(k)] = name

    logger.info(f"Real: {real_cells.shape}, Generated: {gen_cells.shape}")
    logger.info(f"Types: {len(np.unique(group_ids))} real, {len(np.unique(gen_labels))} gen")

    all_results = {}

    # Test 1: Intra-type diversity
    print("\n" + "=" * 60)
    print("TEST 1: Intra-Type Diversity (real-real vs gen-gen)")
    print("=" * 60)
    t1 = test_intratype_diversity(real_cells, group_ids, gen_cells, gen_labels, type_names)
    all_results["test1_intratype_diversity"] = t1
    s = t1["summary"]
    print(f"  Mean diversity ratio: {s['mean_diversity_ratio']:.4f}")
    print(f"  Range: [{s['min_diversity_ratio']:.4f}, {s['max_diversity_ratio']:.4f}]")
    print(f"  Collapsed (<0.5): {s['n_collapsed']}, Healthy (>=0.5): {s['n_healthy']}")

    # Test 2: Memorization
    print("\n" + "=" * 60)
    print("TEST 2: Memorization Check")
    print("=" * 60)
    t2 = test_memorization(real_cells, gen_cells, gen_labels, group_ids)
    all_results["test2_memorization"] = t2
    nn = t2["nn_cosine_distance"]
    print(f"  NN cosine dist: mean={nn['mean']:.6f}, median={nn['median']:.6f}")
    print(f"  Near-copies (d<0.001): {t2['n_very_close']}/{t2['n_sampled']}")
    print(f"  NN same-type fraction: {t2['nn_same_type_frac']:.1%}")

    # Test 3: CFG Sweep
    print("\n" + "=" * 60)
    print("TEST 3: CFG Scale Sweep")
    print("=" * 60)
    model = load_dit(args.dit_checkpoint, device)
    t3 = test_cfg_sweep(model, projected_text, group_ids,
                        cfg_scales=args.cfg_scales,
                        num_per_type=args.num_per_type,
                        num_steps=args.num_steps)
    all_results["test3_cfg_sweep"] = t3

    # Test 4: Noise sensitivity
    print("\n" + "=" * 60)
    print("TEST 4: Noise Sensitivity (same cond, different seeds)")
    print("=" * 60)
    t4 = test_noise_sensitivity(model, projected_text, group_ids,
                                num_per_type=args.num_per_type,
                                num_steps=args.num_steps)
    all_results["test4_noise_sensitivity"] = t4
    print(f"  Mean cross-seed distance: {t4['summary']['mean_cross_seed_dist']:.6f}")

    # Test 5: Condition sensitivity
    print("\n" + "=" * 60)
    print("TEST 5: Centroid vs Condition Noise")
    print("=" * 60)
    t5 = test_condition_sensitivity(model, projected_text, group_ids,
                                    num_per_type=args.num_per_type,
                                    num_steps=args.num_steps)
    all_results["test5_condition_sensitivity"] = t5
    print(f"  Mean diversity gain (noise/centroid): {t5['summary']['mean_diversity_gain']:.4f}x")

    # Test 6: Expression diversity
    print("\n" + "=" * 60)
    print("TEST 6: Expression-Level Diversity")
    print("=" * 60)
    t6 = test_expression_diversity()
    all_results["test6_expression_diversity"] = t6
    if t6:
        o = t6["overall"]
        print(f"  Real gene std: {o['real_mean_gene_std']:.6f}")
        print(f"  Gen gene std:  {o['gen_mean_gene_std']:.6f}")
        print(f"  Ratio (gen/real): {o['gene_std_ratio']:.4f}")

    # Save results
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "diversity_diagnostics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Saved results → {out / 'diversity_diagnostics.json'}")

    # Generate figures
    print("\n" + "=" * 60)
    print("Generating diagnostic figures...")
    print("=" * 60)
    saved = plot_diagnostics(all_results, output_dir=str(out / "figures"), dpi=args.dpi)

    # Final verdict
    print("\n" + "=" * 60)
    print("DIVERSITY DIAGNOSTIC SUMMARY")
    print("=" * 60)
    verdict_lines = []
    if t1["summary"]["mean_diversity_ratio"] < 0.5:
        verdict_lines.append("SEVERE CENTROID COLLAPSE — gen diversity << real diversity")
    elif t1["summary"]["mean_diversity_ratio"] < 0.8:
        verdict_lines.append("MODERATE COLLAPSE — gen diversity somewhat reduced")
    else:
        verdict_lines.append("HEALTHY DIVERSITY — gen ≈ real diversity")

    if t2["n_very_close"] > t2["n_sampled"] * 0.1:
        verdict_lines.append("MEMORIZATION DETECTED — many near-copies of training data")
    else:
        verdict_lines.append("NO MEMORIZATION — generated cells are novel")

    if t5["summary"]["mean_diversity_gain"] > 1.5:
        verdict_lines.append("CONDITION-SENSITIVE — noisy conditions significantly increase diversity")
    elif t5["summary"]["mean_diversity_gain"] > 1.1:
        verdict_lines.append("CONDITION-RESPONSIVE — noisy conditions moderately help diversity")
    else:
        verdict_lines.append("CONDITION-INSENSITIVE — model output similar regardless of condition noise")

    for line in verdict_lines:
        print(f"  {line}")
    print("=" * 60)

    print(f"\nSaved {len(saved)} figures and results to {args.output_dir}/")


if __name__ == "__main__":
    main()
