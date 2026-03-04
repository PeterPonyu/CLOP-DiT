#!/usr/bin/env python3
"""conditioning_analysis.py — Comparative analysis of conditioning strategies.

Generates Panel L (noise-scale tradeoff) and Panel M (conditioning UMAP comparison).

Panel L: Noise-scale vs fidelity/diversity tradeoff curve
  - Sweeps noise_scale ∈ [0, 0.01, 0.02, 0.03, 0.05, 0.08]
  - Plots FD, centroid cosine, and diversity ratio on the same axes

Panel M: UMAP comparison of 3 conditioning modes for 5 selected types
  - Centroid (identical conditions)
  - Condition noise (centroid + ε·N(0,I))
  - Variant blend (CLOP-projected caption variants blended with centroid)
  - Real cells for reference

Usage:
    python scripts/conditioning_analysis.py
    python scripts/conditioning_analysis.py --types 0 5 10 20 30  # specific types
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_embeddings import (
    load_dit, load_clop, load_variant_conditions, generate_all_types,
)
from src.utils.helpers import seed_everything, get_device
from src.evaluation.metrics import GenerationMetrics
from src.visualization.panels_conditioning import plot_panel_l, plot_panel_m

logger = logging.getLogger(__name__)


def pairwise_cosine_mean(X: np.ndarray) -> float:
    """Mean pairwise cosine similarity (upper triangle)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X_n = X / norms
    sim = X_n @ X_n.T
    idx = np.triu_indices(len(X), k=1)
    return float(sim[idx].mean())


def compute_diversity_ratio(real, real_labels, gen, gen_labels):
    """Compute mean diversity ratio across types."""
    ratios = []
    for tid in np.unique(gen_labels):
        r = real[real_labels == tid]
        g = gen[gen_labels == tid]
        if len(r) < 5 or len(g) < 5:
            continue
        # Subsample real to same size as gen for fair comparison
        rng = np.random.default_rng(42)
        r_sub = r[rng.choice(len(r), min(len(r), len(g)), replace=False)]
        real_div = 1.0 - pairwise_cosine_mean(r_sub)
        gen_div = 1.0 - pairwise_cosine_mean(g)
        if real_div > 1e-6:
            ratios.append(gen_div / real_div)
    return float(np.mean(ratios)) if ratios else 0.0


def panel_l_noise_tradeoff(
    model, projected_text, group_ids, real_cells,
    noise_scales, cfg_scale, num_per_type, num_steps,
    device, output_dir, dpi,
):
    """Panel L: Noise-scale vs fidelity/diversity tradeoff (compute data, then plot)."""
    logger.info("Generating Panel L: noise-scale tradeoff...")

    fds, centroids, div_ratios = [], [], []

    for eps in noise_scales:
        seed_everything(42)
        gen, gen_labels, type_ids = generate_all_types(
            model, projected_text, group_ids,
            num_per_type=num_per_type, num_steps=num_steps,
            cfg_scale=cfg_scale, condition_mode="condition_noise" if eps > 0 else "centroid",
            noise_scale=eps, device=device,
        )

        rng = np.random.default_rng(42)
        n = min(len(real_cells), len(gen), 5000)
        r_idx = rng.choice(len(real_cells), n, replace=False)
        g_idx = rng.choice(len(gen), n, replace=False) if len(gen) > n else np.arange(len(gen))
        metrics = GenerationMetrics.full_evaluation(real_cells[r_idx], gen[g_idx])
        fd = metrics.get("frechet_distance", float("nan"))

        cos_sims = []
        for tid in type_ids:
            r = real_cells[group_ids == tid]
            g = gen[gen_labels == tid]
            if len(r) < 5 or len(g) < 5:
                continue
            rc = r.mean(0); rc /= np.linalg.norm(rc) + 1e-8
            gc = g.mean(0); gc /= np.linalg.norm(gc) + 1e-8
            cos_sims.append(float(np.dot(rc, gc)))
        centroid_cos = np.mean(cos_sims)

        dr = compute_diversity_ratio(real_cells, group_ids, gen, gen_labels)

        fds.append(fd)
        centroids.append(centroid_cos)
        div_ratios.append(dr)
        logger.info(f"  ε={eps:.3f}: FD={fd:.4f}, cos={centroid_cos:.4f}, div_ratio={dr:.4f}")

    return plot_panel_l(noise_scales, fds, centroids, div_ratios, output_dir, dpi, cfg_scale)


def panel_m_conditioning_umap(
    model, projected_text, group_ids, real_cells,
    variant_conditions, selected_types,
    cfg_scale, noise_scale, variant_blend,
    num_per_type, num_steps, device, output_dir, dpi,
    type_names=None,
):
    """Panel M: Side-by-side UMAP of 3 conditioning modes (compute data, then plot)."""
    logger.info(f"Generating Panel M: conditioning UMAP for types {selected_types}...")
    type_names = type_names or {}

    modes = {
        "Centroid": {"condition_mode": "centroid"},
        f"Noise (ε={noise_scale})": {"condition_mode": "condition_noise", "noise_scale": noise_scale},
    }
    if variant_conditions:
        modes[f"Variant (blend={variant_blend})"] = {
            "condition_mode": "variant",
            "variant_conditions": variant_conditions,
            "variant_blend": variant_blend,
            "noise_scale": noise_scale,
        }

    results = {}
    for mode_name, kwargs in modes.items():
        seed_everything(42)
        gen, gen_labels, _ = generate_all_types(
            model, projected_text, group_ids,
            num_per_type=num_per_type, num_steps=num_steps,
            cfg_scale=cfg_scale, device=device, **kwargs,
        )
        mask = np.isin(gen_labels, selected_types)
        results[mode_name] = (gen[mask], gen_labels[mask])

    real_mask = np.isin(group_ids, selected_types)
    real_sel = real_cells[real_mask]
    real_lab = group_ids[real_mask]
    rng = np.random.default_rng(42)
    real_sub_idx = []
    for tid in selected_types:
        tidx = np.where(real_lab == tid)[0]
        real_sub_idx.extend(rng.choice(tidx, min(500, len(tidx)), replace=False))
    real_sub = real_sel[real_sub_idx]
    real_sub_lab = real_lab[real_sub_idx]

    all_data = [real_sub]
    all_labels_list = [real_sub_lab]
    all_source = ["Real"] * len(real_sub)
    for mode_name, (gen, gen_labels) in results.items():
        all_data.append(gen)
        all_labels_list.append(gen_labels)
        all_source.extend([mode_name] * len(gen))
    combined = np.concatenate(all_data, axis=0)
    combined_labels = np.concatenate(all_labels_list)
    combined_source = np.array(all_source)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(combined)
    logger.info(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    mode_diversity = {}
    for mode_name, (gen, gen_labels) in results.items():
        divs = []
        for tid in selected_types:
            g = gen[gen_labels == tid]
            if len(g) >= 5:
                norms = np.linalg.norm(g, axis=1, keepdims=True) + 1e-8
                g_n = g / norms
                sim = g_n @ g_n.T
                idx_tri = np.triu_indices(len(g), k=1)
                divs.append(1.0 - float(sim[idx_tri].mean()))
        mode_diversity[mode_name] = np.mean(divs) if divs else 0.0

    real_divs = []
    for tid in selected_types:
        rr = real_sub[real_sub_lab == tid]
        if len(rr) >= 5:
            norms = np.linalg.norm(rr, axis=1, keepdims=True) + 1e-8
            rr_n = rr / norms
            sim = rr_n @ rr_n.T
            idx_tri = np.triu_indices(len(rr), k=1)
            real_divs.append(1.0 - float(sim[idx_tri].mean()))
    real_diversity = np.mean(real_divs) if real_divs else 0.0

    mode_counts = {k: v[0].shape[0] for k, (v, _) in results.items()}
    return plot_panel_m(
        coords, combined_labels, combined_source,
        selected_types, mode_diversity, real_diversity, type_names,
        output_dir, dpi, cfg_scale,
        n_real=real_sub.shape[0], mode_counts=mode_counts,
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Conditioning analysis panels L+M")
    parser.add_argument("--dit-checkpoint", default="models/checkpoints/dit_best.pth")
    parser.add_argument("--clop-checkpoint", default="models/checkpoints/clop_best.pth")
    parser.add_argument("--cache-dir", default="data/cached_latents_v5.2")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--num-per-type", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--noise-scale", type=float, default=0.03)
    parser.add_argument("--variant-blend", type=float, default=0.7)
    parser.add_argument("--types", nargs="+", type=int, default=None,
                        help="Type IDs for Panel M (default: 5 evenly spaced)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    seed_everything(42)
    device = get_device()
    cache = Path(args.cache_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    projected_text = np.load(cache / "projected_text.npy")
    group_ids = np.load(cache / "text_group_ids_dedup.npy")
    real_cells = np.load(cache / "cell_embeddings_dedup_preprocessed.npy")

    # Load type names
    type_names = {}
    cap_path = cache / "text_captions_deduplicated.json"
    if cap_path.exists():
        with open(cap_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            name = v.split(" are ")[0] if " are " in v else v[:50]
            type_names[int(k)] = name

    # Load DiT
    model = load_dit(args.dit_checkpoint, device)

    # Load CLOP + variants
    clop = load_clop(args.clop_checkpoint, device)
    variant_conditions = load_variant_conditions(cache, clop, device)
    del clop
    torch.cuda.empty_cache()

    # ── Panel L: Noise-scale tradeoff ──
    noise_scales = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
    panel_l_noise_tradeoff(
        model, projected_text, group_ids, real_cells,
        noise_scales=noise_scales,
        cfg_scale=args.cfg_scale,
        num_per_type=args.num_per_type,
        num_steps=args.num_steps,
        device=device,
        output_dir=str(out),
        dpi=args.dpi,
    )

    # ── Panel M: Conditioning comparison UMAP ──
    unique_types = np.sort(np.unique(group_ids))
    if args.types:
        selected = args.types
    else:
        # Pick 5 evenly spaced types
        idx = np.linspace(0, len(unique_types) - 1, 5, dtype=int)
        selected = unique_types[idx].tolist()
    logger.info(f"Selected types for Panel M: {selected}")

    panel_m_conditioning_umap(
        model, projected_text, group_ids, real_cells,
        variant_conditions=variant_conditions,
        selected_types=selected,
        cfg_scale=args.cfg_scale,
        noise_scale=args.noise_scale,
        variant_blend=args.variant_blend,
        num_per_type=args.num_per_type,
        num_steps=args.num_steps,
        device=device,
        output_dir=str(out),
        dpi=args.dpi,
        type_names=type_names,
    )

    print(f"\nDone — Panels L+M saved to {out}/")


if __name__ == "__main__":
    main()
