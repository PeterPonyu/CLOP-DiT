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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

logger = logging.getLogger(__name__)

# ── Publication style ──
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "figure.constrained_layout.use": True,
}
TYPE_COLORS = plt.cm.tab10.colors


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
    """Panel L: Noise-scale vs fidelity/diversity tradeoff."""
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

        # FD
        rng = np.random.default_rng(42)
        n = min(len(real_cells), len(gen), 5000)
        r_idx = rng.choice(len(real_cells), n, replace=False)
        g_idx = rng.choice(len(gen), n, replace=False) if len(gen) > n else np.arange(len(gen))
        metrics = GenerationMetrics.full_evaluation(real_cells[r_idx], gen[g_idx])
        fd = metrics.get("frechet_distance", float("nan"))

        # Centroid cosine
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

        # Diversity ratio
        dr = compute_diversity_ratio(real_cells, group_ids, gen, gen_labels)

        fds.append(fd)
        centroids.append(centroid_cos)
        div_ratios.append(dr)
        logger.info(f"  ε={eps:.3f}: FD={fd:.4f}, cos={centroid_cos:.4f}, div_ratio={dr:.4f}")

    # Plot
    matplotlib.rcParams.update(STYLE)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle("Panel L: Noise-Scale Trade-off (CFG=1.5)", fontsize=14, fontweight="bold")

    color_fd = "#1976D2"
    color_cos = "#4CAF50"
    color_div = "#FF7043"

    ax1.plot(noise_scales, fds, "o-", color=color_fd, lw=2.5, markersize=10, label="Fréchet Distance ↓", zorder=3)
    ax1.set_xlabel("Noise Scale (ε)")
    ax1.set_ylabel("Fréchet Distance", color=color_fd)
    ax1.tick_params(axis="y", labelcolor=color_fd)

    ax2 = ax1.twinx()
    ax2.plot(noise_scales, centroids, "s-", color=color_cos, lw=2.5, markersize=10, label="Centroid Cosine ↑")
    ax2.plot(noise_scales, div_ratios, "D-", color=color_div, lw=2.5, markersize=10, label="Diversity Ratio ↑")
    ax2.set_ylabel("Cosine / Ratio")
    ax2.set_ylim(0, 1.2)

    # Shade optimal region
    ax1.axvspan(0.02, 0.04, alpha=0.1, color="green", label="Sweet spot")

    # Mark chosen production config ε=0.03 with prominent vertical line
    chosen_eps = 0.03
    if chosen_eps in noise_scales:
        idx_chosen = noise_scales.index(chosen_eps)
        ax1.axvline(x=chosen_eps, color="#9C27B0", linestyle="-", linewidth=2.5,
                     alpha=0.8, zorder=10, label=f"Production (ε={chosen_eps})")
        # Add annotation box with production metrics
        ax1.annotate(
            f"Production Config\n"
            f"ε={chosen_eps}, CFG={cfg_scale}\n"
            f"FD={fds[idx_chosen]:.3f}\n"
            f"cos={centroids[idx_chosen]:.3f}\n"
            f"div={div_ratios[idx_chosen]:.3f}",
            xy=(chosen_eps, fds[idx_chosen]),
            xytext=(chosen_eps + 0.02, fds[idx_chosen] + 0.05),
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F3E5F5", edgecolor="#9C27B0", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#9C27B0", lw=2),
            zorder=11,
        )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=10)

    # Annotations — best FD point (if different from chosen)
    best_idx = np.argmin(fds)
    if noise_scales[best_idx] != chosen_eps:
        ax1.annotate(f"ε={noise_scales[best_idx]:.2f}\nFD={fds[best_idx]:.3f}",
                     xy=(noise_scales[best_idx], fds[best_idx]),
                     xytext=(noise_scales[best_idx] + 0.01, fds[best_idx] + 0.02),
                     fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))

    path = Path(output_dir) / "panel_l_noise_tradeoff.png"
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi)
    logger.info(f"Saved Panel L → {path}")
    plt.close(fig)
    return path


def panel_m_conditioning_umap(
    model, projected_text, group_ids, real_cells,
    variant_conditions, selected_types,
    cfg_scale, noise_scale, variant_blend,
    num_per_type, num_steps, device, output_dir, dpi,
    type_names=None,
):
    """Panel M: Side-by-side UMAP of 3 conditioning modes for selected types."""
    logger.info(f"Generating Panel M: conditioning UMAP for types {selected_types}...")

    # Generate with each mode for selected types only
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

    results = {}  # mode_name -> (gen, gen_labels)
    for mode_name, kwargs in modes.items():
        seed_everything(42)
        gen, gen_labels, _ = generate_all_types(
            model, projected_text, group_ids,
            num_per_type=num_per_type, num_steps=num_steps,
            cfg_scale=cfg_scale, device=device, **kwargs,
        )
        # Filter to selected types
        mask = np.isin(gen_labels, selected_types)
        results[mode_name] = (gen[mask], gen_labels[mask])

    # Get real cells for selected types
    real_mask = np.isin(group_ids, selected_types)
    real_sel = real_cells[real_mask]
    real_lab = group_ids[real_mask]
    # Subsample real to ~500 per type for visualization
    rng = np.random.default_rng(42)
    real_sub_idx = []
    for tid in selected_types:
        tidx = np.where(real_lab == tid)[0]
        real_sub_idx.extend(rng.choice(tidx, min(500, len(tidx)), replace=False))
    real_sub = real_sel[real_sub_idx]
    real_sub_lab = real_lab[real_sub_idx]

    # Combine all data for consistent PCA/UMAP
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

    # Use PCA (2D) for deterministic, fast visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(combined)
    logger.info(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # Compute per-mode diversity stats for subtitles
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

    # Also compute real diversity for reference
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

    # Plot: one subplot per mode + one for real
    n_modes = 1 + len(results)  # real + each mode
    matplotlib.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 6), squeeze=False)
    axes = axes[0]

    fig.suptitle(
        f"Panel M: Conditioning Mode Comparison (CFG={cfg_scale}, {len(selected_types)} types, PCA 2D)",
        fontsize=14, fontweight="bold",
    )

    type_to_color = {tid: TYPE_COLORS[i % len(TYPE_COLORS)] for i, tid in enumerate(selected_types)}
    type_to_name = {tid: type_names.get(int(tid), f"Type_{tid}")[:25] if type_names else f"Type_{tid}"
                    for tid in selected_types}

    def plot_one(ax, mask, title, alpha=0.4, size=8):
        for tid in selected_types:
            tmask = mask & (combined_labels == tid)
            ax.scatter(coords[tmask, 0], coords[tmask, 1],
                       c=[type_to_color[tid]], s=size, alpha=alpha,
                       label=type_to_name[tid])
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    # Real cells
    real_mask_bool = combined_source == "Real"
    plot_one(axes[0], real_mask_bool,
             f"Real ({real_sub.shape[0]} cells)\ndiversity={real_diversity:.3f}",
             alpha=0.2, size=4)
    axes[0].legend(fontsize=7, markerscale=2, loc="best")

    # Each mode
    for i, mode_name in enumerate(results.keys()):
        mode_mask = combined_source == mode_name
        div_val = mode_diversity.get(mode_name, 0.0)
        plot_one(axes[i + 1], mode_mask,
                 f"{mode_name}\n({mode_mask.sum()} cells, div={div_val:.3f})")

    path = Path(output_dir) / "panel_m_conditioning_umap.png"
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi)
    logger.info(f"Saved Panel M → {path}")
    plt.close(fig)
    return path


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
