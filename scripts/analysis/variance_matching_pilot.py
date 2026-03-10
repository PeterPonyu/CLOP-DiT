#!/usr/bin/env python3
"""Pilot variance-matching experiment using Sliced Wasserstein Distance (SWD).

This script evaluates the feasibility of a latent-space variance-matching
regularizer by computing per-cell-type SWD between real and generated latent
distributions. It also compares per-dimension variance statistics.

This is a diagnostic/analysis script, not a training modification.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sliced_wasserstein_distance(x: np.ndarray, y: np.ndarray, n_proj: int = 100,
                                 seed: int = 42) -> float:
    """Compute Sliced Wasserstein Distance between two point clouds.

    Parameters
    ----------
    x, y : (N, D) and (M, D) arrays
    n_proj : number of random projections
    seed : random seed

    Returns
    -------
    swd : float, average 1D Wasserstein distance across projections
    """
    rng = np.random.RandomState(seed)
    d = x.shape[1]
    # Random projection directions (unit vectors)
    theta = rng.randn(n_proj, d)
    theta = theta / np.linalg.norm(theta, axis=1, keepdims=True)

    # Project both point clouds
    proj_x = x @ theta.T  # (N, n_proj)
    proj_y = y @ theta.T  # (M, n_proj)

    # Sort each projection
    proj_x = np.sort(proj_x, axis=0)
    proj_y = np.sort(proj_y, axis=0)

    # Interpolate to same size if needed
    n, m = proj_x.shape[0], proj_y.shape[0]
    if n != m:
        # Interpolate the smaller to match the larger
        target_n = max(n, m)
        if n < target_n:
            idx = np.linspace(0, n - 1, target_n)
            proj_x = np.array([np.interp(idx, np.arange(n), proj_x[:, j])
                               for j in range(n_proj)]).T
        if m < target_n:
            idx = np.linspace(0, m - 1, target_n)
            proj_y = np.array([np.interp(idx, np.arange(m), proj_y[:, j])
                               for j in range(n_proj)]).T

    # Average absolute difference across quantiles and projections
    swd = np.mean(np.abs(proj_x - proj_y))
    return float(swd)


def main():
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "data" / "cached_latents_v5.2"
    results_dir = project_root / "results"
    output_dir = results_dir / "variance_matching_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load real embeddings and group IDs
    print("[var_pilot] Loading real embeddings and group IDs...")
    real_emb = np.load(cache_dir / "cell_embeddings_dedup_preprocessed.npy")
    group_ids = np.load(cache_dir / "text_group_ids_dedup.npy")

    # Load generated embeddings
    gen_emb = np.load(results_dir / "generated_embeddings.npy")
    gen_labels = np.load(results_dir / "generated_labels.npy")

    # Load captions for labeling
    with open(cache_dir / "text_captions_deduplicated.json") as f:
        captions = json.load(f)

    unique_types = sorted(np.unique(group_ids))
    print(f"[var_pilot] Real: {real_emb.shape}, Gen: {gen_emb.shape}")
    print(f"[var_pilot] {len(unique_types)} cell types")

    results = []

    for gid in unique_types:
        real_mask = group_ids == gid
        gen_mask = gen_labels == gid

        real_subset = real_emb[real_mask]
        gen_subset = gen_emb[gen_mask]

        if len(gen_subset) < 5:
            continue

        # Per-dimension variance
        real_var = np.var(real_subset, axis=0)
        gen_var = np.var(gen_subset, axis=0)
        var_ratio = np.mean(gen_var) / (np.mean(real_var) + 1e-10)
        var_corr = np.corrcoef(real_var, gen_var)[0, 1]

        # SWD
        swd = sliced_wasserstein_distance(real_subset, gen_subset, n_proj=200)

        # Per-dimension mean absolute error
        real_mean = np.mean(real_subset, axis=0)
        gen_mean = np.mean(gen_subset, axis=0)
        mean_cos = float(np.dot(real_mean, gen_mean) /
                         (np.linalg.norm(real_mean) * np.linalg.norm(gen_mean) + 1e-10))

        caption = captions[str(gid)] if str(gid) in captions else f"Type {gid}"
        # Truncate caption
        short_name = caption.split(",")[0][:40] if isinstance(caption, str) else str(caption)[:40]

        results.append({
            "group_id": int(gid),
            "name": short_name,
            "n_real": int(real_mask.sum()),
            "n_gen": int(gen_mask.sum()),
            "swd": swd,
            "var_ratio": float(var_ratio),
            "var_corr": float(var_corr),
            "mean_cos": mean_cos,
            "mean_real_var": float(np.mean(real_var)),
            "mean_gen_var": float(np.mean(gen_var)),
        })

    # Summary statistics
    swd_values = [r["swd"] for r in results]
    var_ratios = [r["var_ratio"] for r in results]
    var_corrs = [r["var_corr"] for r in results]

    summary = {
        "n_types_evaluated": len(results),
        "swd_mean": float(np.mean(swd_values)),
        "swd_median": float(np.median(swd_values)),
        "swd_std": float(np.std(swd_values)),
        "swd_min": float(np.min(swd_values)),
        "swd_max": float(np.max(swd_values)),
        "var_ratio_mean": float(np.mean(var_ratios)),
        "var_ratio_median": float(np.median(var_ratios)),
        "var_corr_mean": float(np.mean(var_corrs)),
        "var_corr_median": float(np.median(var_corrs)),
        "per_type": results,
    }

    with open(output_dir / "swd_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[var_pilot] === Summary ===")
    print(f"  Types evaluated: {len(results)}")
    print(f"  SWD:        mean={np.mean(swd_values):.4f}, median={np.median(swd_values):.4f}")
    print(f"  Var ratio:  mean={np.mean(var_ratios):.4f}, median={np.median(var_ratios):.4f}")
    print(f"  Var corr:   mean={np.mean(var_corrs):.4f}, median={np.median(var_corrs):.4f}")

    # ── Generate figure ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: SWD per type (sorted)
    sorted_results = sorted(results, key=lambda r: r["swd"], reverse=True)
    names = [r["name"][:25] for r in sorted_results]
    swds = [r["swd"] for r in sorted_results]
    ax = axes[0, 0]
    colors = ["#E53935" if s > np.mean(swds) + np.std(swds) else "#1976D2" for s in swds]
    ax.barh(range(len(names)), swds, color=colors, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("Sliced Wasserstein Distance")
    ax.set_title("(a) SWD per Cell Type (latent space)")
    ax.invert_yaxis()
    ax.axvline(np.mean(swds), color="gray", linestyle="--", alpha=0.7, label=f"mean={np.mean(swds):.3f}")
    ax.legend(fontsize=8)

    # Panel B: Variance ratio distribution
    ax = axes[0, 1]
    ax.hist(var_ratios, bins=25, color="#1976D2", alpha=0.8, edgecolor="white")
    ax.axvline(1.0, color="#E53935", linestyle="--", linewidth=2, label="Ideal (1.0)")
    ax.axvline(np.mean(var_ratios), color="#FFA726", linestyle="-.", linewidth=2,
               label=f"Mean={np.mean(var_ratios):.3f}")
    ax.set_xlabel("Variance Ratio (gen/real)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Per-Type Latent Variance Ratio")
    ax.legend()

    # Panel C: Variance correlation distribution
    ax = axes[1, 0]
    ax.hist(var_corrs, bins=25, color="#00897B", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(var_corrs), color="#FFA726", linestyle="-.", linewidth=2,
               label=f"Mean={np.mean(var_corrs):.3f}")
    ax.set_xlabel("Per-Dimension Variance Correlation")
    ax.set_ylabel("Count")
    ax.set_title("(c) Latent Variance Correlation (per dimension)")
    ax.legend()

    # Panel D: SWD vs. cell count
    ax = axes[1, 1]
    n_reals = [r["n_real"] for r in results]
    ax.scatter(n_reals, swd_values, c="#1976D2", alpha=0.6, s=30, edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Number of Real Cells")
    ax.set_ylabel("Sliced Wasserstein Distance")
    ax.set_title("(d) SWD vs. Training Cell Count")
    ax.set_xscale("log")

    # Add correlation annotation
    corr = np.corrcoef(np.log10(np.array(n_reals) + 1), swd_values)[0, 1]
    ax.annotate(f"r(log N, SWD) = {corr:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=9, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    fig_path = output_dir / "variance_matching_pilot.pdf"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[var_pilot] Figure saved to {fig_path}")

    # Also save a copy into the articles/figures/ directory
    article_fig_path = project_root / "articles" / "figures" / "fig18_variance_matching_pilot.pdf"
    import shutil
    shutil.copy(fig_path, article_fig_path)
    shutil.copy(str(fig_path).replace(".pdf", ".png"),
                str(article_fig_path).replace(".pdf", ".png"))
    print(f"[var_pilot] Copied to {article_fig_path}")


if __name__ == "__main__":
    main()
