#!/usr/bin/env python3
"""Gene-gene correlation analysis within cell types.

Computes within-cell-type gene-gene Pearson correlation matrices for real
and generated expression data.  Reports:
  1. Per-type upper-tri correlation between R_real and R_gen (Mantel-style).
  2. Aggregate statistics across all types.
  3. A 4-panel diagnostic figure saved to results/figures/.

Usage:
    python scripts/analysis/gene_gene_correlation.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.visualization.style import apply_style, COLORS, FONT_TITLE, FONT_LABEL

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MIN_CELLS = 10          # need at least this many cells per type
TOP_GENES = 200         # use top variable genes for tractability
SEED = 42

np.random.seed(SEED)


def _load_data():
    gen_expr = np.load(RESULTS / "generated_expression.npy")
    real_expr = np.load(RESULTS / "real_expression.npy")
    gen_labels = np.load(RESULTS / "generated_expression_labels.npy")
    real_labels = np.load(RESULTS / "real_expression_labels.npy")
    with open(RESULTS / "expression_gene_names.json") as f:
        gene_names = json.load(f)
    return gen_expr, real_expr, gen_labels, real_labels, gene_names


def _select_hvg(real_expr, n=TOP_GENES):
    """Select top-n most variable genes by variance across all real cells."""
    var = np.var(real_expr, axis=0)
    idx = np.argsort(var)[-n:]
    return np.sort(idx)


def _upper_tri(mat):
    """Extract strict upper triangle."""
    return mat[np.triu_indices_from(mat, k=1)]


def _corr_matrix(X):
    """Pearson correlation matrix; nan-safe diagonal = 1."""
    n = X.shape[1]
    if X.shape[0] < 3:
        return np.full((n, n), np.nan)
    # center
    X_c = X - X.mean(axis=0, keepdims=True)
    norms = np.sqrt((X_c ** 2).sum(axis=0, keepdims=True))
    norms[norms == 0] = 1.0
    X_n = X_c / norms
    R = X_n.T @ X_n / max(X.shape[0] - 1, 1)  # sample corr
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -1.0, 1.0)


def analyse():
    apply_style()
    gen_expr, real_expr, gen_labels, real_labels, gene_names = _load_data()

    # Select highly variable genes
    hvg_idx = _select_hvg(real_expr, TOP_GENES)
    gene_subset = [gene_names[i] for i in hvg_idx]
    gen_sub = gen_expr[:, hvg_idx]
    real_sub = real_expr[:, hvg_idx]

    types = sorted(set(real_labels) & set(gen_labels))
    per_type_results = {}

    for t in types:
        r_mask = real_labels == t
        g_mask = gen_labels == t
        if r_mask.sum() < MIN_CELLS or g_mask.sum() < MIN_CELLS:
            continue
        R_real = _corr_matrix(real_sub[r_mask])
        R_gen = _corr_matrix(gen_sub[g_mask])
        ut_real = _upper_tri(R_real)
        ut_gen = _upper_tri(R_gen)
        valid = np.isfinite(ut_real) & np.isfinite(ut_gen)
        if valid.sum() < 10:
            continue
        mantel_r = np.corrcoef(ut_real[valid], ut_gen[valid])[0, 1]
        rmse = np.sqrt(np.mean((ut_real[valid] - ut_gen[valid]) ** 2))
        per_type_results[int(t)] = {
            "mantel_r": float(mantel_r),
            "rmse": float(rmse),
            "n_real": int(r_mask.sum()),
            "n_gen": int(g_mask.sum()),
        }

    mantel_vals = [v["mantel_r"] for v in per_type_results.values()]
    rmse_vals = [v["rmse"] for v in per_type_results.values()]

    summary = {
        "n_types_analysed": len(per_type_results),
        "n_genes": TOP_GENES,
        "mean_mantel_r": float(np.mean(mantel_vals)),
        "median_mantel_r": float(np.median(mantel_vals)),
        "std_mantel_r": float(np.std(mantel_vals)),
        "min_mantel_r": float(np.min(mantel_vals)),
        "max_mantel_r": float(np.max(mantel_vals)),
        "mean_rmse": float(np.mean(rmse_vals)),
        "per_type": per_type_results,
    }

    out_json = RESULTS / "gene_gene_correlation.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {out_json}")
    print(f"  Types analysed: {len(per_type_results)}")
    print(f"  Mean Mantel r: {np.mean(mantel_vals):.4f} ± {np.std(mantel_vals):.4f}")
    print(f"  Median Mantel r: {np.median(mantel_vals):.4f}")
    print(f"  Mean RMSE: {np.mean(rmse_vals):.4f}")

    # --- Produce figure ---
    _make_figure(per_type_results, gen_sub, real_sub, gen_labels, real_labels,
                 gene_subset, types)
    return summary


def _make_figure(per_type_results, gen_sub, real_sub, gen_labels, real_labels,
                 gene_subset, types):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    mantel_vals = [v["mantel_r"] for v in per_type_results.values()]

    # (a) Distribution of per-type Mantel correlations
    ax = axes[0, 0]
    ax.hist(mantel_vals, bins=20, color=COLORS["real"], edgecolor="white", alpha=0.85)
    ax.axvline(np.median(mantel_vals), color=COLORS["generated"], ls="--", lw=1.5,
               label=f"median = {np.median(mantel_vals):.3f}")
    ax.axvline(np.mean(mantel_vals), color=COLORS["annotation_dark"], ls="-", lw=1.5,
               label=f"mean = {np.mean(mantel_vals):.3f}")
    ax.set_xlabel("Mantel correlation (upper-tri r)")
    ax.set_ylabel("Number of cell types")
    ax.set_title("(a) Gene-gene correlation preservation")
    ax.legend(fontsize=8, frameon=False)

    # (b) Example: best-preserved cell type - side-by-side heatmaps
    best_type = max(per_type_results, key=lambda k: per_type_results[k]["mantel_r"])
    r_mask = real_labels == best_type
    g_mask = gen_labels == best_type
    R_real = _corr_matrix(real_sub[r_mask][:, :50])
    R_gen = _corr_matrix(gen_sub[g_mask][:, :50])
    ax = axes[0, 1]
    diff = R_gen - R_real
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_title(f"(b) Corr diff (type {best_type}, best)")
    ax.set_xlabel("Gene index (top 50 HVG)")
    ax.set_ylabel("Gene index")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Gen - Real")

    # (c) Example: worst-preserved cell type
    worst_type = min(per_type_results, key=lambda k: per_type_results[k]["mantel_r"])
    r_mask = real_labels == worst_type
    g_mask = gen_labels == worst_type
    R_real_w = _corr_matrix(real_sub[r_mask][:, :50])
    R_gen_w = _corr_matrix(gen_sub[g_mask][:, :50])
    ax = axes[1, 0]
    diff_w = R_gen_w - R_real_w
    im2 = ax.imshow(diff_w, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_title(f"(c) Corr diff (type {worst_type}, worst)")
    ax.set_xlabel("Gene index (top 50 HVG)")
    ax.set_ylabel("Gene index")
    plt.colorbar(im2, ax=ax, shrink=0.8, label="Gen - Real")

    # (d) Mantel r vs cell count scatter
    ax = axes[1, 1]
    n_cells = [per_type_results[t]["n_real"] for t in per_type_results]
    ax.scatter(n_cells, mantel_vals, s=20, alpha=0.7, c=COLORS["real"], edgecolors="none")
    if len(n_cells) > 2:
        r_corr = np.corrcoef(n_cells, mantel_vals)[0, 1]
        if np.isnan(r_corr):
            ax.set_title("(d) Mantel r vs cell count (r=N/A)")
            ax.annotate("Zero variance in cell counts",
                        xy=(0.5, 0.95), xycoords="axes fraction",
                        ha="center", fontsize=8, color=COLORS["error_red"])
        else:
            ax.set_title(f"(d) Mantel r vs cell count (r={r_corr:.3f})")
    else:
        ax.set_title("(d) Mantel r vs cell count")
    ax.set_xlabel("Number of real cells")
    ax.set_ylabel("Mantel correlation")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Gene-Gene Correlation Structure: Real vs Generated", fontsize=12)

    out_path = FIG_DIR / "fig20_gene_gene_correlation.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    analyse()
