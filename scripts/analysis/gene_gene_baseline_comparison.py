#!/usr/bin/env python3
"""Gene-gene correlation baseline comparison.

Extends the gene-gene correlation analysis (scripts/analysis/gene_gene_correlation.py)
by computing Mantel-style correlations for all available baseline methods (Gaussian,
scVI, scGen, EmbeddingVAE) and producing a comparative summary.

Outputs:
    results/gene_gene_correlation_baselines.json   — per-method Mantel stats
    results/figures/gene_gene_correlation_baselines.png/pdf  — comparison figure

Usage:
    python scripts/analysis/gene_gene_baseline_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.visualization.style import (
    apply_style, COLORS, FONT_TITLE, FONT_LABEL, FONT_SMALL,
    add_panel_label, save_with_vcd,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MIN_CELLS = 10
TOP_GENES = 200
SEED = 42
np.random.seed(SEED)


# ── helpers ──────────────────────────────────────────────────────────


def _upper_tri(mat: np.ndarray) -> np.ndarray:
    """Return flattened strict upper triangle of a square matrix."""
    return mat[np.triu_indices_from(mat, k=1)]


def _corr_matrix(X: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation (genes as columns)."""
    if X.shape[0] < 3:
        return np.full((X.shape[1], X.shape[1]), np.nan)
    Xc = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Xc, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    Xn = Xc / norms
    return Xn.T @ Xn / max(X.shape[0] - 1, 1)


def _select_hvg(expr: np.ndarray, n: int = TOP_GENES) -> np.ndarray:
    """Select indices of top-n highly variable genes."""
    var = np.var(expr, axis=0)
    return np.argsort(var)[-n:]


def _mantel_for_type(
    real_expr: np.ndarray,
    gen_expr: np.ndarray,
    hvg_idx: np.ndarray,
) -> float | None:
    """Compute Mantel-style correlation for one cell type."""
    r_real = real_expr[:, hvg_idx]
    r_gen = gen_expr[:, hvg_idx]
    if r_real.shape[0] < MIN_CELLS or r_gen.shape[0] < MIN_CELLS:
        return None
    C_real = _upper_tri(_corr_matrix(r_real))
    C_gen = _upper_tri(_corr_matrix(r_gen))
    mask = np.isfinite(C_real) & np.isfinite(C_gen)
    if mask.sum() < 10:
        return None
    return float(np.corrcoef(C_real[mask], C_gen[mask])[0, 1])


# ── data loading ─────────────────────────────────────────────────────


def _load_main_data():
    """Load real and CLOP-DiT generated expression."""
    real = np.load(RESULTS / "real_expression.npy")
    real_labels = np.load(RESULTS / "real_expression_labels.npy", allow_pickle=True)
    gen = np.load(RESULTS / "generated_expression.npy")
    gen_labels = np.load(RESULTS / "generated_expression_labels.npy", allow_pickle=True)
    return real, real_labels, gen, gen_labels


def _generate_gaussian_baseline(
    real_expr: np.ndarray, real_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian baseline: per-type N(mu, diag(sigma^2))."""
    types = np.unique(real_labels)
    all_gen = []
    all_labels = []
    for t in types:
        mask = real_labels == t
        X = real_expr[mask]
        n = X.shape[0]
        if n < MIN_CELLS:
            continue
        mu = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        samples = np.random.normal(mu, std, size=(n, X.shape[1]))
        all_gen.append(samples)
        all_labels.extend([t] * n)
    return np.vstack(all_gen), np.array(all_labels)


def _load_baseline_expression(method: str):
    """Try to load expression data for a learned baseline method."""
    base_dir = RESULTS / "baselines" / method
    expr_path = base_dir / "generated_expression.npy"
    labels_path = base_dir / "generated_labels.npy"
    if expr_path.exists() and labels_path.exists():
        return np.load(expr_path), np.load(labels_path, allow_pickle=True)
    return None, None


# ── main analysis ────────────────────────────────────────────────────


def compute_baseline_mantel_scores():
    """Compute Mantel correlations for each method."""
    real, real_labels, clopdit_gen, clopdit_labels = _load_main_data()
    hvg_idx = _select_hvg(real)
    types = np.unique(real_labels)

    methods = {}

    # CLOP-DiT
    methods["CLOP-DiT"] = _compute_per_type(real, real_labels, clopdit_gen, clopdit_labels, hvg_idx, types)

    # Gaussian baseline
    gauss_gen, gauss_labels = _generate_gaussian_baseline(real, real_labels)
    methods["Gaussian"] = _compute_per_type(real, real_labels, gauss_gen, gauss_labels, hvg_idx, types)

    # Learned baselines
    for method_name in ["scVI", "scGen", "EmbeddingVAE"]:
        gen_expr, gen_labels = _load_baseline_expression(method_name)
        if gen_expr is not None:
            methods[method_name] = _compute_per_type(
                real, real_labels, gen_expr, gen_labels, hvg_idx, types
            )

    return methods


def _compute_per_type(
    real: np.ndarray,
    real_labels: np.ndarray,
    gen: np.ndarray,
    gen_labels: np.ndarray,
    hvg_idx: np.ndarray,
    types: np.ndarray,
) -> dict:
    """Compute per-type and aggregate Mantel stats for one method."""
    per_type = {}
    for t in types:
        r_mask = real_labels == t
        g_mask = gen_labels == t
        if r_mask.sum() < MIN_CELLS or g_mask.sum() < MIN_CELLS:
            continue
        r = _mantel_for_type(real[r_mask], gen[g_mask], hvg_idx)
        if r is not None:
            per_type[str(t)] = r

    vals = list(per_type.values())
    return {
        "per_type": per_type,
        "mean_mantel_r": float(np.mean(vals)) if vals else None,
        "median_mantel_r": float(np.median(vals)) if vals else None,
        "std_mantel_r": float(np.std(vals)) if vals else None,
        "n_types": len(vals),
    }


def make_comparison_figure(methods: dict, output_stem: str | None = None):
    """Create grouped comparison figure."""
    apply_style()

    method_names = list(methods.keys())
    means = [methods[m]["mean_mantel_r"] or 0 for m in method_names]
    stds = [methods[m]["std_mantel_r"] or 0 for m in method_names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 2]})

    # Panel (a): bar chart of mean Mantel r
    ax = axes[0]
    colors = [COLORS.get(m.lower(), "#888888") for m in method_names]
    bars = ax.bar(range(len(method_names)), means, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=30, ha="right", fontsize=FONT_SMALL)
    ax.set_ylabel("Mean Mantel r", fontsize=FONT_LABEL)
    ax.set_title("Gene-gene correlation\npreservation", fontsize=FONT_TITLE)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    add_panel_label(ax, "a")

    # Panel (b): per-type violin/box comparison (CLOP-DiT vs Gaussian at minimum)
    ax2 = axes[1]
    data_for_violin = []
    labels_for_violin = []
    for m in method_names:
        vals = list(methods[m]["per_type"].values())
        if vals:
            data_for_violin.append(vals)
            labels_for_violin.append(m)

    if data_for_violin:
        parts = ax2.violinplot(data_for_violin, positions=range(len(data_for_violin)),
                               showmeans=True, showmedians=True)
        for i, pc in enumerate(parts.get("bodies", [])):
            c = COLORS.get(labels_for_violin[i].lower(), "#888888")
            pc.set_facecolor(c)
            pc.set_alpha(0.6)
        ax2.set_xticks(range(len(labels_for_violin)))
        ax2.set_xticklabels(labels_for_violin, rotation=30, ha="right", fontsize=FONT_SMALL)
    ax2.set_ylabel("Per-type Mantel r", fontsize=FONT_LABEL)
    ax2.set_title("Distribution of per-type\ncorrelation preservation", fontsize=FONT_TITLE)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    add_panel_label(ax2, "b")

    fig.tight_layout()

    if output_stem is None:
        output_stem = str(FIG_DIR / "gene_gene_correlation_baselines")
    save_with_vcd(fig, output_stem)
    plt.close(fig)
    print(f"Figure saved: {output_stem}.png / .pdf")


def main():
    print("Computing gene-gene correlation across methods...")
    methods = compute_baseline_mantel_scores()

    # Save JSON summary
    summary = {m: {k: v for k, v in d.items() if k != "per_type"} for m, d in methods.items()}
    out_path = RESULTS / "gene_gene_correlation_baselines.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {out_path}")

    # Print table
    print(f"\n{'Method':<20} {'Mean r':>10} {'Median r':>10} {'Std':>10} {'N types':>10}")
    print("-" * 62)
    for m, d in methods.items():
        mean_r = d["mean_mantel_r"]
        med_r = d["median_mantel_r"]
        std_r = d["std_mantel_r"]
        n = d["n_types"]
        print(f"{m:<20} {mean_r:>10.4f} {med_r:>10.4f} {std_r:>10.4f} {n:>10d}")

    make_comparison_figure(methods)
    print("\nDone.")


if __name__ == "__main__":
    main()
