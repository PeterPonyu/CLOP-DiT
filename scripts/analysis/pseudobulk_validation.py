#!/usr/bin/env python3
"""Pseudobulk validation — bridge from embedding-space to expression-level evidence.

For each cell type, computes pseudobulk profiles (mean expression) for both
real and generated cells, then correlates them. This validates that the
generation pipeline preserves biologically meaningful expression patterns
beyond just embedding-space metrics.

Output:
    results/validation/pseudobulk_validation.json
    results/validation/pseudobulk_per_type.json

Usage:
    python scripts/analysis/pseudobulk_validation.py
    python scripts/analysis/pseudobulk_validation.py --top-genes 2000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_pseudobulk(expression: np.ndarray, labels: np.ndarray,
                       type_names: dict) -> dict:
    """Compute mean expression per cell type (pseudobulk profile)."""
    pseudobulk = {}
    for gid in np.unique(labels):
        mask = labels == gid
        n_cells = mask.sum()
        if n_cells < 3:
            continue
        mean_expr = expression[mask].mean(axis=0)
        std_expr = expression[mask].std(axis=0)
        name = type_names.get(int(gid), f"Type_{gid}")
        pseudobulk[name] = {
            "mean": mean_expr,
            "std": std_expr,
            "n_cells": int(n_cells),
            "type_id": int(gid),
        }
    return pseudobulk


def correlate_pseudobulk(real_pb: dict, gen_pb: dict,
                         gene_names: list) -> dict:
    """Correlate pseudobulk profiles between real and generated."""
    shared_types = sorted(set(real_pb.keys()) & set(gen_pb.keys()))
    logger.info(f"Shared cell types for pseudobulk: {len(shared_types)}")

    per_type = {}
    all_pearson = []
    all_spearman = []

    for ct in shared_types:
        real_mean = real_pb[ct]["mean"]
        gen_mean = gen_pb[ct]["mean"]

        # Full-gene correlation
        r_pearson, p_pearson = stats.pearsonr(real_mean, gen_mean)
        r_spearman, p_spearman = stats.spearmanr(real_mean, gen_mean)

        # Top variable genes correlation (genes with highest real variance)
        real_std = real_pb[ct]["std"]
        top_var_idx = np.argsort(real_std)[-500:]
        r_topvar, _ = stats.pearsonr(real_mean[top_var_idx], gen_mean[top_var_idx])

        # Mean absolute error
        mae = float(np.mean(np.abs(real_mean - gen_mean)))

        # Cosine similarity
        norm_real = np.linalg.norm(real_mean)
        norm_gen = np.linalg.norm(gen_mean)
        cosine = float(np.dot(real_mean, gen_mean) / (norm_real * norm_gen + 1e-10))

        # Top expressed genes overlap
        n_top = 100
        real_top = set(np.argsort(real_mean)[-n_top:])
        gen_top = set(np.argsort(gen_mean)[-n_top:])
        top_overlap = len(real_top & gen_top) / n_top

        per_type[ct] = {
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "spearman_rho": float(r_spearman),
            "spearman_p": float(p_spearman),
            "pearson_topvar": float(r_topvar),
            "mae": mae,
            "cosine_sim": cosine,
            "top100_overlap": float(top_overlap),
            "n_real": real_pb[ct]["n_cells"],
            "n_gen": gen_pb[ct]["n_cells"],
        }
        all_pearson.append(r_pearson)
        all_spearman.append(r_spearman)

    # Global pseudobulk: average across all cells regardless of type
    summary = {
        "n_types_evaluated": len(shared_types),
        "pearson_mean": float(np.mean(all_pearson)),
        "pearson_median": float(np.median(all_pearson)),
        "pearson_std": float(np.std(all_pearson)),
        "spearman_mean": float(np.mean(all_spearman)),
        "spearman_median": float(np.median(all_spearman)),
        "spearman_std": float(np.std(all_spearman)),
        "pct_pearson_gt_0.8": float(100 * np.mean(np.array(all_pearson) > 0.8)),
        "pct_pearson_gt_0.9": float(100 * np.mean(np.array(all_pearson) > 0.9)),
    }

    return summary, per_type


def cross_type_correlation(real_pb: dict, gen_pb: dict) -> dict:
    """Cross-type pseudobulk: do real and generated rank cell types similarly?

    For each gene, compute the vector of per-type means (real and gen),
    then correlate across types.
    """
    shared_types = sorted(set(real_pb.keys()) & set(gen_pb.keys()))
    if len(shared_types) < 5:
        return {"error": "Too few shared types for cross-type analysis"}

    real_matrix = np.stack([real_pb[ct]["mean"] for ct in shared_types])  # (T, G)
    gen_matrix = np.stack([gen_pb[ct]["mean"] for ct in shared_types])    # (T, G)

    n_genes = real_matrix.shape[1]
    per_gene_r = np.zeros(n_genes)
    for g in range(n_genes):
        if np.std(real_matrix[:, g]) < 1e-10 or np.std(gen_matrix[:, g]) < 1e-10:
            per_gene_r[g] = 0.0
        else:
            per_gene_r[g] = stats.pearsonr(real_matrix[:, g], gen_matrix[:, g])[0]

    return {
        "cross_type_gene_corr_mean": float(np.nanmean(per_gene_r)),
        "cross_type_gene_corr_median": float(np.nanmedian(per_gene_r)),
        "cross_type_gene_corr_std": float(np.nanstd(per_gene_r)),
        "pct_genes_corr_gt_0.5": float(100 * np.nanmean(per_gene_r > 0.5)),
        "pct_genes_corr_gt_0.8": float(100 * np.nanmean(per_gene_r > 0.8)),
        "n_types": len(shared_types),
        "n_genes": n_genes,
    }


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import RESULTS_DIR, CACHE_DIR

    parser = argparse.ArgumentParser(description="Pseudobulk validation")
    parser.add_argument("--real-expr", default=None)
    parser.add_argument("--gen-expr", default=None)
    parser.add_argument("--real-labels", default=None)
    parser.add_argument("--gen-labels", default=None)
    parser.add_argument("--gene-names", default=None)
    parser.add_argument("--caption-json", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    real_expr_path = args.real_expr or str(RESULTS_DIR / "real_expression.npy")
    gen_expr_path = args.gen_expr or str(RESULTS_DIR / "generated_expression.npy")
    real_labels_path = args.real_labels or str(RESULTS_DIR / "real_expression_labels.npy")
    gen_labels_path = args.gen_labels or str(RESULTS_DIR / "generated_expression_labels.npy")
    gene_names_path = args.gene_names or str(RESULTS_DIR / "expression_gene_names.json")
    caption_path = args.caption_json or str(CACHE_DIR / "text_captions_deduplicated.json")
    output_dir = Path(args.output_dir or str(RESULTS_DIR / "validation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading expression data...")
    real_expr = np.load(real_expr_path).astype(np.float32)
    gen_expr = np.load(gen_expr_path).astype(np.float32)
    real_labels = np.load(real_labels_path)
    gen_labels = np.load(gen_labels_path)

    with open(gene_names_path) as f:
        gene_names = json.load(f)

    type_names = {}
    if Path(caption_path).exists():
        with open(caption_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            name = v.split(" are ")[0] if " are " in v else v[:50]
            type_names[int(k)] = name

    logger.info(f"Real: {real_expr.shape}, Gen: {gen_expr.shape}, Genes: {len(gene_names)}")

    # Compute pseudobulk profiles
    logger.info("Computing pseudobulk profiles...")
    real_pb = compute_pseudobulk(real_expr, real_labels, type_names)
    gen_pb = compute_pseudobulk(gen_expr, gen_labels, type_names)

    # Correlate
    logger.info("Correlating pseudobulk profiles...")
    summary, per_type = correlate_pseudobulk(real_pb, gen_pb, gene_names)

    # Cross-type gene correlation
    logger.info("Computing cross-type gene correlation...")
    cross_type = cross_type_correlation(real_pb, gen_pb)
    summary["cross_type"] = cross_type

    # Save
    with open(output_dir / "pseudobulk_validation.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "pseudobulk_per_type.json", "w") as f:
        json.dump(per_type, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("  Pseudobulk Validation Summary")
    print(f"{'='*60}")
    print(f"  Types evaluated:           {summary['n_types_evaluated']}")
    print(f"  Pearson r (mean +/- std):  {summary['pearson_mean']:.4f} +/- {summary['pearson_std']:.4f}")
    print(f"  Pearson r (median):        {summary['pearson_median']:.4f}")
    print(f"  Spearman rho (mean):       {summary['spearman_mean']:.4f}")
    print(f"  Types with r > 0.9:        {summary['pct_pearson_gt_0.9']:.1f}%")
    print(f"  Types with r > 0.8:        {summary['pct_pearson_gt_0.8']:.1f}%")
    print(f"\n  Cross-type gene correlation:")
    print(f"    Gene corr (mean):        {cross_type.get('cross_type_gene_corr_mean', 'N/A')}")
    print(f"    Genes with r > 0.5:      {cross_type.get('pct_genes_corr_gt_0.5', 'N/A')}%")
    print(f"\n  Saved to: {output_dir}/")

    # Print top/bottom 5 types
    sorted_types = sorted(per_type.items(), key=lambda x: x[1]["pearson_r"], reverse=True)
    print(f"\n  Top 5 types (highest pseudobulk correlation):")
    for name, metrics in sorted_types[:5]:
        print(f"    {name[:35]:35s}  r={metrics['pearson_r']:.4f}  (n_real={metrics['n_real']}, n_gen={metrics['n_gen']})")
    print(f"\n  Bottom 5 types (lowest pseudobulk correlation):")
    for name, metrics in sorted_types[-5:]:
        print(f"    {name[:35]:35s}  r={metrics['pearson_r']:.4f}  (n_real={metrics['n_real']}, n_gen={metrics['n_gen']})")


if __name__ == "__main__":
    main()
