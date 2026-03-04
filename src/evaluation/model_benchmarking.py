"""
model_benchmarking.py — Comprehensive model benchmarking for CLOP-DiT.

Systematically evaluates CLOP-DiT against multiple baselines across all
embedding-level metrics, collecting results into a structured report.

Baselines
---------
1. Gaussian N(μ,σ²I)  — per-type mean + isotropic noise, L2-normalised
2. Shuffled Labels     — CLOP-DiT embeddings with randomised cell-type labels
3. Random N(0,I)       — uninformative prior, L2-normalised
4. Mean-only (collapse) — per-type centroid repeated (zero intra-type diversity)

Metrics (per-method)
--------------------
  Embedding: FD, MMD, Coverage, Density, Mean KL
  Per-type:  Mean Centroid Cosine, Min Centroid Cosine
  Diversity: Diversity Ratio, Fraction Collapsed
  Bootstrap: 95 % CI for FD, Centroid Cosine, Diversity Ratio (B=200)

Output
------
  results/benchmark_report.json   — full structured report
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Bootstrap CI helper
# ─────────────────────────────────────────────────────────────
def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 200,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    return float(values.mean()), float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


# ─────────────────────────────────────────────────────────────
# Per-type evaluation helpers
# ─────────────────────────────────────────────────────────────
def _per_type_metrics(
    real_cells: np.ndarray,
    gen_cells: np.ndarray,
    real_labels: np.ndarray,
    gen_labels: np.ndarray,
    rng: np.random.Generator,
) -> Dict:
    """Compute centroid cosine and diversity ratio per type."""
    unique_types = np.sort(np.unique(real_labels))
    cosines, div_ratios, collapsed = [], [], 0

    for tid in unique_types:
        r = real_cells[real_labels == tid]
        g = gen_cells[gen_labels == tid]
        if len(r) < 5 or len(g) < 5:
            continue

        # Centroid cosine
        rc = r.mean(0)
        rc /= np.linalg.norm(rc) + 1e-8
        gc = g.mean(0)
        gc /= np.linalg.norm(gc) + 1e-8
        cos_val = float(np.dot(rc, gc))
        cosines.append(cos_val)

        # Intra-type diversity ratio
        n_sub = min(100, len(r), len(g))
        r_sub = r[rng.choice(len(r), n_sub, replace=False)]
        g_sub = g[rng.choice(len(g), n_sub, replace=False)]
        r_n = r_sub / (np.linalg.norm(r_sub, axis=1, keepdims=True) + 1e-8)
        g_n = g_sub / (np.linalg.norm(g_sub, axis=1, keepdims=True) + 1e-8)
        rr_sim = (r_n @ r_n.T)[np.triu_indices(len(r_n), k=1)].mean()
        gg_sim = (g_n @ g_n.T)[np.triu_indices(len(g_n), k=1)].mean()
        real_div = 1.0 - rr_sim
        gen_div = 1.0 - gg_sim
        if real_div > 1e-6:
            ratio = gen_div / real_div
            div_ratios.append(ratio)
            if ratio < 0.05:
                collapsed += 1
        else:
            div_ratios.append(1.0)

    cosines_arr = np.array(cosines) if cosines else np.array([0.0])
    div_arr = np.array(div_ratios) if div_ratios else np.array([0.0])
    cos_mean, cos_lo, cos_hi = _bootstrap_ci(cosines_arr)
    div_mean, div_lo, div_hi = _bootstrap_ci(div_arr)

    return {
        "mean_centroid_cosine": cos_mean,
        "centroid_cosine_ci": [cos_lo, cos_hi],
        "min_centroid_cosine": float(cosines_arr.min()),
        "diversity_ratio": div_mean,
        "diversity_ratio_ci": [div_lo, div_hi],
        "fraction_collapsed": collapsed / max(len(unique_types), 1),
        "n_types_evaluated": len(cosines),
    }


# ─────────────────────────────────────────────────────────────
# Generate baselines
# ─────────────────────────────────────────────────────────────
def _generate_baselines(
    real_cells: np.ndarray,
    real_labels: np.ndarray,
    gen_cells: np.ndarray,
    gen_labels: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return {name: (cells, labels)} for each baseline method."""
    unique_types = np.sort(np.unique(real_labels))
    n_per = len(gen_cells) // max(len(unique_types), 1)

    # 1. Gaussian N(μ,σ²I)
    gauss_cells, gauss_labels = [], []
    for tid in unique_types:
        r = real_cells[real_labels == tid]
        centroid = r.mean(axis=0)
        std_val = r.std()
        samples = rng.normal(0, std_val, size=(n_per, real_cells.shape[1])) + centroid
        norms = np.linalg.norm(samples, axis=1, keepdims=True) + 1e-8
        gauss_cells.append(samples / norms)
        gauss_labels.extend([tid] * n_per)
    gauss_cells = np.concatenate(gauss_cells, axis=0)
    gauss_labels = np.array(gauss_labels)

    # 2. Shuffled Labels
    shuf_labels = gen_labels.copy()
    rng.shuffle(shuf_labels)

    # 3. Random N(0,I)
    rand_cells = rng.standard_normal(size=gen_cells.shape)
    rand_cells /= np.linalg.norm(rand_cells, axis=1, keepdims=True) + 1e-8
    rand_labels = rng.choice(unique_types, size=len(gen_cells))

    # 4. Mean-only (collapse)
    mean_cells, mean_labels = [], []
    for tid in unique_types:
        r = real_cells[real_labels == tid]
        centroid = r.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        mean_cells.append(np.tile(centroid, (n_per, 1)))
        mean_labels.extend([tid] * n_per)
    mean_cells = np.concatenate(mean_cells, axis=0)
    mean_labels = np.array(mean_labels)

    return {
        "Gaussian N(μ,σ²I)": (gauss_cells, gauss_labels),
        "Shuffled Labels": (gen_cells, shuf_labels),
        "Random N(0,I)": (rand_cells, rand_labels),
        "Mean-only (collapse)": (mean_cells, mean_labels),
    }


# ─────────────────────────────────────────────────────────────
# Full evaluation for one method
# ─────────────────────────────────────────────────────────────
def _evaluate_method(
    real_cells: np.ndarray,
    gen_cells: np.ndarray,
    real_labels: np.ndarray,
    gen_labels: np.ndarray,
    n_sub: int = 5000,
    rng: np.random.Generator = None,
) -> Dict:
    """Run full metrics suite on a (gen_cells, gen_labels) set."""
    from src.evaluation.metrics import GenerationMetrics

    rng = rng or np.random.default_rng(42)
    n_sub = min(n_sub, len(real_cells), len(gen_cells))
    r_idx = rng.choice(len(real_cells), n_sub, replace=False)
    g_idx = rng.choice(len(gen_cells), n_sub, replace=False)
    real_sub = real_cells[r_idx]
    gen_sub = gen_cells[g_idx]

    # Overall distributional metrics
    fd = GenerationMetrics.frechet_distance(real_sub, gen_sub)
    mmd_val = GenerationMetrics.mmd(real_sub, gen_sub, kernel="rbf")
    cov_den = GenerationMetrics.coverage_and_density(real_sub, gen_sub, k=5)
    kl = GenerationMetrics.kl_per_dimension(real_sub, gen_sub)

    # Bootstrap CI for FD
    fd_boots = []
    for _ in range(50):
        ri = rng.choice(len(real_cells), n_sub, replace=True)
        gi = rng.choice(len(gen_cells), n_sub, replace=True)
        fd_boots.append(GenerationMetrics.frechet_distance(real_cells[ri], gen_cells[gi]))
    fd_arr = np.array(fd_boots)

    # Per-type metrics
    pt = _per_type_metrics(real_cells, gen_cells, real_labels, gen_labels, rng)

    return {
        "frechet_distance": float(fd),
        "fd_ci": [float(np.percentile(fd_arr, 2.5)), float(np.percentile(fd_arr, 97.5))],
        "mmd_rbf": float(mmd_val),
        "coverage": float(cov_den["coverage"]),
        "density": float(cov_den["density"]),
        "mean_kl": float(kl["mean_kl"]),
        **pt,
    }


# ─────────────────────────────────────────────────────────────
# Main benchmarking entry-point
# ─────────────────────────────────────────────────────────────
def run_benchmark(
    cache_dir: str = "data/cached_latents_v5.2",
    results_dir: str = "results",
    output_path: str = "results/benchmark_report.json",
    n_sub: int = 5000,
) -> Dict:
    """Run comprehensive benchmarking of CLOP-DiT vs all baselines.

    Returns the full benchmark report dict and saves to *output_path*.
    """
    t0 = time.time()
    cache = Path(cache_dir)
    res = Path(results_dir)

    # Load data
    cell_path = cache / "cell_embeddings_dedup_preprocessed.npy"
    gid_path = cache / "text_group_ids_dedup.npy"
    gen_path = res / "generated_embeddings.npy"
    gen_lab_path = res / "generated_labels.npy"

    required = [cell_path, gid_path, gen_path, gen_lab_path]
    for p in required:
        if not p.exists():
            logger.error(f"Missing required file: {p}")
            return {}

    real_cells = np.load(cell_path)
    real_labels = np.load(gid_path)
    gen_cells = np.load(gen_path)
    gen_labels = np.load(gen_lab_path)

    logger.info(f"Benchmark data: real={real_cells.shape}, gen={gen_cells.shape}, "
                f"types={len(np.unique(real_labels))}")

    rng = np.random.default_rng(42)

    # Evaluate CLOP-DiT
    logger.info("Evaluating CLOP-DiT...")
    clop_results = _evaluate_method(real_cells, gen_cells, real_labels, gen_labels,
                                    n_sub=n_sub, rng=rng)

    # Load expression metrics if available
    expr_path = res / "expression_metrics.json"
    if expr_path.exists():
        with open(expr_path) as f:
            expr_data = json.load(f)
        clop_results["gene_pearson_r"] = expr_data.get("gene_correlation", {}).get("pearson_r", None)
        clop_results["gene_spearman_rho"] = expr_data.get("gene_correlation", {}).get("spearman_rho", None)

    # Load downstream metrics if available
    for name, key in [
        ("clustering", "ari_gt_vs_leiden"),
        ("clustering", "nmi_gt_vs_leiden"),
        ("classifier", "discriminator_auc"),
        ("classifier", "gen_f1"),
    ]:
        ds_path = res / "downstream" / f"{name}_results.json"
        if ds_path.exists():
            with open(ds_path) as f:
                ds_data = json.load(f)
            if key in ds_data:
                clop_results[key] = ds_data[key]

    # Generate and evaluate baselines
    baselines = _generate_baselines(real_cells, real_labels, gen_cells, gen_labels, rng)
    baseline_results = {}
    for bl_name, (bl_cells, bl_labels) in baselines.items():
        logger.info(f"Evaluating baseline: {bl_name}...")
        bl_res = _evaluate_method(real_cells, bl_cells, real_labels, bl_labels,
                                  n_sub=n_sub, rng=rng)
        baseline_results[bl_name] = bl_res

    # Compute rankings for each metric
    all_methods = {"CLOP-DiT": clop_results, **baseline_results}
    ranking_metrics = [
        ("frechet_distance", "lower"),
        ("mmd_rbf", "lower"),
        ("mean_kl", "lower"),
        ("coverage", "higher"),
        ("density", "higher"),
        ("mean_centroid_cosine", "higher"),
        ("diversity_ratio", "higher"),
    ]
    rankings = {}
    for metric, direction in ranking_metrics:
        vals = {m: all_methods[m].get(metric, float("inf") if direction == "lower" else 0)
                for m in all_methods}
        sorted_methods = sorted(vals.keys(),
                                key=lambda m: vals[m],
                                reverse=(direction == "higher"))
        rankings[metric] = {
            "direction": direction,
            "ranking": sorted_methods,
            "values": {m: vals[m] for m in sorted_methods},
            "best": sorted_methods[0],
        }

    # Composite score: normalise each metric to [0, 1] and average
    composite = {}
    for method in all_methods:
        scores = []
        for metric, direction in ranking_metrics:
            vals = [all_methods[m].get(metric, 0) for m in all_methods]
            mn, mx = min(vals), max(vals)
            rng_val = max(mx - mn, 1e-8)
            raw = all_methods[method].get(metric, 0)
            if direction == "lower":
                norm = 1.0 - (raw - mn) / rng_val
            else:
                norm = (raw - mn) / rng_val
            scores.append(norm)
        composite[method] = float(np.mean(scores))

    elapsed = time.time() - t0
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_sub": n_sub,
        "data_shape": {"real": list(real_cells.shape), "gen": list(gen_cells.shape)},
        "methods": all_methods,
        "rankings": rankings,
        "composite_score": composite,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Benchmark report saved → {out}  ({elapsed:.1f}s)")

    return report


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="CLOP-DiT Model Benchmarking")
    parser.add_argument("--cache-dir", default="data/cached_latents_v5.2")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/benchmark_report.json")
    parser.add_argument("--n-sub", type=int, default=5000)
    args = parser.parse_args()

    report = run_benchmark(
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        output_path=args.output,
        n_sub=args.n_sub,
    )

    if report:
        print("\n=== Composite Scores ===")
        for method, score in sorted(report["composite_score"].items(),
                                    key=lambda x: x[1], reverse=True):
            print(f"  {method:25s}  {score:.4f}")

        print("\n=== Rankings ===")
        for metric, rdata in report["rankings"].items():
            best = rdata["best"]
            print(f"  {metric:30s}  →  {best}")
