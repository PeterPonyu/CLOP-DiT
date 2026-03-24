"""model_benchmarking.py — Comprehensive multi-method benchmarking for CLOP-DiT.

This module evaluates CLOP-DiT, artifact-backed learned baselines, and
built-in synthetic controls under a common registry and artifact contract.
It produces a consolidated JSON report used by Panel S and downstream docs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .baseline_registry import (
    MethodSpec,
    expected_artifact_contract,
    get_method_specs,
    load_baseline_manifest,
    load_method_metadata,
)

logger = logging.getLogger(__name__)


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
    return (
        float(values.mean()),
        float(np.percentile(means, alpha * 100)),
        float(np.percentile(means, (1 - alpha) * 100)),
    )


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

    dim_mismatch = real_cells.shape[1] != gen_cells.shape[1]

    for tid in unique_types:
        r = real_cells[real_labels == tid]
        g = gen_cells[gen_labels == tid]
        if len(r) < 5 or len(g) < 5:
            continue

        # Centroid cosine: skip when dimensions differ (not comparable)
        if not dim_mismatch:
            rc = r.mean(0)
            rc /= np.linalg.norm(rc) + 1e-8
            gc = g.mean(0)
            gc /= np.linalg.norm(gc) + 1e-8
            cosines.append(float(np.dot(rc, gc)))

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


def _generate_synthetic_baselines(
    real_cells: np.ndarray,
    real_labels: np.ndarray,
    clop_cells: np.ndarray,
    clop_labels: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return built-in synthetic baselines in the benchmark registry."""
    unique_types = np.sort(np.unique(real_labels))
    n_per = len(clop_cells) // max(len(unique_types), 1)

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

    shuffled_labels = clop_labels.copy()
    rng.shuffle(shuffled_labels)

    rand_cells = rng.standard_normal(size=clop_cells.shape)
    rand_cells /= np.linalg.norm(rand_cells, axis=1, keepdims=True) + 1e-8
    rand_labels = rng.choice(unique_types, size=len(clop_cells))

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
        "Shuffled Labels": (clop_cells, shuffled_labels),
        "Random N(0,I)": (rand_cells, rand_labels),
        "Mean-only (collapse)": (mean_cells, mean_labels),
    }


def _evaluate_method(
    real_cells: np.ndarray,
    gen_cells: np.ndarray,
    real_labels: np.ndarray,
    gen_labels: np.ndarray,
    n_sub: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Run the full embedding-space metric suite on one method."""
    from src.evaluation.metrics import GenerationMetrics

    rng = rng or np.random.default_rng(42)
    n_sub = min(n_sub, len(real_cells), len(gen_cells))
    r_idx = rng.choice(len(real_cells), n_sub, replace=False)
    g_idx = rng.choice(len(gen_cells), n_sub, replace=False)
    real_sub = real_cells[r_idx]
    gen_sub = gen_cells[g_idx]

    # Handle dimension mismatch: project to common PCA space if needed
    dim_mismatch = real_sub.shape[1] != gen_sub.shape[1]
    if dim_mismatch:
        from sklearn.decomposition import PCA
        common_dim = min(real_sub.shape[1], gen_sub.shape[1], 32)
        pca = PCA(n_components=common_dim, random_state=42)
        pca.fit(real_sub)
        real_proj = pca.transform(real_sub)
        # For gen_sub with different dims, fit a separate PCA and project
        pca_gen = PCA(n_components=common_dim, random_state=42)
        gen_proj = pca_gen.fit_transform(gen_sub)
        logger.warning(
            "Dimension mismatch: real=%dD, gen=%dD → PCA projection to %dD for distributional metrics",
            real_sub.shape[1], gen_sub.shape[1], common_dim,
        )
    else:
        real_proj = real_sub
        gen_proj = gen_sub

    fd = GenerationMetrics.frechet_distance(real_proj, gen_proj)
    mmd_val = GenerationMetrics.mmd(real_proj, gen_proj, kernel="rbf")
    cov_den = GenerationMetrics.coverage_and_density(real_proj, gen_proj, k=5)
    kl = GenerationMetrics.kl_per_dimension(real_proj, gen_proj)

    fd_boots = []
    for _ in range(50):
        ri = rng.choice(len(real_cells), n_sub, replace=True)
        gi = rng.choice(len(gen_cells), n_sub, replace=True)
        r_boot = real_cells[ri]
        g_boot = gen_cells[gi]
        if dim_mismatch:
            r_boot = pca.transform(r_boot)
            g_boot = pca_gen.transform(g_boot)
        fd_boots.append(GenerationMetrics.frechet_distance(r_boot, g_boot))
    fd_arr = np.array(fd_boots)

    pt = _per_type_metrics(real_cells, gen_cells, real_labels, gen_labels, rng)
    result = {
        "frechet_distance": float(fd),
        "fd_ci": [float(np.percentile(fd_arr, 2.5)), float(np.percentile(fd_arr, 97.5))],
        "mmd_rbf": float(mmd_val),
        "coverage": float(cov_den["coverage"]),
        "density": float(cov_den["density"]),
        "mean_kl": float(kl["mean_kl"]),
        **pt,
    }
    if dim_mismatch:
        result["_dim_mismatch_note"] = (
            f"Distributional metrics computed in PCA-{common_dim} space "
            f"(real: {real_sub.shape[1]}D, gen: {gen_sub.shape[1]}D)"
        )
    return result


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_artifact_embeddings(spec: MethodSpec, results_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    emb_path = spec.embeddings_path(results_dir)
    lbl_path = spec.labels_path(results_dir)
    if not emb_path.exists() or not lbl_path.exists():
        return None
    return np.load(emb_path), np.load(lbl_path)


def _downstream_prefix_candidates(spec: MethodSpec) -> List[str]:
    if spec.root_generated:
        return ["", f"{spec.slug}_"]
    return [f"{spec.slug}_"]


def _load_downstream_bundle(spec: MethodSpec, results_dir: Path) -> Dict[str, Optional[Dict]]:
    downstream_dir = results_dir / "downstream"
    if not downstream_dir.exists():
        return {"clustering": None, "classifier": None, "de": None}

    bundle = {"clustering": None, "classifier": None, "de": None}
    for prefix in _downstream_prefix_candidates(spec):
        clust = _load_json(downstream_dir / f"{prefix}clustering_alignment.json")
        clf = _load_json(downstream_dir / f"{prefix}classifier_alignment.json")
        de = _load_json(downstream_dir / f"{prefix}de_concordance.json")
        if clust is not None:
            bundle["clustering"] = clust
        if clf is not None:
            bundle["classifier"] = clf
        if de is not None:
            bundle["de"] = de
    return bundle


def _attach_optional_metrics(metrics: Dict, spec: MethodSpec, results_dir: Path) -> None:
    expr_data = _load_json(spec.expression_metrics_path(results_dir))
    if expr_data:
        gene_corr = expr_data.get("gene_correlation", {})
        metrics["gene_pearson_r"] = gene_corr.get("pearson_r")
        metrics["gene_spearman_rho"] = gene_corr.get("spearman_rho")

    downstream = _load_downstream_bundle(spec, results_dir)
    clustering = downstream["clustering"] or {}
    classifier = downstream["classifier"] or {}
    de_data = downstream["de"] or {}

    for key in ["ari_gt_vs_leiden", "nmi_gt_vs_leiden", "mean_cluster_purity", "mean_mixing_score"]:
        if key in clustering:
            metrics[key] = clustering[key]
    for key in ["gen_accuracy", "gen_f1", "discriminator_auc"]:
        if key in classifier:
            metrics[key] = classifier[key]

    if de_data:
        pearsons = [v.get("logfc_pearson_r") for v in de_data.values() if v.get("logfc_pearson_r") is not None]
        spearmans = [v.get("logfc_spearman_rho") for v in de_data.values() if v.get("logfc_spearman_rho") is not None]
        sign_agreement = [v.get("top_k_sign_agreement") for v in de_data.values() if v.get("top_k_sign_agreement") is not None]
        jaccards = [v.get("top_k_jaccard") for v in de_data.values() if v.get("top_k_jaccard") is not None]
        if pearsons:
            metrics["de_mean_logfc_pearson"] = float(np.mean(pearsons))
        if spearmans:
            metrics["de_mean_logfc_spearman"] = float(np.mean(spearmans))
        if sign_agreement:
            metrics["de_mean_sign_agreement"] = float(np.mean(sign_agreement))
        if jaccards:
            metrics["de_mean_topk_jaccard"] = float(np.mean(jaccards))
        metrics["de_n_contrasts"] = len(de_data)


def _build_rankings(all_methods: Dict[str, Dict]) -> Tuple[Dict, Dict, Dict]:
    ranking_metrics = [
        ("frechet_distance", "lower"),
        ("mmd_rbf", "lower"),
        ("mean_kl", "lower"),
        ("coverage", "higher"),
        ("density", "higher"),
        ("mean_centroid_cosine", "higher"),
        ("min_centroid_cosine", "higher"),
        ("diversity_ratio", "higher"),
        ("fraction_collapsed", "lower"),
        ("gene_pearson_r", "higher"),
        ("gene_spearman_rho", "higher"),
        ("ari_gt_vs_leiden", "higher"),
        ("nmi_gt_vs_leiden", "higher"),
        ("gen_accuracy", "higher"),
        ("gen_f1", "higher"),
        ("de_mean_logfc_pearson", "higher"),
        ("de_mean_sign_agreement", "higher"),
    ]

    active_metrics = []
    for metric, direction in ranking_metrics:
        if any(all_methods[m].get(metric) is not None for m in all_methods):
            active_metrics.append((metric, direction))

    # Identify common metrics (present in ALL methods)
    common_metrics = [
        (metric, direction) for metric, direction in active_metrics
        if all(all_methods[m].get(metric) is not None for m in all_methods)
    ]
    exclusive_metrics = [
        (metric, _) for metric, _ in active_metrics
        if not all(all_methods[m].get(metric) is not None for m in all_methods)
    ]
    logger.info(
        f"Composite metrics: {len(active_metrics)} total, "
        f"{len(common_metrics)} common to all methods, "
        f"{len(exclusive_metrics)} method-exclusive "
        f"({[m for m, _ in exclusive_metrics]})"
    )

    rankings = {}
    for metric, direction in active_metrics:
        default = float("inf") if direction == "lower" else float("-inf")
        vals = {}
        for method_name in all_methods:
            raw = all_methods[method_name].get(metric)
            vals[method_name] = raw if raw is not None else default
        sorted_methods = sorted(
            vals.keys(),
            key=lambda name: vals[name],
            reverse=(direction == "higher"),
        )
        rankings[metric] = {
            "direction": direction,
            "ranking": sorted_methods,
            "values": {name: vals[name] for name in sorted_methods},
            "best": sorted_methods[0],
        }

    def _compute_composite(metrics_subset):
        comp = {}
        for method_name in all_methods:
            scores = []
            for metric, direction in metrics_subset:
                raw_val = all_methods[method_name].get(metric)
                if raw_val is None:
                    scores.append(0.0)
                    continue
                valid_vals = [all_methods[name].get(metric) for name in all_methods if all_methods[name].get(metric) is not None]
                if not valid_vals:
                    scores.append(0.0)
                    continue
                mn, mx = min(valid_vals), max(valid_vals)
                rng_val = mx - mn
                if rng_val < 1e-8:
                    scores.append(1.0)
                    continue
                if direction == "lower":
                    scores.append(1.0 - (raw_val - mn) / rng_val)
                else:
                    scores.append((raw_val - mn) / rng_val)
            comp[method_name] = float(np.mean(scores)) if scores else 0.0
        return comp

    composite_full = _compute_composite(active_metrics)
    composite_common = _compute_composite(common_metrics) if common_metrics else composite_full
    return rankings, composite_full, composite_common


def run_benchmark(
    cache_dir: str = "data/cached_latents",
    results_dir: str = "results",
    output_path: str = "results/benchmark_report.json",
    n_sub: int = 5000,
) -> Dict:
    """Run registry-driven benchmarking for CLOP-DiT and available baselines."""
    t0 = time.time()
    cache = Path(cache_dir)
    res = Path(results_dir)

    cell_path = cache / "cell_embeddings_dedup_preprocessed.npy"
    gid_path = cache / "text_group_ids_dedup.npy"
    clop_gen_path = res / "generated_embeddings.npy"
    clop_lab_path = res / "generated_labels.npy"

    required = [cell_path, gid_path, clop_gen_path, clop_lab_path]
    for path in required:
        if not path.exists():
            logger.error(f"Missing required file: {path}")
            return {}

    real_cells = np.load(cell_path)
    real_labels = np.load(gid_path)
    clop_cells = np.load(clop_gen_path)
    clop_labels = np.load(clop_lab_path)
    logger.info(
        f"Benchmark data: real={real_cells.shape}, clop={clop_cells.shape}, "
        f"types={len(np.unique(real_labels))}"
    )

    rng = np.random.default_rng(42)
    synthetic = _generate_synthetic_baselines(real_cells, real_labels, clop_cells, clop_labels, rng)

    methods = {}
    method_metadata = {}
    skipped_methods = {}

    for spec in get_method_specs(res):
        method_metadata[spec.display_name] = load_method_metadata(spec, res)
        logger.info(f"Evaluating method: {spec.display_name}")

        if spec.built_in:
            bundle = synthetic.get(spec.display_name)
        elif spec.root_generated:
            bundle = (clop_cells, clop_labels)
        else:
            bundle = _load_artifact_embeddings(spec, res)

        if bundle is None:
            skipped_methods[spec.display_name] = "Missing embedding artifacts"
            continue

        gen_cells, gen_labels = bundle
        metrics = _evaluate_method(
            real_cells,
            gen_cells,
            real_labels,
            gen_labels,
            n_sub=n_sub,
            rng=np.random.default_rng(42),
        )
        metrics["method_slug"] = spec.slug
        metrics["method_family"] = spec.family
        _attach_optional_metrics(metrics, spec, res)
        methods[spec.display_name] = metrics

    if not methods:
        logger.error("No methods could be evaluated")
        return {}

    rankings, composite, composite_common = _build_rankings(methods)
    elapsed = time.time() - t0
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_sub": n_sub,
        "data_shape": {"real": list(real_cells.shape), "clop": list(clop_cells.shape)},
        "artifact_contract": expected_artifact_contract(),
        "baseline_manifest": load_baseline_manifest(res),
        "method_metadata": method_metadata,
        "skipped_methods": skipped_methods,
        "methods": methods,
        "rankings": rankings,
        "composite_score": composite,
        "composite_score_common_metrics_only": composite_common,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Benchmark report saved → {out} ({elapsed:.1f}s)")
    return report


if __name__ == "__main__":
    import argparse
    from src.utils.paths import CACHE_DIR, RESULTS_DIR

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="CLOP-DiT model benchmarking")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-sub", type=int, default=5000)
    args = parser.parse_args()

    cache_dir = args.cache_dir or str(CACHE_DIR)
    results_dir = args.results_dir or str(RESULTS_DIR)
    output_path = args.output or str(Path(results_dir) / "benchmark_report.json")

    report = run_benchmark(
        cache_dir=cache_dir,
        results_dir=results_dir,
        output_path=output_path,
        n_sub=args.n_sub,
    )

    if report:
        print("\n=== Composite Scores (all metrics) ===")
        for method, score in sorted(report["composite_score"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {method:25s}  {score:.4f}")

        print("\n=== Composite Scores (common metrics only) ===")
        for method, score in sorted(report["composite_score_common_metrics_only"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {method:25s}  {score:.4f}")

        print("\n=== Rankings ===")
        for metric, rdata in report["rankings"].items():
            print(f"  {metric:30s}  →  {rdata['best']}")
