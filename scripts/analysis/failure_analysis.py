#!/usr/bin/env python3
"""Per-cell-type failure analysis — turn reviewer weakness into a contribution.

Stratifies all generation quality metrics by biological family, sample size,
and embedding overlap to identify systematic patterns in where the model
succeeds and fails. Produces a structured JSON and classification of types
into pass/warn/fail tiers.

Output:
    results/validation/failure_analysis.json
    results/validation/failure_tiers.json

Usage:
    python scripts/analysis/failure_analysis.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Biological family classification for common cell types
# This maps substring patterns to family labels
FAMILY_PATTERNS = {
    "T cell": "Immune: T cells",
    "T lymph": "Immune: T cells",
    "CD4": "Immune: T cells",
    "CD8": "Immune: T cells",
    "NK": "Immune: NK cells",
    "natural killer": "Immune: NK cells",
    "B cell": "Immune: B cells",
    "B lymph": "Immune: B cells",
    "plasma": "Immune: B cells",
    "macrophage": "Immune: Myeloid",
    "monocyte": "Immune: Myeloid",
    "dendritic": "Immune: Myeloid",
    "neutrophil": "Immune: Granulocytes",
    "eosinophil": "Immune: Granulocytes",
    "mast cell": "Immune: Granulocytes",
    "epithelial": "Epithelial",
    "endothelial": "Endothelial",
    "fibroblast": "Stromal",
    "mesenchymal": "Stromal",
    "smooth muscle": "Stromal",
    "pericyte": "Stromal",
    "neuron": "Neural",
    "astrocyte": "Neural",
    "oligodendrocyte": "Neural",
    "microglia": "Neural",
    "hepatocyte": "Parenchymal",
    "cardiomyocyte": "Parenchymal",
    "adipocyte": "Parenchymal",
    "erythrocyte": "Erythroid",
    "erythroid": "Erythroid",
    "megakaryocyte": "Megakaryocyte",
    "platelet": "Megakaryocyte",
    "stem cell": "Stem/Progenitor",
    "progenitor": "Stem/Progenitor",
    "cycling": "Cycling",
    "proliferating": "Cycling",
}


def classify_family(cell_type_name: str) -> str:
    """Classify a cell type into a biological family by name matching."""
    name_lower = cell_type_name.lower()
    for pattern, family in FAMILY_PATTERNS.items():
        if pattern.lower() in name_lower:
            return family
    return "Other"


def compute_per_type_metrics(real_emb, gen_emb, real_labels, gen_labels,
                             type_names):
    """Compute comprehensive per-type generation quality metrics."""
    unique_types = sorted(set(np.unique(real_labels)) & set(np.unique(gen_labels)))
    metrics = {}

    for gid in unique_types:
        real_mask = real_labels == gid
        gen_mask = gen_labels == gid

        real_sub = real_emb[real_mask]
        gen_sub = gen_emb[gen_mask]

        if len(gen_sub) < 3 or len(real_sub) < 3:
            continue

        name = type_names.get(int(gid), f"Type_{gid}")
        family = classify_family(name)

        # 1. Centroid cosine similarity
        real_centroid = real_sub.mean(axis=0)
        gen_centroid = gen_sub.mean(axis=0)
        centroid_cos = float(np.dot(real_centroid, gen_centroid) /
                            (np.linalg.norm(real_centroid) * np.linalg.norm(gen_centroid) + 1e-10))

        # 2. Variance ratio
        real_var = np.var(real_sub, axis=0)
        gen_var = np.var(gen_sub, axis=0)
        var_ratio = float(np.mean(gen_var) / (np.mean(real_var) + 1e-10))
        var_corr = float(np.corrcoef(real_var, gen_var)[0, 1]) if np.std(real_var) > 1e-10 else 0.0

        # 3. KNN accuracy: fraction of generated cells whose NN is same type
        k = min(5, len(real_sub) - 1)
        if k < 1:
            knn_accuracy = 0.0
        else:
            nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(real_emb)
            dists, indices = nn.kneighbors(gen_sub)
            nn_labels = real_labels[indices]
            knn_accuracy = float(np.mean(nn_labels == gid))

        # 4. Per-dimension variance correlation
        if np.std(real_var) > 1e-10 and np.std(gen_var) > 1e-10:
            dim_var_corr = float(stats.pearsonr(real_var, gen_var)[0])
        else:
            dim_var_corr = 0.0

        # 5. Intra-type diversity ratio
        real_pairwise = np.mean(pairwise_distances(real_sub[:min(200, len(real_sub))],
                                                    metric="cosine"))
        gen_pairwise = np.mean(pairwise_distances(gen_sub[:min(200, len(gen_sub))],
                                                   metric="cosine"))
        diversity_ratio = float(gen_pairwise / (real_pairwise + 1e-10))

        # 6. Embedding overlap with other types (confusability)
        # How close is this type's centroid to other centroids?
        other_centroids = []
        for other_gid in unique_types:
            if other_gid == gid:
                continue
            other_mask = real_labels == other_gid
            if other_mask.sum() < 3:
                continue
            other_centroids.append(real_emb[other_mask].mean(axis=0))

        if other_centroids:
            other_centroids = np.stack(other_centroids)
            cos_to_others = np.dot(other_centroids, real_centroid) / (
                np.linalg.norm(other_centroids, axis=1) * np.linalg.norm(real_centroid) + 1e-10)
            max_overlap = float(np.max(cos_to_others))
            mean_overlap = float(np.mean(cos_to_others))
        else:
            max_overlap = 0.0
            mean_overlap = 0.0

        metrics[name] = {
            "type_id": int(gid),
            "family": family,
            "n_real": int(real_mask.sum()),
            "n_gen": int(gen_mask.sum()),
            "centroid_cosine": centroid_cos,
            "variance_ratio": var_ratio,
            "variance_correlation": var_corr,
            "dim_variance_correlation": dim_var_corr,
            "knn_accuracy": knn_accuracy,
            "diversity_ratio": diversity_ratio,
            "max_overlap_with_other_type": max_overlap,
            "mean_overlap_with_other_types": mean_overlap,
        }

    return metrics


def classify_tiers(metrics: dict, thresholds: dict = None) -> dict:
    """Classify each type into pass/warn/fail tiers based on metrics."""
    thresholds = thresholds or {
        "centroid_cosine": {"pass": 0.90, "warn": 0.75},
        "knn_accuracy": {"pass": 0.50, "warn": 0.25},
        "variance_ratio": {"pass_range": (0.5, 1.5), "warn_range": (0.3, 2.0)},
        "diversity_ratio": {"pass_range": (0.5, 1.5), "warn_range": (0.3, 2.0)},
    }

    tiers = {}
    for name, m in metrics.items():
        scores = []

        # Centroid cosine
        cos_val = m["centroid_cosine"]
        if cos_val >= thresholds["centroid_cosine"]["pass"]:
            scores.append(2)
        elif cos_val >= thresholds["centroid_cosine"]["warn"]:
            scores.append(1)
        else:
            scores.append(0)

        # KNN accuracy
        knn_val = m["knn_accuracy"]
        if knn_val >= thresholds["knn_accuracy"]["pass"]:
            scores.append(2)
        elif knn_val >= thresholds["knn_accuracy"]["warn"]:
            scores.append(1)
        else:
            scores.append(0)

        # Variance ratio (should be near 1.0)
        vr = m["variance_ratio"]
        pr = thresholds["variance_ratio"]["pass_range"]
        wr = thresholds["variance_ratio"]["warn_range"]
        if pr[0] <= vr <= pr[1]:
            scores.append(2)
        elif wr[0] <= vr <= wr[1]:
            scores.append(1)
        else:
            scores.append(0)

        # Diversity ratio
        dr = m["diversity_ratio"]
        pr_d = thresholds["diversity_ratio"]["pass_range"]
        wr_d = thresholds["diversity_ratio"]["warn_range"]
        if pr_d[0] <= dr <= pr_d[1]:
            scores.append(2)
        elif wr_d[0] <= dr <= wr_d[1]:
            scores.append(1)
        else:
            scores.append(0)

        total = sum(scores)
        if total >= 7:
            tier = "pass"
        elif total >= 4:
            tier = "warn"
        else:
            tier = "fail"

        tiers[name] = {
            "tier": tier,
            "score": total,
            "max_score": 8,
            "family": m["family"],
            "n_real": m["n_real"],
        }

    return tiers


def analyze_failure_patterns(metrics: dict, tiers: dict) -> dict:
    """Analyze systematic patterns in what drives failures."""
    patterns = {}

    # 1. By biological family
    family_scores = {}
    for name, t in tiers.items():
        family = t["family"]
        if family not in family_scores:
            family_scores[family] = {"scores": [], "tiers": []}
        family_scores[family]["scores"].append(t["score"])
        family_scores[family]["tiers"].append(t["tier"])

    family_analysis = {}
    for family, data in sorted(family_scores.items()):
        n_types = len(data["scores"])
        tier_counts = {t: data["tiers"].count(t) for t in ["pass", "warn", "fail"]}
        family_analysis[family] = {
            "n_types": n_types,
            "mean_score": float(np.mean(data["scores"])),
            "tier_counts": tier_counts,
            "pass_rate": float(tier_counts.get("pass", 0) / n_types) if n_types > 0 else 0,
        }
    patterns["by_family"] = family_analysis

    # 2. By sample size (binned)
    n_reals = [metrics[name]["n_real"] for name in tiers]
    scores = [tiers[name]["score"] for name in tiers]
    bins = [(0, 100, "tiny"), (100, 500, "small"), (500, 2000, "medium"),
            (2000, 10000, "large"), (10000, float("inf"), "very_large")]

    size_analysis = {}
    for lo, hi, label in bins:
        bin_scores = [s for n, s in zip(n_reals, scores) if lo <= n < hi]
        if bin_scores:
            size_analysis[label] = {
                "n_types": len(bin_scores),
                "mean_score": float(np.mean(bin_scores)),
                "score_range": f"[{lo}, {hi})" if hi != float("inf") else f"[{lo}, inf)",
            }
    patterns["by_sample_size"] = size_analysis

    # 3. Sample size -> score correlation
    if len(n_reals) >= 5:
        log_n = np.log10(np.array(n_reals) + 1)
        r, p = stats.spearmanr(log_n, scores)
        patterns["sample_size_correlation"] = {
            "spearman_rho": float(r),
            "p_value": float(p),
        }

    # 4. Overlap -> score correlation
    overlaps = [metrics[name]["max_overlap_with_other_type"] for name in tiers]
    if len(overlaps) >= 5:
        r, p = stats.spearmanr(overlaps, scores)
        patterns["overlap_correlation"] = {
            "spearman_rho": float(r),
            "p_value": float(p),
            "interpretation": "negative = higher overlap (confusability) leads to lower scores"
        }

    return patterns


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import RESULTS_DIR, CACHE_DIR

    output_dir = RESULTS_DIR / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings and labels
    logger.info("Loading data...")
    cache_dir = CACHE_DIR
    real_emb = np.load(str(cache_dir / "cell_embeddings_dedup_preprocessed.npy"))
    gen_emb = np.load(str(RESULTS_DIR / "generated_embeddings.npy"))
    real_labels = np.load(str(cache_dir / "text_group_ids_dedup.npy"))
    gen_labels = np.load(str(RESULTS_DIR / "generated_labels.npy"))

    type_names = {}
    cap_path = cache_dir / "text_captions_deduplicated.json"
    if cap_path.exists():
        with open(cap_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            name = v.split(" are ")[0] if " are " in v else v[:50]
            type_names[int(k)] = name

    logger.info(f"Real: {real_emb.shape}, Gen: {gen_emb.shape}")

    # Compute per-type metrics
    logger.info("Computing per-type metrics...")
    metrics = compute_per_type_metrics(real_emb, gen_emb, real_labels,
                                       gen_labels, type_names)

    # Classify tiers
    logger.info("Classifying failure tiers...")
    tiers = classify_tiers(metrics)

    # Analyze patterns
    logger.info("Analyzing failure patterns...")
    patterns = analyze_failure_patterns(metrics, tiers)

    # Summary
    tier_counts = {"pass": 0, "warn": 0, "fail": 0}
    for t in tiers.values():
        tier_counts[t["tier"]] += 1

    summary = {
        "n_types_evaluated": len(metrics),
        "tier_counts": tier_counts,
        "pass_rate": float(tier_counts["pass"] / len(tiers)) if tiers else 0,
        "patterns": patterns,
    }

    # Save
    with open(output_dir / "failure_analysis.json", "w") as f:
        json.dump({
            "summary": summary,
            "per_type_metrics": {k: v for k, v in metrics.items()},
        }, f, indent=2)

    with open(output_dir / "failure_tiers.json", "w") as f:
        json.dump(tiers, f, indent=2)

    # Print
    print(f"\n{'='*60}")
    print("  Per-Cell-Type Failure Analysis")
    print(f"{'='*60}")
    print(f"  Types evaluated: {len(metrics)}")
    print(f"  Pass: {tier_counts['pass']}  Warn: {tier_counts['warn']}  Fail: {tier_counts['fail']}")
    print(f"  Pass rate: {summary['pass_rate']:.1%}")

    print(f"\n  By Biological Family:")
    for family, data in sorted(patterns["by_family"].items(),
                                key=lambda x: x[1]["mean_score"], reverse=True):
        print(f"    {family:30s}  n={data['n_types']:2d}  "
              f"score={data['mean_score']:.1f}/8  "
              f"pass_rate={data['pass_rate']:.0%}")

    print(f"\n  By Sample Size:")
    for label, data in patterns["by_sample_size"].items():
        print(f"    {label:15s}  n={data['n_types']:2d}  score={data['mean_score']:.1f}/8")

    if "sample_size_correlation" in patterns:
        sc = patterns["sample_size_correlation"]
        print(f"\n  Sample size -> score: rho={sc['spearman_rho']:.3f} (p={sc['p_value']:.4f})")

    if "overlap_correlation" in patterns:
        oc = patterns["overlap_correlation"]
        print(f"  Embedding overlap -> score: rho={oc['spearman_rho']:.3f} (p={oc['p_value']:.4f})")

    # Print top failures
    sorted_tiers = sorted(tiers.items(), key=lambda x: x[1]["score"])
    print(f"\n  Worst-performing types:")
    for name, t in sorted_tiers[:8]:
        print(f"    [{t['tier']:4s}] {name[:40]:40s}  score={t['score']}/8  "
              f"family={t['family']}  n={t['n_real']}")

    print(f"\n  Best-performing types:")
    for name, t in sorted_tiers[-5:]:
        print(f"    [{t['tier']:4s}] {name[:40]:40s}  score={t['score']}/8  "
              f"family={t['family']}  n={t['n_real']}")

    print(f"\n  Saved to: {output_dir}/")


if __name__ == "__main__":
    main()
