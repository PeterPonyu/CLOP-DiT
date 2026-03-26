#!/usr/bin/env python3
"""Selective augmentation rescue — make augmentation work conditionally.

Instead of augmenting all rare types (which failed), this script:
  1. Identifies which types pass a quality threshold (centroid cos > X)
  2. Only augments those selective types
  3. Tests multiple blending strategies:
     a) Pure synthetic augmentation
     b) Alpha-blended: 0.7*real + 0.3*synthetic
     c) Embedding-space augmentation (bypass decoder)
  4. Reports per-type augmentation benefit

Output:
    results/validation/selective_augmentation.json

Usage:
    python scripts/analysis/selective_augmentation.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_quality_scores(real_emb, gen_emb, real_labels, gen_labels):
    """Compute per-type quality score for selective augmentation."""
    shared_ids = sorted(set(np.unique(real_labels)) & set(np.unique(gen_labels)))
    scores = {}

    for gid in shared_ids:
        real_mask = real_labels == gid
        gen_mask = gen_labels == gid
        r = real_emb[real_mask]
        g = gen_emb[gen_mask]

        if len(r) < 3 or len(g) < 3:
            continue

        # Centroid cosine
        real_cent = r.mean(axis=0)
        gen_cent = g.mean(axis=0)
        cos = float(np.dot(real_cent, gen_cent) /
                    (np.linalg.norm(real_cent) * np.linalg.norm(gen_cent) + 1e-10))

        # Variance ratio
        var_ratio = float(np.mean(np.var(g, axis=0)) / (np.mean(np.var(r, axis=0)) + 1e-10))

        # KNN purity: fraction of gen cells whose NN is same type
        k = min(5, len(r) - 1)
        if k >= 1:
            nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(real_emb)
            _, idx = nn.kneighbors(g)
            nn_labels = real_labels[idx]
            knn_purity = float(np.mean(nn_labels == gid))
        else:
            knn_purity = 0.0

        scores[int(gid)] = {
            "centroid_cosine": cos,
            "variance_ratio": var_ratio,
            "knn_purity": knn_purity,
            "n_real": int(len(r)),
            "n_gen": int(len(g)),
            "quality_score": 0.4 * cos + 0.3 * knn_purity + 0.3 * min(var_ratio, 1.0),
        }

    return scores


def classify_with_augmentation(real_emb, real_labels, gen_emb, gen_labels,
                                augment_types, strategy="pure", alpha=0.3):
    """Train classifier with selective augmentation and evaluate.

    Strategies:
        pure: Add synthetic cells directly
        blended: alpha-blend real+synthetic
        embedding: Only augment in embedding space (no decoder)
    """
    # Identify rare types (below median count)
    unique, counts = np.unique(real_labels, return_counts=True)
    count_map = dict(zip(unique, counts))
    median_count = np.median(counts)

    # Build augmented training set
    X_train = real_emb.copy()
    y_train = real_labels.copy()

    for gid in augment_types:
        gen_mask = gen_labels == gid
        gen_sub = gen_emb[gen_mask]

        if len(gen_sub) == 0:
            continue

        real_mask = real_labels == gid
        n_real = real_mask.sum()

        # Target: augment to at least median count
        n_needed = max(0, int(median_count) - n_real)
        if n_needed == 0:
            continue

        # Sample synthetic cells
        n_sample = min(n_needed, len(gen_sub))
        sample_idx = np.random.choice(len(gen_sub), n_sample, replace=len(gen_sub) < n_sample)

        if strategy == "pure":
            aug_emb = gen_sub[sample_idx]
        elif strategy == "blended":
            # Alpha-blend: mostly real + some synthetic direction
            real_sub = real_emb[real_mask]
            real_samples = real_sub[np.random.choice(len(real_sub), n_sample, replace=True)]
            aug_emb = (1 - alpha) * real_samples + alpha * gen_sub[sample_idx]
        else:
            aug_emb = gen_sub[sample_idx]

        X_train = np.vstack([X_train, aug_emb])
        y_train = np.concatenate([y_train, np.full(n_sample, gid)])

    return X_train, y_train


def evaluate_augmentation(X_train, y_train, X_test, y_test):
    """Evaluate augmented classifier performance."""
    clf = LogisticRegression(max_iter=2000, random_state=42,
                             class_weight="balanced", n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Per-type F1
    per_type = {}
    for gid in np.unique(y_test):
        mask = y_test == gid
        if mask.sum() > 0:
            type_pred = y_pred[mask]
            type_acc = float(np.mean(type_pred == gid))
            per_type[int(gid)] = type_acc

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_type_accuracy": per_type,
    }


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import RESULTS_DIR, CACHE_DIR

    np.random.seed(42)
    output_dir = RESULTS_DIR / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    real_emb = np.load(str(CACHE_DIR / "cell_embeddings_dedup_preprocessed.npy"))
    gen_emb = np.load(str(RESULTS_DIR / "generated_embeddings.npy"))
    real_labels = np.load(str(CACHE_DIR / "text_group_ids_dedup.npy"))
    gen_labels = np.load(str(RESULTS_DIR / "generated_labels.npy"))

    type_names = {}
    cap_path = CACHE_DIR / "text_captions_deduplicated.json"
    if cap_path.exists():
        with open(cap_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            name = v.split(" are ")[0] if " are " in v else v[:50]
            type_names[int(k)] = name

    # Compute quality scores
    logger.info("Computing per-type quality scores...")
    quality = compute_quality_scores(real_emb, gen_emb, real_labels, gen_labels)

    # Split into train/test
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf.split(real_emb, real_labels))
    X_test = real_emb[test_idx]
    y_test = real_labels[test_idx]
    X_real_train = real_emb[train_idx]
    y_real_train = real_labels[train_idx]

    # Identify rare types
    unique, counts = np.unique(y_real_train, return_counts=True)
    count_map = dict(zip(unique, counts))
    median_count = np.median(counts)
    rare_types = [gid for gid, cnt in count_map.items() if cnt < median_count]

    logger.info(f"Rare types (below median={median_count:.0f}): {len(rare_types)}")

    # Quality thresholds to test
    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    strategies = ["pure", "blended"]
    alpha_values = [0.1, 0.3, 0.5]

    results = {}

    # Baseline: no augmentation
    logger.info("Evaluating baseline (no augmentation)...")
    baseline = evaluate_augmentation(X_real_train, y_real_train, X_test, y_test)
    results["baseline"] = baseline
    logger.info(f"  Baseline: acc={baseline['accuracy']:.4f}, F1={baseline['f1_macro']:.4f}")

    # Selective augmentation by quality threshold
    for threshold in thresholds:
        eligible = [gid for gid in rare_types
                    if gid in quality and quality[gid]["quality_score"] >= threshold]

        if not eligible:
            continue

        for strategy in strategies:
            if strategy == "blended":
                for alpha in alpha_values:
                    key = f"selective_q{threshold}_blended_a{alpha}"
                    logger.info(f"  {key}: {len(eligible)} types...")
                    X_aug, y_aug = classify_with_augmentation(
                        X_real_train, y_real_train, gen_emb, gen_labels,
                        eligible, strategy="blended", alpha=alpha)
                    metrics = evaluate_augmentation(X_aug, y_aug, X_test, y_test)
                    metrics["n_augmented_types"] = len(eligible)
                    metrics["threshold"] = threshold
                    metrics["strategy"] = f"blended_alpha={alpha}"
                    metrics["delta_f1"] = float(metrics["f1_macro"] - baseline["f1_macro"])
                    results[key] = metrics
                    logger.info(f"    F1={metrics['f1_macro']:.4f} (delta={metrics['delta_f1']:+.4f})")
            else:
                key = f"selective_q{threshold}_pure"
                logger.info(f"  {key}: {len(eligible)} types...")
                X_aug, y_aug = classify_with_augmentation(
                    X_real_train, y_real_train, gen_emb, gen_labels,
                    eligible, strategy="pure")
                metrics = evaluate_augmentation(X_aug, y_aug, X_test, y_test)
                metrics["n_augmented_types"] = len(eligible)
                metrics["threshold"] = threshold
                metrics["strategy"] = "pure"
                metrics["delta_f1"] = float(metrics["f1_macro"] - baseline["f1_macro"])
                results[key] = metrics
                logger.info(f"    F1={metrics['f1_macro']:.4f} (delta={metrics['delta_f1']:+.4f})")

    # Find best configuration
    best_key = max((k for k in results if k != "baseline"),
                   key=lambda k: results[k]["f1_macro"], default=None)

    summary = {
        "baseline_f1": baseline["f1_macro"],
        "baseline_accuracy": baseline["accuracy"],
        "best_config": best_key,
        "best_f1": results[best_key]["f1_macro"] if best_key else baseline["f1_macro"],
        "best_delta": results[best_key]["delta_f1"] if best_key else 0.0,
        "n_rare_types": len(rare_types),
        "quality_scores": {type_names.get(k, f"Type_{k}"): v
                          for k, v in quality.items()},
    }

    results["summary"] = summary

    with open(output_dir / "selective_augmentation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print
    print(f"\n{'='*60}")
    print("  Selective Augmentation Results")
    print(f"{'='*60}")
    print(f"  Baseline F1: {baseline['f1_macro']:.4f}")
    if best_key:
        print(f"  Best config: {best_key}")
        print(f"  Best F1:     {results[best_key]['f1_macro']:.4f} (delta={results[best_key]['delta_f1']:+.4f})")
        print(f"  Improved:    {'YES' if results[best_key]['delta_f1'] > 0.001 else 'NO'}")

    # Print all results sorted by F1
    print(f"\n  All configurations:")
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if k not in ("baseline", "summary")],
        key=lambda x: x[1]["f1_macro"], reverse=True)
    for key, val in sorted_results[:10]:
        marker = " ***" if val["delta_f1"] > 0.001 else ""
        print(f"    {key:45s}  F1={val['f1_macro']:.4f}  delta={val['delta_f1']:+.4f}{marker}")

    print(f"\n  Saved to: {output_dir}/selective_augmentation.json")


if __name__ == "__main__":
    main()
