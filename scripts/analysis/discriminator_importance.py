#!/usr/bin/env python3
"""Discriminator feature importance — what drives real/generated separability?

Instead of just reporting AUC=0.656, this script analyzes WHAT the
discriminator detects. Uses permutation importance to identify which
embedding dimensions or gene programs drive separability.

Output:
    results/validation/discriminator_analysis.json

Usage:
    python scripts/analysis/discriminator_importance.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_discriminator(real_emb, gen_emb):
    """Train a real-vs-generated discriminator and return model + data."""
    X = np.vstack([real_emb, gen_emb])
    y = np.concatenate([np.zeros(len(real_emb)), np.ones(len(gen_emb))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    clf.fit(X_train, y_train)

    auc_train = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    auc_test = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    return clf, X_train, X_test, y_train, y_test, auc_train, auc_test


def analyze_coefficient_patterns(clf, n_dims):
    """Analyze discriminator coefficients to understand separability."""
    coefs = clf.coef_[0]  # (D,)

    # Top positive = dimensions where generated > real
    # Top negative = dimensions where real > generated
    sorted_idx = np.argsort(coefs)
    top_gen_dims = sorted_idx[-20:][::-1]  # generated-favoring dimensions
    top_real_dims = sorted_idx[:20]         # real-favoring dimensions

    # Coefficient magnitude distribution
    abs_coefs = np.abs(coefs)
    concentration = {
        "top10_fraction": float(np.sum(np.sort(abs_coefs)[-10:]) / np.sum(abs_coefs)),
        "top20_fraction": float(np.sum(np.sort(abs_coefs)[-20:]) / np.sum(abs_coefs)),
        "top50_fraction": float(np.sum(np.sort(abs_coefs)[-50:]) / np.sum(abs_coefs)),
        "gini": float(gini_coefficient(abs_coefs)),
    }

    return {
        "top_generated_dims": top_gen_dims.tolist(),
        "top_real_dims": top_real_dims.tolist(),
        "coef_stats": {
            "mean_abs": float(np.mean(abs_coefs)),
            "std_abs": float(np.std(abs_coefs)),
            "max_abs": float(np.max(abs_coefs)),
            "n_nonzero": int(np.sum(abs_coefs > 1e-6)),
        },
        "concentration": concentration,
    }


def gini_coefficient(values):
    """Compute Gini coefficient (0 = equal, 1 = concentrated)."""
    sorted_vals = np.sort(np.abs(values))
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * sorted_vals)) / (n * np.sum(sorted_vals) + 1e-10))


def analyze_variance_vs_mean(real_emb, gen_emb, coefs):
    """Check whether separability is driven by mean shift or variance difference."""
    real_mean = real_emb.mean(axis=0)
    gen_mean = gen_emb.mean(axis=0)
    real_var = real_emb.var(axis=0)
    gen_var = gen_emb.var(axis=0)

    mean_diff = gen_mean - real_mean
    var_diff = gen_var - real_var

    abs_coefs = np.abs(coefs)

    # Correlation between coefficient magnitude and mean/variance differences
    r_mean, p_mean = stats.pearsonr(abs_coefs, np.abs(mean_diff))
    r_var, p_var = stats.pearsonr(abs_coefs, np.abs(var_diff))

    return {
        "coef_vs_mean_diff": {
            "pearson_r": float(r_mean),
            "p_value": float(p_mean),
            "interpretation": "high = discriminator uses mean shifts"
        },
        "coef_vs_var_diff": {
            "pearson_r": float(r_var),
            "p_value": float(p_var),
            "interpretation": "high = discriminator uses variance differences"
        },
        "mean_shift_magnitude": float(np.linalg.norm(mean_diff)),
        "var_shift_magnitude": float(np.linalg.norm(var_diff)),
        "mean_diff_top10_dims": np.argsort(np.abs(mean_diff))[-10:][::-1].tolist(),
        "var_diff_top10_dims": np.argsort(np.abs(var_diff))[-10:][::-1].tolist(),
    }


def per_type_discriminator(real_emb, gen_emb, real_labels, gen_labels, type_names):
    """Run discriminator per cell type to find which types are most separable."""
    shared_ids = sorted(set(np.unique(real_labels)) & set(np.unique(gen_labels)))
    per_type = {}

    for gid in shared_ids:
        real_mask = real_labels == gid
        gen_mask = gen_labels == gid
        r_sub = real_emb[real_mask]
        g_sub = gen_emb[gen_mask]

        if len(r_sub) < 10 or len(g_sub) < 10:
            continue

        X = np.vstack([r_sub, g_sub])
        y = np.concatenate([np.zeros(len(r_sub)), np.ones(len(g_sub))])

        clf = LogisticRegression(max_iter=1000, random_state=42)
        try:
            proba = cross_val_predict(clf, X, y, cv=min(5, min(len(r_sub), len(g_sub))),
                                       method="predict_proba")[:, 1]
            auc = roc_auc_score(y, proba)
        except Exception:
            auc = 0.5

        name = type_names.get(int(gid), f"Type_{gid}")
        per_type[name] = {
            "auc": float(auc),
            "n_real": int(len(r_sub)),
            "n_gen": int(len(g_sub)),
            "separability": "high" if auc > 0.7 else ("medium" if auc > 0.6 else "low"),
        }

    return per_type


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import RESULTS_DIR, CACHE_DIR

    output_dir = RESULTS_DIR / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading embeddings...")
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

    # Train global discriminator
    logger.info("Training global discriminator...")
    clf, X_train, X_test, y_train, y_test, auc_train, auc_test = train_discriminator(
        real_emb, gen_emb)
    logger.info(f"Discriminator AUC: train={auc_train:.4f}, test={auc_test:.4f}")

    # Analyze what drives separability
    logger.info("Analyzing coefficient patterns...")
    coef_analysis = analyze_coefficient_patterns(clf, real_emb.shape[1])

    logger.info("Analyzing mean vs variance contributions...")
    mv_analysis = analyze_variance_vs_mean(real_emb, gen_emb, clf.coef_[0])

    # Permutation importance (slower but more reliable)
    logger.info("Computing permutation importance (may take a minute)...")
    perm_imp = permutation_importance(clf, X_test, y_test, n_repeats=10,
                                       random_state=42, n_jobs=-1)
    top_perm_dims = np.argsort(perm_imp.importances_mean)[-20:][::-1]
    perm_analysis = {
        "top20_dims": top_perm_dims.tolist(),
        "top20_importance": perm_imp.importances_mean[top_perm_dims].tolist(),
        "top10_fraction": float(
            np.sum(np.sort(perm_imp.importances_mean)[-10:]) /
            np.sum(np.maximum(perm_imp.importances_mean, 0) + 1e-10)),
    }

    # Per-type discriminator
    logger.info("Running per-type discriminator analysis...")
    per_type = per_type_discriminator(real_emb, gen_emb, real_labels,
                                      gen_labels, type_names)

    # Sort by AUC
    sorted_types = sorted(per_type.items(), key=lambda x: x[1]["auc"], reverse=True)
    per_type_auc_values = [v["auc"] for v in per_type.values()]

    # Overall summary
    results = {
        "global_discriminator": {
            "auc_train": float(auc_train),
            "auc_test": float(auc_test),
        },
        "coefficient_analysis": coef_analysis,
        "mean_vs_variance": mv_analysis,
        "permutation_importance": perm_analysis,
        "per_type_summary": {
            "n_types": len(per_type),
            "auc_mean": float(np.mean(per_type_auc_values)),
            "auc_median": float(np.median(per_type_auc_values)),
            "n_high_separability": sum(1 for v in per_type.values() if v["separability"] == "high"),
            "n_low_separability": sum(1 for v in per_type.values() if v["separability"] == "low"),
        },
        "per_type": per_type,
    }

    with open(output_dir / "discriminator_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("  Discriminator Feature Importance Analysis")
    print(f"{'='*60}")
    print(f"  Global AUC (test): {auc_test:.4f}")
    print(f"\n  What drives separability:")
    print(f"    Coef vs mean diff:     r={mv_analysis['coef_vs_mean_diff']['pearson_r']:.3f}")
    print(f"    Coef vs variance diff: r={mv_analysis['coef_vs_var_diff']['pearson_r']:.3f}")
    print(f"    → Separability is {'mostly mean-driven' if abs(mv_analysis['coef_vs_mean_diff']['pearson_r']) > abs(mv_analysis['coef_vs_var_diff']['pearson_r']) else 'mostly variance-driven'}")
    print(f"\n  Coefficient concentration (Gini): {coef_analysis['concentration']['gini']:.3f}")
    print(f"    Top 10 dims explain: {coef_analysis['concentration']['top10_fraction']:.1%} of total")
    print(f"    Top 20 dims explain: {coef_analysis['concentration']['top20_fraction']:.1%} of total")

    print(f"\n  Per-Type Separability:")
    print(f"    Mean AUC:            {results['per_type_summary']['auc_mean']:.4f}")
    print(f"    High separability:   {results['per_type_summary']['n_high_separability']}")
    print(f"    Low separability:    {results['per_type_summary']['n_low_separability']}")

    print(f"\n  Most separable types (hardest to make realistic):")
    for name, data in sorted_types[:5]:
        print(f"    {name[:40]:40s}  AUC={data['auc']:.4f}")

    print(f"\n  Least separable types (most realistic):")
    for name, data in sorted_types[-5:]:
        print(f"    {name[:40]:40s}  AUC={data['auc']:.4f}")

    print(f"\n  Saved to: {output_dir}/discriminator_analysis.json")


if __name__ == "__main__":
    main()
