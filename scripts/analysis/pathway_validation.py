#!/usr/bin/env python3
"""Pathway enrichment consistency — validate biology beyond individual genes.

For each cell type, runs gene set enrichment analysis (ORA) on the top
differentially expressed genes from real vs generated data, then checks
whether the same biological pathways are enriched. This provides evidence
that generated cells preserve pathway-level biology, not just gene-level
statistics.

Requires: gseapy (pip install gseapy)

Output:
    results/validation/pathway_concordance.json
    results/validation/pathway_per_type.json

Usage:
    python scripts/analysis/pathway_validation.py
    python scripts/analysis/pathway_validation.py --gene-sets GO_Biological_Process_2021
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_de_genes_per_type(expression, labels, gene_names, type_names,
                          n_top=200, min_cells=10):
    """For each type, compute DE genes (type vs rest) via simple t-test ranking."""
    unique_types = sorted(np.unique(labels))
    de_results = {}

    for gid in unique_types:
        mask = labels == gid
        if mask.sum() < min_cells:
            continue

        type_expr = expression[mask]
        rest_expr = expression[~mask]

        # Fast t-test per gene
        n_genes = expression.shape[1]
        logfc = np.zeros(n_genes)
        pvals = np.ones(n_genes)

        type_mean = type_expr.mean(axis=0)
        rest_mean = rest_expr.mean(axis=0)
        logfc = type_mean - rest_mean  # already in log-space from scGPT

        for g in range(n_genes):
            type_g = type_expr[:, g]
            rest_g = rest_expr[:, g]
            if np.std(type_g) < 1e-10 and np.std(rest_g) < 1e-10:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, pvals[g] = stats.mannwhitneyu(type_g, rest_g, alternative="two-sided")

        # Top upregulated genes
        up_idx = np.argsort(-logfc)[:n_top]
        up_genes = [gene_names[i] for i in up_idx if logfc[i] > 0]

        name = type_names.get(int(gid), f"Type_{gid}")
        de_results[name] = {
            "type_id": int(gid),
            "n_cells": int(mask.sum()),
            "up_genes": up_genes[:n_top],
            "logfc": {gene_names[i]: float(logfc[i]) for i in up_idx[:n_top]},
        }

    return de_results


def run_enrichment(gene_list, gene_sets="GO_Biological_Process_2021",
                   organism="human"):
    """Run over-representation analysis using gseapy/Enrichr."""
    try:
        import gseapy as gp
    except ImportError:
        logger.warning("gseapy not installed. Install with: pip install gseapy")
        return None

    if len(gene_list) < 5:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enr = gp.enrichr(
                gene_list=gene_list[:200],
                gene_sets=gene_sets,
                organism=organism,
                outdir=None,
                no_plot=True,
                verbose=False,
            )
        if enr.results is not None and len(enr.results) > 0:
            sig = enr.results[enr.results["Adjusted P-value"] < 0.05]
            return {
                "n_significant": len(sig),
                "top_terms": sig.head(20)["Term"].tolist() if len(sig) > 0 else [],
                "top_pvals": sig.head(20)["Adjusted P-value"].tolist() if len(sig) > 0 else [],
            }
    except Exception as e:
        logger.debug(f"Enrichment failed: {e}")

    return None


def compare_enrichments(real_enr, gen_enr):
    """Compare enrichment results between real and generated."""
    if real_enr is None or gen_enr is None:
        return None

    real_terms = set(real_enr.get("top_terms", []))
    gen_terms = set(gen_enr.get("top_terms", []))

    if not real_terms and not gen_terms:
        return None

    union = real_terms | gen_terms
    intersection = real_terms & gen_terms
    jaccard = len(intersection) / len(union) if union else 0

    return {
        "n_real_terms": len(real_terms),
        "n_gen_terms": len(gen_terms),
        "n_shared_terms": len(intersection),
        "jaccard": float(jaccard),
        "shared_terms": sorted(intersection),
    }


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.paths import RESULTS_DIR, CACHE_DIR

    parser = argparse.ArgumentParser(description="Pathway enrichment validation")
    parser.add_argument("--gene-sets", default="GO_Biological_Process_2021",
                        help="Enrichr gene set library name")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-top-genes", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or str(RESULTS_DIR / "validation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading expression data...")
    real_expr = np.load(str(RESULTS_DIR / "real_expression.npy")).astype(np.float32)
    gen_expr = np.load(str(RESULTS_DIR / "generated_expression.npy")).astype(np.float32)
    real_labels = np.load(str(RESULTS_DIR / "real_expression_labels.npy"))
    gen_labels = np.load(str(RESULTS_DIR / "generated_expression_labels.npy"))

    with open(str(RESULTS_DIR / "expression_gene_names.json")) as f:
        gene_names = json.load(f)

    type_names = {}
    cap_path = CACHE_DIR / "text_captions_deduplicated.json"
    if cap_path.exists():
        with open(cap_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            name = v.split(" are ")[0] if " are " in v else v[:50]
            type_names[int(k)] = name

    # Compute DE genes for each type
    logger.info("Computing DE genes (real)...")
    real_de = get_de_genes_per_type(real_expr, real_labels, gene_names,
                                     type_names, n_top=args.n_top_genes)
    logger.info("Computing DE genes (generated)...")
    gen_de = get_de_genes_per_type(gen_expr, gen_labels, gene_names,
                                   type_names, n_top=args.n_top_genes)

    # Run enrichment for shared types
    shared_types = sorted(set(real_de.keys()) & set(gen_de.keys()))
    logger.info(f"Running pathway enrichment for {len(shared_types)} shared types...")

    per_type = {}
    jaccard_values = []

    for ct in shared_types:
        logger.info(f"  Enrichment: {ct[:40]}...")
        real_enr = run_enrichment(real_de[ct]["up_genes"], gene_sets=args.gene_sets)
        gen_enr = run_enrichment(gen_de[ct]["up_genes"], gene_sets=args.gene_sets)

        comparison = compare_enrichments(real_enr, gen_enr)

        # Also compute DE gene overlap directly
        real_genes = set(real_de[ct]["up_genes"][:100])
        gen_genes = set(gen_de[ct]["up_genes"][:100])
        de_jaccard = len(real_genes & gen_genes) / len(real_genes | gen_genes) if (real_genes | gen_genes) else 0

        # LogFC correlation for shared genes
        shared_de_genes = set(real_de[ct]["logfc"].keys()) & set(gen_de[ct]["logfc"].keys())
        if len(shared_de_genes) >= 10:
            real_lfc = np.array([real_de[ct]["logfc"][g] for g in shared_de_genes])
            gen_lfc = np.array([gen_de[ct]["logfc"][g] for g in shared_de_genes])
            lfc_corr = float(stats.pearsonr(real_lfc, gen_lfc)[0])
            sign_agree = float(np.mean(np.sign(real_lfc) == np.sign(gen_lfc)))
        else:
            lfc_corr = None
            sign_agree = None

        entry = {
            "de_gene_overlap_jaccard": float(de_jaccard),
            "logfc_correlation": lfc_corr,
            "sign_agreement": sign_agree,
            "n_real_de_genes": len(real_de[ct]["up_genes"]),
            "n_gen_de_genes": len(gen_de[ct]["up_genes"]),
        }
        if comparison is not None:
            entry["pathway"] = comparison
            jaccard_values.append(comparison["jaccard"])
        if real_enr:
            entry["real_enrichment"] = real_enr
        if gen_enr:
            entry["gen_enrichment"] = gen_enr

        per_type[ct] = entry

    # Summary
    all_de_jaccards = [v["de_gene_overlap_jaccard"] for v in per_type.values()]
    all_lfc_corrs = [v["logfc_correlation"] for v in per_type.values()
                     if v["logfc_correlation"] is not None]

    summary = {
        "n_types_evaluated": len(per_type),
        "gene_sets_library": args.gene_sets,
        "de_gene_overlap": {
            "jaccard_mean": float(np.mean(all_de_jaccards)),
            "jaccard_median": float(np.median(all_de_jaccards)),
            "jaccard_std": float(np.std(all_de_jaccards)),
        },
        "logfc_correlation": {
            "mean": float(np.mean(all_lfc_corrs)) if all_lfc_corrs else None,
            "median": float(np.median(all_lfc_corrs)) if all_lfc_corrs else None,
            "std": float(np.std(all_lfc_corrs)) if all_lfc_corrs else None,
        },
    }
    if jaccard_values:
        summary["pathway_concordance"] = {
            "jaccard_mean": float(np.mean(jaccard_values)),
            "jaccard_median": float(np.median(jaccard_values)),
            "jaccard_std": float(np.std(jaccard_values)),
            "n_types_with_enrichment": len(jaccard_values),
        }

    with open(output_dir / "pathway_concordance.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "pathway_per_type.json", "w") as f:
        json.dump(per_type, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("  Pathway Enrichment Concordance")
    print(f"{'='*60}")
    print(f"  Types evaluated:       {summary['n_types_evaluated']}")
    print(f"  DE gene Jaccard:       {summary['de_gene_overlap']['jaccard_mean']:.4f} "
          f"+/- {summary['de_gene_overlap']['jaccard_std']:.4f}")
    if all_lfc_corrs:
        print(f"  LogFC correlation:     {summary['logfc_correlation']['mean']:.4f} "
              f"+/- {summary['logfc_correlation']['std']:.4f}")
    if "pathway_concordance" in summary:
        pc = summary["pathway_concordance"]
        print(f"  Pathway Jaccard:       {pc['jaccard_mean']:.4f} +/- {pc['jaccard_std']:.4f}")
        print(f"  Types with enrichment: {pc['n_types_with_enrichment']}")
    print(f"\n  Saved to: {output_dir}/")


if __name__ == "__main__":
    main()
