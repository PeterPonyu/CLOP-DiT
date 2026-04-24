#!/usr/bin/env python3
"""CellxGene Census tissue ingester for out-of-distribution evaluation.

Pulls healthy, normal-tissue cells from CellxGene Census for a target tissue
(kidney / testis / cerebellum). Writes a single h5ad in the shape the
existing data-preparation pipeline expects: raw counts present, gene symbols,
and cell_type labels. Downstream QC, scGPT caching, and evaluation can then
reuse the existing data-preparation scripts.

Usage:
    python scripts/data_prep/04_cellxgene_census_ingest.py \\
        --tissue kidney --max-cells 80000 \\
        --output data/processed_h5ad/cellxgene_census_kidney.h5ad

Design choices:
    - Census version pinned (stable snapshot) for reproducibility.
    - Healthy-only: excludes any disease ontology term != 'normal'.
    - Gene symbols only (drops cells/genes with Ensembl IDs in feature_name).
    - Saves raw counts as .X and also preserves .layers['counts'] so the
      existing preprocess_adata() step works unchanged.
    - obs columns: cell_type, tissue, organism, dataset_id, donor_id.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

CENSUS_VERSION = "2024-07-01"  # pinned LTS compatible with tiledbsoma 2.3.0


def fetch_tissue(tissue: str, organism: str, max_cells: int, seed: int) -> "anndata.AnnData":
    import cellxgene_census

    obs_filter = (
        f"tissue_general == '{tissue}' "
        f"and is_primary_data == True "
        f"and disease == 'normal' "
        f"and suspension_type in ['cell', 'nucleus']"
    )
    logger.info(f"Opening Census {CENSUS_VERSION} for {organism}, tissue_general={tissue!r}")
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        adata = cellxgene_census.get_anndata(
            census=census,
            organism=organism,
            obs_value_filter=obs_filter,
            obs_column_names=[
                "soma_joinid",
                "cell_type",
                "cell_type_ontology_term_id",
                "tissue",
                "tissue_general",
                "assay",
                "disease",
                "dataset_id",
                "donor_id",
                "sex",
                "suspension_type",
                "development_stage",
            ],
        )
    logger.info(f"Census returned {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    if adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
        idx.sort()
        adata = adata[idx].copy()
        logger.info(f"Subsampled to {adata.n_obs:,} cells (seed={seed})")

    return adata


def enforce_schema(adata, dataset_id: str) -> "anndata.AnnData":
    import anndata as ad

    # Census returns Ensembl IDs as var.index and symbols in var['feature_name'].
    # scGPT needs HGNC/MGI gene symbols — swap if needed.
    if "feature_name" in adata.var.columns:
        mask = adata.var["feature_name"].notna() & (adata.var["feature_name"] != "")
        if not mask.all():
            logger.warning(f"Dropping {(~mask).sum()} genes without feature_name")
            adata = adata[:, mask].copy()
        adata.var_names = adata.var["feature_name"].astype(str).values
        adata.var_names_make_unique()

    # Ensure counts are float32 sparse
    if sp.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)
    else:
        adata.X = sp.csr_matrix(adata.X.astype(np.float32))

    # Mirror the v0.3 pipeline contract: stash raw in layers['counts']
    adata.layers["counts"] = adata.X.copy()

    # Add dataset_id column used by downstream logging
    adata.obs["clopdit_dataset_id"] = dataset_id
    adata.uns["clopdit_dataset_id"] = dataset_id
    adata.uns["clopdit_source"] = f"CellxGene Census {CENSUS_VERSION}"

    return adata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tissue", required=True,
                        choices=["kidney", "testis", "cerebellum"],
                        help="Target tissue (tissue_general in Census schema)")
    parser.add_argument("--organism", default="Homo sapiens",
                        help="Organism (default: Homo sapiens)")
    parser.add_argument("--max-cells", type=int, default=80_000,
                        help="Subsample cap (default: 80000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True,
                        help="Output h5ad path")
    parser.add_argument("--dataset-id", default=None,
                        help="Override dataset_id tag (default: census_<tissue>)")
    args = parser.parse_args()

    setup_logging()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_id = args.dataset_id or f"census_{args.tissue}"
    logger.info(f"=== CellxGene Census ingest: {args.tissue} → {dataset_id} ===")

    adata = fetch_tissue(args.tissue, args.organism, args.max_cells, args.seed)
    adata = enforce_schema(adata, dataset_id)

    logger.info(f"Cell types: {adata.obs['cell_type'].nunique()} unique")
    top5 = adata.obs["cell_type"].value_counts().head(5)
    for name, count in top5.items():
        logger.info(f"  {name}: {count}")
    logger.info(f"Source datasets: {adata.obs['dataset_id'].nunique()} (from Census)")

    adata.write_h5ad(out_path, compression="gzip")
    logger.info(f"Wrote {out_path} ({adata.n_obs:,} cells × {adata.n_vars:,} genes)")

    # Emit a manifest sidecar for provenance
    manifest = {
        "tissue": args.tissue,
        "organism": args.organism,
        "census_version": CENSUS_VERSION,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_unique_cell_types": int(adata.obs["cell_type"].nunique()),
        "n_source_datasets": int(adata.obs["dataset_id"].nunique()),
        "source_datasets": sorted(adata.obs["dataset_id"].unique().tolist()),
        "top_cell_types": top5.to_dict(),
        "output": str(out_path),
        "seed": args.seed,
    }
    sidecar = out_path.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    logger.info(f"Wrote manifest sidecar {sidecar}")


if __name__ == "__main__":
    main()
