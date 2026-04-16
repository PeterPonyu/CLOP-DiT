#!/usr/bin/env python3
"""Adapt a CellxGene Census h5ad into the CLOP-DiT training pipeline format.

Census h5ad convention:
  - adata.X is normalised log-like float (max ~= 10)
  - adata.raw.X holds integer counts
  - var_names are Ensembl IDs (ENSG*)
  - var['feature_name'] holds HGNC gene symbols

Training pipeline convention (see scripts/data_prep/00_prepare_all_data.py):
  - adata.X is raw counts (scGPT rank-binning tokeniser consumes this)
  - var_names are gene symbols
  - Passed through preprocess_adata() for QC / HVG / normalisation

This script does the format swap and runs preprocess_adata() so the
output is schema-identical to the training-corpus processed h5ads.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.logging_config import setup_logging  # noqa: E402

# 00_prepare_all_data.py starts with a digit, so we load it via importlib.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "prep_all_data",
    Path(__file__).resolve().parent / "00_prepare_all_data.py",
)
_preprocess_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_preprocess_mod)

logger = logging.getLogger(__name__)


def census_to_training(adata: ad.AnnData) -> ad.AnnData:
    """Swap raw counts into X, rename var_names to gene symbols, drop raw."""
    if adata.raw is None:
        raise ValueError("adata.raw is None — Census h5ad should have .raw with counts")

    raw_X = adata.raw.X
    raw_var = adata.raw.var.copy()
    if "feature_name" not in raw_var.columns:
        raise ValueError("raw.var missing 'feature_name' column")

    # Build a new AnnData with raw counts as X and symbol var_names
    new = ad.AnnData(
        X=raw_X.astype(np.float32) if sp.issparse(raw_X) else sp.csr_matrix(raw_X, dtype=np.float32),
        obs=adata.obs.copy(),
        var=raw_var[["feature_name"]].copy(),
        uns=dict(adata.uns) if adata.uns is not None else {},
    )
    # Promote feature_name to be var index (gene symbol)
    new.var_names = new.var["feature_name"].astype(str).values
    new.var_names_make_unique()

    logger.info(f"Swapped raw→X, renamed to gene symbols: {new.shape}")
    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Census raw h5ad path")
    parser.add_argument("--output", required=True, help="Processed h5ad output path")
    parser.add_argument("--dataset-id", required=True,
                        help="Dataset id tag, e.g. census_kidney")
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--max-cells", type=int, default=0,
                        help="Subsample to this many cells (0 = all)")
    parser.add_argument("--subsample-seed", type=int, default=0)
    args = parser.parse_args()

    setup_logging()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Census h5ad: {in_path}")
    adata = ad.read_h5ad(in_path)
    logger.info(f"Input shape: {adata.shape}")

    # Format adapt
    adata = census_to_training(adata)

    # Optional subsample before preprocess (mirrors 00_prepare_all_data convention)
    if args.max_cells and adata.n_obs > args.max_cells:
        import scanpy as sc
        sc.pp.subsample(adata, n_obs=args.max_cells, random_state=args.subsample_seed)
        logger.info(f"Subsampled to {adata.n_obs} cells (seed={args.subsample_seed})")

    # Run training-corpus preprocess
    adata = _preprocess_mod.preprocess_adata(
        adata, args.dataset_id, n_top_genes=args.n_top_genes,
    )
    if adata is None:
        logger.error("preprocess_adata returned None (too few cells?)")
        sys.exit(1)

    adata.write_h5ad(out_path, compression="gzip")
    logger.info(f"Wrote {out_path} ({adata.n_obs} cells × {adata.n_vars} genes)")


if __name__ == "__main__":
    main()
