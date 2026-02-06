#!/usr/bin/env python3
# 00_prepare_local_data.py — Prepare local h5ad datasets for CLOP-DiT
"""
Scans local h5ad files, applies standard preprocessing (QC, normalize, HVG),
auto-generates metadata text descriptions, and outputs processed files +
metadata JSON ready for cache_builder.

This replaces 01_fetch_data.py + 02_clean_text.py when using local datasets.

Usage:
    python scripts/00_prepare_local_data.py \
        --data_dir ~/Desktop/datasets/CancerDatasets \
        --output_dir data/processed_h5ad \
        --max_datasets 5
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ============================================================================
#  Auto-generate text descriptions from h5ad metadata + filename
# ============================================================================

TISSUE_KEYWORDS = {
    "lung": "lung", "brain": "brain", "liver": "liver", "kidney": "kidney",
    "pancrea": "pancreas", "heart": "heart", "breast": "breast",
    "colon": "colon", "skin": "skin", "bone": "bone marrow",
    "blood": "blood", "pbmc": "peripheral blood", "ovary": "ovary",
    "prostat": "prostate", "bladder": "bladder", "stomach": "stomach",
    "gastric": "stomach", "esophag": "esophagus", "thyroid": "thyroid",
    "endometri": "endometrium", "cervix": "cervix", "cervical": "cervix",
    "retina": "retina", "muscle": "muscle", "intestin": "intestine",
    "lymph": "lymph node", "spleen": "spleen", "adrenal": "adrenal gland",
    "testis": "testis", "uterus": "uterus", "placenta": "placenta",
}

DISEASE_KEYWORDS = {
    "cancer": "cancer", "tumor": "tumor", "carcinoma": "carcinoma",
    "melanoma": "melanoma", "lymphoma": "lymphoma", "leukemia": "leukemia",
    "adenocarcinoma": "adenocarcinoma", "glioma": "glioma",
    "glioblastoma": "glioblastoma", "sarcoma": "sarcoma",
    "fibro": "fibrosis", "covid": "COVID-19", "inflamm": "inflammation",
    "nsclc": "non-small cell lung cancer", "sclc": "small cell lung cancer",
    "hcc": "hepatocellular carcinoma", "aml": "acute myeloid leukemia",
    "cll": "chronic lymphocytic leukemia", "pdac": "pancreatic ductal adenocarcinoma",
    "normal": "healthy", "healthy": "healthy",
}


def infer_text_from_filename(filename: str) -> str:
    """Infer a biological text description from the h5ad filename."""
    name_lower = filename.lower()

    # Try to extract GSE ID
    gse_match = re.search(r"(GSE\d+)", filename, re.IGNORECASE)
    gse_id = gse_match.group(1) if gse_match else ""

    # Detect tissue
    tissues = []
    for kw, tissue in TISSUE_KEYWORDS.items():
        if kw in name_lower:
            tissues.append(tissue)

    # Detect disease
    diseases = []
    for kw, disease in DISEASE_KEYWORDS.items():
        if kw in name_lower:
            diseases.append(disease)

    # Build description
    parts = ["Single-cell RNA sequencing"]
    if tissues:
        parts.append(f"of {', '.join(set(tissues))} tissue")
    if diseases:
        disease_str = ", ".join(set(diseases))
        if disease_str not in ("healthy",):
            parts.append(f"with {disease_str}")
    parts.append("from human")
    if gse_id:
        parts.append(f"({gse_id})")

    return " ".join(parts) + "."


def infer_text_from_adata(adata, filename: str) -> str:
    """Infer text description from AnnData obs columns and filename."""
    base = infer_text_from_filename(filename)

    # Try to extract cell type info from obs
    cell_type_cols = [c for c in adata.obs.columns
                      if any(kw in c.lower() for kw in ["cell_type", "celltype", "cell.type",
                                                          "annotation", "cluster_label"])]
    if cell_type_cols:
        col = cell_type_cols[0]
        types = adata.obs[col].value_counts().head(5).index.tolist()
        type_str = ", ".join(str(t) for t in types)
        base = base.rstrip(".") + f", including cell types: {type_str}."

    # Check for batch/condition info
    condition_cols = [c for c in adata.obs.columns
                      if any(kw in c.lower() for kw in ["condition", "treatment",
                                                          "disease", "sample_type"])]
    if condition_cols:
        col = condition_cols[0]
        conditions = adata.obs[col].unique().tolist()[:5]
        cond_str = ", ".join(str(c) for c in conditions)
        base = base.rstrip(".") + f", conditions: {cond_str}."

    return base


def preprocess_adata(adata, dataset_id, min_genes=200, min_cells=3,
                     n_top_genes=2000, target_sum=1e4):
    """Standard scanpy preprocessing pipeline."""
    logger.info(f"[{dataset_id}] Raw: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Ensure X is not integer (some h5ad store raw int counts)
    import scipy.sparse as sp
    if sp.issparse(adata.X):
        if adata.X.dtype in (np.int64, np.int32):
            adata.X = adata.X.astype(np.float32)
    else:
        if adata.X.dtype in (np.int64, np.int32):
            adata.X = adata.X.astype(np.float32)

    # Basic QC
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Mitochondrial genes
    adata.var["mt"] = (adata.var_names.str.startswith("MT-") |
                       adata.var_names.str.startswith("mt-"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()

    logger.info(f"[{dataset_id}] After QC: {adata.shape}")

    # Store raw counts before normalization
    adata.layers["counts"] = adata.X.copy()

    # Normalize + log1p
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # HVG selection
    if adata.shape[1] > n_top_genes:
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_top_genes, flavor="seurat_v3",
                layer="counts", subset=False,
            )
        except Exception:
            # Fallback to seurat flavor if seurat_v3 fails
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_top_genes, flavor="seurat",
                subset=False,
            )
        adata = adata[:, adata.var["highly_variable"]].copy()

    logger.info(f"[{dataset_id}] Final: {adata.shape}")
    return adata


def main():
    parser = argparse.ArgumentParser(description="Prepare local h5ad data for CLOP-DiT")
    parser.add_argument("--data_dirs", nargs="+",
                        default=[str(Path.home() / "Desktop/datasets/CancerDatasets")],
                        help="Directories containing h5ad files")
    parser.add_argument("--output_dir", type=str, default="data/processed_h5ad")
    parser.add_argument("--max_datasets", type=int, default=None,
                        help="Max number of datasets to process")
    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument("--max_cells", type=int, default=None,
                        help="Subsample to at most this many cells per dataset")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip preprocessing, just generate metadata")
    args = parser.parse_args()

    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover h5ad files
    h5ad_files = []
    for ddir in args.data_dirs:
        ddir = Path(ddir)
        if ddir.is_file() and ddir.suffix == ".h5ad":
            h5ad_files.append(ddir)
        elif ddir.is_dir():
            h5ad_files.extend(sorted(ddir.glob("*.h5ad")))

    if args.max_datasets:
        h5ad_files = h5ad_files[:args.max_datasets]

    print(f"Found {len(h5ad_files)} h5ad files:")
    for f in h5ad_files:
        print(f"  {f.name}")

    if not h5ad_files:
        print("No h5ad files found!")
        sys.exit(1)

    metadata = {}
    processed_files = []

    for h5ad_path in h5ad_files:
        dataset_id = h5ad_path.stem
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_id}")
        print(f"{'='*60}")

        try:
            adata = sc.read_h5ad(h5ad_path)
            print(f"  Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

            # Generate text description
            text_desc = infer_text_from_adata(adata, h5ad_path.name)
            print(f"  Text: {text_desc}")

            # Subsample if needed
            if args.max_cells and adata.shape[0] > args.max_cells:
                sc.pp.subsample(adata, n_obs=args.max_cells)
                print(f"  Subsampled to {adata.shape[0]} cells")

            if not args.skip_preprocess:
                adata = preprocess_adata(
                    adata, dataset_id, n_top_genes=args.n_top_genes,
                )

                # Save processed
                out_path = output_dir / f"{dataset_id}_processed.h5ad"
                adata.write_h5ad(out_path)
                processed_files.append(str(out_path))
                print(f"  Saved: {out_path}")
            else:
                processed_files.append(str(h5ad_path))

            # Store metadata
            metadata[dataset_id] = {
                "text": text_desc,
                "tissue": "inferred",
                "disease": "inferred",
                "organism": "human",
                "n_cells": int(adata.shape[0]),
                "n_genes": int(adata.shape[1]),
                "source_file": str(h5ad_path),
            }

        except Exception as e:
            logger.error(f"Failed to process {dataset_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save metadata
    meta_path = output_dir / "metadata_structured.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata saved: {meta_path}")

    # Save file list
    filelist_path = output_dir / "processed_files.json"
    with open(filelist_path, "w") as f:
        json.dump(processed_files, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Processed {len(processed_files)} datasets.")
    print(f"  Output dir: {output_dir}")
    print(f"  Metadata:   {meta_path}")
    print(f"Next: Run 03_cache_latents.py to compute embeddings.")


if __name__ == "__main__":
    main()
