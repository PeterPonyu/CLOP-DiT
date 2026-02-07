#!/usr/bin/env python3
# 01_integrate_h5_datasets.py — Integrate 10x h5 filtered feature matrices into h5ad
"""
v0.4 data expansion: Read raw 10x HDF5 (.h5) filtered_feature_bc_matrix files
from scRNA-25100 directory (20 GSE studies), merge per-GSE, preprocess,
and save as individual h5ad files compatible with the CLOP-DiT pipeline.

Each GSE folder contains one or more h5 files from the same study.
Files within the same GSE are concatenated into a single AnnData object.

Text descriptions are derived from verified GEO metadata (fetched Feb 2026).

Usage:
    python scripts/01_integrate_h5_datasets.py \
        --source_dir /path/to/scRNA-25100 \
        --output_dir data/processed_h5ad \
        --max_cells 3000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ============================================================================
#  Verified GEO metadata-based descriptions for 20 GSE studies
#  Source: NCBI GEO (fetched February 2026)
# ============================================================================

NEW_DATASET_DESCRIPTIONS = {
    # ── Cancer studies ───────────────────────────────────────────────────
    "GSE300862": {
        "species": "Mm",
        "text": (
            "Single-cell RNA sequencing of mouse Kras/Trp53-driven lung adenocarcinoma tumors. "
            "This study investigates how Protein Kinase Cι (PKCι) dictates tumor trajectory and cell state "
            "plasticity, revealing that PKCι loss induces a senescent PATS-like tumor cell state and promotes "
            "tertiary lymphoid structure formation in the tumor microenvironment."
        ),
        "category": "cancer",
        "geo_title": "Protein Kinase Cι dictates tumor trajectory, cell state plasticity and immune surveillance in lung adenocarcinoma",
    },
    "GSE289611": {
        "species": "Hm",
        "text": (
            "Single-cell RNA sequencing of human pulmonary pleomorphic carcinoma (PPC), a rare aggressive "
            "subtype of non-small cell lung cancer. This dataset profiles surgically resected primary PPC "
            "tumors from 4 patients, characterizing the clonal evolution and cellular heterogeneity of this "
            "poorly understood lung malignancy."
        ),
        "category": "cancer",
        "geo_title": "Single-cell RNA sequencing of pulmonaly pleomorphic carcinoma",
    },
    "GSE309368": {
        "species": "Hm",
        "text": (
            "Single-cell RNA sequencing of human myeloid cells from peripheral blood of healthy donors "
            "and oral cavity squamous cell carcinoma (OCSCC) patients. This study profiles CD11b+ myeloid "
            "cells and CD45+ tumor-infiltrating leukocytes, revealing how hypoxia-induced EGR1 remodels "
            "neutrophil transcriptional programs to suppress antitumor immunity."
        ),
        "category": "cancer",
        "geo_title": "Transcriptional profiling of human myeloid cells in healthy donors and oral cavity cancer (OCSCC) patients",
    },
    "GSE302433": {
        "species": "Mm",
        "text": (
            "Single-cell RNA sequencing of mouse B16F10 melanoma tumors investigating the role of ZDHHC13 "
            "palmitoyl transferase in the tumor microenvironment. This study demonstrates that ZDHHC13 "
            "modulates tumor-immune cell interactions to suppress melanoma metastasis in a subcutaneous "
            "mouse model."
        ),
        "category": "cancer",
        "geo_title": "ZDHHC13 Modulates Tumor Microenvironment Interactions to Suppress Melanoma Metastasis [scRNA-Seq]",
    },
    "GSE307774": {
        "species": "Mm",
        "text": (
            "Single-cell RNA sequencing of mouse colorectal cancer tumors driven by oncogenic Kras/Braf "
            "MAPK signaling. This study reveals that MAPK pathway activation induces a revival stem cell "
            "phenotype in colon tumors, and that KRAS-G12D inhibition (MRTX1133) triggers rapid cell state "
            "transition to canonical Lgr5+ stem cells, driving therapeutic resistance."
        ),
        "category": "cancer",
        "geo_title": "Oncogenic MAPK signalling induces epithelial cell state change and therapeutic resistance in vivo [scRNA-Seq]",
    },
    "GSE280847": {
        "species": "Mm",
        "text": (
            "Single-cell RNA sequencing of CD45+ tumor-infiltrating leukocytes from mouse MC38 colon "
            "tumors. This study investigates T cell-intrinsic p53-mediated regulation of antitumor immunity, "
            "comparing p53 wild-type and p53-7KQ gain-of-function mutant T cells in the tumor "
            "microenvironment."
        ),
        "category": "cancer",
        "geo_title": "scRNA-seq analysis of Cd45+ cells isolated from MC38 tumors",
    },
    "GSE227719": {
        "species": "Mm",
        "text": (
            "Single-cell RNA sequencing of mouse lung tumor organoids derived from KrasG12D/P53-flox "
            "alveolar type 2 (AT2) cells. This study reveals the tumorigenic potential of the alveolar "
            "progenitor cell state, profiling cell fate transitions during oncogenic transformation of "
            "lung epithelial cells in an organoid model."
        ),
        "category": "cancer",
        "geo_title": "Organoid modeling reveals the tumorigenic potential of the alveolar progenitor cell state",
    },
    "GSE307261": {
        "species": "Hm",
        "text": (
            "Single-cell RNA sequencing of human peripheral blood B cells and T cells from melanoma "
            "patients undergoing checkpoint immunotherapy. This study demonstrates that CTLA-4 blockade "
            "(ipilimumab) shifts the B cell repertoire toward autoimmunity, causing peripheral B cell "
            "tolerance breakdown and immune-related adverse events (irAEs)."
        ),
        "category": "cancer_immunology",
        "geo_title": "CTLA-4 Blockade Shifts the B Cell Repertoire Towards Autoimmunity",
    },
    "GSE283397": {
        "species": "Hm",
        "text": (
            "Single-cell RNA sequencing of human peripheral blood CD8+ T cells from patients with chronic "
            "lymphocytic leukemia (CLL) treated with ibrutinib and pembrolizumab (anti-PD-1). This CITE-seq "
            "dataset profiles the activation and exhaustion dynamics of tumor-reactive T cells during "
            "combination checkpoint and BTK inhibitor therapy."
        ),
        "category": "cancer_immunology",
        "geo_title": "Activation and exhaustion of CD8 T cells in patients with chronic lymphocytic leukemia",
    },

    # ── Disease models ───────────────────────────────────────────────────
    "GSE308428": {
        "species": "Mm",
        "text": (
            "Single-cell RNA sequencing of mouse hippocampal microglia from an autism spectrum disorder "
            "(ASD) model induced by prenatal exposure to anti-Caspr2 antibodies. This study shows that "
            "ACE inhibitor captopril restores microglial homeostasis and reverses ASD-like phenotypes by "
            "normalizing dysregulated eIF2/mTOR and oxidative phosphorylation pathways."
        ),
        "category": "disease_neuro",
        "geo_title": "Captopril Restores Microglial Homeostasis and Reverses ASD-like Phenotype",
    },
    "GSE288211": {
        "species": "Mm",
        "text": (
            "Single-cell RNA sequencing of mouse pancreatic islets from β-cell specific Zzef1 knockout "
            "models (βZKO-Mip and βZKO-Rip) after high-fat diet feeding. This study profiles ribosomal "
            "stress-surveillance mechanisms in pancreatic β-cell dysfunction, providing insight into "
            "type 2 diabetes pathogenesis under metabolic stress."
        ),
        "category": "disease_metabolic",
        "geo_title": "Single-cell gene expression profiles of pancreatic islets from β-cell specific knockout mouse models",
    },
    "GSE303309": {
        "species": "Mm",
        "text": (
            "Single-cell RNA sequencing of brain-infiltrating CD45+ immune cells from App-NL-G-F "
            "Alzheimer's disease mouse model with and without CD8+ T cell-specific Nr4a deletion. "
            "This study profiles the stage-specific roles of clonally expanded CD8+ T cells in "
            "regulating amyloid pathology and neuroinflammation."
        ),
        "category": "disease_neuro",
        "geo_title": "Single-cell Transcriptomic Profile of Brain Immune Cells From Alzheimer's Disease Mouse Models",
    },
    "GSE306676": {
        "species": "Mm",
        "text": (
            "Single-cell RNA sequencing of mouse spinal cord nuclei from SOD1-G93A amyotrophic lateral "
            "sclerosis (ALS) model at early, mid, and end-stage timepoints. This longitudinal study "
            "identifies an emergent disease-associated motor neuron transcriptional state that precedes "
            "cell death, providing a temporal map of selective motor neuron vulnerability."
        ),
        "category": "disease_neuro",
        "geo_title": "An emergent disease-associated motor neuron state precedes cell death in a mouse model of ALS",
    },
    "GSE220913": {
        "species": "Hm",
        "text": (
            "Single-cell RNA sequencing of human HL-60 cell lines (wild-type and F508del-CFTR cystic "
            "fibrosis mutant) and their DMSO-differentiated neutrophils. This study profiles aberrant "
            "immune programming in cystic fibrosis neutrophils, revealing how F508del-CFTR mutation "
            "alters neutrophil transcriptional signatures and inflammatory responses."
        ),
        "category": "disease_other",
        "geo_title": "Transcriptional Signatures of Wild-type and F508del-CF HL-60 Cell Lines",
    },

    # ── Immunology ───────────────────────────────────────────────────────
    "GSE291166": {
        "species": "Mm",
        "text": (
            "Single-cell RNA sequencing of mouse mesenteric lymph node cells investigating maternal "
            "antibody interference with rotavirus vaccination. This study profiles lymphocyte activation "
            "and differentiation dynamics in neonatal mice, characterizing how pre-existing maternal "
            "antibodies modulate the adaptive immune response to oral vaccination."
        ),
        "category": "immunology",
        "geo_title": "Mechanisms of maternal antibody interference to rotavirus vaccination",
    },
    "GSE236565": {
        "species": "Mm",
        "text": (
            "Single-cell RNA sequencing of mouse splenic CD4+ T cells stimulated with anti-CD3/CD28 "
            "with and without retinoic acid treatment. This study investigates the NRIP1 R448G risk "
            "variant in inflammatory bowel disease, showing how it promotes T cell gut-homing and "
            "disrupts the Teff:Treg balance to exacerbate intestinal inflammation."
        ),
        "category": "immunology",
        "geo_title": "Nuclear receptor coregulator NRIP1 R448G modulates T cell gut-homing",
    },

    # ── Development / Neuroscience ───────────────────────────────────────
    "GSE277740": {
        "species": "Mm",
        "text": (
            "Single-cell RNA sequencing of mouse thalamus brain tissue comparing Illumina SBS and "
            "DNBSEQ sequencing platforms. This study profiles the cellular diversity of the mouse "
            "thalamus including neurons, oligodendrocytes, astrocytes, and immune cells, demonstrating "
            "that both sequencing technologies perform comparably for single-cell transcriptomics."
        ),
        "category": "neuroscience",
        "geo_title": "Illumina SBS sequencing and DNBSEQ perform similarly for single-cell transcriptomics",
    },
    "GSE212502": {
        "species": "Hm",
        "text": (
            "Single-cell RNA sequencing of day-56 human forebrain organoids derived from control and "
            "Werner/Williams syndrome (WS) iPSCs. This study profiles neural diversity in brain organoids, "
            "characterizing disease-related changes in neural progenitor composition, neuronal maturation, "
            "and glial cell populations during human neurodevelopment."
        ),
        "category": "development_neuro",
        "geo_title": "Gene expression profile at single cell level of brain organoid at day56",
    },
    "GSE306567": {
        "species": "Mm",
        "text": (
            "Single-cell RNA sequencing of mouse embryonic stem cells (mESCs) and anterior neural "
            "progenitor cells at day 3 and day 6 of differentiation, comparing wild-type and Zic2 "
            "knockout. This study reveals the dual role of ZIC2 as a pioneer transcription factor "
            "and enhancer activator during neural induction."
        ),
        "category": "development",
        "geo_title": "Dual role of ZIC2 during neural induction: from pioneer transcription factor to enhancer activator",
    },

    # ── EXCLUDED ─────────────────────────────────────────────────────────
    # GSE278924: Gallus gallus (chicken) — gene symbols (LOC*) incompatible
    #            with both scGPT human and mouse vocabularies
}

# Species that are incompatible with scGPT
SKIP_GSE = {"GSE278924"}  # Gallus gallus


def read_and_merge_h5_files(gse_dir, gse_id, max_cells=3000):
    """Read all h5 filtered feature matrix files in a GSE directory and merge."""
    h5_files = sorted(gse_dir.glob("*.h5"))
    if not h5_files:
        logger.warning(f"[{gse_id}] No .h5 files found in {gse_dir}")
        return None

    adatas = []
    for h5_file in h5_files:
        try:
            adata = sc.read_10x_h5(h5_file)
            adata.var_names_make_unique()

            # Add sample metadata
            sample_id = h5_file.stem.replace("_filtered_feature_bc_matrix", "")
            sample_id = sample_id.replace(".filtered_feature_bc_matrix", "")
            adata.obs["sample_id"] = sample_id
            adata.obs["source_file"] = h5_file.name

            # Filter to Gene Expression only (for CITE-seq / multimodal data)
            if "feature_types" in adata.var.columns:
                gene_mask = adata.var["feature_types"] == "Gene Expression"
                if gene_mask.sum() < adata.shape[1]:
                    logger.info(f"  [{h5_file.name}] Filtering to Gene Expression: "
                                f"{gene_mask.sum()}/{adata.shape[1]} features")
                    adata = adata[:, gene_mask].copy()

            adatas.append(adata)
            logger.info(f"  [{h5_file.name}] {adata.shape[0]} cells × {adata.shape[1]} genes")

        except Exception as e:
            logger.warning(f"  [{h5_file.name}] Failed to read: {e}")
            continue

    if not adatas:
        return None

    # Merge all samples
    if len(adatas) == 1:
        merged = adatas[0]
    else:
        # Find intersection of genes across samples
        common_genes = set(adatas[0].var_names)
        for a in adatas[1:]:
            common_genes &= set(a.var_names)
        common_genes = sorted(common_genes)

        if len(common_genes) < 1000:
            logger.warning(f"[{gse_id}] Only {len(common_genes)} common genes across {len(adatas)} samples — "
                           f"using union instead")
            merged = ad.concat(adatas, join="outer", fill_value=0)
        else:
            # Subset to common genes, then concat
            adatas_common = [a[:, common_genes].copy() for a in adatas]
            merged = ad.concat(adatas_common, join="inner")

        logger.info(f"[{gse_id}] Merged {len(adatas)} samples → {merged.shape}")

    merged.var_names_make_unique()
    merged.obs_names_make_unique()

    # Subsample if too large
    if max_cells and merged.shape[0] > max_cells:
        sc.pp.subsample(merged, n_obs=max_cells)
        logger.info(f"[{gse_id}] Subsampled to {merged.shape[0]} cells")

    return merged


def preprocess_adata(adata, dataset_id, min_genes=200, min_cells=3,
                     n_top_genes=2000, target_sum=1e4):
    """Standard scanpy preprocessing pipeline (same as 00_prepare_all_data.py)."""
    logger.info(f"[{dataset_id}] Raw: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Ensure float type
    if sp.issparse(adata.X):
        if adata.X.dtype in (np.int64, np.int32):
            adata.X = adata.X.astype(np.float32)
    else:
        if adata.X.dtype in (np.int64, np.int32):
            adata.X = adata.X.astype(np.float32)

    # Check if data is raw counts
    X_sample = adata.X[:100].toarray() if sp.issparse(adata.X) else adata.X[:100]
    if not np.allclose(X_sample, X_sample.astype(int), atol=0.01):
        logger.warning(f"[{dataset_id}] Data appears normalized (not raw counts), skipping")
        return None

    # Check for Ensembl IDs
    gene_sample = list(adata.var_names[:20])
    if any(g.startswith(("ENSG", "ENSMUSG", "ENSGAL")) for g in gene_sample):
        logger.warning(f"[{dataset_id}] Ensembl IDs detected, incompatible with scGPT")
        return None

    # Basic QC filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Mitochondrial gene filtering (human + mouse)
    adata.var["mt"] = (adata.var_names.str.startswith("MT-") |
                       adata.var_names.str.startswith("mt-"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()

    logger.info(f"[{dataset_id}] After QC: {adata.shape}")

    if adata.shape[0] < 50:
        logger.warning(f"[{dataset_id}] Too few cells after QC ({adata.shape[0]}), skipping")
        return None

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
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_top_genes, flavor="seurat",
                subset=False,
            )
        adata = adata[:, adata.var["highly_variable"]].copy()

    logger.info(f"[{dataset_id}] Final: {adata.shape}")
    return adata


def main():
    parser = argparse.ArgumentParser(
        description="Integrate 10x h5 filtered feature matrices into h5ad for CLOP-DiT v0.4"
    )
    parser.add_argument("--source_dir", type=str,
                        default=str(Path.home() / "Desktop/iAODE-LAB/scRNA-25100"))
    parser.add_argument("--output_dir", type=str, default="data/processed_h5ad")
    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument("--max_cells", type=int, default=3000)
    args = parser.parse_args()

    setup_logging()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover GSE folders (format: N-GSEXXXXXX)
    gse_folders = sorted(
        [d for d in source_dir.iterdir() if d.is_dir() and "GSE" in d.name],
        key=lambda x: int(x.name.split("-")[0])
    )

    print(f"\n{'='*70}")
    print(f"CLOP-DiT v0.4 — Integrate 10x h5 Feature Matrices")
    print(f"{'='*70}")
    print(f"Source: {source_dir}")
    print(f"Found {len(gse_folders)} GSE folders")
    print(f"Descriptions available: {len(NEW_DATASET_DESCRIPTIONS)}")
    print(f"Max cells per dataset: {args.max_cells}")
    print(f"{'='*70}\n")

    metadata_new = {}
    processed = []
    skipped = []
    total_cells = 0

    for gse_dir in gse_folders:
        # Extract GSE ID from folder name (e.g., "1-GSE300862" → "GSE300862")
        folder_name = gse_dir.name
        gse_id = folder_name.split("-", 1)[1] if "-" in folder_name else folder_name

        print(f"\n[{folder_name}] Processing {gse_id}")

        # Check if this GSE should be skipped
        if gse_id in SKIP_GSE:
            print(f"  ✗ SKIPPED — incompatible species (not human/mouse)")
            skipped.append(gse_id)
            continue

        # Get verified description
        if gse_id not in NEW_DATASET_DESCRIPTIONS:
            print(f"  ✗ SKIPPED — no verified description available")
            skipped.append(gse_id)
            continue

        desc_info = NEW_DATASET_DESCRIPTIONS[gse_id]
        dataset_name = f"{gse_id}_scRNA25100"

        # Check if already processed
        out_path = output_dir / f"{dataset_name}_processed.h5ad"
        if out_path.exists():
            print(f"  ✓ Already exists: {out_path}")
            # Still add to metadata
            try:
                adata = sc.read_h5ad(out_path, backed="r")
                n_cells = adata.shape[0]
                n_genes = adata.shape[1]
                adata.file.close()
            except Exception:
                n_cells = 0
                n_genes = 0
            metadata_new[dataset_name] = {
                "text": desc_info["text"],
                "n_cells": n_cells,
                "n_genes": n_genes,
                "source_dir": str(gse_dir),
                "species": desc_info["species"],
                "category": desc_info["category"],
                "geo_title": desc_info["geo_title"],
            }
            processed.append(str(out_path))
            total_cells += n_cells
            continue

        # Read and merge h5 files
        merged = read_and_merge_h5_files(gse_dir, gse_id, max_cells=args.max_cells)
        if merged is None:
            skipped.append(gse_id)
            continue

        # Preprocess
        adata = preprocess_adata(merged, gse_id, n_top_genes=args.n_top_genes)
        if adata is None:
            skipped.append(gse_id)
            continue

        # Add metadata
        adata.obs["gse_id"] = gse_id
        adata.obs["species"] = desc_info["species"]
        adata.obs["category"] = desc_info["category"]

        # Save
        adata.write_h5ad(out_path)
        processed.append(str(out_path))
        n_cells = adata.shape[0]
        total_cells += n_cells

        metadata_new[dataset_name] = {
            "text": desc_info["text"],
            "n_cells": n_cells,
            "n_genes": int(adata.shape[1]),
            "source_dir": str(gse_dir),
            "species": desc_info["species"],
            "category": desc_info["category"],
            "geo_title": desc_info["geo_title"],
        }

        print(f"  ✓ Saved: {out_path} ({n_cells} cells × {adata.shape[1]} genes)")
        print(f"  Text: {desc_info['text'][:80]}...")

    # Save new metadata (append-friendly)
    new_meta_path = output_dir / "metadata_scRNA25100.json"
    with open(new_meta_path, "w") as f:
        json.dump(metadata_new, f, indent=2, ensure_ascii=False)

    # Also update the main metadata file if it exists
    main_meta_path = output_dir / "metadata_structured.json"
    if main_meta_path.exists():
        with open(main_meta_path) as f:
            main_meta = json.load(f)
        main_meta.update(metadata_new)
        with open(main_meta_path, "w") as f:
            json.dump(main_meta, f, indent=2, ensure_ascii=False)
        print(f"\n  Updated main metadata: {main_meta_path} ({len(main_meta)} total datasets)")

    print(f"\n{'='*70}")
    print(f"CLOP-DiT v0.4 — Integration Complete")
    print(f"{'='*70}")
    print(f"  New datasets processed: {len(processed)}")
    print(f"  Skipped: {len(skipped)} ({skipped})")
    print(f"  Total new cells: {total_cells:,}")
    print(f"  New metadata: {new_meta_path}")
    print(f"{'='*70}")
    print(f"\nNext: Re-run 03_cache_latents.py to include new datasets")


if __name__ == "__main__":
    main()
