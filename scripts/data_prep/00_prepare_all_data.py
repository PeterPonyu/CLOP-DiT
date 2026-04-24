#!/usr/bin/env python3
# 00_prepare_all_data.py — Prepare ALL local h5ad datasets for CLOP-DiT v0.3
"""
v0.3 data pipeline: Processes all 55 local datasets from 4 directories
with curated biological text descriptions.

Key improvements over v0.2:
  - 55 datasets from 4 directories (vs 43/3-dirs in v0.2)
  - DevelopmentDatasets2 added (12 new datasets)
  - Automatic filtering of invalid datasets:
    · Normalized-only files (no raw counts → scGPT can't encode)
    · Ensembl ID files (incompatible with scGPT gene vocab)
  - Biologically accurate text descriptions → real BiomedBERT alignment
  - Per-dataset subsampling for memory management

Usage:
    python scripts/00_prepare_all_data.py \
        --output_dir data/processed_h5ad \
        --max_cells 3000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ============================================================================
#  Curated biological text descriptions for all 43 datasets
#  (Replaces v0.1 filename-based auto-generation)
# ============================================================================

DATASET_DESCRIPTIONS = {
    # ── CancerDatasets (16) ──────────────────────────────────────────────
    "GSE123813_bccHmCancer": (
        "Single-cell RNA sequencing of human basal cell carcinoma (BCC) skin tumor samples. "
        "This dataset profiles the tumor microenvironment of cutaneous basal cell carcinoma, "
        "capturing malignant keratinocytes, infiltrating immune cells, and stromal populations in human skin cancer."
    ),
    "GSE123813_sccHmCancer": (
        "Single-cell RNA sequencing of human cutaneous squamous cell carcinoma (SCC). "
        "This dataset characterizes the transcriptomic landscape of squamous cell carcinoma of the skin, "
        "including tumor cells, tumor-infiltrating lymphocytes, and the surrounding microenvironment."
    ),
    "GSE123902_LungAdreHmCancer": (
        "Single-cell RNA sequencing of human lung adenocarcinoma. "
        "This dataset profiles malignant epithelial cells, immune infiltrates, and stromal components "
        "within the tumor microenvironment of non-small cell lung cancer (adenocarcinoma subtype)."
    ),
    "GSE132509_acutelymluekPBMCHmCancer": (
        "Single-cell RNA sequencing of bone marrow mononuclear cells from human pediatric patients "
        "with acute lymphoblastic leukemia (ALL). This dataset profiles leukemic lymphoblasts including "
        "Pre-B t(12;21) ETV6-RUNX1 ALL, Pre-B high hyperdiploid ALL, and Pre-T ALL subtypes, alongside "
        "residual normal hematopoietic cells in the bone marrow of childhood leukemia patients."
    ),
    "GSE143423_lbm_CancerBrainHm": (
        "Single-cell RNA sequencing of human leptomeningeal brain metastasis. "
        "This dataset profiles the cerebrospinal fluid and meningeal tumor microenvironment in patients "
        "with leptomeningeal carcinomatosis, capturing metastatic cancer cells and immune populations."
    ),
    "GSE143423_tnbc_CancerBrainHm": (
        "Single-cell RNA sequencing of brain metastases originating from human triple-negative breast cancer (TNBC). "
        "This dataset characterizes the metastatic tumor microenvironment in the brain, including TNBC-derived "
        "malignant cells, microglia, and infiltrating immune cells."
    ),
    "GSE148218_bmALLHmCancer": (
        "Single-cell RNA sequencing of human bone marrow samples from patients with acute lymphoblastic leukemia (ALL). "
        "This dataset profiles leukemic lymphoblasts, residual normal hematopoietic progenitors, and bone marrow "
        "niche cells in the context of B-cell or T-cell ALL."
    ),
    "GSE155109_bcECHmCancer": (
        "Single-cell RNA sequencing of endothelial cells isolated from human breast cancer tumors. "
        "This dataset focuses on tumor-associated endothelial cell heterogeneity, angiogenic programs, "
        "and vascular remodeling within the breast cancer microenvironment."
    ),
    "GSE155109_bcStromaHmCancer": (
        "Single-cell RNA sequencing of stromal cells from human breast cancer tissue. "
        "This dataset characterizes cancer-associated fibroblasts, mesenchymal cells, and other stromal populations "
        "that shape the breast tumor microenvironment and support tumor progression."
    ),
    "GSE183904_GastricHmCancer": (
        "Single-cell RNA sequencing of human gastric cancer (stomach adenocarcinoma). "
        "This dataset profiles the cellular composition of gastric tumors, including malignant epithelial cells, "
        "tumor-infiltrating immune cells, and stromal components of the stomach cancer microenvironment."
    ),
    "GSE222002_TcellsHmCancer": (
        "Single-cell RNA sequencing of T cells isolated from human cancer samples. "
        "This dataset characterizes tumor-infiltrating T lymphocyte subsets, including CD8+ cytotoxic T cells, "
        "CD4+ helper T cells, and regulatory T cells, profiling exhaustion and activation states."
    ),
    "GSE222369_NKsLymphomaHmCancer": (
        "Single-cell RNA sequencing of natural killer (NK) cells from human lymphoma patients. "
        "This dataset profiles NK cell functional states, cytotoxicity programs, and immune evasion mechanisms "
        "in the context of lymphoid malignancy."
    ),
    "GSE225600_breast_CancerHm": (
        "Single-cell RNA sequencing of human breast cancer tumor tissue. "
        "This dataset captures the heterogeneous cellular landscape of breast carcinoma, including malignant "
        "epithelial cells, immune infiltrates, endothelial cells, and cancer-associated fibroblasts."
    ),
    "GSE235787_bcellsALLHmCancer": (
        "Single-cell RNA sequencing of B cells from human patients with acute lymphoblastic leukemia (B-ALL). "
        "This dataset profiles leukemic B-lymphoblasts and residual normal B-cell populations, characterizing "
        "the transcriptomic features of B-cell precursor acute lymphoblastic leukemia."
    ),
    "GSE262288_breastMetasisHmCancer": (
        "Single-cell RNA sequencing of metastatic breast cancer in human patients. "
        "This dataset profiles disseminated tumor cells and the metastatic microenvironment, capturing the "
        "transcriptomic alterations associated with breast cancer metastasis to distant organ sites."
    ),
    "GSE98638_TcellLiverHmCancer": (
        "Single-cell RNA sequencing of T cells from human hepatocellular carcinoma (liver cancer). "
        "This dataset characterizes tumor-infiltrating T lymphocytes in the liver tumor microenvironment, "
        "profiling exhaustion markers, clonal expansion, and functional states of CD8+ and CD4+ T cells."
    ),

    # ── CancerDatasets2 (12) ─────────────────────────────────────────────
    "GSE117988_MCCPBMCCancer": (
        "Single-cell RNA sequencing of peripheral blood mononuclear cells (PBMCs) from patients with "
        "Merkel cell carcinoma (MCC). This dataset profiles circulating immune cell populations and their "
        "functional states in the context of this aggressive neuroendocrine skin cancer."
    ),
    "GSE117988_MCCTumorCancer": (
        "Single-cell RNA sequencing of Merkel cell carcinoma (MCC) tumor tissue. "
        "This dataset characterizes malignant neuroendocrine tumor cells and the intra-tumoral immune "
        "microenvironment of this rare and aggressive cutaneous neoplasm."
    ),
    "GSE120575_melanomaHmCancer": (
        "Single-cell RNA sequencing of human melanoma tumor samples. "
        "This dataset profiles malignant melanocytes, tumor-infiltrating lymphocytes, and stromal cells "
        "within the melanoma microenvironment, capturing immune checkpoint-related transcriptomic signatures."
    ),
    "GSE124310_MMHmCancer": (
        "Single-cell RNA sequencing of human multiple myeloma (MM) bone marrow samples. "
        "This dataset profiles clonal plasma cells, residual normal hematopoietic cells, and the bone marrow "
        "niche microenvironment in the context of this hematological malignancy."
    ),
    "GSE138709_LiverCancer": (
        "Single-cell RNA sequencing of human intrahepatic cholangiocarcinoma (ICC). "
        "This dataset profiles the cellular heterogeneity of bile duct cancer, capturing malignant "
        "cholangiocytes, tumor-associated fibroblasts, immune infiltrates, and the biliary tumor microenvironment."
    ),
    "GSE149655_CAHmCancer": (
        "Single-cell RNA sequencing of human early-stage lung adenocarcinoma with activating somatic KRAS mutations. "
        "This dataset profiles the tumor microenvironment of KRAS-mutant non-small cell lung cancer, capturing "
        "malignant epithelial cells, immune cell populations, and stromal components in LUAD."
    ),
    "GSE163558_stomachHmCancer": (
        "Single-cell RNA sequencing of human stomach cancer. "
        "This dataset profiles the cellular landscape of gastric carcinoma, capturing tumor epithelial cells, "
        "immune cell infiltrates, and stromal populations in the gastric tumor microenvironment."
    ),
    "GSE168181_BreastHmCancer": (
        "Single-cell RNA sequencing of human breast cancer tissue. "
        "This dataset characterizes the intra-tumoral heterogeneity of breast carcinoma, profiling malignant "
        "epithelial cells, immune cell subsets, and the stromal compartment across breast cancer subtypes."
    ),
    "GSE189357_lungAdreHmCancer": (
        "Single-cell RNA sequencing of human lung adenocarcinoma. "
        "This dataset profiles the tumor microenvironment of pulmonary adenocarcinoma, capturing malignant "
        "alveolar and bronchial epithelial cells, tumor-associated macrophages, T cells, and fibroblasts."
    ),
    "GSE225857_liverColonMetasisHmCancer": (
        "Single-cell RNA sequencing of human colorectal cancer liver metastases. "
        "This dataset profiles the metastatic microenvironment of colon cancer cells that have disseminated "
        "to the liver, characterizing tumor-hepatocyte interactions and the metastatic immune niche."
    ),
    "GSE228499_breastHmCancer": (
        "Single-cell RNA sequencing of human breast cancer. "
        "This dataset captures the transcriptomic diversity of breast tumor tissue, including luminal and "
        "basal-like malignant cells, tumor-infiltrating lymphocytes, myeloid cells, and stromal fibroblasts."
    ),
    "GSE283205_hepatoblastomaCancer": (
        "Single-cell RNA sequencing of hepatoblastoma tumor tissue. "
        "This dataset profiles the cellular composition of hepatoblastoma, a rare pediatric liver malignancy, "
        "capturing embryonal and fetal-type hepatoblasts, stromal cells, and immune populations."
    ),

    # ── DevelopmentDatasets (15) ─────────────────────────────────────────
    "bm_GSE120446": (
        "Single-cell RNA sequencing of bone marrow during hematopoietic development. "
        "This dataset profiles the hierarchical differentiation of hematopoietic stem and progenitor cells "
        "into mature blood lineages within the bone marrow niche."
    ),
    "dentate": (
        "Single-cell RNA sequencing of the dentate gyrus in the hippocampus during brain development. "
        "This dataset profiles neurogenesis in the dentate gyrus, capturing neural stem cells, intermediate "
        "progenitors, granule neurons, and glial cells during hippocampal neuronal differentiation."
    ),
    "endo": (
        "Single-cell RNA sequencing of endoderm differentiation. "
        "This dataset profiles the developmental trajectory of definitive endoderm specification, capturing "
        "pluripotent stem cells transitioning through mesendoderm to definitive endoderm progenitors."
    ),
    "GSE120505_bloodAged": (
        "Single-cell RNA sequencing of aged blood cells. "
        "This dataset profiles hematopoietic cell populations in aging blood, characterizing age-associated "
        "changes in immune cell composition, clonal hematopoiesis, and transcriptomic drift."
    ),
    "GSE148215_hESCHSPCD8Hm": (
        "Single-cell RNA sequencing of human embryonic stem cell (hESC)-derived hematopoietic stem and progenitor "
        "cells (HSPCs) differentiating toward CD8+ T cells. This dataset captures the in vitro directed "
        "differentiation trajectory from pluripotent stem cells through hematopoietic progenitors to CD8+ T lymphocytes."
    ),
    "GSE165844_LSKMmBatch": (
        "Single-cell RNA sequencing of mouse Lin-Sca-1+c-Kit+ (LSK) hematopoietic stem and progenitor cells. "
        "This dataset profiles the heterogeneity of the multipotent progenitor compartment in mouse bone marrow, "
        "capturing long-term and short-term hematopoietic stem cells and multipotent progenitors."
    ),
    "GSE167597_spineMm": (
        "Single-cell RNA sequencing of mouse spinal cord during development. "
        "This dataset profiles the cellular diversity of the developing spinal cord, capturing motor neuron "
        "progenitors, interneuron subtypes, oligodendrocyte precursors, and astrocyte lineages."
    ),
    "GSE192857_hESCHmTimes": (
        "Single-cell RNA sequencing time-series of human embryonic stem cell (hESC) differentiation. "
        "This dataset captures transcriptomic dynamics across multiple time points during directed differentiation "
        "of human pluripotent stem cells, profiling lineage commitment and cell fate transitions."
    ),
    "GSE226131_HSCMmAged": (
        "Single-cell RNA sequencing of aged mouse hematopoietic stem cells (HSCs). "
        "This dataset profiles the transcriptomic changes associated with aging in the murine hematopoietic "
        "stem cell compartment, characterizing age-related lineage bias and functional decline."
    ),
    "GSE253355_bmNicheHm": (
        "Single-cell RNA sequencing of the human bone marrow niche. "
        "This dataset profiles the non-hematopoietic stromal microenvironment of human bone marrow, including "
        "mesenchymal stem cells, endothelial cells, adipocytes, and osteoblast lineage cells."
    ),
    "hemato": (
        "Single-cell RNA sequencing of hematopoiesis. "
        "This dataset profiles the continuous differentiation trajectory from hematopoietic stem cells to "
        "mature blood lineages, capturing erythroid, myeloid, and lymphoid progenitor populations."
    ),
    "hESC_GSE144024": (
        "Single-cell RNA sequencing of human embryonic stem cells (hESCs). "
        "This dataset profiles the transcriptomic heterogeneity of pluripotent human embryonic stem cells, "
        "capturing states of self-renewal, priming, and early lineage specification."
    ),
    "ifnHSPC_GSE226824": (
        "Single-cell RNA sequencing of interferon-treated hematopoietic stem and progenitor cells (HSPCs). "
        "This dataset profiles the transcriptomic response of HSPCs to interferon stimulation, characterizing "
        "interferon-stimulated gene programs and differentiation bias induced by inflammatory signaling."
    ),
    "lung": (
        "Single-cell RNA sequencing of lung development. "
        "This dataset profiles the cellular differentiation trajectories during pulmonary organogenesis, "
        "capturing alveolar type I and type II pneumocytes, airway progenitors, and mesenchymal cells."
    ),
    "setty": (
        "Single-cell RNA sequencing of human hematopoiesis from the Setty et al. study. "
        "This dataset profiles the continuous differentiation landscape of human bone marrow hematopoiesis, "
        "capturing the full hierarchy from HSCs to committed erythroid, myeloid, and lymphoid progenitors."
    ),

    # ── DevelopmentDatasets2 (12 — 2 filtered out) ──────────────────────
    "GSE115571_LPSMmDev": (
        "Single-cell RNA sequencing of mouse cells under lipopolysaccharide (LPS) stimulation. "
        "This dataset profiles the innate immune response to LPS, capturing macrophage activation states, "
        "inflammatory gene programs, and myeloid cell polarization during endotoxin challenge."
    ),
    "GSE130148_LungHmDev": (
        "Single-cell RNA sequencing of human fetal lung during development. "
        "This dataset profiles the cellular composition of the developing human lung, capturing airway "
        "progenitors, alveolar epithelial cells, mesenchymal populations, and endothelial lineages."
    ),
    "GSE142653pitHmDev": (
        "Single-cell RNA sequencing of human pituitary gland during development. "
        "This dataset profiles the cellular differentiation trajectories of the anterior pituitary, "
        "capturing hormone-producing cell lineages including corticotrophs, somatotrophs, and gonadotrophs."
    ),
    "GSE145929_ProgastinMmDev": (
        "Single-cell RNA sequencing of adult mouse prostate tissue. "
        "This dataset profiles the epithelial and stromal cell populations of the adult murine prostate, "
        "capturing luminal, basal, and neuroendocrine epithelial subtypes alongside prostatic stroma."
    ),
    "GSE145929_UrineMmDev": (
        "Single-cell RNA sequencing of adult mouse urethra. "
        "This dataset profiles the epithelial and stromal cell populations of the murine lower urinary tract, "
        "capturing urethral epithelial cells, smooth muscle, and surrounding connective tissue populations."
    ),
    "GSE165784_RetinaHmDev": (
        "Single-cell RNA sequencing of human retina during development. "
        "This dataset profiles the differentiation of retinal progenitor cells into photoreceptors, "
        "retinal ganglion cells, amacrine cells, and Müller glia during human retinal organogenesis."
    ),
    "GSE189070_astrocytesSCIMmDev": (
        "Single-cell RNA sequencing of mouse astrocytes following spinal cord injury (SCI). "
        "This dataset profiles the reactive astrocyte response to spinal cord injury, capturing "
        "astrocyte heterogeneity, scar-forming populations, and neuroinflammatory gene programs."
    ),
    "GSE213740_ADHm": (
        "Single-cell RNA sequencing of human ascending aortic wall tissue from patients with sporadic "
        "type A aortic dissection. This dataset profiles the cellular landscape of the diseased aortic wall, "
        "capturing smooth muscle cells, macrophage infiltration, endothelial dysfunction, and extracellular "
        "matrix remodeling associated with acute aortic dissection pathogenesis."
    ),
    # GSE225948_bloodMmStrokeDev — FILTERED OUT: normalized-only data (no raw counts, max=184.46)
    # GSE247719_PanSci_05_Muscle_adata — FILTERED OUT: Ensembl IDs (ENSMUSG), scGPT incompatible
    # GSE247719_PanSci_T_cell_adata — FILTERED OUT: Ensembl IDs (ENSMUSG), scGPT incompatible
    "GSE275119_TeethMmDev": (
        "Single-cell RNA sequencing of mouse dental tissue during tooth development. "
        "This dataset profiles odontogenesis, capturing dental epithelial cells, ameloblasts, odontoblasts, "
        "dental pulp mesenchyme, and periodontal ligament progenitors during mouse tooth morphogenesis."
    ),
}

# Dataset source directories (v0.3: 4 directories)
DATASET_DIRS = [
    Path.home() / "Desktop/datasets/CancerDatasets",
    Path.home() / "Desktop/datasets/CancerDatasets2",
    Path.home() / "Desktop/datasets/DevelopmentDatasets",
    Path.home() / "Desktop/datasets/DevelopmentDatasets2",
]

# Datasets to skip — incompatible with scGPT encoding pipeline
# (normalized-only → no raw counts for rank binning; Ensembl IDs → not in scGPT vocab)
SKIP_DATASETS = {
    "GSE120575_melanomaHmCancer",     # Normalized-only (max=16.35), no raw counts
    "GSE225948_bloodMmStrokeDev",     # Normalized-only (max=184.46), no raw counts
    "GSE148215_hESCHSPCD8Hm",        # Ensembl IDs (ENSG*), scGPT vocab uses gene symbols
    "GSE247719_PanSci_05_Muscle_adata",  # Ensembl IDs (ENSMUSG*), scGPT incompatible
    "GSE247719_PanSci_T_cell_adata",     # Ensembl IDs (ENSMUSG*), scGPT incompatible
}


def preprocess_adata(adata, dataset_id, min_genes=200, min_cells=3,
                     n_top_genes=2000, target_sum=1e4):
    """Standard scanpy preprocessing pipeline."""
    import scipy.sparse as sp

    logger.info(f"[{dataset_id}] Raw: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Ensure float type
    if sp.issparse(adata.X):
        if adata.X.dtype in (np.int64, np.int32):
            adata.X = adata.X.astype(np.float32)
    else:
        if adata.X.dtype in (np.int64, np.int32):
            adata.X = adata.X.astype(np.float32)

    # Basic QC filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Mitochondrial gene filtering
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
    parser = argparse.ArgumentParser(description="Prepare ALL local h5ad data for CLOP-DiT v0.2")
    parser.add_argument("--output_dir", type=str, default="data/processed_h5ad")
    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument("--max_cells", type=int, default=3000,
                        help="Subsample to at most this many cells per dataset")
    parser.add_argument("--subsample_seed", type=int, default=0,
                        help="Random seed for sc.pp.subsample (reproducibility)")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip preprocessing, just generate metadata")
    parser.add_argument("--data_dirs", nargs="+", default=None,
                        help="Override default dataset directories")
    args = parser.parse_args()

    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover h5ad files from all source directories
    data_dirs = [Path(d) for d in args.data_dirs] if args.data_dirs else DATASET_DIRS
    h5ad_files = []
    for ddir in data_dirs:
        if ddir.is_dir():
            h5ad_files.extend(sorted(ddir.glob("*.h5ad")))
        elif ddir.is_file() and ddir.suffix == ".h5ad":
            h5ad_files.append(ddir)

    print(f"\n{'='*70}")
    print(f"CLOP-DiT v0.3 Data Preparation")
    print(f"{'='*70}")
    print(f"Found {len(h5ad_files)} h5ad files across {len(data_dirs)} directories")
    print(f"Curated descriptions available for {len(DATASET_DESCRIPTIONS)} datasets")
    print(f"Datasets to skip (incompatible): {len(SKIP_DATASETS)}")
    print(f"Max cells per dataset: {args.max_cells}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")

    metadata = {}
    processed_files = []
    skipped = []
    total_cells = 0

    for i, h5ad_path in enumerate(h5ad_files):
        dataset_id = h5ad_path.stem
        print(f"\n[{i+1}/{len(h5ad_files)}] Processing: {dataset_id}")
        print(f"  Source: {h5ad_path}")

        # v0.3: Skip incompatible datasets
        if dataset_id in SKIP_DATASETS:
            print(f"  ✗ SKIPPED — incompatible with scGPT (normalized-only or Ensembl IDs)")
            skipped.append(dataset_id)
            continue

        # Get curated text description
        if dataset_id in DATASET_DESCRIPTIONS:
            text_desc = DATASET_DESCRIPTIONS[dataset_id]
            print(f"  ✓ Using curated biological description")
        else:
            # Fallback for any unknown datasets
            text_desc = f"Single-cell RNA sequencing data from dataset {dataset_id}."
            print(f"  ⚠ No curated description, using generic fallback")

        try:
            adata = sc.read_h5ad(h5ad_path)
            print(f"  Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

            # Subsample BEFORE preprocessing to save memory/time
            if args.max_cells and adata.shape[0] > args.max_cells:
                sc.pp.subsample(adata, n_obs=args.max_cells,
                                random_state=args.subsample_seed)
                print(f"  Subsampled to {adata.shape[0]} cells "
                      f"(seed={args.subsample_seed})")

            if not args.skip_preprocess:
                adata = preprocess_adata(
                    adata, dataset_id, n_top_genes=args.n_top_genes,
                )
                if adata is None:
                    skipped.append(dataset_id)
                    continue

                # Save processed
                out_path = output_dir / f"{dataset_id}_processed.h5ad"
                adata.write_h5ad(out_path)
                processed_files.append(str(out_path))
                print(f"  Saved: {out_path} ({adata.shape[0]} cells × {adata.shape[1]} genes)")
            else:
                processed_files.append(str(h5ad_path))

            n_cells = int(adata.shape[0])
            total_cells += n_cells

            # Store metadata with curated text
            metadata[dataset_id] = {
                "text": text_desc,
                "n_cells": n_cells,
                "n_genes": int(adata.shape[1]),
                "source_file": str(h5ad_path),
                "source_dir": h5ad_path.parent.name,
            }

            print(f"  Text: {text_desc[:80]}...")

        except Exception as e:
            logger.error(f"Failed to process {dataset_id}: {e}")
            import traceback
            traceback.print_exc()
            skipped.append(dataset_id)
            continue

    # Save metadata
    meta_path = output_dir / "metadata_structured.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save file list
    filelist_path = output_dir / "processed_files.json"
    with open(filelist_path, "w") as f:
        json.dump(processed_files, f, indent=2)

    print(f"\n{'='*70}")
    print(f"CLOP-DiT v0.3 Data Preparation Complete")
    print(f"{'='*70}")
    print(f"  Processed:  {len(processed_files)} datasets")
    print(f"  Skipped:    {len(skipped)} datasets")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Unique text descriptions: {len(set(m['text'] for m in metadata.values()))}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Metadata:    {meta_path}")
    if skipped:
        print(f"  Skipped datasets: {skipped}")
    print(f"{'='*70}")
    print(f"\nNext: Run 03_cache_latents.py --cell_encoder scgpt --scgpt_dir models/scgpt_human")


if __name__ == "__main__":
    main()
