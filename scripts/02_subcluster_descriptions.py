#!/usr/bin/env python3
# 02_subcluster_descriptions.py — Automated sub-cluster text description generation
"""
CLOP-DiT v0.4: Expand text-cell alignment data by automatically clustering
each dataset and generating per-cluster biological text descriptions.

Motivation:
    CLIP-style contrastive training typically requires 400K+ text-image pairs
    (CLIP used 400M). With only 69 dataset-level descriptions, text diversity
    is too low for effective alignment. This script:
    
    1. Leiden-clusters each preprocessed h5ad file
    2. Identifies top marker genes per cluster (Wilcoxon rank-sum test)
    3. Maps marker genes to known cell type signatures
    4. Generates descriptive text per cluster (automated biological annotation)
    5. Updates metadata to assign per-cluster texts to individual cells

    Result: Expand from 69 dataset texts → ~500-1000 sub-cluster texts,
    giving CLOP a much richer text-cell mapping space.

Usage:
    python scripts/02_subcluster_descriptions.py \
        --h5ad_dir data/processed_h5ad \
        --metadata data/processed_h5ad/metadata_structured.json \
        --output_dir data/processed_h5ad \
        --resolution 0.8
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ============================================================================
#  Known cell type marker gene signatures (curated from CellMarker/PanglaoDB)
# ============================================================================

CELL_TYPE_SIGNATURES = {
    # ── Immune: Lymphoid ──
    "CD8+ T cells": {
        "markers": ["CD8A", "CD8B", "CD3E", "CD3D", "GZMB", "PRF1", "IFNG", "NKG7", "GZMA", "GZMK"],
        "description": "CD8+ cytotoxic T lymphocytes",
    },
    "CD4+ T cells": {
        "markers": ["CD4", "IL7R", "CD3E", "CD3D", "TCF7", "LEF1", "CCR7", "SELL", "CD28"],
        "description": "CD4+ helper T lymphocytes",
    },
    "Regulatory T cells": {
        "markers": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18", "CD4"],
        "description": "CD4+FOXP3+ regulatory T cells (Tregs)",
    },
    "NK cells": {
        "markers": ["NKG7", "KLRD1", "GNLY", "NCR1", "NCAM1", "KLRF1", "FCGR3A", "CD160", "GZMB"],
        "description": "natural killer cells",
    },
    "B cells": {
        "markers": ["CD79A", "CD79B", "MS4A1", "CD19", "PAX5", "BANK1", "BLK", "IGHM", "IGHD"],
        "description": "B lymphocytes",
    },
    "Plasma cells": {
        "markers": ["JCHAIN", "MZB1", "SDC1", "IGHG1", "IGHG2", "IGHA1", "XBP1", "PRDM1"],
        "description": "antibody-secreting plasma cells",
    },

    # ── Immune: Myeloid ──
    "Macrophages": {
        "markers": ["CD68", "CD163", "CSF1R", "MRC1", "MSR1", "MARCO", "C1QA", "C1QB", "APOE"],
        "description": "tissue-resident macrophages",
    },
    "Monocytes": {
        "markers": ["CD14", "FCGR3A", "LYZ", "S100A8", "S100A9", "VCAN", "FCN1", "CST3"],
        "description": "circulating monocytes",
    },
    "Dendritic cells": {
        "markers": ["FCER1A", "CD1C", "CLEC10A", "ITGAX", "HLA-DRA", "HLA-DQA1", "IRF8", "BATF3"],
        "description": "dendritic cells",
    },
    "Neutrophils": {
        "markers": ["S100A8", "S100A9", "CSF3R", "FCGR3B", "CXCR2", "MMP9", "ELANE", "MPO"],
        "description": "neutrophils",
    },
    "Mast cells": {
        "markers": ["KIT", "CPA3", "TPSAB1", "TPSB2", "HPGDS", "MS4A2", "HDC"],
        "description": "mast cells",
    },

    # ── Epithelial ──
    "Epithelial cells": {
        "markers": ["EPCAM", "KRT18", "KRT8", "CDH1", "KRT19", "MUC1", "CLDN4", "TJP1"],
        "description": "epithelial cells",
    },
    "Alveolar type 2": {
        "markers": ["SFTPC", "SFTPA1", "SFTPA2", "SFTPB", "ABCA3", "SLC34A2", "LAMP3"],
        "description": "alveolar type 2 pneumocytes",
    },
    "Basal cells": {
        "markers": ["KRT5", "KRT14", "KRT17", "TP63", "S100A2", "NGFR"],
        "description": "basal epithelial cells",
    },

    # ── Stromal ──
    "Fibroblasts": {
        "markers": ["COL1A1", "COL1A2", "DCN", "LUM", "FAP", "PDGFRA", "VIM", "FN1", "ACTA2"],
        "description": "fibroblasts / mesenchymal stromal cells",
    },
    "Endothelial cells": {
        "markers": ["PECAM1", "VWF", "CDH5", "FLT1", "KDR", "ENG", "CLDN5", "EMCN"],
        "description": "vascular endothelial cells",
    },
    "Smooth muscle cells": {
        "markers": ["ACTA2", "MYH11", "TAGLN", "CNN1", "DES", "CALD1", "MYL9"],
        "description": "smooth muscle cells",
    },
    "Pericytes": {
        "markers": ["RGS5", "PDGFRB", "NOTCH3", "MCAM", "KCNJ8", "ABCC9"],
        "description": "pericytes / mural cells",
    },

    # ── Proliferating ──
    "Proliferating cells": {
        "markers": ["MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "TYMS", "MCM2", "BIRC5", "UBE2C"],
        "description": "actively proliferating cells",
    },

    # ── Neural ──
    "Neurons": {
        "markers": ["RBFOX3", "SNAP25", "SYN1", "SYP", "MAP2", "TUBB3", "ENO2", "SLC17A7"],
        "description": "neurons",
    },
    "Astrocytes": {
        "markers": ["GFAP", "AQP4", "SLC1A3", "S100B", "ALDH1L1", "GJA1", "SOX9"],
        "description": "astrocytes",
    },
    "Oligodendrocytes": {
        "markers": ["MBP", "MOG", "PLP1", "MAG", "OLIG1", "OLIG2", "SOX10"],
        "description": "oligodendrocytes",
    },
    "Microglia": {
        "markers": ["CX3CR1", "P2RY12", "TMEM119", "CSF1R", "AIF1", "HEXB", "TREM2"],
        "description": "microglia",
    },

    # ── Stem / Progenitor ──
    "HSCs/Progenitors": {
        "markers": ["CD34", "KIT", "FLT3", "PROM1", "THY1", "CRHBP", "MLLT3", "HLF", "HOPX"],
        "description": "hematopoietic stem and progenitor cells",
    },
    "Erythroid progenitors": {
        "markers": ["GYPA", "GYPB", "HBB", "HBA1", "HBA2", "ALAS2", "KLF1", "TFRC"],
        "description": "erythroid lineage cells",
    },

    # ── Hepatic ──
    "Hepatocytes": {
        "markers": ["ALB", "APOB", "APOA1", "HP", "TF", "TTR", "SERPINA1", "CYP3A4"],
        "description": "hepatocytes",
    },
    "Cholangiocytes": {
        "markers": ["KRT7", "KRT19", "SOX9", "EPCAM", "SPP1", "ANXA4", "FXYD2"],
        "description": "cholangiocytes / bile duct epithelial cells",
    },
}


def annotate_cluster(
    top_markers: List[str],
    dataset_text: str,
    cluster_id: int,
    n_cells: int,
) -> Tuple[str, str, float]:
    """
    Annotate a cluster based on its top marker genes.
    
    Returns: (cell_type_name, description_text, confidence_score)
    """
    best_type = None
    best_score = 0.0
    best_matched = []

    top_set = set([m.upper() for m in top_markers[:30]])  # use top 30 markers

    for ct_name, ct_info in CELL_TYPE_SIGNATURES.items():
        sig_markers = set([m.upper() for m in ct_info["markers"]])
        overlap = top_set & sig_markers
        # Jaccard-like score weighted toward specificity
        if len(sig_markers) > 0:
            score = len(overlap) / min(len(sig_markers), 8)  # cap denominator
        else:
            score = 0
        if score > best_score:
            best_score = score
            best_type = ct_name
            best_matched = sorted(overlap)

    if best_score >= 0.25 and best_type is not None:
        ct_desc = CELL_TYPE_SIGNATURES[best_type]["description"]
        matched_str = ", ".join(best_matched[:5])
        
        # Build cluster-specific description combining dataset context + cell type
        # Extract tissue/disease context from dataset description
        tissue_context = _extract_context(dataset_text)
        
        description = (
            f"Sub-population of {ct_desc} (cluster {cluster_id}, n={n_cells} cells) "
            f"identified by expression of {matched_str}. "
            f"Context: {tissue_context}"
        )
        return best_type, description, best_score
    else:
        # Unknown cluster — use top markers for description
        top_5 = ", ".join(top_markers[:5])
        tissue_context = _extract_context(dataset_text)
        description = (
            f"Uncharacterized cell population (cluster {cluster_id}, n={n_cells} cells) "
            f"with high expression of {top_5}. "
            f"Context: {tissue_context}"
        )
        return "Unknown", description, best_score


def _extract_context(dataset_text: str) -> str:
    """Extract tissue/disease context from dataset-level description."""
    # Take first sentence as context
    sentences = dataset_text.split(". ")
    if len(sentences) >= 1:
        return sentences[0].strip().rstrip(".")
    return dataset_text[:150]


def cluster_and_annotate(
    adata: sc.AnnData,
    dataset_id: str,
    dataset_text: str,
    resolution: float = 0.8,
    min_cluster_size: int = 20,
) -> Dict:
    """
    Cluster an AnnData and annotate each cluster.
    
    Returns dict mapping cluster_id → {cell_type, text, confidence, n_cells, markers}
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    adata_work = adata.copy()
    
    # PCA → neighbors → Leiden
    try:
        if "X_pca" not in adata_work.obsm:
            n_comps = min(30, adata_work.shape[0] - 1, adata_work.shape[1] - 1)
            if n_comps < 5:
                logger.warning(f"[{dataset_id}] Too small for clustering ({adata_work.shape})")
                return {}
            sc.tl.pca(adata_work, n_comps=n_comps)
        
        sc.pp.neighbors(adata_work, n_neighbors=15, n_pcs=min(30, adata_work.obsm["X_pca"].shape[1]))
        sc.tl.leiden(adata_work, resolution=resolution, key_added="leiden")
    except Exception as e:
        logger.warning(f"[{dataset_id}] Clustering failed: {e}")
        return {}
    
    clusters = adata_work.obs["leiden"].unique()
    logger.info(f"[{dataset_id}] Found {len(clusters)} clusters (resolution={resolution})")
    
    # Rank genes per cluster
    try:
        sc.tl.rank_genes_groups(adata_work, "leiden", method="wilcoxon", n_genes=50)
    except Exception as e:
        logger.warning(f"[{dataset_id}] Marker gene ranking failed: {e}")
        return {}
    
    cluster_annotations = {}
    
    for cluster_id in sorted(clusters, key=int):
        mask = adata_work.obs["leiden"] == cluster_id
        n_cells = int(mask.sum())
        
        if n_cells < min_cluster_size:
            continue
        
        # Get top marker genes for this cluster
        try:
            top_markers = list(adata_work.uns["rank_genes_groups"]["names"][cluster_id][:30])
        except (KeyError, IndexError):
            try:
                # Different scanpy versions store results differently
                result = adata_work.uns["rank_genes_groups"]
                gene_names = [result["names"][i][int(cluster_id)] for i in range(min(30, len(result["names"])))]
                top_markers = gene_names
            except Exception:
                continue
        
        cell_type, description, confidence = annotate_cluster(
            top_markers, dataset_text, int(cluster_id), n_cells
        )
        
        cluster_annotations[str(cluster_id)] = {
            "cell_type": cell_type,
            "text": description,
            "confidence": round(float(confidence), 3),
            "n_cells": int(n_cells),
            "top_markers": top_markers[:10],
            "cell_indices": [int(x) for x in np.where(mask)[0]],
        }
        
        logger.info(
            f"  Cluster {cluster_id}: {cell_type} ({confidence:.2f}) "
            f"— {n_cells} cells — {', '.join(top_markers[:5])}"
        )
    
    return cluster_annotations


def main():
    parser = argparse.ArgumentParser(
        description="CLOP-DiT v0.4: Sub-cluster text description generation"
    )
    parser.add_argument("--h5ad_dir", type=str, default="data/processed_h5ad")
    parser.add_argument("--metadata", type=str, default="data/processed_h5ad/metadata_structured.json")
    parser.add_argument("--output_dir", type=str, default="data/processed_h5ad")
    parser.add_argument("--resolution", type=float, default=0.8,
                        help="Leiden clustering resolution (higher = more clusters)")
    parser.add_argument("--min_cluster_size", type=int, default=20,
                        help="Minimum cells per cluster to annotate")
    args = parser.parse_args()

    setup_logging()
    
    h5ad_dir = Path(args.h5ad_dir)
    output_dir = Path(args.output_dir)
    
    # Load structured metadata
    with open(args.metadata) as f:
        metadata = json.load(f)
    
    # Find all processed h5ad files
    h5ad_files = sorted(h5ad_dir.glob("*_processed.h5ad"))
    
    print(f"\n{'='*70}")
    print(f"CLOP-DiT v0.4 Sub-Cluster Text Generation")
    print(f"{'='*70}")
    print(f"Datasets: {len(h5ad_files)}")
    print(f"Resolution: {args.resolution}")
    print(f"Min cluster size: {args.min_cluster_size}")
    print(f"{'='*70}\n")
    
    all_subcluster_meta = {}
    total_clusters = 0
    total_annotated = 0
    unique_texts = set()
    
    for i, h5ad_path in enumerate(h5ad_files):
        dataset_id = h5ad_path.stem.replace("_processed", "")
        print(f"\n[{i+1}/{len(h5ad_files)}] {dataset_id}")
        
        # Get dataset-level text
        if dataset_id in metadata:
            meta = metadata[dataset_id]
            dataset_text = meta["text"] if isinstance(meta, dict) else meta
        else:
            dataset_text = f"Single-cell RNA sequencing data from {dataset_id}."
            print(f"  ⚠ No metadata for {dataset_id}, using generic text")
        
        # Load and cluster
        try:
            adata = sc.read_h5ad(h5ad_path)
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue
        
        cluster_annotations = cluster_and_annotate(
            adata, dataset_id, dataset_text,
            resolution=args.resolution,
            min_cluster_size=args.min_cluster_size,
        )
        
        if cluster_annotations:
            all_subcluster_meta[dataset_id] = {
                "dataset_text": dataset_text,
                "n_cells_total": int(adata.shape[0]),
                "n_clusters": len(cluster_annotations),
                "clusters": cluster_annotations,
            }
            total_clusters += len(cluster_annotations)
            
            for cid, cinfo in cluster_annotations.items():
                if cinfo["confidence"] >= 0.25:
                    total_annotated += 1
                unique_texts.add(cinfo["text"])
            
            print(f"  ✓ {len(cluster_annotations)} clusters annotated")
        else:
            print(f"  ✗ No clusters generated")
    
    # Save sub-cluster metadata
    subcluster_path = output_dir / "subcluster_metadata.json"
    with open(subcluster_path, "w") as f:
        json.dump(all_subcluster_meta, f, indent=2, ensure_ascii=False)
    
    # Create expanded metadata for cache builder
    # This maps each cell to its sub-cluster text (more fine-grained than dataset-level)
    expanded_metadata = {}
    expanded_id = 0
    
    for dataset_id, ds_info in all_subcluster_meta.items():
        for cluster_id, cluster_info in ds_info["clusters"].items():
            expanded_key = f"{dataset_id}__cluster_{cluster_id}"
            expanded_metadata[expanded_key] = {
                "text": cluster_info["text"],
                "cell_type": cluster_info["cell_type"],
                "confidence": cluster_info["confidence"],
                "n_cells": cluster_info["n_cells"],
                "source_dataset": dataset_id,
                "cluster_id": cluster_id,
                "top_markers": cluster_info["top_markers"],
            }
            expanded_id += 1
    
    expanded_path = output_dir / "metadata_subclusters.json"
    with open(expanded_path, "w") as f:
        json.dump(expanded_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"Sub-Cluster Analysis Complete")
    print(f"{'='*70}")
    print(f"  Datasets processed:    {len(all_subcluster_meta)}")
    print(f"  Total clusters:        {total_clusters}")
    print(f"  Annotated (conf≥0.25): {total_annotated}")
    print(f"  Unique text descriptions: {len(unique_texts)}")
    print(f"  Expansion ratio:       {len(unique_texts)} / {len(metadata)} "
          f"= {len(unique_texts)/max(len(metadata),1):.1f}x")
    print(f"  Sub-cluster metadata:  {subcluster_path}")
    print(f"  Expanded metadata:     {expanded_path}")
    print(f"{'='*70}")
    print(f"\nNext: Run 03_cache_latents.py with --subcluster_metadata {expanded_path}")


if __name__ == "__main__":
    main()
