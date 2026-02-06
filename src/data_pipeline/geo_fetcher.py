# geo_fetcher.py — GEO data acquisition and preprocessing
"""
Automated GEO dataset fetcher for building the CLOP-DiT training corpus.

Handles:
    1. Querying NCBI GEO for scRNA-seq datasets by keyword
    2. Downloading expression matrices (h5ad, mtx, csv)
    3. Extracting raw metadata text for SFT cleaning
    4. Basic QC filtering (min genes, min cells, doublet removal)
    5. HVG selection and normalization
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import scanpy as sc
import anndata as ad

logger = logging.getLogger(__name__)


class GEOFetcher:
    """Fetch and preprocess single-cell datasets from NCBI GEO.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save processed h5ad files.
    raw_dir : str or Path
        Directory for raw downloads.
    min_genes : int
        Minimum genes per cell for QC filtering.
    min_cells : int
        Minimum cells per gene for QC filtering.
    n_top_genes : int
        Number of highly variable genes to select.
    target_sum : float or None
        Target sum for normalization. None = median.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "data/processed_h5ad",
        raw_dir: Union[str, Path] = "data/raw_geo",
        min_genes: int = 200,
        min_cells: int = 3,
        n_top_genes: int = 2000,
        target_sum: Optional[float] = 1e4,
    ):
        self.output_dir = Path(output_dir)
        self.raw_dir = Path(raw_dir)
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.n_top_genes = n_top_genes
        self.target_sum = target_sum

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch_geo_metadata(self, gse_id: str) -> Dict:
        """Fetch metadata for a GEO series.

        Parameters
        ----------
        gse_id : str
            GEO Series accession (e.g., 'GSE123456').

        Returns
        -------
        metadata : dict
            Raw metadata including title, summary, organism, etc.
        """
        try:
            import GEOparse
        except ImportError:
            raise ImportError("GEOparse is required: pip install GEOparse")

        logger.info(f"Fetching metadata for {gse_id}...")
        gse = GEOparse.get_GEO(geo=gse_id, destdir=str(self.raw_dir), silent=True)

        metadata = {
            "gse_id": gse_id,
            "title": gse.metadata.get("title", [""])[0],
            "summary": gse.metadata.get("summary", [""])[0],
            "overall_design": gse.metadata.get("overall_design", [""])[0],
            "organism": gse.metadata.get("platform_organism", [""])[0] if "platform_organism" in gse.metadata else "",
            "type": gse.metadata.get("type", [""])[0],
            "pubmed_id": gse.metadata.get("pubmed_id", [""]),
            "samples": {},
        }

        # Extract sample-level metadata
        for gsm_name, gsm in gse.gsms.items():
            sample_meta = {
                "title": gsm.metadata.get("title", [""])[0],
                "source": gsm.metadata.get("source_name_ch1", [""])[0],
                "characteristics": gsm.metadata.get("characteristics_ch1", []),
                "description": gsm.metadata.get("description", [""])[0],
                "organism": gsm.metadata.get("organism_ch1", [""])[0],
            }
            metadata["samples"][gsm_name] = sample_meta

        return metadata

    def fetch_batch(
        self,
        gse_ids: List[str],
        metadata_file: str = "metadata_raw.json",
    ) -> Dict:
        """Fetch metadata for multiple GEO series.

        Parameters
        ----------
        gse_ids : list of str
            List of GSE accessions.
        metadata_file : str
            Output filename for combined metadata.

        Returns
        -------
        all_metadata : dict
            Combined metadata from all series.
        """
        all_metadata = {}
        failed = []

        for gse_id in gse_ids:
            try:
                meta = self.fetch_geo_metadata(gse_id)
                all_metadata[gse_id] = meta
                logger.info(f"✓ {gse_id}: {meta['title'][:80]}")
            except Exception as e:
                logger.warning(f"✗ {gse_id}: {e}")
                failed.append(gse_id)

        # Save
        out_path = self.output_dir / metadata_file
        with open(out_path, "w") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Fetched {len(all_metadata)}/{len(gse_ids)} datasets. "
                     f"Failed: {len(failed)}. Saved to {out_path}")

        return all_metadata

    def preprocess_h5ad(
        self,
        adata: ad.AnnData,
        dataset_id: str = "unknown",
    ) -> ad.AnnData:
        """Standard preprocessing pipeline for scRNA-seq data.

        Parameters
        ----------
        adata : AnnData
            Raw count matrix.
        dataset_id : str
            Identifier for logging.

        Returns
        -------
        adata : AnnData
            Preprocessed AnnData with HVGs selected.
        """
        logger.info(f"[{dataset_id}] Raw: {adata.shape[0]} cells × {adata.shape[1]} genes")

        # Basic QC
        sc.pp.filter_cells(adata, min_genes=self.min_genes)
        sc.pp.filter_genes(adata, min_cells=self.min_cells)

        # Mitochondrial gene filtering
        adata.var["mt"] = adata.var_names.str.startswith("MT-") | adata.var_names.str.startswith("mt-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        adata = adata[adata.obs["pct_counts_mt"] < 20].copy()

        logger.info(f"[{dataset_id}] After QC: {adata.shape[0]} cells × {adata.shape[1]} genes")

        # Store raw counts
        adata.layers["counts"] = adata.X.copy()

        # Normalize + log1p
        sc.pp.normalize_total(adata, target_sum=self.target_sum)
        sc.pp.log1p(adata)

        # HVG selection
        if adata.shape[1] > self.n_top_genes:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=self.n_top_genes, flavor="seurat_v3",
                layer="counts", subset=False,
            )
            adata = adata[:, adata.var["highly_variable"]].copy()

        logger.info(f"[{dataset_id}] Final: {adata.shape[0]} cells × {adata.shape[1]} genes")

        return adata

    def process_and_save(
        self,
        h5ad_path: Union[str, Path],
        dataset_id: str,
    ) -> Path:
        """Load, preprocess, and save an h5ad file.

        Parameters
        ----------
        h5ad_path : path
            Path to raw h5ad file.
        dataset_id : str
            Dataset identifier.

        Returns
        -------
        output_path : Path
            Path to saved processed file.
        """
        adata = sc.read_h5ad(h5ad_path)
        adata = self.preprocess_h5ad(adata, dataset_id=dataset_id)

        output_path = self.output_dir / f"{dataset_id}_processed.h5ad"
        adata.write_h5ad(output_path)
        logger.info(f"Saved processed data to {output_path}")

        return output_path


# ============================================================================
#  Utility: Curated Dataset Lists
# ============================================================================

# Example lung cancer datasets for initial pilot
LUNG_CANCER_PILOT = [
    "GSE131907",  # NSCLC single-cell atlas
    "GSE148071",  # Lung adenocarcinoma
    "GSE117570",  # Lung cancer microenvironment
    "GSE143423",  # Lung adenocarcinoma treatment
    "GSE154826",  # NSCLC immunotherapy
    "GSE136246",  # Small cell lung cancer
    "GSE127465",  # Lung myeloid cells
    "GSE150660",  # COVID/Lung tissue
    "GSE135893",  # Pulmonary fibrosis
    "GSE130148",  # Lung development
]

MULTI_TISSUE_PILOT = {
    "brain": ["GSE138852", "GSE160936"],
    "liver": ["GSE136103", "GSE124395"],
    "pancreas": ["GSE84133", "GSE114297"],
    "kidney": ["GSE131685", "GSE140989"],
    "heart": ["GSE109816", "GSE121893"],
}
