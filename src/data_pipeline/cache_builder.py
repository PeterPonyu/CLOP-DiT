# cache_builder.py — Pre-compute and cache latent embeddings
"""
Latent Cache Builder: Pre-computes cell embeddings (scGPT) and text embeddings
(PubMedBERT) and saves them as memory-mapped numpy arrays.

This is the critical optimization step that decouples the heavy encoder models
from the lightweight CLOP and DiT training loops.

Outputs:
    cached_latents/
    ├── cell_embeddings.npy     # (N_total, cell_dim)
    ├── text_embeddings.npy     # (N_total, text_dim)
    ├── sample_ids.npy          # (N_total,) integer sample IDs
    ├── metadata.json           # Mapping of sample_id → text description
    └── manifest.json           # Dataset statistics and configuration
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import scanpy as sc
import anndata as ad

logger = logging.getLogger(__name__)


class LatentCacheBuilder:
    """Pre-compute and cache embeddings from frozen encoders.

    Parameters
    ----------
    cache_dir : str or Path
        Directory to save cached embeddings.
    cell_encoder : str
        Cell encoder type ('scgpt' or 'scanpy_pca').
    text_encoder : str
        Text encoder model name.
    cell_dim : int
        Expected cell embedding dimension.
    text_dim : int
        Expected text embedding dimension.
    device : str
        Computation device.
    batch_size : int
        Batch size for encoding.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/cached_latents",
        cell_encoder: str = "scgpt",
        text_encoder: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        cell_dim: int = 512,
        text_dim: int = 768,
        device: str = "cuda",
        batch_size: int = 64,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cell_encoder_type = cell_encoder
        self.text_encoder_name = text_encoder
        self.cell_dim = cell_dim
        self.text_dim = text_dim
        self.device = device
        self.batch_size = batch_size

        self._text_model = None
        self._text_tokenizer = None

    # ========================================================================
    #  Text Encoding (PubMedBERT)
    # ========================================================================

    def _load_text_encoder(self):
        """Load PubMedBERT for text encoding."""
        if self._text_model is not None:
            return

        from transformers import AutoTokenizer, AutoModel

        logger.info(f"Loading text encoder: {self.text_encoder_name}")
        self._text_tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
        self._text_model = AutoModel.from_pretrained(self.text_encoder_name).to(self.device)
        self._text_model.eval()
        logger.info("Text encoder loaded.")

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text strings to embeddings.

        Parameters
        ----------
        texts : list of str
            Text descriptions to encode.

        Returns
        -------
        embeddings : (N, text_dim) numpy array
        """
        self._load_text_encoder()

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self._text_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            outputs = self._text_model(**inputs)
            # Use [CLS] token embedding
            cls_emb = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_emb.cpu().numpy())

            if (i // self.batch_size) % 10 == 0:
                logger.info(f"Text encoding: {i + len(batch_texts)}/{len(texts)}")

        return np.concatenate(all_embeddings, axis=0)

    # ========================================================================
    #  Cell Encoding (scGPT or PCA fallback)
    # ========================================================================

    @torch.no_grad()
    def encode_cells_scgpt(
        self,
        adata: ad.AnnData,
        scgpt_model_dir: str,
    ) -> np.ndarray:
        """Encode cells using scGPT.

        Parameters
        ----------
        adata : AnnData
            Preprocessed single-cell data.
        scgpt_model_dir : str
            Path to scGPT model weights.

        Returns
        -------
        embeddings : (N_cells, cell_dim) numpy array
        """
        from ..architecture.scgpt_embed import ScGPTCellEncoder

        if not hasattr(self, "_scgpt_encoder") or self._scgpt_encoder is None:
            self._scgpt_encoder = ScGPTCellEncoder(
                model_dir=scgpt_model_dir,
                device=self.device,
                max_length=1200,
                batch_size=self.batch_size,
            )

        embeddings = self._scgpt_encoder.encode(adata)
        logger.info(f"scGPT embeddings: {embeddings.shape}")
        return embeddings.astype(np.float32)

    def encode_cells_pca(
        self,
        adata: ad.AnnData,
        n_components: int = 512,
    ) -> np.ndarray:
        """Encode cells using PCA as a fallback.

        Suitable for initial development and testing before scGPT integration.

        Parameters
        ----------
        adata : AnnData
            Preprocessed single-cell data.
        n_components : int
            Number of PCA components.

        Returns
        -------
        embeddings : (N_cells, n_components) numpy array
        """
        logger.info(f"Computing PCA embeddings ({n_components} components)...")

        # Ensure data is dense for PCA
        import scipy.sparse as sp
        if sp.issparse(adata.X):
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=min(n_components, adata.shape[1] - 1))
            embeddings = svd.fit_transform(adata.X)
        else:
            sc.tl.pca(adata, n_comps=min(n_components, adata.shape[1] - 1))
            embeddings = adata.obsm["X_pca"]

        # Pad to cell_dim if needed
        if embeddings.shape[1] < self.cell_dim:
            pad = np.zeros((embeddings.shape[0], self.cell_dim - embeddings.shape[1]))
            embeddings = np.concatenate([embeddings, pad], axis=1)

        logger.info(f"PCA embeddings: {embeddings.shape}")
        return embeddings[:, :self.cell_dim].astype(np.float32)

    # ========================================================================
    #  Full Cache Building Pipeline
    # ========================================================================

    def build_cache(
        self,
        h5ad_files: List[Union[str, Path]],
        metadata_file: Union[str, Path],
        cell_encoder_method: str = "pca",
        scgpt_model_dir: Optional[str] = None,
    ) -> Dict:
        """Build the complete latent cache from processed h5ad files and metadata.

        Parameters
        ----------
        h5ad_files : list of paths
            Preprocessed h5ad files (one per dataset).
        metadata_file : path
            Structured metadata JSON (from TextCleaner).
        cell_encoder_method : str
            'scgpt' or 'pca'.
        scgpt_model_dir : str, optional
            Required if cell_encoder_method == 'scgpt'.

        Returns
        -------
        manifest : dict
            Cache statistics.
        """
        # Load structured metadata
        with open(metadata_file) as f:
            metadata = json.load(f)

        all_cell_emb = []
        all_text_emb = []
        all_sample_ids = []
        id_to_text = {}
        sample_counter = 0

        for h5ad_path in h5ad_files:
            h5ad_path = Path(h5ad_path)
            dataset_id = h5ad_path.stem.replace("_processed", "")

            logger.info(f"Processing {dataset_id}...")

            # Load AnnData
            adata = sc.read_h5ad(h5ad_path)

            # Cell encoding
            if cell_encoder_method == "scgpt" and scgpt_model_dir:
                cell_emb = self.encode_cells_scgpt(adata, scgpt_model_dir)
            else:
                cell_emb = self.encode_cells_pca(adata)

            # Get text description for this dataset
            dataset_meta = metadata.get(dataset_id, {})
            if isinstance(dataset_meta, dict) and "text" in dataset_meta:
                text_desc = dataset_meta["text"]
            elif isinstance(dataset_meta, dict) and "series" in dataset_meta:
                text_desc = self._metadata_to_text(dataset_meta["series"])
            elif isinstance(dataset_meta, dict):
                text_desc = self._metadata_to_text(dataset_meta)
            else:
                text_desc = f"Single-cell RNA sequencing data from {dataset_id}."

            # Text encoding (same text for all cells in the dataset)
            text_emb_single = self.encode_texts([text_desc])  # (1, text_dim)
            text_emb = np.repeat(text_emb_single, cell_emb.shape[0], axis=0)

            # Sample IDs
            sample_ids = np.full(cell_emb.shape[0], sample_counter, dtype=np.int64)
            id_to_text[str(sample_counter)] = text_desc
            sample_counter += 1

            all_cell_emb.append(cell_emb)
            all_text_emb.append(text_emb)
            all_sample_ids.append(sample_ids)

            logger.info(f"  {dataset_id}: {cell_emb.shape[0]} cells, text='{text_desc[:60]}...'")

        # Concatenate
        cell_embeddings = np.concatenate(all_cell_emb, axis=0)
        text_embeddings = np.concatenate(all_text_emb, axis=0)
        sample_ids = np.concatenate(all_sample_ids, axis=0)

        # Save
        np.save(self.cache_dir / "cell_embeddings.npy", cell_embeddings)
        np.save(self.cache_dir / "text_embeddings.npy", text_embeddings)
        np.save(self.cache_dir / "sample_ids.npy", sample_ids)

        with open(self.cache_dir / "metadata.json", "w") as f:
            json.dump(id_to_text, f, indent=2)

        manifest = {
            "total_cells": int(cell_embeddings.shape[0]),
            "cell_dim": int(cell_embeddings.shape[1]),
            "text_dim": int(text_embeddings.shape[1]),
            "num_datasets": sample_counter,
            "cell_encoder": cell_encoder_method,
            "text_encoder": self.text_encoder_name,
        }
        with open(self.cache_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Cache built: {manifest}")
        return manifest

    @staticmethod
    def _metadata_to_text(meta: Dict) -> str:
        """Convert structured metadata dict to natural language."""
        parts = []

        tissue = meta.get("tissue", "unknown")
        disease = meta.get("disease", "unknown")
        cell_type = meta.get("cell_type", "unknown")
        drug = meta.get("drug_treatment", "none")
        organism = meta.get("organism", "unknown")

        if tissue != "unknown":
            parts.append(f"{tissue} tissue")
        if disease not in ("unknown", "healthy", "normal"):
            parts.append(f"with {disease}")
        if cell_type not in ("unknown", "mixed"):
            parts.append(f"focusing on {cell_type}")
        if drug != "none":
            parts.append(f"treated with {drug}")
        if organism != "unknown":
            parts.append(f"from {organism}")

        if parts:
            return "Single-cell RNA sequencing of " + " ".join(parts) + "."
        else:
            summary = meta.get("condition_summary", "")
            return summary if summary else "Single-cell RNA sequencing data."
