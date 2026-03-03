# cache_builder.py — Pre-compute and cache latent embeddings
"""
Latent Cache Builder: Pre-computes cell embeddings (scGPT pan-cancer) and
text embeddings (BiomedBERT-large) and saves them as memory-mapped numpy arrays.

This is the critical optimization step that decouples the heavy encoder models
from the lightweight CLOP and DiT training loops.

v6.2: Supports DEDUPLICATED storage format and enriched multi-caption texts.

Outputs (v6.2 — deduplicated):
    cached_latents/
    ├── cell_embeddings.npy          # (N_total, cell_dim)
    ├── text_embeddings_unique.npy   # (N_unique_texts, text_dim)  ← deduplicated
    ├── text_group_ids.npy           # (N_total,) maps cell → unique text index
    ├── sample_ids.npy               # (N_total,) integer dataset IDs
    ├── text_strings.json            # {text_group_id: raw_text_string}
    ├── text_variants.json           # {text_group_id: [variant_1, variant_2, ...]}
    ├── metadata.json                # sample_id → dataset-level text
    └── manifest.json                # Cache statistics and configuration

Outputs (legacy — duplicated, for backward compatibility):
    cached_latents/
    ├── text_embeddings.npy          # (N_total, text_dim) — full duplication
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
        cache_dir: Union[str, Path] = "data/cached_latents_v5.2",
        cell_encoder: str = "scgpt",
        text_encoder: str = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        cell_dim: int = 512,
        text_dim: int = 1024,
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

        # Clamp components to valid range
        max_components = min(n_components, adata.shape[0] - 1, adata.shape[1] - 1)
        if max_components < 1:
            logger.warning(f"Dataset too small for PCA ({adata.shape}), returning zeros.")
            return np.zeros((adata.shape[0], self.cell_dim), dtype=np.float32)

        # Ensure data is dense for PCA
        import scipy.sparse as sp
        if sp.issparse(adata.X):
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=max_components)
            embeddings = svd.fit_transform(adata.X)
        else:
            sc.tl.pca(adata, n_comps=max_components)
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
        cell_encoder_method: str = "scgpt",
        scgpt_model_dir: str = "models/scgpt_human",
        subcluster_metadata_file: Optional[Union[str, Path]] = None,
        deduplicate: bool = True,
        use_variants: bool = False,
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
        subcluster_metadata_file : path, optional
            Sub-cluster metadata JSON (from 02_subcluster_descriptions.py or
            02b_enrich_descriptions.py). If provided, assigns per-cluster
            text descriptions to individual cells.
        deduplicate : bool
            If True (default), store text embeddings in deduplicated format
            (text_embeddings_unique.npy + text_group_ids.npy) saving ~99% space.
            Also saves legacy text_embeddings.npy for backward compatibility.
        use_variants : bool
            If True and subcluster_metadata has 'text_variants', encode all
            variants and store in text_variant_embeddings_unique.npy.

        Returns
        -------
        manifest : dict
            Cache statistics.
        """
        # Load structured metadata
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Load sub-cluster metadata if available
        subcluster_meta = None
        if subcluster_metadata_file and Path(subcluster_metadata_file).exists():
            with open(subcluster_metadata_file) as f:
                subcluster_meta = json.load(f)
            # subcluster_metadata.json structure:
            #   dataset_id → {dataset_text, n_cells_total, n_clusters,
            #                  clusters: {cluster_id → {text, cell_indices, ...}}}
            logger.info(f"Loaded sub-cluster metadata for {len(subcluster_meta)} datasets")

        all_cell_emb = []
        all_text_emb = []
        all_sample_ids = []
        all_cell_text_strings = []    # v6.2: raw text strings per cell
        all_cell_text_variants = []   # v6.2: caption variants per cell
        id_to_text = {}
        sample_counter = 0

        for h5ad_path in h5ad_files:
            h5ad_path = Path(h5ad_path)
            dataset_id = h5ad_path.stem.replace("_processed", "")

            logger.info(f"Processing {dataset_id}...")

            # Load AnnData
            adata = sc.read_h5ad(h5ad_path)

            # Cell encoding — scGPT is the standard encoder
            if cell_encoder_method == "scgpt":
                if not scgpt_model_dir:
                    raise ValueError("scGPT model directory required. Set --scgpt_dir.")
                try:
                    cell_emb = self.encode_cells_scgpt(adata, scgpt_model_dir)
                except ValueError as e:
                    logger.warning(
                        f"  scGPT encoding failed for {dataset_id}: {e}. "
                        f"SKIPPING this dataset (PCA embeddings are incompatible "
                        f"with scGPT embedding space)."
                    )
                    continue
            elif cell_encoder_method == "pca":
                logger.warning(
                    "PCA fallback is deprecated. scGPT provides universal pretrained "
                    "embeddings — use --cell_encoder scgpt for production runs."
                )
                cell_emb = self.encode_cells_pca(adata)
            else:
                raise ValueError(f"Unknown cell encoder: {cell_encoder_method}")

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

            # ── Sub-cluster text assignment ─────────────────────────────────
            # If sub-cluster metadata available, assign per-cluster texts to cells
            if subcluster_meta and dataset_id in subcluster_meta:
                ds_info = subcluster_meta[dataset_id]
                n_cells = cell_emb.shape[0]

                # Build per-cell text assignments
                # Cells not in any annotated cluster get the dataset-level text
                cell_texts = [text_desc] * n_cells
                cell_variants = [[] for _ in range(n_cells)]  # v6.2: variants per cell

                for cid, cinfo in ds_info.get("clusters", {}).items():
                    cluster_text = cinfo.get("text", text_desc)
                    text_variants = cinfo.get("text_variants", [])
                    cell_indices = cinfo.get("cell_indices", [])
                    for idx in cell_indices:
                        if idx < n_cells:
                            cell_texts[idx] = cluster_text
                            if text_variants:
                                cell_variants[idx] = text_variants

                # Get unique texts and encode them
                unique_texts = list(set(cell_texts))
                text_to_idx = {t: i for i, t in enumerate(unique_texts)}
                text_embs_unique = self.encode_texts(unique_texts)

                # Map cell_texts → text embeddings
                text_emb = np.zeros((n_cells, text_embs_unique.shape[1]), dtype=np.float32)
                for ci in range(n_cells):
                    text_emb[ci] = text_embs_unique[text_to_idx[cell_texts[ci]]]

                # Use one sample_id per dataset for grouping; text diversity is in embeddings
                sample_ids = np.full(n_cells, sample_counter, dtype=np.int64)
                id_to_text[str(sample_counter)] = text_desc
                sample_counter += 1

                # v6.2: collect raw text strings and variants
                all_cell_text_strings.append(cell_texts)
                all_cell_text_variants.append(cell_variants)

                logger.info(
                    f"  {dataset_id}: {n_cells} cells, "
                    f"{len(unique_texts)} sub-cluster texts (from {ds_info['n_clusters']} clusters)"
                )

            else:
                # Standard: single text for all cells in dataset
                text_emb_single = self.encode_texts([text_desc])  # (1, text_dim)
                text_emb = np.repeat(text_emb_single, cell_emb.shape[0], axis=0)

                # Sample IDs
                sample_ids = np.full(cell_emb.shape[0], sample_counter, dtype=np.int64)
                id_to_text[str(sample_counter)] = text_desc
                sample_counter += 1

                # v6.2: raw text strings (all same for this dataset)
                all_cell_text_strings.append([text_desc] * cell_emb.shape[0])
                all_cell_text_variants.append([[] for _ in range(cell_emb.shape[0])])

                logger.info(f"  {dataset_id}: {cell_emb.shape[0]} cells, text='{text_desc[:60]}...'")

            all_cell_emb.append(cell_emb)
            all_text_emb.append(text_emb)
            all_sample_ids.append(sample_ids)

            logger.info(f"  {dataset_id}: {cell_emb.shape[0]} cells, text='{text_desc[:60]}...'")

        # Concatenate
        cell_embeddings = np.concatenate(all_cell_emb, axis=0)
        text_embeddings = np.concatenate(all_text_emb, axis=0)
        sample_ids = np.concatenate(all_sample_ids, axis=0)

        # Save cell embeddings and sample IDs (always same format)
        np.save(self.cache_dir / "cell_embeddings.npy", cell_embeddings)
        np.save(self.cache_dir / "sample_ids.npy", sample_ids)

        # ── Deduplicated storage (v6.2) ─────────────────────────────────
        if deduplicate:
            # Find unique text embeddings and build index mapping
            # Use byte-level comparison for exact matching
            text_bytes = text_embeddings.view(np.uint8).reshape(text_embeddings.shape[0], -1)
            _, unique_idx, inverse_idx = np.unique(
                text_bytes, axis=0, return_index=True, return_inverse=True
            )

            text_emb_unique = text_embeddings[unique_idx]  # (N_unique, text_dim)
            text_group_ids = inverse_idx.astype(np.int32)  # (N_total,) → index into unique

            np.save(self.cache_dir / "text_embeddings_unique.npy", text_emb_unique)
            np.save(self.cache_dir / "text_group_ids.npy", text_group_ids)

            # Also collect raw text strings for each unique group
            # Build cell_idx → text_string mapping from alltext lists
            all_cell_texts_flat = []
            for text_list in all_cell_text_strings:
                all_cell_texts_flat.extend(text_list)

            text_strings = {}
            for uid in range(len(unique_idx)):
                # Find a cell with this group ID and get its text
                cell_idx = unique_idx[uid]
                text_strings[str(uid)] = all_cell_texts_flat[cell_idx]

            with open(self.cache_dir / "text_strings.json", "w") as f:
                json.dump(text_strings, f, indent=2, ensure_ascii=False)

            # Save text variants if available
            if all_cell_text_variants:
                all_variants_flat = []
                for variant_list in all_cell_text_variants:
                    all_variants_flat.extend(variant_list)

                text_variants_map = {}
                for uid in range(len(unique_idx)):
                    cell_idx = unique_idx[uid]
                    variants = all_variants_flat[cell_idx]
                    if variants:
                        text_variants_map[str(uid)] = variants

                if text_variants_map:
                    with open(self.cache_dir / "text_variants.json", "w") as f:
                        json.dump(text_variants_map, f, indent=2, ensure_ascii=False)

                    # Encode variant embeddings if requested
                    if use_variants:
                        all_variant_texts = []
                        variant_group_map = []  # (group_id, variant_idx)
                        for uid_str, variants in text_variants_map.items():
                            for vi, vtext in enumerate(variants):
                                all_variant_texts.append(vtext)
                                variant_group_map.append((int(uid_str), vi))

                        if all_variant_texts:
                            logger.info(f"Encoding {len(all_variant_texts)} caption variants...")
                            variant_embs = self.encode_texts(all_variant_texts)
                            np.save(self.cache_dir / "text_variant_embeddings.npy", variant_embs)
                            with open(self.cache_dir / "text_variant_map.json", "w") as f:
                                json.dump(variant_group_map, f)
                            logger.info(f"  Saved {variant_embs.shape[0]} variant embeddings")

            dedup_size = text_emb_unique.shape[0] * text_emb_unique.shape[1] * 4
            orig_size = text_embeddings.shape[0] * text_embeddings.shape[1] * 4
            logger.info(
                f"Deduplicated: {text_embeddings.shape[0]} → {text_emb_unique.shape[0]} "
                f"unique texts ({dedup_size/1e6:.1f} MB vs {orig_size/1e6:.1f} MB, "
                f"{100*(1-dedup_size/orig_size):.1f}% savings)"
            )

        # Legacy format: full duplicated text_embeddings.npy (backward compatible)
        np.save(self.cache_dir / "text_embeddings.npy", text_embeddings)

        with open(self.cache_dir / "metadata.json", "w") as f:
            json.dump(id_to_text, f, indent=2)

        manifest = {
            "total_cells": int(cell_embeddings.shape[0]),
            "cell_dim": int(cell_embeddings.shape[1]),
            "text_dim": int(text_embeddings.shape[1]),
            "num_datasets": sample_counter,
            "cell_encoder": cell_encoder_method,
            "text_encoder": self.text_encoder_name,
            "deduplicated": deduplicate,
            "format_version": "6.2",
        }
        if deduplicate:
            manifest["num_unique_texts"] = int(text_emb_unique.shape[0])

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
