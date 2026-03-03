#!/usr/bin/env python3
"""
Re-embed Polished Texts with BiomedBERT
========================================

After text polishing, all BiomedBERT embeddings must be recomputed because:
1. Polished texts are 2.4× longer and semantically richer
2. Cell-type biology is now front-loaded (was buried under tissue prefix)
3. Old embeddings were from rigid templates

This script:
1. Loads polished text_strings.json
2. Encodes all 1088 texts with BiomedBERT-large
3. Saves new text_embeddings.npy
4. Optionally recomputes ZCA whitening transforms

Usage:
    python scripts/re_embed_polished_texts.py \\
        --cache_dir data/cached_latents_v5.2 \\
        --recompute_whitening \\
        --batch_size 64
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BiomedBERTEncoder:
    """BiomedBERT-large encoder for text descriptions."""
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        device: str = "cuda",
        max_length: int = 512,
    ):
        self.device = device
        self.max_length = max_length
        
        logger.info(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        logger.info(f"✓ BiomedBERT loaded on {device}")
    
    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts to [CLS] embeddings.
        
        Parameters
        ----------
        texts : list of str
            Text descriptions to encode.
        
        Returns
        -------
        embeddings : (B, 1024) numpy array
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs)
        
        # Extract [CLS] embeddings
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
    
    def encode_all(
        self,
        text_dict: Dict[str, str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode all texts from a dictionary.
        
        Parameters
        ----------
        text_dict : dict {text_id: text_string}
            Dictionary of texts to encode.
        batch_size : int
            Batch size for encoding.
        
        Returns
        -------
        embeddings : (N, 1024) numpy array
            Ordered by sorted text_id keys.
        """
        # Sort by text_id to ensure consistent ordering
        sorted_ids = sorted(text_dict.keys(), key=int)
        sorted_texts = [text_dict[tid] for tid in sorted_ids]
        
        all_embeddings = []
        num_batches = (len(sorted_texts) + batch_size - 1) // batch_size
        
        logger.info(f"Encoding {len(sorted_texts)} texts in {num_batches} batches...")
        
        for i in tqdm(range(0, len(sorted_texts), batch_size), desc="Encoding"):
            batch_texts = sorted_texts[i:i + batch_size]
            batch_emb = self.encode_batch(batch_texts)
            all_embeddings.append(batch_emb)
        
        embeddings = np.vstack(all_embeddings)
        logger.info(f"✓ Encoded shape: {embeddings.shape}")
        
        return embeddings


def recompute_zca_whitening(
    text_embeddings: np.ndarray,
    cell_embeddings: np.ndarray,
    eps: float = 1e-5,
) -> Dict:
    """Recompute ZCA whitening transforms for both modalities.
    
    Parameters
    ----------
    text_embeddings : (N, D_text)
    cell_embeddings : (N, D_cell)
    eps : float
        Regularization for numerical stability.
    
    Returns
    -------
    transforms : dict with keys:
        "text_mean", "text_W_zca" (D_text, D_text)
        "cell_mean", "cell_W_zca" (D_cell, D_cell)
    """
    logger.info("Computing ZCA whitening transforms...")
    
    transforms = {}
    
    # Text whitening
    text_mean = text_embeddings.mean(axis=0, keepdims=True)
    text_centered = text_embeddings - text_mean
    text_cov = np.cov(text_centered, rowvar=False)
    
    U, S, Vt = np.linalg.svd(text_cov)
    text_W_zca = U @ np.diag(1 / np.sqrt(S + eps)) @ Vt
    
    transforms["text_mean"] = text_mean
    transforms["text_W_zca"] = text_W_zca
    
    # Compute whitened norms for reporting
    text_whitened = (text_centered @ text_W_zca.T)
    text_cov_after = np.cov(text_whitened, rowvar=False)
    text_cov_diag_mean = np.diag(text_cov_after).mean()
    text_cov_offdiag_mean = (text_cov_after.sum() - np.diag(text_cov_after).sum()) / (text_cov_after.size - len(text_cov_after))
    
    logger.info(f"  Text: cov_diag={text_cov_diag_mean:.6f}, cov_offdiag={text_cov_offdiag_mean:.6f}")
    
    # Cell whitening
    cell_mean = cell_embeddings.mean(axis=0, keepdims=True)
    cell_centered = cell_embeddings - cell_mean
    cell_cov = np.cov(cell_centered, rowvar=False)
    
    U, S, Vt = np.linalg.svd(cell_cov)
    cell_W_zca = U @ np.diag(1 / np.sqrt(S + eps)) @ Vt
    
    transforms["cell_mean"] = cell_mean
    transforms["cell_W_zca"] = cell_W_zca
    
    # Compute whitened norms for reporting
    cell_whitened = (cell_centered @ cell_W_zca.T)
    cell_cov_after = np.cov(cell_whitened, rowvar=False)
    cell_cov_diag_mean = np.diag(cell_cov_after).mean()
    cell_cov_offdiag_mean = (cell_cov_after.sum() - np.diag(cell_cov_after).sum()) / (cell_cov_after.size - len(cell_cov_after))
    
    logger.info(f"  Cell: cov_diag={cell_cov_diag_mean:.6f}, cov_offdiag={cell_cov_offdiag_mean:.6f}")
    
    return transforms


def main():
    parser = argparse.ArgumentParser(description="Re-embed polished texts with BiomedBERT")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/cached_latents_v5.2",
        help="Cache directory containing text_strings.json",
    )
    parser.add_argument(
        "--recompute_whitening",
        action="store_true",
        help="Recompute ZCA whitening transforms (recommended after polishing)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for BiomedBERT",
    )
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return
    
    # ========================================================================
    # Step 1: Load polished texts
    # ========================================================================
    text_strings_path = cache_dir / "text_strings.json"
    logger.info(f"Loading polished texts from {text_strings_path}...")
    
    with open(text_strings_path) as f:
        text_dict = json.load(f)
    
    logger.info(f"✓ Loaded {len(text_dict)} polished texts")
    
    # Quick quality check
    sample_ids = ["0", "1", "2"]
    for tid in sample_ids:
        if tid in text_dict:
            logger.info(f"  Sample [{tid}]: {text_dict[tid][:100]}...")
    
    # ========================================================================
    # Step 2: Encode all texts with BiomedBERT
    # ========================================================================
    encoder = BiomedBERTEncoder(device=args.device)
    text_embeddings = encoder.encode_all(text_dict, batch_size=args.batch_size)
    
    # Backup old embeddings before overwriting
    old_emb_path = cache_dir / "text_embeddings.npy"
    if old_emb_path.exists():
        backup_path = cache_dir / "text_embeddings_original_templates.npy"
        logger.info(f"Backing up old embeddings to {backup_path}...")
        import shutil
        shutil.copy(old_emb_path, backup_path)
    
    # Save new embeddings
    new_emb_path = cache_dir / "text_embeddings.npy"
    logger.info(f"Saving new embeddings to {new_emb_path}...")
    np.save(new_emb_path, text_embeddings)
    logger.info(f"✓ Saved {text_embeddings.shape}")
    
    # ========================================================================
    # Step 3: Recompute ZCA whitening (optional but highly recommended)
    # ========================================================================
    if args.recompute_whitening:
        # Load cell embeddings
        cell_emb_path = cache_dir / "cell_embeddings.npy"
        if not cell_emb_path.exists():
            logger.warning(f"Cell embeddings not found at {cell_emb_path}, skipping whitening")
        else:
            logger.info(f"Loading cell embeddings from {cell_emb_path}...")
            cell_embeddings = np.load(cell_emb_path)
            logger.info(f"✓ Loaded cell embeddings: {cell_embeddings.shape}")
            
            # Load text→cell mapping
            text_assignments_path = cache_dir / "text_assignments.npy"
            if not text_assignments_path.exists():
                logger.warning(f"Text assignments not found, using all embeddings")
                # Assume one-to-one mapping (may not be correct)
                text_emb_for_whitening = text_embeddings
                cell_emb_for_whitening = cell_embeddings
            else:
                text_assignments = np.load(text_assignments_path)
                logger.info(f"✓ Loaded text assignments: {text_assignments.shape}")
                
                # Expand text embeddings to match cell count
                text_emb_for_whitening = text_embeddings[text_assignments]
                cell_emb_for_whitening = cell_embeddings
            
            # Compute whitening transforms
            transforms = recompute_zca_whitening(
                text_emb_for_whitening,
                cell_emb_for_whitening,
            )
            
            # Backup old transforms
            for key in ["text_mean", "text_W_zca", "cell_mean", "cell_W_zca"]:
                old_path = cache_dir / f"{key}.npy"
                if old_path.exists():
                    backup_path = cache_dir / f"{key}_original_templates.npy"
                    logger.info(f"Backing up {key} to {backup_path.name}...")
                    import shutil
                    shutil.copy(old_path, backup_path)
            
            # Save new transforms
            for key, value in transforms.items():
                save_path = cache_dir / f"{key}.npy"
                np.save(save_path, value)
                logger.info(f"✓ Saved {key}: {value.shape}")
    
    # ========================================================================
    # Step 4: Summary
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("RE-EMBEDDING COMPLETE")
    logger.info("="*80)
    logger.info(f"✓ Encoded {len(text_dict)} polished texts")
    logger.info(f"✓ Saved to: {new_emb_path}")
    if args.recompute_whitening:
        logger.info(f"✓ Recomputed ZCA whitening transforms")
    logger.info("\nNext steps:")
    logger.info("1. Retrain CLOP with new text embeddings")
    logger.info("2. Expected improvement: 15-25% val_proto_acc (vs 10.45% before)")
    logger.info("3. Monitor cross-tissue generalization (same cell type in different tissues)")
    logger.info("="*80)


if __name__ == "__main__":
    main()
