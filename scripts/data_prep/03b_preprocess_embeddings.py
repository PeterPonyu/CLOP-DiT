#!/usr/bin/env python3
# 03b_preprocess_embeddings.py — Fix embedding space collapse before CLOP training
"""
CRITICAL preprocessing step for CLOP-DiT v6.

Applies ZCA whitening to both text (BiomedBERT) and cell (scGPT) embeddings
to fix the fundamental space collapse problem that caused all v3.0–v5.2 CLOP
alignment failures.

Before preprocessing:
    Text embeddings pairwise cosine:  mean = 0.954 (collapsed → indistinguishable)
    Cell embeddings pairwise cosine:  mean = 0.991 (collapsed → indistinguishable)

After preprocessing:
    Text embeddings pairwise cosine:  mean ≈ 0.05  (well-spread → discriminable)
    Cell embeddings pairwise cosine:  mean ≈ 0.10  (well-spread → discriminable)

Usage:
    python scripts/03b_preprocess_embeddings.py
    python scripts/03b_preprocess_embeddings.py --cache_dir data/cached_latents
    python scripts/03b_preprocess_embeddings.py --text_method whiten --cell_method center_norm
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logging_config import setup_logging
from src.data_pipeline.embedding_preprocessor import preprocess_cached_embeddings

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess cached embeddings (whitening)")
    parser.add_argument("--cache_dir", type=str, default="data/cached_latents",
                        help="Directory with cached embeddings")
    parser.add_argument("--text_method", type=str, default="whiten",
                        choices=["whiten", "whiten_pca", "center_norm", "none"],
                        help="Text embedding preprocessing method")
    parser.add_argument("--cell_method", type=str, default="whiten",
                        choices=["whiten", "whiten_pca", "center_norm", "none"],
                        help="Cell embedding preprocessing method")
    args = parser.parse_args()

    setup_logging("preprocess_embeddings")

    stats = preprocess_cached_embeddings(
        cache_dir=args.cache_dir,
        text_method=args.text_method,
        cell_method=args.cell_method,
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Text cosine: {stats['text']['raw_cosine_mean']:.4f} → {stats['text']['processed_cosine_mean']:.4f}")
    print(f"Cell cosine: {stats['cell']['raw_cosine_mean']:.4f} → {stats['cell']['processed_cosine_mean']:.4f}")
    print(f"\nOutput files:")
    print(f"  {args.cache_dir}/text_embeddings_preprocessed.npy")
    print(f"  {args.cache_dir}/cell_embeddings_preprocessed.npy")
    print(f"\nNow run: python scripts/04a_train_clop.py --config configs/clop.yaml")


if __name__ == "__main__":
    main()
