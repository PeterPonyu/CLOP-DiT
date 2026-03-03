#!/usr/bin/env python3
# 03_cache_latents.py — Pre-compute cell and text embeddings
"""
Step 3: Cache cell and text embeddings as .npy files.

This eliminates the need to load heavy encoder models during CLOP/DiT training.

Usage:
    python scripts/03_cache_latents.py \
        --h5ad_dir data/processed_h5ad \
        --metadata data/processed_h5ad/metadata_structured.json \
        --output_dir data/cached_latents_v5.2 \
        --cell_encoder pca
"""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline.cache_builder import LatentCacheBuilder
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Cache Latent Embeddings")
    parser.add_argument("--h5ad_dir", type=str, default="data/processed_h5ad")
    parser.add_argument("--metadata", type=str, default="data/processed_h5ad/metadata_structured.json")
    parser.add_argument("--output_dir", type=str, default="data/cached_latents_v5.2")
    parser.add_argument("--cell_encoder", type=str, default="scgpt",
                        choices=["scgpt", "pca"], help="Cell encoder method (scGPT recommended)")
    parser.add_argument("--scgpt_dir", type=str, default="models/scgpt_pancancer",
                        help="scGPT weights directory (if using scgpt)")
    parser.add_argument("--text_encoder", type=str,
                        default="microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
    parser.add_argument("--cell_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--subcluster_metadata", type=str, default=None,
                        help="Sub-cluster metadata JSON from 02_subcluster_descriptions.py. "
                             "Enables per-cluster text assignments for fine-grained alignment.")
    args = parser.parse_args()

    setup_logging()

    # Find all processed h5ad files
    h5ad_files = sorted(glob.glob(str(Path(args.h5ad_dir) / "*_processed.h5ad")))
    if not h5ad_files:
        h5ad_files = sorted(glob.glob(str(Path(args.h5ad_dir) / "*.h5ad")))

    if not h5ad_files:
        print(f"No h5ad files found in {args.h5ad_dir}")
        print("Run 01_fetch_data.py first, or place processed h5ad files there.")
        sys.exit(1)

    print(f"Found {len(h5ad_files)} h5ad files:")
    for f in h5ad_files:
        print(f"  {f}")

    builder = LatentCacheBuilder(
        cache_dir=args.output_dir,
        cell_encoder=args.cell_encoder,
        text_encoder=args.text_encoder,
        cell_dim=args.cell_dim,
        batch_size=args.batch_size,
    )

    manifest = builder.build_cache(
        h5ad_files=h5ad_files,
        metadata_file=args.metadata,
        cell_encoder_method=args.cell_encoder,
        scgpt_model_dir=args.scgpt_dir if args.cell_encoder == "scgpt" else None,
        subcluster_metadata_file=args.subcluster_metadata,
    )

    print(f"\n{'='*50}")
    print(f"Cache built successfully!")
    print(f"  Total cells:  {manifest['total_cells']:,}")
    print(f"  Cell dim:     {manifest['cell_dim']}")
    print(f"  Text dim:     {manifest['text_dim']}")
    print(f"  Datasets:     {manifest['num_datasets']}")
    print(f"  Output:       {args.output_dir}")
    print(f"{'='*50}")
    print("Next: Run 04_train_clop.py for contrastive alignment.")


if __name__ == "__main__":
    main()
