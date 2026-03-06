#!/usr/bin/env python3
# 02_subcluster_descriptions.py — Automated sub-cluster text description generation
"""
CLOP-DiT v0.4: Expand text-cell alignment data by automatically clustering
each dataset and generating per-cluster biological text descriptions.

Usage:
    python scripts/02_subcluster_descriptions.py \\
        --h5ad_dir data/processed_h5ad \\
        --metadata data/processed_h5ad/metadata_structured.json \\
        --output_dir data/processed_h5ad \\
        --resolution 0.8
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logging_config import setup_logging
from src.data_pipeline.subcluster_annotation import run_subcluster_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="CLOP-DiT v0.4: Sub-cluster text description generation"
    )
    parser.add_argument("--h5ad_dir", type=str, default="data/processed_h5ad")
    parser.add_argument("--metadata", type=str, default="data/processed_h5ad/metadata_structured.json")
    parser.add_argument("--output_dir", type=str, default="data/processed_h5ad")
    parser.add_argument("--resolution", type=float, default=0.8,
                        help="Base Leiden clustering resolution (higher = more clusters)")
    parser.add_argument("--resolution_mode", choices=["fixed", "adaptive"], default="adaptive",
                        help="fixed: use --resolution; adaptive: adjust by dataset size")
    parser.add_argument("--min_cluster_size", type=int, default=20,
                        help="Minimum cells per cluster to annotate")
    parser.add_argument("--signature_db", type=str, default="",
                        help="Optional JSON file with additional marker signatures")
    args = parser.parse_args()

    setup_logging()

    run_subcluster_pipeline(
        h5ad_dir=args.h5ad_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        resolution=args.resolution,
        resolution_mode=args.resolution_mode,
        min_cluster_size=args.min_cluster_size,
        signature_db_path=args.signature_db or "",
    )

    print("\nNext: Run 03_cache_latents.py with --subcluster_metadata <output_dir>/subcluster_metadata.json")


if __name__ == "__main__":
    main()
