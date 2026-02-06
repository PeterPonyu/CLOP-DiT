#!/usr/bin/env python3
# 01_fetch_data.py — Download and preprocess GEO datasets
"""
Step 1: Fetch GEO metadata and preprocess expression matrices.

Usage:
    python scripts/01_fetch_data.py --gse_ids GSE131907 GSE148071 --output_dir data/processed_h5ad
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline.geo_fetcher import GEOFetcher, LUNG_CANCER_PILOT
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Fetch GEO Data")
    parser.add_argument("--gse_ids", nargs="+", default=None,
                        help="GSE IDs to fetch. Default: LUNG_CANCER_PILOT")
    parser.add_argument("--output_dir", type=str, default="data/processed_h5ad")
    parser.add_argument("--raw_dir", type=str, default="data/raw_geo")
    parser.add_argument("--n_top_genes", type=int, default=2000)
    args = parser.parse_args()

    setup_logging()

    fetcher = GEOFetcher(
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        n_top_genes=args.n_top_genes,
    )

    gse_ids = args.gse_ids or LUNG_CANCER_PILOT
    print(f"Fetching {len(gse_ids)} datasets...")

    metadata = fetcher.fetch_batch(gse_ids, metadata_file="metadata_raw.json")
    print(f"\nDone! Fetched metadata for {len(metadata)} datasets.")
    print("Next: Run 02_clean_text.py to extract structured labels.")


if __name__ == "__main__":
    main()
