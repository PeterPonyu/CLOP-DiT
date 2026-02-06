#!/usr/bin/env python3
# 02_clean_text.py — SFT text cleaning pipeline
"""
Step 2: Extract structured metadata from raw GEO text using Llama-3.

Usage:
    python scripts/02_clean_text.py \
        --input data/processed_h5ad/metadata_raw.json \
        --output data/processed_h5ad/metadata_structured.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline.text_cleaner import TextCleaner
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="SFT Text Cleaning")
    parser.add_argument("--input", type=str, default="data/processed_h5ad/metadata_raw.json")
    parser.add_argument("--output", type=str, default="data/processed_h5ad/metadata_structured.json")
    parser.add_argument("--model", type=str, default="unsloth/llama-3-8b-Instruct-bnb-4bit")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")
    args = parser.parse_args()

    setup_logging()

    cleaner = TextCleaner(
        model_name=args.model,
        adapter_path=args.adapter,
    )

    print(f"Processing: {args.input}")
    structured = cleaner.extract_batch(args.input, args.output)
    print(f"\nDone! Structured metadata saved to {args.output}")
    print(f"Processed {len(structured)} datasets.")
    print("Next: Run 03_cache_latents.py to pre-compute embeddings.")


if __name__ == "__main__":
    main()
