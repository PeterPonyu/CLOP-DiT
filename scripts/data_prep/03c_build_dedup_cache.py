#!/usr/bin/env python3
# 03c_build_dedup_cache.py — Build deduplicated cache (Level 2b) for CLOP-DiT.
"""
Build dedup arrays and captions so downstream (generate_embeddings, diversity_diagnostics,
decode_expression, panels) can use the 69-type space.

Prerequisites in cache_dir (from 03_cache_latents):
  - cell_embeddings.npy, text_embeddings_unique.npy, text_group_ids.npy, sample_ids.npy

Required (create with deduplicate_captions.py or scripts/fix_dedup_captions.py):
  - text_group_mapping.json   (old group_id -> new 0..68 or None for uncharacterized)
  - text_captions_deduplicated.json  (new_id -> caption string)

Outputs (into cache_dir):
  - cell_embeddings_dedup.npy, text_embeddings_dedup.npy
  - text_group_ids_dedup.npy, sample_ids_dedup.npy
  - text_captions_deduplicated.json (written if not already present)
  - METADATA_DEDUP.json

Then run 03b_preprocess_embeddings (which will also produce *_dedup_preprocessed.npy when dedup files exist).

Usage:
  python scripts/03c_build_dedup_cache.py
  python scripts/03c_build_dedup_cache.py --cache-dir data/cached_latents
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.paths import CACHE_DIR
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def filter_and_remap(
    cell_emb: np.ndarray,
    text_group_ids: np.ndarray,
    sample_ids: np.ndarray,
    text_mapping: dict,
) -> tuple:
    """Filter out uncharacterized cells; remap group IDs to 0..68."""
    num_cells = cell_emb.shape[0]
    kept_indices = []
    new_group_ids = []
    kept_by_group = defaultdict(int)

    for i in range(num_cells):
        old_group_id = int(text_group_ids[i])
        old_str = str(old_group_id)
        if old_str not in text_mapping:
            continue
        new_id = text_mapping[old_str]
        if new_id is None:
            continue
        kept_indices.append(i)
        new_group_ids.append(new_id)
        kept_by_group[new_id] += 1

    kept_indices = np.array(kept_indices, dtype=np.int32)
    new_group_ids = np.array(new_group_ids, dtype=np.int32)
    filtered_cell = cell_emb[kept_indices]
    filtered_sample_ids = sample_ids[kept_indices]
    stats = {
        "original_cells": int(num_cells),
        "filtered_cells": len(kept_indices),
        "cells_removed": num_cells - len(kept_indices),
        "kept_by_group": dict(kept_by_group),
    }
    return filtered_cell, new_group_ids, filtered_sample_ids, stats


def get_text_embeddings_for_groups(
    text_emb_all: np.ndarray,
    text_mapping: dict,
) -> np.ndarray:
    """Extract one text embedding per deduplicated group (69)."""
    reverse = defaultdict(list)
    for old_str, new_id in text_mapping.items():
        if new_id is not None:
            reverse[new_id].append(int(old_str))
    dedup_list = []
    for new_id in sorted(reverse.keys()):
        old_id = reverse[new_id][0]
        dedup_list.append(text_emb_all[old_id])
    return np.array(dedup_list, dtype=np.float32)


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Build deduplicated cache (Level 2b)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help=f"Cache directory (default: {CACHE_DIR})")
    args = parser.parse_args()
    cache = Path(args.cache_dir) if args.cache_dir else CACHE_DIR
    cache = cache.resolve()

    mapping_path = cache / "text_group_mapping.json"
    captions_path = cache / "text_captions_deduplicated.json"
    if not mapping_path.is_file():
        logger.error(
            "text_group_mapping.json not found in %s. Create it first with "
            "deduplicate_captions.py or scripts/fix_dedup_captions.py.",
            cache,
        )
        return 1
    if not captions_path.is_file():
        logger.error(
            "text_captions_deduplicated.json not found in %s. Create it first.",
            cache,
        )
        return 1

    cell_path = cache / "cell_embeddings.npy"
    text_unique_path = cache / "text_embeddings_unique.npy"
    group_path = cache / "text_group_ids.npy"
    sample_path = cache / "sample_ids.npy"
    for p, name in [
        (cell_path, "cell_embeddings.npy"),
        (text_unique_path, "text_embeddings_unique.npy"),
        (group_path, "text_group_ids.npy"),
        (sample_path, "sample_ids.npy"),
    ]:
        if not p.is_file():
            logger.error("%s not found. Run 03_cache_latents.py first.", name)
            return 1

    logger.info("Loading cache...")
    cell_emb = np.load(cell_path, mmap_mode="r")
    text_emb_all = np.load(text_unique_path, mmap_mode="r")
    text_group_ids = np.load(group_path)
    sample_ids = np.load(sample_path)
    with open(mapping_path) as f:
        text_mapping = json.load(f)
    with open(captions_path) as f:
        dedup_captions = json.load(f)
    # Normalize caption keys to int for consistency
    dedup_captions = {int(k): v for k, v in dedup_captions.items()}

    logger.info("Filtering and remapping cells...")
    filtered_cell, new_group_ids, filtered_sample_ids, stats = filter_and_remap(
        cell_emb, text_group_ids, sample_ids, text_mapping
    )
    logger.info("Building dedup text embeddings...")
    text_dedup = get_text_embeddings_for_groups(text_emb_all, text_mapping)

    logger.info("Saving dedup cache...")
    np.save(cache / "cell_embeddings_dedup.npy", filtered_cell)
    np.save(cache / "text_embeddings_dedup.npy", text_dedup)
    np.save(cache / "text_group_ids_dedup.npy", new_group_ids)
    np.save(cache / "sample_ids_dedup.npy", filtered_sample_ids)
    with open(cache / "text_captions_deduplicated.json", "w") as f:
        json.dump(dedup_captions, f, indent=2)

    metadata = {
        "processing_stage": "deduplication_filtering",
        "original_cells": stats["original_cells"],
        "filtered_cells": stats["filtered_cells"],
        "cells_removed": stats["cells_removed"],
        "deduplicated_captions": len(dedup_captions),
        "statistics": {"kept_by_group": stats["kept_by_group"]},
    }
    with open(cache / "METADATA_DEDUP.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Dedup cache complete: %d cells, %d types. Next: python scripts/03b_preprocess_embeddings.py --cache_dir %s",
        stats["filtered_cells"],
        len(dedup_captions),
        cache,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
