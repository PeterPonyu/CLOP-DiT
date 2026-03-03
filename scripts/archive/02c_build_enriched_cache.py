#!/usr/bin/env python3
# 02c_build_enriched_cache.py — Build deduplicated cache from enriched metadata
"""
CLOP-DiT v6.2: Convert existing text_embeddings.npy + enriched metadata into
the efficient deduplicated cache format.

This script:
1. Reads the existing cached text embeddings (text_embeddings.npy)
2. Deduplicates them into text_embeddings_unique.npy + text_group_ids.npy
3. Reads enriched metadata and maps text_group_ids to enriched text strings
4. Encodes caption variants through BiomedBERT (optional, requires GPU)
5. Updates the manifest

No re-encoding of the PRIMARY text embeddings is needed — this just
reorganizes the existing data into the efficient format.

Usage:
    # Step 1: Just deduplicate existing cache (no re-encoding)
    python scripts/02c_build_enriched_cache.py --mode dedup

    # Step 2: Re-encode with enriched texts (requires GPU + BiomedBERT)
    python scripts/02c_build_enriched_cache.py --mode re-encode

    # Step 3: Encode caption variants (requires GPU + BiomedBERT)
    python scripts/02c_build_enriched_cache.py --mode variants
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def deduplicate_existing_cache(cache_dir: str):
    """Convert existing duplicated text_embeddings.npy to deduplicated format.

    No re-encoding needed — just reorganizes the data.
    """
    cache_dir = Path(cache_dir)

    logger.info("Loading existing text embeddings...")
    text_emb = np.load(cache_dir / "text_embeddings.npy")
    logger.info(f"  Shape: {text_emb.shape}, Size: {text_emb.nbytes / 1e6:.1f} MB")

    # Find unique embeddings using byte-level comparison
    logger.info("Finding unique text embeddings...")
    text_bytes = text_emb.view(np.uint8).reshape(text_emb.shape[0], -1)
    _, unique_idx, inverse_idx = np.unique(
        text_bytes, axis=0, return_index=True, return_inverse=True
    )

    text_emb_unique = text_emb[unique_idx]
    text_group_ids = inverse_idx.astype(np.int32)

    logger.info(f"  Unique texts: {text_emb_unique.shape[0]}")
    logger.info(f"  Dedup ratio: {text_emb.shape[0] / text_emb_unique.shape[0]:.1f}x")

    # Save deduplicated
    np.save(cache_dir / "text_embeddings_unique.npy", text_emb_unique)
    np.save(cache_dir / "text_group_ids.npy", text_group_ids)

    orig_size = text_emb.nbytes
    new_size = text_emb_unique.nbytes + text_group_ids.nbytes
    logger.info(
        f"  Saved: {orig_size / 1e6:.1f} MB → {new_size / 1e6:.1f} MB "
        f"({100 * (1 - new_size / orig_size):.1f}% savings)"
    )

    # Also deduplicate preprocessed if it exists
    pp_path = cache_dir / "text_embeddings_preprocessed.npy"
    if pp_path.exists():
        logger.info("Deduplicating preprocessed text embeddings...")
        text_emb_pp = np.load(pp_path)
        text_emb_pp_unique = text_emb_pp[unique_idx]
        np.save(cache_dir / "text_embeddings_unique_preprocessed.npy", text_emb_pp_unique)
        logger.info(f"  Saved preprocessed unique: {text_emb_pp_unique.shape}")

    # Build text_strings.json from subcluster metadata
    # Map each unique text_group_id → the raw text string
    sub_meta_path = cache_dir.parent / "processed_h5ad" / "subcluster_metadata.json"
    enriched_meta_path = cache_dir.parent / "processed_h5ad" / "subcluster_metadata_enriched.json"
    metadata_path = cache_dir / "metadata.json"
    sample_ids = np.load(cache_dir / "sample_ids.npy")

    # Try enriched first, fall back to original
    meta_path = enriched_meta_path if enriched_meta_path.exists() else sub_meta_path
    if meta_path.exists():
        logger.info(f"Building text_strings.json from {meta_path.name}...")
        with open(meta_path) as f:
            sub_meta = json.load(f)
        with open(metadata_path) as f:
            id_to_text = json.load(f)

        # Reconstruct the per-cell text assignment used during cache building
        # We need to figure out which text string corresponds to each unique embedding
        # Use the sample_ids + cluster cell_indices to reconstruct
        all_cell_texts = [""] * text_emb.shape[0]
        all_cell_variants = [[] for _ in range(text_emb.shape[0])]

        # Build dataset_id → sample_id mapping
        # (Iterate over metadata to reconstruct the order)
        for sid_str, ds_text in id_to_text.items():
            sid = int(sid_str)
            mask = sample_ids == sid
            cell_indices_global = np.where(mask)[0]

            # Find the matching dataset in subcluster meta
            ds_match = None
            for ds_id, ds_info in sub_meta.items():
                if ds_info["dataset_text"][:80] == ds_text[:80]:
                    ds_match = (ds_id, ds_info)
                    break

            if ds_match is None:
                # Fallback: use dataset-level text
                for gi in cell_indices_global:
                    all_cell_texts[gi] = ds_text
                continue

            ds_id, ds_info = ds_match
            # Default: dataset-level text
            for gi in cell_indices_global:
                all_cell_texts[gi] = ds_text

            # Override with cluster-level texts
            n_ds_cells = int(mask.sum())
            start_idx = int(cell_indices_global[0])

            for cid, cinfo in ds_info.get("clusters", {}).items():
                cluster_text = cinfo.get("text", ds_text)
                text_variants = cinfo.get("text_variants", [])
                for local_idx in cinfo.get("cell_indices", []):
                    global_idx = start_idx + local_idx
                    if global_idx < text_emb.shape[0] and sample_ids[global_idx] == sid:
                        all_cell_texts[global_idx] = cluster_text
                        if text_variants:
                            all_cell_variants[global_idx] = text_variants

        # Map unique group IDs → text strings
        text_strings = {}
        text_variants_map = {}
        for uid in range(len(unique_idx)):
            cell_idx = unique_idx[uid]
            text_strings[str(uid)] = all_cell_texts[cell_idx]
            if all_cell_variants[cell_idx]:
                text_variants_map[str(uid)] = all_cell_variants[cell_idx]

        with open(cache_dir / "text_strings.json", "w") as f:
            json.dump(text_strings, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved text_strings.json: {len(text_strings)} entries")

        if text_variants_map:
            with open(cache_dir / "text_variants.json", "w") as f:
                json.dump(text_variants_map, f, indent=2, ensure_ascii=False)
            logger.info(f"  Saved text_variants.json: {len(text_variants_map)} entries with variants")

    # Update manifest
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}

    manifest["deduplicated"] = True
    manifest["format_version"] = "6.2"
    manifest["num_unique_texts"] = int(text_emb_unique.shape[0])

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Deduplication complete!")
    return text_group_ids, text_emb_unique


def reencode_enriched_texts(cache_dir: str, enriched_meta_file: str, device: str = "cuda"):
    """Re-encode primary texts from enriched metadata through BiomedBERT.

    This replaces the text embeddings with embeddings of the enriched
    (evidence-dense) descriptions instead of the original templates.
    """
    cache_dir = Path(cache_dir)

    from src.data_pipeline.cache_builder import LatentCacheBuilder

    logger.info("Loading enriched metadata...")
    with open(enriched_meta_file) as f:
        enriched = json.load(f)

    # Load existing dedup data
    text_group_ids = np.load(cache_dir / "text_group_ids.npy")
    with open(cache_dir / "text_strings.json") as f:
        old_strings = json.load(f)

    n_unique = len(old_strings)
    logger.info(f"  {n_unique} unique text groups to re-encode")

    # Collect all enriched texts in group order
    # Each group ID maps to its enriched primary text
    enriched_texts = []
    for uid in range(n_unique):
        enriched_texts.append(old_strings[str(uid)])

    # Now try to map each to its enriched version
    # Build lookup: original_text → enriched_text
    text_upgrade_map = {}
    for ds_id, ds_info in enriched.items():
        for cid, cinfo in ds_info["clusters"].items():
            if "text_original" in cinfo and "text" in cinfo:
                text_upgrade_map[cinfo["text_original"]] = cinfo["text"]

    n_upgraded = 0
    new_text_strings = {}
    for uid in range(n_unique):
        old_text = old_strings[str(uid)]
        if old_text in text_upgrade_map:
            enriched_texts[uid] = text_upgrade_map[old_text]
            n_upgraded += 1
        new_text_strings[str(uid)] = enriched_texts[uid]

    logger.info(f"  Upgraded {n_upgraded}/{n_unique} texts to enriched versions")

    # Encode through BiomedBERT
    builder = LatentCacheBuilder(cache_dir=str(cache_dir), device=device)
    logger.info("Encoding enriched texts through BiomedBERT...")
    new_emb = builder.encode_texts(enriched_texts)

    # Save
    np.save(cache_dir / "text_embeddings_unique.npy", new_emb)
    with open(cache_dir / "text_strings.json", "w") as f:
        json.dump(new_text_strings, f, indent=2, ensure_ascii=False)

    # Also rebuild the full duplicated version for legacy compatibility
    text_emb_full = new_emb[text_group_ids]
    np.save(cache_dir / "text_embeddings.npy", text_emb_full)

    logger.info(f"  Saved re-encoded text_embeddings_unique.npy: {new_emb.shape}")
    logger.info("  Re-encoding complete!")


def encode_variants(cache_dir: str, device: str = "cuda"):
    """Encode all caption variants through BiomedBERT."""
    cache_dir = Path(cache_dir)

    from src.data_pipeline.cache_builder import LatentCacheBuilder

    variants_path = cache_dir / "text_variants.json"
    if not variants_path.exists():
        logger.error("text_variants.json not found. Run dedup first.")
        return

    with open(variants_path) as f:
        text_variants = json.load(f)

    # Flatten all variants
    all_variant_texts = []
    variant_group_map = []  # [(group_id, variant_idx), ...]

    for gid_str, variants in text_variants.items():
        gid = int(gid_str)
        for vi, vtext in enumerate(variants):
            all_variant_texts.append(vtext)
            variant_group_map.append([gid, vi])

    logger.info(f"Encoding {len(all_variant_texts)} caption variants...")

    builder = LatentCacheBuilder(cache_dir=str(cache_dir), device=device)
    variant_embs = builder.encode_texts(all_variant_texts)

    np.save(cache_dir / "text_variant_embeddings.npy", variant_embs)
    with open(cache_dir / "text_variant_map.json", "w") as f:
        json.dump(variant_group_map, f)

    logger.info(f"  Saved {variant_embs.shape} variant embeddings + map")


def main():
    parser = argparse.ArgumentParser(description="Build deduplicated enriched cache")
    parser.add_argument("--cache_dir", default="data/cached_latents_v5.2",
                       help="Cache directory")
    parser.add_argument("--enriched_meta",
                       default="data/processed_h5ad/subcluster_metadata_enriched.json",
                       help="Enriched metadata file")
    parser.add_argument("--mode", choices=["dedup", "re-encode", "variants", "all"],
                       default="dedup",
                       help="Operation mode: dedup (no GPU), re-encode (GPU), variants (GPU), all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_level", default="INFO")

    args = parser.parse_args()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=level)

    if args.mode in ("dedup", "all"):
        deduplicate_existing_cache(args.cache_dir)

    if args.mode in ("re-encode", "all"):
        reencode_enriched_texts(args.cache_dir, args.enriched_meta, args.device)

    if args.mode in ("variants", "all"):
        encode_variants(args.cache_dir, args.device)


if __name__ == "__main__":
    main()
