#!/usr/bin/env python3
"""
Fix deduplicated captions and text embeddings.

Issues addressed:
1. Stutter bug: captions had duplicated cell type name prefix
2. Text embeddings used only first original; now AVERAGES all originals per group
3. Regenerates clean caption text from the original polished captions

Usage:
    python scripts/fix_dedup_captions.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path("data/cached_latents")


def main():
    print("=" * 60)
    print("Fixing deduplicated captions and text embeddings")
    print("=" * 60)

    # Load original polished captions (these are well-formatted)
    with open(CACHE_DIR / "text_strings_polished_pre_v3.json") as f:
        original_captions = json.load(f)
    print(f"Loaded {len(original_captions)} original captions")

    # Load mapping: original_id -> dedup_id (or None for uncharacterized)
    with open(CACHE_DIR / "text_group_mapping.json") as f:
        mapping = json.load(f)
    print(f"Loaded mapping for {len(mapping)} entries")

    # Load original text embeddings
    text_emb_orig = np.load(CACHE_DIR / "text_embeddings_unique.npy")
    print(f"Loaded original text embeddings: {text_emb_orig.shape}")

    # Build reverse mapping: dedup_id -> list of original_ids
    reverse = defaultdict(list)
    for old_id_str, new_id in mapping.items():
        if new_id is not None:
            reverse[new_id].append(int(old_id_str))

    n_groups = len(reverse)
    print(f"Found {n_groups} deduplicated groups")

    # --- Fix 1: Clean captions from originals ---
    # For each dedup group, pick the BEST original caption as representative
    # "Best" = longest original caption that is NOT a dataset-level abstract
    fixed_captions = {}
    for new_id in sorted(reverse.keys()):
        old_ids = reverse[new_id]
        # Get all original captions for this group
        candidates = []
        for oid in old_ids:
            cap = original_captions.get(str(oid), "")
            # Skip dataset-level abstracts (they start with "Single-cell RNA sequencing")
            if cap.startswith("Single-cell RNA sequencing"):
                continue
            candidates.append(cap)

        if not candidates:
            # All were dataset-level; just use the first one
            candidates = [original_captions[str(old_ids[0])]]

        # Pick the longest non-abstract caption as representative
        best = max(candidates, key=len)
        fixed_captions[str(new_id)] = best

    # Verify: check for stutter bug
    stutter_count = 0
    for cid, cap in fixed_captions.items():
        # Check if first 30 chars have a repeated pattern
        words = cap.split()
        if len(words) >= 4:
            # The stutter pattern was: "lowercase are Titlecase are ..."
            if words[0].lower() == words[0] and "are" in words[:3]:
                half = cap[: len(cap) // 3]
                if half.lower() in cap[len(half) :].lower():
                    stutter_count += 1
    print(f"Stutter check: {stutter_count} captions still have stutter (should be 0)")

    # --- Fix 2: Average text embeddings per group ---
    averaged_embs = np.zeros((n_groups, text_emb_orig.shape[1]), dtype=np.float32)
    for new_id in sorted(reverse.keys()):
        old_ids = reverse[new_id]
        group_embs = text_emb_orig[old_ids]  # (K, 1024)
        averaged_embs[new_id] = group_embs.mean(axis=0)

    # L2 normalize the averaged embeddings (important for cosine similarity)
    norms = np.linalg.norm(averaged_embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    averaged_embs = averaged_embs / norms

    print(f"Created averaged text embeddings: {averaged_embs.shape}")
    print(f"Embedding norms after normalization: min={np.linalg.norm(averaged_embs, axis=1).min():.4f}, max={np.linalg.norm(averaged_embs, axis=1).max():.4f}")

    # --- Save fixed files ---
    # Overwrite the broken deduplicated captions
    with open(CACHE_DIR / "text_captions_deduplicated.json", "w") as f:
        json.dump(fixed_captions, f, indent=2, ensure_ascii=False)
    print(f"Saved fixed captions: {len(fixed_captions)} entries")

    # Overwrite the text embeddings with averaged version
    np.save(CACHE_DIR / "text_embeddings_dedup.npy", averaged_embs)
    print(f"Saved averaged text embeddings: {averaged_embs.shape}")

    # --- Verification ---
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    # Show first 5 fixed captions (truncated)
    for i in range(min(5, n_groups)):
        cap = fixed_captions[str(i)]
        print(f"  [{i}] {cap[:120]}...")

    # Show embedding statistics
    print(f"\nText embedding stats:")
    print(f"  Shape: {averaged_embs.shape}")
    print(f"  Mean: {averaged_embs.mean():.6f}")
    print(f"  Std:  {averaged_embs.std():.6f}")

    # Cross-check array alignment
    cell_dedup = np.load(CACHE_DIR / "cell_embeddings_dedup.npy", mmap_mode="r")
    gids_dedup = np.load(CACHE_DIR / "text_group_ids_dedup.npy")
    print(f"\nAlignment check:")
    print(f"  cell_embeddings_dedup: {cell_dedup.shape}")
    print(f"  text_embeddings_dedup: {averaged_embs.shape}")
    print(f"  text_group_ids_dedup:  {gids_dedup.shape}, range [{gids_dedup.min()}, {gids_dedup.max()}]")
    assert gids_dedup.max() < averaged_embs.shape[0], "Group ID exceeds text embedding count!"
    assert cell_dedup.shape[0] == gids_dedup.shape[0], "Cell and group ID count mismatch!"
    print("  All alignment checks PASSED")

    # Show group size distribution
    unique, counts = np.unique(gids_dedup, return_counts=True)
    print(f"\nGroup distribution:")
    print(f"  Groups: {len(unique)}")
    print(f"  Min cells/group:    {counts.min()}")
    print(f"  Max cells/group:    {counts.max()}")
    print(f"  Median cells/group: {int(np.median(counts))}")
    print(f"  Mean cells/group:   {counts.mean():.1f}")

    print("\nDONE - All fixes applied successfully")


if __name__ == "__main__":
    main()
