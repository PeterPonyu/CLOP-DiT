#!/usr/bin/env python3
"""
Prepare deduplicated captions for CLOP training.

This script filters the cached embeddings by removing uncharacterized cells
and creates deduplicated training data with remapped group IDs.

Input:
  - cell_embeddings.npy (220304 × 512): Cell embeddings
  - text_embeddings_unique.npy (1088 × 1024): Text embeddings for all unique captions
  - text_group_ids.npy (220304,): Original text group IDs for each cell
  - sample_ids.npy (220304,): Sample IDs for each cell
  - text_group_mapping.json: Maps old group IDs to new group IDs (None = uncharacterized)

Output:
  - cell_embeddings_dedup.npy: Filtered cell embeddings (167245 × 512)
  - text_embeddings_dedup.npy: Deduplicated text embeddings (69 × 1024)
  - text_group_ids_dedup.npy: Remapped group IDs (167245,) in range [0, 68]
  - sample_ids_dedup.npy: Filtered sample IDs (167245,)
  - METADATA_DEDUP.json: Comprehensive metadata about the deduplication
  - compatibility_check.json: Statistics for validation
"""

import json
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_input_data(cache_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load all input data files."""
    print("[*] Loading input data...")

    cell_emb = np.load(f"{cache_dir}/cell_embeddings.npy", mmap_mode="r")
    text_emb = np.load(f"{cache_dir}/text_embeddings_unique.npy", mmap_mode="r")
    text_group_ids = np.load(f"{cache_dir}/text_group_ids.npy", mmap_mode="r")
    sample_ids = np.load(f"{cache_dir}/sample_ids.npy", mmap_mode="r")

    with open(f"{cache_dir}/text_group_mapping.json") as f:
        text_mapping = json.load(f)

    print(f"  ✓ cell_embeddings.npy: {cell_emb.shape}")
    print(f"  ✓ text_embeddings_unique.npy: {text_emb.shape}")
    print(f"  ✓ text_group_ids.npy: {text_group_ids.shape}")
    print(f"  ✓ sample_ids.npy: {sample_ids.shape}")
    print(f"  ✓ text_group_mapping.json: {len(text_mapping)} mappings")

    return cell_emb, text_emb, text_group_ids, sample_ids, text_mapping


def load_deduplicated_captions(cache_dir: str) -> Dict[int, str]:
    """Load the deduplicated captions."""
    print("[*] Loading deduplicated captions...")

    with open(f"{cache_dir}/text_captions_deduplicated.json") as f:
        captions = json.load(f)

    # Convert string keys to ints for consistency
    captions = {int(k): v for k, v in captions.items()}

    print(f"  ✓ Loaded {len(captions)} deduplicated captions")
    return captions


def filter_and_remap(
    cell_emb: np.ndarray,
    text_group_ids: np.ndarray,
    sample_ids: np.ndarray,
    text_mapping: Dict[str, int],
    cache_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Filter out uncharacterized cells and remap group IDs.

    Returns:
      - filtered_cell_emb: Filtered cell embeddings
      - filtered_text_group_ids: Remapped group IDs (0-68)
      - filtered_sample_ids: Filtered sample IDs
      - stats: Dictionary with filtering statistics
    """
    print("[*] Filtering cells and remapping group IDs...")

    num_cells = cell_emb.shape[0]
    kept_indices = []
    new_group_ids = []

    # Stats tracking
    removed_by_group = defaultdict(int)
    kept_by_group = defaultdict(int)

    # Process each cell
    for i in range(num_cells):
        old_group_id = int(text_group_ids[i])

        # Check if this group ID is in the mapping
        old_group_str = str(old_group_id)
        if old_group_str not in text_mapping:
            print(f"  WARNING: Group ID {old_group_id} not found in text_mapping!")
            removed_by_group[old_group_id] += 1
            continue

        new_group_id = text_mapping[old_group_str]

        # Skip if uncharacterized (None mapping)
        if new_group_id is None:
            removed_by_group[old_group_id] += 1
            continue

        # Keep this cell
        kept_indices.append(i)
        new_group_ids.append(new_group_id)
        kept_by_group[new_group_id] += 1

        if (i + 1) % 50000 == 0:
            print(f"  [{i+1:7d}/{num_cells}] Processed {len(kept_indices)} kept cells")

    kept_indices = np.array(kept_indices, dtype=np.int32)
    new_group_ids = np.array(new_group_ids, dtype=np.int32)

    print(f"\n[*] Filtering complete:")
    print(f"  ✓ Original cells: {num_cells:,}")
    print(f"  ✓ Kept cells: {len(kept_indices):,}")
    print(f"  ✓ Removed cells: {num_cells - len(kept_indices):,}")
    print(f"  ✓ Removal rate: {100 * (num_cells - len(kept_indices)) / num_cells:.2f}%")

    # Load actual cell embeddings into memory for filtering
    print("\n[*] Filtering embeddings...")
    filtered_cell_emb = cell_emb[kept_indices]
    filtered_sample_ids = sample_ids[kept_indices]

    stats = {
        "original_cells": int(num_cells),
        "filtered_cells": int(len(kept_indices)),
        "cells_removed": int(num_cells - len(kept_indices)),
        "removed_by_group": dict(removed_by_group),
        "kept_by_group": dict(kept_by_group),
    }

    return filtered_cell_emb, new_group_ids, filtered_sample_ids, stats


def get_text_embeddings_for_groups(
    text_embeddings_all: np.ndarray,
    text_mapping: Dict[str, int],
) -> np.ndarray:
    """
    Extract text embeddings only for the 69 deduplicated groups.

    Maps original group indices (0-1087) through text_mapping to get
    the specific embeddings needed for the 69 deduplicated groups.
    """
    print("[*] Extracting deduplicated text embeddings...")

    # Build reverse mapping: new_group_id -> old_group_ids
    reverse_mapping = defaultdict(list)
    for old_id_str, new_id in text_mapping.items():
        if new_id is not None:
            old_id = int(old_id_str)
            reverse_mapping[new_id].append(old_id)

    num_groups = len(reverse_mapping)
    print(f"  Found {num_groups} deduplicated groups")

    # For each deduplicated group, we'll use the first original group's embedding
    # (since they're supposed to be semantically similar after deduplication)
    dedup_embeddings = []
    for new_id in sorted(reverse_mapping.keys()):
        old_ids = reverse_mapping[new_id]
        # Use the first one (could also average, but the captions are already deduplicated)
        old_id = old_ids[0]
        dedup_embeddings.append(text_embeddings_all[old_id])

        if (new_id + 1) % 10 == 0:
            print(f"  [{new_id + 1:3d}/69] Extracted embedding for group {new_id}")

    dedup_embeddings = np.array(dedup_embeddings, dtype=np.float32)
    print(f"  ✓ Text embeddings shape: {dedup_embeddings.shape}")

    return dedup_embeddings


def save_outputs(
    output_dir: str,
    cell_emb: np.ndarray,
    text_emb: np.ndarray,
    text_group_ids: np.ndarray,
    sample_ids: np.ndarray,
    deduplicated_captions: Dict[int, str],
    caption_metadata: Dict,
    stats: Dict,
) -> None:
    """Save all output files."""
    print(f"\n[*] Saving outputs to {output_dir}...")

    # Save numpy files
    output_path = Path(output_dir)

    np.save(str(output_path / "cell_embeddings_dedup.npy"), cell_emb)
    print(f"  ✓ Saved cell_embeddings_dedup.npy {cell_emb.shape}")

    np.save(str(output_path / "text_embeddings_dedup.npy"), text_emb)
    print(f"  ✓ Saved text_embeddings_dedup.npy {text_emb.shape}")

    np.save(str(output_path / "text_group_ids_dedup.npy"), text_group_ids)
    print(f"  ✓ Saved text_group_ids_dedup.npy {text_group_ids.shape}")

    np.save(str(output_path / "sample_ids_dedup.npy"), sample_ids)
    print(f"  ✓ Saved sample_ids_dedup.npy {sample_ids.shape}")

    # Save captions (already in proper format)
    with open(output_path / "text_captions_deduplicated.json", "w") as f:
        json.dump(deduplicated_captions, f, indent=2)
    print(f"  ✓ Saved text_captions_deduplicated.json ({len(deduplicated_captions)} captions)")

    # Save caption metadata if provided
    if caption_metadata:
        with open(output_path / "text_caption_metadata.json", "w") as f:
            json.dump(caption_metadata, f, indent=2)
        print(f"  ✓ Saved text_caption_metadata.json")

    # Create comprehensive metadata
    metadata = {
        "processing_stage": "deduplication_filtering",
        "timestamp": str(np.datetime64("now")),
        "original_cells": stats["original_cells"],
        "filtered_cells": stats["filtered_cells"],
        "cells_removed": stats["cells_removed"],
        "removal_rate_percent": round(100.0 * stats["cells_removed"] / stats["original_cells"], 2),
        "original_captions": 1088,
        "deduplicated_captions": 69,
        "cell_embeddings": {
            "shape": list(cell_emb.shape),
            "dtype": str(cell_emb.dtype),
            "path": "cell_embeddings_dedup.npy",
            "description": "Filtered cell embeddings (only from characterized cells)",
        },
        "text_embeddings": {
            "shape": list(text_emb.shape),
            "dtype": str(text_emb.dtype),
            "path": "text_embeddings_dedup.npy",
            "description": "Deduplicated text embeddings (69 groups only)",
        },
        "text_group_ids": {
            "shape": list(text_group_ids.shape),
            "dtype": str(text_group_ids.dtype),
            "path": "text_group_ids_dedup.npy",
            "value_range": [int(text_group_ids.min()), int(text_group_ids.max())],
            "description": "Remapped group IDs (0-68) for each cell",
        },
        "sample_ids": {
            "shape": list(sample_ids.shape),
            "dtype": str(sample_ids.dtype),
            "path": "sample_ids_dedup.npy",
            "description": "Sample IDs of filtered cells (original indices)",
        },
        "text_captions": {
            "count": len(deduplicated_captions),
            "path": "text_captions_deduplicated.json",
            "description": "Deduplicated captions for all 69 groups",
        },
        "text_metadata": {
            "path": "text_caption_metadata.json",
            "description": "Metadata about captions (if available)",
        },
        "filtering_reason": "Remove uncharacterized cells (mapped to None in text_group_mapping.json)",
        "statistics": {
            "cells_per_group": dict(stats["kept_by_group"]),
            "groups_with_no_cells": 0,  # Will calculate
        },
    }

    # Calculate groups with no cells
    groups_with_cells = set(stats["kept_by_group"].keys())
    expected_groups = set(range(69))
    empty_groups = list(expected_groups - groups_with_cells)
    metadata["statistics"]["groups_with_no_cells"] = len(empty_groups)
    if empty_groups:
        metadata["statistics"]["empty_groups"] = empty_groups

    with open(output_path / "METADATA_DEDUP.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved METADATA_DEDUP.json")


def generate_compatibility_check(
    stats: Dict,
    deduplicated_captions: Dict[int, str],
    output_dir: str,
) -> Dict:
    """
    Generate a compatibility check report.
    """
    print("\n[*] Generating compatibility check...")

    check = {
        "validation": {
            "total_cells": stats["filtered_cells"],
            "total_captions": len(deduplicated_captions),
            "cells_per_caption": f"{stats['filtered_cells'] / len(deduplicated_captions):.1f} average",
            "status": "READY_FOR_TRAINING",
        },
        "cell_removal": {
            "reason": "Uncharacterized cells (None in text_group_mapping)",
            "total_removed": stats["cells_removed"],
            "percent_removed": f"{100 * stats['cells_removed'] / stats['original_cells']:.2f}%",
        },
        "cells_per_group": {
            "min_cells": min(stats["kept_by_group"].values()),
            "max_cells": max(stats["kept_by_group"].values()),
            "median_cells": int(np.median(list(stats["kept_by_group"].values()))),
            "mean_cells": f"{np.mean(list(stats['kept_by_group'].values())):.1f}",
            "groups_below_100_cells": sum(1 for v in stats["kept_by_group"].values() if v < 100),
            "groups_below_500_cells": sum(1 for v in stats["kept_by_group"].values() if v < 500),
        },
        "detailed_group_stats": {
            int(group_id): {
                "cell_count": cell_count,
                "caption": deduplicated_captions[int(group_id)],
            }
            for group_id, cell_count in sorted(stats["kept_by_group"].items())
        },
        "warnings": [],
    }

    # Add warnings for potential issues
    if check["cells_per_group"]["groups_below_100_cells"] > 0:
        check["warnings"].append(
            f"WARNING: {check['cells_per_group']['groups_below_100_cells']} groups have < 100 cells"
        )

    if check["cells_per_group"]["min_cells"] < 50:
        check["warnings"].append(
            f"WARNING: Minimum cells per group is {check['cells_per_group']['min_cells']} (very sparse)"
        )

    output_file = Path(output_dir) / "compatibility_check.json"
    with open(output_file, "w") as f:
        json.dump(check, f, indent=2)

    print(f"  ✓ Saved compatibility_check.json")

    # Print summary
    print(f"\n[*] Compatibility Summary:")
    print(f"  Total cells: {check['validation']['total_cells']:,}")
    print(f"  Total captions: {check['validation']['total_captions']}")
    print(f"  Average cells/caption: {check['validation']['cells_per_caption']}")
    print(f"  Min cells/group: {check['cells_per_group']['min_cells']}")
    print(f"  Max cells/group: {check['cells_per_group']['max_cells']}")
    print(f"  Median cells/group: {check['cells_per_group']['median_cells']}")
    if check["warnings"]:
        print(f"\n  Warnings:")
        for warning in check["warnings"]:
            print(f"    - {warning}")

    return check


def create_config_template(output_dir: str) -> None:
    """Create a config template for CLOP training with deduplicated data."""
    print("\n[*] Creating config template...")

    config_template = """# CLOP Training Configuration - Deduplicated Captions
# This configuration uses the deduplicated and filtered training data

data:
  cache_dir: "data/cached_latents_v5.2"
  use_deduplicated: true
  use_preprocessed: false

  # Deduplicated data files
  cell_embeddings: "cell_embeddings_dedup.npy"
  text_embeddings: "text_embeddings_dedup.npy"
  text_group_ids: "text_group_ids_dedup.npy"
  sample_ids: "sample_ids_dedup.npy"
  text_captions: "text_captions_deduplicated.json"
  text_metadata: "text_caption_metadata.json"

model:
  cell_embedding_dim: 512
  text_embedding_dim: 1024
  hidden_dim: 768

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.0001
  warmup_steps: 1000

  # Weighted sampling based on deduplicated group sizes
  use_weighted_sampling: true

optimization:
  optimizer: "adamw"
  scheduler: "cosine"
  gradient_clip: 1.0

logging:
  log_dir: "logs/clop_dedup"
  save_every_n_steps: 1000
  eval_every_n_steps: 500

metadata:
  filtered_cells: 167245
  deduplicated_captions: 69
  original_cells_before_filtering: 220304
  cells_removed_reason: "uncharacterized (null entries in text_group_mapping)"
"""

    config_path = Path(output_dir) / "clop_dedup.yaml"
    with open(config_path, "w") as f:
        f.write(config_template)

    print(f"  ✓ Saved clop_dedup.yaml")


def main():
    """Main execution."""
    cache_dir = "/home/zeyufu/Desktop/CLOP-DiT/data/cached_latents_v5.2"
    output_dir = cache_dir  # Save in the same directory

    print("=" * 70)
    print("CLOP Deduplicated Training Data Preparation")
    print("=" * 70)

    try:
        # Load inputs
        cell_emb, text_emb_all, text_group_ids, sample_ids, text_mapping = load_input_data(cache_dir)
        deduplicated_captions = load_deduplicated_captions(cache_dir)

        # Check for caption metadata
        caption_metadata_path = Path(cache_dir) / "text_caption_metadata.json"
        caption_metadata = None
        if caption_metadata_path.exists():
            with open(caption_metadata_path) as f:
                caption_metadata = json.load(f)
            print(f"  ✓ Loaded caption metadata ({len(caption_metadata)} entries)")

        # Filter and remap
        filtered_cell_emb, new_group_ids, filtered_sample_ids, stats = filter_and_remap(
            cell_emb, text_group_ids, sample_ids, text_mapping, cache_dir
        )

        # Get deduplicated text embeddings
        filtered_text_emb = get_text_embeddings_for_groups(text_emb_all, text_mapping)

        # Save outputs
        save_outputs(
            output_dir,
            filtered_cell_emb,
            filtered_text_emb,
            new_group_ids,
            filtered_sample_ids,
            deduplicated_captions,
            caption_metadata,
            stats,
        )

        # Generate compatibility check
        check = generate_compatibility_check(
            stats, deduplicated_captions, output_dir
        )

        # Create config template
        create_config_template(output_dir)

        print("\n" + "=" * 70)
        print("SUCCESS: Deduplicated training data prepared!")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
