#!/usr/bin/env python3
"""
Example: Loading and using deduplicated captions for CLOP training.

This script shows how to:
1. Load the deduplicated data
2. Access aligned cell/text embeddings
3. Implement weighted sampling for imbalanced groups
4. Create train/test splits
"""

import numpy as np
import json
from collections import Counter
from typing import Dict, Tuple


class CLOPDedupDataLoader:
    """Data loader for deduplicated CLOP training data."""

    def __init__(self, cache_dir: str = "data/cached_latents_v5.2"):
        """Initialize and load all deduplicated data."""
        self.cache_dir = cache_dir
        self._load_data()

    def _load_data(self):
        """Load all necessary files."""
        print("[*] Loading deduplicated CLOP data...")

        self.cell_embeddings = np.load(f"{self.cache_dir}/cell_embeddings_dedup.npy")
        self.text_embeddings = np.load(f"{self.cache_dir}/text_embeddings_dedup.npy")
        self.text_group_ids = np.load(f"{self.cache_dir}/text_group_ids_dedup.npy")
        self.sample_ids = np.load(f"{self.cache_dir}/sample_ids_dedup.npy")

        with open(f"{self.cache_dir}/text_captions_deduplicated.json") as f:
            self.captions = json.load(f)

        with open(f"{self.cache_dir}/compatibility_check.json") as f:
            self.compatibility = json.load(f)

        print(f"  ✓ Loaded cell embeddings: {self.cell_embeddings.shape}")
        print(f"  ✓ Loaded text embeddings: {self.text_embeddings.shape}")
        print(f"  ✓ Loaded {len(self.captions)} captions")
        print(f"  ✓ Total cells: {len(self.cell_embeddings):,}")

    def print_summary(self):
        """Print summary statistics."""
        print("\n[*] Deduplicated Data Summary:")
        print(f"  Total cells: {len(self.cell_embeddings):,}")
        print(f"  Total cell types: {len(self.captions)}")
        print(f"  Cell embedding dim: {self.cell_embeddings.shape[1]}")
        print(f"  Text embedding dim: {self.text_embeddings.shape[1]}")

        # Group stats
        group_stats = self.compatibility["cells_per_group"]
        print(f"\n  Cell counts per group:")
        print(f"    Min: {group_stats['min_cells']}")
        print(f"    Max: {group_stats['max_cells']}")
        print(f"    Median: {group_stats['median_cells']}")
        print(f"    Mean: {group_stats['mean_cells']}")

    def get_group_distribution(self) -> Dict[int, int]:
        """Get number of cells per group."""
        counts = Counter(self.text_group_ids)
        return {int(k): int(v) for k, v in counts.items()}

    def get_group_weights(self, method: str = "inverse") -> np.ndarray:
        """
        Compute weights for weighted sampling.

        Args:
            method: "inverse" (1/count) or "log" (1/log(count))

        Returns:
            Array of weights for each cell
        """
        distribution = self.get_group_distribution()
        total_groups = max(distribution.keys()) + 1

        weights = np.zeros(len(self.cell_embeddings))

        for group_id, count in distribution.items():
            mask = self.text_group_ids == group_id

            if method == "inverse":
                weight = 1.0 / count
            elif method == "log":
                weight = 1.0 / np.log(count + 2)
            else:
                raise ValueError(f"Unknown method: {method}")

            weights[mask] = weight

        # Normalize
        weights /= weights.sum()
        return weights

    def get_caption_for_group(self, group_id: int) -> str:
        """Get full caption text for a group."""
        return self.captions[str(group_id)]

    def get_cells_for_group(self, group_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get cell and sample IDs for a specific group."""
        mask = self.text_group_ids == group_id
        cell_indices = np.where(mask)[0]
        cells = self.cell_embeddings[cell_indices]
        samples = self.sample_ids[cell_indices]
        return cells, samples

    def create_train_test_split(
        self, train_ratio: float = 0.8, random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create stratified train/test split respecting group proportions.

        Returns:
            train_indices, test_indices (as numpy arrays)
        """
        np.random.seed(random_seed)
        train_indices = []
        test_indices = []

        distribution = self.get_group_distribution()

        for group_id, count in distribution.items():
            mask = self.text_group_ids == group_id
            group_indices = np.where(mask)[0]

            # Shuffle within group
            np.random.shuffle(group_indices)

            # Split
            split_point = int(count * train_ratio)
            train_indices.extend(group_indices[:split_point])
            test_indices.extend(group_indices[split_point:])

        return np.array(train_indices), np.array(test_indices)


def example_basic_usage():
    """Example 1: Basic usage."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Data Loading")
    print("=" * 70)

    loader = CLOPDedupDataLoader()
    loader.print_summary()

    # Get caption for first cell type
    print(f"\nCaption for group 0:")
    print(f"  {loader.get_caption_for_group(0)}")


def example_group_analysis():
    """Example 2: Analyze group-level statistics."""
    print("\n" + "=" * 70)
    print("Example 2: Group Analysis")
    print("=" * 70)

    loader = CLOPDedupDataLoader()
    distribution = loader.get_group_distribution()

    print(f"\nTop 5 cell types by count:")
    for i, (group_id, count) in enumerate(sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:5]):
        caption = loader.get_caption_for_group(group_id)[:60] + "..."
        print(f"  {i+1}. Group {group_id}: {count:5d} cells - {caption}")

    print(f"\nBottom 5 cell types by count:")
    for i, (group_id, count) in enumerate(sorted(distribution.items(), key=lambda x: x[1])[:5]):
        caption = loader.get_caption_for_group(group_id)[:60] + "..."
        print(f"  {i+1}. Group {group_id}: {count:5d} cells - {caption}")


def example_weighted_sampling():
    """Example 3: Weighted sampling for imbalanced groups."""
    print("\n" + "=" * 70)
    print("Example 3: Weighted Sampling (Handling Imbalance)")
    print("=" * 70)

    loader = CLOPDedupDataLoader()

    # Get weights for inverse weighting
    weights = loader.get_group_weights(method="inverse")

    print(f"\nWeight statistics:")
    print(f"  Min weight: {weights.min():.6f}")
    print(f"  Max weight: {weights.max():.6f}")
    print(f"  Mean weight: {weights.mean():.6f}")
    print(f"  Total weight: {weights.sum():.6f}")

    # Sample from distribution
    print(f"\nSampling 1000 cells with weighted distribution:")
    sampled_indices = np.random.choice(len(weights), 1000, p=weights)
    sampled_groups = loader.text_group_ids[sampled_indices]

    # Check distribution in sample
    sample_dist = Counter(sampled_groups)
    print(f"  Each group appears roughly equally in sample:")
    for group_id in sorted(sample_dist.keys()):
        count = sample_dist[group_id]
        print(f"    Group {group_id:2d}: {count:3d} cells")


def example_data_access():
    """Example 4: Efficient data access patterns."""
    print("\n" + "=" * 70)
    print("Example 4: Data Access Patterns")
    print("=" * 70)

    loader = CLOPDedupDataLoader()

    # Example 1: Access cells from a specific group
    print(f"\nAccessing cells from Group 2 (Heat-shock response):")
    cells, samples = loader.get_cells_for_group(2)
    print(f"  Found {len(cells)} cells")
    print(f"  Cell embeddings shape: {cells.shape}")
    print(f"  Sample IDs (first 5): {samples[:5]}")

    # Example 2: Get paired cell/text embeddings
    print(f"\nAccessing paired embeddings for a batch:")
    batch_size = 32
    random_indices = np.random.choice(len(loader.cell_embeddings), batch_size)

    batch_cell_emb = loader.cell_embeddings[random_indices]
    batch_group_ids = loader.text_group_ids[random_indices]
    batch_text_emb = loader.text_embeddings[batch_group_ids]

    print(f"  Cell embeddings: {batch_cell_emb.shape}")
    print(f"  Text embeddings: {batch_text_emb.shape}")
    print(f"  Group IDs: {batch_group_ids[:5]}")


def example_train_test_split():
    """Example 5: Creating stratified train/test splits."""
    print("\n" + "=" * 70)
    print("Example 5: Stratified Train/Test Split")
    print("=" * 70)

    loader = CLOPDedupDataLoader()

    # Create split
    train_idx, test_idx = loader.create_train_test_split(train_ratio=0.8)

    print(f"\nSplit results:")
    print(f"  Total cells: {len(train_idx) + len(test_idx):,}")
    print(f"  Training cells: {len(train_idx):,} (80%)")
    print(f"  Testing cells: {len(test_idx):,} (20%)")

    # Verify stratification
    train_distribution = Counter(loader.text_group_ids[train_idx])
    test_distribution = Counter(loader.text_group_ids[test_idx])

    print(f"\nDistribution preserved across split:")
    all_distribution = loader.get_group_distribution()
    for group_id in sorted(all_distribution.keys())[:5]:  # Show first 5
        all_count = all_distribution[group_id]
        train_count = train_distribution.get(group_id, 0)
        test_count = test_distribution.get(group_id, 0)
        train_pct = 100 * train_count / all_count
        print(f"  Group {group_id}: {train_count} train / {test_count} test (split: {train_pct:.1f}% train)")


def main():
    """Run all examples."""
    example_basic_usage()
    example_group_analysis()
    example_weighted_sampling()
    example_data_access()
    example_train_test_split()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
