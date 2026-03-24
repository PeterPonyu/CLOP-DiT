#!/usr/bin/env python3
"""Data audit for CLOP-DiT pipeline.

Validates preprocessing, split coverage, and embedding quality
to catch silent data issues before training.

Usage:
    python scripts/audit_data.py [--cache-dir data/cached_latents]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def audit(cache_dir: str) -> dict:
    """Run comprehensive data audit and return results dict."""
    p = Path(cache_dir)
    report = {"status": "ok", "warnings": [], "errors": []}

    # ── 1. File existence ──
    required_dedup = [
        "cell_embeddings_dedup_preprocessed.npy",
        "text_embeddings_dedup_preprocessed.npy",
        "text_group_ids_dedup.npy",
        "sample_ids_dedup.npy",
        "text_captions_deduplicated.json",
    ]
    for fname in required_dedup:
        if not (p / fname).exists():
            report["errors"].append(f"Missing required file: {fname}")

    if report["errors"]:
        report["status"] = "FAIL"
        return report

    # ── 2. Load data ──
    cell_emb = np.load(p / "cell_embeddings_dedup_preprocessed.npy", mmap_mode="r")
    text_emb = np.load(p / "text_embeddings_dedup_preprocessed.npy", mmap_mode="r")
    group_ids = np.load(p / "text_group_ids_dedup.npy")
    sample_ids = np.load(p / "sample_ids_dedup.npy")

    with open(p / "text_captions_deduplicated.json") as f:
        captions = json.load(f)

    n_cells = len(cell_emb)
    n_types = len(np.unique(group_ids))
    n_datasets = len(np.unique(sample_ids))
    n_captions = len(captions)

    report["shape"] = {
        "cell_embeddings": list(cell_emb.shape),
        "text_embeddings": list(text_emb.shape),
        "group_ids": list(group_ids.shape),
        "sample_ids": list(sample_ids.shape),
    }
    report["counts"] = {
        "cells": n_cells,
        "cell_types": n_types,
        "datasets": n_datasets,
        "captions": n_captions,
    }

    # ── 3. Dimension consistency ──
    if len(group_ids) != n_cells:
        report["errors"].append(
            f"group_ids length ({len(group_ids)}) != cells ({n_cells})"
        )
    if len(sample_ids) != n_cells:
        report["errors"].append(
            f"sample_ids length ({len(sample_ids)}) != cells ({n_cells})"
        )
    if text_emb.shape[0] != n_types:
        report["warnings"].append(
            f"text_emb rows ({text_emb.shape[0]}) != unique types ({n_types})"
        )
    if n_captions != n_types:
        report["warnings"].append(
            f"caption count ({n_captions}) != unique types ({n_types})"
        )

    # ── 4. Group ID range ──
    gid_min, gid_max = int(group_ids.min()), int(group_ids.max())
    if gid_min != 0 or gid_max != n_types - 1:
        report["warnings"].append(
            f"group_ids range [{gid_min}, {gid_max}] expected [0, {n_types-1}]"
        )

    # ── 5. Embedding norms (should be ~1.0 after preprocessing) ──
    cell_sample = np.array(cell_emb[:5000])
    cell_norms = np.linalg.norm(cell_sample, axis=1)
    text_norms = np.linalg.norm(np.array(text_emb), axis=1)

    report["norms"] = {
        "cell_mean": float(np.mean(cell_norms)),
        "cell_std": float(np.std(cell_norms)),
        "cell_min": float(np.min(cell_norms)),
        "cell_max": float(np.max(cell_norms)),
        "text_mean": float(np.mean(text_norms)),
        "text_std": float(np.std(text_norms)),
    }

    if abs(np.mean(cell_norms) - 1.0) > 0.05:
        report["warnings"].append(
            f"Cell embeddings NOT unit-normed (mean norm={np.mean(cell_norms):.4f})"
        )
    if abs(np.mean(text_norms) - 1.0) > 0.05:
        report["warnings"].append(
            f"Text embeddings NOT unit-normed (mean norm={np.mean(text_norms):.4f})"
        )

    # ── 6. Pairwise cosine similarity (collapse check) ──
    # High mean cos_sim = collapsed embeddings (bad preprocessing)
    cell_cos = float(np.mean(cell_sample @ cell_sample.T))
    text_arr = np.array(text_emb)
    text_cos_matrix = text_arr @ text_arr.T
    np.fill_diagonal(text_cos_matrix, 0)
    n_t = text_arr.shape[0]
    text_cos_mean = float(text_cos_matrix.sum() / (n_t * (n_t - 1)))

    report["cosine_similarity"] = {
        "cell_mean_pairwise": cell_cos,
        "text_mean_pairwise": text_cos_mean,
    }

    if cell_cos > 0.5:
        report["warnings"].append(
            f"Cell embeddings may be collapsed (mean pairwise cos_sim={cell_cos:.4f})"
        )
    if text_cos_mean > 0.5:
        report["warnings"].append(
            f"Text embeddings may be collapsed (mean pairwise cos_sim={text_cos_mean:.4f})"
        )

    # ── 7. Type distribution ──
    unique_types, type_counts = np.unique(group_ids, return_counts=True)
    sort_idx = np.argsort(type_counts)
    smallest_types = [(int(unique_types[i]), int(type_counts[i])) for i in sort_idx[:5]]
    largest_types = [(int(unique_types[i]), int(type_counts[i])) for i in sort_idx[-5:]]

    report["type_distribution"] = {
        "min_cells": int(type_counts.min()),
        "max_cells": int(type_counts.max()),
        "median_cells": int(np.median(type_counts)),
        "mean_cells": float(np.mean(type_counts)),
        "smallest_5": smallest_types,
        "largest_5": largest_types,
        "types_under_100": int(np.sum(type_counts < 100)),
        "types_under_50": int(np.sum(type_counts < 50)),
    }

    # ── 8. Dataset coverage per type ──
    types_in_one_dataset = 0
    for gid in unique_types:
        mask = group_ids == gid
        n_ds = len(np.unique(sample_ids[mask]))
        if n_ds == 1:
            types_in_one_dataset += 1

    report["coverage"] = {
        "types_in_single_dataset": types_in_one_dataset,
    }
    if types_in_one_dataset > 0:
        report["warnings"].append(
            f"{types_in_one_dataset} cell type(s) exist in only 1 dataset (fragile)"
        )

    # ── 9. Stratified split simulation ──
    rng = np.random.default_rng(42)
    train_idx, val_idx = [], []
    for gid in unique_types:
        tidx = np.where(group_ids == gid)[0]
        rng.shuffle(tidx)
        n_val = max(1, int(len(tidx) * 0.1))
        val_idx.extend(tidx[:n_val].tolist())
        train_idx.extend(tidx[n_val:].tolist())

    train_types = len(np.unique(group_ids[train_idx]))
    val_types = len(np.unique(group_ids[val_idx]))

    report["stratified_split"] = {
        "train_cells": len(train_idx),
        "val_cells": len(val_idx),
        "train_types": train_types,
        "val_types": val_types,
        "all_types_in_val": val_types == n_types,
    }

    # ── 10. Confusable text pairs ──
    text_cos_full = text_arr @ text_arr.T
    np.fill_diagonal(text_cos_full, -999)
    confusable = []
    for i in range(n_t):
        for j in range(i + 1, n_t):
            if text_cos_full[i, j] > 0.5:
                confusable.append({
                    "type_a": int(i),
                    "type_b": int(j),
                    "cosine_sim": float(text_cos_full[i, j]),
                })
    report["confusable_pairs"] = confusable
    if confusable:
        report["warnings"].append(
            f"{len(confusable)} text embedding pair(s) with cos_sim > 0.5 "
            f"(biologically similar types that may be hard to distinguish)"
        )

    # ── Final status ──
    if report["errors"]:
        report["status"] = "FAIL"
    elif report["warnings"]:
        report["status"] = "WARN"

    return report


def main():
    parser = argparse.ArgumentParser(description="CLOP-DiT data audit")
    parser.add_argument(
        "--cache-dir",
        default="data/cached_latents",
        help="Path to cached latents directory",
    )
    args = parser.parse_args()

    print(f"Auditing: {args.cache_dir}")
    print("=" * 60)

    report = audit(args.cache_dir)

    # Pretty print
    print(f"\nStatus: {report['status']}")
    print(f"\nData Shape:")
    for k, v in report["shape"].items():
        print(f"  {k}: {v}")
    print(f"\nCounts:")
    for k, v in report["counts"].items():
        print(f"  {k}: {v}")
    print(f"\nEmbedding Norms:")
    for k, v in report["norms"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nCosine Similarity (collapse check):")
    for k, v in report["cosine_similarity"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nType Distribution:")
    td = report["type_distribution"]
    print(f"  Range: [{td['min_cells']}, {td['max_cells']}], median={td['median_cells']}")
    print(f"  Types <100 cells: {td['types_under_100']}, <50 cells: {td['types_under_50']}")
    print(f"\nStratified Split (val_split=0.1):")
    ss = report["stratified_split"]
    print(f"  Train: {ss['train_cells']} cells ({ss['train_types']} types)")
    print(f"  Val:   {ss['val_cells']} cells ({ss['val_types']} types)")
    print(f"  All types in val: {ss['all_types_in_val']}")

    if report["confusable_pairs"]:
        print(f"\nConfusable Text Pairs (cos_sim > 0.5):")
        for pair in report["confusable_pairs"]:
            print(f"  Type {pair['type_a']} ↔ {pair['type_b']}: {pair['cosine_sim']:.3f}")

    if report["warnings"]:
        print(f"\nWarnings ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"  ⚠ {w}")

    if report["errors"]:
        print(f"\nErrors ({len(report['errors'])}):")
        for e in report["errors"]:
            print(f"  ✗ {e}")

    # Save JSON report
    out_path = Path("results/clop_data_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved: {out_path}")

    return 0 if report["status"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
