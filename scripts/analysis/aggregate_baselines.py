#!/usr/bin/env python3
"""Aggregate baselines — collect per-method metrics into a unified comparison table.

Reads the baselines manifest and per-method metric files, computes:
  - Per-method summary statistics (mean, std across seeds where available)
  - Metric rankings across all methods
  - LaTeX-ready comparison table

Output:
  results/baselines/aggregated_comparison.json
  results/baselines/comparison_table.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import RESULTS_DIR

logger = logging.getLogger(__name__)

BASELINES_DIR = RESULTS_DIR / "baselines"
MANIFEST_PATH = BASELINES_DIR / "manifest.json"

# Canonical embedding metrics to aggregate
EMBEDDING_METRICS = [
    "frechet_distance",
    "mean_centroid_cosine",
    "diversity_ratio",
    "coverage",
    "knn_top1",
    "knn_top5",
    "steering_accuracy",
    "linear_accuracy",
]

# Metrics where lower is better
LOWER_IS_BETTER = {"frechet_distance"}


def load_manifest() -> dict:
    """Load baseline manifest."""
    if not MANIFEST_PATH.exists():
        logger.warning("Manifest not found: %s", MANIFEST_PATH)
        return {}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def load_method_metrics(method_slug: str) -> dict:
    """Load per-method metrics from various files."""
    method_dir = BASELINES_DIR / method_slug
    metrics = {}

    # Try embeddings-level metrics
    for candidate in ["metrics.json", "generation_metrics.json", "benchmark_metrics.json"]:
        path = method_dir / candidate
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    metrics.update(data)

    return metrics


def load_clop_dit_metrics() -> dict:
    """Load CLOP-DiT's own metrics from the main results."""
    metrics = {}
    for candidate in ["generation_metrics.json", "baseline_metrics.json"]:
        path = RESULTS_DIR / candidate
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # generation_metrics.json is flat; baseline_metrics has sub-dicts
                    if any(isinstance(v, dict) for v in data.values()):
                        continue  # Skip nested (this is baseline_metrics)
                    metrics.update(data)
    return metrics


def aggregate():
    """Main aggregation logic."""
    manifest = load_manifest()
    methods_info = manifest.get("methods", {})

    all_results = {}

    # CLOP-DiT (our method)
    ours = load_clop_dit_metrics()
    if ours:
        all_results["CLOP-DiT"] = {
            "display_name": "CLOP-DiT",
            "family": "ours",
            "metrics": {k: ours.get(k) for k in EMBEDDING_METRICS if k in ours},
        }

    # Statistical baselines from baseline_metrics.json
    baseline_path = RESULTS_DIR / "baseline_metrics.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baselines = json.load(f)
        for name, bm in baselines.items():
            if isinstance(bm, dict):
                all_results[name] = {
                    "display_name": name,
                    "family": "statistical",
                    "metrics": {k: bm.get(k) for k in EMBEDDING_METRICS if k in bm},
                }

    # External baselines from manifest
    for slug, info in methods_info.items():
        method_metrics = load_method_metrics(slug)
        if method_metrics:
            display_name = info.get("display_name", slug)
            all_results[display_name] = {
                "display_name": display_name,
                "family": info.get("family", "external"),
                "metrics": {k: method_metrics.get(k) for k in EMBEDDING_METRICS if k in method_metrics},
            }

    # Compute rankings per metric
    rankings = {}
    for metric in EMBEDDING_METRICS:
        values = []
        for name, data in all_results.items():
            val = data["metrics"].get(metric)
            if val is not None:
                values.append((name, float(val)))
        if not values:
            continue
        reverse = metric not in LOWER_IS_BETTER
        values.sort(key=lambda x: x[1], reverse=reverse)
        rankings[metric] = {name: rank + 1 for rank, (name, _) in enumerate(values)}

    # Build comparison table
    table_rows = []
    for name, data in all_results.items():
        row = {"method": name, "family": data["family"]}
        row.update(data["metrics"])
        # Add mean rank
        method_ranks = [rankings[m].get(name, len(all_results)) for m in rankings]
        row["mean_rank"] = np.mean(method_ranks) if method_ranks else None
        table_rows.append(row)

    # Sort by mean rank
    table_rows.sort(key=lambda r: r.get("mean_rank") or 999)

    # Save
    output = {
        "methods": all_results,
        "rankings": rankings,
        "comparison_table": table_rows,
    }
    out_json = BASELINES_DIR / "aggregated_comparison.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Saved aggregated comparison: %s", out_json)

    # Save CSV
    out_csv = BASELINES_DIR / "comparison_table.csv"
    if table_rows:
        import csv
        fieldnames = ["method", "family"] + EMBEDDING_METRICS + ["mean_rank"]
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(table_rows)
        logger.info("Saved comparison table: %s", out_csv)

    # Print summary
    print("\n" + "=" * 70)
    print("Baseline Aggregation Summary")
    print("=" * 70)
    for row in table_rows:
        rank_str = f"(rank {row['mean_rank']:.1f})" if row.get("mean_rank") else ""
        metrics_str = ", ".join(
            f"{k}={row[k]:.4f}" for k in EMBEDDING_METRICS if row.get(k) is not None
        )
        print(f"  {row['method']:30s} {rank_str:12s} {metrics_str}")
    print(f"\nMethods: {len(all_results)} | Metrics: {len(rankings)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    aggregate()
