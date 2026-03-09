#!/usr/bin/env python3
"""Multi-seed validation for CLOP-DiT.

Trains CLOP + DiT end-to-end with 3 independent seeds, runs evaluation
for each, and produces an aggregated report (mean +/- std) for core metrics
(KNN accuracy, steering accuracy, diversity ratio, centroid cosine).

This addresses Reviewer Concern #1: single-seed evaluation.

Usage:
    python scripts/analysis/run_multi_seed_validation.py
    python scripts/analysis/run_multi_seed_validation.py --seeds 42,123,456 --skip-training
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _parse_seeds(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def run_cmd(cmd: list[str], label: str) -> int:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED ({result.returncode})"
    print(f"  [{status}] {label} ({elapsed:.1f}s)")
    return result.returncode


def train_single_seed(seed: int, output_base: Path) -> int:
    """Train CLOP + DiT for a single seed, saving checkpoints separately."""
    seed_dir = output_base / f"seed_{seed}"
    ckpt_dir = seed_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Train CLOP
    code = run_cmd([
        sys.executable, "scripts/training/04a_train_clop.py",
        "--config", "configs/clop.yaml",
        "--save_dir", str(ckpt_dir),
        "--seed", str(seed),
    ], f"CLOP training (seed={seed})")
    if code != 0:
        return code

    # Train DiT
    code = run_cmd([
        sys.executable, "scripts/training/04b_train_dit.py",
        "--config", "configs/dit.yaml",
        "--save_dir", str(ckpt_dir),
        "--seed", str(seed),
    ], f"DiT training (seed={seed})")
    if code != 0:
        return code

    return 0


def evaluate_single_seed(seed: int, output_base: Path) -> dict | None:
    """Run generation + evaluation for a single seed."""
    seed_dir = output_base / f"seed_{seed}"
    ckpt_dir = seed_dir / "checkpoints"
    results_dir = seed_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoints
    clop_ckpt = ckpt_dir / "CLOP" / "best" / "clop_best.pth"
    dit_ckpt = ckpt_dir / "DiT" / "best" / "dit_best.pth"

    # Fall back to default checkpoints if seed-specific ones don't exist
    if not clop_ckpt.exists():
        clop_ckpt = _PROJECT_ROOT / "models" / "checkpoints" / "CLOP" / "best" / "clop_best.pth"
    if not dit_ckpt.exists():
        dit_ckpt = _PROJECT_ROOT / "models" / "checkpoints" / "DiT" / "best" / "dit_best.pth"

    if not clop_ckpt.exists() or not dit_ckpt.exists():
        print(f"  [SKIP] Checkpoints not found for seed {seed}")
        return None

    # Generate embeddings
    code = run_cmd([
        sys.executable, "scripts/inference/generate_embeddings.py",
        "--condition-mode", "condition_noise",
        "--noise-scale", "0.03",
        "--cfg-scale", "2.0",
        "--num-per-type", "200",
        "--num-steps", "10",
        "--solver", "euler",
        "--output-dir", str(results_dir),
        "--seed", str(seed),
    ], f"Generate embeddings (seed={seed}, CFG=2.0 Euler)")
    if code != 0:
        return None

    # Also generate with high-diversity config
    results_div_dir = seed_dir / "results_diversity"
    results_div_dir.mkdir(parents=True, exist_ok=True)
    code = run_cmd([
        sys.executable, "scripts/inference/generate_embeddings.py",
        "--condition-mode", "condition_noise",
        "--noise-scale", "0.03",
        "--cfg-scale", "1.0",
        "--num-per-type", "200",
        "--num-steps", "10",
        "--solver", "midpoint",
        "--output-dir", str(results_div_dir),
        "--seed", str(seed),
    ], f"Generate embeddings (seed={seed}, CFG=1.0 Midpoint)")

    # Run diversity diagnostics
    code = run_cmd([
        sys.executable, "scripts/analysis/diversity_diagnostics.py",
        "--num-per-type", "200",
        "--output-dir", str(results_dir),
        "--seed", str(seed),
    ], f"Diversity diagnostics (seed={seed})")

    # Collect metrics from JSON files
    metrics = {"seed": seed}

    # Load generation metrics
    gen_metrics_path = results_dir / "generation_metrics.json"
    if gen_metrics_path.exists():
        with open(gen_metrics_path) as f:
            gm = json.load(f)
        metrics.update({
            k: gm.get(k) for k in [
                "knn_top1", "knn_top5", "steering_accuracy",
                "diversity_ratio", "linear_accuracy", "frechet_distance",
                "centroid_cosine",
            ] if k in gm
        })

    # Load diversity diagnostics
    div_path = results_dir / "diversity_diagnostics.json"
    if div_path.exists():
        with open(div_path) as f:
            dm = json.load(f)
        if "diversity_ratio" not in metrics and "diversity_ratio" in dm:
            metrics["diversity_ratio"] = dm["diversity_ratio"]

    return metrics


def aggregate_results(all_metrics: list[dict]) -> dict:
    """Compute mean +/- std across seeds."""
    metric_keys = [
        "knn_top1", "knn_top5", "steering_accuracy",
        "diversity_ratio", "linear_accuracy", "frechet_distance",
        "centroid_cosine",
    ]

    summary = {
        "n_seeds": len(all_metrics),
        "seeds": [m["seed"] for m in all_metrics],
    }

    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m and m[key] is not None]
        if values:
            arr = np.array(values)
            summary[key] = {
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else 0.0,
                "min": round(float(arr.min()), 4),
                "max": round(float(arr.max()), 4),
                "values": [round(float(v), 4) for v in arr],
            }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-seed validation for CLOP-DiT")
    parser.add_argument("--seeds", default="42,123,456",
                        help="Comma-separated list of random seeds")
    parser.add_argument("--output-dir", default="results/multi_seed",
                        help="Output directory for seed-specific results")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only run evaluation (use existing checkpoints)")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    output_base = _PROJECT_ROOT / args.output_dir
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Multi-seed validation: seeds={seeds}")
    print(f"Output: {output_base}")

    # Phase 1: Training (optional)
    if not args.skip_training:
        for seed in seeds:
            code = train_single_seed(seed, output_base)
            if code != 0:
                print(f"Training failed for seed {seed}")
                return code

    # Phase 2: Evaluation
    all_metrics = []
    for seed in seeds:
        metrics = evaluate_single_seed(seed, output_base)
        if metrics is not None:
            all_metrics.append(metrics)

    if not all_metrics:
        print("No evaluation results collected. Check logs above.")
        return 1

    # Phase 3: Aggregation
    summary = aggregate_results(all_metrics)

    # Save raw per-seed and aggregated results
    report = {
        "per_seed": all_metrics,
        "aggregated": summary,
    }
    report_path = output_base / "multi_seed_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  MULTI-SEED VALIDATION RESULTS ({len(all_metrics)} seeds)")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*70}")
    for key in ["knn_top1", "knn_top5", "steering_accuracy",
                 "diversity_ratio", "linear_accuracy", "centroid_cosine"]:
        if key in summary:
            s = summary[key]
            print(f"{key:<25} {s['mean']:>10.4f} {s['std']:>10.4f} "
                  f"{s['min']:>10.4f} {s['max']:>10.4f}")
    print(f"{'='*70}")
    print(f"Report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
