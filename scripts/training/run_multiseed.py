#!/usr/bin/env python3
"""Run DiT training across multiple seeds for statistical confidence.

Trains the DiT model N times with different random seeds, saving each
run's checkpoint and history to a seed-specific subdirectory. After all
runs complete, aggregates headline metrics (mean +/- std) into a
summary JSON for reporting.

Usage:
    python scripts/training/run_multiseed.py --config configs/dit.yaml
    python scripts/training/run_multiseed.py --config configs/dit.yaml --seeds 42 123 456
    python scripts/training/run_multiseed.py --config configs/dit.yaml --seeds 42 123 456 789 1024
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.train_dit import DiTTrainer
from src.utils.helpers import seed_everything
from src.utils.logging_config import setup_logging


DEFAULT_SEEDS = [42, 123, 456]


def train_single_seed(config: dict, seed: int, base_save_dir: Path) -> dict:
    """Train one DiT run with a specific seed. Returns the training history."""
    run_dir = base_save_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = dict(config)
    run_config["seed"] = seed
    run_config["save_dir"] = str(run_dir)

    print(f"\n{'='*60}")
    print(f"  DiT Training — Seed {seed}")
    print(f"  Save dir: {run_dir}")
    print(f"{'='*60}\n")

    seed_everything(seed)
    trainer = DiTTrainer.from_config(run_config)

    param_info = trainer.model.count_parameters()
    print(f"DiT Model: {param_info['trainable_M']} trainable parameters")

    history = trainer.train()

    # Run final generation evaluation
    gen_metrics = trainer.evaluate_generation(num_samples=512)

    # Save run summary
    run_summary = {
        "seed": seed,
        "best_val_loss": trainer.best_val_loss,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "final_val_cosine": history["val_cosine_sim"][-1] if history["val_cosine_sim"] else None,
        "generation_metrics": gen_metrics,
    }

    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    return run_summary


def aggregate_results(summaries: list, output_path: Path):
    """Aggregate results across seeds into mean +/- std summary."""
    # Scalar metrics
    scalar_keys = ["best_val_loss", "final_train_loss", "final_val_loss", "final_val_cosine"]
    agg = {"n_seeds": len(summaries), "seeds": [s["seed"] for s in summaries]}

    for key in scalar_keys:
        values = [s[key] for s in summaries if s[key] is not None]
        if values:
            agg[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

    # Generation metrics
    gen_keys = set()
    for s in summaries:
        if "generation_metrics" in s and s["generation_metrics"]:
            gen_keys.update(s["generation_metrics"].keys())

    gen_agg = {}
    for key in sorted(gen_keys):
        values = []
        for s in summaries:
            gm = s.get("generation_metrics", {})
            if gm and key in gm and isinstance(gm[key], (int, float)):
                values.append(float(gm[key]))
        if values:
            gen_agg[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
    agg["generation_metrics"] = gen_agg

    with open(output_path, "w") as f:
        json.dump(agg, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Multi-Seed Aggregate ({len(summaries)} seeds)")
    print(f"{'='*60}")
    for key in scalar_keys:
        if key in agg:
            m, s = agg[key]["mean"], agg[key]["std"]
            print(f"  {key}: {m:.6f} +/- {s:.6f}")
    if gen_agg:
        print(f"\n  Generation metrics:")
        for key in sorted(gen_agg.keys()):
            m, s = gen_agg[key]["mean"], gen_agg[key]["std"]
            print(f"    {key}: {m:.4f} +/- {s:.4f}")
    print(f"\n  Saved to: {output_path}")

    return agg


def main():
    parser = argparse.ArgumentParser(description="Multi-seed DiT training")
    parser.add_argument("--config", type=str, default="configs/dit.yaml",
                        help="YAML config file")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds to use")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output directory (default: models/checkpoints/multiseed)")
    args = parser.parse_args()

    setup_logging()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    base_save_dir = Path(args.output_dir or config.get("save_dir", "models/checkpoints"))
    base_save_dir = base_save_dir / "multiseed"
    base_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Multi-seed training: seeds={args.seeds}")
    print(f"Output directory: {base_save_dir}")

    summaries = []
    for seed in args.seeds:
        summary = train_single_seed(config, seed, base_save_dir)
        summaries.append(summary)

    aggregate_results(summaries, base_save_dir / "aggregate_summary.json")

    print(f"\nAll {len(args.seeds)} seeds complete!")
    print(f"Results in: {base_save_dir}/")


if __name__ == "__main__":
    main()
