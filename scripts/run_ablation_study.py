#!/usr/bin/env python3
"""CLOP Ablation Study & Replication Framework.

Runs a series of structured ablation experiments with proper artifact
retention for publication. Each experiment gets an isolated output
directory with config snapshots, checkpoints, histories, and logs.

This framework is designed for:
  1. Systematic ablation studies (one change at a time)
  2. Reproducible experiments (deterministic seeds, config snapshots)
  3. Publication-ready result aggregation (tables, comparisons)

Usage:
    # Run all ablation experiments sequentially
    python scripts/run_ablation_study.py --base_config configs/clop_v8.2.yaml

    # Run a specific ablation only
    python scripts/run_ablation_study.py --base_config configs/clop_v8.2.yaml --ablation no_mixup

    # List available ablations
    python scripts/run_ablation_study.py --list

    # Generate comparison report from completed experiments
    python scripts/run_ablation_study.py --report

    # Resume from a specific ablation (skip completed ones)
    python scripts/run_ablation_study.py --base_config configs/clop_v8.2.yaml --resume
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "ablations"
PYTHON = sys.executable


# ── Ablation Definitions ──────────────────────────────────────────────
# Each ablation modifies one aspect of the base config.
# Format: { "name": {"description": ..., "overrides": {key: value}} }

ABLATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "description": "Full v8.2 config (all improvements enabled)",
        "overrides": {},
    },
    "no_variants": {
        "description": "Disable caption variant augmentation",
        "overrides": {"variant_prob": 0.0},
    },
    "no_mixup": {
        "description": "Disable embedding MixUp regularization",
        "overrides": {"mixup_alpha": 0.0},
    },
    "no_rdrop": {
        "description": "Disable R-Drop consistency regularization",
        "overrides": {"rdrop_weight": 0.0},
    },
    "no_cell_noise": {
        "description": "Disable cell embedding noise augmentation",
        "overrides": {"cell_noise_std": 0.0},
    },
    "low_temperature_cap": {
        "description": "Temperature cap at 20 (v8.1 level) vs 50",
        "overrides": {"max_temperature": 20.0},
    },
    "high_variant_prob": {
        "description": "Higher variant probability (0.5 vs 0.3)",
        "overrides": {"variant_prob": 0.5},
    },
    "no_label_smoothing": {
        "description": "Disable label smoothing",
        "overrides": {"label_smoothing": 0.0},
    },
    "wider_proj": {
        "description": "Wider projection head (768-dim vs 512-dim)",
        "overrides": {"proj_dim": 768},
    },
    "lighter_dropout": {
        "description": "Lower dropout (0.15 vs 0.30)",
        "overrides": {"dropout": 0.15},
    },
    "no_regularization": {
        "description": "Remove all new regularization (variants, MixUp, R-Drop)",
        "overrides": {
            "variant_prob": 0.0,
            "mixup_alpha": 0.0,
            "rdrop_weight": 0.0,
        },
    },
}


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: Dict, path: Path):
    """Save config as YAML."""
    import yaml
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def apply_overrides(config: Dict, overrides: Dict) -> Dict:
    """Apply ablation overrides to a config dict."""
    cfg = copy.deepcopy(config)
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def get_experiment_dir(ablation_name: str) -> Path:
    """Get the output directory for an ablation experiment."""
    return RESULTS_DIR / ablation_name


def is_experiment_complete(ablation_name: str) -> bool:
    """Check if an experiment has completed successfully."""
    exp_dir = get_experiment_dir(ablation_name)
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        return False
    with open(summary_path) as f:
        summary = json.load(f)
    return summary.get("status") == "completed"


def run_single_ablation(
    ablation_name: str,
    base_config: Dict,
    overrides: Dict,
    description: str,
    seed: int = 42,
    dry_run: bool = False,
) -> Optional[Dict]:
    """Run a single ablation experiment.

    Parameters
    ----------
    ablation_name : str
        Name of the ablation.
    base_config : dict
        Base config to modify.
    overrides : dict
        Config overrides for this ablation.
    description : str
        Human-readable description.
    seed : int
        Random seed for reproducibility.
    dry_run : bool
        If True, only save the config without running.

    Returns
    -------
    summary : dict or None
        Experiment summary if completed, None if failed.
    """
    exp_dir = get_experiment_dir(ablation_name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Prepare config
    config = apply_overrides(base_config, overrides)
    config["save_dir"] = str(exp_dir / "checkpoints")
    config["log_file"] = str(exp_dir / "train.log")
    config["run_name"] = f"ablation_{ablation_name}"

    # Save config snapshot
    config_path = exp_dir / "config.yaml"
    save_config(config, config_path)

    # Save ablation metadata
    meta = {
        "ablation_name": ablation_name,
        "description": description,
        "overrides": overrides,
        "base_config_diff": {k: {"base": base_config.get(k), "ablation": v}
                            for k, v in overrides.items()},
        "seed": seed,
        "started_at": datetime.now().isoformat(),
        "status": "running",
    }
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    if dry_run:
        print(f"  [DRY RUN] Config saved to {config_path}")
        return None

    # Run training
    print(f"\n{'='*70}")
    print(f"  ABLATION: {ablation_name}")
    print(f"  {description}")
    print(f"  Overrides: {overrides or '(none — baseline)'}")
    print(f"  Output: {exp_dir}")
    print(f"{'='*70}")

    log_path = exp_dir / "console.log"
    cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "04a_train_clop.py"),
        "--config", str(config_path),
    ]

    start_time = time.time()
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
    elapsed = time.time() - start_time

    # Parse results
    summary = {
        "ablation_name": ablation_name,
        "description": description,
        "overrides": overrides,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "elapsed_human": f"{elapsed/60:.1f} min",
        "return_code": result.returncode,
        "status": "completed" if result.returncode == 0 else "failed",
        "completed_at": datetime.now().isoformat(),
    }

    # Try to load training history
    history_path = exp_dir / "checkpoints" / "clop_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        summary["metrics"] = extract_best_metrics(history)
    else:
        summary["metrics"] = None

    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Update metadata
    meta["status"] = summary["status"]
    meta["completed_at"] = summary["completed_at"]
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    status_str = "✓" if result.returncode == 0 else "✗"
    print(f"\n  [{status_str}] {ablation_name}: {elapsed/60:.1f} min")
    if summary["metrics"]:
        m = summary["metrics"]
        print(f"      Best val_proto_acc: {m['best_val_proto_acc']:.4f} "
              f"(epoch {m['best_epoch']})")
        print(f"      Best val_proto_top5: {m.get('best_val_proto_top5', 0):.4f}")
        print(f"      Best val_proto_top10: {m.get('best_val_proto_top10', 0):.4f}")
        print(f"      Gap (train/val): {m.get('gap_at_best', 0):.2f}x")

    return summary


def extract_best_metrics(history: Dict) -> Dict:
    """Extract key metrics from a training history dict."""
    metrics = {}

    # Find best epoch by val_proto_acc
    val_proto = history.get("val_proto_acc", [])
    if not val_proto:
        return {"error": "No val_proto_acc in history"}

    best_idx = int(np.argmax(val_proto))
    metrics["best_val_proto_acc"] = val_proto[best_idx]
    metrics["best_epoch"] = best_idx + 1
    metrics["total_epochs"] = len(val_proto)

    # Top-5 and Top-10
    for key in ["val_proto_top5", "val_proto_top10"]:
        series = history.get(key, [])
        if series:
            metrics[f"best_{key}"] = series[best_idx]
            metrics[f"final_{key}"] = series[-1]

    # Train-val gap
    train_proto = history.get("train_proto_acc", [])
    if train_proto and train_proto[best_idx] > 0:
        metrics["train_proto_at_best"] = train_proto[best_idx]
        metrics["gap_at_best"] = train_proto[best_idx] / max(val_proto[best_idx], 1e-6)

    # Final metrics
    metrics["final_val_proto_acc"] = val_proto[-1]
    if train_proto:
        metrics["final_train_proto_acc"] = train_proto[-1]

    # Temperature
    temp = history.get("temperature", [])
    if temp:
        metrics["temp_first"] = temp[0]
        metrics["temp_last"] = temp[-1]
        metrics["temp_at_best"] = temp[best_idx]

    # Loss
    val_loss = history.get("val_loss", [])
    if val_loss:
        metrics["best_val_loss"] = val_loss[best_idx]

    return metrics


def generate_report(include_historical: bool = True) -> str:
    """Generate a comparison report from completed ablation experiments.

    Parameters
    ----------
    include_historical : bool
        Include historical results from the experiment registry.

    Returns
    -------
    report : str
        Formatted markdown report.
    """
    lines = []
    lines.append("# CLOP Ablation Study Results\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Collect results
    results = []

    # Historical versions (from experiment registry)
    if include_historical:
        registry_path = PROJECT_ROOT / "results" / "clop_experiment_registry.json"
        if registry_path.exists():
            with open(registry_path) as f:
                registry = json.load(f)
            for name, exp in registry.get("experiments", {}).items():
                m = exp.get("metrics", {})
                if m.get("best_val_proto_acc"):
                    results.append({
                        "name": f"[hist] {name}",
                        "val_proto_acc": m["best_val_proto_acc"],
                        "val_proto_top5": m.get("best_val_proto_top5", 0),
                        "val_proto_top10": m.get("best_val_proto_top10", 0),
                        "best_epoch": m.get("best_epoch", 0),
                        "gap": m.get("gap_at_best", m.get("gap_train_val_at_best", 0)),
                        "description": exp.get("note", ""),
                        "is_historical": True,
                    })

    # Ablation results
    if RESULTS_DIR.exists():
        for exp_dir in sorted(RESULTS_DIR.iterdir()):
            summary_path = exp_dir / "summary.json"
            if not summary_path.exists():
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            m = summary.get("metrics")
            if not m or "error" in m:
                results.append({
                    "name": summary["ablation_name"],
                    "val_proto_acc": 0,
                    "val_proto_top5": 0,
                    "val_proto_top10": 0,
                    "best_epoch": 0,
                    "gap": 0,
                    "description": f"FAILED: {m.get('error', 'unknown') if m else 'no metrics'}",
                    "is_historical": False,
                })
                continue

            results.append({
                "name": summary["ablation_name"],
                "val_proto_acc": m["best_val_proto_acc"],
                "val_proto_top5": m.get("best_val_proto_top5", 0),
                "val_proto_top10": m.get("best_val_proto_top10", 0),
                "best_epoch": m["best_epoch"],
                "gap": m.get("gap_at_best", 0),
                "description": summary["description"],
                "overrides": summary.get("overrides", {}),
                "elapsed": summary.get("elapsed_human", ""),
                "is_historical": False,
            })

    if not results:
        return "No results found. Run ablation experiments first.\n"

    # Sort by val_proto_acc descending
    results.sort(key=lambda x: x["val_proto_acc"], reverse=True)

    # Table
    lines.append("## Results Table\n")
    lines.append("| Rank | Experiment | Val Proto Acc | Top-5 | Top-10 | Best Epoch | Gap | Description |")
    lines.append("|------|-----------|--------------|-------|--------|------------|-----|-------------|")

    for i, r in enumerate(results, 1):
        marker = " **" if r["val_proto_acc"] == max(x["val_proto_acc"] for x in results) else ""
        end_marker = "**" if marker else ""
        lines.append(
            f"| {i} | {marker}{r['name']}{end_marker} | "
            f"{r['val_proto_acc']*100:.2f}% | "
            f"{r['val_proto_top5']*100:.1f}% | "
            f"{r['val_proto_top10']*100:.1f}% | "
            f"{r['best_epoch']} | "
            f"{r['gap']:.1f}x | "
            f"{r['description'][:50]} |"
        )

    # Ablation analysis
    ablation_results = [r for r in results if not r.get("is_historical")]
    baseline_result = next((r for r in ablation_results if r["name"] == "baseline"), None)

    if baseline_result and len(ablation_results) > 1:
        lines.append("\n## Ablation Impact Analysis\n")
        lines.append("| Ablation | Δ Val Proto Acc | Δ Top-10 | Impact |")
        lines.append("|----------|----------------|----------|--------|")

        for r in ablation_results:
            if r["name"] == "baseline":
                continue
            delta_acc = (r["val_proto_acc"] - baseline_result["val_proto_acc"]) * 100
            delta_top10 = (r["val_proto_top10"] - baseline_result["val_proto_top10"]) * 100
            impact = "positive" if delta_acc > 0.5 else "negative" if delta_acc < -0.5 else "neutral"
            lines.append(
                f"| {r['name']} | {delta_acc:+.2f}pp | {delta_top10:+.1f}pp | {impact} |"
            )

    # Key findings
    if ablation_results:
        lines.append("\n## Key Findings\n")
        best = max(ablation_results, key=lambda x: x["val_proto_acc"])
        worst = min(ablation_results, key=lambda x: x["val_proto_acc"])
        lines.append(f"- **Best configuration**: {best['name']} ({best['val_proto_acc']*100:.2f}%)")
        lines.append(f"- **Worst configuration**: {worst['name']} ({worst['val_proto_acc']*100:.2f}%)")
        if baseline_result:
            lines.append(f"- **Baseline (v8.2 full)**: {baseline_result['val_proto_acc']*100:.2f}%")

    report = "\n".join(lines)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="CLOP Ablation Study & Replication Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all ablations
    python scripts/run_ablation_study.py --base_config configs/clop_v8.2.yaml

    # Run specific ablation
    python scripts/run_ablation_study.py --base_config configs/clop_v8.2.yaml --ablation no_mixup

    # Quick ablation (fewer epochs for initial screening)
    python scripts/run_ablation_study.py --base_config configs/clop_v8.2.yaml --quick

    # Just generate comparison report
    python scripts/run_ablation_study.py --report

    # List available ablations
    python scripts/run_ablation_study.py --list
        """,
    )
    parser.add_argument("--base_config", type=str,
                        help="Base config YAML (typically v8.2)")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Run a specific ablation (by name)")
    parser.add_argument("--list", action="store_true",
                        help="List available ablations and exit")
    parser.add_argument("--report", action="store_true",
                        help="Generate comparison report and exit")
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick screening mode (30 epochs)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Save configs without running")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # List ablations
    if args.list:
        print("\nAvailable ablation experiments:")
        print(f"{'='*60}")
        for name, info in ABLATION_REGISTRY.items():
            done = "✓" if is_experiment_complete(name) else " "
            print(f"  [{done}] {name:25s} — {info['description']}")
            if info["overrides"]:
                for k, v in info["overrides"].items():
                    print(f"        {k}: {v}")
        print(f"\n  Total: {len(ABLATION_REGISTRY)} ablations")
        return

    # Generate report
    if args.report:
        report = generate_report(include_historical=True)
        print(report)

        # Also save to file
        report_path = RESULTS_DIR / "ablation_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")
        return

    # Validate base config
    if not args.base_config:
        parser.error("--base_config is required for running ablations")

    base_config = load_config(args.base_config)

    # Quick mode: reduce epochs
    if args.quick:
        base_config["num_epochs"] = 30
        base_config["early_stopping_patience"] = 10
        print(f"  [QUICK MODE] Reduced to {base_config['num_epochs']} epochs")

    # Determine which ablations to run
    if args.ablation:
        if args.ablation not in ABLATION_REGISTRY:
            print(f"Unknown ablation: {args.ablation}")
            print(f"Available: {', '.join(ABLATION_REGISTRY.keys())}")
            return
        ablations = {args.ablation: ABLATION_REGISTRY[args.ablation]}
    else:
        ablations = ABLATION_REGISTRY

    # Run ablations
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"  CLOP ABLATION STUDY")
    print(f"  Base config: {args.base_config}")
    print(f"  Ablations: {len(ablations)}")
    print(f"  Output: {RESULTS_DIR}")
    print(f"{'='*70}")

    for name, info in ablations.items():
        if args.resume and is_experiment_complete(name):
            print(f"\n  [SKIP] {name} — already completed")
            # Load existing summary
            with open(get_experiment_dir(name) / "summary.json") as f:
                all_summaries.append(json.load(f))
            continue

        summary = run_single_ablation(
            ablation_name=name,
            base_config=base_config,
            overrides=info["overrides"],
            description=info["description"],
            seed=args.seed,
            dry_run=args.dry_run,
        )
        if summary:
            all_summaries.append(summary)

    total_elapsed = time.time() - start_time

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  ABLATION STUDY COMPLETE")
    print(f"  Total time: {total_elapsed/60:.1f} min")
    print(f"  Experiments: {len(all_summaries)} completed")
    print(f"{'='*70}")

    if all_summaries and not args.dry_run:
        # Generate and save report
        report = generate_report(include_historical=True)
        report_path = RESULTS_DIR / "ablation_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\n  Report: {report_path}")

        # Save aggregated results
        agg_path = RESULTS_DIR / "all_summaries.json"
        with open(agg_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"  Summaries: {agg_path}")


if __name__ == "__main__":
    main()
