#!/usr/bin/env python3
"""Train CLOP and compare with historical results.

This script:
1. Trains CLOP with a provided config
2. Compares training dynamics with v6.3, v6.4, v6.4.1
3. Generates a comparison report

Usage:
    python scripts/09_train_and_compare_v7.py [--epochs N] [--dry-run]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_clop import CLOPTrainer
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.logging_config import setup_logging


def load_historical_results():
    """Load experiment registry and historical training histories."""
    registry_path = PROJECT_ROOT / "results" / "clop_experiment_registry.json"
    history = {}

    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
        for vid, data in registry.get("experiments", {}).items():
            history[vid] = data.get("metrics", {})

    # Load full histories for available versions
    archive_base = PROJECT_ROOT / "models" / "checkpoints" / "archive"
    for version_dir in ["v63", "v641"]:
        hist_path = archive_base / version_dir / "clop_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                h = json.load(f)
            vid = version_dir.replace("v63", "v6.3").replace("v641", "v6.4.1")
            history[vid]["full_history"] = h

    return history


def compare_training_dynamics(run_history, historical, run_label: str):
    """Generate comparison report between current run and historical versions."""
    report = []
    report.append("=" * 70)
    report.append("CLOP Version Comparison Report")
    report.append("=" * 70)
    report.append("")

    # Current run metrics
    vpa_v7 = run_history.get("val_proto_acc", [])
    tpa_v7 = run_history.get("train_proto_acc", [])
    temps_v7 = run_history.get("temperature", [])
    vpt5_v7 = run_history.get("val_proto_top5", [])
    vpt10_v7 = run_history.get("val_proto_top10", [])

    if vpa_v7:
        best_epoch_v7 = int(np.argmax(vpa_v7)) + 1
        best_vpa_v7 = max(vpa_v7)
        best_tpa_v7 = tpa_v7[best_epoch_v7 - 1] if tpa_v7 else 0
        gap_v7 = best_tpa_v7 / max(best_vpa_v7, 0.001)

        report.append(f"{'Version':<12} {'Best VPA':>10} {'Best Epoch':>12} {'Train PA':>10} {'Gap':>8} {'Temp(final)':>12}")
        report.append("-" * 70)

        # Current run
        temp_final_v7 = temps_v7[-1] if temps_v7 else 0
        report.append(f"{run_label:<12} {best_vpa_v7:>10.4f} {best_epoch_v7:>12d} {best_tpa_v7:>10.4f} {gap_v7:>7.1f}x {temp_final_v7:>12.2f}")

        # Historical versions
        for vid in ["v6.3", "v6.4", "v6.4.1"]:
            if vid in historical:
                m = historical[vid]
                bvpa = m.get("best_val_proto_acc", 0)
                bepoch = m.get("best_epoch", m.get("epochs_seen", "?"))
                btpa = m.get("best_train_proto_acc", m.get("train_proto_series_tail", [0])[-1])
                gap = m.get("gap_train_val_at_best", btpa / max(bvpa, 0.001) if bvpa > 0 else 0)
                temp = m.get("temp_last", 0)
                report.append(f"{vid:<12} {bvpa:>10.4f} {str(bepoch):>12s} {btpa:>10.4f} {gap:>7.1f}x {temp:>12.2f}")

        report.append("")

        # Key improvements analysis
        report.append("Key Analysis:")
        report.append("-" * 70)

        # Temperature behavior
        if temps_v7:
            temp_max = max(temps_v7)
            temp_at_epoch_20 = temps_v7[min(19, len(temps_v7)-1)]
            temp_saturated = temp_max >= 14.9
            report.append(f"  Temperature: max={temp_max:.2f}, at epoch 20={temp_at_epoch_20:.2f}, saturated={'YES' if temp_saturated else 'NO'}")
            if not temp_saturated:
                report.append(f"  ✓ Temperature regularization prevented saturation!")
            else:
                report.append(f"  ⚠ Temperature still saturated — consider stronger temp_reg_weight")

        # Overfitting gap
        if gap_v7 < 3.0:
            report.append(f"  ✓ Train/val gap ({gap_v7:.1f}x) improved below 3.0x target!")
        else:
            report.append(f"  ⚠ Train/val gap ({gap_v7:.1f}x) still above 3.0x target")

        # Compare with v6.4.1
        v641_vpa = historical.get("v6.4.1", {}).get("best_val_proto_acc", 0)
        if v641_vpa > 0:
            improvement = (best_vpa_v7 - v641_vpa) / v641_vpa * 100
            report.append(f"  Val proto acc vs v6.4.1: {improvement:+.1f}% ({'improved' if improvement > 0 else 'regressed'})")

        # Top-k metrics
        if vpt5_v7:
            report.append(f"  Best val_proto_top5: {max(vpt5_v7):.4f}")
        if vpt10_v7:
            report.append(f"  Best val_proto_top10: {max(vpt10_v7):.4f}")

        # Training curve snapshot
        report.append("")
        report.append(f"{run_label} Training Trajectory:")
        report.append(f"  {'Epoch':>6} {'VPA':>8} {'TPA':>8} {'Gap':>6} {'Temp':>6} {'Top5':>8} {'Top10':>8}")
        for epoch in [1, 5, 10, 20, 30, 50, 70, 100, 150, 200]:
            if epoch <= len(vpa_v7):
                vpa = vpa_v7[epoch - 1]
                tpa = tpa_v7[epoch - 1] if epoch <= len(tpa_v7) else 0
                temp = temps_v7[epoch - 1] if epoch <= len(temps_v7) else 0
                g = tpa / max(vpa, 0.001)
                t5 = vpt5_v7[epoch - 1] if epoch <= len(vpt5_v7) else 0
                t10 = vpt10_v7[epoch - 1] if epoch <= len(vpt10_v7) else 0
                report.append(f"  {epoch:>6d} {vpa:>8.4f} {tpa:>8.4f} {g:>5.1f}x {temp:>6.2f} {t5:>8.4f} {t10:>8.4f}")

    report.append("")
    report.append("=" * 70)
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Train CLOP and compare with history")
    parser.add_argument("--config", default="configs/clop_v7.yaml", help="Config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--dry-run", action="store_true", help="Test with 5 epochs only")
    args = parser.parse_args()

    # Setup logging
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{Path(args.config).stem}_train.log"
    setup_logging(log_path)
    logger = logging.getLogger(__name__)

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["num_epochs"] = args.epochs
    if args.dry_run:
        config["num_epochs"] = 5
        config["early_stopping_patience"] = 0
        logger.info("DRY RUN: 5 epochs only")

    # Setup experiment tracker
    tracker = ExperimentTracker(base_dir="results")
    version_id = config.get("version_id", f"{Path(args.config).stem}_{int(time.time())}")
    description = config.get("version_description", f"CLOP training run ({Path(args.config).stem})")
    changes = config.get("version_changes", "Config-driven CLOP training run")

    try:
        version_dir = tracker.create_version(
            version_id=version_id,
            description=description,
            config=config,
            changes=changes,
        )
        logger.info(f"Created experiment version: {version_id} at {version_dir}")

        # Update save_dir to version-specific directory
        config["save_dir"] = str(tracker.get_checkpoint_dir(version_id))
    except ValueError as e:
        logger.warning(f"Version exists: {e}. Using timestamped ID.")
        version_id = f"{Path(args.config).stem}_{int(time.time())}"
        version_dir = tracker.create_version(
            version_id=version_id,
            description=description,
            config=config,
        )
        config["save_dir"] = str(tracker.get_checkpoint_dir(version_id))

    tracker.update_status(version_id, "training")

    # Train
    logger.info("=" * 60)
    logger.info(f"Starting CLOP training: {version_id}")
    logger.info(f"Config: temp_reg_weight={config.get('temp_reg_weight', 0)}, "
                f"max_temperature={config.get('max_temperature', 20)}, "
                f"preprocess_cell_method={config.get('preprocess_cell_method', 'whiten')}")
    logger.info("=" * 60)

    trainer = CLOPTrainer.from_config(config)
    history = trainer.train()

    # Save history to version metrics
    tracker.save_metrics(version_id, "clop_history.json", history)

    # Compute summary
    vpa = history.get("val_proto_acc", [])
    tpa = history.get("train_proto_acc", [])
    summary = {}
    if vpa:
        best_idx = int(np.argmax(vpa))
        summary = {
            "epochs": len(vpa),
            "best_val_proto_acc": float(max(vpa)),
            "best_epoch": best_idx + 1,
            "final_val_proto_acc": float(vpa[-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "best_train_proto_acc": float(tpa[best_idx]) if tpa else 0,
            "gap_at_best": float(tpa[best_idx] / max(vpa[best_idx], 0.001)) if tpa else 0,
            "temp_first": float(history["temperature"][0]) if history.get("temperature") else 0,
            "temp_last": float(history["temperature"][-1]) if history.get("temperature") else 0,
            "best_val_proto_top5": float(max(history.get("val_proto_top5", [0]))),
            "best_val_proto_top10": float(max(history.get("val_proto_top10", [0]))),
        }

    tracker.save_metrics(version_id, "training_summary.json", summary)
    tracker.update_status(version_id, "completed", summary=summary)

    logger.info(f"Training complete. Summary: {json.dumps(summary, indent=2)}")

    # Load historical and compare
    historical = load_historical_results()
    report = compare_training_dynamics(history, historical, version_id)

    # Save report
    report_path = tracker.get_metrics_dir(version_id) / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Also print to console
    print("\n" + report)

    # Update experiment registry with this run results
    registry_path = PROJECT_ROOT / "results" / "clop_experiment_registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"experiments": {}}

    registry["experiments"][version_id] = {
        "history": str(tracker.get_metrics_dir(version_id) / "clop_history.json"),
        "best_ckpt": str(tracker.get_checkpoint_dir(version_id) / "clop_best.pth"),
        "log": log_path,
        "metrics": summary,
        "config": args.config,
    }
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info(f"Updated experiment registry with {version_id} results")


if __name__ == "__main__":
    main()
