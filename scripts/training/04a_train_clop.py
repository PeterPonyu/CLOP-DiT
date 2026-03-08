#!/usr/bin/env python3
# 04a_train_clop.py — Train CLOP contrastive alignment
"""
Step 4a: Train CLOP alignment between text and cell embeddings.

Usage:
    python scripts/04a_train_clop.py --config configs/clop.yaml
    python scripts/04a_train_clop.py --cache_dir data/cached_latents_v5.2 --epochs 100
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.train_clop import CLOPTrainer
from src.utils.helpers import seed_everything, get_device
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train CLOP Alignment")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--proj_dim", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=None,
                        help="Validation split fraction (used when --n_folds <= 1)")
    parser.add_argument("--n_folds", type=int, default=None,
                        help="Number of group-level CV folds (set >1 to enable K-fold)")
    parser.add_argument("--fold_idx", type=int, default=None,
                        help="Validation fold index (0-based) when --n_folds > 1")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    seed_everything(args.seed)

    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "cache_dir": args.cache_dir or "data/cached_latents_v5.2",
            "save_dir": args.save_dir or "models/checkpoints",
            "num_epochs": args.epochs or 200,
            "batch_size": args.batch_size or 256,
            "lr": args.lr or 3e-4,
            "proj_dim": args.proj_dim or 256,
            "device": args.device or "cuda",
        }

    # CLI overrides — only apply if explicitly set (not None)
    if args.cache_dir is not None:
        config["cache_dir"] = args.cache_dir
    if args.save_dir is not None:
        config["save_dir"] = args.save_dir
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr
    if args.proj_dim is not None:
        config["proj_dim"] = args.proj_dim
    if args.device is not None:
        config["device"] = args.device

    if args.val_split is not None:
        config["val_split"] = args.val_split
    if args.n_folds is not None:
        config["n_folds"] = args.n_folds
    if args.fold_idx is not None:
        config["fold_idx"] = args.fold_idx

    print("CLOP Training Configuration:")
    print(json.dumps(config, indent=2))

    trainer = CLOPTrainer.from_config(config)
    history = trainer.train()

    # Project and save text embeddings for DiT training
    cache_dir = config.get("cache_dir", "data/cached_latents_v5.2")
    trainer.project_and_save(
        output_path=str(Path(cache_dir) / "projected_text.npy")
    )

    print("\nCLOP training complete!")
    print(f"Best val acc: {trainer.best_val_acc:.4f}")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print("Next: Run 04b_train_dit.py for flow matching training.")


if __name__ == "__main__":
    main()
