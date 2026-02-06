#!/usr/bin/env python3
# 04a_train_clop.py — Train CLOP contrastive alignment
"""
Step 4a: Train CLOP alignment between text and cell embeddings.

Usage:
    python scripts/04a_train_clop.py --config configs/clop.yaml
    python scripts/04a_train_clop.py --cache_dir data/cached_latents --epochs 100
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train_clop import CLOPTrainer
from src.utils.helpers import seed_everything, get_device
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train CLOP Alignment")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--cache_dir", type=str, default="data/cached_latents")
    parser.add_argument("--save_dir", type=str, default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
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
            "cache_dir": args.cache_dir,
            "save_dir": args.save_dir,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "proj_dim": args.proj_dim,
            "device": args.device,
        }

    print("CLOP Training Configuration:")
    print(json.dumps(config, indent=2))

    trainer = CLOPTrainer.from_config(config)
    history = trainer.train()

    # Project and save text embeddings for DiT training
    trainer.project_and_save(
        output_path=str(Path(args.cache_dir) / "projected_text.npy")
    )

    print("\nCLOP training complete!")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print("Next: Run 04b_train_dit.py for flow matching training.")


if __name__ == "__main__":
    main()
