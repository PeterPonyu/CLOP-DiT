#!/usr/bin/env python3
# 04b_train_dit.py — Train DiT Flow Matching model
"""
Step 4b: Train the DiT generative model with Flow Matching.

Usage:
    python scripts/04b_train_dit.py --config configs/dit.yaml
    python scripts/04b_train_dit.py --cache_dir data/cached_latents --epochs 200
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train_dit import DiTTrainer
from src.utils.helpers import seed_everything, get_device
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train DiT Flow Matching")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--cache_dir", type=str, default="data/cached_latents")
    parser.add_argument("--save_dir", type=str, default="models/checkpoints")
    parser.add_argument("--projected_text", type=str, default=None,
                        help="Path to CLOP-projected text embeddings")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--cond_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    setup_logging()
    seed_everything(args.seed)

    # Auto-detect projected text
    projected_text = args.projected_text
    if projected_text is None:
        default_path = Path(args.cache_dir) / "projected_text.npy"
        if default_path.exists():
            projected_text = str(default_path)
            print(f"Found CLOP-projected text at {projected_text}")
        else:
            print("WARNING: No projected text found. Using raw text embeddings.")
            print("  Run 04a_train_clop.py first for optimal results.")

    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "cache_dir": args.cache_dir,
            "save_dir": args.save_dir,
            "projected_text_path": projected_text,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "num_blocks": args.num_blocks,
            "num_heads": args.num_heads,
            "cond_dim": args.cond_dim,
            "device": args.device,
        }

    print("DiT Training Configuration:")
    print(json.dumps(config, indent=2))

    # CLI overrides for speed optimization
    if args.resume:
        config['resume'] = args.resume

    trainer = DiTTrainer.from_config(config)

    # Print model summary
    param_info = trainer.model.count_parameters()
    print(f"\nDiT Model: {param_info['trainable_M']} trainable parameters")

    history = trainer.train()

    print("\nDiT training complete!")
    print(f"Best val loss: {trainer.best_val_loss:.6f}")
    print("Next: Run 05_inference.py to generate cells!")


if __name__ == "__main__":
    main()
