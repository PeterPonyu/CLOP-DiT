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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.train_dit import DiTTrainer
from src.utils.helpers import seed_everything, get_device
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train DiT Flow Matching")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--projected_text", type=str, default=None,
                        help="Path to CLOP-projected text embeddings")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val_split", type=float, default=None,
                        help="Validation split fraction (used when --n_folds <= 1)")
    parser.add_argument("--n_folds", type=int, default=None,
                        help="Number of group-level CV folds (set >1 to enable K-fold)")
    parser.add_argument("--fold_idx", type=int, default=None,
                        help="Validation fold index (0-based) when --n_folds > 1")
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--cond_dim", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    setup_logging()
    seed_everything(args.seed)

    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "cache_dir": args.cache_dir or "data/cached_latents",
            "save_dir": args.save_dir or "models/checkpoints",
            "num_epochs": args.epochs or 200,
            "batch_size": args.batch_size or 512,
            "lr": args.lr or 1e-4,
            "latent_dim": args.latent_dim or 512,
            "hidden_dim": args.hidden_dim or 384,
            "num_blocks": args.num_blocks or 8,
            "num_heads": args.num_heads or 6,
            "cond_dim": args.cond_dim or 256,
            "device": args.device or "cuda",
        }

    # CLI overrides — only apply if explicitly set
    for key, arg_val in [
        ("cache_dir", args.cache_dir), ("save_dir", args.save_dir),
        ("num_epochs", args.epochs), ("batch_size", args.batch_size),
        ("lr", args.lr), ("latent_dim", args.latent_dim),
        ("hidden_dim", args.hidden_dim), ("num_blocks", args.num_blocks),
        ("num_heads", args.num_heads), ("cond_dim", args.cond_dim),
        ("device", args.device),
    ]:
        if arg_val is not None:
            config[key] = arg_val

    # Auto-detect projected text
    cache_dir = config.get("cache_dir", "data/cached_latents")
    if args.projected_text is not None:
        config["projected_text_path"] = args.projected_text
    elif "projected_text_path" not in config or not Path(config["projected_text_path"]).exists():
        default_path = Path(cache_dir) / "projected_text.npy"
        if default_path.exists():
            config["projected_text_path"] = str(default_path)
            print(f"Found CLOP-projected text at {config['projected_text_path']}")
        else:
            config["projected_text_path"] = None
            print("WARNING: No projected text found. Using raw text embeddings.")
            print("  Run 04a_train_clop.py first for optimal results.")

    if args.val_split is not None:
        config["val_split"] = args.val_split
    if args.n_folds is not None:
        config["n_folds"] = args.n_folds
    if args.fold_idx is not None:
        config["fold_idx"] = args.fold_idx

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
