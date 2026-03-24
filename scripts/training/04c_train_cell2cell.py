#!/usr/bin/env python3
# 04c_train_cell2cell.py — Train Cell2Cell Conditional Flow Matching
"""
Step 4c: Train the Cell2Cell model for cell-to-cell latent editing.

This model learns cross-condition cell transport in scGPT latent space,
analogous to Stable Diffusion img2img but for single-cell biology.

Usage:
    python scripts/04c_train_cell2cell.py --cache_dir data/cached_latents --epochs 200
    python scripts/04c_train_cell2cell.py --identity_weight 0.15 --lr 5e-5

Prerequisites:
    - Run 03_cache_builder.py first (creates data/cached_latents/)
    - Run 04a_train_clop.py first (creates projected_text.npy)
    - Optionally pre-train DiT with 04b_train_dit.py (for weight init)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.train_cell2cell import Cell2CellTrainer
from src.utils.helpers import seed_everything, get_device
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train Cell2Cell Flow Matching")
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
    parser.add_argument("--identity_weight", type=float, default=0.1,
                        help="Weight for identity preservation regularization")
    parser.add_argument("--src_drop_prob", type=float, default=0.1,
                        help="Source dropout probability for CFG training")
    parser.add_argument("--cond_drop_prob", type=float, default=0.1,
                        help="Condition dropout probability for CFG training")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_from_dit", type=str, default=None,
                        help="Initialize shared weights from pre-trained DiT checkpoint")
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

    # Auto-detect cond_dim from projected text
    if projected_text:
        cond_dim = np.load(projected_text).shape[1]
        print(f"Auto-detected cond_dim = {cond_dim} from projected text")
    else:
        text_path = Path(args.cache_dir) / "text_embeddings.npy"
        if text_path.exists():
            cond_dim = np.load(text_path).shape[1]
            print(f"Auto-detected cond_dim = {cond_dim} from raw text")
        else:
            cond_dim = 256  # fallback

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
            "cond_dim": cond_dim,
            "device": args.device,
            "identity_weight": args.identity_weight,
            "src_drop_prob": args.src_drop_prob,
            "cond_drop_prob": args.cond_drop_prob,
        }

    print("\n" + "=" * 60)
    print("Cell2Cell Training Configuration")
    print("=" * 60)
    print(json.dumps(config, indent=2))

    trainer = Cell2CellTrainer.from_config(config)

    # Optional: initialize shared weights from pre-trained DiT
    if args.init_from_dit:
        dit_ckpt = torch.load(args.init_from_dit, map_location="cpu", weights_only=False)
        dit_state = dit_ckpt.get("ema_state_dict", dit_ckpt.get("model_state_dict", {}))
        model_state = trainer.model.state_dict()

        transferred = 0
        for k, v in dit_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                transferred += 1

        trainer.model.load_state_dict(model_state)
        print(f"\nTransferred {transferred} weight tensors from DiT checkpoint")

    # Print model summary
    param_info = trainer.model.count_parameters()
    print(f"\nCell2Cell Model: {param_info['trainable_M']} trainable parameters")
    print(f"  Architecture: {config.get('num_blocks', 8)} blocks × "
          f"{config.get('hidden_dim', 384)} hidden × "
          f"{config.get('num_heads', 6)} heads")
    print(f"  Identity weight: {config.get('identity_weight', 0.1)}")
    print(f"  Source dropout: {config.get('src_drop_prob', 0.1)}")

    history = trainer.train()

    print("\n" + "=" * 60)
    print("Cell2Cell training complete!")
    print(f"Best val loss: {trainer.best_val_loss:.6f}")
    print("Next: Run 06_cell2cell_inference.py to edit cells!")
    print("=" * 60)


if __name__ == "__main__":
    main()
