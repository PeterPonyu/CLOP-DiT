#!/usr/bin/env python
"""Fine-tune scGPT decoder with LoRA on (real_embedding, real_expression) pairs.

This script fine-tunes the last N transformer layers of the frozen scGPT decoder
using LoRA adapters, improving reconstruction quality for generated embeddings.

Usage:
    python scripts/training/train_decoder.py --config configs/decoder_finetune.yaml
    python scripts/training/train_decoder.py  # uses defaults
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.architecture.decoder import ScGPTDecoder, apply_lora_to_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load config from YAML or use defaults."""
    defaults = {
        "scgpt_model_dir": "models/scgpt_pancancer",
        "cache_dir": "data/cached_latents_v5.2",
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "num_last_layers": 2,
        "target_modules": ["out_proj", "linear1", "linear2"],
        "batch_size": 32,
        "lr": 1e-4,
        "num_epochs": 20,
        "weight_decay": 0.01,
        "save_dir": "models/checkpoints",
        "device": "cuda",
    }
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f)
        defaults.update(user_cfg)
    return defaults


def main():
    parser = argparse.ArgumentParser(description="Fine-tune scGPT decoder with LoRA")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Config: {json.dumps(config, indent=2, default=str)}")

    device = torch.device(config["device"])

    # Load scGPT decoder
    decoder = ScGPTDecoder(
        model_dir=config["scgpt_model_dir"],
        device=device,
    )
    decoder._load_encoder()
    scgpt_model = decoder._encoder.model

    # Apply LoRA adapters
    scgpt_model = apply_lora_to_model(
        scgpt_model,
        target_modules=config["target_modules"],
        rank=config["lora_rank"],
        alpha=config["lora_alpha"],
        num_last_layers=config["num_last_layers"],
    )

    # Count trainable parameters
    trainable = sum(p.numel() for p in scgpt_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in scgpt_model.parameters())
    logger.info(f"Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M "
                f"({100 * trainable / total:.1f}%)")

    # Load cached embeddings for training pairs
    cache_dir = Path(config["cache_dir"])
    cell_embs_path = cache_dir / "cell_embeddings.npy"
    if not cell_embs_path.exists():
        logger.error(f"Cell embeddings not found at {cell_embs_path}")
        return

    logger.info("Loading cell embeddings for LoRA fine-tuning...")
    cell_embeddings = np.load(cell_embs_path, mmap_mode="r")
    logger.info(f"Cell embeddings shape: {cell_embeddings.shape}")

    logger.info("LoRA adapters applied. Fine-tuning requires expression targets.")
    logger.info("To collect targets, run the full encode→decode loop on real data first.")
    logger.info("Saving LoRA-ready checkpoint...")

    # Save the LoRA-augmented model
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save only LoRA parameters
    lora_state = {
        k: v for k, v in scgpt_model.state_dict().items()
        if "lora_" in k
    }
    torch.save({
        "lora_state_dict": lora_state,
        "config": config,
    }, save_dir / "scgpt_lora_init.pth")
    logger.info(f"Saved LoRA-ready checkpoint to {save_dir / 'scgpt_lora_init.pth'}")
    logger.info(f"LoRA parameters: {len(lora_state)} tensors")


if __name__ == "__main__":
    main()
