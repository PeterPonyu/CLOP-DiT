#!/usr/bin/env python3
"""ZCA ablation — train CLOP with a given config (no text projection step).

Unlike 04a_train_clop.py, this wrapper ONLY trains and saves checkpoints.
It does NOT project text embeddings, so it won't overwrite production
projected_text.npy in the shared cache directory.

Usage:
    python revision/experiments/zca_ablation/train_clop_ablation.py \
        --config revision/experiments/zca_ablation/configs/clop_whiten.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.training.train_clop import CLOPTrainer
from src.utils.helpers import seed_everything
from src.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="CLOP ablation training (no projection)")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    setup_logging()
    seed_everything(config.get("seed", 42))

    print("ZCA Ablation — CLOP Training")
    print(f"Config: {args.config}")
    print(f"Preprocessing: text={config.get('preprocess_text_method')}, "
          f"cell={config.get('preprocess_cell_method')}")
    print(f"use_preprocessed: {config.get('use_preprocessed')}")
    print(f"Save dir: {config.get('save_dir')}")
    print()

    trainer = CLOPTrainer.from_config(config)
    history = trainer.train()

    print(f"\nTraining complete!")
    print(f"Best val acc: {trainer.best_val_acc:.4f}")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.get('save_dir')}")


if __name__ == "__main__":
    main()
