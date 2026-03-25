#!/usr/bin/env python3
"""train_mlp_decoder.py — Train a direct MLP decoder: embedding → gene expression.

Bypasses scGPT generate() by learning a direct mapping from cell embeddings
to gene expression. Trained on real (embedding, expression) pairs from
the cached h5ad data.

Usage:
    python scripts/training/train_mlp_decoder.py
    python scripts/training/train_mlp_decoder.py --hidden-dims 1024 2048 --epochs 50
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.architecture.decoder import LinearDecoder
from src.utils.paths import (
    CACHE_DIR, CHECKPOINT_DIR, PROCESSED_H5AD_DIR, SCGPT_DIR,
)
from src.utils.helpers import seed_everything, get_device

logger = logging.getLogger(__name__)


def load_training_data(cache_dir, h5ad_dir, max_cells=20000, seed=42):
    """Load matched (embedding, expression) pairs for training.

    Uses the preprocessed embeddings and decodes the corresponding
    real expression from h5ad files.
    """
    cache = Path(cache_dir)

    # Load preprocessed embeddings and labels
    embeddings = np.load(cache / "cell_embeddings_dedup_preprocessed.npy")
    labels = np.load(cache / "text_group_ids_dedup.npy")

    # Load the embedding preprocessor for inverse transform
    from src.data_pipeline.embedding_preprocessor import EmbeddingPreprocessor
    preprocessor = EmbeddingPreprocessor.load(
        str(cache / "cell_preprocessor_preprocessed.npz")
    )

    # Compute pre-norm scale
    raw_emb = np.load(cache / "cell_embeddings_dedup.npy")
    from scripts.analysis.decode_expression import compute_pre_norm_scale
    pre_norm_scale = compute_pre_norm_scale(preprocessor, raw_emb, n_samples=5000)
    del raw_emb

    # Get real expression by decoding through scGPT (the "ground truth")
    # Subsample first to keep training tractable
    rng = np.random.default_rng(seed)
    if max_cells < len(embeddings):
        idx = rng.choice(len(embeddings), max_cells, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    logger.info(f"Training data: {len(embeddings)} cells")

    # Inverse transform to raw scGPT space for decoding
    raw_space = preprocessor.inverse_transform(embeddings, target_norm=pre_norm_scale)

    # Decode via scGPT to get "ground truth" expression
    from src.architecture.decoder import ScGPTDecoder
    device = get_device()
    scgpt = ScGPTDecoder(model_dir=str(SCGPT_DIR), device=device, batch_size=64)
    scgpt._load_encoder()

    # Get gene reference
    import anndata as ad
    h5ad_files = sorted(Path(h5ad_dir).glob("*_processed.h5ad"))
    ref = ad.read_h5ad(h5ad_files[0])
    scgpt._encoder.encode(ref[:2].copy())
    gene_ids = scgpt._encoder._ref_gene_ids
    gene_names = scgpt._encoder._ref_gene_names
    del ref

    result = scgpt.decode(raw_space, gene_ids=gene_ids, gene_names=gene_names)
    expression = result["expression"]
    del scgpt, raw_space

    logger.info(f"Expression shape: {expression.shape}, mean: {expression.mean():.2f}")

    return embeddings, expression, labels, gene_names


def train_mlp(embeddings, expression, labels, gene_names,
              hidden_dims=(1024, 2048), epochs=50, batch_size=256,
              lr=1e-3, val_frac=0.1, seed=42, device=None):
    """Train a LinearDecoder on (embedding → expression) pairs."""
    if device is None:
        device = get_device()

    seed_everything(seed)
    n = len(embeddings)
    n_val = int(n * val_frac)

    # Stratified split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train = torch.from_numpy(embeddings[train_idx]).float()
    Y_train = torch.from_numpy(expression[train_idx]).float()
    X_val = torch.from_numpy(embeddings[val_idx]).float()
    Y_val = torch.from_numpy(expression[val_idx]).float()

    train_ds = TensorDataset(X_train, Y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=True, num_workers=0)

    num_genes = expression.shape[1]
    model = LinearDecoder(
        embed_dim=512, num_genes=num_genes,
        hidden_dims=list(hidden_dims), output_activation="none",
        dropout=0.1,
    ).to(device)

    logger.info(f"MLP decoder: {sum(p.numel() for p in model.parameters()):,} params")
    logger.info(f"  Architecture: 512 → {' → '.join(map(str, hidden_dims))} → {num_genes}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, Y_val.to(device)).item()

        train_loss = np.mean(train_losses)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}: "
                        f"train={train_loss:.6f} val={val_loss:.6f} "
                        f"best={best_val_loss:.6f}")

    # Save checkpoint
    model.load_state_dict(best_state)
    ckpt_path = CHECKPOINT_DIR / "mlp_decoder_best.pth"
    torch.save({
        "model_state_dict": best_state,
        "config": {
            "hidden_dims": list(hidden_dims),
            "num_genes": num_genes,
            "embed_dim": 512,
            "output_activation": "none",
        },
        "gene_names": gene_names,
        "val_loss": best_val_loss,
        "history": history,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }, ckpt_path)

    logger.info(f"Saved MLP decoder → {ckpt_path} (val_loss={best_val_loss:.6f})")
    return model, history


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train MLP decoder")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--h5ad-dir", default=str(PROCESSED_H5AD_DIR))
    parser.add_argument("--max-cells", type=int, default=20000)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[1024, 2048])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()

    embeddings, expression, labels, gene_names = load_training_data(
        args.cache_dir, args.h5ad_dir, args.max_cells, args.seed,
    )

    model, history = train_mlp(
        embeddings, expression, labels, gene_names,
        hidden_dims=tuple(args.hidden_dims),
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed,
    )

    elapsed = time.time() - t0
    logger.info(f"Total training time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
