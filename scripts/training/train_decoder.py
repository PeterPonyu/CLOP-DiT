#!/usr/bin/env python
"""Fine-tune scGPT decoder with LoRA on (real_embedding, real_expression) pairs.

This script fine-tunes the last N transformer layers of the frozen scGPT decoder
using LoRA adapters, with a marker-aware loss that upweights biologically
important genes to improve cell-type-specific marker reconstruction.

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
from torch.amp import GradScaler, autocast

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.architecture.decoder import ScGPTDecoder, apply_lora_to_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load config from YAML or use defaults."""
    defaults = {
        "scgpt_model_dir": "models/scgpt_human",
        "cache_dir": "data/cached_latents",
        "h5ad_dir": "data/processed_h5ad",
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "num_last_layers": 2,
        "target_modules": ["out_proj", "linear1", "linear2"],
        "batch_size": 32,
        "lr": 1e-4,
        "num_epochs": 20,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "marker_weight": 10.0,
        "marker_genes_config": "configs/marker_genes.yaml",
        "val_split": 0.1,
        "max_train_samples": 10000,
        "save_dir": "models/checkpoints",
        "device": "cuda",
    }
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f)
        defaults.update(user_cfg)
    return defaults


def load_marker_gene_set(config_path: str) -> set:
    """Load marker gene names from config for loss upweighting."""
    marker_genes = set()
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Marker genes config not found at {path}, using uniform loss")
        return marker_genes
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    panel = cfg.get("marker_panel_genes", {})
    for lineage, genes in panel.items():
        for g in genes:
            marker_genes.add(g.upper())
    logger.info(f"Loaded {len(marker_genes)} marker genes for upweighted loss")
    return marker_genes


def build_expression_targets(decoder, cell_embeddings: np.ndarray, config: dict):
    """Build (embedding, expression) training pairs using teacher-forced decode.

    We use the decoder's own frozen weights to generate target expression from
    real cell embeddings. The LoRA fine-tuning then learns to reconstruct these
    targets more faithfully, especially for marker genes.
    """
    logger.info("Building expression targets via scGPT decode...")
    result = decoder.decode(cell_embeddings, batch_size=config["batch_size"])
    expression = result["expression"]  # (N, G)
    gene_names = result["gene_names"]  # list of G strings
    logger.info(f"Expression targets: {expression.shape}, {len(gene_names)} genes")
    return expression, gene_names


def teacher_forced_forward(
    scgpt_model,
    cell_emb: torch.Tensor,
    src: torch.Tensor,
    values: torch.Tensor,
    src_key_padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Teacher-forced forward pass through scGPT with gradients enabled.

    Same pathway as TransformerModel.generate() but without @torch.no_grad(),
    so gradients flow through LoRA parameters.

    Returns (B, seq_len) predicted expression values.
    """
    # Encode gene tokens and values
    src_embs = scgpt_model.encoder(src)           # (B, seq_len, d_model)
    val_embs = scgpt_model.value_encoder(values)  # (B, seq_len, d_model)

    if scgpt_model.input_emb_style == "scaling":
        total_embs = src_embs * val_embs.unsqueeze(2)
    else:
        total_embs = src_embs + val_embs

    # Inject cell_emb at position 0 (replacing [CLS] token embedding)
    total_embs[:, 0, :] = cell_emb

    if getattr(scgpt_model, "bn", None) is not None:
        total_embs = scgpt_model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

    # Single forward pass through transformer encoder
    output = scgpt_model.transformer_encoder(
        total_embs, src_key_padding_mask=src_key_padding_mask
    )

    # ExprDecoder: per-gene scalar expression
    mlm_output = scgpt_model.decoder(output)  # {"pred": (B, seq_len)}
    return mlm_output["pred"]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune scGPT decoder with LoRA")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Config: {json.dumps(config, indent=2, default=str)}")

    device = torch.device(config["device"])

    # ── Load scGPT decoder ──
    decoder = ScGPTDecoder(
        model_dir=config["scgpt_model_dir"],
        device=device,
    )
    decoder._load_encoder()
    scgpt_model = decoder._encoder.model

    # ── Populate reference gene set from an h5ad file ──
    # The decoder needs gene_ids from encode() to know which genes to decode.
    # Load one h5ad file to cross-reference gene names against scGPT vocab.
    h5ad_dir = Path(config.get("h5ad_dir", "data/processed_h5ad"))
    h5ad_files = sorted(h5ad_dir.glob("*_processed.h5ad"))
    if not h5ad_files:
        logger.error(f"No h5ad files found in {h5ad_dir} for gene reference")
        return
    import anndata as ad
    ref_adata = ad.read_h5ad(h5ad_files[0])
    logger.info(f"Loading gene reference from {h5ad_files[0].name} ({ref_adata.n_vars} genes)")
    # Run encode on a tiny subset just to populate _ref_gene_ids
    tiny_adata = ref_adata[:2].copy()
    decoder._encoder.encode(tiny_adata)
    del ref_adata, tiny_adata
    logger.info(f"Reference gene set: {len(decoder._encoder._ref_gene_ids)} genes")

    # ── Load cached cell embeddings ──
    cache_dir = Path(config["cache_dir"])
    cell_embs_path = cache_dir / "cell_embeddings_dedup_preprocessed.npy"
    if not cell_embs_path.exists():
        cell_embs_path = cache_dir / "cell_embeddings.npy"
    if not cell_embs_path.exists():
        logger.error(f"Cell embeddings not found in {cache_dir}")
        return

    logger.info(f"Loading cell embeddings from {cell_embs_path}")
    cell_embeddings = np.load(cell_embs_path)
    logger.info(f"Cell embeddings shape: {cell_embeddings.shape}")

    # ── Subsample for speed ──
    max_samples = config.get("max_train_samples", 0)
    if max_samples and max_samples < cell_embeddings.shape[0]:
        rng = np.random.default_rng(42)
        keep_idx = rng.choice(cell_embeddings.shape[0], max_samples, replace=False)
        cell_embeddings = cell_embeddings[keep_idx]
        logger.info(f"Subsampled to {max_samples} cells for LoRA training")

    # ── Build expression targets (frozen decode of real embeddings) ──
    expression_targets, gene_names = build_expression_targets(
        decoder, cell_embeddings, config
    )

    # ── Build marker gene weight mask ──
    marker_genes = load_marker_gene_set(config["marker_genes_config"])
    gene_weights = torch.ones(len(gene_names), device=device)
    if marker_genes:
        n_marked = 0
        for i, gname in enumerate(gene_names):
            if gname.upper() in marker_genes:
                gene_weights[i] = config["marker_weight"]
                n_marked += 1
        logger.info(
            f"Marker-weighted loss: {n_marked}/{len(gene_names)} genes at "
            f"{config['marker_weight']}x weight"
        )

    # ── Apply LoRA adapters ──
    scgpt_model = apply_lora_to_model(
        scgpt_model,
        target_modules=config["target_modules"],
        rank=config["lora_rank"],
        alpha=config["lora_alpha"],
        num_last_layers=config["num_last_layers"],
    )
    # Move LoRA params to device (they're created on CPU by default)
    scgpt_model = scgpt_model.to(device)

    # Count trainable parameters
    trainable = sum(p.numel() for p in scgpt_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in scgpt_model.parameters())
    logger.info(f"Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M "
                f"({100 * trainable / total:.1f}%)")

    # ── Prepare fixed gene token sequence ──
    encoder = decoder._encoder
    if encoder._ref_gene_ids is None:
        logger.error("No reference gene IDs available. Run encode() on data first.")
        return

    gene_ids = encoder._ref_gene_ids
    cls_token_id = encoder.vocab["<cls>"]
    pad_token_id = encoder.vocab["<pad>"]
    pad_value = encoder.model_configs.get("pad_value", -2)

    fixed_genes = np.insert(gene_ids, 0, cls_token_id)
    fixed_values = np.full(len(fixed_genes), 0.0, dtype=np.float32)
    fixed_values[0] = pad_value

    src_fixed = torch.from_numpy(fixed_genes).long().to(device)
    values_fixed = torch.from_numpy(fixed_values).float().to(device)
    mask_fixed = src_fixed.eq(pad_token_id)

    # ── Train/val split ──
    N = cell_embeddings.shape[0]
    indices = np.random.default_rng(42).permutation(N)
    n_val = int(N * config["val_split"])
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_emb = torch.from_numpy(cell_embeddings[train_idx]).float()
    train_expr = torch.from_numpy(expression_targets[train_idx]).float()
    val_emb = torch.from_numpy(cell_embeddings[val_idx]).float()
    val_expr = torch.from_numpy(expression_targets[val_idx]).float()

    train_dataset = TensorDataset(train_emb, train_expr)
    val_dataset = TensorDataset(val_emb, val_expr)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ── Optimizer (only LoRA params) ──
    lora_params = [p for p in scgpt_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scaler = GradScaler("cuda")

    # ── Training loop ──
    best_val_loss = float("inf")
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    seq_len = len(fixed_genes)

    for epoch in range(config["num_epochs"]):
        scgpt_model.train()
        train_loss_sum = 0
        n_batches = 0
        n_total_batches = len(train_loader)

        for batch_emb, batch_expr in train_loader:
            batch_emb = batch_emb.to(device)
            batch_expr = batch_expr.to(device)
            B = batch_emb.shape[0]

            # Expand fixed gene sequence to batch
            src_batch = src_fixed.unsqueeze(0).expand(B, -1)
            val_batch = values_fixed.unsqueeze(0).expand(B, -1)
            mask_batch = mask_fixed.unsqueeze(0).expand(B, -1)

            optimizer.zero_grad()

            with autocast("cuda"):
                pred = teacher_forced_forward(
                    scgpt_model, batch_emb, src_batch, val_batch, mask_batch
                )
                # pred: (B, seq_len), skip position 0 (cls)
                pred_expr = pred[:, 1:]  # (B, G)

                # Marker-weighted MSE loss
                diff_sq = (pred_expr - batch_expr) ** 2  # (B, G)
                weighted_loss = (diff_sq * gene_weights.unsqueeze(0)).mean()

            scaler.scale(weighted_loss).backward()
            if config["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(lora_params, config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += weighted_loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                logger.info(
                    f"  [Epoch {epoch+1}][Step {n_batches}/{n_total_batches}] "
                    f"loss={weighted_loss.item():.6f}"
                )

        avg_train = train_loss_sum / max(n_batches, 1)

        # ── Validation ──
        scgpt_model.eval()
        val_loss_sum = 0
        val_batches = 0

        with torch.no_grad():
            for batch_emb, batch_expr in val_loader:
                batch_emb = batch_emb.to(device)
                batch_expr = batch_expr.to(device)
                B = batch_emb.shape[0]

                src_batch = src_fixed.unsqueeze(0).expand(B, -1)
                val_batch = values_fixed.unsqueeze(0).expand(B, -1)
                mask_batch = mask_fixed.unsqueeze(0).expand(B, -1)

                with autocast("cuda"):
                    pred = teacher_forced_forward(
                        scgpt_model, batch_emb, src_batch, val_batch, mask_batch
                    )
                    pred_expr = pred[:, 1:]
                    diff_sq = (pred_expr - batch_expr) ** 2
                    loss = (diff_sq * gene_weights.unsqueeze(0)).mean()

                val_loss_sum += loss.item()
                val_batches += 1

        avg_val = val_loss_sum / max(val_batches, 1)

        logger.info(
            f"Epoch {epoch+1}/{config['num_epochs']} | "
            f"Train loss: {avg_train:.6f} | Val loss: {avg_val:.6f}"
        )

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            lora_state = {
                k: v for k, v in scgpt_model.state_dict().items()
                if "lora_" in k
            }
            torch.save({
                "lora_state_dict": lora_state,
                "config": config,
                "epoch": epoch + 1,
                "val_loss": avg_val,
                "gene_names": gene_names,
            }, save_dir / "scgpt_lora_best.pth")
            logger.info(f"  → Saved best model (val_loss={avg_val:.6f})")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
    logger.info(f"Best checkpoint: {save_dir / 'scgpt_lora_best.pth'}")


if __name__ == "__main__":
    main()
