#!/usr/bin/env python3
"""Evaluate v6.3 checkpoint — measure train AND val proto metrics side-by-side."""

import sys, json, torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.architecture.clop import CLOPAligner
from src.data_pipeline.dataset import create_dataloaders

CKPT = Path("models/checkpoints/clop_best.pth")
CONFIG = Path("configs/clop.yaml")

# Load config
import yaml
with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

# Create data loaders
train_loader, val_loader = create_dataloaders(
    cache_dir=cfg["cache_dir"],
    batch_size=cfg["batch_size"],
    val_split=cfg["val_split"],
    num_workers=0,
    use_preprocessed=cfg.get("use_preprocessed", False),
    variant_prob=0.0,
)

# Load model from checkpoint
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = CLOPAligner(
    text_dim=ckpt["config"]["text_dim"],
    cell_dim=ckpt["config"]["cell_dim"],
    proj_dim=ckpt["config"]["proj_dim"],
    text_layers=ckpt["config"]["text_layers"],
    cell_layers=ckpt["config"]["cell_layers"],
    dropout=ckpt["config"]["dropout"],
    loss_type=ckpt["config"]["loss_type"],
    temperature=ckpt["config"].get("temperature", 14.0),
    max_temperature=ckpt["config"].get("max_temperature_siglip", ckpt["config"].get("max_temperature", 20.0)),
    cohesion_weight=ckpt["config"].get("cohesion_weight", 0.1),
    use_batch_norm=ckpt["config"].get("use_batch_norm", False),
    cell_noise_std=0.0,  # no noise for eval
)
model.load_state_dict(ckpt["model_state_dict"])
model = model.cuda().eval()

print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_acc={ckpt['metrics'].get('val_acc', '?')}")
print(f"Temperature: {model.criterion.temperature.item():.4f}")
print(f"Bias: {model.criterion.bias.item():.4f}")
print()


def evaluate_split(loader, split_name, max_batches=None):
    """Compute detailed metrics on a data split."""
    total_loss = 0
    total_acc_t2c = 0
    total_acc_c2t = 0
    total_proto_acc_t2c = 0
    total_proto_acc_c2t = 0
    total_n_groups = 0
    total_cohesion = 0
    n_batches = 0
    
    # Top-k accuracy
    total_top5_t2c = 0
    total_top5_c2t = 0
    total_top10_t2c = 0
    total_top10_c2t = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            text_emb, cell_emb, sample_ids = batch
            text_emb = text_emb.cuda()
            cell_emb = cell_emb.cuda()

            loss, metrics = model(text_emb, cell_emb)

            total_loss += loss.item()
            total_acc_t2c += metrics["acc_t2c"]
            total_acc_c2t += metrics["acc_c2t"]
            if "proto_acc_t2c" in metrics:
                total_proto_acc_t2c += metrics["proto_acc_t2c"]
                total_proto_acc_c2t += metrics["proto_acc_c2t"]
                total_n_groups += metrics["n_groups"]
                total_cohesion += metrics.get("cohesion_loss", 0)
            n_batches += 1

            # Compute top-k proto accuracy manually
            # We need to recompute prototypes — pull from the loss internals
            # Actually let's compute from the model forward
            text_proj = model.text_projector(model._whiten_text(text_emb))
            cell_proj = model.cell_projector(cell_emb)
            import torch.nn.functional as F
            text_proj = F.normalize(text_proj, dim=-1)
            cell_proj = F.normalize(cell_proj, dim=-1)
            
            group_ids = model._compute_group_ids(text_emb)
            unique_gids, inverse_ids = group_ids.unique(return_inverse=True)
            ng = unique_gids.shape[0]
            
            prototypes = torch.zeros(ng, cell_proj.shape[-1], device=cell_proj.device)
            prototypes.scatter_add_(0, inverse_ids.unsqueeze(-1).expand_as(cell_proj), cell_proj)
            counts = torch.zeros(ng, 1, device=cell_proj.device)
            counts.scatter_add_(0, inverse_ids.unsqueeze(-1), torch.ones(cell_proj.shape[0], 1, device=cell_proj.device))
            prototypes = prototypes / counts.clamp(min=1)
            prototypes = F.normalize(prototypes, dim=-1)
            
            first = torch.zeros(ng, dtype=torch.long, device=cell_proj.device)
            seen = torch.zeros(ng, dtype=torch.bool, device=cell_proj.device)
            for j in range(cell_proj.shape[0]):
                g = inverse_ids[j].item()
                if not seen[g]:
                    first[g] = j
                    seen[g] = True
            text_reps = F.normalize(text_proj[first], dim=-1)
            
            logits = text_reps @ prototypes.T
            targets = torch.arange(ng, device=logits.device)
            
            # Top-k
            for k in [5, 10]:
                if ng >= k:
                    topk_t2c = logits.topk(k, dim=-1).indices
                    topk_c2t = logits.T.topk(k, dim=-1).indices
                    hit_t2c = (topk_t2c == targets.unsqueeze(-1)).any(dim=-1).float().mean()
                    hit_c2t = (topk_c2t == targets.unsqueeze(-1)).any(dim=-1).float().mean()
                    if k == 5:
                        total_top5_t2c += hit_t2c.item()
                        total_top5_c2t += hit_c2t.item()
                    else:
                        total_top10_t2c += hit_t2c.item()
                        total_top10_c2t += hit_c2t.item()

    print(f"\n{'='*50}")
    print(f"  {split_name} Evaluation ({n_batches} batches)")
    print(f"{'='*50}")
    print(f"  Loss:          {total_loss / n_batches:.4f}")
    print(f"  Indiv Acc:     {(total_acc_t2c + total_acc_c2t) / (2 * n_batches) * 100:.2f}%")
    print(f"  Proto Acc:     {(total_proto_acc_t2c + total_proto_acc_c2t) / (2 * n_batches) * 100:.2f}%")
    print(f"  Proto Top-5:   {(total_top5_t2c + total_top5_c2t) / (2 * n_batches) * 100:.2f}%")
    print(f"  Proto Top-10:  {(total_top10_t2c + total_top10_c2t) / (2 * n_batches) * 100:.2f}%")
    print(f"  Avg Groups:    {total_n_groups / n_batches:.0f}")
    print(f"  Cohesion:      {total_cohesion / n_batches:.4f}")
    print(f"  Proto t2c:     {total_proto_acc_t2c / n_batches * 100:.2f}%")
    print(f"  Proto c2t:     {total_proto_acc_c2t / n_batches * 100:.2f}%")
    
    return {
        "loss": total_loss / n_batches,
        "proto_acc": (total_proto_acc_t2c + total_proto_acc_c2t) / (2 * n_batches),
        "top5": (total_top5_t2c + total_top5_c2t) / (2 * n_batches),
        "top10": (total_top10_t2c + total_top10_c2t) / (2 * n_batches),
        "n_groups": total_n_groups / n_batches,
    }


print("=" * 60)
print("  V6.3 CHECKPOINT EVALUATION")
print("=" * 60)

# Val evaluation (all batches)
val_results = evaluate_split(val_loader, "VALIDATION")

# Train evaluation (first 20 batches for speed)
train_results = evaluate_split(train_loader, "TRAIN (sample)", max_batches=20)

# Gap analysis
print(f"\n{'='*50}")
print(f"  TRAIN-VAL GAP ANALYSIS")
print(f"{'='*50}")
print(f"  Loss gap:       {val_results['loss'] - train_results['loss']:.4f} ({val_results['loss']/train_results['loss']:.1f}x)")
print(f"  Proto acc gap:  {train_results['proto_acc']*100:.1f}% (train) vs {val_results['proto_acc']*100:.1f}% (val)")
print(f"  Top-5 gap:      {train_results['top5']*100:.1f}% (train) vs {val_results['top5']*100:.1f}% (val)")
print(f"  Top-10 gap:     {train_results['top10']*100:.1f}% (train) vs {val_results['top10']*100:.1f}% (val)")
print(f"  Groups:         {train_results['n_groups']:.0f} (train) vs {val_results['n_groups']:.0f} (val)")

print("\n  Interpretation:")
gap_ratio = train_results['proto_acc'] / max(val_results['proto_acc'], 1e-8)
if gap_ratio > 3:
    print(f"  ⚠ Large train-val gap ({gap_ratio:.1f}x) → significant overfitting")
    print(f"  → Consider: stronger regularization, balanced batches, more augmentation")
elif gap_ratio > 1.5:
    print(f"  ⚡ Moderate gap ({gap_ratio:.1f}x) → some overfitting, normal range")
else:
    print(f"  ✓ Small gap ({gap_ratio:.1f}x) → good generalization")
