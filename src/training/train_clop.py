# train_clop.py — CLOP Contrastive Alignment Training
"""
Trainer for CLOP (Contrastive Language-Omics Pre-training).

Trains the text and cell projectors to align embeddings in a shared space
using InfoNCE contrastive loss.

Usage:
    python -m src.training.train_clop --config configs/clop.yaml
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from ..architecture.clop import CLOPAligner
from ..data_pipeline.dataset import CLOPDataset, create_dataloaders
from .schedulers import CosineWarmupScheduler

logger = logging.getLogger(__name__)


class CLOPTrainer:
    """Trainer for CLOP contrastive alignment.

    Parameters
    ----------
    model : CLOPAligner
        The CLOP alignment model.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    lr : float
        Learning rate.
    weight_decay : float
        AdamW weight decay.
    num_epochs : int
        Number of training epochs.
    warmup_epochs : int
        Warmup epochs for cosine scheduler.
    save_dir : str
        Directory to save checkpoints.
    device : str
        Training device.
    use_amp : bool
        Use automatic mixed precision.
    grad_clip : float
        Gradient clipping norm. 0 = no clipping.
    log_interval : int
        Log every N steps.
    """

    def __init__(
        self,
        model: CLOPAligner,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 100,
        warmup_epochs: int = 5,
        save_dir: str = "models/checkpoints",
        device: str = "cuda",
        use_amp: bool = True,
        grad_clip: float = 1.0,
        log_interval: int = 50,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer — separate LR for temperature
        param_groups = [
            {"params": model.text_projector.parameters(), "lr": lr},
            {"params": model.cell_projector.parameters(), "lr": lr},
            {"params": [model.criterion.log_temperature], "lr": lr * 10},
        ]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        # Scheduler
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = steps_per_epoch * warmup_epochs
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps
        )

        # AMP
        self.scaler = GradScaler("cuda") if use_amp else None

        # Tracking
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch.

        Returns
        -------
        metrics : dict
        """
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        for step, batch in enumerate(self.train_loader):
            text_emb, cell_emb, sample_ids = batch
            text_emb = text_emb.to(self.device)
            cell_emb = cell_emb.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast("cuda"):
                    loss, metrics = self.model(text_emb, cell_emb)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
            else:
                loss, metrics = self.model(text_emb, cell_emb)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

            total_loss += loss.item()
            total_acc += (metrics["acc_t2c"] + metrics["acc_c2t"]) / 2
            num_batches += 1

            if step % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[Epoch {epoch}][Step {step}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} | "
                    f"Acc_t2c: {metrics['acc_t2c']:.3f} | "
                    f"Acc_c2t: {metrics['acc_c2t']:.3f} | "
                    f"Temp: {metrics['temperature']:.4f} | "
                    f"LR: {lr:.2e}"
                )

        return {
            "train_loss": total_loss / num_batches,
            "train_acc": total_acc / num_batches,
        }

    @torch.no_grad()
    def validate(self) -> Dict:
        """Run validation.

        Returns
        -------
        metrics : dict
        """
        self.model.eval()
        total_loss = 0
        total_acc_t2c = 0
        total_acc_c2t = 0
        num_batches = 0

        for batch in self.val_loader:
            text_emb, cell_emb, _ = batch
            text_emb = text_emb.to(self.device)
            cell_emb = cell_emb.to(self.device)

            loss, metrics = self.model(text_emb, cell_emb)

            total_loss += loss.item()
            total_acc_t2c += metrics["acc_t2c"]
            total_acc_c2t += metrics["acc_c2t"]
            num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "val_acc_t2c": total_acc_t2c / num_batches,
            "val_acc_c2t": total_acc_c2t / num_batches,
            "val_acc": (total_acc_t2c + total_acc_c2t) / (2 * num_batches),
        }

    def train(self) -> Dict:
        """Full training loop.

        Returns
        -------
        history : dict
        """
        logger.info("=" * 60)
        logger.info("CLOP Alignment Training")
        logger.info(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Device: {self.device} | AMP: {self.use_amp}")
        logger.info("=" * 60)

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            elapsed = time.time() - t0

            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_acc"].append(val_metrics["val_acc"])

            logger.info(
                f"\n{'='*60}\n"
                f"Epoch {epoch}/{self.num_epochs} ({elapsed:.1f}s)\n"
                f"  Train Loss: {train_metrics['train_loss']:.4f}\n"
                f"  Val   Loss: {val_metrics['val_loss']:.4f}\n"
                f"  Val   Acc:  {val_metrics['val_acc']:.3f}\n"
                f"{'='*60}"
            )

            # Save best
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint("clop_best.pth", epoch, val_metrics)
                logger.info(f"  ✓ New best model saved (val_loss={self.best_val_loss:.4f})")

            # Save periodic
            if epoch % 10 == 0:
                self.save_checkpoint(f"clop_epoch_{epoch}.pth", epoch, val_metrics)

        # Save final
        self.save_checkpoint("clop_final.pth", self.num_epochs, val_metrics)

        # Save history
        with open(self.save_dir / "clop_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        path = self.save_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": {
                "proj_dim": self.model.proj_dim,
            },
        }, path)

    @classmethod
    def from_config(cls, config: Dict) -> "CLOPTrainer":
        """Create trainer from configuration dict.

        Parameters
        ----------
        config : dict
            Training configuration.

        Returns
        -------
        trainer : CLOPTrainer
        """
        # Build model
        model = CLOPAligner(
            text_dim=config.get("text_dim", 768),
            cell_dim=config.get("cell_dim", 512),
            proj_dim=config.get("proj_dim", 256),
            text_layers=config.get("text_layers", 3),
            cell_layers=config.get("cell_layers", 3),
            dropout=config.get("dropout", 0.1),
            temperature=config.get("temperature", 0.07),
            label_smoothing=config.get("label_smoothing", 0.1),
            use_ema=config.get("use_ema", False),
        )

        # Build dataloaders
        train_loader, val_loader = create_dataloaders(
            cache_dir=config.get("cache_dir", "data/cached_latents"),
            batch_size=config.get("batch_size", 256),
            val_split=config.get("val_split", 0.1),
            num_workers=config.get("num_workers", 4),
            stage="clop",
        )

        return cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 0.01),
            num_epochs=config.get("num_epochs", 100),
            warmup_epochs=config.get("warmup_epochs", 5),
            save_dir=config.get("save_dir", "models/checkpoints"),
            device=config.get("device", "cuda"),
            use_amp=config.get("use_amp", True),
            grad_clip=config.get("grad_clip", 1.0),
        )

    def project_and_save(self, output_path: str = "data/cached_latents/projected_text.npy"):
        """After training, project all text embeddings through the trained projector.

        This creates the condition vectors for DiT training.

        Parameters
        ----------
        output_path : str
            Path to save projected text embeddings.
        """
        self.model.eval()
        # Navigate DataLoader → Subset → CLOPDataset or DataLoader → CLOPDataset
        ds = self.train_loader.dataset
        if hasattr(ds, 'dataset'):  # Subset wrapper from random_split
            ds = ds.dataset
        if hasattr(ds, 'cache_dir'):
            cache_dir = Path(ds.cache_dir)
        else:
            cache_dir = Path(output_path).parent

        text_emb = np.load(cache_dir / "text_embeddings.npy")
        projected = []

        with torch.no_grad():
            for i in range(0, len(text_emb), 512):
                batch = torch.from_numpy(text_emb[i:i+512]).float().to(self.device)
                proj = self.model.project_text(batch)
                projected.append(proj.cpu().numpy())

        projected = np.concatenate(projected, axis=0)
        np.save(output_path, projected)
        logger.info(f"Projected text embeddings saved: {projected.shape} → {output_path}")
