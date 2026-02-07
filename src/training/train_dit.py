# train_dit.py — Flow Matching DiT Training
"""
Trainer for DiT (Diffusion Transformer) with Flow Matching objective.

Trains the 1D-DiT to learn the velocity field v(z_t, t, c) for generating
cell embeddings conditioned on text descriptions.

Loss: MSE(v_pred, v_target) where v_target = z_1 - z_0

Usage:
    python -m src.training.train_dit --config configs/dit.yaml
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from ..architecture.dit import DiT1D
from ..data_pipeline.dataset import DiTDataset, create_dataloaders
from .schedulers import CosineWarmupScheduler

logger = logging.getLogger(__name__)


class DiTTrainer:
    """Trainer for DiT Flow Matching generation.

    Parameters
    ----------
    model : DiT1D
        The DiT model.
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
        Warmup epochs.
    save_dir : str
        Checkpoint directory.
    device : str
        Training device.
    use_amp : bool
        Use automatic mixed precision.
    grad_clip : float
        Gradient clipping norm.
    ema_decay : float
        EMA decay for model weights (0 = no EMA).
    log_interval : int
        Log every N steps.
    eval_interval : int
        Run generation evaluation every N epochs.
    """

    def __init__(
        self,
        model: DiT1D,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 200,
        warmup_epochs: int = 10,
        save_dir: str = "models/checkpoints",
        device: str = "cuda",
        use_amp: bool = True,
        grad_clip: float = 1.0,
        ema_decay: float = 0.9999,
        log_interval: int = 50,
        eval_interval: int = 10,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.ema_decay = ema_decay
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
        )

        # Scheduler
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = steps_per_epoch * warmup_epochs
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps
        )

        # AMP
        self.scaler = GradScaler("cuda") if use_amp else None

        # EMA
        if ema_decay > 0:
            self.ema_model = self._create_ema()
        else:
            self.ema_model = None

        # Tracking
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.start_epoch = 1
        self.history = {
            "train_loss": [], "val_loss": [],
            "val_cosine_sim": [], "lr": [],
        }

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a saved checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        """
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            logger.info("  Loaded optimizer state from checkpoint")
        else:
            logger.info("  No optimizer state in checkpoint, using fresh optimizer")

        if self.ema_model is not None and "ema_state_dict" in ckpt:
            self.ema_model.load_state_dict(ckpt["ema_state_dict"])

        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", 0)

        metrics = ckpt.get("metrics", {})
        self.best_val_loss = metrics.get("val_loss", float("inf"))

        logger.info(
            f"  Resumed at epoch {self.start_epoch}, "
            f"global_step={self.global_step}, "
            f"best_val_loss={self.best_val_loss:.6f}"
        )

    def _create_ema(self) -> DiT1D:
        """Create EMA copy of model."""
        import copy
        ema = copy.deepcopy(self.model)
        ema.eval()
        for p in ema.parameters():
            p.requires_grad = False
        return ema

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA model parameters."""
        if self.ema_model is None:
            return
        for s_param, e_param in zip(self.model.parameters(), self.ema_model.parameters()):
            e_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1 - self.ema_decay)

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute flow matching MSE loss.

        Parameters
        ----------
        batch : dict with keys 'z_t', 't', 'v_target', 'cond'

        Returns
        -------
        loss : scalar tensor
        """
        z_t = batch["z_t"].to(self.device)
        t = batch["t"].to(self.device)
        v_target = batch["v_target"].to(self.device)
        cond = batch["cond"].to(self.device)

        v_pred = self.model(z_t, t, cond)
        loss = F.mse_loss(v_pred, v_target)

        return loss

    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch.

        Returns
        -------
        metrics : dict
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for step, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast("cuda"):
                    loss = self.compute_loss(batch)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.compute_loss(batch)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.scheduler.step()
            self._update_ema()
            self.global_step += 1

            total_loss += loss.item()
            num_batches += 1

            if step % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[Epoch {epoch}][Step {step}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.6f} | LR: {lr:.2e}"
                )

        return {"train_loss": total_loss / num_batches}

    @torch.no_grad()
    def validate(self) -> Dict:
        """Run validation.

        Returns
        -------
        metrics : dict
        """
        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()

        total_loss = 0
        total_cosine = 0
        num_batches = 0

        for batch in self.val_loader:
            z_t = batch["z_t"].to(self.device)
            t = batch["t"].to(self.device)
            v_target = batch["v_target"].to(self.device)
            cond = batch["cond"].to(self.device)
            z_1 = batch["z_1"].to(self.device)

            v_pred = model(z_t, t, cond)
            loss = F.mse_loss(v_pred, v_target)

            # Also compute cosine similarity between predicted and target velocity
            cosine = F.cosine_similarity(v_pred, v_target, dim=-1).mean()

            total_loss += loss.item()
            total_cosine += cosine.item()
            num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "val_cosine_sim": total_cosine / num_batches,
        }

    @torch.no_grad()
    def evaluate_generation(self, num_samples: int = 256) -> Dict:
        """Evaluate generation quality using full metric suite.

        Generates embeddings and compares with real data using Fréchet Distance,
        MMD, Coverage/Density, KL divergence, and cosine similarity.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.

        Returns
        -------
        metrics : dict
        """
        from ..evaluation.metrics import GenerationMetrics

        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()

        # Get real data samples and conditions
        real_embs = []
        conditions = []
        for batch in self.val_loader:
            real_embs.append(batch["z_1"])
            conditions.append(batch["cond"])
            if sum(e.shape[0] for e in real_embs) >= num_samples:
                break

        real_embs = torch.cat(real_embs)[:num_samples].to(self.device)
        conditions = torch.cat(conditions)[:num_samples].to(self.device)

        # Generate
        generated = model.sample(conditions, num_steps=4, cfg_scale=3.0)

        real_np = real_embs.cpu().numpy()
        gen_np = generated.cpu().numpy()

        # Full evaluation via GenerationMetrics
        metrics = GenerationMetrics.full_evaluation(real_np, gen_np)

        # Additional quick stats
        cosine_sims = F.cosine_similarity(
            generated.unsqueeze(1), real_embs.unsqueeze(0), dim=-1
        )
        metrics["mean_nearest_cosine"] = cosine_sims.max(dim=1).values.mean().item()
        metrics["real_norm_mean"] = real_embs.norm(dim=-1).mean().item()
        metrics["gen_norm_mean"] = generated.norm(dim=-1).mean().item()

        logger.info(f"Generation eval: {json.dumps(metrics, indent=2)}")
        return metrics

    def train(self) -> Dict:
        """Full training loop.

        Returns
        -------
        history : dict
        """
        param_info = self.model.count_parameters()
        logger.info("=" * 60)
        logger.info("DiT Flow Matching Training")
        logger.info(f"Model params: {param_info['trainable_M']}")
        logger.info(f"Device: {self.device} | AMP: {self.use_amp} | EMA: {self.ema_decay}")
        logger.info("=" * 60)

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            elapsed = time.time() - t0

            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_cosine_sim"].append(val_metrics["val_cosine_sim"])
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            logger.info(
                f"\n{'='*60}\n"
                f"Epoch {epoch}/{self.num_epochs} ({elapsed:.1f}s)\n"
                f"  Train Loss: {train_metrics['train_loss']:.6f}\n"
                f"  Val   Loss: {val_metrics['val_loss']:.6f}\n"
                f"  Val  Cosine: {val_metrics['val_cosine_sim']:.4f}\n"
                f"{'='*60}"
            )

            # Save best
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint("dit_best.pth", epoch, val_metrics)
                logger.info(f"  ✓ New best model (val_loss={self.best_val_loss:.6f})")

            # Periodic evaluation (skip optimizer state to reduce I/O)
            if epoch % self.eval_interval == 0:
                gen_metrics = self.evaluate_generation()
                self.save_checkpoint(
                    f"dit_epoch_{epoch}.pth", epoch, val_metrics,
                    include_optimizer=False,
                )

        # Final save (with optimizer for potential future resume)
        self.save_checkpoint("dit_final.pth", self.num_epochs, val_metrics)

        with open(self.save_dir / "dit_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict,
                        include_optimizer: bool = True):
        """Save model checkpoint.

        Parameters
        ----------
        filename : str
        epoch : int
        metrics : dict
        include_optimizer : bool
            If False, skip optimizer state to reduce file size (~170MB vs ~338MB).
        """
        path = self.save_dir / filename
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "config": {
                "latent_dim": self.model.latent_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_tokens": self.model.num_tokens,
            },
        }
        if include_optimizer:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.ema_model is not None:
            state["ema_state_dict"] = self.ema_model.state_dict()
        torch.save(state, path)

    @classmethod
    def from_config(cls, config: Dict) -> "DiTTrainer":
        """Create trainer from configuration dict.

        Parameters
        ----------
        config : dict

        Returns
        -------
        trainer : DiTTrainer
        """
        model = DiT1D(
            latent_dim=config.get("latent_dim", 512),
            hidden_dim=config.get("hidden_dim", 384),
            cond_dim=config.get("cond_dim", 256),
            num_blocks=config.get("num_blocks", 8),
            num_heads=config.get("num_heads", 6),
            mlp_ratio=config.get("mlp_ratio", 4.0),
            num_tokens=config.get("num_tokens", 16),
            cond_drop_prob=config.get("cond_drop_prob", 0.1),
            attn_drop=config.get("attn_drop", 0.0),
            proj_drop=config.get("proj_drop", 0.1),
        )

        train_loader, val_loader = create_dataloaders(
            cache_dir=config.get("cache_dir", "data/cached_latents"),
            batch_size=config.get("batch_size", 512),
            val_split=config.get("val_split", 0.1),
            num_workers=config.get("num_workers", 4),
            stage="dit",
            projected_text_path=config.get("projected_text_path", None),
        )

        trainer = cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
            num_epochs=config.get("num_epochs", 200),
            warmup_epochs=config.get("warmup_epochs", 10),
            save_dir=config.get("save_dir", "models/checkpoints"),
            device=config.get("device", "cuda"),
            use_amp=config.get("use_amp", True),
            grad_clip=config.get("grad_clip", 1.0),
            ema_decay=config.get("ema_decay", 0.9999),
            log_interval=config.get("log_interval", 50),
            eval_interval=config.get("eval_interval", 10),
        )

        # Resume from checkpoint if specified
        resume_path = config.get("resume")
        if resume_path:
            trainer.resume_from_checkpoint(resume_path)

        return trainer
