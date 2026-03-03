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
        early_stopping_patience: int = 0,
        temp_lr_multiplier: float = 10.0,
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

        # Optimizer — separate LR for temperature/bias (loss params learn faster)
        loss_params = [p for p in model.criterion.parameters() if p.requires_grad]
        param_groups = [
            {"params": model.text_projector.parameters(), "lr": lr},
            {"params": model.cell_projector.parameters(), "lr": lr},
            {"params": loss_params, "lr": lr * temp_lr_multiplier},
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
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = early_stopping_patience
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "temperature": []}

    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch.

        Returns
        -------
        metrics : dict
        """
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_proto_acc_t2c = 0
        total_proto_acc_c2t = 0
        total_n_groups = 0
        total_cohesion = 0
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
            if "proto_acc_t2c" in metrics:
                total_proto_acc_t2c += metrics["proto_acc_t2c"]
                total_proto_acc_c2t += metrics["proto_acc_c2t"]
                total_n_groups += metrics.get("n_groups", 0)
                total_cohesion += metrics.get("cohesion_loss", 0)
            num_batches += 1

            if step % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                proto_str = ""
                if "proto_acc_t2c" in metrics:
                    proto_str = (
                        f" | Proto_t2c: {metrics['proto_acc_t2c']:.3f}"
                        f" | Proto_c2t: {metrics['proto_acc_c2t']:.3f}"
                        f" | Groups: {metrics.get('n_groups', '?')}"
                    )
                logger.info(
                    f"[Epoch {epoch}][Step {step}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} | "
                    f"Acc_t2c: {metrics['acc_t2c']:.3f} | "
                    f"Acc_c2t: {metrics['acc_c2t']:.3f} | "
                    f"Temp: {metrics['temperature']:.4f} | "
                    f"LR: {lr:.2e}{proto_str}"
                )

        result = {
            "train_loss": total_loss / num_batches,
            "train_acc": total_acc / num_batches,
        }
        if total_proto_acc_t2c > 0:
            result["train_proto_acc"] = (total_proto_acc_t2c + total_proto_acc_c2t) / (2 * num_batches)
            result["train_n_groups"] = total_n_groups / num_batches
            result["train_cohesion"] = total_cohesion / num_batches
        return result

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
        total_proto_acc_t2c = 0
        total_proto_acc_c2t = 0
        total_proto_top5_t2c = 0
        total_proto_top5_c2t = 0
        total_proto_top10_t2c = 0
        total_proto_top10_c2t = 0
        total_n_groups = 0
        num_batches = 0

        for batch in self.val_loader:
            text_emb, cell_emb, sample_ids = batch
            text_emb = text_emb.to(self.device)
            cell_emb = cell_emb.to(self.device)

            loss, metrics = self.model(text_emb, cell_emb)

            total_loss += loss.item()
            total_acc_t2c += metrics["acc_t2c"]
            total_acc_c2t += metrics["acc_c2t"]
            if "proto_acc_t2c" in metrics:
                total_proto_acc_t2c += metrics["proto_acc_t2c"]
                total_proto_acc_c2t += metrics["proto_acc_c2t"]
                total_proto_top5_t2c += metrics.get("proto_top5_t2c", 0)
                total_proto_top5_c2t += metrics.get("proto_top5_c2t", 0)
                total_proto_top10_t2c += metrics.get("proto_top10_t2c", 0)
                total_proto_top10_c2t += metrics.get("proto_top10_c2t", 0)
                total_n_groups += metrics.get("n_groups", 0)
            num_batches += 1

        result = {
            "val_loss": total_loss / num_batches,
            "val_acc_t2c": total_acc_t2c / num_batches,
            "val_acc_c2t": total_acc_c2t / num_batches,
            "val_acc": (total_acc_t2c + total_acc_c2t) / (2 * num_batches),
        }
        if total_proto_acc_t2c > 0:
            result["val_proto_acc"] = (total_proto_acc_t2c + total_proto_acc_c2t) / (2 * num_batches)
            result["val_proto_top5"] = (total_proto_top5_t2c + total_proto_top5_c2t) / (2 * num_batches)
            result["val_proto_top10"] = (total_proto_top10_t2c + total_proto_top10_c2t) / (2 * num_batches)
            result["val_n_groups"] = total_n_groups / num_batches
        return result

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

            # Get current temperature
            current_temp = self.model.criterion.temperature.item()

            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_acc"].append(val_metrics["val_acc"])
            self.history["temperature"].append(current_temp)

            # Track prototype-level accuracies (more meaningful for PrototypeSigLIP)
            val_proto_acc = val_metrics.get("val_proto_acc", val_metrics["val_acc"])
            if "val_proto_acc" not in self.history:
                self.history["val_proto_acc"] = []
            self.history["val_proto_acc"].append(val_proto_acc)

            val_proto_top5 = val_metrics.get("val_proto_top5", val_proto_acc)
            if "val_proto_top5" not in self.history:
                self.history["val_proto_top5"] = []
            self.history["val_proto_top5"].append(val_proto_top5)

            val_proto_top10 = val_metrics.get("val_proto_top10", val_proto_top5)
            if "val_proto_top10" not in self.history:
                self.history["val_proto_top10"] = []
            self.history["val_proto_top10"].append(val_proto_top10)

            train_proto_acc = train_metrics.get("train_proto_acc", train_metrics["train_acc"])
            if "train_proto_acc" not in self.history:
                self.history["train_proto_acc"] = []
            self.history["train_proto_acc"].append(train_proto_acc)
            if "train_acc" not in self.history:
                self.history["train_acc"] = []
            self.history["train_acc"].append(train_metrics["train_acc"])
            if "train_cohesion" not in self.history:
                self.history["train_cohesion"] = []
            self.history["train_cohesion"].append(train_metrics.get("train_cohesion", 0))

            proto_str = ""
            if "val_proto_acc" in val_metrics:
                proto_str = (
                    f"\n  Val Proto:  {val_metrics['val_proto_acc']:.3f}"
                    f" | Top5: {val_proto_top5:.3f}"
                    f" | Top10: {val_proto_top10:.3f}"
                    f" (groups={val_metrics['val_n_groups']:.0f})"
                )
            train_proto_str = ""
            gap_str = ""
            if "train_proto_acc" in train_metrics:
                train_proto_str = (
                    f"\n  Train Proto: {train_metrics['train_proto_acc']:.3f}"
                    f" (groups={train_metrics['train_n_groups']:.0f})"
                )
                if val_proto_acc > 0.001:
                    gap = train_metrics['train_proto_acc'] / val_proto_acc
                    gap_str = f"\n  Gap (train/val): {gap:.1f}x"

            logger.info(
                f"\n{'='*60}\n"
                f"Epoch {epoch}/{self.num_epochs} ({elapsed:.1f}s)\n"
                f"  Train Loss: {train_metrics['train_loss']:.4f}{train_proto_str}\n"
                f"  Val   Loss: {val_metrics['val_loss']:.4f}\n"
                f"  Val   Acc:  {val_metrics['val_acc']:.3f}{proto_str}\n"
                f"  Temp:       {current_temp:.4f}{gap_str}\n"
                f"{'='*60}"
            )

            # Use prototype-level acc for early stopping (more stable metric)
            tracking_acc = val_proto_acc if "val_proto_acc" in val_metrics else val_metrics["val_acc"]

            # Save best by tracking accuracy
            if tracking_acc > self.best_val_acc:
                self.best_val_acc = tracking_acc
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
                self.save_checkpoint("clop_best.pth", epoch, val_metrics)
                logger.info(f"  ✓ New best model saved (val_proto_acc={self.best_val_acc:.4f})")
            else:
                self.patience_counter += 1

            # Save periodic
            if epoch % 10 == 0:
                self.save_checkpoint(f"clop_epoch_{epoch}.pth", epoch, val_metrics)

            # Early stopping
            if self.early_stopping_patience > 0 and self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"  ⏹ Early stopping at epoch {epoch} "
                    f"(no val_acc improvement for {self.early_stopping_patience} epochs)"
                )
                break

        # Save final
        self.save_checkpoint("clop_final.pth", epoch, val_metrics)

        # Save history
        with open(self.save_dir / "clop_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint with full architectural config for reconstruction."""
        path = self.save_dir / filename

        # Extract all architectural params needed to reconstruct the model
        model = self.model
        config = {
            "text_dim": model.text_projector.net[0].in_features,
            "cell_dim": model.cell_projector.net[0].in_features,
            "proj_dim": model.proj_dim,
            "use_batch_norm": model.use_batch_norm,
            "use_whitening": model.use_whitening,
            "use_ema": model.use_ema,
            "cell_noise_std": model.cell_noise_std,
            "loss_type": model.loss_type,
            "auto_duplicate_mask": model.auto_duplicate_mask,
            # Temperature params
            "temperature": model.criterion.temperature.item(),
        }

        # Loss-specific config
        if hasattr(model.criterion, 'min_temp'):
            config["min_temperature"] = model.criterion.min_temp
            config["max_temperature"] = model.criterion.max_temp
            config["use_soft_labels"] = model.criterion.use_soft_labels
            config["soft_label_alpha"] = model.criterion.soft_label_alpha
            config["soft_label_bias"] = model.criterion.soft_label_bias
            config["label_smoothing"] = model.criterion.label_smoothing
        if hasattr(model.criterion, 'bias'):
            config["siglip_bias"] = model.criterion.bias.item()
        if hasattr(model.criterion, 'cohesion_weight'):
            config["cohesion_weight"] = model.criterion.cohesion_weight
        if hasattr(model.criterion, 'max_temperature'):
            config["max_temperature_siglip"] = model.criterion.max_temperature

        # Infer layer counts from sequential structure
        text_linear_count = sum(1 for m in model.text_projector.net if isinstance(m, nn.Linear))
        cell_linear_count = sum(1 for m in model.cell_projector.net if isinstance(m, nn.Linear))
        config["text_layers"] = text_linear_count
        config["cell_layers"] = cell_linear_count

        # Infer dropout from projector if possible
        for m in model.text_projector.net:
            if isinstance(m, nn.Dropout):
                config["dropout"] = m.p
                break

        # Whitening eps
        if model.use_whitening and model.text_whitening is not None:
            config["whitening_eps"] = model.text_whitening.eps

        # EMA decay
        if model.use_ema:
            config["ema_decay"] = model.ema_decay

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
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
        # Resolve per-modality preprocessing
        preprocess_text = config.get("preprocess_text_method", "whiten") != "none"
        preprocess_cell = config.get("preprocess_cell_method", "whiten") != "none"

        # Build dataloaders
        train_loader, val_loader = create_dataloaders(
            cache_dir=config.get("cache_dir", "data/cached_latents_v5.2"),
            batch_size=config.get("batch_size", 256),
            val_split=config.get("val_split", 0.1),
            n_folds=config.get("n_folds", 1),
            fold_idx=config.get("fold_idx", 0),
            num_workers=config.get("num_workers", 4),
            stage="clop",
            use_preprocessed=config.get("use_preprocessed", False),
            variant_prob=config.get("variant_prob", 0.0),
            preprocess_text=preprocess_text,
            preprocess_cell=preprocess_cell,
            group_aware_sampling=config.get("group_aware_sampling", False),
            groups_per_batch=config.get("groups_per_batch", 128),
            hard_negative_ratio=config.get("hard_negative_ratio", 0.5),
            hard_negative_k=config.get("hard_negative_k", 20),
            text_embeddings_path=config.get("text_embeddings_path"),
            variant_emb_path=config.get("variant_emb_path"),
            variant_map_path=config.get("variant_map_path"),
        )

        # Auto-detect dimensions from cached data if not specified
        ds = train_loader.dataset
        if hasattr(ds, 'dataset'):  # Subset wrapper from random_split
            ds = ds.dataset
        actual_text_dim = ds.text_dim if hasattr(ds, 'text_dim') else config.get("text_dim", 768)
        actual_cell_dim = ds.cell_dim if hasattr(ds, 'cell_dim') else config.get("cell_dim", 512)
        logger.info(f"Auto-detected dims: text_dim={actual_text_dim}, cell_dim={actual_cell_dim}")

        # Build model
        model = CLOPAligner(
            text_dim=config.get("text_dim", actual_text_dim),
            cell_dim=config.get("cell_dim", actual_cell_dim),
            proj_dim=config.get("proj_dim", 256),
            text_layers=config.get("text_layers", 3),
            cell_layers=config.get("cell_layers", 3),
            dropout=config.get("dropout", 0.1),
            use_batch_norm=config.get("use_batch_norm", True),
            temperature=config.get("temperature", 0.07),
            min_temperature=config.get("min_temperature", 0.01),
            max_temperature=config.get("max_temperature", 0.5),
            label_smoothing=config.get("label_smoothing", 0.1),
            use_ema=config.get("use_ema", False),
            ema_decay=config.get("ema_decay", 0.999),
            use_soft_labels=config.get("use_soft_labels", False),
            soft_label_alpha=config.get("soft_label_alpha", 2.0),
            soft_label_bias=config.get("soft_label_bias", 5.0),
            cell_noise_std=config.get("cell_noise_std", 0.0),
            use_whitening=config.get("use_whitening", False),
            whitening_eps=config.get("whitening_eps", 1e-4),
            loss_type=config.get("loss_type", "infonce"),
            auto_duplicate_mask=config.get("auto_duplicate_mask", False),
            cohesion_weight=config.get("cohesion_weight", 0.1),
            temp_reg_weight=config.get("temp_reg_weight", 0.0),
        )

        # Run embedding preprocessing if needed and preprocessed files don't exist
        if config.get("use_preprocessed", False):
            from ..data_pipeline.embedding_preprocessor import preprocess_cached_embeddings
            cache_dir = Path(config.get("cache_dir", "data/cached_latents_v5.2"))
            text_pp = cache_dir / "text_embeddings_preprocessed.npy"
            cell_pp = cache_dir / "cell_embeddings_preprocessed.npy"
            if not text_pp.exists() or not cell_pp.exists():
                logger.info("Preprocessed embeddings not found — running preprocessing...")
                preprocess_cached_embeddings(
                    cache_dir=str(cache_dir),
                    text_method=config.get("preprocess_text_method", "whiten"),
                    cell_method=config.get("preprocess_cell_method", "whiten"),
                )

        # Compute whitening statistics from training text embeddings
        if config.get("use_whitening", False):
            train_subset = train_loader.dataset
            if hasattr(train_subset, 'dataset'):
                full_dataset = train_subset.dataset
            else:
                full_dataset = train_subset
            train_indices = train_subset.indices if hasattr(train_subset, 'indices') else list(range(len(train_subset)))
            train_text_embs = np.array(full_dataset.text_emb[train_indices])
            model.text_whitening.compute_stats(train_text_embs)
            logger.info(
                f"Text whitening computed from {len(train_indices)} training samples. "
                f"Mean norm: {np.linalg.norm(train_text_embs.mean(axis=0)):.4f}"
            )

        return cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 0.01),
            num_epochs=config.get("num_epochs", 200),
            warmup_epochs=config.get("warmup_epochs", 5),
            save_dir=config.get("save_dir", "models/checkpoints"),
            device=config.get("device", "cuda"),
            use_amp=config.get("use_amp", True),
            grad_clip=config.get("grad_clip", 1.0),
            early_stopping_patience=config.get("early_stopping_patience", 0),
            temp_lr_multiplier=config.get("temp_lr_multiplier", 10.0),
        )

    def project_and_save(
        self,
        output_path: str = "data/cached_latents_v5.2/projected_text.npy",
        use_best: bool = True,
    ):
        """Project all text embeddings through the trained projector.

        This creates the condition vectors for DiT training.
        IMPORTANT: Downstream inference must load the SAME checkpoint used here.

        Parameters
        ----------
        output_path : str
            Path to save projected text embeddings.
        use_best : bool
            If True (default), loads the best checkpoint for projection.
            If False, uses the current (final) model state.
            The choice must match the checkpoint loaded at inference time.
        """
        if use_best and self.save_dir:
            best_path = Path(self.save_dir) / "clop_best.pth"
            if best_path.exists():
                ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt["model_state_dict"])
                logger.info(
                    f"Loaded best checkpoint (epoch {ckpt.get('epoch', '?')}) "
                    f"for projection. Downstream must use clop_best.pth."
                )
            else:
                logger.warning(
                    f"Best checkpoint not found at {best_path}; "
                    f"using current (final) model state."
                )

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
