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
from ..evaluation.embedding_quality import compute_all_quality_metrics
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
        mixup_alpha: float = 0.0,
        rdrop_weight: float = 0.0,
        temp_schedule: str = "none",
        temp_schedule_max: float = 20.0,
        temp_schedule_min: float = 10.0,
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

        # Temperature schedule (overrides learned temperature when active)
        self.temp_schedule = temp_schedule
        self.temp_schedule_max = temp_schedule_max
        self.temp_schedule_min = temp_schedule_min

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

        # Regularization
        self.mixup_alpha = mixup_alpha
        self.rdrop_weight = rdrop_weight

        # Tracking
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = early_stopping_patience
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "temperature": []}

    def _apply_temp_schedule(self, epoch: int):
        """Apply temperature schedule if configured.

        Cosine annealing from temp_max → temp_min over training epochs.
        Directly sets the loss criterion's log_temperature parameter.
        """
        import math
        if self.temp_schedule == "none" or self.temp_schedule is None:
            return
        if not hasattr(self.model.criterion, 'log_temperature'):
            return

        progress = epoch / max(self.num_epochs, 1)  # 0 → 1
        if self.temp_schedule == "cosine":
            # Cosine annealing: starts at max, ends at min
            temp = self.temp_schedule_min + 0.5 * (self.temp_schedule_max - self.temp_schedule_min) * (
                1 + math.cos(math.pi * progress)
            )
        elif self.temp_schedule == "linear":
            temp = self.temp_schedule_max + (self.temp_schedule_min - self.temp_schedule_max) * progress
        else:
            return

        with torch.no_grad():
            self.model.criterion.log_temperature.fill_(math.log(max(temp, 1e-6)))

    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch.

        Returns
        -------
        metrics : dict
        """
        self._apply_temp_schedule(epoch)
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_proto_acc_t2c = 0
        total_proto_acc_c2t = 0
        total_n_groups = 0
        total_cohesion = 0
        num_batches = 0

        for step, batch in enumerate(self.train_loader):
            # Dataset returns (text, cell, sample_id, text_group_id)
            if len(batch) == 4:
                text_emb, cell_emb, sample_ids, group_ids = batch
                group_ids = group_ids.to(self.device)
            else:
                text_emb, cell_emb, sample_ids = batch
                group_ids = None
            text_emb = text_emb.to(self.device)
            cell_emb = cell_emb.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast("cuda"):
                    loss, metrics = self.model(
                        text_emb, cell_emb, mixup_alpha=self.mixup_alpha,
                        group_ids=group_ids,
                    )
                    # R-Drop: second forward pass with different dropout
                    if self.rdrop_weight > 0:
                        loss2, _ = self.model(
                            text_emb, cell_emb, mixup_alpha=self.mixup_alpha,
                            group_ids=group_ids,
                        )
                        loss = (loss + loss2) / 2 + self.rdrop_weight * (loss - loss2).abs()
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
            else:
                loss, metrics = self.model(
                    text_emb, cell_emb, mixup_alpha=self.mixup_alpha,
                    group_ids=group_ids,
                )
                if self.rdrop_weight > 0:
                    loss2, _ = self.model(
                        text_emb, cell_emb, mixup_alpha=self.mixup_alpha,
                        group_ids=group_ids,
                    )
                    loss = (loss + loss2) / 2 + self.rdrop_weight * (loss - loss2).abs()
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
        """Run validation with embedding quality metrics.

        Computes standard contrastive metrics (loss, proto accuracy) PLUS
        structural quality metrics that better predict downstream generation:

        - **alignment**: Mean squared L2 distance between matched pairs.
          Lower = better.  (Wang & Isola, ICML 2020)
        - **uniformity_{text,cell}**: Log-average Gaussian kernel between all
          pairs within each modality.  Lower = more spread.  (ibid.)
        - **mean_cosine_sim**: Average cosine similarity of matched text–cell
          pairs.  Higher = better.  Analogous to "CLIP score".
        - **inter_sep**: Mean cosine distance between cell-type centroids.
          Higher = better type discrimination.
        - **intra_cohesion**: Mean cosine sim of cells to their group centroid.
          Higher = tighter clusters.
        - **text_cell_align**: Mean cosine sim between each group's text
          centroid and cell centroid.  Measures conditioning fidelity.

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

        # Collect projected embeddings for embedding quality metrics
        all_text_proj = []
        all_cell_proj = []
        all_group_ids = []

        for batch in self.val_loader:
            # Dataset returns (text, cell, sample_id, text_group_id)
            if len(batch) == 4:
                text_emb, cell_emb, sample_ids, group_ids = batch
                group_ids = group_ids.to(self.device)
            else:
                text_emb, cell_emb, sample_ids = batch
                group_ids = None
            text_emb = text_emb.to(self.device)
            cell_emb = cell_emb.to(self.device)

            loss, metrics = self.model(text_emb, cell_emb, group_ids=group_ids)

            # Collect projected embeddings for quality metrics
            text_proj = self.model.project_text(text_emb)
            cell_proj = self.model.project_cell(cell_emb)
            all_text_proj.append(text_proj)
            all_cell_proj.append(cell_proj)
            if group_ids is not None:
                all_group_ids.append(group_ids)

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

        # ── Embedding quality metrics (structural space quality) ──
        try:
            all_text_proj = torch.cat(all_text_proj, dim=0)
            all_cell_proj = torch.cat(all_cell_proj, dim=0)
            gids = torch.cat(all_group_ids, dim=0) if all_group_ids else None
            quality = compute_all_quality_metrics(all_text_proj, all_cell_proj, gids)
            result.update({f"val_{k}": v for k, v in quality.items()})
        except Exception as e:
            logger.warning(f"Embedding quality metrics failed: {e}")

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

            # ── Track embedding quality metrics ──
            quality_keys = [
                "val_alignment", "val_uniformity_text", "val_uniformity_cell",
                "val_mean_cosine_sim", "val_inter_sep", "val_intra_cohesion",
                "val_text_cell_align",
            ]
            for qk in quality_keys:
                if qk not in self.history:
                    self.history[qk] = []
                self.history[qk].append(val_metrics.get(qk, float("nan")))

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

            # Embedding quality summary
            eq_str = ""
            align_val = val_metrics.get("val_alignment")
            if align_val is not None:
                eq_str = (
                    f"\n  Embed Quality:"
                    f" align={align_val:.4f}"
                    f" uniform_t={val_metrics.get('val_uniformity_text', 0):.3f}"
                    f" uniform_c={val_metrics.get('val_uniformity_cell', 0):.3f}"
                    f" cos_sim={val_metrics.get('val_mean_cosine_sim', 0):.4f}"
                )
                if "val_inter_sep" in val_metrics:
                    eq_str += (
                        f"\n  Group Quality:"
                        f" sep={val_metrics['val_inter_sep']:.4f}"
                        f" coh={val_metrics.get('val_intra_cohesion', 0):.4f}"
                        f" t↔c={val_metrics.get('val_text_cell_align', 0):.4f}"
                    )

            logger.info(
                f"\n{'='*60}\n"
                f"Epoch {epoch}/{self.num_epochs} ({elapsed:.1f}s)\n"
                f"  Train Loss: {train_metrics['train_loss']:.4f}{train_proto_str}\n"
                f"  Val   Loss: {val_metrics['val_loss']:.4f}\n"
                f"  Val   Acc:  {val_metrics['val_acc']:.3f}{proto_str}\n"
                f"  Temp:       {current_temp:.4f}{gap_str}"
                f"{eq_str}\n"
                f"{'='*60}"
            )

            # ── Best-model selection: composite quality score ──
            # Combines prototype accuracy (discrimination) with text-cell
            # alignment (conditioning fidelity), inter-type separation, and
            # mean cosine similarity. All higher = better.
            # This predicts downstream generation quality better than any
            # single metric alone.
            cos_sim = val_metrics.get("val_mean_cosine_sim", 0.0)
            tc_align = val_metrics.get("val_text_cell_align", 0.0)
            sep = val_metrics.get("val_inter_separation", 0.0)
            proto = val_metrics.get("val_proto_acc", 0.0)

            # quality_score: higher is better
            # proto ∈ [0,1], tc_align ∈ [-1,1], sep ∈ [0,2], cos_sim ∈ [-1,1]
            quality_score = 0.3 * proto + 0.3 * tc_align + 0.2 * sep + 0.2 * cos_sim

            # Fall back to proto acc if quality metrics aren't available
            if cos_sim == 0.0 and tc_align == 0.0 and sep == 0.0:
                tracking_metric = val_proto_acc if "val_proto_acc" in val_metrics else val_metrics["val_acc"]
            else:
                tracking_metric = quality_score

            # Save best by tracking metric
            if tracking_metric > self.best_val_acc:
                self.best_val_acc = tracking_metric
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
                self.save_checkpoint("clop_best.pth", epoch, val_metrics)
                logger.info(f"  ✓ New best model saved (quality_score={self.best_val_acc:.4f})")
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
        from .reproducibility import seed_everything
        seed_everything(
            seed=config.get("seed", 42),
            deterministic=config.get("deterministic", False),
        )

        # Resolve per-modality preprocessing
        preprocess_text = config.get("preprocess_text_method", "whiten") != "none"
        preprocess_cell = config.get("preprocess_cell_method", "whiten") != "none"

        # Build dataloaders
        train_loader, val_loader = create_dataloaders(
            cache_dir=config.get("cache_dir", "data/cached_latents"),
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
            use_deduplicated=config.get("use_deduplicated", False),
            split_strategy=config.get("split_strategy", "stratified"),
            class_weight_power=config.get("class_weight_power", 0.0),
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
            separation_margin=config.get("separation_margin", 0.0),
            separation_threshold=config.get("separation_threshold", 0.3),
        )

        # Freeze temperature if configured as non-learnable (prevents temp runaway)
        if not config.get("temperature_learnable", True):
            if hasattr(model, 'criterion') and hasattr(model.criterion, 'log_temperature'):
                model.criterion.log_temperature.requires_grad_(False)
                logger.info(f"Temperature FIXED at {model.criterion.temperature:.1f} (non-learnable)")
            if hasattr(model, 'criterion') and hasattr(model.criterion, 'bias'):
                model.criterion.bias.requires_grad_(False)
                logger.info("SigLIP bias FIXED (non-learnable)")

        # Run embedding preprocessing if needed and preprocessed files don't exist
        if config.get("use_preprocessed", False):
            from ..data_pipeline.embedding_preprocessor import preprocess_cached_embeddings
            cache_dir = Path(config.get("cache_dir", "data/cached_latents"))
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
            mixup_alpha=config.get("mixup_alpha", 0.0),
            rdrop_weight=config.get("rdrop_weight", 0.0),
            temp_schedule=config.get("temp_schedule", "none"),
            temp_schedule_max=config.get("temp_schedule_max", 20.0),
            temp_schedule_min=config.get("temp_schedule_min", 10.0),
        )

    def project_and_save(
        self,
        output_path: str = "data/cached_latents/projected_text.npy",
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

        # ── Resolve text embedding file (v6.2+ compatible) ──
        # Priority: preprocessed > unique (deduplicated) > raw
        # Must match what the training dataset actually loaded.
        text_emb = None
        text_group_ids = None

        # Check if dataset used deduplicated storage
        if hasattr(ds, '_deduplicated') and ds._deduplicated:
            # Load unique text embeddings and group_ids to expand
            if hasattr(ds, 'text_emb_unique') and ds.text_emb_unique is not None:
                text_emb_unique = np.array(ds.text_emb_unique)
            else:
                for candidate in ["text_embeddings_unique_preprocessed.npy",
                                  "text_embeddings_unique.npy",
                                  "text_embeddings_dedup.npy"]:
                    p = cache_dir / candidate
                    if p.exists():
                        text_emb_unique = np.load(p)
                        break
                else:
                    raise FileNotFoundError(
                        f"No unique text embedding file found in {cache_dir}"
                    )

            # Load group mapping to expand unique → per-cell
            # IMPORTANT: use_deduplicated must use dedup group_ids (0-68)
            # not the original group_ids (0-862) which index 1088 sub-clusters
            if hasattr(ds, 'use_deduplicated') and ds.use_deduplicated:
                gid_order = ["text_group_ids_dedup.npy"]
            else:
                gid_order = ["text_group_ids.npy", "text_group_ids_dedup.npy"]
            for gid_name in gid_order:
                gid_path = cache_dir / gid_name
                if gid_path.exists():
                    text_group_ids = np.load(gid_path)
                    break

            if text_group_ids is not None:
                # Expand: project unique embeddings, then index by group_ids
                logger.info(
                    f"Projecting {text_emb_unique.shape[0]} unique text embeddings, "
                    f"then expanding to {len(text_group_ids)} cells via group_ids"
                )
                projected_unique = []
                with torch.no_grad():
                    for i in range(0, len(text_emb_unique), 512):
                        batch = torch.from_numpy(
                            text_emb_unique[i:i+512]
                        ).float().to(self.device)
                        proj = self.model.project_text(batch)
                        projected_unique.append(proj.cpu().numpy())
                projected_unique = np.concatenate(projected_unique, axis=0)
                projected = projected_unique[text_group_ids]
                np.save(output_path, projected)
                logger.info(
                    f"Projected text embeddings saved: {projected.shape} → {output_path}"
                )
                return
            else:
                # Unique embeddings but no group IDs; project unique only
                text_emb = text_emb_unique
        else:
            # Legacy duplicated format: load full per-cell text embeddings
            for candidate in ["text_embeddings_preprocessed.npy",
                              "text_embeddings.npy"]:
                p = cache_dir / candidate
                if p.exists():
                    text_emb = np.load(p)
                    logger.info(f"Loading text embeddings from {candidate}")
                    break

        if text_emb is None:
            raise FileNotFoundError(
                f"No text embedding file found in {cache_dir}. "
                f"Expected one of: text_embeddings_preprocessed.npy, "
                f"text_embeddings.npy, text_embeddings_unique.npy"
            )

        projected = []
        with torch.no_grad():
            for i in range(0, len(text_emb), 512):
                batch = torch.from_numpy(text_emb[i:i+512]).float().to(self.device)
                proj = self.model.project_text(batch)
                projected.append(proj.cpu().numpy())

        projected = np.concatenate(projected, axis=0)
        np.save(output_path, projected)
        logger.info(f"Projected text embeddings saved: {projected.shape} → {output_path}")
