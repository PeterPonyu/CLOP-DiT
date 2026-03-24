# train_cell2cell.py — Cell→Cell Conditional Flow Matching Trainer
"""
Trainer for Cell2Cell latent editing via conditional flow matching.

Training data: pairs or unpaired samples with source and target conditions.
  - Paired: (z_src, z_tgt, y_edit)
  - Unpaired: (z_src, y_src) + (z_tgt, y_tgt) with OT matching

Loss: MSE(v_pred, v_target) same as DiT, but the model also receives z_src.

Key differences from DiT training:
  1. Source cell z_src is sampled from the same cache (another cell from different condition)
  2. Edit condition y = projected_text[tgt] - projected_text[src] (delta) or just projected_text[tgt]
  3. Edit strength is randomized during training for robustness
  4. Supports both paired and unpaired regimes
"""

import json
import time
import copy
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

from ..architecture.cell2cell import Cell2CellDiT
from .schedulers import CosineWarmupScheduler

logger = logging.getLogger(__name__)


# ============================================================================
#  Cell2Cell Dataset
# ============================================================================

class Cell2CellDataset(Dataset):
    """Dataset for Cell2Cell flow matching training.

    Creates (source, target, condition) triples from cached embeddings.
    Source and target are sampled from DIFFERENT datasets/conditions
    to learn cross-condition transport.

    For unpaired data: source and target are randomly paired across conditions.
    For paired data: explicit source-target mappings are provided.

    Parameters
    ----------
    cache_dir : str or Path
        Directory containing cached embeddings.
    projected_text_path : str or Path, optional
        CLOP-projected text embeddings.
    mode : str
        'unpaired' (random cross-condition pairing) or 'paired'.
    """

    def __init__(
        self,
        cache_dir: str = "data/cached_latents",
        projected_text_path: Optional[str] = None,
        mode: str = "unpaired",
    ):
        cache_dir = Path(cache_dir)

        self.cell_emb = np.load(cache_dir / "cell_embeddings.npy")
        self.sample_ids = np.load(cache_dir / "sample_ids.npy")

        # Use projected text conditions
        if projected_text_path and Path(projected_text_path).exists():
            self.text_cond = np.load(projected_text_path)
        else:
            self.text_cond = np.load(cache_dir / "text_embeddings.npy")

        # Pre-convert to tensors
        self._cell_tensor = torch.from_numpy(self.cell_emb).float()
        self._cond_tensor = torch.from_numpy(self.text_cond).float()

        # Build per-condition indices for cross-condition sampling
        self.unique_ids = np.unique(self.sample_ids)
        self.id_to_indices = {}
        for sid in self.unique_ids:
            self.id_to_indices[sid] = np.where(self.sample_ids == sid)[0]

        # Pre-build exclusion maps for fast cross-condition sampling
        self._other_sids = {}
        for sid in self.unique_ids:
            others = self.unique_ids[self.unique_ids != sid]
            if len(others) == 0:
                others = self.unique_ids
            self._other_sids[sid] = others

        self.mode = mode
        logger.info(
            f"Cell2CellDataset: {len(self.cell_emb)} cells, "
            f"{len(self.unique_ids)} conditions, mode={mode}"
        )

    def __len__(self) -> int:
        return len(self.cell_emb)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns a (source, target, condition) triple.

        Target = cell at idx (from dataset D_tgt)
        Source = random cell from a DIFFERENT dataset D_src
        Condition = target's text condition

        This forces the model to learn cross-condition transport.
        """
        # Target cell and condition
        z_tgt = self._cell_tensor[idx]
        cond_tgt = self._cond_tensor[idx]
        tgt_sid = int(self.sample_ids[idx])

        # Source cell: sample from a DIFFERENT condition (pre-built map)
        other_sids = self._other_sids[tgt_sid]
        src_sid = other_sids[np.random.randint(len(other_sids))]
        src_indices = self.id_to_indices[src_sid]
        src_idx = src_indices[np.random.randint(len(src_indices))]
        z_src = self._cell_tensor[src_idx]

        # Sample timestep and edit strength
        t = torch.rand(1).squeeze()
        edit_strength = torch.rand(1).squeeze() * 0.8 + 0.1  # ∈ [0.1, 0.9]

        # Flow matching: noised interpolation between noise and target
        z_0 = torch.randn_like(z_tgt)
        z_t = (1 - t) * z_0 + t * z_tgt
        v_target = z_tgt - z_0

        return {
            "z_t": z_t,
            "t": t,
            "v_target": v_target,
            "z_src": z_src,
            "cond": cond_tgt,
            "edit_strength": edit_strength,
            "z_tgt": z_tgt,
        }

    @property
    def latent_dim(self) -> int:
        return self.cell_emb.shape[1]

    @property
    def cond_dim(self) -> int:
        return self.text_cond.shape[1]


# ============================================================================
#  Cell2Cell Trainer
# ============================================================================

class Cell2CellTrainer:
    """Trainer for Cell2Cell conditional flow matching.

    Parameters
    ----------
    model : Cell2CellDiT
    train_loader, val_loader : DataLoader
    lr, weight_decay, num_epochs, ... : training hyperparameters
    identity_weight : float
        Weight for identity preservation loss (source reconstruction).
    """

    def __init__(
        self,
        model: Cell2CellDiT,
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
        identity_weight: float = 0.1,
        log_interval: int = 50,
        eval_interval: int = 25,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.ema_decay = ema_decay
        self.identity_weight = identity_weight
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
        )

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = steps_per_epoch * warmup_epochs
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps
        )

        self.scaler = GradScaler("cuda") if use_amp else None

        # EMA
        if ema_decay > 0:
            self.ema_model = copy.deepcopy(model).eval()
            for p in self.ema_model.parameters():
                p.requires_grad = False
        else:
            self.ema_model = None

        self.best_val_loss = float("inf")
        self.global_step = 0
        self.history = {
            "train_loss": [], "val_loss": [],
            "val_cosine_sim": [], "lr": [],
        }

    @torch.no_grad()
    def _update_ema(self):
        if self.ema_model is None:
            return
        for s_param, e_param in zip(self.model.parameters(), self.ema_model.parameters()):
            e_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1 - self.ema_decay)

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute Cell2Cell flow matching loss.

        Total loss = flow_matching_loss + identity_weight * identity_loss

        Identity loss: when edit_strength ≈ 0, the model should approximately
        reconstruct the source (i.e., no edit → identity mapping).
        """
        z_t = batch["z_t"].to(self.device)
        t = batch["t"].to(self.device)
        v_target = batch["v_target"].to(self.device)
        z_src = batch["z_src"].to(self.device)
        cond = batch["cond"].to(self.device)
        edit_strength = batch["edit_strength"].to(self.device)

        # Primary: flow matching velocity prediction
        v_pred = self.model(z_t, t, z_src, cond, edit_strength)
        flow_loss = F.mse_loss(v_pred, v_target)

        # Identity regularization: only every 4th batch (saves ~25% compute)
        # and only after warmup (step > 1000)
        total_loss = flow_loss

        if (self.identity_weight > 0
                and self.global_step > 1000
                and self.global_step % 4 == 0):
            with torch.no_grad():
                z_0_id = torch.randn_like(z_src)
                t_id = torch.rand(z_src.shape[0], device=self.device)
                z_t_id = (1 - t_id.unsqueeze(1)) * z_0_id + t_id.unsqueeze(1) * z_src
                v_target_id = z_src - z_0_id
            s_zero = torch.zeros(z_src.shape[0], device=self.device)
            v_pred_id = self.model(z_t_id, t_id, z_src, cond, s_zero)
            id_loss = F.mse_loss(v_pred_id, v_target_id)
            total_loss = flow_loss + self.identity_weight * id_loss

        return total_loss

    def train_epoch(self, epoch: int) -> Dict:
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
        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()

        total_loss = 0
        total_cosine = 0
        num_batches = 0

        for batch in self.val_loader:
            z_t = batch["z_t"].to(self.device)
            t = batch["t"].to(self.device)
            v_target = batch["v_target"].to(self.device)
            z_src = batch["z_src"].to(self.device)
            cond = batch["cond"].to(self.device)
            edit_strength = batch["edit_strength"].to(self.device)

            v_pred = model(z_t, t, z_src, cond, edit_strength)
            loss = F.mse_loss(v_pred, v_target)
            cosine = F.cosine_similarity(v_pred, v_target, dim=-1).mean()

            total_loss += loss.item()
            total_cosine += cosine.item()
            num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "val_cosine_sim": total_cosine / num_batches,
        }

    @torch.no_grad()
    def evaluate_editing(self, num_samples: int = 128) -> Dict:
        """Evaluate editing quality: edit cells and compare with target distribution."""
        from ..evaluation.metrics import GenerationMetrics

        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()

        real_tgts = []
        conditions = []
        sources = []

        for batch in self.val_loader:
            real_tgts.append(batch["z_tgt"])
            conditions.append(batch["cond"])
            sources.append(batch["z_src"])
            if sum(e.shape[0] for e in real_tgts) >= num_samples:
                break

        real_tgts = torch.cat(real_tgts)[:num_samples].to(self.device)
        conditions = torch.cat(conditions)[:num_samples].to(self.device)
        sources = torch.cat(sources)[:num_samples].to(self.device)

        # Edit with medium strength
        edited = model.edit(
            sources, conditions,
            edit_strength=0.5,
            num_steps=20, cfg_scale=3.0, src_cfg_scale=1.5
        )

        real_np = real_tgts.cpu().numpy()
        gen_np = edited.cpu().numpy()

        metrics = GenerationMetrics.full_evaluation(real_np, gen_np)

        # Source→target vs source→edited cosine
        src_tgt_cos = F.cosine_similarity(sources, real_tgts, dim=-1).mean().item()
        src_edit_cos = F.cosine_similarity(sources, edited, dim=-1).mean().item()
        edit_tgt_cos = F.cosine_similarity(edited, real_tgts, dim=-1).mean().item()

        metrics["src_tgt_cosine"] = src_tgt_cos
        metrics["src_edit_cosine"] = src_edit_cos
        metrics["edit_tgt_cosine"] = edit_tgt_cos

        logger.info(f"Edit eval: {json.dumps(metrics, indent=2)}")
        return metrics

    def train(self) -> Dict:
        param_info = self.model.count_parameters()
        logger.info("=" * 60)
        logger.info("Cell2Cell Flow Matching Training")
        logger.info(f"Model params: {param_info['trainable_M']}")
        logger.info(f"Device: {self.device} | AMP: {self.use_amp} | EMA: {self.ema_decay}")
        logger.info(f"Identity weight: {self.identity_weight}")
        logger.info("=" * 60)

        for epoch in range(1, self.num_epochs + 1):
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

            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint("cell2cell_best.pth", epoch, val_metrics)
                logger.info(f"  ✓ New best model (val_loss={self.best_val_loss:.6f})")

            if epoch % 10 == 0:
                self.save_checkpoint(f"cell2cell_epoch_{epoch}.pth", epoch, val_metrics)

            if epoch % self.eval_interval == 0:
                edit_metrics = self.evaluate_editing()

        self.save_checkpoint("cell2cell_final.pth", self.num_epochs, val_metrics)

        with open(self.save_dir / "cell2cell_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        path = self.save_dir / filename
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": {
                "latent_dim": self.model.latent_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_tokens": self.model.num_tokens,
            },
        }
        if self.ema_model is not None:
            state["ema_state_dict"] = self.ema_model.state_dict()
        torch.save(state, path)

    @classmethod
    def from_config(cls, config: Dict) -> "Cell2CellTrainer":
        """Create trainer from config dict."""
        cache_dir = config.get("cache_dir", "data/cached_latents")
        projected_text_path = config.get("projected_text_path", None)

        # Auto-detect projected text
        if projected_text_path is None:
            default_path = Path(cache_dir) / "projected_text.npy"
            if default_path.exists():
                projected_text_path = str(default_path)

        # Build dataset
        full_ds = Cell2CellDataset(
            cache_dir=cache_dir,
            projected_text_path=projected_text_path,
            mode=config.get("mode", "unpaired"),
        )

        # Train/val split
        n = len(full_ds)
        n_val = int(n * config.get("val_split", 0.1))
        n_train = n - n_val
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

        nw = config.get("num_workers", 4)
        train_loader = DataLoader(
            train_ds,
            batch_size=config.get("batch_size", 512),
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            drop_last=True,
            persistent_workers=nw > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.get("batch_size", 512),
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
        )

        # Build model
        model = Cell2CellDiT(
            latent_dim=config.get("latent_dim", full_ds.latent_dim),
            hidden_dim=config.get("hidden_dim", 384),
            cond_dim=config.get("cond_dim", full_ds.cond_dim),
            num_blocks=config.get("num_blocks", 8),
            num_heads=config.get("num_heads", 6),
            mlp_ratio=config.get("mlp_ratio", 4.0),
            num_tokens=config.get("num_tokens", 16),
            cond_drop_prob=config.get("cond_drop_prob", 0.1),
            src_drop_prob=config.get("src_drop_prob", 0.1),
        )

        return cls(
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
            identity_weight=config.get("identity_weight", 0.1),
        )
