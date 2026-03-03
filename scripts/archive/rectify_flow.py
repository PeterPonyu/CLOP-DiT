#!/usr/bin/env python3
"""
Rectified Flow — Path Straightening for CLOP-DiT
=================================================

After initial flow matching training, rectified flow performs one additional
training pass to straighten the learned velocity paths. This enables even
faster sampling (often 4-8 steps instead of 20-50).

Methodology (Liu et al., 2022/2023):
1. Sample (z_0, cond) pairs from training data
2. Generate z_1 using the current flow model (ODE integration)
3. Create new straight training pairs: (z_0, z_1_generated, cond)
4. Retrain the model on these rectified pairs

Result: The new model learns straighter paths between noise and data,
reducing curvature and accelerating convergence during sampling.

Usage:
    python scripts/rectify_flow.py \\
        --dit_checkpoint models/checkpoints/dit_epoch_200.pt \\
        --config configs/dit.yaml \\
        --num_rectification_samples 50000 \\
        --rectify_epochs 50
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RectifiedFlowTrainer:
    """Rectified flow training for path straightening."""
    
    def __init__(
        self,
        dit_model,
        device: str = "cuda",
        num_ode_steps: int = 20,
        cfg_scale: float = 1.0,
    ):
        """
        Parameters
        ----------
        dit_model : DiT1D
            Pretrained DiT model.
        device : str
            Computation device.
        num_ode_steps : int
            Number of ODE steps for generating rectified z_1.
        cfg_scale : float
            Classifier-free guidance scale (1.0 = no guidance).
        """
        self.dit_model = dit_model.to(device)
        self.device = device
        self.num_ode_steps = num_ode_steps
        self.cfg_scale = cfg_scale
    
    @torch.no_grad()
    def generate_rectified_pairs(
        self,
        z_0_real: torch.Tensor,
        cond: torch.Tensor,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Generate z_1 from z_0 using current flow model.
        
        Parameters
        ----------
        z_0_real : (N, D) real noise samples
        cond : (N, D_cond) condition vectors
        batch_size : int
            Batch size for generation (to avoid OOM).
        
        Returns
        -------
        z_1_generated : (N, D) generated data endpoints
        """
        self.dit_model.eval()
        
        all_z1 = []
        num_batches = (len(z_0_real) + batch_size - 1) // batch_size
        
        logger.info(f"Generating rectified pairs for {len(z_0_real)} samples...")
        
        for i in tqdm(range(0, len(z_0_real), batch_size), total=num_batches, desc="Rectifying"):
            batch_z0 = z_0_real[i:i + batch_size].to(self.device)
            batch_cond = cond[i:i + batch_size].to(self.device)
            
            # Integrate forward with current model
            z = batch_z0.clone()
            dt = 1.0 / self.num_ode_steps
            
            for step in range(self.num_ode_steps):
                t_val = step / self.num_ode_steps
                t = torch.full((len(z),), t_val, device=self.device)
                
                # Use CFG if scale > 1.0
                if self.cfg_scale > 1.0:
                    v = self.dit_model.forward_with_cfg(z, t, batch_cond, cfg_scale=self.cfg_scale)
                else:
                    v = self.dit_model(z, t, batch_cond)
                
                z = z + v * dt
            
            all_z1.append(z.cpu())
        
        z_1_generated = torch.cat(all_z1, dim=0)
        logger.info(f"✓ Generated {len(z_1_generated)} rectified endpoints")
        
        return z_1_generated
    
    def create_rectified_dataset(
        self,
        original_loader: DataLoader,
        num_samples: int = 50000,
    ) -> TensorDataset:
        """Create rectified training dataset.
        
        Parameters
        ----------
        original_loader : DataLoader
            Original training data loader.
        num_samples : int
            Number of rectified pairs to generate.
        
        Returns
        -------
        dataset : TensorDataset
            Rectified (z_0, z_1_generated, cond) pairs.
        """
        logger.info(f"Creating rectified dataset with {num_samples} samples...")
        
        # Collect z_1 (real data endpoints) and conditions from original loader
        all_z1_real = []
        all_cond = []
        
        for batch in tqdm(original_loader, desc="Collecting real pairs"):
            # Original batch has z_1 (real cell embeddings)
            all_z1_real.append(batch["z_1"])
            all_cond.append(batch["cond"])
            
            if sum(len(x) for x in all_z1_real) >= num_samples:
                break
        
        z_1_real = torch.cat(all_z1_real, dim=0)[:num_samples]
        cond = torch.cat(all_cond, dim=0)[:num_samples]
        
        # Sample z_0 ~ N(0, I)
        z_0 = torch.randn_like(z_1_real)
        
        # Generate z_1 using current flow model
        z_1_generated = self.generate_rectified_pairs(z_0, cond)
        
        # Create straight path dataset: (z_0, z_1_generated)
        logger.info("✓ Rectified dataset created")
        
        return TensorDataset(z_0, z_1_generated, cond)
    
    def compute_rectified_loss(
        self,
        batch: tuple,
        t_dist: str = "uniform",
    ) -> torch.Tensor:
        """Compute loss for rectified flow training.
        
        Parameters
        ----------
        batch : tuple (z_0, z_1, cond)
        t_dist : str
            't' distribution: 'uniform' or 'logit_normal'.
        
        Returns
        -------
        loss : scalar tensor
        """
        z_0, z_1, cond = batch
        z_0 = z_0.to(self.device)
        z_1 = z_1.to(self.device)
        cond = cond.to(self.device)
        
        # Sample timesteps
        if t_dist == "uniform":
            t = torch.rand(len(z_0), device=self.device)
        else:  # logit_normal
            t = torch.randn(len(z_0), device=self.device) * 0.5
            t = torch.sigmoid(t)
        
        # Linear interpolation (now between z_0 and rectified z_1)
        z_t = (1 - t[:, None]) * z_0 + t[:, None] * z_1
        
        # Target velocity: z_1 - z_0 (straight path)
        v_target = z_1 - z_0
        
        # Predicted velocity
        v_pred = self.dit_model(z_t, t, cond)
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        
        return loss
    
    def train_rectified(
        self,
        rectified_loader: DataLoader,
        num_epochs: int = 50,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        save_interval: int = 10,
        save_dir: str = "models/checkpoints/rectified",
    ) -> Dict:
        """Train on rectified dataset.
        
        Parameters
        ----------
        rectified_loader : DataLoader
            Rectified training data.
        num_epochs : int
            Training epochs.
        lr : float
            Learning rate (typically lower than initial training).
        weight_decay : float
            AdamW weight decay.
        warmup_epochs : int
            Warmup epochs.
        save_interval : int
            Save checkpoint every N epochs.
        save_dir : str
            Checkpoint directory.
        
        Returns
        -------
        history : dict
            Training history.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.dit_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Scheduler (cosine with warmup)
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs * len(rectified_loader),
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=(num_epochs - warmup_epochs) * len(rectified_loader),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs * len(rectified_loader)],
        )
        
        # AMP
        scaler = GradScaler("cuda")
        
        history = {"epoch": [], "train_loss": []}
        
        logger.info("="*80)
        logger.info("RECTIFIED FLOW TRAINING")
        logger.info("="*80)
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Batches per epoch: {len(rectified_loader)}")
        logger.info("="*80)
        
        for epoch in range(1, num_epochs + 1):
            self.dit_model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in tqdm(rectified_loader, desc=f"Epoch {epoch}/{num_epochs}"):
                optimizer.zero_grad()
                
                with autocast("cuda"):
                    loss = self.compute_rectified_loss(batch)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.dit_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            history["epoch"].append(epoch)
            history["train_loss"].append(avg_loss)
            
            logger.info(f"[Epoch {epoch}] Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint
            if epoch % save_interval == 0 or epoch == num_epochs:
                ckpt_path = save_path / f"rectified_epoch_{epoch}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.dit_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "history": history,
                }, ckpt_path)
                logger.info(f"✓ Saved checkpoint: {ckpt_path}")
        
        logger.info("="*80)
        logger.info("RECTIFIED FLOW TRAINING COMPLETE")
        logger.info("="*80)
        
        return history


def main():
    parser = argparse.ArgumentParser(description="Rectified Flow Training")
    parser.add_argument(
        "--dit_checkpoint",
        type=str,
        required=True,
        help="Path to pretrained DiT checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dit.yaml",
        help="DiT config file",
    )
    parser.add_argument(
        "--num_rectification_samples",
        type=int,
        default=50000,
        help="Number of rectified pairs to generate",
    )
    parser.add_argument(
        "--rectify_epochs",
        type=int,
        default=50,
        help="Number of training epochs on rectified data",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (typically lower than initial training)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for rectified training",
    )
    parser.add_argument(
        "--num_ode_steps",
        type=int,
        default=20,
        help="ODE steps for rectification trajectory generation",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="CFG scale for rectification (1.0 = no guidance)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    args = parser.parse_args()
    
    # ========================================================================
    # Load DiT model
    # ========================================================================
    logger.info(f"Loading DiT checkpoint from {args.dit_checkpoint}...")
    
    # Import DiT model (assuming it's in src.architecture)
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.architecture.dit import DiT1D
    import yaml
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create model
    dit_model = DiT1D(
        latent_dim=config.get("latent_dim", 512),
        cond_dim=config.get("cond_dim", 512),
        hidden_dim=config.get("hidden_dim", 512),
        num_blocks=config.get("num_blocks", 8),
        num_heads=config.get("num_heads", 8),
        dropout=config.get("dropout", 0.1),
        cond_dropout_prob=config.get("cond_dropout_prob", 0.1),
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.dit_checkpoint, map_location=args.device)
    dit_model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"✓ Loaded DiT from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # ========================================================================
    # Load original training data
    # ========================================================================
    from src.data_pipeline.dataset import create_dataloaders
    
    logger.info("Loading original training data...")
    train_loader, val_loader = create_dataloaders(
        cache_dir=config.get("cache_dir", "data/cached_latents_v5.2"),
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=4,
        use_preprocessed=True,
    )
    logger.info(f"✓ Loaded {len(train_loader.dataset)} training samples")
    
    # ========================================================================
    # Create rectified dataset
    # ========================================================================
    trainer = RectifiedFlowTrainer(
        dit_model=dit_model,
        device=args.device,
        num_ode_steps=args.num_ode_steps,
        cfg_scale=args.cfg_scale,
    )
    
    rectified_dataset = trainer.create_rectified_dataset(
        original_loader=train_loader,
        num_samples=args.num_rectification_samples,
    )
    
    rectified_loader = DataLoader(
        rectified_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    logger.info(f"✓ Rectified dataset: {len(rectified_dataset)} samples")
    
    # ========================================================================
    # Train on rectified pairs
    # ========================================================================
    history = trainer.train_rectified(
        rectified_loader=rectified_loader,
        num_epochs=args.rectify_epochs,
        lr=args.lr,
        save_dir="models/checkpoints/rectified",
    )
    
    # Save training history
    history_path = Path("models/checkpoints/rectified/history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"✓ Saved training history to {history_path}")


if __name__ == "__main__":
    main()
