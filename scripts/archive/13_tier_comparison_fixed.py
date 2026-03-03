#!/usr/bin/env python3
"""
Tier 1 vs Tier 2 Architecture Comparison — FIXED
==================================================

Key fix: Standardize cell embeddings BEFORE flow matching so the model
doesn't waste capacity learning the global mean direction (norm 21).

This makes the transport from N(0,I) → standardized data much easier
and allows fair comparison with fewer epochs.

Tier 1: Raw BiomedBERT (1024-d) → DiT (no CLOP)
Tier 2: Fixed CLOP (text-centroid contrastive) → DiT (256-d conds)
"""

import sys
import gc
import json
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.architecture.dit import DiT1D
from src.architecture.clop import CLOPAligner, ProjectionHead
from src.evaluation.metrics import GenerationMetrics
from src.utils.helpers import seed_everything

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path("data/cached_latents_v5.2")
SAVE_DIR = Path("models/checkpoints")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
#  Logging tee
# ============================================================================

class Tee:
    def __init__(self, logfile):
        self.terminal = sys.__stdout__
        self.logfile = open(logfile, "a")
    def write(self, msg):
        self.terminal.write(msg)
        self.logfile.write(msg)
        self.logfile.flush()
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
    def isatty(self):
        return False


# ============================================================================
#  Standardization utilities
# ============================================================================

def compute_standardization(cell_emb):
    """Compute per-dimension mean and std from training data."""
    mu = cell_emb.mean(axis=0)
    sigma = cell_emb.std(axis=0) + 1e-8
    return mu, sigma


def standardize(cell_emb, mu, sigma):
    """Standardize cell embeddings to ~N(0,1) per dimension."""
    return (cell_emb - mu) / sigma


def destandardize(gen, mu, sigma):
    """Reverse standardization: z_gen * sigma + mu."""
    return gen * sigma + mu


# ============================================================================
#  Dataset with standardization
# ============================================================================

class FlexDiTDataset(Dataset):
    """DiT dataset with pre-standardized cell embeddings."""
    
    def __init__(self, cell_emb_std, text_cond, sample_ids,
                 time_sampling="logit_normal"):
        self.cell_emb = torch.from_numpy(cell_emb_std).float()
        self.text_cond = torch.from_numpy(text_cond).float()
        self.sample_ids = sample_ids
        self.time_sampling = time_sampling
    
    def __len__(self):
        return len(self.cell_emb)
    
    def __getitem__(self, idx):
        z_1 = self.cell_emb[idx]
        cond = self.text_cond[idx]
        z_0 = torch.randn_like(z_1)
        
        if self.time_sampling == "logit_normal":
            u = torch.normal(mean=0.0, std=1.0, size=(1,))
            t = torch.sigmoid(u).squeeze().clamp(1e-5, 1-1e-5)
        else:
            t = torch.rand(1).squeeze()
        
        z_t = (1 - t) * z_0 + t * z_1
        v_target = z_1 - z_0
        
        return {"z_t": z_t, "t": t, "v_target": v_target, "cond": cond, "z_1": z_1}


# ============================================================================
#  Training loop
# ============================================================================

def train_dit(model, train_loader, val_loader, num_epochs=200, lr=2e-4,
              warmup_epochs=10, save_prefix=""):
    """Train DiT and return EMA model."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    
    cond_dim_val = model.c_embedder.null_cond.shape[-1]
    ema_model = DiT1D(
        latent_dim=model.latent_dim,
        hidden_dim=model.hidden_dim,
        cond_dim=cond_dim_val,
        num_blocks=len(model.blocks),
        num_heads=model.blocks[0].attn.num_heads,
    ).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    ema_decay = 0.9999
    
    total_steps = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    history = {"train_loss": [], "val_loss": [], "val_cosine": []}
    best_val_loss = float("inf")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            z_t = batch["z_t"].to(DEVICE)
            t = batch["t"].to(DEVICE)
            v_target = batch["v_target"].to(DEVICE)
            cond = batch["cond"].to(DEVICE)
            
            with torch.amp.autocast("cuda"):
                v_pred = model(z_t, t, cond)
                loss = F.mse_loss(v_pred, v_target)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            with torch.no_grad():
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
            
            train_losses.append(loss.item())
        
        avg_train = np.mean(train_losses)
        
        model.eval()
        val_losses = []
        cosine_sims = []
        with torch.no_grad():
            for batch in val_loader:
                z_t = batch["z_t"].to(DEVICE)
                t = batch["t"].to(DEVICE)
                v_target = batch["v_target"].to(DEVICE)
                cond = batch["cond"].to(DEVICE)
                
                v_pred = model(z_t, t, cond)
                loss = F.mse_loss(v_pred, v_target)
                val_losses.append(loss.item())
                
                cos = F.cosine_similarity(v_pred, v_target, dim=-1).mean()
                cosine_sims.append(cos.item())
        
        avg_val = np.mean(val_losses)
        avg_cos = np.mean(cosine_sims)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_cosine"].append(avg_cos)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": ema_model.state_dict(),
                "val_loss": avg_val,
            }, SAVE_DIR / f"{save_prefix}_best.pth")
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{num_epochs}  "
                  f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  "
                  f"val_cos={avg_cos:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
    
    ckpt = torch.load(SAVE_DIR / f"{save_prefix}_best.pth",
                       map_location=DEVICE, weights_only=False)
    ema_model.load_state_dict(ckpt["model_state_dict"])
    history["best_epoch"] = ckpt["epoch"]
    print(f"  Best val_loss={best_val_loss:.4f} at epoch {ckpt['epoch']}")
    
    return ema_model, history


# ============================================================================
#  Evaluation
# ============================================================================

def evaluate_generation(dit_model, real_emb, cond_vectors, cond_dim,
                        mu, sigma, num_per_type=200, cfg_scale=3.0,
                        num_steps=20):
    """Generate, de-standardize, compute FD/Coverage/Density.
    
    cond_vectors: dict of name → (1, cond_dim) tensor
    mu, sigma: standardization parameters for de-standardization
    """
    dit_model.eval()
    gen_all = []
    per_type_metrics = {}
    
    for name, cond in cond_vectors.items():
        cond_batch = cond.expand(num_per_type, -1).to(DEVICE)
        # Generate in standardized space
        gen_std = dit_model.sample(cond_batch, num_steps=num_steps, cfg_scale=cfg_scale)
        # De-standardize
        gen_np = destandardize(gen_std.cpu().numpy(), mu, sigma)
        gen_all.append(gen_np)
        
        real_sub = real_emb[np.random.choice(len(real_emb), num_per_type, replace=False)]
        try:
            pm = GenerationMetrics.full_evaluation(real_sub, gen_np)
            per_type_metrics[name] = pm
        except Exception:
            pass
    
    gen = np.concatenate(gen_all, axis=0)
    real_sub = real_emb[np.random.choice(len(real_emb), len(gen), replace=False)]
    overall = GenerationMetrics.full_evaluation(real_sub, gen)
    
    # Unconditional: random conditions matching training distribution
    cond_random = torch.randn(len(gen), cond_dim).to(DEVICE)
    cond_random = F.normalize(cond_random, dim=-1)
    gen_random_std = dit_model.sample(cond_random, num_steps=num_steps, cfg_scale=cfg_scale)
    gen_random_np = destandardize(gen_random_std.cpu().numpy(), mu, sigma)
    uncond = GenerationMetrics.full_evaluation(real_sub, gen_random_np)
    
    # Inter-type cosine
    type_means = np.array([g.mean(axis=0) for g in gen_all])
    type_means_norm = type_means / (np.linalg.norm(type_means, axis=1, keepdims=True) + 1e-10)
    inter_type_sim = type_means_norm @ type_means_norm.T
    np.fill_diagonal(inter_type_sim, 0)
    triu = np.triu_indices_from(inter_type_sim, k=1)
    inter_type_cos = inter_type_sim[triu].mean()
    
    return {
        "overall": overall,
        "unconditional": uncond,
        "per_type": per_type_metrics,
        "inter_type_cosine": float(inter_type_cos),
        "type_controllability": 1.0 - float(inter_type_cos),
    }


# ============================================================================
#  Tier 2: TextCentroidCLOP
# ============================================================================

class TextCentroidCLOP(nn.Module):
    """Fixed CLOP: text-centroid level contrastive + regression."""
    
    def __init__(self, text_dim=1024, cell_dim=512, proj_dim=256):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(384, proj_dim),
        )
        self.cell_proj = nn.Sequential(
            nn.Linear(cell_dim, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(384, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(384, proj_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(np.log(1/0.07)))
        self.regression_head = nn.Sequential(
            nn.Linear(proj_dim, 384), nn.GELU(),
            nn.Linear(384, cell_dim),
        )
    
    def project_text(self, text_emb):
        z = self.text_proj(text_emb)
        return F.normalize(z, dim=-1)
    
    def project_cell(self, cell_emb):
        z = self.cell_proj(cell_emb)
        return F.normalize(z, dim=-1)
    
    def forward(self, text_emb, cell_emb, group_ids):
        text_z = self.project_text(text_emb)
        cell_z = self.project_cell(cell_emb)
        
        temp = self.temperature.exp().clamp(max=100.0)
        
        # Compute centroids per group
        unique_groups = group_ids.unique()
        text_centroids = []
        cell_centroids = []
        group_list = []
        
        for g in unique_groups:
            mask = group_ids == g
            if mask.sum() < 2:
                continue
            text_centroids.append(F.normalize(text_z[mask].mean(0), dim=-1))
            cell_centroids.append(F.normalize(cell_z[mask].mean(0), dim=-1))
            group_list.append(g.item())
        
        if len(text_centroids) < 2:
            return torch.tensor(0.0, device=text_emb.device, requires_grad=True), {
                "accuracy": 0, "contrastive_loss": 0, "regression_loss": 0,
                "n_groups": 0, "temperature": temp.item()
            }
        
        text_c = torch.stack(text_centroids)
        cell_c = torch.stack(cell_centroids)
        
        # Contrastive on centroids
        logits = text_c @ cell_c.T * temp
        labels = torch.arange(len(text_c), device=logits.device)
        ctr_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        
        acc = ((logits.argmax(dim=-1) == labels).float().mean().item() +
               (logits.T.argmax(dim=-1) == labels).float().mean().item()) / 2
        
        # Regression: projected text centroid → predict cell centroid (in original space)
        cell_mean_orig = []
        for g in unique_groups:
            mask = group_ids == g
            if mask.sum() < 2:
                continue
            cell_mean_orig.append(cell_emb[mask].mean(0))
        cell_mean_orig = torch.stack(cell_mean_orig)
        
        text_pred = self.regression_head(text_c)
        reg_loss = F.mse_loss(text_pred, cell_mean_orig)
        
        total_loss = ctr_loss + 0.5 * reg_loss
        
        metrics = {
            "accuracy": acc,
            "contrastive_loss": ctr_loss.item(),
            "regression_loss": reg_loss.item(),
            "n_groups": len(text_centroids),
            "temperature": temp.item(),
        }
        return total_loss, metrics


class CentroidCLOPDataset(Dataset):
    def __init__(self, text_emb, cell_emb, sample_ids):
        self.text_emb = torch.from_numpy(text_emb).float()
        self.cell_emb = torch.from_numpy(cell_emb).float()
        
        unique_texts, self.text_group_ids = np.unique(
            text_emb.round(decimals=5), axis=0, return_inverse=True
        )
        print(f"  CentroidCLOP: {len(unique_texts)} unique text groups "
              f"for {len(cell_emb)} cells")
    
    def __len__(self):
        return len(self.text_emb)
    
    def __getitem__(self, idx):
        return (self.text_emb[idx], self.cell_emb[idx],
                torch.tensor(self.text_group_ids[idx], dtype=torch.long))


def train_centroid_clop(text_emb, cell_emb, sample_ids, num_epochs=100,
                        batch_size=512, lr=3e-4):
    """Train text-centroid CLOP and return model + projected embeddings."""
    print("\n--- Training Tier 2 CLOP (Text-Centroid Contrastive) ---")
    
    dataset = CentroidCLOPDataset(text_emb, cell_emb, sample_ids)
    
    unique_sids = np.unique(sample_ids)
    rng = np.random.default_rng(42)
    shuffled = unique_sids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.1))
    val_ids = set(shuffled[:n_val].tolist())
    
    train_idx = [i for i, s in enumerate(sample_ids) if s not in val_ids]
    val_idx = [i for i, s in enumerate(sample_ids) if s in val_ids]
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    
    model = TextCentroidCLOP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_metrics = defaultdict(list)
        for text_b, cell_b, group_b in train_loader:
            text_b = text_b.to(DEVICE)
            cell_b = cell_b.to(DEVICE)
            group_b = group_b.to(DEVICE)
            loss, m = model(text_b, cell_b, group_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    train_metrics[k].append(v)
        
        model.eval()
        val_metrics = defaultdict(list)
        with torch.no_grad():
            for text_b, cell_b, group_b in val_loader:
                text_b = text_b.to(DEVICE)
                cell_b = cell_b.to(DEVICE)
                group_b = group_b.to(DEVICE)
                _, m = model(text_b, cell_b, group_b)
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        val_metrics[k].append(v)
        
        val_acc = np.mean(val_metrics["accuracy"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  "
                  f"train_acc={np.mean(train_metrics['accuracy']):.3f}  "
                  f"val_acc={val_acc:.3f}  "
                  f"ctr_loss={np.mean(train_metrics['contrastive_loss']):.3f}  "
                  f"reg_loss={np.mean(train_metrics['regression_loss']):.3f}  "
                  f"n_groups={np.mean(train_metrics['n_groups']):.0f}  "
                  f"temp={np.mean(train_metrics['temperature']):.4f}")
    
    model.load_state_dict(best_state)
    print(f"  Best val_acc={best_val_acc:.3f}")
    
    # Save CLOP model
    torch.save(best_state, SAVE_DIR / "tier2_clop_centroid.pth")
    
    # Project all text
    model.eval()
    projected = []
    with torch.no_grad():
        for i in range(0, len(text_emb), 1024):
            batch = torch.from_numpy(text_emb[i:i+1024]).float().to(DEVICE)
            proj = model.project_text(batch)
            projected.append(proj.cpu().numpy())
    projected = np.concatenate(projected, axis=0)
    
    return model, projected, best_val_acc


# ============================================================================
#  BiomedBERT prompts
# ============================================================================

def encode_prompts_biomedbert(prompts, device=DEVICE):
    from transformers import AutoTokenizer, AutoModel
    import gc
    model_name = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load on CPU to avoid GPU OOM, then move inputs only
    bert = AutoModel.from_pretrained(model_name).eval()
    
    embeddings = {}
    with torch.no_grad():
        for name, text in prompts.items():
            inputs = tokenizer(text, return_tensors="pt", padding=True,
                             truncation=True, max_length=512)
            outputs = bert(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].to(device)
            embeddings[name] = emb
    
    del bert, tokenizer
    gc.collect()
    return embeddings


# ============================================================================
#  Main
# ============================================================================

def main():
    seed_everything(42)
    logfile = LOG_DIR / "tier_comparison_v2.log"
    sys.stdout = Tee(logfile)
    print(f"Logging to {logfile}")
    
    CELL_TYPE_PROMPTS = {
        "CD8_T": "CD8-positive T cell from tumor microenvironment with cytotoxic phenotype",
        "Macrophage": "Tumor-associated macrophage with M2 polarization signature",
        "Epithelial_tumor": "Malignant epithelial cell from primary tumor with high proliferation",
        "Fibroblast": "Cancer-associated fibroblast with activated myofibroblast features",
        "NK_cell": "Natural killer cell with activated cytotoxic function",
        "B_cell": "B lymphocyte from tumor-draining lymph node",
    }
    
    DIT_EPOCHS = 200
    CLOP_EPOCHS = 100
    
    print("=" * 70)
    print("  TIER 1 vs TIER 2 — FIXED (standardized flow matching, 200 epochs)")
    print("=" * 70)
    
    # Load data
    cell_emb = np.load(CACHE_DIR / "cell_embeddings.npy")
    text_emb = np.load(CACHE_DIR / "text_embeddings.npy")
    sample_ids = np.load(CACHE_DIR / "sample_ids.npy")
    print(f"  Cells: {cell_emb.shape}, Text: {text_emb.shape}")
    
    # Standardize cell embeddings
    mu, sigma = compute_standardization(cell_emb)
    cell_std = standardize(cell_emb, mu, sigma)
    print(f"  Standardized: mean={cell_std.mean():.4f}, std={cell_std.std():.4f}, "
          f"norm={np.linalg.norm(cell_std, axis=1).mean():.2f}")
    print(f"  Data mean norm (pre-std): {np.linalg.norm(mu):.2f}")
    
    # Group split
    unique_sids = np.unique(sample_ids)
    rng = np.random.default_rng(42)
    shuffled = unique_sids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.1))
    val_ids = set(shuffled[:n_val].tolist())
    train_idx = [i for i, s in enumerate(sample_ids) if s not in val_ids]
    val_idx = [i for i, s in enumerate(sample_ids) if s in val_ids]
    real_emb = cell_emb  # Original space for evaluation
    
    results = {}
    
    # ==================================================================
    #  Tier 1: Raw BiomedBERT (1024-d) → DiT (no CLOP)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TIER 1: Raw BiomedBERT → DiT (standardized flow matching)")
    print("=" * 70)
    
    t1_start = time.time()
    
    t1_train_ds = FlexDiTDataset(cell_std[train_idx], text_emb[train_idx],
                                  sample_ids[train_idx])
    t1_val_ds = FlexDiTDataset(cell_std[val_idx], text_emb[val_idx],
                                sample_ids[val_idx])
    t1_train_loader = DataLoader(t1_train_ds, batch_size=1024, shuffle=True,
                                  num_workers=4, drop_last=True, pin_memory=True)
    t1_val_loader = DataLoader(t1_val_ds, batch_size=1024, shuffle=True,
                                num_workers=4, pin_memory=True)
    
    t1_dit = DiT1D(latent_dim=512, hidden_dim=384, cond_dim=1024,
                    num_blocks=8, num_heads=6, cond_drop_prob=0.15)
    print(f"  DiT params: {sum(p.numel() for p in t1_dit.parameters())/1e6:.2f}M")
    
    t1_ckpt_path = SAVE_DIR / "t1_std_dit_best.pth"
    if t1_ckpt_path.exists():
        print(f"  [SKIP] Loading existing Tier 1 checkpoint: {t1_ckpt_path}")
        ckpt = torch.load(t1_ckpt_path, map_location=DEVICE, weights_only=False)
        t1_dit.load_state_dict(ckpt["model_state_dict"])
        t1_model = t1_dit.to(DEVICE)
        t1_hist = {"best_epoch": ckpt["epoch"]}
        t1_time = 0.0
        print(f"  Best epoch was {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    else:
        t1_model, t1_hist = train_dit(
            t1_dit, t1_train_loader, t1_val_loader,
            num_epochs=DIT_EPOCHS, save_prefix="t1_std_dit"
        )
        t1_time = time.time() - t1_start
    
    # Evaluate
    # Free GPU before eval
    del t1_dit
    torch.cuda.empty_cache()
    import gc; gc.collect()
    
    print("\n  Evaluating Tier 1...")
    bert_embs = encode_prompts_biomedbert(CELL_TYPE_PROMPTS)
    t1_conds = {name: emb for name, emb in bert_embs.items()}
    
    t1_results = evaluate_generation(
        t1_model, real_emb, t1_conds, cond_dim=1024,
        mu=mu, sigma=sigma
    )
    t1_results["training_time"] = t1_time
    t1_results["history"] = t1_hist
    results["tier1"] = t1_results
    
    print(f"\n  TIER 1 RESULTS:")
    print(f"    FD:     {t1_results['overall']['frechet_distance']:.3f}")
    print(f"    Cov:    {t1_results['overall']['coverage']:.3f}")
    print(f"    Dens:   {t1_results['overall']['density']:.3f}")
    print(f"    Uncond FD: {t1_results['unconditional']['frechet_distance']:.3f}")
    print(f"    Inter-type cos: {t1_results['inter_type_cosine']:.4f}")
    print(f"    Controllability: {t1_results['type_controllability']:.4f}")
    print(f"    Time:   {t1_time:.0f}s")
    
    del t1_model, t1_train_ds, t1_val_ds
    torch.cuda.empty_cache()
    gc.collect()
    
    # ==================================================================
    #  Tier 2: Fixed CLOP (text-centroid) → DiT
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TIER 2: Fixed CLOP (Text-Centroid) → DiT (standardized)")
    print("=" * 70)
    
    t2_start = time.time()
    
    # Train CLOP
    t2_clop, t2_projected, t2_clop_acc = train_centroid_clop(
        text_emb, cell_emb, sample_ids,
        num_epochs=CLOP_EPOCHS, batch_size=512
    )
    
    # DiT with CLOP-projected conditions
    t2_train_ds = FlexDiTDataset(cell_std[train_idx], t2_projected[train_idx],
                                  sample_ids[train_idx])
    t2_val_ds = FlexDiTDataset(cell_std[val_idx], t2_projected[val_idx],
                                sample_ids[val_idx])
    t2_train_loader = DataLoader(t2_train_ds, batch_size=1024, shuffle=True,
                                  num_workers=4, drop_last=True, pin_memory=True)
    t2_val_loader = DataLoader(t2_val_ds, batch_size=1024, shuffle=True,
                                num_workers=4, pin_memory=True)
    
    t2_dit = DiT1D(latent_dim=512, hidden_dim=384, cond_dim=256,
                    num_blocks=8, num_heads=6, cond_drop_prob=0.15)
    print(f"\n  DiT params: {sum(p.numel() for p in t2_dit.parameters())/1e6:.2f}M")
    
    t2_model, t2_hist = train_dit(
        t2_dit, t2_train_loader, t2_val_loader,
        num_epochs=DIT_EPOCHS, save_prefix="t2_std_dit"
    )
    t2_time = time.time() - t2_start
    
    # Free training model before evaluation
    del t2_dit
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n  Evaluating Tier 2...")
    t2_conds = {}
    t2_clop.eval()
    with torch.no_grad():
        for name, emb in bert_embs.items():
            proj = t2_clop.project_text(emb.to(DEVICE))
            t2_conds[name] = proj
    
    t2_results = evaluate_generation(
        t2_model, real_emb, t2_conds, cond_dim=256,
        mu=mu, sigma=sigma
    )
    t2_results["training_time"] = t2_time
    t2_results["history"] = t2_hist
    t2_results["clop_val_acc"] = float(t2_clop_acc)
    results["tier2"] = t2_results
    
    print(f"\n  TIER 2 RESULTS:")
    print(f"    CLOP val_acc: {t2_clop_acc:.3f}")
    print(f"    FD:     {t2_results['overall']['frechet_distance']:.3f}")
    print(f"    Cov:    {t2_results['overall']['coverage']:.3f}")
    print(f"    Dens:   {t2_results['overall']['density']:.3f}")
    print(f"    Uncond FD: {t2_results['unconditional']['frechet_distance']:.3f}")
    print(f"    Inter-type cos: {t2_results['inter_type_cosine']:.4f}")
    print(f"    Controllability: {t2_results['type_controllability']:.4f}")
    print(f"    Time:   {t2_time:.0f}s")
    
    del t2_model, t2_clop
    torch.cuda.empty_cache()
    gc.collect()
    
    # ==================================================================
    #  Also test ORIGINAL DiT (200 epochs, original CLOP) for baseline
    # ==================================================================
    print("\n" + "=" * 70)
    print("  BASELINE: Original CLOP→DiT (200 epochs, no standardization)")
    print("=" * 70)
    
    orig_dit = DiT1D(latent_dim=512, hidden_dim=384, cond_dim=256,
                      num_blocks=8, num_heads=6).cuda()
    ckpt = torch.load(SAVE_DIR / "dit_best.pth", weights_only=False)
    orig_dit.load_state_dict(ckpt["model_state_dict"])
    orig_dit.eval()
    
    # Original uses projected_text from original CLOP
    orig_proj = np.load(CACHE_DIR / "projected_text.npy")
    
    # Re-evaluate original with same eval pipeline (no destandardization!)
    orig_conds = {}
    orig_clop = CLOPAligner(text_dim=1024, cell_dim=512, proj_dim=256).cuda()
    orig_clop_ckpt = torch.load(SAVE_DIR / "clop_best.pth", weights_only=False)
    orig_clop.load_state_dict(orig_clop_ckpt["model_state_dict"])
    orig_clop.eval()
    with torch.no_grad():
        for name, emb in bert_embs.items():
            proj = orig_clop.project_text(emb.to(DEVICE))
            orig_conds[name] = proj
    
    # Original model has NO standardization — evaluate in original space
    orig_results = evaluate_generation(
        orig_dit, real_emb, orig_conds, cond_dim=256,
        mu=np.zeros(512), sigma=np.ones(512)  # identity transform
    )
    results["original"] = orig_results
    
    print(f"  Original 200-epoch DiT:")
    print(f"    FD:     {orig_results['overall']['frechet_distance']:.3f}")
    print(f"    Cov:    {orig_results['overall']['coverage']:.3f}")
    print(f"    Dens:   {orig_results['overall']['density']:.3f}")
    print(f"    Uncond FD: {orig_results['unconditional']['frechet_distance']:.3f}")
    print(f"    Controllability: {orig_results['type_controllability']:.4f}")
    
    del orig_dit, orig_clop
    torch.cuda.empty_cache()
    
    # ==================================================================
    #  Summary Table
    # ==================================================================
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    
    def fd(r):
        return r["overall"]["frechet_distance"]
    def cov(r):
        return r["overall"]["coverage"]
    def dens(r):
        return r["overall"]["density"]
    def ufd(r):
        return r["unconditional"]["frechet_distance"]
    def ctrl(r):
        return r["type_controllability"]
    
    print(f"\n{'Method':<40} {'FD↓':>8} {'Cov↑':>8} {'Dens↑':>8} "
          f"{'Uncond FD':>10} {'Ctrl↑':>8}")
    print("-" * 85)
    print(f"{'Original CLOP→DiT (200ep, no std)':<40} {fd(orig_results):8.3f} "
          f"{cov(orig_results):8.3f} {dens(orig_results):8.3f} "
          f"{ufd(orig_results):10.3f} {ctrl(orig_results):8.4f}")
    print(f"{'Tier 1: BiomedBERT→DiT (200ep + std)':<40} {fd(t1_results):8.3f} "
          f"{cov(t1_results):8.3f} {dens(t1_results):8.3f} "
          f"{ufd(t1_results):10.3f} {ctrl(t1_results):8.4f}")
    print(f"{'Tier 2: Fixed CLOP→DiT (200ep + std)':<40} {fd(t2_results):8.3f} "
          f"{cov(t2_results):8.3f} {dens(t2_results):8.3f} "
          f"{ufd(t2_results):10.3f} {ctrl(t2_results):8.4f}")
    
    # Key analysis
    print(f"\n  KEY QUESTION: Does conditioning HELP?")
    print(f"    Original: Cond FD={fd(orig_results):.3f} vs Uncond FD={ufd(orig_results):.3f}"
          f" → {'HELPS' if fd(orig_results) < ufd(orig_results) else 'HURTS'}")
    print(f"    Tier 1:   Cond FD={fd(t1_results):.3f} vs Uncond FD={ufd(t1_results):.3f}"
          f" → {'HELPS' if fd(t1_results) < ufd(t1_results) else 'HURTS'}")
    print(f"    Tier 2:   Cond FD={fd(t2_results):.3f} vs Uncond FD={ufd(t2_results):.3f}"
          f" → {'HELPS' if fd(t2_results) < ufd(t2_results) else 'HURTS'}")
    
    # Save
    # Convert numpy types for JSON serialization
    def numpy_convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    save_path = Path("results/v5_final/tier_comparison_v2.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=numpy_convert)
    print(f"\n  Results saved to {save_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
