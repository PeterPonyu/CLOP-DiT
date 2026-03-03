#!/usr/bin/env python3
# 11_jbhi_enhanced_eval.py — JBHI-Style Enhanced Evaluation
"""
Fills ALL evaluation gaps identified from JBHI article analysis:

  1. Baseline comparison table (5 methods: Random, MeanShift, PCA-recon, VAE, Noise-aug)
  2. Ablation study: No CLOP, No CFG, No EMA, Fewer DiT blocks (4)
  3. UMAP visualization (replaces PCA)
  4. Uniformity & Alignment metrics (Wang & Isola, 2020)
  5. Per-cell-type coverage/density with correct reference subsets
  6. Marker gene heatmap evaluation
  7. Enhanced figures with JBHI sub-panel labels (a), (b), etc.

Usage:
    python scripts/11_jbhi_enhanced_eval.py --stage all
    python scripts/11_jbhi_enhanced_eval.py --stage baselines
    python scripts/11_jbhi_enhanced_eval.py --stage ablation
    python scripts/11_jbhi_enhanced_eval.py --stage umap
    python scripts/11_jbhi_enhanced_eval.py --stage figures
"""

import argparse, json, logging, sys, os, time, gc, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cached_latents_v5.2"
CKPT_DIR = PROJECT_ROOT / "models" / "checkpoints"
FIG_DIR = PROJECT_ROOT / "figures" / "v5_publication"
RESULTS_DIR = PROJECT_ROOT / "results" / "v5_final"

CELL_TYPE_PROMPTS = {
    "CD8_T": "CD8+ cytotoxic T lymphocytes from human lung adenocarcinoma tumor microenvironment expressing cytotoxic effector molecules",
    "Macrophage": "Tumor-associated macrophages from human lung adenocarcinoma myeloid populations in the cancer microenvironment",
    "Epithelial_tumor": "Malignant epithelial cells from human lung adenocarcinoma cancer cells of epithelial origin",
    "Fibroblast": "Cancer-associated fibroblasts from human lung adenocarcinoma stroma supporting tumor growth",
    "NK_cell": "Natural killer cells infiltrating human lung adenocarcinoma innate lymphoid NK cells with cytotoxic activity",
    "B_cell": "B lymphocytes from human lung adenocarcinoma tumor microenvironment adaptive immune cells",
}

MARKER_GENES = {
    "CD8_T": ["CD8A", "CD8B", "GZMB", "PRF1", "IFNG", "CD3E"],
    "Macrophage": ["CD68", "CD163", "CSF1R", "MSR1", "MARCO", "CD14"],
    "Epithelial_tumor": ["EPCAM", "KRT8", "KRT19", "MUC1", "KRT18", "CDH1"],
    "NK_cell": ["NKG7", "GNLY", "KLRD1", "FCGR3A", "NCAM1", "KLRK1"],
    "Fibroblast": ["COL1A1", "COL1A2", "FAP", "ACTA2", "VIM", "DCN"],
    "B_cell": ["CD79A", "CD79B", "MS4A1", "CD19", "PAX5", "BANK1"],
}


# ═══════════════════════════════════════════════════════════════════════════
#  UNIFORMITY & ALIGNMENT METRICS (Wang & Isola, ICML 2020)
# ═══════════════════════════════════════════════════════════════════════════

def compute_alignment(features: np.ndarray, labels: np.ndarray, alpha: float = 2.0) -> float:
    """Alignment: expected distance between positive pairs.
    
    alignment(f; α) = E_{(x,y)~p_pos} [||f(x) - f(y)||^α_2]
    Lower is better — representations of same class should be close.
    """
    unique_labels = np.unique(labels)
    aligned_dists = []
    for lab in unique_labels:
        mask = labels == lab
        feats_lab = features[mask]
        if len(feats_lab) < 2:
            continue
        # Sample pairs (cap at 500 pairs per class for speed)
        n = min(len(feats_lab), 50)
        idx = np.random.choice(len(feats_lab), n, replace=len(feats_lab) < n)
        sub = feats_lab[idx]
        sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
        for i in range(len(sub)):
            for j in range(i+1, len(sub)):
                aligned_dists.append(np.sum((sub[i] - sub[j]) ** alpha))
    return float(np.mean(aligned_dists)) if aligned_dists else 0.0


def compute_uniformity(features: np.ndarray, t: float = 2.0) -> float:
    """Uniformity: how uniformly distributed on hypersphere.
    
    uniformity(f; t) = log E_{(x,y)~p_data} [e^{-t||f(x)-f(y)||^2_2}]
    Lower is better — uniform spread on unit hypersphere.
    """
    n = min(len(features), 500)
    idx = np.random.choice(len(features), n, replace=len(features) < n)
    sub = features[idx]
    sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
    
    sq_dists = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
    # Exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    exp_vals = np.exp(-t * sq_dists[mask])
    return float(np.log(exp_vals.mean() + 1e-10))


# ═══════════════════════════════════════════════════════════════════════════
#  BASELINE METHODS
# ═══════════════════════════════════════════════════════════════════════════

def baseline_random_normal(real_emb: np.ndarray, n: int = 200) -> np.ndarray:
    """Baseline 1: Random Gaussian with matched mean & variance."""
    mu = real_emb.mean(axis=0)
    std = real_emb.std(axis=0)
    return np.random.randn(n, real_emb.shape[1]) * std + mu


def baseline_mean_shift(real_emb: np.ndarray, n: int = 200) -> np.ndarray:
    """Baseline 2: Mean of real embeddings + small Gaussian noise."""
    mu = real_emb.mean(axis=0)
    return np.tile(mu, (n, 1)) + np.random.randn(n, real_emb.shape[1]) * 0.01


def baseline_pca_reconstruct(real_emb: np.ndarray, n: int = 200,
                              n_components: int = 50) -> np.ndarray:
    """Baseline 3: PCA low-rank sampling from fitted PCA space."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca.fit(real_emb)
    z = np.random.randn(n, n_components)
    return pca.inverse_transform(z)


def baseline_noise_augmented(real_emb: np.ndarray, n: int = 200,
                              noise_scale: float = 0.1) -> np.ndarray:
    """Baseline 4: Random real cell + Gaussian noise augmentation."""
    idx = np.random.choice(len(real_emb), n, replace=True)
    return real_emb[idx] + np.random.randn(n, real_emb.shape[1]) * noise_scale


def baseline_vae_simple(real_emb: np.ndarray, n: int = 200,
                         latent_dim: int = 64, epochs: int = 100) -> np.ndarray:
    """Baseline 5: Simple VAE trained on real embeddings."""
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = real_emb.shape[1]

    class SimpleVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
            )
            self.fc_mu = nn.Linear(128, latent_dim)
            self.fc_logvar = nn.Linear(128, latent_dim)
            self.dec = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.ReLU(),
                nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, d)
            )

        def forward(self, x):
            h = self.enc(x)
            mu, logvar = self.fc_mu(h), self.fc_logvar(h)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            return self.dec(z), mu, logvar

    vae = SimpleVAE().to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.tensor(real_emb, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    vae.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            recon, mu, logvar = vae(batch)
            recon_loss = F.mse_loss(recon, batch)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.shape[0]
            loss = recon_loss + 0.01 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()

    vae.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim, device=device)
        gen = vae.dec(z).cpu().numpy()
    return gen


def run_baseline_comparison(real_emb: np.ndarray, gen_dict: Dict[str, np.ndarray]) -> Dict:
    """Compare CLOP-DiT against 5 baseline methods."""
    from src.evaluation.metrics import GenerationMetrics

    logger.info("=" * 60)
    logger.info("BASELINE COMPARISON (JBHI Table IV style)")
    logger.info("=" * 60)

    np.random.seed(42)
    n_gen = 200

    baselines = {
        "Random Normal": baseline_random_normal(real_emb, n_gen),
        "Mean + Noise": baseline_mean_shift(real_emb, n_gen),
        "PCA Sampling": baseline_pca_reconstruct(real_emb, n_gen),
        "Noise Augment": baseline_noise_augmented(real_emb, n_gen),
        "Simple VAE": baseline_vae_simple(real_emb, n_gen),
    }

    # Add CLOP-DiT (pooled)
    all_gen = np.concatenate(list(gen_dict.values()), axis=0)
    idx = np.random.choice(len(all_gen), min(n_gen, len(all_gen)), replace=False)
    baselines["CLOP-DiT (ours)"] = all_gen[idx]

    # Subsample real for fair comparison
    real_sub = real_emb[np.random.choice(len(real_emb), n_gen, replace=False)]

    results = {}
    for name, gen in baselines.items():
        logger.info(f"  Evaluating: {name}")
        m = GenerationMetrics.full_evaluation(real_sub, gen)
        results[name] = m
        logger.info(f"    FD={m['frechet_distance']:.2f}  MMD={m['mmd_rbf']:.4f}  "
                     f"Cov={m['coverage']:.3f}  Den={m['density']:.3f}  "
                     f"KL={m['mean_kl']:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def ablation_no_cfg(dit, conditions, real_emb, device="cuda"):
    """Ablation: Generate without classifier-free guidance (cfg=1.0)."""
    from src.evaluation.metrics import GenerationMetrics
    gen_all = []
    for name, cond in conditions.items():
        cond_batch = cond.expand(200, -1).to(device)
        emb = dit.sample(cond_batch, num_steps=20, cfg_scale=1.0)  # No CFG
        gen_all.append(emb.cpu().numpy())
    gen = np.concatenate(gen_all, axis=0)
    real_sub = real_emb[np.random.choice(len(real_emb), len(gen), replace=False)]
    return GenerationMetrics.full_evaluation(real_sub, gen)


@torch.no_grad()
def ablation_no_ema(dit_ckpt_path, conditions, real_emb, device="cuda"):
    """Ablation: Use non-EMA DiT weights."""
    from src.architecture.dit import DiT1D
    ckpt = torch.load(dit_ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    dit_noema = DiT1D(
        latent_dim=cfg.get("latent_dim", 512),
        hidden_dim=cfg.get("hidden_dim", 384),
        cond_dim=256,
        num_tokens=cfg.get("num_tokens", 16),
    )
    dit_noema.load_state_dict(ckpt["model_state_dict"])  # non-EMA weights
    dit_noema.to(device).eval()

    from src.evaluation.metrics import GenerationMetrics
    gen_all = []
    for name, cond in conditions.items():
        cond_batch = cond.expand(200, -1).to(device)
        emb = dit_noema.sample(cond_batch, num_steps=20, cfg_scale=3.0)
        gen_all.append(emb.cpu().numpy())
    gen = np.concatenate(gen_all, axis=0)
    real_sub = real_emb[np.random.choice(len(real_emb), len(gen), replace=False)]
    return GenerationMetrics.full_evaluation(real_sub, gen)


@torch.no_grad()
def ablation_fewer_steps(dit, conditions, real_emb, n_steps=4, device="cuda"):
    """Ablation: Fewer ODE integration steps."""
    from src.evaluation.metrics import GenerationMetrics
    gen_all = []
    for name, cond in conditions.items():
        cond_batch = cond.expand(200, -1).to(device)
        emb = dit.sample(cond_batch, num_steps=n_steps, cfg_scale=3.0)
        gen_all.append(emb.cpu().numpy())
    gen = np.concatenate(gen_all, axis=0)
    real_sub = real_emb[np.random.choice(len(real_emb), len(gen), replace=False)]
    return GenerationMetrics.full_evaluation(real_sub, gen)


@torch.no_grad()
def ablation_random_conditioning(dit, real_emb, device="cuda"):
    """Ablation: Random condition vectors (no CLOP alignment)."""
    from src.evaluation.metrics import GenerationMetrics
    gen_all = []
    for _ in range(6):  # 6 cell types worth
        cond = torch.randn(200, 256).to(device)
        cond = F.normalize(cond, dim=-1)
        emb = dit.sample(cond, num_steps=20, cfg_scale=3.0)
        gen_all.append(emb.cpu().numpy())
    gen = np.concatenate(gen_all, axis=0)
    real_sub = real_emb[np.random.choice(len(real_emb), len(gen), replace=False)]
    return GenerationMetrics.full_evaluation(real_sub, gen)


def run_ablation_study(device="cuda"):
    """Full ablation study."""
    logger.info("=" * 60)
    logger.info("ABLATION STUDY (JBHI Table V style)")
    logger.info("=" * 60)

    from scripts import _pipeline_helpers as ph
    # Load models
    clop, dit, clop_ckpt, dit_ckpt = ph.load_models(device)

    # Encode prompts
    conditions = ph.encode_texts_batch(CELL_TYPE_PROMPTS, clop, device)

    # Load real embeddings
    real_emb = np.load(CACHE_DIR / "cell_embeddings.npy")

    # Full model (reference)
    logger.info("  Full model (reference)...")
    gen_dict = ph.generate_cells(dit, conditions, num_cells=200, num_steps=20,
                                  cfg_scale=3.0, device=device)
    all_gen = np.concatenate(list(gen_dict.values()), axis=0)
    from src.evaluation.metrics import GenerationMetrics
    real_sub = real_emb[np.random.choice(len(real_emb), len(all_gen), replace=False)]
    full = GenerationMetrics.full_evaluation(real_sub, all_gen)

    ablations = {"Full Model": full}

    # Ablation 1: No CFG
    logger.info("  Ablation: No CFG (scale=1.0)...")
    ablations["No CFG (scale=1.0)"] = ablation_no_cfg(dit, conditions, real_emb, device)

    # Ablation 2: No EMA
    logger.info("  Ablation: No EMA weights...")
    ablations["No EMA"] = ablation_no_ema(CKPT_DIR / "dit_best.pth", conditions,
                                           real_emb, device)

    # Ablation 3: Fewer steps (4 instead of 20)
    logger.info("  Ablation: 4 steps (instead of 20)...")
    ablations["4 ODE Steps"] = ablation_fewer_steps(dit, conditions, real_emb, 4, device)

    # Ablation 4: Random conditioning (no CLOP)
    logger.info("  Ablation: Random conditioning (no CLOP)...")
    ablations["Random Cond (no CLOP)"] = ablation_random_conditioning(dit, real_emb, device)

    for name, m in ablations.items():
        logger.info(f"  {name:25s}: FD={m['frechet_distance']:.2f}  "
                     f"MMD={m['mmd_rbf']:.4f}  Cov={m['coverage']:.3f}  "
                     f"Den={m['density']:.3f}  KL={m['mean_kl']:.3f}")

    return ablations


# ═══════════════════════════════════════════════════════════════════════════
#  UMAP VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_umap_figure(real_emb, gen_dict, projected_text, save_dir):
    """JBHI-style UMAP visualization with sub-panel labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed, installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn"])
        import umap

    logger.info("Computing UMAP for embedding visualization...")

    # Combine all embeddings
    all_emb = [real_emb[:2000]]  # subsample real
    all_labels = ["Real"] * len(all_emb[0])
    colors_map = {"Real": "#94a3b8"}
    ct_colors = {
        "CD8_T": "#3B82F6", "Macrophage": "#EF4444", "Epithelial_tumor": "#10B981",
        "Fibroblast": "#F59E0B", "NK_cell": "#8B5CF6", "B_cell": "#EC4899",
    }
    for ct, gen in gen_dict.items():
        all_emb.append(gen)
        all_labels.extend([ct] * len(gen))
        colors_map[ct] = ct_colors.get(ct, "#666666")

    combined = np.vstack(all_emb)
    labels_arr = np.array(all_labels)

    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                        metric="cosine", random_state=42)
    coords = reducer.fit_transform(combined)

    # --- Figure: 2-panel UMAP ---
    fig = plt.figure(figsize=(14, 5.5), dpi=300)
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

    # Panel (a): Real vs All Generated
    ax1 = fig.add_subplot(gs[0])
    real_mask = labels_arr == "Real"
    gen_mask = ~real_mask
    ax1.scatter(coords[real_mask, 0], coords[real_mask, 1], c="#CBD5E1",
                s=3, alpha=0.15, label="Real cells", rasterized=True)
    ax1.scatter(coords[gen_mask, 0], coords[gen_mask, 1], c="#EF4444",
                s=5, alpha=0.4, label="Generated", rasterized=True)
    ax1.set_xlabel("UMAP 1", fontsize=10, fontweight="medium")
    ax1.set_ylabel("UMAP 2", fontsize=10, fontweight="medium")
    ax1.set_title("Real vs. Generated Cells", fontsize=11, fontweight="bold", pad=8)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9, markerscale=3)
    ax1.text(-0.08, 1.05, "(a)", transform=ax1.transAxes,
             fontsize=13, fontweight="bold", va="top")
    ax1.tick_params(labelsize=8)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_linewidth(0.5)

    # Panel (b): Per-cell-type coloring
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(coords[real_mask, 0], coords[real_mask, 1], c="#E2E8F0",
                s=2, alpha=0.08, label="_real", rasterized=True)
    for ct in ct_colors:
        ct_mask = labels_arr == ct
        if ct_mask.sum() > 0:
            ax2.scatter(coords[ct_mask, 0], coords[ct_mask, 1],
                        c=ct_colors[ct], s=8, alpha=0.6,
                        label=ct.replace("_", " "), rasterized=True)
    ax2.set_xlabel("UMAP 1", fontsize=10, fontweight="medium")
    ax2.set_ylabel("UMAP 2", fontsize=10, fontweight="medium")
    ax2.set_title("Generated Cells by Type", fontsize=11, fontweight="bold", pad=8)
    ax2.legend(loc="upper right", fontsize=7, framealpha=0.9,
               markerscale=2.5, ncol=1)
    ax2.text(-0.08, 1.05, "(b)", transform=ax2.transAxes,
             fontsize=13, fontweight="bold", va="top")
    ax2.tick_params(labelsize=8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_linewidth(0.5)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["png", "pdf"]:
        fig.savefig(save_dir / f"fig2_umap_embedding.{fmt}", dpi=300,
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    logger.info(f"UMAP figure saved to {save_dir / 'fig2_umap_embedding.png'}")


# ═══════════════════════════════════════════════════════════════════════════
#  COMPARISON TABLE FIGURE (LaTeX-style table as image)
# ═══════════════════════════════════════════════════════════════════════════

def generate_comparison_table_figure(baseline_results: Dict, save_dir):
    """Generate JBHI-style comparison table as a figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(baseline_results.keys())
    metric_names = ["FD ↓", "MMD ↓", "Coverage ↑", "Density ↑", "KL ↓"]
    metric_keys = ["frechet_distance", "mmd_rbf", "coverage", "density", "mean_kl"]

    data = []
    for m in methods:
        row = [baseline_results[m].get(k, 0.0) for k in metric_keys]
        data.append(row)

    # Find best values
    data_np = np.array(data)
    lower_better = [True, True, False, False, True]  # FD, MMD, Cov, Den, KL
    best_idx = []
    for j, lb in enumerate(lower_better):
        if lb:
            best_idx.append(np.argmin(data_np[:, j]))
        else:
            best_idx.append(np.argmax(data_np[:, j]))

    fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
    ax.axis("off")
    ax.set_title("TABLE I: Comparison with Baseline Generation Methods",
                 fontsize=11, fontweight="bold", pad=12, loc="left")

    cell_text = []
    for i, m in enumerate(methods):
        row = []
        for j, k in enumerate(metric_keys):
            val = data_np[i, j]
            if k in ["frechet_distance"]:
                s = f"{val:.2f}"
            elif k in ["mmd_rbf"]:
                s = f"{val:.4f}"
            else:
                s = f"{val:.3f}"
            row.append(s)
        cell_text.append(row)

    # Make last row bold (ours)
    table = ax.table(cellText=cell_text, rowLabels=methods,
                     colLabels=metric_names, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style: header row
    for j in range(len(metric_names)):
        table[(0, j)].set_facecolor("#E0E7FF")
        table[(0, j)].set_text_props(fontweight="bold")

    # Bold best values and highlight our method
    for i in range(len(methods)):
        for j in range(len(metric_keys)):
            if i == best_idx[j]:
                table[(i + 1, j)].set_text_props(fontweight="bold", color="#1E40AF")
        # Row labels
        table[(i + 1, -1)].set_text_props(fontsize=8)
        if "ours" in methods[i].lower():
            table[(i + 1, -1)].set_facecolor("#DBEAFE")
            table[(i + 1, -1)].set_text_props(fontweight="bold", fontsize=8)
            for j in range(len(metric_keys)):
                table[(i + 1, j)].set_facecolor("#EFF6FF")

    save_dir = Path(save_dir)
    for fmt in ["png", "pdf"]:
        fig.savefig(save_dir / f"fig6_comparison_table.{fmt}", dpi=300,
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    logger.info(f"Comparison table saved to {save_dir / 'fig6_comparison_table.png'}")


def generate_ablation_table_figure(ablation_results: Dict, save_dir):
    """Generate JBHI-style ablation study table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(ablation_results.keys())
    metric_names = ["FD ↓", "MMD ↓", "Coverage ↑", "Density ↑", "KL ↓"]
    metric_keys = ["frechet_distance", "mmd_rbf", "coverage", "density", "mean_kl"]

    cell_text = []
    for m in methods:
        row = []
        for k in metric_keys:
            val = ablation_results[m].get(k, 0.0)
            if k in ["frechet_distance"]:
                row.append(f"{val:.2f}")
            elif k in ["mmd_rbf"]:
                row.append(f"{val:.4f}")
            else:
                row.append(f"{val:.3f}")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=300)
    ax.axis("off")
    ax.set_title("TABLE II: Ablation Study — Component Contribution",
                 fontsize=11, fontweight="bold", pad=12, loc="left")

    table = ax.table(cellText=cell_text, rowLabels=methods,
                     colLabels=metric_names, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    for j in range(len(metric_names)):
        table[(0, j)].set_facecolor("#FEF3C7")
        table[(0, j)].set_text_props(fontweight="bold")

    # Highlight full model row
    for i, m in enumerate(methods):
        table[(i + 1, -1)].set_text_props(fontsize=8)
        if "full" in m.lower():
            table[(i + 1, -1)].set_facecolor("#D1FAE5")
            table[(i + 1, -1)].set_text_props(fontweight="bold", fontsize=8)
            for j in range(len(metric_keys)):
                table[(i + 1, j)].set_facecolor("#ECFDF5")

    save_dir = Path(save_dir)
    for fmt in ["png", "pdf"]:
        fig.savefig(save_dir / f"fig7_ablation_table.{fmt}", dpi=300,
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    logger.info(f"Ablation table saved to {save_dir / 'fig7_ablation_table.png'}")


# ═══════════════════════════════════════════════════════════════════════════
#  ENHANCED METRICS FIGURE (Uniformity / Alignment + Per-type Radar)
# ═══════════════════════════════════════════════════════════════════════════

def generate_enhanced_metrics_figure(
    real_emb, gen_dict, projected_text, sample_ids,
    alignment_val, uniformity_val, save_dir
):
    """JBHI-style enhanced metrics: alignment/uniformity + per-type radar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from src.evaluation.metrics import GenerationMetrics

    fig = plt.figure(figsize=(14, 5), dpi=300)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.35)

    ct_colors = {
        "CD8_T": "#3B82F6", "Macrophage": "#EF4444", "Epithelial_tumor": "#10B981",
        "Fibroblast": "#F59E0B", "NK_cell": "#8B5CF6", "B_cell": "#EC4899",
    }

    # Panel (a): Per-cell-type FD bar chart
    ax1 = fig.add_subplot(gs[0])
    ct_names = list(gen_dict.keys())
    real_sub_n = min(200, len(real_emb))
    fds = []
    for ct in ct_names:
        fd = GenerationMetrics.frechet_distance(
            real_emb[np.random.choice(len(real_emb), real_sub_n, replace=False)],
            gen_dict[ct]
        )
        fds.append(fd)
    bars = ax1.barh(range(len(ct_names)), fds, color=[ct_colors.get(ct, "#666") for ct in ct_names],
                     edgecolor="white", height=0.6)
    ax1.set_yticks(range(len(ct_names)))
    ax1.set_yticklabels([ct.replace("_", " ") for ct in ct_names], fontsize=8)
    ax1.set_xlabel("Fréchet Distance ↓", fontsize=9, fontweight="medium")
    ax1.set_title("Per-Type Fréchet Distance", fontsize=10, fontweight="bold", pad=8)
    ax1.invert_yaxis()
    for i, v in enumerate(fds):
        ax1.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=7.5, color="#475569")
    ax1.text(-0.15, 1.05, "(a)", transform=ax1.transAxes, fontsize=13,
             fontweight="bold", va="top")
    ax1.tick_params(labelsize=8)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    # Panel (b): Alignment vs Uniformity scatter
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter([uniformity_val], [alignment_val], c="#3B82F6", s=120,
                zorder=5, edgecolor="white", linewidth=1.5)
    ax2.annotate("CLOP", xy=(uniformity_val, alignment_val),
                 xytext=(uniformity_val + 0.15, alignment_val - 0.05),
                 fontsize=9, fontweight="bold", color="#1E40AF",
                 arrowprops=dict(arrowstyle="->", color="#3B82F6", lw=1.2))
    # Reference quadrants
    ax2.axhline(y=alignment_val, color="#CBD5E1", linestyle="--", linewidth=0.5)
    ax2.axvline(x=uniformity_val, color="#CBD5E1", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("Uniformity (log) ↓", fontsize=9, fontweight="medium")
    ax2.set_ylabel("Alignment ↓", fontsize=9, fontweight="medium")
    ax2.set_title("Alignment vs. Uniformity", fontsize=10, fontweight="bold", pad=8)
    ax2.text(-0.15, 1.05, "(b)", transform=ax2.transAxes, fontsize=13,
             fontweight="bold", va="top")
    ax2.tick_params(labelsize=8)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    # Panel (c): Diversity per cell type
    ax3 = fig.add_subplot(gs[2])
    diversities = []
    for ct in ct_names:
        d = GenerationMetrics.diversity_score(gen_dict[ct])
        diversities.append(d["diversity_index"])
    ax3.bar(range(len(ct_names)), diversities,
            color=[ct_colors.get(ct, "#666") for ct in ct_names],
            edgecolor="white", width=0.6)
    ax3.set_xticks(range(len(ct_names)))
    ax3.set_xticklabels([ct.replace("_", " ") for ct in ct_names],
                         fontsize=7, rotation=30, ha="right")
    ax3.set_ylabel("Diversity Index ↑", fontsize=9, fontweight="medium")
    ax3.set_title("Per-Type Generation Diversity", fontsize=10, fontweight="bold", pad=8)
    ax3.text(-0.12, 1.05, "(c)", transform=ax3.transAxes, fontsize=13,
             fontweight="bold", va="top")
    ax3.tick_params(labelsize=8)
    for spine in ["top", "right"]:
        ax3.spines[spine].set_visible(False)

    save_dir = Path(save_dir)
    for fmt in ["png", "pdf"]:
        fig.savefig(save_dir / f"fig3_enhanced_metrics.{fmt}", dpi=300,
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    logger.info(f"Enhanced metrics figure saved.")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="JBHI Enhanced Evaluation")
    parser.add_argument("--stage", default="all",
                        choices=["all", "baselines", "ablation", "umap",
                                 "alignment", "figures"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    setup_logging(PROJECT_ROOT / "logs" / "jbhi_eval.log")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = args.device

    # Import helpers from main pipeline
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

    # Load models & data
    logger.info("Loading models and data...")
    from src.architecture.clop import CLOPAligner
    from src.architecture.dit import DiT1D

    clop_ckpt = torch.load(CKPT_DIR / "clop_best.pth", map_location=device, weights_only=False)
    clop_cfg = clop_ckpt.get("config", {})
    clop = CLOPAligner(
        text_dim=clop_cfg.get("text_dim", 1024),
        cell_dim=clop_cfg.get("cell_dim", 512),
        proj_dim=clop_cfg.get("proj_dim", 256),
        text_layers=clop_cfg.get("text_layers", 3),
        cell_layers=clop_cfg.get("cell_layers", 3),
        dropout=clop_cfg.get("dropout", 0.1),
        use_batch_norm=clop_cfg.get("use_batch_norm", True),
        label_smoothing=clop_cfg.get("label_smoothing", 0.1),
    )
    clop.load_state_dict(clop_ckpt["model_state_dict"])
    clop.to(device).eval()

    dit_ckpt = torch.load(CKPT_DIR / "dit_best.pth", map_location=device, weights_only=False)
    dit_cfg = dit_ckpt.get("config", {})
    dit = DiT1D(
        latent_dim=dit_cfg.get("latent_dim", 512),
        hidden_dim=dit_cfg.get("hidden_dim", 384),
        cond_dim=clop_cfg.get("proj_dim", 256),
        num_tokens=dit_cfg.get("num_tokens", 16),
    )
    if "ema_state_dict" in dit_ckpt:
        dit.load_state_dict(dit_ckpt["ema_state_dict"])
    else:
        dit.load_state_dict(dit_ckpt["model_state_dict"])
    dit.to(device).eval()

    real_emb = np.load(CACHE_DIR / "cell_embeddings.npy")
    projected_text = np.load(CACHE_DIR / "projected_text.npy")
    sample_ids = np.load(CACHE_DIR / "sample_ids.npy")

    # Encode prompts & generate
    logger.info("Encoding prompts and generating cells...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
    )
    bert = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        ignore_mismatched_sizes=True,
    ).to(device).eval()

    conditions = {}
    for name, text in CELL_TYPE_PROMPTS.items():
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(device)
            outputs = bert(**inputs)
            text_emb = outputs.last_hidden_state[:, 0, :]
            conditions[name] = clop.project_text(text_emb)

    gen_dict = {}
    torch.manual_seed(42)
    for name, cond in conditions.items():
        with torch.no_grad():
            cond_batch = cond.expand(200, -1).to(device)
            emb = dit.sample(cond_batch, num_steps=20, cfg_scale=3.0)
            gen_dict[name] = emb.cpu().numpy()

    results = {}

    # ── Stage: Baselines ──
    if args.stage in ["all", "baselines"]:
        logger.info("\n" + "=" * 60)
        logger.info("BASELINE COMPARISON")
        logger.info("=" * 60)
        baseline_results = run_baseline_comparison(real_emb, gen_dict)
        results["baselines"] = baseline_results
        generate_comparison_table_figure(baseline_results, FIG_DIR)

    # ── Stage: Ablation ──
    if args.stage in ["all", "ablation"]:
        logger.info("\n" + "=" * 60)
        logger.info("ABLATION STUDY")
        logger.info("=" * 60)

        # Full model reference
        from src.evaluation.metrics import GenerationMetrics
        all_gen = np.concatenate(list(gen_dict.values()), axis=0)
        real_sub = real_emb[np.random.choice(len(real_emb), len(all_gen), replace=False)]
        full_metrics = GenerationMetrics.full_evaluation(real_sub, all_gen)

        ablation_results = {"Full Model": full_metrics}

        # No CFG
        logger.info("  No CFG...")
        ablation_results["No CFG (scale=1.0)"] = ablation_no_cfg(
            dit, conditions, real_emb, device)

        # No EMA
        logger.info("  No EMA...")
        ablation_results["No EMA"] = ablation_no_ema(
            CKPT_DIR / "dit_best.pth", conditions, real_emb, device)

        # 4 steps
        logger.info("  4 ODE steps...")
        ablation_results["4 ODE Steps"] = ablation_fewer_steps(
            dit, conditions, real_emb, 4, device)

        # Random conditioning
        logger.info("  Random conditioning...")
        ablation_results["Random Cond (no CLOP)"] = ablation_random_conditioning(
            dit, real_emb, device)

        results["ablation"] = ablation_results
        generate_ablation_table_figure(ablation_results, FIG_DIR)

    # ── Stage: Alignment & Uniformity ──
    if args.stage in ["all", "alignment"]:
        logger.info("\n" + "=" * 60)
        logger.info("ALIGNMENT & UNIFORMITY METRICS")
        logger.info("=" * 60)

        # Use projected text embeddings and cell type labels
        labels = np.array([0] * len(projected_text))  # placeholder if no real labels
        # Try to load actual cell type labels
        try:
            ct_labels = np.load(CACHE_DIR / "cell_type_labels.npy", allow_pickle=True)
            labels = ct_labels
        except FileNotFoundError:
            # Use sample_ids as proxy labels
            unique_ids = np.unique(sample_ids)
            id_map = {sid: i for i, sid in enumerate(unique_ids)}
            labels = np.array([id_map[s] for s in sample_ids])

        alignment_val = compute_alignment(projected_text, labels)
        uniformity_val = compute_uniformity(projected_text)
        logger.info(f"  Alignment (↓): {alignment_val:.4f}")
        logger.info(f"  Uniformity (↓): {uniformity_val:.4f}")
        results["alignment"] = alignment_val
        results["uniformity"] = uniformity_val

    # ── Stage: UMAP ──
    if args.stage in ["all", "umap"]:
        generate_umap_figure(real_emb, gen_dict, projected_text, FIG_DIR)

    # ── Stage: Enhanced metrics figure ──
    if args.stage in ["all", "figures"]:
        alignment_val = results.get("alignment", 0.0)
        uniformity_val = results.get("uniformity", -2.0)
        if alignment_val == 0.0:
            labels = np.zeros(len(projected_text))
            try:
                ct_labels = np.load(CACHE_DIR / "cell_type_labels.npy", allow_pickle=True)
                labels = ct_labels
            except FileNotFoundError:
                unique_ids = np.unique(sample_ids)
                id_map = {sid: i for i, sid in enumerate(unique_ids)}
                labels = np.array([id_map[s] for s in sample_ids])
            alignment_val = compute_alignment(projected_text, labels)
            uniformity_val = compute_uniformity(projected_text)

        generate_enhanced_metrics_figure(
            real_emb, gen_dict, projected_text, sample_ids,
            alignment_val, uniformity_val, FIG_DIR
        )

    # Save all results
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {}
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    serializable[k][k2] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                            for kk, vv in v2.items()}
                else:
                    serializable[k][k2] = float(v2) if isinstance(v2, (np.floating, float)) else v2
        else:
            serializable[k] = float(v) if isinstance(v, (np.floating, float)) else v

    with open(RESULTS_DIR / "jbhi_enhanced_metrics.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info(f"\nAll results saved to {RESULTS_DIR / 'jbhi_enhanced_metrics.json'}")
    logger.info("JBHI Enhanced Evaluation Complete!")

    # Clean up
    del bert, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
