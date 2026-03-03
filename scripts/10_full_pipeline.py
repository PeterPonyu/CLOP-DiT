#!/usr/bin/env python3
# 10_full_pipeline.py — Complete CLOP-DiT v6 pipeline
"""
Full pipeline with industrial-grade fixes for alignment training.

v6 changes:
  0. Preprocess embeddings (ZCA whitening) — fixes space collapse
  1. Train CLOP (SigLIP loss + whitened embeddings) → project text embeddings
  2. Train DiT (200 epochs, flow matching)
  3. Generate cells for multiple cell types
  4. Comprehensive evaluation (embedding + biological + robustness)
  5. Generate 5 publication-quality multi-panel figures
  6. Visual conflict detection

Usage:
    python scripts/10_full_pipeline.py --stage all
    python scripts/10_full_pipeline.py --stage preprocess
    python scripts/10_full_pipeline.py --stage train_clop
    python scripts/10_full_pipeline.py --stage train_dit
    python scripts/10_full_pipeline.py --stage evaluate
    python scripts/10_full_pipeline.py --stage figures
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

# ── Cell type prompts (Reviewer concern: multi-cell-type coverage) ──
CELL_TYPE_PROMPTS = {
    "CD8_T": "CD8+ cytotoxic T lymphocytes from human lung adenocarcinoma tumor microenvironment expressing cytotoxic effector molecules",
    "Macrophage": "Tumor-associated macrophages from human lung adenocarcinoma myeloid populations in the cancer microenvironment",
    "Epithelial_tumor": "Malignant epithelial cells from human lung adenocarcinoma cancer cells of epithelial origin",
    "Fibroblast": "Cancer-associated fibroblasts from human lung adenocarcinoma stroma supporting tumor growth",
    "NK_cell": "Natural killer cells infiltrating human lung adenocarcinoma innate lymphoid NK cells with cytotoxic activity",
    "B_cell": "B lymphocytes from human lung adenocarcinoma tumor microenvironment adaptive immune cells",
}

# ── Prompt robustness variants (Reviewer concern: prompt sensitivity) ──
PROMPT_VARIANTS = {
    "CD8_T": [
        "CD8+ cytotoxic T lymphocytes from human lung adenocarcinoma tumor microenvironment expressing cytotoxic effector molecules",
        "CD8 positive T cells in lung cancer tissue with cytotoxic activity",
        "Tumor-infiltrating CD8+ T lymphocytes from pulmonary adenocarcinoma",
        "Cytotoxic T cells expressing CD8A and granzyme B in lung tumor",
    ],
    "Macrophage": [
        "Tumor-associated macrophages from human lung adenocarcinoma myeloid populations in the cancer microenvironment",
        "Macrophage cells in lung cancer expressing CD68 and CD163",
        "Myeloid macrophages infiltrating human pulmonary adenocarcinoma",
        "TAMs from lung tumor microenvironment with M2 polarization markers",
    ],
}

# ── Marker genes (Reviewer concern: biological validation) ──
MARKER_GENES = {
    "CD8_T": ["CD8A", "CD8B", "GZMB", "PRF1", "IFNG", "CD3E"],
    "Macrophage": ["CD68", "CD163", "CSF1R", "MSR1", "MARCO", "CD14"],
    "Epithelial_tumor": ["EPCAM", "KRT8", "KRT19", "MUC1", "KRT18", "CDH1"],
    "NK_cell": ["NKG7", "GNLY", "KLRD1", "FCGR3A", "NCAM1", "KLRK1"],
    "Fibroblast": ["COL1A1", "COL1A2", "FAP", "ACTA2", "VIM", "DCN"],
    "B_cell": ["CD79A", "CD79B", "MS4A1", "CD19", "PAX5", "BANK1"],
}


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: CLOP Training
# ═══════════════════════════════════════════════════════════════════════════

def train_clop(num_epochs=200, resume=False):
    """Train CLOP alignment with monitoring."""
    import yaml
    from src.training.train_clop import CLOPTrainer

    logger.info("=" * 70)
    logger.info("STAGE 1: CLOP Contrastive Alignment Training")
    logger.info("=" * 70)

    config_path = PROJECT_ROOT / "configs" / "clop.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["num_epochs"] = num_epochs
    trainer = CLOPTrainer.from_config(config)

    t0 = time.time()
    history = trainer.train()
    elapsed = time.time() - t0

    logger.info(f"CLOP training complete in {elapsed/3600:.1f}h")
    logger.info(f"  Best val_acc: {max(history['val_acc']):.4f}")
    logger.info(f"  Final temperature: {history['temperature'][-1]:.4f}")

    # Project text embeddings for DiT
    logger.info("Projecting text embeddings for DiT training...")
    trainer.project_and_save(
        output_path=str(CACHE_DIR / "projected_text.npy"),
        use_best=True,
    )

    return history


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: DiT Training
# ═══════════════════════════════════════════════════════════════════════════

def train_dit(num_epochs=200, resume=False):
    """Train DiT flow matching with monitoring."""
    import yaml
    from src.training.train_dit import DiTTrainer

    logger.info("=" * 70)
    logger.info("STAGE 2: DiT Flow Matching Training")
    logger.info("=" * 70)

    config_path = PROJECT_ROOT / "configs" / "dit.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["num_epochs"] = num_epochs
    if resume:
        best_ckpt = CKPT_DIR / "dit_best.pth"
        if best_ckpt.exists():
            config["resume"] = str(best_ckpt)

    trainer = DiTTrainer.from_config(config)

    t0 = time.time()
    history = trainer.train()
    elapsed = time.time() - t0

    logger.info(f"DiT training complete in {elapsed/3600:.1f}h")
    logger.info(f"  Best val_loss: {min(history['val_loss']):.6f}")

    return history


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: Generation + Comprehensive Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_embeddings():
    """Stage 0: Preprocess embeddings with ZCA whitening.

    This is the KEY step that fixes the v5.2 alignment failure.
    Both BiomedBERT text embeddings (cosine=0.96) and scGPT cell embeddings
    (cosine=0.99) are whitened to spread the collapsed spaces.
    """
    from src.data_pipeline.embedding_preprocessor import preprocess_cached_embeddings

    logger.info("=" * 70)
    logger.info("STAGE 0: Embedding Preprocessing (ZCA Whitening)")
    logger.info("=" * 70)

    # Check if already done
    text_pp = CACHE_DIR / "text_embeddings_preprocessed.npy"
    cell_pp = CACHE_DIR / "cell_embeddings_preprocessed.npy"
    if text_pp.exists() and cell_pp.exists():
        logger.info("Preprocessed embeddings already exist — skipping.")
        logger.info(f"  Text: {text_pp}")
        logger.info(f"  Cell: {cell_pp}")
        return

    stats = preprocess_cached_embeddings(
        cache_dir=str(CACHE_DIR),
        text_method="whiten",
        cell_method="whiten",
    )

    logger.info(f"Text cosine: {stats['text']['raw_cosine_mean']:.4f} → "
                f"{stats['text']['processed_cosine_mean']:.4f}")
    logger.info(f"Cell cosine: {stats['cell']['raw_cosine_mean']:.4f} → "
                f"{stats['cell']['processed_cosine_mean']:.4f}")


def load_models(device="cuda"):
    """Load trained CLOP + DiT models (v6-compatible)."""
    from src.architecture.clop import CLOPAligner
    from src.architecture.dit import DiT1D

    # CLOP
    clop_ckpt = torch.load(CKPT_DIR / "clop_best.pth", map_location=device, weights_only=False)
    clop_cfg = clop_ckpt.get("config", {})
    clop = CLOPAligner(
        text_dim=clop_cfg.get("text_dim", 1024),
        cell_dim=clop_cfg.get("cell_dim", 512),
        proj_dim=clop_cfg.get("proj_dim", 512),
        text_layers=clop_cfg.get("text_layers", 3),
        cell_layers=clop_cfg.get("cell_layers", 3),
        dropout=clop_cfg.get("dropout", 0.2),
        use_batch_norm=clop_cfg.get("use_batch_norm", False),
        label_smoothing=clop_cfg.get("label_smoothing", 0.1),
        loss_type=clop_cfg.get("loss_type", "prototype_siglip"),
        auto_duplicate_mask=clop_cfg.get("auto_duplicate_mask", True),
        temperature=clop_cfg.get("temperature", 10.0),
        use_whitening=clop_cfg.get("use_whitening", False),
        cohesion_weight=clop_cfg.get("cohesion_weight", 0.1),
        max_temperature=clop_cfg.get("max_temperature", 100.0),
    )
    clop.load_state_dict(clop_ckpt["model_state_dict"])
    clop.to(device).eval()

    # DiT
    dit_ckpt = torch.load(CKPT_DIR / "dit_best.pth", map_location=device, weights_only=False)
    dit_cfg = dit_ckpt.get("config", {})
    dit = DiT1D(
        latent_dim=dit_cfg.get("latent_dim", 512),
        hidden_dim=dit_cfg.get("hidden_dim", 384),
        cond_dim=clop_cfg.get("proj_dim", 512),
        num_tokens=dit_cfg.get("num_tokens", 16),
    )
    if "ema_state_dict" in dit_ckpt:
        dit.load_state_dict(dit_ckpt["ema_state_dict"])
    else:
        dit.load_state_dict(dit_ckpt["model_state_dict"])
    dit.to(device).eval()

    return clop, dit, clop_ckpt, dit_ckpt


@torch.no_grad()
def encode_text(text: str, clop, device="cuda"):
    """Encode text → CLOP condition vector."""
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
    )
    model = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        ignore_mismatched_sizes=True,
    ).to(device).eval()

    inputs = tokenizer(text, return_tensors="pt", padding=True,
                       truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    text_emb = outputs.last_hidden_state[:, 0, :]
    cond = clop.project_text(text_emb)

    del model, tokenizer
    torch.cuda.empty_cache()
    return cond


@torch.no_grad()
def encode_texts_batch(texts: Dict[str, str], clop, device="cuda"):
    """Encode multiple text prompts → condition vectors."""
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
    )
    model = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        ignore_mismatched_sizes=True,
    ).to(device).eval()

    conditions = {}
    for name, text in texts.items():
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        text_emb = outputs.last_hidden_state[:, 0, :]
        conditions[name] = clop.project_text(text_emb)

    del model, tokenizer
    torch.cuda.empty_cache()
    return conditions


@torch.no_grad()
def generate_cells(dit, conditions, num_cells=200, num_steps=20, cfg_scale=3.0,
                   device="cuda", seed=42):
    """Generate cell embeddings for each condition."""
    torch.manual_seed(seed)
    results = {}
    for name, cond in conditions.items():
        cond_batch = cond.expand(num_cells, -1).to(device)
        emb = dit.sample(cond_batch, num_steps=num_steps, cfg_scale=cfg_scale)
        results[name] = emb.cpu().numpy()
        logger.info(f"Generated {num_cells} cells for {name}: shape={emb.shape}")
    return results


def evaluate_embedding_quality(real_emb, gen_emb_dict, projected_text, sample_ids):
    """Comprehensive embedding-space evaluation."""
    from src.evaluation.metrics import GenerationMetrics
    from scipy import stats

    metrics = {}

    # Per-cell-type generation quality
    for name, gen in gen_emb_dict.items():
        m = GenerationMetrics.full_evaluation(real_emb[:len(gen)], gen)
        for k, v in m.items():
            metrics[f"{name}/{k}"] = v

    # Overall generation quality (pool all generated)
    all_gen = np.concatenate(list(gen_emb_dict.values()), axis=0)
    idx = np.random.choice(len(real_emb), min(len(all_gen), len(real_emb)), replace=False)
    overall = GenerationMetrics.full_evaluation(real_emb[idx], all_gen[:len(idx)])
    for k, v in overall.items():
        metrics[f"overall/{k}"] = v

    return metrics


def evaluate_prompt_robustness(dit, clop, device="cuda"):
    """Evaluate sensitivity to prompt wording variants."""
    logger.info("Evaluating prompt robustness...")
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
    )
    model = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        ignore_mismatched_sizes=True,
    ).to(device).eval()

    robustness = {}
    for ct, variants in PROMPT_VARIANTS.items():
        embeddings = []
        for text in variants:
            inputs = tokenizer(text, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            text_emb = outputs.last_hidden_state[:, 0, :]
            cond = clop.project_text(text_emb)

            # Generate a small set
            cond_batch = cond.expand(50, -1)
            gen = dit.sample(cond_batch, num_steps=20, cfg_scale=3.0)
            embeddings.append(gen.cpu().numpy())

        # Compute pairwise cosine similarity between variant means
        means = [e.mean(axis=0) for e in embeddings]
        sims = []
        for i in range(len(means)):
            for j in range(i+1, len(means)):
                sim = np.dot(means[i], means[j]) / (
                    np.linalg.norm(means[i]) * np.linalg.norm(means[j]) + 1e-8
                )
                sims.append(sim)

        robustness[ct] = {
            "mean_cross_sim": float(np.mean(sims)),
            "std_cross_sim": float(np.std(sims)),
            "min_cross_sim": float(np.min(sims)),
            "n_variants": len(variants),
        }
        logger.info(f"  {ct}: mean_sim={np.mean(sims):.4f} ± {np.std(sims):.4f}")

    del model, tokenizer
    torch.cuda.empty_cache()
    return robustness


def run_full_evaluation(device="cuda"):
    """Complete evaluation pipeline."""
    logger.info("=" * 70)
    logger.info("STAGE 3: Comprehensive Evaluation")
    logger.info("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    clop, dit, clop_ckpt, dit_ckpt = load_models(device)

    # Load cached data
    real_emb = np.load(CACHE_DIR / "cell_embeddings.npy")
    projected_text = np.load(CACHE_DIR / "projected_text.npy")
    sample_ids = np.load(CACHE_DIR / "sample_ids.npy")

    # Encode prompts
    logger.info("Encoding cell type prompts...")
    conditions = encode_texts_batch(CELL_TYPE_PROMPTS, clop, device)

    # Generate cells (Reviewer concern: use consistent num_steps=20)
    logger.info("Generating cells...")
    gen_dict = generate_cells(dit, conditions, num_cells=200, num_steps=20,
                              cfg_scale=3.0, device=device)

    # Save generated embeddings
    for name, emb in gen_dict.items():
        np.save(RESULTS_DIR / f"gen_{name}.npy", emb)

    # Embedding quality
    logger.info("Computing embedding quality metrics...")
    emb_metrics = evaluate_embedding_quality(real_emb, gen_dict, projected_text, sample_ids)

    # Prompt robustness
    robustness = evaluate_prompt_robustness(dit, clop, device)

    # Compile all metrics
    all_metrics = {
        "embedding_quality": emb_metrics,
        "prompt_robustness": robustness,
        "training": {
            "clop_epoch": clop_ckpt.get("epoch", "?"),
            "clop_val_acc": clop_ckpt.get("metrics", {}).get("val_acc", "?"),
            "dit_epoch": dit_ckpt.get("epoch", "?"),
            "dit_val_loss": dit_ckpt.get("metrics", {}).get("val_loss", "?"),
        },
        "generation": {
            "num_steps": 20,
            "cfg_scale": 3.0,
            "num_cells_per_type": 200,
            "cell_types": list(CELL_TYPE_PROMPTS.keys()),
        },
    }

    with open(RESULTS_DIR / "metrics_v5.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    logger.info(f"All metrics saved to {RESULTS_DIR / 'metrics_v5.json'}")

    return all_metrics, gen_dict, real_emb, projected_text, sample_ids


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: Publication-Quality Multi-Panel Figures (5-6 figures)
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_figures(metrics, gen_dict, real_emb, projected_text, sample_ids):
    """Generate 5 publication-quality multi-panel figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    from scipy import stats
    from sklearn.decomposition import PCA

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Publication style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    COLORS = {
        "real": "#2166ac",
        "generated": "#d6604d",
        "CD8_T": "#e41a1c",
        "Macrophage": "#377eb8",
        "Epithelial_tumor": "#4daf4a",
        "NK_cell": "#984ea3",
        "Fibroblast": "#ff7f00",
        "B_cell": "#a65628",
    }

    # ── Load training histories ──
    clop_hist = json.load(open(CKPT_DIR / "clop_history.json"))
    dit_hist = json.load(open(CKPT_DIR / "dit_history.json"))

    # ──────────────────────────────────────────────────────────────────────
    # FIGURE 1: Training Dynamics (2×2 panel)
    # a) CLOP loss curves  b) CLOP accuracy + temperature
    # c) DiT loss curves   d) DiT velocity cosine similarity
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Generating Figure 1: Training Dynamics...")
    fig1, axes1 = plt.subplots(2, 2, figsize=(8, 6.5))

    # 1a: CLOP loss
    ax = axes1[0, 0]
    epochs_c = range(1, len(clop_hist["train_loss"]) + 1)
    ax.plot(epochs_c, clop_hist["train_loss"], color="#2166ac", lw=1.5, label="Train")
    ax.plot(epochs_c, clop_hist["val_loss"], color="#d6604d", lw=1.5, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) CLOP Contrastive Loss", fontweight="bold")
    ax.legend(frameon=False)
    ax.set_xlim(1, len(clop_hist["train_loss"]))

    # 1b: CLOP accuracy + temperature
    ax = axes1[0, 1]
    ax.plot(epochs_c, clop_hist["val_acc"], color="#4daf4a", lw=1.5, label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy", color="#4daf4a")
    ax.tick_params(axis="y", labelcolor="#4daf4a")
    ax.set_title("(b) CLOP Alignment Quality", fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(epochs_c, clop_hist["temperature"], color="#984ea3", lw=1.0, ls="--",
             label="Temperature")
    ax2.set_ylabel("Temperature", color="#984ea3")
    ax2.tick_params(axis="y", labelcolor="#984ea3")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center right")

    # 1c: DiT loss
    ax = axes1[1, 0]
    epochs_d = range(1, len(dit_hist["train_loss"]) + 1)
    ax.plot(epochs_d, dit_hist["train_loss"], color="#2166ac", lw=1.5, label="Train")
    ax.plot(epochs_d, dit_hist["val_loss"], color="#d6604d", lw=1.5, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("(c) DiT Flow Matching Loss", fontweight="bold")
    ax.legend(frameon=False)
    ax.set_xlim(1, len(dit_hist["train_loss"]))

    # 1d: DiT cosine similarity
    ax = axes1[1, 1]
    if "val_cosine_sim" in dit_hist:
        ax.plot(epochs_d, dit_hist["val_cosine_sim"], color="#ff7f00", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("(d) Velocity Prediction Quality", fontweight="bold")
    ax.set_xlim(1, len(dit_hist["train_loss"]))
    ax.set_ylim(0, 1)

    fig1.suptitle("Figure 1: CLOP-DiT Training Dynamics", fontsize=13, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(FIG_DIR / "fig1_training_dynamics.pdf", dpi=300, bbox_inches="tight")
    fig1.savefig(FIG_DIR / "fig1_training_dynamics.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    logger.info("  Figure 1 saved.")

    # ──────────────────────────────────────────────────────────────────────
    # FIGURE 2: Embedding Space Analysis (1×3 panel)
    # a) PCA: Real vs Generated  b) Per-cell-type PCA  c) Norm distributions
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Generating Figure 2: Embedding Space Analysis...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4.5))

    # Subsample real for speed
    n_sub = min(2000, len(real_emb))
    idx_sub = np.random.choice(len(real_emb), n_sub, replace=False)
    real_sub = real_emb[idx_sub]

    all_gen = np.concatenate(list(gen_dict.values()), axis=0)
    combined = np.vstack([real_sub, all_gen])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)
    n_real = len(real_sub)

    # 2a: Real vs Generated
    ax = axes2[0]
    ax.scatter(coords[:n_real, 0], coords[:n_real, 1], s=3, alpha=0.3,
               color=COLORS["real"], label="Real", rasterized=True)
    ax.scatter(coords[n_real:, 0], coords[n_real:, 1], s=5, alpha=0.5,
               color=COLORS["generated"], label="Generated", rasterized=True)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("(a) Real vs Generated", fontweight="bold")
    ax.legend(frameon=False, markerscale=3)

    # 2b: Per-cell-type
    ax = axes2[1]
    ax.scatter(coords[:n_real, 0], coords[:n_real, 1], s=2, alpha=0.15,
               color="#cccccc", label="Real (all)", rasterized=True)
    offset = n_real
    for name in gen_dict:
        n = len(gen_dict[name])
        ax.scatter(coords[offset:offset+n, 0], coords[offset:offset+n, 1],
                   s=8, alpha=0.7, color=COLORS.get(name, "#333333"),
                   label=name, rasterized=True)
        offset += n
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("(b) Cell-Type Specific Generation", fontweight="bold")
    ax.legend(frameon=False, markerscale=2, fontsize=7, ncol=2)

    # 2c: Norm distributions
    ax = axes2[2]
    real_norms = np.linalg.norm(real_sub, axis=1)
    gen_norms = np.linalg.norm(all_gen, axis=1)
    ax.hist(real_norms, bins=50, alpha=0.6, color=COLORS["real"],
            label=f"Real (μ={real_norms.mean():.1f})", density=True)
    ax.hist(gen_norms, bins=50, alpha=0.6, color=COLORS["generated"],
            label=f"Gen (μ={gen_norms.mean():.1f})", density=True)
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Density")
    ax.set_title("(c) Embedding Norm Distribution", fontweight="bold")
    ax.legend(frameon=False)

    fig2.suptitle("Figure 2: Embedding Space Analysis", fontsize=13, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(FIG_DIR / "fig2_embedding_space.pdf", dpi=300, bbox_inches="tight")
    fig2.savefig(FIG_DIR / "fig2_embedding_space.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    logger.info("  Figure 2 saved.")

    # ──────────────────────────────────────────────────────────────────────
    # FIGURE 3: Quantitative Metrics Dashboard (2×3 panel)
    # a) FD by cell type  b) MMD by cell type  c) Coverage & Density
    # d) KL divergence    e) Cosine similarity  f) Dim-wise correlation
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Generating Figure 3: Metrics Dashboard...")
    fig3, axes3 = plt.subplots(2, 3, figsize=(14, 8))

    emb_metrics = metrics.get("embedding_quality", {})
    ct_names = list(gen_dict.keys())

    # 3a: FD by cell type
    ax = axes3[0, 0]
    fd_values = [emb_metrics.get(f"{ct}/frechet_distance", 0) for ct in ct_names]
    bars = ax.bar(range(len(ct_names)), fd_values, color=[COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("Fréchet Distance ↓")
    ax.set_title("(a) Fréchet Distance", fontweight="bold")

    # 3b: MMD by cell type
    ax = axes3[0, 1]
    mmd_values = [emb_metrics.get(f"{ct}/mmd_rbf", 0) for ct in ct_names]
    ax.bar(range(len(ct_names)), mmd_values, color=[COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("MMD (RBF) ↓")
    ax.set_title("(b) Maximum Mean Discrepancy", fontweight="bold")

    # 3c: Coverage & Density
    ax = axes3[0, 2]
    cov_values = [emb_metrics.get(f"{ct}/coverage", 0) for ct in ct_names]
    den_values = [emb_metrics.get(f"{ct}/density", 0) for ct in ct_names]
    x = np.arange(len(ct_names))
    w = 0.35
    ax.bar(x - w/2, cov_values, w, label="Coverage ↑", color="#2166ac", alpha=0.8)
    ax.bar(x + w/2, den_values, w, label="Density ↑", color="#d6604d", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title("(c) Coverage & Density", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)

    # 3d: KL divergence
    ax = axes3[1, 0]
    kl_values = [emb_metrics.get(f"{ct}/mean_kl", 0) for ct in ct_names]
    ax.bar(range(len(ct_names)), kl_values, color=[COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("Mean KL ↓")
    ax.set_title("(d) Per-Dimension KL Divergence", fontweight="bold")

    # 3e: Overall metrics summary
    ax = axes3[1, 1]
    overall_keys = ["overall/frechet_distance", "overall/mmd_rbf",
                    "overall/coverage", "overall/density", "overall/mean_kl"]
    labels = ["FD", "MMD", "Coverage", "Density", "KL"]
    values = [emb_metrics.get(k, 0) for k in overall_keys]
    # Normalize for radar-like viz
    ax.barh(range(len(labels)), values, color="#377eb8", alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Score")
    ax.set_title("(e) Overall Metrics Summary", fontweight="bold")

    # 3f: Prompt robustness
    ax = axes3[1, 2]
    rob = metrics.get("prompt_robustness", {})
    if rob:
        rob_cts = list(rob.keys())
        rob_sims = [rob[ct]["mean_cross_sim"] for ct in rob_cts]
        rob_stds = [rob[ct]["std_cross_sim"] for ct in rob_cts]
        ax.bar(range(len(rob_cts)), rob_sims, yerr=rob_stds,
               color=[COLORS.get(ct, "#999") for ct in rob_cts],
               capsize=3, alpha=0.8)
        ax.set_xticks(range(len(rob_cts)))
        ax.set_xticklabels([ct.replace("_", "\n") for ct in rob_cts], fontsize=7)
        ax.set_ylabel("Cross-Variant Cosine Sim ↑")
        ax.set_ylim(0, 1)
    ax.set_title("(f) Prompt Robustness", fontweight="bold")

    fig3.suptitle("Figure 3: Generation Quality Metrics", fontsize=13, fontweight="bold")
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig(FIG_DIR / "fig3_metrics_dashboard.pdf", dpi=300, bbox_inches="tight")
    fig3.savefig(FIG_DIR / "fig3_metrics_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)
    logger.info("  Figure 3 saved.")

    # ──────────────────────────────────────────────────────────────────────
    # FIGURE 4: Biological Validation (2×2 panel)
    # a) Per-cell-type embedding separation (UMAP or PCA)
    # b) Inter-type cosine distance heatmap
    # c) Diversity: intra-type variance
    # d) Conditioning fidelity: condition→embedding correlation
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Generating Figure 4: Biological Validation...")
    fig4, axes4 = plt.subplots(2, 2, figsize=(10, 9))

    # 4a: Cell-type separation (PCA with generated only)
    ax = axes4[0, 0]
    gen_labels = []
    gen_all_list = []
    for name, emb in gen_dict.items():
        gen_all_list.append(emb)
        gen_labels.extend([name] * len(emb))
    gen_all_arr = np.vstack(gen_all_list)
    pca_gen = PCA(n_components=2)
    coords_gen = pca_gen.fit_transform(gen_all_arr)

    for name in gen_dict:
        mask = np.array(gen_labels) == name
        ax.scatter(coords_gen[mask, 0], coords_gen[mask, 1], s=10, alpha=0.6,
                   color=COLORS.get(name, "#333"), label=name, rasterized=True)
    ax.set_xlabel(f"PC1 ({pca_gen.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_gen.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("(a) Generated Cell-Type Separation", fontweight="bold")
    ax.legend(frameon=False, markerscale=2, fontsize=7, ncol=2)

    # 4b: Inter-type cosine distance heatmap
    ax = axes4[0, 1]
    type_means = {}
    for name, emb in gen_dict.items():
        type_means[name] = emb.mean(axis=0)
    names = list(type_means.keys())
    n_types = len(names)
    sim_matrix = np.zeros((n_types, n_types))
    for i in range(n_types):
        for j in range(n_types):
            v1, v2 = type_means[names[i]], type_means[names[j]]
            sim_matrix[i, j] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(n_types))
    ax.set_yticks(range(n_types))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    for i in range(n_types):
        for j in range(n_types):
            ax.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine Sim")
    ax.set_title("(b) Inter-Type Similarity", fontweight="bold")

    # 4c: Intra-type diversity (variance)
    ax = axes4[1, 0]
    intra_vars = []
    for name in gen_dict:
        v = np.var(gen_dict[name], axis=0).mean()
        intra_vars.append(v)
    ax.bar(range(len(ct_names)), intra_vars, color=[COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("Mean Variance ↑")
    ax.set_title("(c) Intra-Type Diversity", fontweight="bold")

    # 4d: Conditioning fidelity (condition → nearest real cosine)
    ax = axes4[1, 1]
    fidelity_scores = []
    for name, emb in gen_dict.items():
        gen_mean = emb.mean(axis=0)
        gen_mean_n = gen_mean / (np.linalg.norm(gen_mean) + 1e-8)
        # Find nearest real neighbors
        real_norms_mat = real_emb / (np.linalg.norm(real_emb, axis=1, keepdims=True) + 1e-8)
        sims_to_real = real_norms_mat @ gen_mean_n
        top_k_sim = np.sort(sims_to_real)[-50:].mean()
        fidelity_scores.append(top_k_sim)

    ax.bar(range(len(ct_names)), fidelity_scores, color=[COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("Top-50 Real Cosine Sim ↑")
    ax.set_title("(d) Conditioning Fidelity", fontweight="bold")

    fig4.suptitle("Figure 4: Biological Validation", fontsize=13, fontweight="bold")
    fig4.tight_layout(rect=[0, 0, 1, 0.96])
    fig4.savefig(FIG_DIR / "fig4_biological_validation.pdf", dpi=300, bbox_inches="tight")
    fig4.savefig(FIG_DIR / "fig4_biological_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)
    logger.info("  Figure 4 saved.")

    # ──────────────────────────────────────────────────────────────────────
    # FIGURE 5: Dimension-Level Analysis & ODE Integration (2×2 panel)
    # a) Per-dim distribution comparison (violin)
    # b) Dim-wise correlation real vs gen
    # c) Sampling trajectory visualization
    # d) CFG scale ablation 
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Generating Figure 5: Dimension & Sampling Analysis...")
    fig5, axes5 = plt.subplots(2, 2, figsize=(9, 7.5))

    # 5a: Per-dimension distribution (first 20 dims)
    ax = axes5[0, 0]
    n_dims_show = 16
    real_sub_small = real_sub[:500]
    gen_sub_small = all_gen[:500]
    positions = []
    data_real, data_gen = [], []
    for d in range(n_dims_show):
        data_real.append(real_sub_small[:, d])
        data_gen.append(gen_sub_small[:, d])
    bp_r = ax.boxplot(data_real, positions=np.arange(n_dims_show)*3,
                      widths=0.8, patch_artist=True, showfliers=False)
    bp_g = ax.boxplot(data_gen, positions=np.arange(n_dims_show)*3 + 1,
                      widths=0.8, patch_artist=True, showfliers=False)
    for patch in bp_r["boxes"]:
        patch.set_facecolor(COLORS["real"])
        patch.set_alpha(0.6)
    for patch in bp_g["boxes"]:
        patch.set_facecolor(COLORS["generated"])
        patch.set_alpha(0.6)
    ax.set_xticks(np.arange(n_dims_show)*3 + 0.5)
    ax.set_xticklabels([str(i) for i in range(n_dims_show)], fontsize=7)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Value")
    ax.set_title("(a) Per-Dimension Distribution", fontweight="bold")
    ax.legend([bp_r["boxes"][0], bp_g["boxes"][0]], ["Real", "Generated"], frameon=False)

    # 5b: Dim-wise mean correlation
    ax = axes5[0, 1]
    real_dim_mean = real_sub.mean(axis=0)
    gen_dim_mean = all_gen.mean(axis=0)
    r_corr, p_val = stats.pearsonr(real_dim_mean, gen_dim_mean)
    ax.scatter(real_dim_mean, gen_dim_mean, s=5, alpha=0.5, color="#377eb8")
    lims = [min(real_dim_mean.min(), gen_dim_mean.min()),
            max(real_dim_mean.max(), gen_dim_mean.max())]
    ax.plot(lims, lims, "r--", alpha=0.5, lw=1)
    ax.set_xlabel("Real Mean")
    ax.set_ylabel("Generated Mean")
    ax.set_title("(b) Dimension-wise Correlation", fontweight="bold")
    ax.text(0.05, 0.95, f"r = {r_corr:.4f}\np = {p_val:.2e}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # 5c: Sampling trajectory visualization
    ax = axes5[1, 0]
    # Show how z evolves from noise to data for a single example
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        clop_m, dit_m, _, _ = load_models(device)
        cond_ex = encode_text(CELL_TYPE_PROMPTS["CD8_T"], clop_m, device)
        cond_batch = cond_ex.expand(1, -1)
        z = torch.randn(1, 512, device=device)
        trajectory = [z.cpu().numpy().flatten()]
        n_show_steps = 20
        dt = 1.0 / n_show_steps
        for i in range(n_show_steps):
            t_val = i / n_show_steps
            t = torch.full((1,), t_val, device=device)
            v = dit_m.forward_with_cfg(z, t, cond_batch, cfg_scale=3.0)
            z = z + v * dt
            trajectory.append(z.cpu().numpy().flatten())
        del clop_m, dit_m
        torch.cuda.empty_cache()

        traj = np.array(trajectory)
        traj_pca = PCA(n_components=2).fit_transform(traj)
        colors_traj = plt.cm.viridis(np.linspace(0, 1, len(traj_pca)))
        for i in range(len(traj_pca)-1):
            ax.annotate("", xy=traj_pca[i+1], xytext=traj_pca[i],
                        arrowprops=dict(arrowstyle="->", color=colors_traj[i], lw=1.5))
        ax.scatter(traj_pca[0, 0], traj_pca[0, 1], s=50, marker="o",
                   color="blue", zorder=5, label="z₀ (noise)")
        ax.scatter(traj_pca[-1, 0], traj_pca[-1, 1], s=50, marker="*",
                   color="red", zorder=5, label="z₁ (cell)")
        ax.legend(frameon=False)
    except Exception as e:
        ax.text(0.5, 0.5, f"Trajectory unavailable\n{e}", transform=ax.transAxes,
                ha="center", va="center")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("(c) ODE Sampling Trajectory", fontweight="bold")

    # 5d: CFG scale ablation
    ax = axes5[1, 1]
    cfg_scales = [1.0, 2.0, 3.0, 5.0, 7.0]
    cfg_fds = []
    try:
        clop_m, dit_m, _, _ = load_models(device)
        cond_ex = encode_text(CELL_TYPE_PROMPTS["CD8_T"], clop_m, device)
        for cfg in cfg_scales:
            cond_batch = cond_ex.expand(100, -1)
            gen_cfg = dit_m.sample(cond_batch, num_steps=20, cfg_scale=cfg)
            gen_np = gen_cfg.cpu().numpy()
            # Compute FD against real
            from src.evaluation.metrics import GenerationMetrics
            fd = GenerationMetrics.frechet_distance(real_emb[:100], gen_np)
            cfg_fds.append(fd)
        del clop_m, dit_m
        torch.cuda.empty_cache()
        ax.plot(cfg_scales, cfg_fds, "o-", color="#377eb8", lw=2, markersize=6)
        ax.axvline(x=3.0, color="red", ls="--", alpha=0.5, label="default (3.0)")
        ax.legend(frameon=False)
    except Exception as e:
        ax.text(0.5, 0.5, f"CFG ablation unavailable\n{e}", transform=ax.transAxes,
                ha="center", va="center")
    ax.set_xlabel("CFG Scale")
    ax.set_ylabel("Fréchet Distance ↓")
    ax.set_title("(d) CFG Scale Ablation", fontweight="bold")

    fig5.suptitle("Figure 5: Dimension & Sampling Analysis", fontsize=13, fontweight="bold")
    fig5.tight_layout(rect=[0, 0, 1, 0.96])
    fig5.savefig(FIG_DIR / "fig5_dimension_sampling.pdf", dpi=300, bbox_inches="tight")
    fig5.savefig(FIG_DIR / "fig5_dimension_sampling.png", dpi=300, bbox_inches="tight")
    plt.close(fig5)
    logger.info("  Figure 5 saved.")

    logger.info(f"All 5 figures saved to {FIG_DIR}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5: Visual Conflict Detection
# ═══════════════════════════════════════════════════════════════════════════

def run_visual_conflict_check():
    """Check figures for visual consistency issues."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    logger.info("=" * 70)
    logger.info("STAGE 5: Visual Conflict Detection")
    logger.info("=" * 70)

    issues = []
    fig_files = sorted(FIG_DIR.glob("*.png"))

    for fpath in fig_files:
        try:
            img = Image.open(fpath)
            w, h = img.size

            # Check resolution
            if w < 1000 or h < 800:
                issues.append(f"{fpath.name}: LOW_RESOLUTION ({w}x{h})")

            # Check aspect ratio
            ratio = w / h
            if ratio > 3.5 or ratio < 0.28:
                issues.append(f"{fpath.name}: EXTREME_ASPECT_RATIO ({ratio:.2f})")

            # Check file size (too small might mean empty)
            size_kb = fpath.stat().st_size / 1024
            if size_kb < 10:
                issues.append(f"{fpath.name}: SUSPICIOUSLY_SMALL ({size_kb:.1f}KB)")

            # Check for mostly-white (potential blank figure)
            arr = np.array(img.convert("L"))
            white_frac = (arr > 250).mean()
            if white_frac > 0.97:
                issues.append(f"{fpath.name}: MOSTLY_BLANK ({white_frac*100:.1f}% white)")

            logger.info(f"  ✓ {fpath.name}: {w}x{h}, {size_kb:.0f}KB, white={white_frac*100:.0f}%")

        except Exception as e:
            issues.append(f"{fpath.name}: LOAD_ERROR ({e})")

    # Check logical consistency
    expected_figs = [
        "fig1_training_dynamics",
        "fig2_embedding_space",
        "fig3_metrics_dashboard",
        "fig4_biological_validation",
        "fig5_dimension_sampling",
    ]
    for expected in expected_figs:
        if not any(expected in f.name for f in fig_files):
            issues.append(f"MISSING: {expected}")

    if issues:
        logger.warning(f"Visual conflict issues found ({len(issues)}):")
        for issue in issues:
            logger.warning(f"  ⚠ {issue}")
    else:
        logger.info("  ✓ No visual conflicts detected. All figures pass checks.")

    # Save report
    report = {
        "total_figures": len(fig_files),
        "issues": issues,
        "figures_checked": [f.name for f in fig_files],
        "status": "PASS" if not issues else "ISSUES_FOUND",
    }
    with open(FIG_DIR / "visual_conflict_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CLOP-DiT v6 Full Pipeline")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "preprocess", "train_clop", "train_dit", "evaluate", "figures", "check"])
    parser.add_argument("--clop_epochs", type=int, default=200)
    parser.add_argument("--dit_epochs", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    setup_logging(log_file=str(PROJECT_ROOT / "logs" / "pipeline_v6.log"))

    stages = args.stage
    if stages == "all":
        stages = ["preprocess", "train_clop", "train_dit", "evaluate", "figures", "check"]
    else:
        stages = [stages]

    for stage in stages:
        if stage == "preprocess":
            preprocess_embeddings()

        elif stage == "train_clop":
            train_clop(num_epochs=args.clop_epochs, resume=args.resume)

        elif stage == "train_dit":
            train_dit(num_epochs=args.dit_epochs, resume=args.resume)

        elif stage == "evaluate":
            metrics, gen_dict, real_emb, proj_text, sids = run_full_evaluation(args.device)
            # Cache for figures stage
            np.savez(RESULTS_DIR / "eval_cache.npz",
                     real_emb=real_emb[:2000],
                     proj_text=proj_text[:2000],
                     sample_ids=sids[:2000])

        elif stage == "figures":
            # Load evaluation results
            if (RESULTS_DIR / "metrics_v5.json").exists():
                metrics = json.load(open(RESULTS_DIR / "metrics_v5.json"))
            else:
                metrics, gen_dict, real_emb, proj_text, sids = run_full_evaluation(args.device)

            # Load generated embeddings
            gen_dict = {}
            for ct in CELL_TYPE_PROMPTS:
                path = RESULTS_DIR / f"gen_{ct}.npy"
                if path.exists():
                    gen_dict[ct] = np.load(path)

            real_emb = np.load(CACHE_DIR / "cell_embeddings.npy")
            proj_text = np.load(CACHE_DIR / "projected_text.npy")
            sids = np.load(CACHE_DIR / "sample_ids.npy")

            generate_all_figures(metrics, gen_dict, real_emb, proj_text, sids)

        elif stage == "check":
            run_visual_conflict_check()

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
