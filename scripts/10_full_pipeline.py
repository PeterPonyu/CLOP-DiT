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
from src.utils.paths import (
    PROJECT_ROOT,
    CACHE_DIR,
    RESULTS_DIR,
    FIG_DIR,
    CHECKPOINT_DIR,
    CONFIG_DIR,
    LOG_DIR,
)
from src.evaluation.run_metrics import load_models_for_pipeline
from src.visualization.full_pipeline_figures import generate_all_figures

logger = logging.getLogger(__name__)

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

    config_path = CONFIG_DIR / "clop.yaml"
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

    config_path = CONFIG_DIR / "dit.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["num_epochs"] = num_epochs
    if resume:
        best_ckpt = CHECKPOINT_DIR / "dit_best.pth"
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
    """Load trained CLOP + DiT models via shared API."""
    return load_models_for_pipeline(
        CHECKPOINT_DIR / "clop_best.pth",
        CHECKPOINT_DIR / "dit_best.pth",
        device=device,
    )


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
# STAGE 4: Publication-Quality Multi-Panel Figures (delegated to full_pipeline_figures)
# ═══════════════════════════════════════════════════════════════════════════


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
            ratio = max(w, h) / max(min(w, h), 1)
            if ratio > 3.5 or ratio < 0.28:
                issues.append(f"{fpath.name}: EXTREME_ASPECT_RATIO ({ratio:.2f})")
            size_kb = fpath.stat().st_size / 1024
            if size_kb < 10:
                issues.append(f"{fpath.name}: SUSPICIOUSLY_SMALL ({size_kb:.1f}KB)")
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

    setup_logging(log_file=str(LOG_DIR / "pipeline_v6.log"))

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
