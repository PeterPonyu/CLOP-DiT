#!/usr/bin/env python3
"""5-Fold Group Cross-Validation for CLOP-DiT Pipeline.

Runs the full CLOP → DiT → Evaluate pipeline for each of 5 folds,
then aggregates results (mean ± std) for publication.

Usage:
    python scripts/training/run_5fold_cv.py --config configs/clop.yaml --dit_config configs/dit.yaml
    python scripts/training/run_5fold_cv.py --resume_from_fold 3   # resume from fold 3
"""

import argparse
import json
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.paths import CACHE_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR

PYTHON = sys.executable


def run_command(cmd: list, log_file: str, description: str) -> bool:
    """Run a subprocess command and log output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Log: {log_file}")
    print(f"{'='*60}")
    start = time.time()

    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT)
        )

    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"  [{status}] {elapsed:.1f}s elapsed")

    if result.returncode != 0:
        print(f"  ERROR: Check {log_file} for details")
        # Print last 20 lines of log
        with open(log_file) as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(f"    {line.rstrip()}")

    return result.returncode == 0


def evaluate_fold(fold_idx: int, ckpt_dir: Path, cache_dir: Path,
                  device: str = "cuda") -> dict:
    """Evaluate a single fold using saved checkpoints."""
    import torch

    from src.evaluation.run_metrics import load_clop, load_dit
    from src.evaluation.metrics import GenerationMetrics

    clop_path = ckpt_dir / "clop_best.pth"
    dit_path = ckpt_dir / "dit_best.pth"

    clop, clop_cfg = load_clop(clop_path, device)
    dit, _ = load_dit(dit_path, clop_config=clop_cfg, device=device)
    clop_ckpt = torch.load(clop_path, map_location=device, weights_only=False)
    dit_ckpt = torch.load(dit_path, map_location=device, weights_only=False)

    # Load real embeddings and encode prompts
    real_emb = np.load(cache_dir / "cell_embeddings.npy")

    CELL_TYPE_PROMPTS = {
        "CD8_T": "CD8+ cytotoxic T lymphocytes from human lung adenocarcinoma tumor microenvironment expressing cytotoxic effector molecules",
        "Macrophage": "Tumor-associated macrophages from human lung adenocarcinoma myeloid populations in the cancer microenvironment",
        "Epithelial_tumor": "Malignant epithelial cells from human lung adenocarcinoma cancer cells of epithelial origin",
        "Fibroblast": "Cancer-associated fibroblasts from human lung adenocarcinoma stroma supporting tumor growth",
        "NK_cell": "Natural killer cells infiltrating human lung adenocarcinoma innate lymphoid NK cells with cytotoxic activity",
        "B_cell": "B lymphocytes from human lung adenocarcinoma tumor microenvironment adaptive immune cells",
    }

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
    )
    bert = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        ignore_mismatched_sizes=True,
    ).to(device).eval()

    # Generate cells
    gen_all = []
    torch.manual_seed(42)
    for name, text in CELL_TYPE_PROMPTS.items():
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(device)
            outputs = bert(**inputs)
            text_emb = outputs.last_hidden_state[:, 0, :]
            cond = clop.project_text(text_emb)
            cond_batch = cond.expand(200, -1)
            emb = dit.sample(cond_batch, num_steps=20, cfg_scale=3.0)
            gen_all.append(emb.cpu().numpy())

    gen = np.concatenate(gen_all, axis=0)
    real_sub = real_emb[np.random.choice(len(real_emb), len(gen), replace=False)]

    metrics = GenerationMetrics.full_evaluation(real_sub, gen)

    # Also get CLOP training metrics
    clop_train_acc = clop_ckpt.get("train_acc", 0.0)
    clop_val_acc = clop_ckpt.get("val_acc", 0.0)
    clop_val_loss = clop_ckpt.get("val_loss", 0.0)

    dit_val_loss = dit_ckpt.get("val_loss", 0.0)

    metrics["clop_train_acc"] = float(clop_train_acc) if clop_train_acc else 0.0
    metrics["clop_val_acc"] = float(clop_val_acc) if clop_val_acc else 0.0
    metrics["dit_val_loss"] = float(dit_val_loss) if dit_val_loss else 0.0

    # Cleanup
    del bert, tokenizer, clop, dit
    torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="5-Fold Group CV")
    parser.add_argument("--clop_config", default="configs/clop.yaml")
    parser.add_argument("--dit_config", default="configs/dit.yaml")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--resume_from_fold", type=int, default=0,
                        help="Resume from this fold (skip earlier folds)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only aggregate results (skip training)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cache_dir = CACHE_DIR
    results_dir = RESULTS_DIR / "5fold_cv"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = LOG_DIR / "5fold_cv"
    log_dir.mkdir(parents=True, exist_ok=True)

    fold_results = {}
    total_start = time.time()

    for fold in range(args.n_folds):
        fold_dir = CHECKPOINT_DIR / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        projected_text_path = fold_dir / "projected_text.npy"

        if fold < args.resume_from_fold or args.eval_only:
            print(f"\n[Fold {fold}] Skipping training (resume_from_fold={args.resume_from_fold})")
        else:
            print(f"\n{'#'*60}")
            print(f"  FOLD {fold}/{args.n_folds-1}")
            print(f"{'#'*60}")

            # ── Step 1: CLOP Training ──
            clop_cmd = [
                PYTHON, "scripts/training/04a_train_clop.py",
                "--config", args.clop_config,
                "--cache_dir", str(cache_dir),
                "--n_folds", str(args.n_folds),
                "--fold_idx", str(fold),
                "--save_dir", str(fold_dir),
            ]
            ok = run_command(
                clop_cmd,
                str(log_dir / f"clop_fold{fold}.log"),
                f"Fold {fold}: CLOP Training"
            )
            if not ok:
                print(f"CLOP training failed for fold {fold}. Stopping.")
                break

            # The project_and_save writes to cache_dir/projected_text.npy
            # We need to move it to the fold directory
            default_proj = cache_dir / "projected_text.npy"
            if default_proj.exists():
                import shutil
                shutil.copy2(str(default_proj), str(projected_text_path))
                print(f"  Copied projected_text.npy to {projected_text_path}")

            # ── Step 2: DiT Training ──
            dit_cmd = [
                PYTHON, "scripts/training/04b_train_dit.py",
                "--config", args.dit_config,
                "--cache_dir", str(cache_dir),
                "--n_folds", str(args.n_folds),
                "--fold_idx", str(fold),
                "--save_dir", str(fold_dir),
                "--projected_text", str(projected_text_path),
            ]
            ok = run_command(
                dit_cmd,
                str(log_dir / f"dit_fold{fold}.log"),
                f"Fold {fold}: DiT Training"
            )
            if not ok:
                print(f"DiT training failed for fold {fold}. Stopping.")
                break

        # ── Step 3: Evaluate fold ──
        if (fold_dir / "clop_best.pth").exists() and (fold_dir / "dit_best.pth").exists():
            print(f"\n  Evaluating fold {fold}...")
            try:
                metrics = evaluate_fold(fold, fold_dir, cache_dir, args.device)
                fold_results[f"fold_{fold}"] = metrics
                print(f"  Fold {fold} — FD={metrics['frechet_distance']:.2f}  "
                      f"Cov={metrics['coverage']:.3f}  Den={metrics['density']:.3f}  "
                      f"KL={metrics['mean_kl']:.3f}")
            except Exception as e:
                print(f"  Evaluation failed for fold {fold}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  Fold {fold}: checkpoints not found, skipping evaluation")

    # ── Aggregate results ──
    if fold_results:
        print(f"\n{'='*60}")
        print("  5-FOLD CV AGGREGATE RESULTS")
        print(f"{'='*60}")

        metric_keys = [
            "frechet_distance", "mmd_rbf", "coverage", "density",
            "mean_kl", "mean_pairwise_cosine_sim", "diversity_index",
        ]
        agg = {}
        for key in metric_keys:
            values = [fold_results[f].get(key, 0.0) for f in fold_results]
            if values:
                agg[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }
                print(f"  {key:30s}: {agg[key]['mean']:.4f} ± {agg[key]['std']:.4f}")

        output = {
            "n_folds": args.n_folds,
            "per_fold": fold_results,
            "aggregate": agg,
            "total_time_seconds": time.time() - total_start,
        }

        out_path = results_dir / "5fold_cv_results.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")

    total_elapsed = time.time() - total_start
    print(f"\n  Total time: {total_elapsed/3600:.1f} hours")


if __name__ == "__main__":
    main()
