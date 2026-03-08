#!/usr/bin/env python3
"""run_pipeline.py — Single entry point for the full CLOP-DiT pipeline.

Runs stages in order; paths come from configs/pipeline.yaml and src.utils.paths.
Override with env: CLOPDIT_CACHE_DIR, CLOPDIT_RESULTS_DIR, CLOPDIT_FIG_DIR, etc.

Stages (in order):
  data_prep       — 00/01/02 prepare h5ad + metadata (optional; often done once)
  cache           — 03_cache_latents
  dedup           — 03c_build_dedup_cache (requires text_group_mapping + text_captions_deduplicated)
  preprocess      — 03b_preprocess_embeddings
  train_clop      — 04a_train_clop
  train_dit       — 04b_train_dit
  generate        — generate_embeddings
  decode          — decode_expression
  diversity       — diversity_diagnostics
  conditioning    — conditioning_analysis
  downstream      — downstream_biology
  benchmark       — model_benchmarking
  figures         — generate_architecture_figure + results_visualizer
  article_delivery — verify + symlink 15 PDFs to articles/figures
  build_article   — latexmk -pdf (optional)

Usage:
  python scripts/pipeline/run_pipeline.py --stage all
  python scripts/pipeline/run_pipeline.py --from generate   # from generate through article_delivery
  python scripts/pipeline/run_pipeline.py --stage figures
  python scripts/pipeline/run_pipeline.py --stage all --build-article
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.paths import (
    CACHE_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    FIG_DIR,
    ARTICLE_DIR,
    ARTICLE_TEX,
    PROCESSED_H5AD_DIR,
)

STAGES_IN_ORDER = [
    "data_prep",
    "cache",
    "dedup",
    "preprocess",
    "train_clop",
    "train_dit",
    "generate",
    "decode",
    "diversity",
    "conditioning",
    "downstream",
    "benchmark",
    "figures",
    "article_delivery",
]


def run(cmd: list[str], cwd: Path | None = None) -> int:
    cwd = cwd or ROOT
    return subprocess.run(cmd, cwd=cwd).returncode


def run_stage(stage: str, extra: list[str] | None = None) -> int:
    extra = extra or []
    if stage == "data_prep":
        return run(["python", "scripts/data_prep/00_prepare_all_data.py", "--output_dir", str(PROCESSED_H5AD_DIR), *extra])
    if stage == "cache":
        return run(["python", "scripts/data_prep/03_cache_latents.py", "--h5ad_dir", str(PROCESSED_H5AD_DIR), "--output_dir", str(CACHE_DIR), *extra])
    if stage == "dedup":
        return run(["python", "scripts/data_prep/03c_build_dedup_cache.py", "--cache-dir", str(CACHE_DIR), *extra])
    if stage == "preprocess":
        return run(["python", "scripts/data_prep/03b_preprocess_embeddings.py", "--cache_dir", str(CACHE_DIR), *extra])
    if stage == "train_clop":
        return run([
            "python", "scripts/training/04a_train_clop.py",
            "--config", "configs/clop.yaml",
            "--cache_dir", str(CACHE_DIR),
            "--save_dir", str(CHECKPOINT_DIR),
            *extra,
        ])
    if stage == "train_dit":
        return run([
            "python", "scripts/training/04b_train_dit.py",
            "--config", "configs/dit.yaml",
            "--cache_dir", str(CACHE_DIR),
            "--save_dir", str(CHECKPOINT_DIR),
            *extra,
        ])
    if stage == "generate":
        return run([
            "python", "scripts/inference/generate_embeddings.py",
            "--condition-mode", "condition_noise", "--noise-scale", "0.03", "--cfg-scale", "1.5",
            "--num-per-type", "100", "--num-steps", "20",
            *extra,
        ])
    if stage == "decode":
        return run(["python", "scripts/analysis/decode_expression.py", *extra])
    if stage == "diversity":
        return run(["python", "scripts/analysis/diversity_diagnostics.py", "--num-per-type", "100", *extra])
    if stage == "conditioning":
        return run(["python", "scripts/analysis/conditioning_analysis.py", "--num-per-type", "100", "--cfg-scale", "1.5", "--noise-scale", "0.03", *extra])
    if stage == "downstream":
        return run(["python", "-m", "src.evaluation.downstream_biology", "--output-dir", str(RESULTS_DIR / "downstream"), *extra])
    if stage == "benchmark":
        return run(["python", "-m", "src.evaluation.model_benchmarking", *extra])
    if stage == "figures":
        code = run(["python", "scripts/analysis/generate_architecture_figure.py"])
        if code != 0:
            return code
        return run(["python", "-m", "src.visualization.results_visualizer", *extra])
    if stage == "article_delivery":
        return run(["python", "-m", "src.visualization.article_delivery", *extra])
    print(f"Unknown stage: {stage}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="CLOP-DiT pipeline runner")
    parser.add_argument("--stage", default="all", help="Stage to run, or 'all'")
    parser.add_argument("--from", dest="from_stage", default=None,
                        help="Start from this stage (inclusive) through article_delivery")
    parser.add_argument("--build-article", action="store_true",
                        help="After article_delivery, run latexmk to build PDF")
    args = parser.parse_args()

    if args.from_stage:
        try:
            start_idx = STAGES_IN_ORDER.index(args.from_stage)
        except ValueError:
            print(f"Unknown --from stage: {args.from_stage}. Choose from: {STAGES_IN_ORDER}", file=sys.stderr)
            return 1
        stages = STAGES_IN_ORDER[start_idx:]
    elif args.stage == "all":
        stages = STAGES_IN_ORDER.copy()
    else:
        if args.stage not in STAGES_IN_ORDER:
            print(f"Unknown stage: {args.stage}. Choose from: {STAGES_IN_ORDER}", file=sys.stderr)
            return 1
        stages = [args.stage]

    for stage in stages:
        print(f"\n▶ Running stage: {stage}")
        code = run_stage(stage)
        if code != 0:
            print(f"Stage {stage} failed with exit code {code}", file=sys.stderr)
            return code

    if args.build_article:
        print("\n▶ Building article PDF...")
        code = run(["latexmk", "-pdf", ARTICLE_TEX], cwd=ARTICLE_DIR)
        if code != 0:
            print("latexmk failed", file=sys.stderr)
            return code
        print(f"  → {ARTICLE_DIR / ARTICLE_TEX.replace('.tex', '.pdf')}")

    print("\nPipeline run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
