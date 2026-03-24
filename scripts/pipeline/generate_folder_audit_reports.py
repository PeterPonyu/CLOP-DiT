#!/usr/bin/env python3
"""
Generate FOLDER_REVIEW.md in each project subdirectory for audit handoff.

Excludes VCS/cache dirs. Idempotent: overwrites FOLDER_REVIEW.md when re-run.

Usage (from repo root):
  python scripts/pipeline/generate_folder_audit_reports.py
"""
from __future__ import annotations

import os
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "node_modules",
    }
)


def should_skip_dir(name: str) -> bool:
    if name in SKIP_DIR_NAMES:
        return True
    if name.startswith(".") and name not in {".github"}:
        return True
    return False


# Long-form context for reviewers (path uses forward slashes, relative to repo root, no leading slash)
CONTEXT: dict[str, tuple[str, str]] = {
    "": (
        "Repository root",
        "Installable Python package (`pip install -e .`), entry docs (`README.md`, `PIPELINE.md`, `REPRODUCIBILITY.md`), "
        "and orchestration. Prefer `src.utils.paths` and `configs/pipeline.yaml` for paths.",
    ),
    "src": (
        "Core library (`clopdit` package)",
        "Imported by scripts and tests. Subpackages: architecture, data_pipeline, evaluation, training, utils, visualization, experiments.",
    ),
    "src/architecture": (
        "Neural modules",
        "DiT, CLOP aligner (`clop/`), scGPT integration, Cell2Cell. Training scripts live under `scripts/training/`.",
    ),
    "src/architecture/clop": (
        "CLOP contrastive head",
        "Aligner, Prototype-SigLIP / loss definitions. Coordinate with `configs/clop.yaml` and `04a_train_clop.py`.",
    ),
    "src/data_pipeline": (
        "Datasets and caches",
        "PyTorch datasets, ZCA preprocessing, GEO helpers, samplers. Reads from `data/cached_latents_*` per pipeline config.",
    ),
    "src/evaluation": (
        "Metrics and benchmarking",
        "Embedding metrics, biological validation package, baseline registry, `model_benchmarking.py`. Outputs under `results/`.",
    ),
    "src/evaluation/biological_validation": (
        "Text-to-cell biological validation",
        "Generation, marker/DE metrics, optional figures. See `src/evaluation/README.md`.",
    ),
    "src/experiments": (
        "Experiment drivers",
        "OOD and auxiliary experiment code (e.g. `ood_evaluation.py`).",
    ),
    "src/training": (
        "Training loop implementations",
        "Used by `scripts/training/*.py`; includes schedulers and Cell2Cell training.",
    ),
    "src/utils": (
        "Shared utilities",
        "`paths.py` is canonical for FIG_DIR, RESULTS_DIR, etc. Env overrides: `CLOPDIT_*`.",
    ),
    "src/visualization": (
        "Figures and article delivery",
        "`article_delivery.py` manifest, `style.py`, `results_visualizer.py`, `fig*.py` modules. Saves to `results/figures/`.",
    ),
    "scripts": (
        "Runnable scripts (not the importable package)",
        "Organized by stage: data_prep, training, inference, analysis, pipeline, baselines, vcd. See `scripts/README.md`.",
    ),
    "scripts/data_prep": (
        "Data preparation pipeline",
        "Steps 00–03: integrate h5ad, captions, cache latents, ZCA, dedup. Heavy I/O.",
    ),
    "scripts/training": (
        "Training entrypoints",
        "`04a_train_clop.py`, `04b_train_dit.py`, ablations, CV, robustness runners.",
    ),
    "scripts/inference": (
        "Generation and evaluation",
        "Inference, decode, biological validation scripts, cell2cell inference.",
    ),
    "scripts/analysis": (
        "Post-hoc analysis and paper figures",
        "Architecture diagrams, gene–gene correlation, DE, dashboards, supplementary tables.",
    ),
    "scripts/pipeline": (
        "Orchestration and article build",
        "`run_pipeline.py`, figure regeneration, LaTeX build helpers. This audit generator lives here.",
    ),
    "scripts/baselines": (
        "Baseline method training",
        "Embedding-VAE, scGen, scVI-style baselines; configs under `configs/baselines/`.",
    ),
    "scripts/vcd": (
        "Visual conflict detection (VCD)",
        "Automated figure QA for matplotlib outputs. Font files may live under `fonts/`.",
    ),
    "scripts/archive": (
        "Archived scripts",
        "Legacy or superseded scripts; verify before use.",
    ),
    "articles": (
        "Primary LaTeX manuscript",
        "`clop_dit_biology.tex`, `articles/figures/` symlinks to `results/figures/`. Build: `scripts/pipeline/build_article.sh`.",
    ),
    "articles/figures": (
        "Article figure symlinks",
        "Populated by `article_delivery.py`; do not commit large binary copies—prefer symlinks.",
    ),
    "articles/Definitions": (
        "LaTeX snippet definitions",
        "Shared macros or definition files for the article.",
    ),
    "articles_elsevier": (
        "Elsevier-variant article tree",
        "Parallel layout to `articles/`; keep figure paths consistent if maintained in parallel.",
    ),
    "articles_elsevier copy": (
        "Duplicate Elsevier tree (copy)",
        "Likely a working copy; consider consolidating to avoid drift.",
    ),
    "docs": (
        "Project documentation",
        "Methodology notes, quick start, figure organization—supplement README.",
    ),
    "tests": (
        "Pytest suite",
        "Run `pytest tests/`. Covers configs, visualization delivery, CLIP loss, etc.",
    ),
    "configs": (
        "YAML and JSON configuration",
        "`pipeline.yaml` drives paths; `clop.yaml`, `dit.yaml`, `models.yaml`, prompts, baselines.",
    ),
    "configs/baselines": (
        "Baseline configs",
        "Per-method YAML for scVI, embedding VAE, etc.",
    ),
    "configs/archive": (
        "Archived configs",
        "Historical runs; match to `results/archive/` or old checkpoints when auditing.",
    ),
    "configs/prompts": (
        "Prompt assets",
        "Default or variant prompt JSON for conditioning experiments.",
    ),
    "data": (
        "Data and derived caches (large)",
        "Not all data may be in git; `processed_h5ad/`, `cached_latents_*` per `configs/pipeline.yaml`.",
    ),
    "data/processed_h5ad": (
        "Integrated h5ad inputs",
        "Upstream of latent caching; verify provenance in data prep scripts.",
    ),
    "data/cached_latents": (
        "Cached embeddings / latents",
        "Primary training cache path in default pipeline config; may include `archive/` subtrees.",
    ),
    "results": (
        "Experiment outputs",
        "Metrics JSON, figures, baselines, ablations, robustness—primary benchmarking surface for the paper.",
    ),
    "results/figures": (
        "Generated figure PDFs/PNGs",
        "Source for article delivery; VCD reports may reference `_live_vcd/` during runs.",
    ),
    "results/baselines": (
        "Baseline artifacts",
        "Per-method folders with `embeddings.npy`, `metadata.json` per evaluation README contract.",
    ),
    "results/ablations": (
        "CLOP ablation runs",
        "One subfolder per ablation; typically contains `checkpoints/` and metric JSON. Compare against primary `models/checkpoints/`.",
    ),
    "results/archive": (
        "Archived result bundles",
        "Versioned metrics/checkpoints snapshots (e.g. v7.x); use for historical comparison only.",
    ),
    "results/downstream": (
        "Downstream biology outputs",
        "Clustering/classifier/DE artifacts consumed by downstream panels.",
    ),
    "results/robustness": (
        "Cross-dataset robustness",
        "Bioval_* style runs; pair with `scripts/training/run_robustness_experiments.py`.",
    ),
    "results/ood_evaluation": (
        "Out-of-distribution evaluation",
        "OOD metrics and caches for generalization analysis.",
    ),
    "results/conditioning_ablation": (
        "Conditioning ablations",
        "Noise/conditioning sweeps; align with `fig*` conditioning panels.",
    ),
    "results/multi_seed": (
        "Multi-seed runs",
        "Variance across seeds; see Fig 22 / multi-seed visualization.",
    ),
    "results/marker_independence": (
        "Marker independence analysis",
        "Marker completeness / leakage style analyses.",
    ),
    "results/rare_cell_augmentation": (
        "Rare-cell augmentation experiments",
        "Augmentation or sampling experiments for rare types.",
    ),
    "results/variance_matching_pilot": (
        "Variance matching pilot outputs",
        "Pilot metrics/figures for variance matching (related to Fig 19 / pilot scripts).",
    ),
    "results/conditioning_cache": (
        "Conditioning analysis cache",
        "Intermediate arrays for conditioning analyses.",
    ),
    "models": (
        "Pretrained weights and scGPT assets",
        "Large binaries; often gitignored partially. `checkpoints/` for CLOP/DiT training outputs.",
    ),
    "models/checkpoints": (
        "Training checkpoints",
        "`CLOP/`, `DiT/`, `ablations/`, `archive/`. `CURRENT_VERSION.txt` may indicate active pointer.",
    ),
    "models/checkpoints/CLOP": (
        "CLOP checkpoints",
        "`best/`, `final/`, `versions/` — align with `04a_train_clop.py` and `configs/clop.yaml`.",
    ),
    "models/checkpoints/DiT": (
        "DiT checkpoints",
        "`best/`, `final/`, `versions/` — align with `04b_train_dit.py` and `configs/dit.yaml`.",
    ),
    "models/checkpoints/ablations": (
        "Ablation checkpoints (models tree)",
        "Mirrors naming under `results/ablations/`; confirm pairing when auditing.",
    ),
    "models/scgpt_pancancer": (
        "scGPT pancancer decoder weights",
        "Cancer-focused checkpoint; kept for ablation presets.",
    ),
    "models/scgpt_human": (
        "scGPT whole-human decoder weights",
        "Default decoder path in pipeline config; required for decode and training cache.",
    ),
    "models/baselines": (
        "Baseline model weights",
        "Stored weights for VAE/scGen/etc. if not only under `results/baselines/`.",
    ),
    "logs": (
        "Training and run logs",
        "Versioned log trees (v6, v7, v8.x); useful for reproducing hyperparameters and failures.",
    ),
    "references": (
        "Bibliography / reference PDFs",
        "External papers or citation assets—keep licensing in mind.",
    ),
    "fonts": (
        "Bundled fonts for figures",
        "May duplicate `scripts/vcd/fonts/`; keep consistent for VCD and matplotlib.",
    ),
    ".github": (
        "GitHub metadata",
        "CI workflows (e.g. `workflows/ci.yml`).",
    ),
    ".github/workflows": (
        "CI workflows",
        "Lint/test on push; extend when adding new test modules.",
    ),
}


def review(
    verdict: str,
    evidence: list[str],
    strengths: list[str],
    gaps: list[str],
    next_checks: list[str],
) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "evidence": evidence,
        "strengths": strengths,
        "gaps": gaps,
        "next_checks": next_checks,
    }


MANUAL_REVIEWS: dict[str, dict[str, Any]] = {
    "": review(
        verdict=(
            "This checkout supports an artifact-level audit: local cached data, "
            "checkpoint histories, benchmark outputs, figures, logs, and a built "
            "article are all present. The overall system looks substantially trained "
            "and documented, but several data-tail, documentation, and result-quality "
            "caveats remain important for later refinement."
        ),
        evidence=[
            "`README.md`, `PIPELINE.md`, `REPRODUCIBILITY.md`, and the generated audit scaffold.",
            "`results/clop_data_audit.json`, `models/checkpoints/CLOP/versions/latest/clop_history.json`, and `models/checkpoints/DiT/versions/latest/dit_history.json`.",
            "`results/benchmark_report.json`, `results/comprehensive_summary.json`, `results/vcd_report_summary.md`, and `articles/clop_dit_biology.tex`.",
        ],
        strengths=[
            "Clear stage separation across data prep, training, inference, evaluation, visualization, and article delivery.",
            "Centralized path resolution via `src.utils.paths` reduces hardcoded-path drift.",
            "Artifact coverage is real rather than hypothetical: processed h5ad files, cached latents, checkpoints, logs, benchmark JSON, figure PDFs, and an article PDF are all present locally.",
            "Training evidence is strong enough to review: CLOP runs the configured 60 epochs and ends around `val_loss ~= 0.0070` with `val_proto_acc ~= 0.9978`, while DiT runs 200 epochs and ends around `val_loss ~= 0.0323` with `val_cosine ~= 0.9898`.",
            "Evaluation is broader than a single metric: embedding quality, downstream biology, benchmarking, and VCD-backed figures all exist.",
        ],
        gaps=[
            "The data audit still returns `WARN`: `167245` cells across `69` cell types and `79` datasets looks broad, but two cell types appear in only one dataset, one type has only `77` cells, and seven text-embedding pairs are unusually confusable.",
            "Documentation around figure/manuscript counts still drifts across `17`, `19`, `20`, and `30`, which weakens result clarity for external reviewers.",
            "Some scientifically important checks remain incomplete or weak: `coverage = 0.0282`, `discriminator_auc ~= 0.656`, `de_mean_logfc_pearson ~= 0.394`, and several `comprehensive_summary.json` sections remain null.",
            "VCD still reports six warnings across four figures and missing live sidecars for `fig04_embedding_space.pdf`, `fig29_embedding_augmentation.pdf`, and `fig30_validation_summary.pdf`.",
        ],
        next_checks=[
            "Clarify whether the remaining weak metrics are acceptable for the current paper version or require more training/data refinement.",
            "Reconcile figure and article numbering across README files, visualization docs, and regeneration scripts.",
            "Record config snapshots, seeds, model revisions, and artifact hashes alongside the published results bundle.",
            "Investigate the tail of the training data and the missing/null validation blocks before claiming the audit is closed.",
        ],
    ),
    ".github": review(
        verdict=(
            "Automation is present and useful, but it is a lightweight regression gate "
            "rather than a full scientific reproducibility gate."
        ),
        evidence=[
            "`.github/workflows/ci.yml`.",
        ],
        strengths=[
            "CI runs on pushes and pull requests to `main` and `master`.",
            "Python `3.10` and `3.11` are both exercised.",
            "The workflow installs the package, lints `src/`, and runs the test suite.",
        ],
        gaps=[
            "No coverage reporting, type checking, formatting gate, or dependency review is visible here.",
            "No end-to-end pipeline smoke test or GPU-aware validation is present.",
        ],
        next_checks=[
            "Add at least one minimal pipeline smoke test built around tiny fixtures or cached toy arrays.",
            "Consider separate fast PR checks and slower nightly checks for heavier evaluation paths.",
        ],
    ),
    ".github/workflows": review(
        verdict=(
            "The current workflow provides a reasonable fast-fail signal for imports, "
            "config integrity, and unit behavior, but it is not sufficient to certify "
            "training, data, or result quality."
        ),
        evidence=[
            "`ci.yml`.",
        ],
        strengths=[
            "Uses a simple and maintainable single-job workflow.",
            "Runs both linting and tests after `pip install -e .`.",
        ],
        gaps=[
            "`flake8` targets `src/` only, so `tests/` and `scripts/` are not linted in CI.",
            "`pytest -x` stops on the first failure, which can hide the full failure surface of a broken branch.",
            "The workflow never exercises real training, cached data, figure generation, or article build steps.",
        ],
        next_checks=[
            "Add targeted checks for `scripts/` and `tests/` if those folders are part of the supported developer surface.",
            "Add a scheduled job for figure regeneration or benchmark-contract validation when local artifacts are available.",
        ],
    ),
    "configs": review(
        verdict=(
            "The main CLOP and DiT configs are sufficiently detailed for code-level "
            "training reproduction, but they are not yet a complete audit trail on their own."
        ),
        evidence=[
            "`configs/README.md`, `clop.yaml`, `dit.yaml`, `models.yaml`, and `pipeline.yaml`.",
            "`configs/baselines/*.yaml`, `thresholds.yaml`, `marker_genes.yaml`, and `ood_test_prompts.yaml`.",
        ],
        strengths=[
            "Core hyperparameters, split strategy, cache path, and checkpoint targets are explicit for CLOP and DiT.",
            "The configs capture both model training knobs and downstream evaluation assets such as thresholds and marker panels.",
            "Baseline and prompt subfolders exist rather than hiding those settings inside code.",
        ],
        gaps=[
            "The main CLOP and DiT YAMLs do not declare `seed`, even though the training scripts default to `42` at the CLI layer.",
            "Path authority is split between `pipeline.yaml` and per-stage YAMLs, which can drift silently.",
            "`models.yaml` names Hugging Face models but does not pin revisions or commit hashes.",
            "`dit.yaml` and `clop.yaml` config descriptions should be kept in sync.",
        ],
        next_checks=[
            "Add explicit `seed` keys to every trainable config.",
            "Choose one path authority model and document when duplication is allowed.",
            "Pin external model revisions or record them in emitted run metadata.",
        ],
    ),
    "configs/baselines": review(
        verdict=(
            "Baseline configuration coverage exists, but not all baselines are equally reproducible."
        ),
        evidence=[
            "`embedding_vae.yaml` and `scvi.yaml`.",
        ],
        strengths=[
            "Both baseline configs declare an explicit `seed`.",
            "The VAE baseline reads directly from the deduplicated cache used elsewhere in the project.",
        ],
        gaps=[
            "The scVI baseline depends on derived files under `results/` rather than raw expression inputs, so its provenance depends on an upstream export step that is not captured here.",
            "This folder does not explain how baseline outputs should be compared to the main model beyond the artifact contract in `src/evaluation/README.md`.",
        ],
        next_checks=[
            "Document which script exports `results/real_expression.npy` and related label/gene-name files.",
            "Add example provenance metadata for baseline outputs so later audits can compare methods fairly.",
        ],
    ),
    "configs/prompts": review(
        verdict=(
            "Prompt robustness assets are present and useful, but prompt coverage is not "
            "yet documented strongly enough to call it exhaustive."
        ),
        evidence=[
            "`default_prompts.json`, `marker_dropped_prompts.json`, `context_shuffled_prompts.json`, and `expanded_variants.json`.",
            "`robustness_prompt_variants.example.json`.",
        ],
        strengths=[
            "The prompt set goes beyond a single default phrasing and includes ablated and shuffled variants.",
            "The expanded variant file suggests the project is already thinking about prompt sensitivity systematically.",
        ],
        gaps=[
            "This folder does not state how short prompt keys map onto the canonical biological labels used by downstream evaluation.",
            "There is no explicit coverage rationale for why these prompt variants are sufficient for robustness claims.",
        ],
        next_checks=[
            "Document the mapping between prompt keys and evaluation cell-type names.",
            "Write down which prompt subsets support each robustness claim in the manuscript.",
        ],
    ),
    "docs": review(
        verdict=(
            "Documentation is helpful and above average for a research repo, but it is "
            "not synchronized tightly enough to serve as a single unambiguous audit source."
        ),
        evidence=[
            "`docs/README.md`, `figure_organization.md`, `QUICK_START.md`, `dataset_expansion.md`, and `ablation_methodology.md`.",
        ],
        strengths=[
            "There is dedicated documentation for figure organization, quick start, ablations, and dataset expansion.",
            "The docs folder points readers back to root-level pipeline and reproducibility guides instead of duplicating everything.",
        ],
        gaps=[
            "`docs/README.md` says the canonical figure mapping covers `17` article figures, while other repo docs and scripts refer to `20` or `30` figures.",
            "`dataset_expansion.md` exists but is not indexed in `docs/README.md`.",
            "There are multiple plausible 'canonical' regeneration commands across the docs and pipeline scripts.",
        ],
        next_checks=[
            "Make one document authoritative for figure numbering and one command authoritative for regeneration.",
            "Index every maintained markdown document in `docs/README.md`.",
        ],
    ),
    "fonts": review(
        verdict=(
            "Local font fallback is practical for figure reproducibility, but this folder "
            "needs a clearer source-of-truth and licensing story."
        ),
        evidence=[
            "Bundled Arial `.ttf` files under `fonts/`.",
            "`src/visualization/style.py`, which registers fonts from both `fonts/` and `scripts/vcd/fonts/`.",
        ],
        strengths=[
            "Matplotlib can resolve project-local fonts even on clean systems.",
            "Figure style and VCD font discovery already know how to look here.",
        ],
        gaps=[
            "The same Arial assets are duplicated under `scripts/vcd/fonts/`, which invites silent drift.",
            "`REPRODUCIBILITY.md` says no proprietary fonts are distributed, but this checkout includes Arial files.",
        ],
        next_checks=[
            "Decide whether `fonts/` or `scripts/vcd/fonts/` is canonical and sync or remove the duplicate copy.",
            "Clarify the repository's font redistribution and fallback-font policy in the docs.",
        ],
    ),
    "models": review(
        verdict=(
            "The model directory contains enough checkpoint material to review training "
            "completion and version lineage for the current CLOP and DiT runs."
        ),
        evidence=[
            "`models/checkpoints/CURRENT_VERSION.txt`.",
            "`models/checkpoints/CLOP/best/clop_best.pth`, `models/checkpoints/CLOP/versions/latest/{clop_best.pth,clop_final.pth,clop_history.json}`.",
            "`models/checkpoints/DiT/best/dit_best.pth`, `models/checkpoints/DiT/versions/latest/{dit_best.pth,dit_final.pth,dit_history.json}`.",
        ],
        strengths=[
            "A lightweight version pointer exists in `CURRENT_VERSION.txt`.",
            "Both CLOP and DiT have best/final checkpoints and versioned history JSON rather than a single opaque weight file.",
        ],
        gaps=[
            "`generation_metadata.json` points to `models/checkpoints/dit_best.pth`, while the concrete stored checkpoint is nested under `models/checkpoints/DiT/...`, which is a path-drift risk.",
            "Checkpoint folders still need explicit hashes, config snapshots, and seed metadata if they are to function as release-grade evidence bundles.",
            "`CURRENT_VERSION.txt` names CLOP, while DiT is versioned separately under `v2.0`; that split is reasonable but should be made explicit in human-facing docs.",
        ],
        next_checks=[
            "Record which exact checkpoint pairs correspond to the reported paper version.",
            "Add hashes or manifest files for the best/final checkpoints and align all downstream metadata with the nested checkpoint paths.",
        ],
    ),
    "scripts": review(
        verdict=(
            "The scripts tree is operationally broad and well organized, but it still has "
            "more than one plausible execution path for major tasks."
        ),
        evidence=[
            "`scripts/README.md`.",
            "`scripts/pipeline/run_pipeline.py` plus the subfolder inventories under `data_prep/`, `training/`, `inference/`, `analysis/`, `baselines/`, and `vcd/`.",
        ],
        strengths=[
            "The stage-oriented folder structure makes the intended workflow easy to follow.",
            "A canonical runner exists instead of requiring manual copy-paste of every step.",
        ],
        gaps=[
            "The README omits some scripts that exist on disk, such as `train_scgen_baseline.py`.",
            "Some usage strings still refer to old top-level paths rather than the current subfolder layout.",
            "The canonical runner does not include every scientifically important helper script.",
        ],
        next_checks=[
            "Mark each script as canonical, auxiliary, exploratory, or legacy.",
            "Normalize usage examples so they match the current tree exactly.",
        ],
    ),
    "scripts/analysis": review(
        verdict=(
            "This folder contains many of the strongest audit-oriented utilities for result "
            "interpretation, but most of that value depends on result artifacts that are not present here."
        ),
        evidence=[
            "`decode_expression.py`, `diversity_diagnostics.py`, `conditioning_analysis.py`, and `audit_data.py`.",
            "Figure-generation and downstream-support scripts in the same folder.",
        ],
        strengths=[
            "The folder covers decoding, diversity, conditioning, marker analysis, bootstrap CIs, and explicit data auditing.",
            "`audit_data.py` is a useful pre-training sanity check rather than relying on training failure to reveal data problems.",
        ],
        gaps=[
            "`audit_data.py` is not part of `run_pipeline.py`, so it is easy to skip.",
            "No `results/` artifacts are available in this checkout, so the analysis scripts cannot be used here to judge current output quality.",
        ],
        next_checks=[
            "Run and store `results/clop_data_audit.json` whenever a new cache build is produced.",
            "Document which analysis outputs are required evidence for manuscript-level claims.",
        ],
    ),
    "scripts/baselines": review(
        verdict=(
            "Baseline method support exists, but it currently behaves more like a side "
            "workflow than a first-class branch of the main pipeline."
        ),
        evidence=[
            "`train_embedding_vae_baseline.py`, `train_scvi_baseline.py`, and `train_scgen_baseline.py`.",
        ],
        strengths=[
            "The repo does not rely on a single weak baseline.",
            "Baseline-specific configs are separated cleanly into `configs/baselines/`.",
        ],
        gaps=[
            "Baseline training is not orchestrated by `run_pipeline.py`.",
            "The main scripts README does not fully describe the baseline surface.",
            "No baseline artifacts are present locally, so comparative sufficiency cannot be checked.",
        ],
        next_checks=[
            "Add a documented baseline regeneration sequence and expected output contract.",
            "Persist baseline metadata in a way that can be compared automatically to the main model.",
        ],
    ),
    "scripts/data_prep": review(
        verdict=(
            "The data-prep stage is substantial and plausibly sufficient to construct the "
            "training cache, but the actual prepared data is absent from this checkout."
        ),
        evidence=[
            "The numbered `00`-`03` scripts plus helper scripts like `02b_enrich_descriptions.py`, `generate_prompt_variants.py`, and `02d_rebuild_v63_cache.py`.",
        ],
        strengths=[
            "The folder separates integration, caption generation, caching, deduplication, and preprocessing.",
            "The numbered scripts create a legible main path even though auxiliary helpers also exist.",
        ],
        gaps=[
            "No raw or processed data artifacts are present locally, so data sufficiency can only be inferred from the code and docs.",
            "Several helper scripts are valuable but sit outside the canonical stage list, which can hide important provenance steps.",
        ],
        next_checks=[
            "Store dataset manifests, accession lists, and cache hashes beside the generated cache.",
            "Document which helper scripts are mandatory for the current paper version and which are exploratory.",
        ],
    ),
    "scripts/inference": review(
        verdict=(
            "Inference and evaluation entrypoints are broad enough for rich analysis, but "
            "the folder currently supports multiple parallel workflows that need clearer signposting."
        ),
        evidence=[
            "`05_inference.py`, `06_evaluate.py`, `06_cell2cell_inference.py`, `08_biological_validation.py`, and `generate_embeddings.py`.",
        ],
        strengths=[
            "There is both a canonical embedding-generation path and more interactive end-to-end inference scripts.",
            "Biological validation is represented here rather than being hidden inside a notebook.",
        ],
        gaps=[
            "`run_pipeline.py` only uses `generate_embeddings.py`, so the relationship between the other scripts and the main results path is easy to misunderstand.",
            "Some usage blocks still reference outdated path layouts.",
        ],
        next_checks=[
            "Document which script should be used for paper regeneration versus ad hoc exploration.",
            "Add a small decision table explaining when to use each inference entrypoint.",
        ],
    ),
    "scripts/pipeline": review(
        verdict=(
            "This folder is the closest thing to a canonical operational contract, but it "
            "still carries some documentation drift and coverage gaps."
        ),
        evidence=[
            "`run_pipeline.py`, `run_regeneration.py`, `build_article.sh`, and `generate_folder_audit_reports.py`.",
        ],
        strengths=[
            "There is a single ordered stage list for the main CLOP-DiT workflow.",
            "Figure regeneration and article delivery are automated rather than left as manual follow-up.",
        ],
        gaps=[
            "`run_pipeline.py` mentions article delivery of `20` PDFs, while `run_regeneration.py` describes `30` figures and docs elsewhere mention `17` or `19`.",
            "Important audit helpers such as `audit_data.py`, baseline runs, biological validation, and decoder/Cell2Cell training are outside the canonical runner.",
            "There are multiple orchestration surfaces, including shell wrappers, which can drift over time.",
        ],
        next_checks=[
            "Reconcile figure counts and declare one canonical paper-regeneration path.",
            "Add preflight checks for expected checkpoints and result prerequisites before each stage runs.",
        ],
    ),
    "scripts/training": review(
        verdict=(
            "The training script surface is rich enough for ablations and robustness work, "
            "but only part of it is treated as canonical by the pipeline runner."
        ),
        evidence=[
            "`04a_train_clop.py`, `04b_train_dit.py`, `04c_train_cell2cell.py`, `run_5fold_cv.py`, `run_ablation_study.py`, `run_robustness_experiments.py`, and `train_decoder.py`.",
        ],
        strengths=[
            "The repo distinguishes main training, CV, ablation, robustness, decoder, and Cell2Cell workflows.",
            "The main CLOP and DiT entrypoints already apply seeding and load YAML configs cleanly.",
        ],
        gaps=[
            "Only CLOP and DiT training are part of `run_pipeline.py`; the rest require separate orchestration.",
            "No local logs or checkpoints are present here, so actual training sufficiency cannot be judged from this checkout.",
            "Some usage strings still reflect a pre-subfolder script layout.",
        ],
        next_checks=[
            "Persist history JSON, seed, config snapshot, and checkpoint identifiers for every training mode.",
            "Clarify which training modes are paper-critical and which are exploratory extensions.",
        ],
    ),
    "scripts/vcd": review(
        verdict=(
            "VCD is a strong readability and layout-quality layer, but it should not be "
            "confused with scientific result validation."
        ),
        evidence=[
            "`scripts/pipeline/run_regeneration.py` and the VCD package layout.",
        ],
        strengths=[
            "Figure QA is automated rather than depending on manual eyeballing alone.",
            "The project has an explicit place for visual readability checks and follow-up reports.",
        ],
        gaps=[
            "VCD validates presentation quality, not biological correctness or statistical sufficiency.",
            "The font assets used by VCD are duplicated elsewhere in the repo.",
        ],
        next_checks=[
            "Keep VCD reports near the corresponding figure outputs.",
            "Document VCD as a secondary audit layer beneath data, training, and evaluation evidence.",
        ],
    ),
    "scripts/vcd/fonts": review(
        verdict=(
            "This folder is a utility asset mirror rather than a scientific evidence source."
        ),
        evidence=[
            "Bundled Arial `.ttf` files under `scripts/vcd/fonts/`.",
        ],
        strengths=[
            "It reduces the chance that VCD runs fail on systems without Arial installed.",
        ],
        gaps=[
            "The same font set also exists under `fonts/`, so updates can drift.",
            "Licensing and redistribution questions should be answered once, not per duplicate folder.",
        ],
        next_checks=[
            "Keep this folder synchronized with `fonts/` or replace it with one canonical shared location.",
        ],
    ),
    "src": review(
        verdict=(
            "The importable package is structured like a serious research codebase and "
            "already separates most major concerns cleanly."
        ),
        evidence=[
            "`src/README.md`, `src/architecture/README.md`, `src/evaluation/README.md`, and `src/visualization/README.md`.",
        ],
        strengths=[
            "Architecture, data, evaluation, training, experiments, utils, and visualization are separated into coherent subpackages.",
            "The folder-level docs make the package easier to audit than a script-only project.",
        ],
        gaps=[
            "Runtime sufficiency still depends on missing local artifacts rather than code structure alone.",
            "Package documentation also inherits the repo-wide figure-count drift problem.",
        ],
        next_checks=[
            "Keep package docs synchronized with the actual delivery manifest and paper numbering.",
            "Continue moving script-only logic into importable APIs where possible.",
        ],
    ),
    "src/architecture": review(
        verdict=(
            "The architecture layer appears sufficient for the intended CLOP + DiT + "
            "decoder stack, with optional extension paths for Cell2Cell work."
        ),
        evidence=[
            "`src/architecture/README.md` and the exported symbols in `src/architecture/__init__.py`.",
        ],
        strengths=[
            "The main model components are explicit rather than entangled in a single monolithic file.",
            "The architecture layer already exposes both the primary path and auxiliary modules such as Cell2Cell.",
        ],
        gaps=[
            "This folder alone cannot reveal which exact model pairing produced the published results.",
            "Checkpoint/config provenance must still be carried by run metadata outside the module definitions.",
        ],
        next_checks=[
            "Record the active encoder/decoder preset and checkpoint lineage whenever a run is emitted.",
        ],
    ),
    "src/architecture/clop": review(
        verdict=(
            "The CLOP submodule is one of the stronger technical areas in the repo and "
            "looks flexible enough for the current contrastive alignment claims."
        ),
        evidence=[
            "`aligner.py`, `losses.py`, `configs/clop.yaml`, and the validation hooks in `src/training/train_clop.py`.",
        ],
        strengths=[
            "Multiple alignment objectives exist, including Prototype-SigLIP, plain SigLIP, and InfoNCE.",
            "Whitening and projection logic are kept with the contrastive module rather than scattered through scripts.",
        ],
        gaps=[
            "Documentation can drift around which loss is the true default, so audits must verify the configured loss rather than trusting prose alone.",
            "The published training evidence is not bundled with the module in this checkout.",
        ],
        next_checks=[
            "Store the exact loss type, temperature mode, and preprocessing flags in emitted histories and checkpoint metadata.",
        ],
    ),
    "src/data_pipeline": review(
        verdict=(
            "This is one of the strongest folders for reproducible structure, but its "
            "claims about dataset sufficiency cannot be validated here without the actual artifacts."
        ),
        evidence=[
            "`dataset.py`, `cache_builder.py`, and the repo docs describing `data/cached_latents`.",
            "`scripts/analysis/audit_data.py` as the intended cache sanity check.",
        ],
        strengths=[
            "The data layer distinguishes dataset loading, cache building, preprocessing, and sampling.",
            "Deduplicated and preprocessed cache naming reduces ambiguity about what the trainers consume.",
        ],
        gaps=[
            "No raw h5ad files, cached arrays, or manifest files were detected locally.",
            "Dataset sufficiency is therefore inferred from docs, not from independently inspectable artifacts.",
        ],
        next_checks=[
            "Version and hash the cached arrays and accession manifest used for each paper release.",
            "Persist the output of `audit_data.py` with the cache build.",
        ],
    ),
    "src/evaluation": review(
        verdict=(
            "Evaluation breadth is a major strength of the repository and should support "
            "a strong audit once result artifacts are populated."
        ),
        evidence=[
            "`src/evaluation/README.md`, `downstream_biology.py`, `baseline_registry.py`, and `model_benchmarking.py`.",
        ],
        strengths=[
            "The folder covers embedding metrics, downstream biology, biological validation, and multi-method benchmarking.",
            "There is an explicit artifact contract for baseline methods rather than implicit ad hoc loading.",
        ],
        gaps=[
            "No local `results/` outputs are available, so evaluation adequacy cannot be checked on actual runs here.",
            "Some evaluation paths depend on external data and optional tools that CI does not exercise.",
        ],
        next_checks=[
            "Bundle benchmark outputs, downstream JSON, and biological-validation summaries with each release candidate.",
            "Add artifact-level regression tests for the benchmark contract where possible.",
        ],
    ),
    "src/evaluation/biological_validation": review(
        verdict=(
            "This folder is critical for answering whether results are biologically clear "
            "and plausible, and its presence is a major strength of the codebase."
        ),
        evidence=[
            "The biological validation package structure documented in `src/evaluation/README.md`.",
        ],
        strengths=[
            "The package targets marker enrichment, DE behavior, expression-space similarity, and figure generation rather than a single superficial score.",
            "It supports end-to-end text-to-cell generation rather than only embedding-space inspection.",
        ],
        gaps=[
            "The folder depends on external reference datasets and optional tooling that are not present in this checkout.",
            "Biological validation is not part of the default `run_pipeline.py` stage list.",
        ],
        next_checks=[
            "Document the required datasets, expected outputs, and acceptance thresholds per validation run.",
            "Promote the most important biological validation path into the canonical release checklist.",
        ],
    ),
    "src/experiments": review(
        verdict=(
            "The experiments package is useful for research extension work, but it is not "
            "yet clean enough to be treated as canonical evidence without caveats."
        ),
        evidence=[
            "`ood_evaluation.py` and the subpackage description in `src/__init__.py`.",
        ],
        strengths=[
            "OOD and rare-cell augmentation ideas are implemented as code rather than only as notes.",
        ],
        gaps=[
            "`ood_evaluation.py` reaches into `scripts/inference/05_inference.py` via `importlib` and `sys.path` injection, which is fragile.",
            "Experimental code can blur the line between supported workflow and prototype workflow if not labeled carefully.",
        ],
        next_checks=[
            "Refactor cross-package imports into a stable API or explicitly label this folder as experimental-only.",
        ],
    ),
    "src/training": review(
        verdict=(
            "The trainer implementations include more than just loss minimization and are "
            "a meaningful source of rigor, even though this checkout lacks the resulting artifacts."
        ),
        evidence=[
            "`train_clop.py` validation hooks for `compute_all_quality_metrics` and optional early stopping.",
            "`train_dit.py` generation-time evaluation via `GenerationMetrics.full_evaluation`.",
            "`reproducibility.py` and scheduler utilities.",
        ],
        strengths=[
            "CLOP training tracks validation accuracy and embedding-space quality, not only loss.",
            "DiT training includes EMA and periodic generation evaluation rather than a train-loss-only loop.",
            "The package includes explicit reproducibility helpers instead of leaving seeding ad hoc.",
        ],
        gaps=[
            "No history files, checkpoints, or logs are present here, so actual convergence cannot be judged from the current checkout.",
            "Config-level seeding is still implicit for the main models because the YAMLs omit `seed`.",
        ],
        next_checks=[
            "Make config snapshots and seeds mandatory in every saved training bundle.",
            "Persist trainer outputs in a form that later audits can compare automatically.",
        ],
    ),
    "src/utils": review(
        verdict=(
            "Utilities are one of the main reasons the repo remains auditable at all, "
            "especially around paths and run metadata."
        ),
        evidence=[
            "`paths.py` and the utility surface described in `src/README.md`.",
        ],
        strengths=[
            "Path resolution follows a sensible env -> YAML -> default hierarchy.",
            "Threshold and marker config loaders centralize common configuration instead of hardcoding it repeatedly.",
            "An experiment-tracker module exists, which is exactly the right direction for future auditability.",
        ],
        gaps=[
            "Not every script clearly uses the experiment tracker, so run metadata capture may still be uneven.",
            "Some older scripts and docs still show hardcoded path examples that compete with `paths.py`.",
        ],
        next_checks=[
            "Standardize experiment tracking across all train/eval/generation entrypoints.",
            "Remove or update stale examples that bypass `src.utils.paths`.",
        ],
    ),
    "src/visualization": review(
        verdict=(
            "The figure system is rich and publication-oriented, but result clarity is "
            "held back by inconsistent figure-count narratives across the repo."
        ),
        evidence=[
            "`src/visualization/README.md`, `style.py`, `article_delivery.py`, and `results_visualizer.py`.",
            "`scripts/pipeline/run_regeneration.py`.",
        ],
        strengths=[
            "Style, panel rendering, orchestration, and article delivery are explicitly separated.",
            "Fonts and visual QA hooks are already integrated into the visualization flow.",
        ],
        gaps=[
            "The folder README says `19` panels, the article-delivery description mentions `17` article figure basenames, and regeneration code talks about `30` figures.",
            "No rendered figure artifacts are present locally, so actual visual clarity cannot be judged here.",
        ],
        next_checks=[
            "Reconcile panel, figure, and article manifest terminology in one canonical place.",
            "Attach VCD summaries and figure inventories to each released result bundle.",
        ],
    ),
    "tests": review(
        verdict=(
            "The test suite is useful and relevant, but it is not broad enough to certify "
            "training sufficiency, dataset sufficiency, or full-paper reproducibility by itself."
        ),
        evidence=[
            "Eight test modules in `tests/` and the CI workflow that runs them.",
            "`test_visualization.py`, which illustrates the current emphasis on imports and path sanity.",
        ],
        strengths=[
            "Tests cover configs, paths, visualization imports, article delivery, embedding quality, layout, and CLOP loss behavior.",
            "The suite gives quick feedback on structural regressions without requiring large artifacts.",
        ],
        gaps=[
            "Most tests are smoke or unit tests rather than end-to-end pipeline tests.",
            "CI stops at the first failure and never exercises real artifacts, heavy evaluation, or figure regeneration.",
        ],
        next_checks=[
            "Add a tiny fixture-based end-to-end smoke test for cache -> train -> generate -> evaluate.",
            "Add artifact-contract tests for baseline outputs and downstream result JSON where practical.",
        ],
    ),
    "data": review(
        verdict=(
            "The local data bundle is broad enough to support the current 69-type audit, "
            "but it is not uniformly strong across every type and dataset."
        ),
        evidence=[
            "`results/clop_data_audit.json`.",
            "`data/processed_h5ad/` and `data/cached_latents/` inventories.",
            "`REPRODUCIBILITY.md` and the default cache path in the configs.",
        ],
        strengths=[
            "`clop_data_audit.json` reports `167245` cells, `69` cell types, and `79` datasets, with all `69` types present in both train and validation under the stratified split.",
            "Embedding norms are effectively unit-normalized and the mean pairwise cosine values do not suggest collapse.",
            "`data/processed_h5ad/` contains `80` processed `.h5ad` files and `data/cached_latents/` contains the expected cached arrays and metadata sidecars.",
        ],
        gaps=[
            "The data audit status is `WARN`, not `PASS`.",
            "Two cell types exist in only one dataset, one type has only `77` cells, and seven text-embedding pairs have cosine similarity above `0.5`.",
            "There is a mild accounting mismatch between `80` processed `.h5ad` files on disk and `79` datasets reported by `clop_data_audit.json`, which deserves clarification.",
        ],
        next_checks=[
            "Investigate the `80`-vs-`79` dataset-count difference and document whether one source is merged or excluded intentionally.",
            "Strengthen the thin tail of rare cell types before claiming uniform data sufficiency across all 69 classes.",
            "Map the seven confusable text pairs back to biological labels and decide whether caption refinement is needed.",
        ],
    ),
    "results": review(
        verdict=(
            "Results are present, structured, and broadly reviewable, but the current "
            "bundle still mixes strong benchmark evidence with stale documentation and a few incomplete validation blocks."
        ),
        evidence=[
            "`benchmark_report.json`, `comprehensive_summary.json`, `generation_metadata.json`, and `generation_metrics.json`.",
            "`results/README.md`, `results/figures/`, and `vcd_report_summary.md`.",
        ],
        strengths=[
            "The folder contains a real benchmark report, experiment registry, data audit, generation metadata, downstream outputs, and figure PDFs.",
            "CLOP-DiT leads the stored composite benchmark with `composite_score ~= 0.838`, well above the next-best stored method.",
            "The benchmarked primary model evaluates all `69` cell types with `fraction_collapsed = 0.0`, `mean_centroid_cosine ~= 0.897`, and `diversity_ratio ~= 0.944`.",
        ],
        gaps=[
            "`results/README.md` is stale: it still documents `panel_*` outputs and the wrong regeneration script path (`scripts/regenerate_report.sh` instead of `scripts/pipeline/regenerate_report.sh`).",
            "Several result sections still signal incomplete validation rather than closure: `coverage = 0.0282`, `discriminator_auc ~= 0.656`, `de_mean_logfc_pearson ~= 0.394`, OOD marker-hit rate is only `0.1667`, and embedding augmentation is explicitly marked `negative_result`.",
            "`comprehensive_summary.json` still contains null-valued blocks for cross-dataset validation, expanded DE concordance, and marker completeness.",
            "VCD reports six warnings and three figures missing live audit sidecars.",
        ],
        next_checks=[
            "Refresh `results/README.md` so the documented file names and commands match the live 30-figure workflow.",
            "Decide which null-valued validation sections are intentionally pending versus silently broken.",
            "Resolve the remaining VCD warnings and regenerate missing live sidecars before calling the presentation audit complete.",
        ],
    ),
    "logs": review(
        verdict=(
            "Logs are present and versioned, so they can support convergence and provenance review if paired carefully with checkpoints."
        ),
        evidence=[
            "The root `logs/` inventory plus versioned subfolders such as `v6/`, `v7/`, `v8.1/`, `v8.2/`, `v8.3/`, and `5fold_cv/`.",
        ],
        strengths=[
            "The repo reserves a dedicated log area rather than relying entirely on ephemeral terminal output.",
            "Multiple versioned log bundles are already present, which is better than a single rolling log file.",
        ],
        gaps=[
            "The log tree still needs explicit linkage back to checkpoint folders and config snapshots to be maximally useful.",
            "A versioned log directory does not by itself guarantee that the right run was used for the paper figures.",
        ],
        next_checks=[
            "Cross-link each paper-critical checkpoint bundle to its matching log file and history JSON.",
            "Keep log naming conventions stable enough that later auditors can trace the active run without guesswork.",
        ],
    ),
    "articles": review(
        verdict=(
            "The manuscript layer is populated and reviewable: the main TeX source, built "
            "PDF, and delivered article figures are all present."
        ),
        evidence=[
            "`articles/clop_dit_biology.tex`, `articles/clop_dit_biology.pdf`, and `articles/figures/`.",
            "`src/visualization/article_delivery.py` and the delivered figure inventory.",
        ],
        strengths=[
            "The repo intends article delivery to be automated from generated figures rather than hand-copied.",
            "The built manuscript PDF and the article figure set are already present locally.",
        ],
        gaps=[
            "`article_delivery.py` still describes itself as the source of truth for `20` article figures, while the live manifest and article include `30` figures.",
            "A populated article tree does not eliminate the need to verify that every delivered figure matches the latest results and VCD state.",
        ],
        next_checks=[
            "Reconcile the human-facing documentation with the actual 30-figure manifest.",
            "Verify that every `\\includegraphics` entry in the article is backed by the current delivered PDF and not a stale copy.",
        ],
    ),
    "articles_elsevier": review(
        verdict=(
            "A parallel article tree can be useful, but it introduces drift risk unless it "
            "is maintained with a very explicit ownership model."
        ),
        evidence=[
            "The folder's presence in the audit scaffold and repository root inventory.",
        ],
        strengths=[
            "It suggests the project anticipates multiple manuscript targets.",
        ],
        gaps=[
            "Parallel article trees are a common source of stale figures, divergent numbering, and conflicting prose.",
        ],
        next_checks=[
            "Document whether this folder is active, derived, or archival.",
        ],
    ),
    "articles_elsevier copy": review(
        verdict=(
            "A duplicated article working copy is high-risk for audit drift and should be "
            "treated as a temporary workspace, not a long-term source of truth."
        ),
        evidence=[
            "The folder's presence in the audit scaffold and repository root inventory.",
        ],
        strengths=[
            "It may preserve an in-progress variant without blocking work in the primary article tree.",
        ],
        gaps=[
            "The name itself signals duplication rather than controlled versioning.",
            "A copied article tree can diverge silently from both `articles/` and `articles_elsevier/`.",
        ],
        next_checks=[
            "Replace copy-based branching with a documented variant workflow or archive/delete the duplicate tree.",
        ],
    ),
    "references": review(
        verdict=(
            "This folder matters more for manuscript traceability than for model-quality "
            "audit, but it still deserves basic provenance and licensing hygiene."
        ),
        evidence=[
            "The folder's declared role in the audit scaffold.",
        ],
        strengths=[
            "Keeping reference assets separate from code is structurally sensible.",
        ],
        gaps=[
            "Reference licensing and synchronization with citation metadata are not visible from the current code-only audit.",
        ],
        next_checks=[
            "Track citation-source provenance and any redistribution limits for bundled papers or assets.",
        ],
    ),
}


def has_any_pattern(base: Path, patterns: tuple[str, ...]) -> bool:
    if not base.exists():
        return False
    for pattern in patterns:
        try:
            next(base.rglob(pattern))
            return True
        except StopIteration:
            continue
    return False


def collect_artifact_status(repo: Path) -> dict[str, bool]:
    return {
        "data": has_any_pattern(repo / "data", ("*.npy", "*.h5ad", "*.h5", "*.csv", "*.tsv", "*.json", "*.txt")),
        "models": has_any_pattern(repo / "models", ("*.pth", "*.pt", "*.ckpt")),
        "results": has_any_pattern(repo / "results", ("*.json", "*.pdf", "*.png", "*.csv", "*.md")),
        "logs": has_any_pattern(repo / "logs", ("*.log", "*.txt", "*.json")),
        "articles": has_any_pattern(repo / "articles", ("*.tex", "*.pdf")),
    }


def has_non_review_file(dir_path: Path) -> bool:
    if not dir_path.exists():
        return False
    for root, _, files in os.walk(dir_path):
        for name in files:
            if name != "FOLDER_REVIEW.md":
                return True
    return False


def clone_review(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": entry["verdict"],
        "evidence": list(entry["evidence"]),
        "strengths": list(entry["strengths"]),
        "gaps": list(entry["gaps"]),
        "next_checks": list(entry["next_checks"]),
    }


def prefix_review(rel: str, dir_path: Path) -> dict[str, Any] | None:
    has_files = has_non_review_file(dir_path)
    if rel.startswith("data/"):
        verdict = (
            "This is a data-bearing subfolder; its sufficiency depends on the actual local "
            "artifacts rather than code structure alone."
        )
        if not has_files:
            verdict = (
                "This data-bearing subfolder is currently empty or audit-only in this checkout, "
                "so its contents cannot support data-sufficiency claims yet."
            )
        return review(
            verdict=verdict,
            evidence=[
                "Directory inventory plus the parent `data/` audit.",
            ],
            strengths=[
                "The parent pipeline already documents where processed h5ad and cached latent artifacts belong.",
            ],
            gaps=[
                "Data provenance, split coverage, and artifact hashes must come from the files in this folder, not from folder names alone.",
            ],
            next_checks=[
                "Verify accession manifests, cache hashes, and `audit_data.py` output for this specific subtree.",
            ],
        )
    if rel.startswith("results/"):
        verdict = (
            "This results subfolder should carry claim-level evidence, but no local payload "
            "was detected here during audit generation."
        )
        if has_files:
            verdict = (
                "This results subfolder contains local outputs and should be treated as an "
                "artifact-level evidence source during subsequent audits."
            )
        return review(
            verdict=verdict,
            evidence=[
                "Directory inventory plus the parent `results/` audit.",
            ],
            strengths=[
                "The surrounding codebase already expects structured result artifacts rather than untracked ad hoc files.",
            ],
            gaps=[
                "Folder existence alone does not prove metric quality, figure clarity, or result provenance.",
            ],
            next_checks=[
                "Check timestamps, config snapshots, and benchmark/downstream JSON inside this subtree before using it as evidence.",
            ],
        )
    if rel.startswith("models/"):
        verdict = (
            "This model subtree matters for checkpoint provenance, but no local binary payload "
            "was detected here during audit generation."
        )
        if has_files:
            verdict = (
                "This model subtree contains local artifacts and should be checked against config version, seed, and training history metadata."
            )
        return review(
            verdict=verdict,
            evidence=[
                "Directory inventory plus the parent `models/` audit.",
            ],
            strengths=[
                "The repo already separates model storage from source code and configs.",
            ],
            gaps=[
                "Without explicit pairing to config and history files, checkpoint folders can become hard to trust.",
            ],
            next_checks=[
                "Verify checkpoint names, hashes, and the exact config/history bundle used to create them.",
            ],
        )
    if rel.startswith("articles/"):
        verdict = (
            "This article subtree is part of the presentation layer and should be audited "
            "for figure numbering and manuscript consistency when populated."
        )
        if not has_files:
            verdict = (
                "This article subtree is currently empty or audit-only in this checkout, so presentation consistency cannot be checked here yet."
            )
        return review(
            verdict=verdict,
            evidence=[
                "Directory inventory plus the parent `articles/` audit.",
            ],
            strengths=[
                "The project intends article delivery to be generated, not manually hand-copied.",
            ],
            gaps=[
                "A missing or stale manuscript subtree can hide numbering drift even when figures exist elsewhere.",
            ],
            next_checks=[
                "Check includes, symlinks, and figure basenames against the delivery manifest.",
            ],
        )
    if rel.startswith("logs/"):
        verdict = (
            "This log subtree should capture convergence and failure evidence; if it stays empty, later training audits will stay weak."
        )
        if not has_files:
            verdict = (
                "No local log payload was detected here, so this subtree currently contributes no direct training evidence."
            )
        return review(
            verdict=verdict,
            evidence=[
                "Directory inventory plus the parent `logs/` audit.",
            ],
            strengths=[
                "A dedicated log subtree makes it possible to keep provenance outside transient terminals.",
            ],
            gaps=[
                "Logs are only useful if they are versioned, labeled, and paired with checkpoints.",
            ],
            next_checks=[
                "Store trainer stdout/stderr, history JSON, and failure summaries here per run.",
            ],
        )
    if rel.startswith("references/"):
        return review(
            verdict=(
                "This subtree is relevant for manuscript provenance rather than model-quality proof."
            ),
            evidence=[
                "Directory inventory plus the parent `references/` audit.",
            ],
            strengths=[
                "Separating reference assets from code is structurally sensible.",
            ],
            gaps=[
                "Subtree contents still need basic citation and licensing hygiene.",
            ],
            next_checks=[
                "Track source provenance and redistribution status for each bundled reference asset.",
            ],
        )
    return None


def resolve_review(rel: str, dir_path: Path, artifact_status: dict[str, bool]) -> dict[str, Any]:
    if rel in MANUAL_REVIEWS:
        entry = clone_review(MANUAL_REVIEWS[rel])
        if rel == "":
            entry["gaps"].append(
                "Artifact scan at generation time: "
                + ", ".join(
                    [
                        f"`data/`={'present' if artifact_status['data'] else 'missing'}",
                        f"`models/`={'present' if artifact_status['models'] else 'missing'}",
                        f"`results/`={'present' if artifact_status['results'] else 'missing'}",
                        f"`logs/`={'present' if artifact_status['logs'] else 'missing'}",
                        f"`articles/`={'present' if artifact_status['articles'] else 'missing'}",
                    ]
                )
                + "."
            )
        return entry
    fallback = prefix_review(rel, dir_path)
    if fallback is not None:
        return fallback
    return review(
        verdict=(
            "This folder has a stable structural role, but it has not yet received a "
            "deep manual review beyond inventory and parent-folder context."
        ),
        evidence=[
            "Directory inventory, parent `FOLDER_REVIEW.md`, and any local README discovered here.",
        ],
        strengths=[
            "The generator still records this folder's role and contents for later refinement.",
        ],
        gaps=[
            "No path-specific scientific or operational verdict has been encoded yet.",
        ],
        next_checks=[
            "Promote this folder into `MANUAL_REVIEWS` once it becomes part of a stable release or audit story.",
        ],
    )


def fallback_context(rel_posix: str) -> tuple[str, str]:
    base = Path(rel_posix).name
    if re.match(r"^results/clop_v[\d.]+_\d+$", rel_posix):
        return (
            f"Versioned training/eval bundle: {base}",
            "Timestamped run under `results/`; contains `metrics/` and `checkpoints/`—map to paper tables and `models/checkpoints/` pointers.",
        )
    if rel_posix.startswith("results/ablations/"):
        return (
            f"Ablation run: {base}",
            "Compare metrics and checkpoints to baseline ablation folder; document which config flag this folder disables.",
        )
    if rel_posix.startswith("models/checkpoints/ablations/"):
        return (
            f"Ablation checkpoint: {base}",
            "Weights for a single CLOP (or related) ablation; pair with `results/ablations/{name}/`.",
        )
    if rel_posix.startswith("results/archive/"):
        return (
            f"Archived results: {base}",
            "Historical bundle; prefer current `results/` + `models/checkpoints/` for main paper numbers unless reproducing old versions.",
        )
    if rel_posix.startswith("logs/"):
        return (
            f"Log bundle: {base}",
            "Training or experiment stdout/stderr logs; cross-reference timestamps with checkpoints.",
        )
    if "checkpoints" in rel_posix.split("/"):
        return (
            f"Checkpoints: {base}",
            "Binary `.pt`/`.pth` or similar; verify compatibility with current model code before loading.",
        )
    if rel_posix.startswith("results/robustness/"):
        return (
            f"Robustness run: {base}",
            "Dataset- or cohort-specific robustness output; used in cross-dataset figures.",
        )
    return (
        f"Directory: {base or rel_posix}",
        "Review README or parent `FOLDER_REVIEW.md`; add domain notes when this folder gains a stable role.",
    )


def inventory(dir_path: Path) -> tuple[list[str], Counter[str], list[str]]:
    subdirs = []
    exts: Counter[str] = Counter()
    top_files: list[str] = []
    try:
        entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError:
        return [], exts, []
    for p in entries:
        if p.name.startswith(".") and p.name not in {".gitkeep"}:
            continue
        if p.is_dir():
            if not should_skip_dir(p.name):
                subdirs.append(p.name)
        else:
            _, e = os.path.splitext(p.name)
            key = e.lower() if e else "<noext>"
            exts[key] += 1
            if len(top_files) < 12:
                top_files.append(p.name)
    return subdirs, exts, top_files


def read_readme_hint(dir_path: Path) -> str | None:
    for name in ("README.md", "README.rst", "README.txt"):
        rp = dir_path / name
        if not rp.is_file():
            continue
        try:
            text = rp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None
        title = lines[0][:200]
        return f"Local readme: **{name}** — {title}"
    return None


def rel_posix(repo: Path, dir_path: Path) -> str:
    rel = dir_path.relative_to(repo)
    return "" if str(rel) == "." else str(rel).replace(os.sep, "/")


def render_markdown(
    dir_path: Path,
    rel: str,
    title: str,
    purpose: str,
    subdirs: list[str],
    exts: Counter[str],
    top_files: list[str],
    readme_hint: str | None,
    artifact_status: dict[str, bool],
) -> str:
    today = date.today().isoformat()
    review_data = resolve_review(rel, dir_path, artifact_status)
    lines = [
        f"# Folder audit: `{rel or '.'}`",
        "",
        f"**Generated:** {today} (tool: `scripts/pipeline/generate_folder_audit_reports.py`)",
        "",
        "## Role",
        "",
        f"**{title}**",
        "",
        purpose.strip(),
        "",
    ]
    if readme_hint:
        lines.extend(["## Local documentation", "", readme_hint, ""])
    lines.extend(
        [
            "## Contents snapshot",
            "",
            f"- **Subdirectories:** {len(subdirs)}",
            f"- **Files (this directory):** {sum(exts.values())}",
            "",
        ]
    )
    if subdirs:
        show = subdirs[:40]
        rest = len(subdirs) - len(show)
        lines.append("**Subdirectories:**")
        for s in show:
            lines.append(f"- `{s}/`")
        if rest > 0:
            lines.append(f"- … and {rest} more")
        lines.append("")
    if exts:
        lines.append("**File types (this directory only):**")
        for ext, n in exts.most_common():
            lines.append(f"- `{ext}`: {n}")
        lines.append("")
    if top_files:
        lines.append("**Sample files:**")
        for f in top_files:
            lines.append(f"- `{f}`")
        lines.append("")
    lines.extend(
        [
            "## Audit notes for follow-up models",
            "",
            "- Re-run this generator after large structural changes: `python scripts/pipeline/generate_folder_audit_reports.py`.",
            "- Prefer `src.utils.paths` / `configs/pipeline.yaml` over hardcoded paths in new code.",
            "- Large binaries (`data/`, `models/`, `results/`) may be absent in minimal clones—check `REPRODUCIBILITY.md`.",
            "",
        ]
    )
    lines.extend(
        [
            "## Evidence-based review",
            "",
            f"**Verdict:** {review_data['verdict']}",
            "",
            "**Evidence reviewed:**",
        ]
    )
    for item in review_data["evidence"]:
        lines.append(f"- {item}")
    lines.extend(["", "**Strengths:**"])
    for item in review_data["strengths"]:
        lines.append(f"- {item}")
    lines.extend(["", "**Gaps / risks:**"])
    for item in review_data["gaps"]:
        lines.append(f"- {item}")
    lines.extend(["", "**Next audit checks:**"])
    for item in review_data["next_checks"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    artifact_status = collect_artifact_status(REPO)
    all_dirs: list[Path] = [REPO]
    for dirpath, dirnames, _ in os.walk(REPO):
        dp = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        for d in dirnames:
            all_dirs.append(dp / d)

    written = 0
    for dir_path in sorted(set(all_dirs)):
        if not dir_path.is_dir():
            continue
        rel = rel_posix(REPO, dir_path)
        title, purpose = CONTEXT.get(rel, fallback_context(rel))
        subdirs, exts, top_files = inventory(dir_path)
        readme_hint = read_readme_hint(dir_path)
        md = render_markdown(
            dir_path,
            rel,
            title,
            purpose,
            subdirs,
            exts,
            top_files,
            readme_hint,
            artifact_status,
        )
        out = dir_path / "FOLDER_REVIEW.md"
        out.write_text(md, encoding="utf-8")
        written += 1

    index_path = REPO / "AUDIT_INDEX.md"
    index_lines = [
        "# Repository audit index",
        "",
        f"**Generated:** {date.today().isoformat()}",
        "",
        "Each directory under this repository (excluding VCS/cache folders) contains a **`FOLDER_REVIEW.md`** "
        "with role, content snapshot, and evidence-based audit notes for later models and humans.",
        "",
        "## Regeneration",
        "",
        "```bash",
        "python scripts/pipeline/generate_folder_audit_reports.py",
        "```",
        "",
        "## Repository-level findings",
        "",
        "- This checkout contains real local artifacts for `data/`, `models/`, `results/`, `logs/`, and `articles/`, so the review can use actual caches, checkpoints, histories, figures, and manuscript outputs rather than code-only assumptions.",
        "- `results/clop_data_audit.json` reports `167245` cells across `69` cell types and `79` datasets with all types represented in both train and validation under the stratified split, but the audit still returns `WARN` because two cell types appear in only one dataset, one type has only `77` cells, and seven text-embedding pairs are highly confusable.",
        "- Training appears substantially complete from the stored histories: CLOP runs the configured `60` epochs and improves from `val_loss ~= 0.082` to `0.007`, while DiT runs `200` epochs and improves from `val_loss ~= 0.999` to `0.032` with `val_cosine` reaching about `0.990`.",
        "- Results are structurally strong: `benchmark_report.json`, `comprehensive_summary.json`, `generation_metadata.json`, `results/figures/*.pdf`, `articles/figures/*.pdf`, and `articles/clop_dit_biology.pdf` are all present, and CLOP-DiT leads the stored composite benchmark with `composite_score ~= 0.838`.",
        "- The main unresolved issues are result interpretation and documentation drift: `coverage = 0.0282`, `discriminator_auc ~= 0.656`, `de_mean_logfc_pearson ~= 0.394`, several summary blocks are null, VCD reports six warnings plus three missing live sidecars, and docs still drift across `17`, `19`, `20`, and `30` figure narratives.",
        "",
        "## Local artifact status",
        "",
        f"- `data/`: {'artifacts detected' if artifact_status['data'] else 'no local artifacts detected'}",
        f"- `models/`: {'artifacts detected' if artifact_status['models'] else 'no local artifacts detected'}",
        f"- `results/`: {'artifacts detected' if artifact_status['results'] else 'no local artifacts detected'}",
        f"- `logs/`: {'artifacts detected' if artifact_status['logs'] else 'no local artifacts detected'}",
        f"- `articles/`: {'artifacts detected' if artifact_status['articles'] else 'no local artifacts detected'}",
        "",
        "## Highest-priority follow-up",
        "",
        "- Reconcile figure and article counts across docs, visualization manifests, and regeneration scripts, then refresh `results/README.md` to match the live `figNN_*` workflow.",
        "- Investigate the data-tail risks surfaced by `clop_data_audit.json`, especially the `80` processed-h5ad files versus `79` dataset count, the two single-dataset types, and the rarest cell types.",
        "- Align downstream metadata with real checkpoint locations; for example, `generation_metadata.json` still points to `models/checkpoints/dit_best.pth` even though the stored versioned checkpoint lives under `models/checkpoints/DiT/...`.",
        "- Resolve the remaining VCD warnings and missing live sidecars, and decide whether the null-valued cross-dataset / expanded-DE / marker-completeness sections are expected gaps or broken analyses.",
        "- Make `seed` explicit in the main YAMLs and store hashes or manifests for cached latents, checkpoints, and result bundles.",
        "",
        "## Top-level folders",
        "",
    ]
    for name in sorted(REPO.iterdir()):
        if not name.is_dir() or should_skip_dir(name.name):
            continue
        rel = name.name
        t, p = CONTEXT.get(rel, ("(see folder)", ""))
        index_lines.append(f"- **`{rel}/`** — {t}")
    index_lines.extend(
        [
            "",
            "## Conventions",
            "",
            "- **Article figures:** `results/figures/` → `articles/figures/` via `src/visualization/article_delivery.py`.",
            "- **Benchmarking contract:** `src/evaluation/README.md` (baselines under `results/baselines/{method}/`).",
            "- **Paths:** `configs/pipeline.yaml` + env `CLOPDIT_*` (see `src/utils/paths.py`).",
            "",
        ]
    )
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"Wrote {written} FOLDER_REVIEW.md files and AUDIT_INDEX.md")


if __name__ == "__main__":
    main()
