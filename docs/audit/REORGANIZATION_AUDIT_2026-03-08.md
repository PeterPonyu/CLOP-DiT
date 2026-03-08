# Project Reorganization Audit Report

**Date:** 2026-03-08
**Scope:** Audit of the deep project reorganization (commits `ad0ffff`, `3c9a800`, `2c6c415`) and subsequent repair of all broken references.

---

## Executive Summary

The project reorganization moved 45 scripts into phase-based subdirectories (`data_prep/`, `training/`, `inference/`, `analysis/`, `pipeline/`), 28 docs into category subdirectories, and restructured checkpoint directories. While the directory layout was well-designed, the reorganization left **118 broken references** across Python imports, shell scripts, subprocess calls, LaTeX paths, and Markdown documentation. This audit identified and repaired all 118 references.

---

## Issues Found and Fixed

### 1. CRITICAL: sys.path Broken in 30 Scripts (22 original + 8 additional)

**Problem:** Scripts moved from `scripts/` root into subdirectories (e.g., `scripts/data_prep/`) used `Path(__file__).resolve().parent.parent` to find the project root. After the move, `.parent.parent` resolves to `scripts/` instead of the project root, breaking all `from src.` imports.

**Fix:** Changed `.parent.parent` to `.parent.parent.parent` in all 30 affected scripts.

**Impact:** Without this fix, no moved script could import any `src.*` module.

### 2. CRITICAL: Shell Script cd Bug in 4 Scripts

**Problem:** Shell scripts in `scripts/pipeline/` used `cd "$(dirname "$0")/.."` to set the working directory to the project root. After the move, `..` resolves to `scripts/` instead of the project root.

**Fix:** Changed `..` to `../..` in:
- `scripts/pipeline/regenerate_report.sh`
- `scripts/pipeline/build_article.sh`
- `scripts/pipeline/cleanup_heavy.sh`
- `scripts/pipeline/cleanup_light.sh`

### 3. HIGH: 10 Broken Subprocess Paths in run_pipeline.py

**Problem:** `scripts/pipeline/run_pipeline.py` invokes scripts via subprocess using old root-level paths (e.g., `scripts/04a_train_clop.py`).

**Fix:** Updated all 10 references to use new subdirectory paths.

### 4. HIGH: Broken Paths in Other Orchestrators

| File | Broken | Fix |
|------|--------|-----|
| `scripts/training/run_5fold_cv.py` | 2 refs | -> `scripts/training/` |
| `scripts/training/run_robustness_experiments.py` | 3 refs | -> `scripts/inference/` |
| `scripts/training/run_ablation_study.py` | 1 ref | -> `scripts/training/` |

### 5. HIGH: Broken importlib Paths in src/experiments/

| File | Line | Old Path | New Path |
|------|------|----------|----------|
| `src/experiments/ood_evaluation.py` | 54 | `scripts/05_inference.py` | `scripts/inference/05_inference.py` |
| `src/experiments/rare_cell_augmentation.py` | 59 | `scripts/05_inference.py` | `scripts/inference/05_inference.py` |

### 6. MEDIUM: 5 Stale Producer Paths in article_delivery.py

Updated figure-to-script mapping in `src/visualization/article_delivery.py` to reflect new script locations.

### 7. MEDIUM: Test Config Path

`tests/test_configs.py` referenced `configs/clop_v9.3.yaml` (moved to archive). Updated to `configs/clop.yaml`.

### 8. MEDIUM: 5 Broken LaTeX Paths

Updated `\texttt{}` file path references in `articles/clop_dit_biology.tex`:
- `scripts/02b_enrich_descriptions.py` -> `scripts/data_prep/02b_enrich_descriptions.py`
- `configs/clop_v9.3.yaml` -> `configs/clop.yaml` (3 locations)
- `regenerate_report.sh` -> `scripts/pipeline/regenerate_report.sh` (4 locations)
- `docs/QUICK_START.md` -> `docs/operational/QUICK_START.md` (2 locations)

### 9. HIGH: ~35 Broken Markdown Documentation Paths

Updated references in `README.md` (~16), `PIPELINE.md` (~12), and `REPRODUCIBILITY.md` (~7) to reflect new subdirectory paths.

### 10. STRUCTURAL: 13 Scripts Left Loose in scripts/ Root

**Problem:** The original reorganization moved scripts into subdirectories but left 13 scripts stranded in `scripts/` root, undocumented in INDEX.md.

**Fix:** Moved all 13 to proper subdirectories:

| Script | Destination |
|--------|-------------|
| `generate_embeddings.py` | `scripts/inference/` |
| `generate_caption_variants.py` | `scripts/data_prep/` |
| `generate_supplementary_tables.py` | `scripts/analysis/` |
| `get_article_paths.py` | `scripts/pipeline/` |
| `polish_captions_v3.py` | `scripts/data_prep/` |
| `remove_outdated_figures.sh` | `scripts/pipeline/` |
| `run_5fold_cv.py` | `scripts/training/` |
| `run_ablation_study.py` | `scripts/training/` |
| `run_downstream_for_baselines.py` | `scripts/analysis/` |
| `run_pipeline.py` | `scripts/pipeline/` |
| `run_robustness_experiments.py` | `scripts/training/` |
| `verify_article_figures.sh` | `scripts/pipeline/` |
| `visual_conflict_detector.py` | `scripts/analysis/` |

Each move included updating the script's own sys.path and all cross-references project-wide (Python imports, subprocess calls, shell invocations, documentation).

---

## Verification Results

| Check | Result |
|-------|--------|
| Old-style root script paths in .py/.sh | 0 remaining |
| Loose files in `scripts/` root | 0 remaining |
| sys.path depth correctness | All correct |
| Broken symlinks | 0 found |
| LaTeX PDF build | Clean (34 pages) |

---

## Remaining Known Issues (Not Addressed)

1. **`data/dataset_accessions.tsv` and `data/validation_studies.txt`** referenced in REPRODUCIBILITY.md do not exist. These are planned supplementary data files that have not been created yet.

2. **`scripts/training/full_retrain_pipeline.sh`** references 3 archived scripts (`polish_texts.py`, `re_embed_polished_texts.py`, `rectify_flow.py`) and one deleted script (`06_generate_from_text.py`). This script is a legacy pipeline from v5-v7 era and may need retirement or archival.

3. **`archive/` subdirectories** described in `VERSIONS.md` (e.g., `archive/2025-03-08_configs/`) were never created. The per-component archive directories (`configs/archive/`, `scripts/archive/`, `docs/archive/`) serve this purpose instead.

4. **`models/checkpoints/CLOP/`, `DiT/`, `ablations/`** are untracked. Their INDEX.md files and directory structure metadata would be lost if the directories are removed. Consider adding `.gitkeep` or INDEX.md files to git.

---

## Article LaTeX Improvements (Separate from Reorganization)

During this session, the article (`articles/clop_dit_biology.tex`) also received content improvements:

1. **Float placement**: 17 `[H]` figures + 3 `[H]` tables changed to `[!htbp]`
2. **`\raggedbottom`** and `\usepackage{placeins}` added
3. **6 equations** wrapped in `\begin{linenomath}...\end{linenomath}`
4. **10 formulaic transitions** ("This subsection...") removed
5. **Temperature notation** clarified (tau = 14.0 logit-scale parameter)
6. **ZCA whitening** description added to Methods
7. **Logit-normal timestep** justification added with SD3 citation
8. **Validation split criteria** and **sample size justification** added
9. **"First method" claim** properly qualified with Cell2Sentence/GenePT citations
10. **"In-distribution" qualifiers** added to Results
11. **FD caveat** added where Frechet distance first appears
12. **Bibliography expanded** from 47 to 67 references
13. **Grammar fixes**: "Section" -> "Sections", informal phrasing tightened

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Broken sys.path entries fixed | 30 |
| Shell script cd bugs fixed | 4 |
| Broken subprocess paths fixed | 16 |
| Broken importlib paths fixed | 2 |
| Stale metadata paths fixed | 5 |
| Test config paths fixed | 3 |
| LaTeX path references fixed | 10 |
| Markdown path references fixed | ~35 |
| Scripts moved to subdirectories | 13 |
| **Total broken references repaired** | **~118** |
