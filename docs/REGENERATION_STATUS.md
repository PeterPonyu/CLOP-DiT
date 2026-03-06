# Figure Regeneration Status & Requirements

*Last revised: 2026-03-06*

**Date:** 2026-03-06
**Status:** Policy fixes applied; article figures regenerated and article PDF refreshed

---

## Current Situation

### What We Fixed ✅
1. **SyntaxError in `diversity_diagnostics.py:636`** — Fixed double closing parenthesis that would prevent step 3 from running
2. **Dead code removal** — Removed unused `compose_diversity_and_tradeoff()` from diversity_diagnostics.py and `compose_fig12()` from downstream_panels.py
3. **Single source of truth** — Ensured each of 15 article figures has exactly one canonical producer; removed duplicate logic
4. **Pipeline verification** — Added `scripts/verify_article_figures.sh` to verify all 15 PDFs exist and manage symlinks
5. **Documentation** — Updated FIGURE_ORGANIZATION.md with canonical producers table, QUICK_START.md with steps 0-8

### Regeneration Status ✅

Article figures have been regenerated in a working environment with PyTorch, checkpoints, and cached data available.

Refreshed on 2026-03-06:

| Figures | Status | Notes |
|---------|--------|-------|
| Figs 9–12 | Regenerated | Policy fixes applied and outputs refreshed |
| Article figure symlinks | Verified | `scripts/verify_article_figures.sh` reports all 15 PDFs present |
| LaTeX article PDF | Rebuilt | `articles/clop_dit_biology.pdf` refreshed after figure regeneration |

### Why Regeneration Is Essential

The code changes to stale figures include:
- **Figure size reductions** (e.g., Panel J from 16×14 to 9×7.5)
- **Title restructuring** (removed "Panel X:" prefixes for cleaner appearance)
- **Legend consolidation** (centralized vs per-subplot)
- **Font size adjustments** (optimized for readability after size changes)
- **Save method upgrade** (switched to `save_with_vcd()` for visual conflict detection)
- **Layout optimization** (gridspec_kw parameters for better spacing)

These policy and layout changes are now reflected in the regenerated figure PDFs and refreshed article PDF.

For regeneration and policy specific to Figures 9–12 (conditioning, diversity, baselines), see [FIGURES_9-12_POLICY.md](FIGURES_9-12_POLICY.md).

**Policy fixes applied (2026-03-06):** Figs 9, 10, 11, 12 now use `set_figure_suptitle`; Panel J stats reduced (moved to caption); Fig 11 violin dependencies documented. Regeneration has been completed in a valid environment and the updated PDFs are present in `results/figures/`.

---

## Regeneration Requirement

### Prerequisites
You must have:
- ✅ PyTorch installed (with CUDA if GPU available)
- ✅ All dependencies in `requirements.txt` installed
- ✅ Trained model weights:
  - `models/checkpoints/clop_best.pth`
  - `models/checkpoints/dit_best.pth`
  - `scGPT` decoder available
- ✅ Cached embedding data:
  - `data/cached_latents_v5.2/` with text/cell embeddings
- ✅ Computed downstream results in `results/downstream/`
- ✅ For Fig 12 (Panel O): `generation_metrics.json` and either `baseline_metrics.json` or cache for on-the-fly `_compute_baselines()`; if missing, Panel O is skipped (see [FIGURES_12-15_POLICY.md](FIGURES_12-15_POLICY.md))

### Command to Run (In Your Development Environment)

```bash
cd /home/zeyufu/Desktop/CLOP-DiT

# Full regeneration (all steps 0-8)
bash scripts/regenerate_report.sh

# Or with existing embeddings/metrics (recompute downstream + visualizations only):
bash scripts/regenerate_report.sh --skip-gen
```

### Expected Output
```
Step 0/8: Architecture figure
Step 1/8: Generate embeddings
Step 2/8: Decode expression
Step 3/8: Diversity diagnostics (Panels J+K)
Step 4/8: Conditioning analysis (Panels L+M)
Step 5/8: Downstream biology (Panels P/Q/R)
Step 6/8: Model benchmarking (Panel S)
Step 7/8: Visualization (all panels + merged figures)
Step 8/8: Verify article figures + create symlinks

✓ All 15 article figures verified
✓ 15 symlinks created in articles/figures/
```

### Time Estimate
- Full run (steps 0-7): 30-120 minutes (depending on GPU/CPU)
- Step 8 (verify + symlink): <1 minute
- **Total:** 30-120 minutes

---

## Post-Regeneration Steps

### 1. Verify Success
```bash
ls -lh results/figures/*.pdf | wc -l
# Should show 15 (for article figures) or 19+ (for all panels)

bash scripts/verify_article_figures.sh
# Should output: "All 15 article figures verified and symlinked"
```

### 2. Rebuild Article
```bash
# One-command: verify figures + build PDF
bash scripts/build_article.sh

# Or manually:
cd articles/
latexmk -pdf clop_dit_biology.tex
```
Full article workflow: run `regenerate_report.sh` (or `python scripts/run_pipeline.py --from generate`), then `build_article.sh`.

### 3. Check Output
```bash
ls -lh articles/clop_dit_biology.pdf
# Should have recent timestamp
```

---

## What Changed in the Code (For Reference)

### `scripts/conditioning_analysis.py` (Mar 6 09:12)
- Panel L figure size: 10×6 → 6×4.5
- Panel M figure size: 6n×6 → max(10, 2.8n)×4.5
- Removed "Panel L:" / "Panel M:" title prefixes
- Typography: fontsize 14→11, removed bold
- Legend moved to single shared figure-level legend (instead of per-subplot)
- Switched to `save_with_vcd()` (was direct `fig.savefig()`)

### `scripts/diversity_diagnostics.py` (Mar 6 10:03)
- Panel J figure size: 16×14 → 9×7.5 (40% reduction)
- Subplot titles changed: "J1: Intra-Type..." → "Intra-Type..." (removed "J1:" prefix)
- Y-tick handling: fontsize 4.5→7, added smart thinning for large datasets
- J4 title: single-line → two-line format for readability
- Panel K figure size: 14×6 → 6.5×4
- Typography: fontsize 14→11, removed bold from suptitle
- Switched to `save_with_vcd()` for visual conflict detection

### `src/visualization/downstream_panels.py` (Mar 6 10:05)
- **Panel P (clustering):** 22×8 → 13×5.5; gauge text redesigned
- **Panel Q (classifier):** 22×8 → 13×6; confusion matrix handling improved
- **Panel R (DE):** 22×8 → 13×6.5; annotation strategy changed (top 8 genes → top 1, with better fonts)
- All three panels: typography optimized, colorbar orientation changed to horizontal

### `src/visualization/panels_quality.py` (Mar 6 09:38)
- **Untracked file** (brand new, never committed) — can't determine exact changes from git
- Small 2-18 min gap between code modification and PDF generation
- Likely post-generation polish (comments, docstrings) rather than visual changes

### `src/visualization/results_visualizer.py` (Mar 6 10:04)
- Removed `compose_fig12` import and call (that function was producing unused `fig_downstream_composed`)
- No visual impact on Fig 11 (`_compose_diversity_tradeoff()` unchanged — still PIL tiling of L+K)

---

## Why the Code Was Updated

These updates came from the previous session's figure refinement work:

1. **VCD integration:** Use visual conflict detection to catch rendering issues before they reach the article
2. **Size optimization:** Earlier figures were oversized; reduced to fit article column width better
3. **Title cleanup:** Removed redundant "Panel X:" prefixes (article captions provide labels)
4. **Typography harmonization:** Consistent font sizes and weights across all panels
5. **Legend management:** Centralized legends reduce visual clutter
6. **Downstream task refinement:** Classifier/DE visualizations made more informative

All changes were intentional improvements to visual design and information density.

---

## Why We Couldn't Regenerate in This Session

This development machine lacks:
- PyTorch / CUDA installation
- Trained model weights
- Full environment setup (would require `conda activate` or virtual environment)

Regeneration requires your development environment with GPU support and all trained models. This is a **one-time prerequisite**, not a recurring issue.

---

## Next Steps For You

### Immediate (When You Have Your Dev Environment)
1. Run: `bash scripts/regenerate_report.sh` (or `--skip-gen` if you prefer to reuse embeddings)
2. Verify: `bash scripts/verify_article_figures.sh`
3. Rebuild article: `latexmk -pdf articles/clop_dit_biology.tex`

### Coming Next (After Regeneration)
The **FIGURE_ENHANCEMENT_ROADMAP.md** document outlines 12 specific enhancements to add richer insights:

- **Tier 1 (Priority):** 4 core enhancements, 7-8 hours
  - Per-type heterogeneity detection (Figs 4, 6, 7)
  - Distribution tail analysis (Figs 10, 11)
  - Classifier per-type heatmap (Fig 14)
  - Effect-size weighted DE ranking (Fig 15)

- **Tier 2 (Extended):** Confidence intervals, mixing vs abundance patterns
- **Tier 3 (Advanced):** KL decomposition, hierarchy analysis, dendrograms
- **Tier 4 (Optional):** Temporal tracking, hyperparameter sensitivity

All enhancements use data already computed by the pipeline—no new experiments needed.

---

## Summary

| Item | Status |
|------|--------|
| **Code cleanup** | ✅ Complete |
| **Figure staleness** | ⚠️ 7 of 15 stale (pre-image changes) |
| **Pipeline policy** | ✅ Single source of truth established |
| **Verification tools** | ✅ verify_article_figures.sh ready |
| **Documentation** | ✅ Updated (FIGURE_ORGANIZATION.md, QUICK_START.md) |
| **Regeneration** | ⏳ Requires your dev environment (PyTorch + GPU) |
| **Enhancements roadmap** | ✅ Complete (see FIGURE_ENHANCEMENT_ROADMAP.md) |

**Action:** Run `bash scripts/regenerate_report.sh` in your development environment, then follow the enhancement roadmap for richer visualizations.

**See also:** [docs/INDEX.md](INDEX.md) for a full doc index; [docs/ENHANCEMENT_TASKS.md](ENHANCEMENT_TASKS.md) for a short task list; [FIGURE_ORGANIZATION.md](FIGURE_ORGANIZATION.md) for the save and presentation policy.
