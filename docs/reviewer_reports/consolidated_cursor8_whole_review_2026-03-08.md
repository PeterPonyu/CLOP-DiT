# Consolidated Master Review Report: CLOP-DiT Manuscript

**Date:** 2026-03-08
**Scope:** Merged from 8 individual reviewer reports (`bay`, `dcn`, `epw`, `etb`, `jth`, `kjc`, `mmg`, `rjp`), 8 parallel Cursor worktree figure reviews (`ahz`, `eeu`, `eru`, `fhu`, `jjf`, `mdv`, `tqv`, `wvp`), 2 prior consolidated reports (LaTeX format, whole-review), and 4 audit files.
**Authoritative source:** `articles/clop_dit_biology.tex`

---

## 1. Methodology

This report de-duplicates and frequency-ranks all findings from **16 independent reviews** (8 reviewer reports + 8 figure-level worktree audits) into a single actionable record. Each concern is tagged with:

- **Frequency**: how many reports raised it (out of 16, or out of subgroup of 8)
- **Current status**: resolved / partially resolved / open
- **Priority**: Critical / High / Medium / Low
- **Action owner**: manuscript (.tex), code (Python), docs (markdown), or submission packaging

---

## 2. Consensus Concerns Across All Reviews

### 2.1 Manuscript-Level Issues

| Priority | Concern | Frequency | Current Status (2026-03-08) |
|----------|---------|-----------|---------------------------|
| Critical | Cross-section metric framing: 51.1% LinAcc vs 30.8% downstream classifier | 8/8 reviewer | **RESOLVED** in .tex — Table 1 caption explicitly distinguishes LinAcc (embedding-space) from downstream expression-space classifier (Fig 16). Abstract uses 30.8% with full context. |
| Critical | logFC Pearson r: 0.17 vs ~0.39 | 8/8 reviewer | **RESOLVED** in .tex — Abstract states "mean logFC Pearson r ≈ 0.39 across three contrasts, range 0.17–0.56" |
| Critical | Composite benchmark: 0.844 vs 0.668 | 5/8 reviewer | **PARTIALLY RESOLVED** — 0.844 is authoritative in .tex; stale 0.668 remains in `docs/roadmaps/FIGURE_DESIGN_AND_PROMPT_DIVERSITY_PLAN.md` |
| Critical | Reproducibility: no public code/data URL | 8/8 reviewer | **OPEN** — requires author decision on publication posture |
| Critical | Supplementary Tables S1/S2 referenced but not created | 6/8 reviewer | **OPEN** — .tex references them; actual files needed |
| High | Uncertainty/CI for headline metrics | 8/8 reviewer | **PARTIALLY RESOLVED** — composite has bootstrap CIs; individual metric CIs still absent from abstract/Table 1 |
| High | Figure artifacts not generated in workspace | 7/8 reviewer | **OPEN** — requires pipeline execution in target runtime |
| High | Learned baselines absent (scVI, etc.) | 6/8 reviewer | **PARTIALLY RESOLVED** — Baseline Scope section in Discussion acknowledges gap; scVI dependency issue documented |
| Medium | Script name `build_text_captions.py` does not exist | 4/8 reviewer | **OPEN** — needs correction to actual paths |
| Medium | Dimension discrepancy: line 332 says "shared 256-dimensional" but rest of paper says 512-d | 3/8 figure | **OPEN** — single-site typo at line 332 |
| Medium | Derivative markdown drift vs authoritative .tex | 5/8 reviewer | **OPEN** — needs deprecation notice |
| Low | ORCID placeholder at line 82 | 1/8 audit | **RESOLVED** — line 81 now has real ORCID `0009-0001-8329-0108` |

### 2.2 Figure-Level Issues (from 8 Cursor Worktree Reviews)

#### Critical Issues (must fix before submission)

| # | Concern | Figures Affected | Freq (of 8 fig reviews) | Code Location |
|---|---------|-----------------|------------------------|---------------|
| F1 | Missing panel labels (a)-(f) | Fig 3 | 8/8 | `src/visualization/panels_merged.py` — add `add_panel_label()` |
| F2 | Missing panel (b) label | Fig 14 | 8/8 | `src/visualization/baseline_panels.py` — label targets placeholder axes |
| F3 | Duplicate panel labels (a)(b)(c) in merged figures | Figs 5/6, 12 | 6/8 | `panels_merged.py`, `results_visualizer.py` — need label_offset |
| F4 | MaxNLocator bug hides gene name xtick labels | Fig 8 | 6/8 | `src/visualization/panels_expression.py` **line 247** |

#### High Issues — Stale Suptitles

| Concern | Figures Affected | Freq | Action |
|---------|-----------------|------|--------|
| Stale `fig.suptitle()` in rendered PNGs | Figs 2-10, 12-17 (15 of 17 figures) | 8/8 | Remove all suptitle calls; regenerate |
| Fig 2 stats banner via `fig.text()` acting as suptitle | Fig 2 | 7/8 | Remove `fig.text()` banner at lines 147-152 of `panels_training.py` |

#### Medium Issues — Font, Layout, Caption

| # | Concern | Figures | Freq | Action |
|---|---------|---------|------|--------|
| M1 | Font sizes below 7pt VCD minimum at print scale | Figs 3, 7, 9, 10, 16 | 7/8 | Use `style.py` constants (FONT_TICK >= 7pt) |
| M2 | Caption-panel content mismatches | Figs 3, 5, 7, 8, 12, 14, 15, 17 | 7/8 | Align captions with actual rendered content |
| M3 | Hard-coded font sizes instead of style.py constants | Figs 1, 2, 7, 9, 10 | 6/8 | Replace with FONT_TITLE, FONT_LABEL, etc. |
| M4 | Cell type name truncation `[:20]` mid-word | Figs 7, 11, 17 | 7/8 | Implement intelligent abbreviation |
| M5 | Fig 1: colored text should be black with colored backgrounds | Fig 1 | 7/8 | Fix in `generate_architecture_figure.py` |
| M6 | Fig 1: sublabel clipped at box boundary | Fig 1 | 6/8 | Increase box padding |
| M7 | Fig 10: heatmap annotation fontsize=6 below 7pt min | Fig 10 | 7/8 | Increase to >=7pt |
| M8 | Fig 14 panel O3: extreme 3000% bars compress smaller bars | Fig 14 | 7/8 | Cap display at 300% |
| M9 | Fig 15 panel S3: EmbeddingVAE FD dominates y-axis | Fig 15 | 7/8 | Use log-scale or broken axis |
| M10 | Fig 17: too few gene annotations in scatter, too few DE contrasts | Fig 17 | 7/8 | Add annotations and contrasts |
| M11 | Figure S1 referenced but never defined | Supplementary | 5/8 | Create or remove reference |

---

## 3. Reconciled Interpretations (No Action Needed)

These apparent contradictions are already properly resolved in the current .tex:

| Apparent Issue | Resolution in .tex |
|----------------|-------------------|
| 51.1% vs 30.8% classifier | 51.1% = LinAcc (embedding-space logistic regression, Table 1); 30.8% = downstream expression-space classifier (Fig 16, Conclusions). Table 1 caption explicitly distinguishes them. |
| r = 0.17 vs ~0.39 | Abstract: "mean logFC Pearson r ≈ 0.39 across three contrasts, range 0.17–0.56". Properly framed. |
| τ = 14.0 vs 0.07 | Line 228: "τ = 14.0 is a fixed logit-scale parameter... effective softmax temperature 1/τ ≈ 0.071". Clear. |
| CUDA 12 vs 13 | Line 210: already says "CUDA~13.0". Resolved. |
| Composite 0.844 | Authoritative manuscript value. Legacy 0.668 flagged as stale in addendum. |

---

## 4. Action Plan: Parallel Workstreams

### Workstream A: Manuscript .tex Fixes (Critical/High)

1. **Fix dimension typo** at line 332: "shared 256-dimensional" → "shared 512-dimensional"
2. **Fix script reference**: `build_text_captions.py` → actual paths (`src/data_pipeline/subcluster_annotation.py` and `scripts/data_prep/02b_enrich_descriptions.py`)
3. **Add uncertainty paragraph** in Methods: state that bootstrap 95% CIs are available for composite and key metrics
4. **Add sample size justification sentence**: reference standard practice for 100 types, 200 cells/group
5. **Strengthen ablation discussion**: acknowledge that removing regularization improves CLOP val accuracy but harms downstream generation quality

### Workstream B: Supplementary Materials Creation

1. **Create `articles/supplementary_tables.tex`** with Table S1 (GEO accessions) and S2 (validation datasets)
2. **Verify S3 (CFG sweep)** is already in the longtable appendix
3. **Create or remove Figure S1 reference**

### Workstream C: Figure Code Fixes

1. Fix panel labels: Figs 3, 5/6 merged, 12 merged, 14
2. Remove all suptitle calls across 15 figures
3. Fix MaxNLocator bug at `panels_expression.py:247`
4. Fix Fig 1: black text, padding
5. Fix font sizes below 7pt minimum
6. Fix cell type name truncation

### Workstream D: Documentation Cleanup

1. Remove stale 0.668 references from `docs/roadmaps/FIGURE_DESIGN_AND_PROMPT_DIVERSITY_PLAN.md`
2. Add deprecation notice to derivative markdown docs
3. Update `SUBMISSION_CHECKLIST.md` checkmarks for completed items
4. Update `REVIEWER_CONCERNS_AND_NEXT_STEPS.md` with resolution status
5. Sync all script path references to new reorganized structure

### Workstream E: Submission Verification (Post-Fix)

1. Regenerate all 17 figure PDFs
2. Run VCD verification (target: 0 warnings)
3. Run `verify_article_figures.sh`
4. LaTeX build verification: `latexmk -pdf clop_dit_biology.tex`
5. Confirm no TODO/placeholder remnants

---

## 5. Per-Figure Concern Summary Table

| Figure | Content | Critical | High | Medium | Low |
|--------|---------|----------|------|--------|-----|
| 1 | Architecture Overview | 0 | 0 | 3 (text color, clipping, font) | 2 |
| 2 | Training Dynamics | 0 | 2 (suptitle, stats banner) | 1 (legend) | 1 |
| 3 | Embedding Space | **1** (panel labels) | 1 (suptitle) | 2 (font, figsize) | 1 |
| 4 | Metrics Dashboard | 0 | 1 (suptitle) | 2 (alignment, FD caveat) | 2 |
| 5 | Per-Type Fidelity | **1** (dup labels) | 1 (suptitle) | 2 (scatter, raster) | 1 |
| 6 | Text-Cell Alignment | **1** (dup labels) | 1 (suptitle) | 2 (raster, canonical) | 1 |
| 7 | Marker Gene | 0 | 1 (suptitle) | 3 (y-axis, fold-change, legend) | 2 |
| 8 | Expression Correlation | **1** (MaxNLocator) | 1 (suptitle) | 2 (offset, stats) | 0 |
| 9 | Expression Distribution | 0 | 1 (suptitle) | 2 (legend font, CV axis) | 1 |
| 10 | Conditioning Landscape | 0 | 1 (suptitle) | 3 (legend pos, font, panels) | 2 |
| 11 | Diversity Diagnostics | 0 | 0 | 3 (type IDs, legend, truncation) | 1 |
| 12 | Noise-Scale Trade-Off | **1** (dup labels) | 1 (suptitle) | 2 (border, canonical) | 1 |
| 13 | Expression Diversity | 0 | 1 (suptitle) | 1 (label conflict) | 1 |
| 14 | Baseline Comparison | **1** (panel b missing) | 1 (suptitle) | 2 (bars, caption) | 1 |
| 15 | Composite Benchmark | 0 | 1 (suptitle) | 3 (FD axis, legend, CIs) | 1 |
| 16 | Downstream P+Q | 0 | 1 (suptitle) | 3 (font, overlap, legacy) | 1 |
| 17 | DE Concordance | 0 | 1 (suptitle) | 3 (annotations, contrasts, truncation) | 1 |
| **Total** | | **6** | **16** | **37** | **20** |

---

## 6. Execution Priority Order

**Phase 1 — Critical code fixes** (must fix before any regeneration):
1. Panel labels: Figs 3, 5/6, 12, 14
2. MaxNLocator bug: Fig 8 (`panels_expression.py:247`)
3. Dimension typo in .tex line 332

**Phase 2 — High: suptitle removal + .tex fixes**:
1. Remove all `suptitle()` and `fig.text()` banner calls
2. Fix script name reference in .tex
3. Add uncertainty/CI and sample size text
4. Address ablation paradox

**Phase 3 — Medium: per-figure content improvements**:
1. Font sizes, legends, axis labels
2. Cell type name truncation logic
3. Fig 1 text color and padding
4. Caption-panel alignment verification

**Phase 4 — Documentation and packaging**:
1. Remove stale 0.668 references
2. Update submission checklist
3. Add deprecation notices
4. Create supplementary tables S1/S2

**Phase 5 — Verification** (post all fixes):
1. Regenerate all figures
2. VCD check
3. LaTeX build
4. Final consistency sweep

---

## 7. Items Already Resolved (No Action Needed)

The following items from reviewer reports are confirmed resolved in the current .tex:

1. Author metadata, affiliation, ORCID, correspondence — complete
2. Ethics statement, informed consent, CRediT contributions — complete
3. Data Availability Statement — present (references S1/S2)
4. Float placement: all `[H]` → `[!htbp]` — done
5. `\raggedbottom` and `\usepackage{placeins}` — added
6. 6 equations in `linenomath` — done
7. Formulaic transitions removed — done
8. Temperature τ = 14.0 notation clarified — done
9. ZCA whitening added to Methods — present in Fig 1 caption
10. Logit-normal timestep justified with SD3 citation — done
11. "First method" claim qualified with Cell2Sentence/GenePT citations — done
12. Cell2Sentence citation added — done
13. Bibliography expanded to 67 references — done
14. FD caveat at first introduction — done (line 342)
15. OOD limitation explicit in Discussion — done (line 558)
16. Validation split criteria documented — done (line 204)
17. CUDA version: already says CUDA 13.0 — done
18. Primary vs exploratory operating points designated — done
19. Text caption template documented with correct script paths — done (line 202)
20. 80/20 split and 100-type design justified — done (line 206)
21. Baseline scope limitations acknowledged — done (line 560)
22. Marker gene circularity acknowledged — done (line 562)

---

## 8. Blocking Items Before Submission QA

1. Create supplementary table files (S1: 80 GEO accessions, S2: 8 validation IDs)
2. Decide code/data publication posture (public URL vs controlled request with SLA)
3. Fix dimension typo (256→512) at line 332
4. Fix all critical figure code issues (panel labels, MaxNLocator)
5. Regenerate all 17 figure PDFs
6. Run VCD verification to 0 warnings
7. Final LaTeX build clean
