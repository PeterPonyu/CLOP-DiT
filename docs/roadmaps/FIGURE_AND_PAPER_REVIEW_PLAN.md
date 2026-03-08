# CLOP-DiT Figure and Paper Review Plan

**Date:** 2026-03-07
**Scope:** Single paper `articles/clop_dit_biology.tex` and all 15 article figures

---

## Executive Summary

A thorough audit of all 15 figures, the LaTeX manuscript, bibliography, abbreviations, and data availability sections revealed the following critical issues:

1. **Panel labels (a)(b)(c):** 3 figures missing panel labels entirely, 2 figures have duplicate/conflicting labels
2. **Stale suptitles:** 12 of 15 figure PNGs show suptitles that the code has since removed -- all PNGs need regeneration
3. **References:** Only 27 references (need 45-50+); bibliography not ordered by first appearance
4. **Abbreviations:** 19 additional abbreviations used in text but missing from the abbreviations table
5. **CUDA version:** Reported as "12" in 3 locations; actual is 13.0
6. **Figure S1:** Referenced in supplementary text but never defined
7. **Font consistency:** Multiple hard-coded font sizes; some below 7pt VCD minimum
8. **Per-figure issues:** Detailed below for each of the 15 figures

---

## Part 1: Cross-Cutting Issues

### 1.1 Panel Labels (a)(b)(c) Status Across All Figures

| Fig | Source | Has Labels? | Issue |
|-----|--------|-------------|-------|
| 1 | generate_architecture_figure.py | Yes (manual ax.text) | Not using `add_panel_label()` -- inconsistent API |
| 2 | panels_training.py | Yes (a)-(h) | OK |
| 3 | panels_merged.py (embedding) | **NO** | **CRITICAL: 6 panels, zero labels. Caption references (a)-(f)** |
| 4 | panels_metrics.py | Yes (a)-(d) | OK |
| 5 | panels_merged.py (fidelity) | **DUPLICATE** | Both sub-figs use (a)(b)(c); needs (a)-(f) sequentially |
| 6 | panels_expression.py (markers) | Yes (a)-(d) | OK |
| 7 | panels_expression.py (corr) | Yes (a)-(d) | OK |
| 8 | panels_expression.py (analysis) | Yes (a)-(d) | OK |
| 9 | panels_conditioning.py | Yes (a)-(j) | OK but 10 labels is excessive |
| 10 | panels_diversity.py | Yes (a)-(d) | OK |
| 11 | results_visualizer.py (composed) | **DUPLICATE** | Panel L has (a), Panel K has (a)(b) -- conflict |
| 12 | baseline_panels.py | **MISSING (b)** | (b) added to placeholder axes then removed |
| 13 | benchmark_panels.py | Yes (a)-(d) | OK |
| 14 | downstream_panels.py | Yes (a)-(f) | OK |
| 15 | panels_de_concordance.py | Yes (a)-(c) | OK but caption doesn't reference them |

**Action:** Fix labels in Figs 3, 5, 11, 12. Update Fig 1 to use `add_panel_label()`.

### 1.2 Stale Suptitles in Rendered PNGs

All figures except Fig 10 have suptitles visible in the rendered PNG that the code has since commented out. The PNGs on disk are stale and must be regenerated.

| Fig | Stale Suptitle Text in PNG |
|-----|---------------------------|
| 2 | "Training Dynamics (CLOP + DiT)" |
| 3 | "Embedding Space Analysis" |
| 4 | "Core Evaluation Metrics Dashboard" |
| 5 top | "Per-Type Generation Fidelity" |
| 5 bot | "Text--Cell Alignment" |
| 6 | "Marker Gene Comparison" |
| 7 | "Gene Expression Recovery -- r=1.000000, rho=0.999999, n=1790" |
| 8 | "Expression Decoder -- 1932 real, 2000 gen, 1790 genes" |
| 9 | "Conditioning Mode Comparison (CFG=1.5, 5 types, PCA 2D)" |
| 11 | "Diversity Trade-off and Expression Variance" |
| 12 | "CLOP-DiT vs Baselines" |
| 13 | "Model Benchmarking -- CLOP-DiT vs Baselines" |
| 14 | "Downstream Validation: Clustering & Classifier Alignment" |
| 15 | "DE Concordance -- Real vs Generated" |

**Action:** Regenerate ALL figures after code changes are complete.

### 1.3 Figure-Level Suptitle Removal (per user request)

The user requires: *"Remove all figure-level titles (suptitles). Move information into figure captions or subplot titles."*

Current code status: Most suptitles are already commented out. But some figures still have `fig.text()` calls that act as de facto suptitles (e.g., Fig 2 line 147-152 has a stats banner). These decorative `fig.text()` calls that present summary statistics at the top of the figure should be moved into the subplot area or removed.

### 1.4 Subplot Titles Policy

The user clarifies: *"Don't remove subplot titles -- only the overall figure title (suptitle)."*

Subplot titles (per-axis titles via `ax.set_title()`) should be KEPT. They provide essential context for each panel.

---

## Part 2: Per-Figure Detailed Review and Action Plan

### Figure 1: Architecture Diagram

**Source:** `scripts/generate_architecture_figure.py`
**figsize:** (7.8, 3.8)

**User feedback:** "Some content is overlapping. Use colored backgrounds/boxes but keep main text in black for academic look."

**Issues found:**
1. Text color uses colored text matching the stage palette (blue for text path, green for cell path, purple for shared space). User wants BLACK text for main labels.
2. "frozen | 340M | 1024-d" sublabel slightly clipped at box boundary
3. Panel labels use manual `ax.text()` instead of `add_panel_label()`
4. Some hard-coded font sizes (8, 9) not using style.py constants
5. figsize can be reduced for compactness

**Actions:**
- [ ] Change all box label text to black (`color="black"`)
- [ ] Keep colored backgrounds on boxes/stages (visual distinction)
- [ ] Reduce figsize from (7.8, 3.8) to (7.8, 3.0)
- [ ] Tighten ylim from (-0.3, 3.4) to (-0.15, 3.15)
- [ ] Replace manual panel label code with `add_panel_label()`
- [ ] Verify no text overlap after height reduction

### Figure 2: Training Dynamics

**Source:** `src/visualization/panels_training.py`, `plot_training_dynamics_combined()`
**figsize:** (14.4, 8.6) -- 2x4 grid, panels (a)-(h)

**User feedback:** "Can you fill the bottom-right with a statistics panel? If computation is needed, use GPU."

**Current state:** Panel (h) was already filled with a DiT Summary text box in the previous session. The user may want something more visual/statistical.

**Issues found:**
1. `fig.text()` at lines 147-152 draws a stats banner at the top acting as a de facto suptitle. Should be removed.
2. Panel (h) is a plain text box -- could be enhanced with a mini bar chart or metric gauge

**Actions:**
- [ ] Remove the fig.text() stats banner (lines 147-152)
- [ ] Enhance panel (h): convert text summary to a small horizontal bar chart showing final metric values
- [ ] Ensure suptitle is gone after regeneration

### Figure 3: Embedding Space

**Source:** `src/visualization/panels_merged.py`, `plot_embedding_space_merged()`
**figsize:** (14.2, 11.2) -- 2x3 grid

**User feedback:** "Can we add more diverse display conditions? Different input conditions generating different UMAP embeddings to show generalization."

**Issues found:**
1. **CRITICAL: No panel labels.** Must add (a)-(f) for the 6 subplots
2. Stale suptitle "Embedding Space Analysis" in PNG
3. Caption references panel (f) as "norm comparison" but code has no such panel
4. 6pt y-tick labels on bar chart (below 7pt VCD minimum)
5. figsize at 14.2" means heavy scaling at \textwidth, making all text tiny

**Actions:**
- [ ] Add `add_panel_label()` calls for all 6 axes: (a)-(f)
- [ ] Fix 6pt ytick to at least 7pt
- [ ] Verify/fix caption-panel (f) mismatch
- [ ] Consider adding a 3rd row showing different conditioning inputs (e.g., different CFG values or different cell types) to demonstrate diversity -- this would require generating additional UMAP data

### Figure 4: Metrics Summary Dashboard

**Source:** `src/visualization/panels_metrics.py`, `plot_metrics_summary()`
**figsize:** (12.6, 8.0) -- 2x2 grid

**User feedback:** "Radar chart needs left-alignment with panel below. Check if content can be expanded with additional experiments."

**Issues found:**
1. Stale suptitle in PNG
2. Radar chart (panel b) sits at top-right; panel (d) below may not be left-aligned
3. Caption doesn't use bold (a)-(d) panel references
4. LaTeX figure missing `\centering`

**Actions:**
- [ ] Adjust gridspec width_ratios to left-align radar with panel below
- [ ] Add `\centering` before `\includegraphics` in LaTeX
- [ ] Update caption to use bold panel letter references: `(\textbf{a})`...
- [ ] Consider adding a 5th panel (expanding to 2x3 or 3x2) with additional metrics like per-type KNN distribution or composite score comparison

### Figure 5: Fidelity + Alignment (Merged G+F)

**Source:** `src/visualization/panels_merged.py`, `plot_fidelity_and_alignment_merged()`
**Composition:** PIL raster stacking (non-vector)

**User feedback:** "Add more text annotations. Add (a)-(f) labels. Remove suptitle. Can the right-upper panel be clearer?"

**Issues found:**
1. **DUPLICATE panel labels:** Both sub-figs use (a)(b)(c) independently. Need sequential (a)-(f)
2. Two stale suptitles (one per sub-figure) visible in PNG
3. PIL raster composition loses vector quality
4. Right-upper panel (c, "Fidelity vs Abundance" scatter) meaning could be annotated more clearly

**Actions:**
- [ ] Fix panel labels: top row (a)(b)(c), bottom row (d)(e)(f) by passing label_offset to sub-functions
- [ ] Add brief text annotations to key data points in the scatter plot (panel c/f)
- [ ] Consider converting from PIL composition to native GridSpec for vector output
- [ ] Ensure panel (c) "Fidelity vs Abundance" has clearer axis labels and annotations

### Figure 6: Marker Gene Comparison

**Source:** `src/visualization/panels_expression.py`, `plot_marker_gene_comparison()`
**figsize:** (11.0, 7.5) -- 2x2 grid

**User feedback:** "Add more text or cell type display. Avoid overlap. Don't add too many labels that become unreadable."

**Issues found:**
1. Stale suptitle in PNG
2. Panel N1 y-axis starts at 0 but data in 27-30 range -- wasted space
3. Panel N4 fold changes (0.96-1.04 range) nearly invisible
4. Type names hard-truncated mid-word ([:20])

**Actions:**
- [ ] Add more cell type names to panels where space permits (increase from ~10 to ~15 visible)
- [ ] Adjust N1 y-axis to start near data minimum (e.g., 25) to better show variation
- [ ] Add a "zoomed inset" or adjust N4 y-axis to better show fold-change deviations from 1.0
- [ ] Use intelligent abbreviation instead of hard truncation

### Figure 7: Expression Correlation

**Source:** `src/visualization/panels_expression.py`, `plot_expression_correlation()`
**figsize:** (11.2, 7.8) -- 2x2 grid

**User feedback:** "Move suptitle info to caption. Why does bottom-left panel (c) have only 3 x-tick labels?"

**Issues found:**
1. Stale suptitle "Gene Expression Recovery -- r=1.000000" shows detailed stats that should go to caption
2. **BUG:** Panel H3 `MaxNLocator(nbins=4)` at line 247 overrides explicit xtick labels set at line 236, hiding gene names
3. Panel H2 x-axis tick formatting near r=1.0 is confusing ("+1" offset notation)

**Actions:**
- [ ] Remove/verify suptitle removal in code
- [ ] **Fix bug:** Remove the MaxNLocator call at line 247 of panels_expression.py that overrides gene name xtick labels
- [ ] Move the suptitle stats (r, rho, n) into the caption or as an annotation in panel (a)
- [ ] Improve H2 x-axis formatting to avoid the confusing "+1" offset

### Figure 8: Expression Analysis

**Source:** `src/visualization/panels_expression.py`, `plot_expression_analysis()`
**figsize:** (9.8, 7.4) -- 2x2 grid

**User feedback:** "Can content be enriched somehow?"

**Issues found:**
1. Stale suptitle in PNG
2. Panel I2 legend at fontsize=7 -- at print scale becomes ~4.9pt
3. Panel I1 CV values extremely small (1e-5 range), axis labeling confusing

**Actions:**
- [ ] Add annotation text in panels with summary statistics
- [ ] Consider adding a 5th panel or enriching existing panels with additional detail (e.g., highlight top variable genes with labels)
- [ ] Increase I2 legend fontsize from 7 to 8

### Figure 9: Conditioning UMAP

**Source:** `src/visualization/panels_conditioning.py`, `plot_panel_m()`
**figsize:** (11.2, 12.0) -- 3 rows

**User feedback:** "Move Row 1 legends closer (not at bottom). Show PCA as UMAP instead. Reduce height for compactness."

**Issues found:**
1. Stale suptitle in PNG
2. Figure-level type-color legend at very bottom (`bbox_to_anchor=(0.5, 0.005)`) -- far from Row 1 where colors are used
3. Heatmap cell annotations at fontsize=6 -- extremely small
4. 10 panel labels (a-j) for one figure is excessive

**Actions:**
- [ ] Move the type-color legend from bottom to directly below Row 1
- [ ] Reduce hspace between rows to compact the figure
- [ ] Increase heatmap annotation fontsize from 6 to at least 7
- [ ] Consider reducing from 3 rows to 2 rows (merge Row 2 and Row 3 metrics into fewer panels)
- [ ] Consider showing UMAP instead of PCA for Row 1 visualizations

### Figure 10: Diversity Diagnostics

**Source:** `src/visualization/panels_diversity.py`, `plot_diagnostics()`
**figsize:** (9.0, 7.5) -- 2x2 grid

**User feedback:** "Add (a)-(d) labels and enhance figure caption."

**Issues found:**
1. Panel labels (a)-(d) already present in code
2. No stale suptitle (only figure without one!)
3. J4 x-axis shows numeric type IDs instead of names
4. J1 legend overlaps with bar data at lower-right
5. Caption claims "abbreviated" type names but code does hard truncation

**Actions:**
- [ ] Fix J4 to show type names (or abbreviated names) instead of numeric IDs
- [ ] Move J1 legend to upper-right to avoid overlap with bars
- [ ] Fix caption claim about "abbreviated" names
- [ ] Enhance caption with quantitative summary

### Figure 11: Diversity Trade-off (Merged L+K+Violin)

**Source:** `results_visualizer.py`, `_compose_diversity_tradeoff()`
**Composition:** Hybrid PIL raster + matplotlib

**Issues found:**
1. **DUPLICATE panel labels:** Panel L has (a), Panel K has (a)(b) -- conflict
2. Stale suptitle visible
3. The merged figure is NOT used in the LaTeX (standalone L and K are used instead)
4. The violin panel at bottom has a text box with border -- user wants no border/shadow/background

**Actions:**
- [ ] Since LaTeX already uses standalone panels, the merged composition may not need fixing
- [ ] If keeping merged: fix panel labels to be sequential, remove duplicate (a)
- [ ] In violin panel (`panels_metrics.py`): remove bbox border from inset text annotation
- [ ] Verify which version (merged or standalone) the LaTeX actually uses

### Figure 12: Baseline Comparison

**Source:** `src/visualization/baseline_panels.py`, `plot_baseline_comparison()`
**figsize:** (13.8, 5.8) -- 1x3 grid

**User feedback:** "Enrich content."

**Issues found:**
1. **MISSING panel (b) label** -- added to placeholder axes that gets removed
2. Stale suptitle in PNG
3. Panel O3 has extreme improvement values (up to 3000%) that compress smaller bars
4. Caption says "heatmap" but code renders horizontal bars

**Actions:**
- [ ] Fix (b) label: add `add_panel_label()` to the actual `ax_radar` after it's created (not the placeholder)
- [ ] Cap O3 improvement at a reasonable max (e.g., 300%) with annotation for extremes
- [ ] Fix caption: "heatmap" -> "horizontal bar chart"
- [ ] Consider adding a 4th panel showing per-metric comparison detail

### Figure 13: Benchmark

**Source:** `src/visualization/benchmark_panels.py`, `plot_benchmark_panel()`
**figsize:** (15.0, 9.5) -- 2x2 grid

**User feedback:** "Enrich content."

**Issues found:**
1. Stale suptitle in PNG
2. S3 EmbeddingVAE FD (~1.8) dominates y-axis, hiding other methods
3. S4 legend partially truncated at right edge
4. S4 method names truncated with ellipsis

**Actions:**
- [ ] Use log-scale or broken axis for metrics where one value dominates (FD in S3)
- [ ] Move S4 legend inside the plot area or increase right margin
- [ ] Show full method names on S4 y-axis
- [ ] Consider adding a 5th panel or splitting into more focused comparisons

### Figure 14: Downstream (Merged P+Q)

**Source:** `src/visualization/downstream_panels.py`, `plot_clustering_and_classifier_merged()`
**figsize:** (15.2, 10.6) -- 2x3 grid

**User feedback:** "Check for potential content enrichment."

**Issues found:**
1. Stale suptitle in PNG
2. Panel P2 y-tick labels at fontsize=7 (at print scale ~3.3pt)
3. Q2 stat box slightly overlaps with heatmap corner

**Actions:**
- [ ] Increase P2 y-tick fontsize from 7 to 8
- [ ] Adjust Q2 stat box position to avoid overlap
- [ ] Consider adding per-type detail (e.g., highlight best/worst 5 types with labels in P1 UMAP)

### Figure 15: DE Concordance

**Source:** `src/visualization/panels_de_concordance.py`, `plot_de_concordance_panel()`
**figsize:** (14.8, 7.2) -- 1x3 grid

**User feedback:** "Left panel shows too few names. Middle heatmap only 3 rows -- add more."

**Issues found:**
1. Stale suptitle in PNG
2. R1 scatter shows few gene name annotations (limited to top genes by effect size)
3. R2 heatmap has only 3 rows (3 DE contrasts) x 4 metrics
4. R3 x-tick labels truncated to 8 chars with "..."
5. Caption doesn't use bold (a)(b)(c) references despite labels in figure

**Actions:**
- [ ] R1: Increase number of annotated gene names (currently seems limited)
- [ ] R2: Add more DE contrasts (e.g., add 2-3 more biologically meaningful contrasts from the downstream data)
- [ ] R3: Show full contrast names (remove 8-char truncation)
- [ ] Update caption to reference (a)(b)(c) with bold formatting

---

## Part 3: Bibliography Expansion

### Current: 27 references. Target: 45-50+

### 20 Suggested New References (by topic):

**Diffusion/Flow (foundational):**
1. Ho et al. "Denoising diffusion probabilistic models." NeurIPS 2020
2. Song et al. "Score-based generative modeling through SDEs." ICLR 2021
3. Tong et al. "Improving flow-based models with minibatch OT." TMLR 2024

**Single-cell foundation models:**
4. Hao et al. "Large-scale foundation model on single-cell transcriptomics (scFoundation)." Nat Methods 2024
5. Wen et al. "CellPLM: Pre-training of cell language model." ICLR 2024
6. Rosen et al. "Universal cell embeddings (UCE)." NeurIPS Workshop 2023

**Single-cell generative models:**
7. Bunne et al. "Learning single-cell perturbation responses using neural OT (CellOT)." Nat Methods 2023
8. Lotfollahi et al. "Mapping single-cell data to reference atlases (scArches)." Nat Biotechnol 2022
9. Gayoso et al. "scvi-tools: a Python library for probabilistic scRNA-seq." Nat Biotechnol 2022

**Text-to-image / CLIP:**
10. Ramesh et al. "Hierarchical text-conditional image generation with CLIP latents (DALL-E 2)." arXiv 2022
11. Saharia et al. "Photorealistic text-to-image diffusion (Imagen)." NeurIPS 2022

**Contrastive / evaluation:**
12. Oord et al. "Representation learning with contrastive predictive coding (InfoNCE)." arXiv 2018
13. Heusel et al. "GANs trained by a two time-scale update rule (FID)." NeurIPS 2017
14. Luecken et al. "Benchmarking atlas-level data integration." Nat Methods 2022

**Benchmark / atlas:**
15. Tabula Muris Consortium. "Single-cell transcriptomics of 20 mouse organs." Nature 2018
16. Regev et al. "The Human Cell Atlas." eLife 2017

**Technical foundations:**
17. Kingma & Welling. "Auto-encoding variational Bayes (VAE)." ICLR 2014
18. Hu et al. "LoRA: Low-rank adaptation." ICLR 2022

**Additional biology:**
19. Zheng et al. "Massively parallel digital transcriptional profiling of single cells (10x Genomics)." Nat Commun 2017
20. Stuart et al. "Comprehensive integration of single-cell data (Seurat)." Cell 2019

### Citation Ordering

All \bibitem entries must be reordered to match first-appearance order in the text body. Current order has at least 5 violations.

---

## Part 4: Missing Abbreviations

### 19 abbreviations to add:

| Abbreviation | Expansion |
|---|---|
| MSI | Marker Specificity Index |
| BCC | Basal Cell Carcinoma |
| AML | Acute Myeloid Leukemia |
| ALL | Acute Lymphoblastic Leukemia |
| PBMC | Peripheral Blood Mononuclear Cell |
| NK | Natural Killer |
| ESC | Embryonic Stem Cell |
| HBV | Hepatitis B Virus |
| LPS | Lipopolysaccharide |
| IFN | Interferon |
| ALS | Amyotrophic Lateral Sclerosis |
| ASD | Autism Spectrum Disorder |
| CE | Cross-Entropy |
| InfoNCE | Info Noise-Contrastive Estimation |
| SigLIP | Sigmoid Loss for Language-Image Pre-training |
| LoRA | Low-Rank Adaptation |
| CRISPR | Clustered Regularly Interspaced Short Palindromic Repeats |
| CLIP | Contrastive Language-Image Pre-training |
| BERT | Bidirectional Encoder Representations from Transformers |

---

## Part 5: Data Availability & Factual Verification

### CUDA Version Fix
- **3 locations** report "CUDA~12" — actual is CUDA 13.0
  - Line 208: `PyTorch~2.1 (CUDA~12)`
  - Line 211: `PyTorch~2.1 with CUDA~12`
  - Line 595 (data availability): `CUDA~12`

### File Path Verification (ALL PASS)
All 8 referenced file paths verified to exist on disk:
- `requirements.txt` -- exists
- `configs/clop_v9.3.yaml` -- exists
- `configs/dit.yaml` -- exists
- `scripts/regenerate_report.sh` -- exists
- `docs/QUICK_START.md` -- exists
- `src/data_pipeline/subcluster_annotation.py` -- exists
- `scripts/02b_enrich_descriptions.py` -- exists
- `src.evaluation.downstream_biology` module -- exists

### Figure S1 Reference
Supplementary text references "Figure S1 (per-type KNN accuracy)" but no corresponding figure environment exists. Must either create the figure or remove the reference.

---

## Part 6: Execution Order

### Group A (parallel, no dependencies):
- Fix panel labels in Figs 3, 5, 11, 12 (code changes)
- Remove stale suptitle remnants and fig.text() banners
- Fix CUDA version in 3 locations
- Add missing abbreviations
- Fix Figure S1 reference

### Group B (parallel, no dependencies):
- Per-figure content improvements (Figs 1-15 as detailed above)
- Add 20 new bibliography entries
- Reorder bibliography by first appearance

### Group C (after Groups A & B):
- Font consistency audit across all panels
- Regenerate ALL figures via `python -m src.visualization.results_visualizer`
- Regenerate architecture figure via `python scripts/generate_architecture_figure.py`

### Group D (after Group C):
- Run VCD on all regenerated figures
- Verify LaTeX compilation
- Git commit

---

## Workflow Documentation (for future Claude sessions)

### Figure Generation Pipeline:
1. **Individual panels:** `python -m src.visualization.results_visualizer --no-umap --dpi 300`
2. **Architecture:** `python scripts/generate_architecture_figure.py`
3. **Article delivery (symlinks):** `python -m src.visualization.article_delivery`
4. **LaTeX compilation:** `cd articles && latexmk -pdf clop_dit_biology.tex`
5. **VCD checking:** Automatic via `save_with_vcd()` during generation

### Key style constants (from `src/visualization/style.py`):
- Title: 11pt (`FONT_TITLE`)
- Labels: 10pt (`FONT_LABEL`)
- Ticks: 10pt (`FONT_TICK`), dense: 8pt (`FONT_TICK_DENSE`)
- Panel labels: 12pt bold (via `add_panel_label()`)
- Minimum: 7pt (`FONT_SMALL`)
- Composed scale factor: 0.95 (single-column MDPI)

### VCD minimum thresholds:
- Min font: 5.5pt effective
- Min contrast: 3.0 (WCAG)
- Label density: 0.92
