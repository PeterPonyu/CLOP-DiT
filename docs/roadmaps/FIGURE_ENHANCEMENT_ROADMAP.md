# Figure Enhancement Roadmap

This document outlines a prioritized roadmap for enhancing the 15 article figures to show richer, more insightful results. All enhancements build on data already computed by the pipeline; no new computations are required for most.

---

## Current State Analysis

### Hidden Data (Computed but Not Visualized)
The pipeline computes rich statistics that are underutilized:

| Data | Location | Current Usage | Potential |
|------|----------|----------------|-----------|
| Per-type centroid cosine / FD | `generation_metrics.json` | Summary stats only | Per-type failure detection |
| Per-type cosine distributions | `diversity_diagnostics.json` | Aggregate diversity | Distribution tails, outliers |
| Per-type classifier accuracy | `classifier_alignment.json` | Aggregate score (0.308) | Type-by-type performance matrix |
| Per-gene DE logFC + p-values | `de_concordance.json` | Top-k overlap only | Effect-size weighted concordance |
| Bootstrap CIs (FD, cosine, diversity) | `benchmark_report.json` | Shown in Panel S only | Propagate to all metrics |
| Confusion matrix | Downstream inference | Overall accuracy | Per-type precision/recall |

**Key insight:** The 7 "stale" figures (Figs 3, 4, 5, 9, 10, 11, 14, 15) weren't just updated for bug fixes—they received **significant visual enhancements** (figure sizes reduced, titles lost panel-letter prefixes, legends restructured). These changes are **not yet reflected in the PDFs**. Regenerating will unlock the baseline for further enhancement work.

---

## Enhancement Priority Matrix

### **TIER 1: High Impact + High Feasibility (1-3 hours each, implement immediately after regeneration)**

#### 1.1 Per-Type Performance Heterogeneity (Enhancement to Figs 4, 6, 7)
**Problem:** Current figures hide catastrophic failures. Example: Inhibitory GABAergic neurons have centroid cosine of 0.39 vs 0.92+ for others.

**Solution:** Add 2-panel scatter plot:
- **Panel A (Per-type Cosine vs Cell Count):**
  - X: Number of real cells (cell type abundance)
  - Y: Per-type centroid cosine similarity (generation quality)
  - Size: Point size = per-type variance
  - Color: Cell type category (immune, epithelial, neural)
  - Trend: Add regression line to show abundance-quality correlation

- **Panel B (Per-type Frechet with CI):**
  - X: Cell type (sorted by FD)
  - Y: Per-type Frechet Distance with bootstrap CI whiskers
  - Highlight outliers (FD > 2σ above mean)
  - Color code by tissue

**Implementation:**
```python
# File: src/visualization/panels_quality.py
# Add function: plot_per_type_heterogeneity()
# Input: generation_metrics.json (per_type section: dict with cosine, fd stats)
# Output: fig_per_type_heterogeneity.pdf (2-panel scatter + bar)
```

**Data depends on:**
- `generation_metrics.json['per_type']` — contains `cosine_sim`, `frechet_distance` per type
- Cell type metadata (abundance) from training data

**Expected finding:** Reveals **underrepresented cell types struggle** or **biologically distinct types are inherently hard**.

---

#### 1.2 Distribution Tails & Outlier Analysis in Diversity (Enhancement to Figs 10, 11) — DONE for Fig 11
**Problem:** Current diversity metrics show aggregate ratios, not whether diversity is distributed or tail-weighted.

**Solution:** Replace or supplement current Fig 11 with **violin plots** (implemented in `plot_diversity_distributions_violin`, composed in `_compose_diversity_tradeoff`):
- **For top 6 cell types** (by diversity):
  - X-axis: Cell type
  - Y-axis: Cosine similarity distribution
  - Three violin curves per type: Real-real, Gen-gen, Real-gen
  - Overlay: 0.25, 0.5, 0.75 quantile bands (horizontal lines)
  - Point overlay: Individual samples (alpha=0.3)

**Implementation:**
```python
# File: src/visualization/diversity_diagnostics.py
# Add function: plot_diversity_distributions_violin()
# Input: diversity_diagnostics.json (nested distribution data per type)
# Output: fig_diversity_distributions.pdf
```

**Data depends on:**
- `diversity_diagnostics.json['per_type'][type_name]['distributions']` — mean/std/median/raw quantiles

**Expected finding:** Shows if diversity is **genuine breadth vs artificial tail inflation**; informs biological realism assessment.

---

#### 1.3 Classifier Per-Type Performance Heatmap (Enhancement to Fig 14)
**Problem:** Panel Q shows overall accuracy (0.308), but masks that some types reach 1.0 while others are 0.0.

**Solution:** Replace simple bar chart with **precision-recall heatmap**:
- **Rows:** Predicted cell type
- **Columns:** True cell type
- **Cell color intensity:** Per-cell-type accuracy / F1 score
- **Cell annotations:** Count of samples (real/generated) in diagonal; predicted/true on off-diagonal
- **Row/column ordering:** Hierarchical cluster by similarity

**Implementation:**
```python
# File: src/visualization/downstream_panels.py
# Modify: plot_classifier_panel() to include heatmap
# Add function: plot_classifier_per_type_heatmap()
# Input: classifier_alignment.json (confusion matrix)
# Output: Enhanced Panel Q with 2-panel layout (global ROC + per-type heatmap)
```

**Data depends on:**
- `classifier_alignment.json['confusion_matrix']` — compute per-type precision/recall from this

**Expected finding:** Identifies **which cell types are catastrophic failures** vs successes; explains downstream task feasibility.

---

#### 1.4 DE Concordance: Effect Size Weighted Gene Ranking (Enhancement to Fig 15)
**Problem:** Current overlap visualization treats all genes equally; ignores biological significance (effect size, p-value).

**Solution:** Add **effect-size-weighted scatter plot** for top contrast:
- **X-axis:** Real logFC
- **Y-axis:** Generated logFC
- **Point size:** |logFC| (effect size magnitude — larger points = stronger effects)
- **Point color:** -log10(p_adj) significance (warmer colors = more significant)
- **Highlight:** Points in opposite quadrants (sign-disagreement = concerning)
- **Marginalia:** Histograms on axes showing logFC distributions

**Implementation:**
```python
# File: src/visualization/downstream_panels.py
# Add function: plot_de_effect_size_weighted()
# Input: de_concordance.json (per-contrast per-gene logFC, pval)
# Output: fig_de_effect_size.pdf (6-panel, one per contrast)
```

**Data depends on:**
- `de_concordance.json['contrasts'][contrast_name]` — logFC, scores, p-values per gene

**Expected finding:** Reveals **whether the method agrees on *biologically important* genes** vs just overlapping in rank order; addresses clinical relevance.

---

### **TIER 2: High Impact + Moderate Feasibility (2-4 hours each, implement in phase 2)**

#### 2.1 Confidence Intervals Propagated Through Downstream Tasks (Figs 14, 15)
**Problem:** Classifier accuracy shown as point (0.308); no uncertainty intervals anywhere except Panel S.

**Solution:** Add **bootstrap CI bars** to:
- **Panel Q:** Per-type accuracy bars → compute binomial proportion CI (95% Wilson)
- **Panel R:** Contrast concordance metrics → resampling CI across contrast splits
- **Panel S:** Already has CIs; amplify their visibility with error band background shading

**Implementation:**
```python
# File: src/visualization/downstream_panels.py
# Add function: compute_binomial_ci() (Wilson score interval)
# Modify: plot_classifier_panel(), plot_de_concordance_panel()
# Output: Enhanced Panels Q+R with CI whiskers/bands
```

**Data depends on:**
- Raw confusion matrices (for binomial CI)
- Per-contrast logFC arrays (for resampling CI)

**Expected finding:** Communicates **which findings are statistically robust vs noisy**; differentiates signal from sampling artifacts.

---

#### 2.2 Mixing Score vs Cell Type Size (Enhancement to Fig 14)
**Problem:** Mixing scores range 0.15–0.57; no contextualization by cell type abundance or category.

**Solution:** **Scatter plot with trend**:
- **X-axis:** Real cell count (cell type abundance in data)
- **Y-axis:** kNN mixing score
- **Color:** Cell type category (immune, epithelial, neural, other)
- **Size:** Marker size = mixing score magnitude
- **Trend:** Add LOWESS regression line with 95% CI band
- **Annotations:** Label outliers (types with mismatched abundance/mixing)

**Implementation:**
```python
# File: src/visualization/downstream_panels.py
# Add function: plot_mixing_vs_abundance()
# Input: clustering_alignment.json (mixing scores), generation_metrics.json (cell counts)
# Output: fig_mixing_heterogeneity.pdf
```

**Data depends on:**
- `clustering_alignment.json` — per-type mixing scores
- Training data cell type counts (from cached latents metadata)

**Expected finding:** Shows **whether underrepresented types are systematically harder to mix** (important for rare disease modeling).

---

### **TIER 3: Moderate Impact + Moderate-High Feasibility (2-4 hours each, consider for future versions)**

#### 3.1 KL Divergence Decomposition by Latent Dimensions (Specialized new panel)
**Problem:** Aggregate KL divergence shown (0.024), but **where do actual distribution mismatches occur?**

**Solution:** Create **per-dimension KL heatmap**:
- **Left panel:** 1D distribution overlays for top 5 PCs (real blue vs generated red histograms)
- **Right panel:** Bar chart of per-PC KL contribution to total
- **Annotations:** PC1 contributes 45%, PC2 contributes 23%, etc.

**Implementation:**
```python
# File: src/visualization/results_visualizer.py
# Add function: plot_kl_by_dimension()
# Requires: Pre-computed PCA on embeddings + per-PC KL computation
# Output: fig_kl_divergence_decomposition.pdf
```

**Data depends on:**
- Embedding-level data (not summary statistics)
- PCA decomposition + per-PC KL (requires new computation if not cached)

**Expected finding:** Explains whether mismatches are in **rare modes (tail effects) or core distribution shift** (fundamental modeling failure).

---

#### 3.2 Per-Contrast DE Patterns by Cell Type Hierarchy (Enhancement to Fig 15)
**Problem:** Current Fig R shows 3 contrasts but doesn't contextualize whether within-tissue contrasts are easier than cross-tissue.

**Solution:** **Small-multiples matrix**:
- One subplot per contrast (6-12 contrasts total)
- Each subplot: Per-type accuracy / concordance score
- Row ordering: Hierarchical (immune, epithelial, neural)
- Meta-annotation: Highlight within-tissue (same color) vs cross-tissue (different color) contrasts

**Implementation:**
```python
# File: src/visualization/downstream_panels.py
# Add function: plot_de_by_hierarchy()
# Input: de_concordance.json (per-contrast stats), cell type hierarchy metadata
# Output: fig_de_hierarchy.pdf (small multiples, 6-12 subplots)
```

**Data depends on:**
- `de_concordance.json` — all contrasts + concordance scores
- Cell type hierarchy (tissue category, lineage) metadata

**Expected finding:** Reveals whether **cross-tissue contrasts are systematically harder** (informs applicability to batch integration/cross-species studies).

---

### **TIER 4: Lower Priority / Optional (Only if strong reviewer feedback)**

#### 4.1 Temporal Diversity Emergence: Checkpoint Ablation (Requires new data collection)
**Problem:** Figures 2 (training) and 10 (diversity) show final state, not when diversity emerges.

**Solution:** **Line plot with dual Y-axis:**
- Y1 (left): Validation loss over training
- Y2 (right): Diversity ratio, Coverage, Centroid cosine
- Highlight point where each metric plateaus
- Shade region where quality-diversity trade-off is visible

**Implementation challenge:** Requires checkpoint metrics saved during training (may not currently be captured). Would need to modify training script to log per-checkpoint diversity metrics.

**Data depends on:**
- Training checkpoint data (diversity_diagnostics.json computed per checkpoint)

---

#### 4.2 Heatmap Dendrogram for DE Genes (Enhancement to Fig 15)
**Problem:** Top genes per contrast shown but no hierarchical grouping; misses co-regulation patterns.

**Solution:** Replace simple heatmap with **hierarchical clustered version**:
- Rows: Top DE genes (hierarchical clustered by logFC pattern across contrasts)
- Columns: Contrasts
- Color: Real logFC (left half) vs Generated logFC (right half) side-by-side
- Dendrograms show which genes co-regulate together

**Implementation:**
```python
# File: src/visualization/downstream_panels.py
# Add function: plot_de_dendrogram_heatmap()
# Input: de_concordance.json (all genes + logFC)
# Output: fig_de_dendrogram.pdf (hierarchical heatmap)
```

**Expected finding:** Shows if **real & generated agree on gene regulatory modules** (co-expression patterns), addressing systems-level biological fidelity.

---

## Implementation Checklist

### Phase 1 (Post-regeneration): Core 4 enhancements
- [ ] **1.1** Per-type heterogeneity scatter (2h)
  - [ ] Read `generation_metrics.json` per_type section
  - [ ] Create scatter function with CI bands
  - [ ] Add trend line (abundance vs quality)
  - [ ] Test and save fig_per_type_heterogeneity.pdf

- [ ] **1.2** Diversity violin plots (2h)
  - [ ] Extract distribution data from `diversity_diagnostics.json`
  - [ ] Create violin plot function
  - [ ] Overlay quantile marks
  - [ ] Save fig_diversity_distributions.pdf

- [ ] **1.3** Classifier heatmap (1.5h)
  - [ ] Parse confusion matrix from `classifier_alignment.json`
  - [ ] Compute per-type precision/recall
  - [ ] Create hierarchical heatmap
  - [ ] Integrate into downstream_panels.py

- [ ] **1.4** DE effect-size scatter (2h)
  - [ ] Extract per-gene logFC + pvalues
  - [ ] Create scatter with size/color encoding
  - [ ] Add effect-size marginal histograms
  - [ ] Save fig_de_effect_size.pdf

### Phase 2 (If time permits): Extended confidence intervals
- [ ] **2.1** CI propagation (3h)
  - [ ] Implement Wilson score binomial CI
  - [ ] Add to classifier per-type bars
  - [ ] Add resampling CI to DE metrics
  - [ ] Integrate into Panels Q + R

- [ ] **2.2** Mixing vs abundance (1.5h)
  - [ ] Create scatter plot with LOWESS trend
  - [ ] Cross-reference abundance + mixing data
  - [ ] Add category coloring
  - [ ] Save fig_mixing_heterogeneity.pdf

### Phase 3 (Future versions): Advanced analytics
- [ ] **3.1** KL divergence decomposition (3h, requires embedding data)
- [ ] **3.2** DE by cell hierarchy (2h)
- [ ] **4.1** Temporal diversity tracking (4h, requires checkpoint data)
- [ ] **4.2** DE gene dendrogram (2h)

---

## Expected Outcomes

### Before Enhancements
- 15 figures show overall method performance
- Reader impression: "CLOP-DiT works well"
- Hidden: ~9 figures are stale (pre-regeneration)
- Reviewers may ask: "Which cell types fail? Why?"

### After Core Enhancements (Phase 1)
- 15 figures + 4 new panels (19 total visualizations)
- Reader impression: "CLOP-DiT works well **for most cell types** — here are the specific failures and why"
- Visible: Per-type heterogeneity, true distribution characteristics, classifier-level details, biologically-weighted concordance
- Reviewer concerns addressed: Clear failure modes, confidence intervals, effect sizes matter

### With Full Enhancement Suite
- **Robustness:** Abundance-quality correlation shows what matters
- **Biological realism:** Distribution tails & outliers scrutinized
- **Limitations:** Per-type failures highlighted; downstream task feasibility clear
- **Statistical rigor:** CIs throughout, effect sizes weighted appropriately
- **Mechanistic clarity:** KL decomposition shows where model struggles; hierarchy analysis shows cross-tissue challenges

---

## File Modification Strategy

### New functions to add:
1. `src/visualization/panels_quality.py::plot_per_type_heterogeneity()` — Figs 4, 6, 7 enhancement
2. `src/visualization/diversity_diagnostics.py::plot_diversity_distributions_violin()` — Figs 10, 11 enhancement
3. `src/visualization/downstream_panels.py::plot_classifier_per_type_heatmap()` — Fig 14 enhancement
4. `src/visualization/downstream_panels.py::plot_de_effect_size_weighted()` — Fig 15 enhancement
5. `src/visualization/downstream_panels.py::compute_binomial_ci()` — Helper for Tier 2

### Files to modify:
- `src/visualization/results_visualizer.py` — add calls to new functions, optionally extend report generation
- `src/visualization/style.py` — ensure color schemes consistent across new plots
- `scripts/pipeline/regenerate_report.sh` — optionally document new outputs

### Integration points:
- Call new functions from `ResultsVisualizer.generate_full_report()` after base panels are generated
- Save new PDFs to `results/figures/` following naming convention

---

## Next Steps (In Order)

1. ✅ **Complete code cleanup** (removed dead code, fixed SyntaxError) — DONE
2. ✅ **Verify figure pipeline** (confirm single source of truth for each figure) — DONE
3. ⏭️ **Regenerate pipeline** (run `bash scripts/pipeline/regenerate_report.sh` in your environment with PyTorch)
   ```bash
   # In your development environment with CUDA/PyTorch:
   cd /home/zeyufu/Desktop/CLOP-DiT
   bash scripts/pipeline/regenerate_report.sh
   ```
4. ⏭️ **Verify regeneration succeeded** (check all 15 PDFs updated)
   ```bash
   bash scripts/pipeline/verify_article_figures.sh
   ```
5. ⏭️ **Rebuild article** (confirm LaTeX builds with new figures)
   ```bash
   cd articles && latexmk -pdf clop_dit_biology.tex
   ```
6. ⏭️ **Implement Tier 1 enhancements** (4 new panels, 7-8 hours total)
7. ⏭️ **Test regeneration + enhancements** (full pipeline with new panels)
8. ⏭️ **Update FIGURE_ORGANIZATION.md** with new panel descriptions
9. ⏭️ **Rebuild article** with full panel suite

---

## Estimated Timeline
- **Regeneration:** ~30-60 min (depends on GPU/CPU)
- **Tier 1 enhancements:** ~8 hours (1.5-2.5h per panel)
- **Tier 2 extensions:** ~6 hours (spread across 2-3 enhancements)
- **Testing + refinement:** ~4 hours
- **Total:** ~24-30 hours for full suite (or ~10-12 hours for Core 4)

---

## Important Notes

- **All enhancement data already exists** in JSON outputs; no new metrics to compute
- **Visual design consistency:** Use existing color schemes and font settings from `style.py`
- **VCD compatibility:** New plots should pass `save_with_vcd()` checks; test with VCD enabled
- **Article impact:** Each new panel adds ~2-3 KB to article PDF; 4 new panels = ~10 KB (acceptable)
- **Reproducibility:** All enhancements deterministic (same data → same visual); no stochasticity
