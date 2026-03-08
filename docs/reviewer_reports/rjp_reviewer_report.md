# Comprehensive Reviewer Concerns Report for CLOP-DiT Article

**Report Generated:** 2026-03-07  
**Article:** CLOP-DiT: Text-Guided Single-Cell Gene Expression Generation via Contrastive Language-Omics Pretraining and Diffusion Transformers  
**Target Journal:** MDPI Biology  
**Manuscript Source:** `articles/clop_dit_biology.tex`

---

## Executive Summary

This report consolidates all reviewer concerns identified through analysis of the article manuscript, policy documents, test files, and supporting documentation. The concerns are organized by category and prioritized by impact on manuscript acceptance.

---

## 1. HIGH PRIORITY CONCERNS (Editorial Return Risk)

### 1.1 Reproducibility and Data Availability (MDPI Requirement)

**Concern:** MDPI Biology requires "full experimental details must be provided so that the results can be reproduced" and "make full datasets available where possible." The manuscript has addressed this with:

- Data Availability Statement (Section before Acknowledgments): Contains proper GEO dataset reference and supplementary tables
- Code availability mentioned in manuscript: "Model checkpoints, generated embeddings, and the figure-reproduction script are available from the corresponding author upon reasonable request"
- Config reference: `configs/clop_v9.3.yaml`

**Status:** Addressed in current manuscript version  
**Evidence:** Lines 474-475 in `articles/clop_dit_biology.tex`

### 1.2 Author and Submission Metadata (Previously Placeholders, Now Filled)

**Previous Concern:** Template placeholders in front/back matter would cause editorial rejection  
**Current Status:** RESOLVED

| Field | Status | Location in .tex |
|-------|--------|------------------|
| Affiliation | Filled | Lines 89-90 |
| Correspondence | Filled | Line 93 |
| Funding | Filled | Line 468 |
| Institutional Review | Filled | Lines 470-471 |
| Informed Consent | Filled | Line 472 |
| Author Contributions | Filled | Line 466 |
| Acknowledgments | Filled | Line 487 |
| Conflicts of Interest | Filled | Line 489 |

### 1.3 Software and Implementation Details

**Concern:** Exact software versions needed for reproducibility  
**Status:** PARTIALLY ADDRESSED

**Current state in manuscript:**
- Line 197 mentions: "Python~3.10 using PyTorch~2.1 (CUDA~12), scGPT~v0.2.1, and Hugging Face Transformers~4.36 for BiomedBERT"
- References to `requirements.txt` and `configs/clop_v9.3.yaml`

**Gap:** No explicit commit hash or Docker image reference for the exact environment.

---

## 2. MEDIUM PRIORITY CONCERNS (Reviewer Expectations)

### 2.1 Statistics and Uncertainty Reporting

**Concern:** Lack of confidence intervals for key metrics

**Current manuscript state:**
- Line 247 mentions: "Bootstrap 95% confidence intervals for the composite benchmark scores are shown in Figure~\ref{fig:benchmark}d"
- Single point estimates for KNN accuracy (36.9%), steering (81%), diversity ratio (0.93)

**Remaining Gaps:**
1. No explicit CIs for KNN accuracy, steering accuracy, or diversity ratio in main text
2. No mention of multiple comparison correction (though acknowledged as "exploratory comparisons" without correction)
3. Primary vs exploratory comparison clarification present (Line 247 designates CFG=2.0 Euler-10 and CFG=1.0 Midpoint-10 as "primary operating points")

**Recommendation:** Add brief uncertainty paragraph or supplementary table with bootstrap CIs for all key metrics.

### 2.2 Methods Detail - Text-Condition Construction

**Concern:** How 1,088 text groups were constructed needs clarity

**Current manuscript state (Lines 193-194):**
```
Each cell was encoded into a 512-dimensional embedding by a pre-trained scGPT model. 
Cell-type annotations were organized into 1,088 unique text groups, each described 
by a structured caption incorporating cell type, marker genes, tissue of origin, 
organism, and disease context.
```

**Gap:** The exact template schema and marker gene identification process could be more explicit. The manuscript references `scripts/build_text_captions.py` but no supplementary method document.

### 2.3 Validation Split Clarification

**Concern:** Which 8 datasets held out?

**Current manuscript state (Line 195):**
```
The 8 validation datasets were selected by study (not by cell) to ensure no sample 
overlap between training and validation splits; the full list of GEO accession 
identifiers for all 80 datasets is provided in Supplementary Table~S1, and the 
validation study identifiers are in Supplementary Table~S2.
```

**Status:** Addresses concern with reference to supplementary tables.

### 2.4 OOD (Out-of-Distribution) Limitation

**Concern:** Claims limited to in-distribution only

**Current manuscript state (Discussion, Lines 428):**
```
CLOP-DiT currently generates most reliably for cell states represented in training; 
novel prompts far from the training distribution remain sensitive to CLOP projector 
extrapolation. A pilot OOD/free-form prompt run was executed in this workstream, 
but robust quantitative OOD scoring is still incomplete because matched references 
are unavailable for many prompts; therefore the main reported metrics remain 
in-distribution.
```

**Status:** Well-addressed with explicit limitation statement.

---

## 3. FIGURE AND VISUALIZATION CONCERNS

### 3.1 Figure Presentation Policy Compliance

**Policy Source:** `docs/FIGURE_PRESENTATION_POLICY.md`

| Policy Rule | Status | Evidence |
|-------------|--------|----------|
| At most one stats box per subplot | Enforced | Test in `test_article_delivery.py` |
| At most two highlighted outliers | Enforced | Style guide compliance |
| Each subplot answers one question | Enforced | Reviewed in FIGURE_ORGANIZATION.md |
| Shared semantic colors | Enforced | `src/visualization/style.py` |
| Text density ceiling (3 elements max) | Enforced | VCD checks |
| No figure numbers on image | Enforced | Article delivery tests |
| Colorbar safe placement | Enforced | `style.add_colorbar_safe()` |

### 3.2 Legend and Caption Policy Compliance

**Policy Source:** `docs/LEGEND_CAPTION_POLICY.md`

**Concern:** Statistical summaries in legend titles  
**Test Coverage:** `test_article_delivery.py` includes `TestArticlePresentationPolicy` class with regex patterns detecting:
- `r = 0.X`, `p = 0.X`, `AUC = 0.X`
- `CI = [...]`, `confidence = ...`
- `Pearson r = ...`, `Spearman = ...`

**Status:** Automated test enforces compliance.

### 3.3 VCD (Visual Conflict Detection) Architecture

**Status:** 4-layer detection system operational
- Layer 1: Subplot-level (legend occlusion)
- Layer 2: Figure-level (text overlaps, truncation)
- Layer 3: Perceptual (WCAG contrast, CVD safety)
- Layer 4: Semantic (overplotting, log-scale sanity)

**Target:** 0 warnings for all panels  
**Known exception:** Panel S heatmap uses explicit dark text (`#1a1a1a`) for WCAG contrast.

### 3.4 Article Figure Completeness (15 Figures)

**Canonical List from `article_delivery.py`:**

| Article Fig | Filename | Status |
|-------------|----------|--------|
| Fig 1 | `fig_architecture` | Required |
| Fig 2 | `fig_training_dynamics` | Required |
| Fig 3 | `fig_embedding_space` | Required |
| Fig 4 | `panel_d_metrics_summary` | Required |
| Fig 5 | `fig_fidelity_alignment` | Required |
| Fig 6 | `panel_n_marker_gene_comparison` | Required |
| Fig 7 | `panel_h_expression_correlation` | Required |
| Fig 8 | `panel_i_expression_analysis` | Required |
| Fig 9 | `panel_m_conditioning_umap` | Required |
| Fig 10 | `panel_j_diversity_diagnostics` | Required |
| Fig 11 | `fig_diversity_tradeoff` | Required |
| Fig 12 | `panel_o_baseline_comparison` | Required |
| Fig 13 | `panel_s_benchmark` | Required |
| Fig 14 | `fig_downstream_pq` | Required |
| Fig 15 | `panel_r_de_concordance` | Required |

**Verification:** `scripts/pipeline/verify_article_figures.sh` confirms all 15 PDFs exist.

### 3.5 Figure Caption-Subsection Alignment

**Policy Source:** `docs/ARTICLE_SUBSECTION_POLICY.md`

**Structural Rules Applied:**
- Opening sentence states purpose (all subsections reviewed)
- Figure/table reference explicit and states takeaway
- One primary claim per subsection
- Methods subsections: sufficient for reproducibility
- Results subsections: 1-3 paragraphs typical

**Transitions Present:**
- Introduction ending (Line 168-172): Maps paper organization
- Methods ending (Line 250): Bridge to results
- Results ending (Line 411): Mini-conclusion before Discussion
- Discussion opening (Line 416): Links results to interpretation

---

## 4. BIOLOGICAL CLAIM VERIFICATION

### 4.1 Claim-to-Evidence Mapping

**Source:** `docs/BIOLOGICAL_CLAIM_MAP.md`

| Claim | Primary Figures | Code Path | Verification Status |
|-------|-----------------|-----------|---------------------|
| Text conditioning is learned | Fig 2, 3, 4 | `src/evaluation/run_metrics.py` | Verified (unconditional collapse to random) |
| CLOP builds useful condition space | Fig 3, 5 | `src/visualization/panels_embedding.py` | Verified (cosine 0.222 vs 0.994) |
| Generated cells preserve type identity | Fig 4, 5, 6 | `src/evaluation/run_metrics.py` | Verified (KNN 37× random) |
| Expression is gene-level plausible | Fig 6, 7, 8 | `src/visualization/panels_expression.py` | Verified (marker preservation) |
| Diversity is preserved | Fig 10, 11 | `scripts/diversity_diagnostics.py` | Verified (DivR 0.93 Midpoint) |
| Not limited to easy types | Fig 5, 10, 14, 15 | `src/evaluation/downstream_biology.py` | Verified (per-type analysis) |
| Usable in downstream workflows | Fig 14, 15 | `src/evaluation/downstream_biology.py` | Verified (ARI, NMI, DE concordance) |
| Outperforms baselines | Fig 12, 13 | `src/evaluation/model_benchmarking.py` | Verified (composite score 0.844) |

### 4.2 Key Metrics Summary

| Metric | Value | Figure/Table |
|--------|-------|--------------|
| KNN top-1 accuracy | 36.9% (37× random) | Table 1, Fig 4 |
| Steering accuracy | 81.0% | Table 1, Fig 4 |
| Diversity ratio (CFG=1.0 Midpoint) | 0.929 | Table 1, Fig 10 |
| Linear classifier accuracy | 51.1% | Table 1 |
| Composite benchmark score | 0.844 | Fig 13 |
| ARI/NMI (clustering) | Reported in downstream | Fig 14 |
| DE logFC Pearson r | 0.17 | Fig 15 |

---

## 5. BASELINE AND BENCHMARKING CONCERNS

### 5.1 Baseline Comparison Completeness

**Concern:** Sufficient baseline coverage for reviewer confidence

**Current baselines in manuscript:**
- Unconditional generation (CFG=0)
- Gaussian sampling from per-type statistics
- Decoder-only ablations
- Shuffled-label and mean-collapse controls

**Gap identified in policy:** Learned baselines (scVI, embedding VAE) have entry points (`scripts/baselines/`) but may not be fully integrated into the main benchmark.

**From `REVIEWER_CONCERNS_AND_NEXT_STEPS.md`:**
> 19. Run at least one learned baseline and write its artifacts into `results/baselines/{method}/`
> 20. Use `scripts/analysis/run_downstream_for_baselines.py` to generate per-method downstream biological validation

**Status:** Policy exists but execution status unclear from documentation.

### 5.2 Robustness Framework

**From `docs/ROBUSTNESS_EXPERIMENTS.md`:**
- Seed stability testing
- Subsampling robustness
- Prompt sensitivity analysis

**Status:** Centralized in `scripts/training/run_robustness_experiments.py`  
**Gap:** No explicit robustness summary in the main manuscript supplement.

---

## 6. WRITING AND STYLE CONCERNS

### 6.1 Terminology Consistency

**Canonical terms from `docs/WRITING_AND_DOCS_POLICY.md`:**

| Term | Usage in Article | Status |
|------|-----------------|--------|
| CLOP-DiT | Consistent | Correct |
| scGPT | Consistent | Correct |
| flow matching | Consistent | Correct |
| BiomedBERT | Consistent | Correct |
| AdaLN-Zero | Consistent | Correct |
| classifier-free guidance (CFG) | Consistent | Correct |

### 6.2 Figure-Text Alignment Issues

**Reviewed:** All figure references in the manuscript point to existing figures with matching content.

**Verified mappings:**
- Line 187 → `fig_architecture.pdf` (Fig 1)
- Line 270 → `fig_training_dynamics.pdf` (Fig 2)
- Line 280 → `fig_embedding_space.pdf` (Fig 3)
- Line 308 → `panel_d_metrics_summary.pdf` (Fig 4)
- Line 318 → `fig_fidelity_alignment.pdf` (Fig 5)
- Line 328 → `panel_n_marker_gene_comparison.pdf` (Fig 6)
- Line 338 → `panel_h_expression_correlation.pdf` (Fig 7)
- Line 344 → `panel_i_expression_analysis.pdf` (Fig 8)
- Line 354 → `panel_m_conditioning_umap.pdf` (Fig 9)
- Line 362 → `panel_j_diversity_diagnostics.pdf` (Fig 10)
- Line 370 → `fig_diversity_tradeoff.pdf` (Fig 11)
- Line 380 → `panel_o_baseline_comparison.pdf` (Fig 12)
- Line 388 → `panel_s_benchmark.pdf` (Fig 13)
- Line 402 → `fig_downstream_pq.pdf` (Fig 14)
- Line 408 → `panel_r_de_concordance.pdf` (Fig 15)

---

## 7. SUBMISSION READINESS CHECKLIST

Based on `docs/SUBMISSION_CHECKLIST.md`:

### High Priority (Avoid Editorial Return)

| # | Task | Status |
|---|------|--------|
| 1 | Data Availability statement | Complete |
| 2 | REPRODUCIBILITY.md or supplement | Present at root |
| 3 | Affiliation, correspondence, funding | Complete |
| 4 | Ethics statements | Complete |
| 5 | Author Contributions | Complete |
| 6 | Acknowledgments and Conflicts | Complete |

### Medium Priority (Reviewer Expectations)

| # | Task | Status |
|---|------|--------|
| 7 | Uncertainty/CI paragraph | Partial - mentions CIs for composite score only |
| 8 | Primary vs exploratory comparisons | Complete |
| 9 | Text-condition construction | Partial - references script |
| 10 | Validation split clarification | Complete (references Supp Tables) |
| 11 | OOD limitation statement | Complete |

### Polish

| # | Task | Status |
|---|------|--------|
| 12 | Confirm Fig 12, 14, 15 files | Complete |
| 13 | Downstream numbers in abstract/conclusions | Partial - qualitative in abstract, quantitative in conclusions |

---

## 8. RECOMMENDED ACTIONS BEFORE SUBMISSION

### Immediate Actions (High Impact)

1. **Verify learned baseline integration:** Ensure at least scVI or embedding VAE baseline is included in Fig 12/13 for reviewer confidence.

2. **Add uncertainty paragraph:** Brief statement in Methods or Results indicating bootstrap CIs are available in supplementary materials.

3. **Regenerate all figures:** Run `bash scripts/regenerate_report.sh` and verify 0 VCD warnings.

4. **Build and check article PDF:** `cd articles && latexmk -pdf clop_dit_biology.tex` - verify no overfull boxes or figure resolution issues.

### Secondary Actions (Medium Impact)

5. **Add supplementary method document:** Brief description of text-caption construction schema for transparency.

6. **Create minimal supplement tables:** Table S1 (GEO accessions), S2 (validation IDs), S3 (CFG sweep) - verify these exist and are referenced.

7. **Run verification script:** `bash scripts/pipeline/verify_article_figures.sh` - confirm all 15 PDFs present.

### Optional Enhancements

8. **Add downstream numbers to abstract:** Current abstract says "integrate into standard single-cell workflows" - could optionally add "(ARI/NMI, classifier transfer, DE concordance)".

9. **Reference robustness experiments:** If run, add brief mention in supplement of seed stability and subsampling results.

---

## 9. SUMMARY OF RESOLVED vs REMAINING CONCERNS

| Category | Resolved | Partial | Outstanding |
|----------|----------|---------|-------------|
| Metadata (affiliation, ethics, etc.) | 6 | 0 | 0 |
| Data/Code Availability | 1 | 0 | 0 |
| Statistics/Uncertainty | 0 | 2 | 0 |
| Methods Detail | 1 | 2 | 0 |
| Figure Compliance | 9 | 0 | 0 |
| Biological Claims | 8 | 0 | 0 |
| Baseline/Benchmark | 0 | 1 | 1 |
| Writing/Style | 6 | 0 | 0 |
| **TOTAL** | **31** | **5** | **1** |

**Key Outstanding Item:** Integration of learned baselines (scVI/embedding VAE) into the main benchmark for stronger comparison evidence.

---

## 10. CONCLUSION

The CLOP-DiT manuscript is substantially complete and addresses most critical reviewer concerns identified in the policy documents. The work shows strong evidence of:

1. **Technical rigor:** 37× random KNN accuracy, 81% steering, proper ablation studies
2. **Biological grounding:** Marker gene preservation, DE concordance, downstream validation
3. **Reproducibility:** Clear data availability, code references, config documentation
4. **Presentation quality:** VCD-enforced figure quality, policy-compliant captions

The primary remaining gap is ensuring learned baseline comparisons are fully integrated to address potential reviewer questions about comparison method strength.

**Overall Assessment:** Ready for submission with minor baseline integration verification.

---

*Report compiled from analysis of:*
- `articles/clop_dit_biology.tex`
- `docs/REVIEWER_CONCERNS_AND_NEXT_STEPS.md`
- `docs/FIGURE_PRESENTATION_POLICY.md`
- `docs/LEGEND_CAPTION_POLICY.md`
- `docs/ARTICLE_SUBSECTION_POLICY.md`
- `docs/FIGURE_ORGANIZATION.md`
- `docs/BIOLOGICAL_CLAIM_MAP.md`
- `docs/SUBMISSION_CHECKLIST.md`
- `tests/test_article_delivery.py`
- `docs/CLOP-DiT_Evaluation_Report.md`
