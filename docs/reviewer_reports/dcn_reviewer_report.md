# Comprehensive Reviewer Report: CLOP-DiT Article

*Report compiled: 2026-03-07*
*Article: "CLOP-DiT: Text-Guided Single-Cell Gene Expression Generation via Contrastive Language-Omics Pretraining and Diffusion Transformers"*
*Target Journal: MDPI Biology*

---

## Executive Summary

This report provides a comprehensive review of the CLOP-DiT article covering (1) results validity and claims, (2) visualization and figure presentation policies, (3) figure organization and quality, and (4) writing quality and consistency. The article presents a two-stage pipeline for text-conditioned single-cell gene expression generation using contrastive learning (CLOP) and diffusion transformers (DiT).

**Overall Assessment**: The article presents scientifically interesting results with a well-structured evaluation framework. However, several critical issues must be addressed before submission, particularly around metadata completion, uncertainty reporting, and figure generation verification.

---

## PART 1: RESULTS VALIDITY AND CLAIMS ANALYSIS

### 1.1 Core Claims and Evidence Chain

| Claim | Evidence Presented | Strength | Concern |
|-------|-------------------|----------|---------|
| Text conditioning drives type-specific generation | KNN 36.9% vs 1.0% random (37x); steering 81% vs 50% random; unconditional collapse to 1.0% | Strong | Well-demonstrated through multiple controls |
| CLOP creates well-separated condition space | Pairwise cosine: raw 0.994 vs projected 0.222 | Strong | 130x amplification factor is compelling |
| DiT learns accurate velocity field | Val cosine 0.976 (angle ~12.6deg) | Strong | EMA model shows stable convergence |
| Generated cells preserve biological meaning | Marker genes (Fig 6), expression correlation (Fig 7-8), DE concordance (Fig 15) | Moderate-Strong | Gene-level validation is thorough |
| Downstream workflow integration | Clustering alignment, classifier transfer, DE analysis (Fig 14-15) | Moderate | ARI/NMI values not prominently reported |

### 1.2 Critical Results Concerns

**Concern 1.2.1: Uncertainty and Confidence Intervals**
- **Issue**: All metrics reported as point estimates without confidence intervals or standard errors
- **Impact**: Reviewers will question the stability of reported values (36.9% KNN, 81% steering, 0.93 diversity ratio)
- **Recommendation**: Add bootstrap 95% CIs for key metrics in supplementary materials; mention variability in main text

**Concern 1.2.2: Multiple Comparisons and Primary Endpoints**
- **Issue**: CFG sweep across 8 scales x 2 solvers x multiple step counts without multiple comparison correction
- **Impact**: Risk of false discovery in reported "optimal" configurations
- **Recommendation**: Explicitly designate primary operating points (CFG=2.0 Euler-10 for fidelity, CFG=1.0 Midpoint-10 for diversity) and label others as exploratory sensitivity analysis

**Concern 1.2.3: In-Distribution vs Out-of-Distribution Limitation**
- **Issue**: Claims limited to in-distribution conditions; OOD evaluation incomplete
- **Impact**: Generalization claims are constrained
- **Recommendation**: Add explicit statement in Limitations that OOD/free-form prompts were not quantitatively evaluated

**Concern 1.2.4: Sample Size Justification**
- **Issue**: No justification for 100 cell types, 200 cells per group, or 80/20 split
- **Impact**: Statistical power considerations unclear
- **Recommendation**: Briefly address in Methods or acknowledge as practical constraints

---

## PART 2: VISUALIZATION POLICIES ASSESSMENT

### 2.1 Policy Framework (Good)

The project has a comprehensive visualization policy framework:

| Policy Document | Purpose | Status |
|----------------|---------|--------|
| `FIGURE_PRESENTATION_POLICY.md` | Annotation budget, text density, color semantics | Well-defined |
| `LEGEND_CAPTION_POLICY.md` | Legend vs caption content separation | Well-defined |
| `FIGURE_STYLE_AUDIT.md` | Style consistency and shared constants | Recently updated |
| `FIGURE_ORGANIZATION.md` | Panel mapping to article figures | Comprehensive |

### 2.2 Policy Compliance Strengths

1. **Semantic Color System**: Shared `quality_color()` helper with standardized good/warn/bad thresholds
2. **Font Consistency**: Centralized font constants (`FONT_SUPTITLE`, `FONT_TITLE`, etc.) in `style.py`
3. **VCD Integration**: 30-pass Visual Conflict Detection system for layout validation
4. **Figure-Caption Separation**: Clear policy on what belongs in legends vs captions

### 2.3 Policy Concerns

**Concern 2.3.1: Legend Density in Dense Panels**
- Panels with >6-8 legend entries may still be at risk of crowding
- Policy recommends smaller font or "see caption" truncation, but implementation depends on panel-specific code

**Concern 2.3.2: Colorbar Placement**
- Policy requires `add_colorbar_safe()` with specific padding/shrink values
- Must verify all panels comply during figure generation

**Concern 2.3.3: Article vs Report Distinction**
- Article panels must have minimal annotation; report panels can carry more diagnostics
- Risk of report-style density leaking into article figures if not carefully reviewed

---

## PART 3: FIGURE PRESENTATIONS ANALYSIS

### 3.1 Figure Organization (15 Article Figures)

| Figure | Content | Producer | Status Concern |
|--------|---------|----------|----------------|
| Fig 1 | Architecture diagram | `generate_architecture_figure.py` | Vector PDF required |
| Fig 2 | Training dynamics (merged A+C) | `training_panels.py` | Merged layout |
| Fig 3 | Embedding space (merged B+E) | `panels_quality.py` | Merged layout |
| Fig 4 | Metrics dashboard (D) | `panels_metrics.py` | Standalone |
| Fig 5 | Fidelity & alignment (merged G+F) | `panels_quality.py` | PIL raster stack - consider vector refactor |
| Fig 6 | Marker genes (N) | `panels_expression.py` | Standalone |
| Fig 7 | Expression correlation (H) | `panels_expression.py` | Standalone |
| Fig 8 | Expression analysis (I) | `panels_expression.py` | Standalone |
| Fig 9 | Conditioning UMAP (M) | `conditioning_analysis.py` | Standalone |
| Fig 10 | Diversity diagnostics (J) | `diversity_diagnostics.py` | Standalone |
| Fig 11 | Diversity tradeoff (merged L+K) | `results_visualizer.py` | Mixed raster/vector - needs verification |
| Fig 12 | Baselines (O) | `baseline_panels.py` | Standalone |
| Fig 13 | Benchmark (S) | `benchmark_panels.py` | Heatmap contrast fixed |
| Fig 14 | Downstream P+Q (merged) | `downstream_panels.py` | GridSpec vector |
| Fig 15 | DE concordance (R) | `downstream_panels.py` | Standalone |

### 3.2 Figure Generation Pipeline

**Critical Finding**: The figure directories (`results/figures/` and `articles/figures/`) do not exist.

**Implications**:
1. Figures have not been generated in the current worktree
2. Cannot verify visual quality, VCD compliance, or layout correctness
3. Article cannot be compiled with `\includegraphics` commands

**Required Action**:
```bash
bash scripts/regenerate_report.sh
bash scripts/pipeline/verify_article_figures.sh
cd articles && latexmk -pdf clop_dit_biology.tex
```

### 3.3 Figure-Specific Concerns

**Concern 3.3.1: Merged Figures (Figs 2, 3, 5, 11, 14)**
- Fig 5 uses PIL raster stack (G and F rendered separately, then stacked)
- Fig 11 uses mixed approach (raster top row from PNG, vector violin bottom row)
- Recommendation: Refactor to single GridSpec for full vector output where possible

**Concern 3.3.2: Panel S Heatmap Text**
- Fixed to use explicit dark color (`#1a1a1a`) for WCAG contrast
- Must verify 0 VCD warnings after regeneration

**Concern 3.3.3: Legend Statistics**
- Policy prohibits statistical summaries in legend titles (no "r =", "p =", "Sign =")
- Must verify compliance in all expression and DE panels

---

## PART 4: ARTICLE WRITING ANALYSIS

### 4.1 Writing Strengths

1. **Clear Structure**: Well-organized with signposting paragraphs in Methods and Results
2. **Consistent Terminology**: CLOP-DiT, scGPT, flow matching, BiomedBERT, AdaLN-Zero, CFG used consistently
3. **Evidence Chain**: Logical flow from training dynamics → generation quality → downstream validation
4. **Subsection Policy**: Opening sentences state purpose; figures referenced explicitly

### 4.2 Critical Writing Issues

**Issue 4.2.1: Template Placeholders (HIGH PRIORITY)**
The `.tex` file contains resolved metadata (based on current reading), but checklist indicates need for verification:
- [x] Affiliation: Present (Army Medical University, Chongqing)
- [x] Correspondence: Present (zeyu.fu@tmmu.edu.cn)
- [x] Funding: Present ("This research received no external funding")
- [x] Ethics: Present (IRB waiver for public GEO data)
- [x] Author Contributions: Present (CRediT-style)
- [x] Acknowledgments: Present
- [x] Conflicts: Present ("no conflicts of interest")

**Status**: All placeholders appear resolved in current version.

**Issue 4.2.2: Data Availability Statement**
- Present: States GEO data source, mentions supplementary tables S1 and S2
- Mentions code availability "upon reasonable request"
- **Verification needed**: Ensure all three elements are explicit:
  1. GEO accession list (Supplementary Table S1)
  2. Validation dataset IDs (Supplementary Table S2)
  3. Code repository or "available upon request" with justification

**Issue 4.2.3: Statistics and Uncertainty Reporting**
- **Current**: "Bootstrap 95\% confidence intervals for the composite benchmark scores are shown in Figure~\ref{fig:benchmark}d"
- **Missing**: CIs for individual metrics (KNN, steering, diversity ratio)
- **Recommendation**: Add statement that single-measurement metrics are characterized by CFG sweep in Appendix B

**Issue 4.2.4: Primary vs Exploratory Designations**
- **Current**: "Two configurations are designated as \emph{primary operating points}: CFG\,=\,2.0 with 10-step Euler integration (high-fidelity regime) and CFG\,=\,1.0 with 10-step Midpoint integration (high-diversity regime)"
- **Good**: Clear designation of primary configurations
- **Note**: "no multiple-comparison correction was applied to these exploratory comparisons" - appropriate caveat

### 4.3 Writing Recommendations

1. **Abstract Enhancement**: Consider adding one phrase about downstream numbers (ARI, NMI, DE r) to make biological utility claim more quantitative

2. **Conclusions Enhancement**: Add specific downstream numbers to take-home summary (e.g., "DE concordance Pearson r ≈ 0.39, sign agreement ≈ 0.94")

3. **Methods Detail**: Text-condition construction paragraph is present but could be more specific about exact template used

4. **Software Versions**: Present (Python 3.10, PyTorch 2.1, scGPT v0.2.1, Transformers 4.36) - good

---

## PART 5: COMPREHENSIVE REVIEWER CONCERNS SUMMARY

### 5.1 High Priority (Must Address Before Submission)

| # | Concern | Action Required |
|---|---------|-----------------|
| 1 | **Figures Not Generated** | Run `regenerate_report.sh` and `scripts/pipeline/verify_article_figures.sh` |
| 2 | **Uncertainty Reporting** | Add bootstrap CIs for key metrics in supplement |
| 3 | **Validation Split Clarity** | Confirm explicit statement on how 8 validation datasets were chosen |
| 4 | **OOD Limitation** | Add sentence in Limitations about OOD evaluation scope |

### 5.2 Medium Priority (Reviewer Expectations)

| # | Concern | Action Required |
|---|---------|-----------------|
| 5 | **VCD Verification** | Confirm 0 warnings after figure regeneration |
| 6 | **Legend Compliance** | Verify no statistical summaries in legend titles |
| 7 | **Merged Figure Quality** | Consider vector refactor for Figs 5 and 11 |
| 8 | **Supplement List** | Explicitly enumerate S1, S2, S3 in Data Availability |

### 5.3 Low Priority (Polish)

| # | Concern | Action Required |
|---|---------|-----------------|
| 9 | **Abstract Enhancement** | Optional: add downstream metric summary |
| 10 | **Conclusions Enhancement** | Optional: add quantitative DE concordance numbers |
| 11 | **Reference Pass** | Verify all citations match intended references |

---

## PART 6: SUBMISSION READINESS CHECKLIST

Based on `SUBMISSION_CHECKLIST.md` and this review:

### High Priority Items
- [x] Data Availability statement present (verify completeness)
- [x] REPRODUCIBILITY.md exists at project root
- [x] Affiliation, correspondence, funding placeholders resolved
- [x] Ethics statements present
- [x] Author Contributions filled
- [x] Acknowledgments and Conflicts set
- [ ] **Figures regenerated and verified** (CRITICAL - directories currently missing)

### Medium Priority Items
- [ ] Uncertainty/CI paragraph or table added
- [x] Primary vs exploratory comparisons clarified
- [x] Text-condition construction paragraph present
- [x] Software versions listed
- [ ] Validation split clarification verified
- [x] OOD limitation statement present

### Pre-Submission Verification
- [ ] Run `bash scripts/pipeline/verify_article_figures.sh` - all 15 PDFs present
- [ ] Run VCD - confirm 0 warnings
- [ ] Article builds: `cd articles && latexmk -pdf clop_dit_biology.tex`
- [ ] No TODO or "Please add" placeholders remain

---

## CONCLUSIONS AND RECOMMENDATIONS

### Summary of Findings

1. **Scientific Content**: The article presents a compelling methodology with strong evidence chain (37x random accuracy, 81% steering, comprehensive biological validation)

2. **Policy Framework**: Excellent visualization policies in place with VCD integration, semantic color systems, and clear legend/caption separation

3. **Figure Status**: **CRITICAL ISSUE** - Figure directories do not exist; regeneration required before any submission

4. **Writing Quality**: Generally strong with good structure and terminology consistency; minor enhancements possible for uncertainty reporting and downstream numbers

### Priority Actions for Submission

1. **IMMEDIATE**: Generate all figures using `regenerate_report.sh` and verify with `scripts/pipeline/verify_article_figures.sh`
2. **HIGH**: Verify VCD compliance (0 warnings target)
3. **HIGH**: Confirm all 15 article figure PDFs exist and are properly symlinked
4. **MEDIUM**: Add bootstrap CIs to supplementary materials for key metrics
5. **MEDIUM**: Verify legend compliance with no statistics in titles

### Estimated Reviewer Reception

With the high-priority items addressed, this article should receive a positive reception from reviewers. The methodology is novel (first text-conditioned single-cell generation), the evaluation is thorough (4 axes, 15 figures, downstream validation), and the biological claims are well-supported by gene-level and workflow-level evidence.

The primary risks for reviewer concerns are:
1. **Missing figures** (if not regenerated)
2. **Uncertainty quantification** (if CIs not provided)
3. **Generalization claims** (if OOD limitation not stated)

All of these are addressable with the documented action items above.

---

*Report compiled by comprehensive review of:*
- `articles/clop_dit_biology.tex` (full article source)
- `docs/FIGURE_PRESENTATION_POLICY.md`
- `docs/LEGEND_CAPTION_POLICY.md`
- `docs/FIGURE_STYLE_AUDIT.md`
- `docs/FIGURE_ORGANIZATION.md`
- `docs/WRITING_AND_DOCS_POLICY.md`
- `docs/ARTICLE_SUBSECTION_POLICY.md`
- `docs/REVIEWER_CONCERNS_AND_NEXT_STEPS.md`
- `docs/SUBMISSION_CHECKLIST.md`
- `docs/BIOLOGICAL_CLAIM_MAP.md`
- `docs/CLOP-DiT_Evaluation_Report.md`

*Reviewer Concerns Report - CLOP-DiT Article - 2026-03-07*
