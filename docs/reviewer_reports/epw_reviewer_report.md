# Comprehensive Reviewer Concerns Report

## CLOP-DiT: Text-Guided Single-Cell Gene Expression Generation

**Report Date:** 2026-03-07  
**Article:** `articles/clop_dit_biology.tex` (MDPI Biology format)  
**Status:** Pre-submission review

---

## Executive Summary

This report consolidates potential reviewer concerns across multiple dimensions: reproducibility and data availability, statistical rigor, methods detail, figure presentation, article writing quality, biological validation, and baseline comparisons. Each concern is categorized by severity (Critical/High/Medium/Low) with specific evidence from the codebase and recommended corrective actions.

---

## 1. REPRODUCIBILITY AND DATA AVAILABILITY (Critical Priority)

### Concern 1.1: Incomplete Data Availability Statement
**Severity:** Critical  
**MDPI Requirement:** Biology requires "full experimental details must be provided so that the results can be reproduced" and "make full datasets available where possible."

**Current Status:**
- The manuscript has a Data Availability section (lines 474-475 in `.tex`) but lacks specific GEO accession tables
- No supplementary table listing all 80 GEO datasets (Supplementary Table S1 referenced but may not exist)
- No supplementary table listing the 8 held-out validation datasets (Supplementary Table S2 referenced)
- No clear code repository URL or license information

**Evidence:**
```latex
% From clop_dit_biology.tex lines 474-475
\dataavailability{The training and validation data were curated from publicly available 
GEO datasets; a full list of GEO accession identifiers is provided in Supplementary Table~S1...}
```

**Reviewer Impact:** Editorial return likely without concrete data availability documentation.

**Recommended Actions:**
1. Create Supplementary Table S1 with all 80 GEO accession identifiers
2. Create Supplementary Table S2 with the 8 validation dataset identifiers
3. Add explicit GitHub/code repository URL or state "available upon request" with justification
4. Reference `docs/QUICK_START.md` and `scripts/regenerate_report.sh` in the data availability statement

---

### Concern 1.2: Missing Reproducibility Documentation
**Severity:** High  
**MDPI Requirement:** Reproducibility instructions must enable independent verification.

**Current Status:**
- `docs/QUICK_START.md` exists but may not be referenced in the manuscript
- `scripts/regenerate_report.sh` provides one-command reproduction
- No explicit REPRODUCIBILITY.md supplement referenced in the article

**Evidence:**
- File `docs/QUICK_START.md` exists with reproduction instructions
- File `scripts/regenerate_report.sh` handles figure regeneration

**Recommended Actions:**
1. Ensure `docs/QUICK_START.md` is complete and accurate
2. Add a supplement or appendix referencing the reproduction pipeline
3. Include expected output hashes or metrics for verification

---

## 2. AUTHOR AND SUBMISSION METADATA (High Priority)

### Concern 2.1: Template Placeholders in Front Matter
**Severity:** High  
**Issue:** Some front matter fields in the LaTeX may still contain placeholders (though main ones appear filled based on the article read).

**Verified Status (from article read):**
- ✅ Affiliation: Filled (State Key Laboratory, Army Medical University)
- ✅ Correspondence: Filled (zeyu.fu@tmmu.edu.cn)
- ✅ Funding: Filled ("This research received no external funding")
- ✅ Author Contributions: Filled (CRediT-style)
- ✅ Acknowledgments: Filled
- ✅ Conflicts of Interest: Filled
- ✅ Ethics Statement: Filled (IRB waiver for public GEO data)

**Assessment:** This concern appears **RESOLVED** based on the current article state.

---

## 3. STATISTICS AND UNCERTAINTY REPORTING (High Priority)

### Concern 3.1: No Confidence Intervals or Uncertainty Measures
**Severity:** High  
**Issue:** Single point estimates without uncertainty quantification.

**Current Status:**
- Core metrics (KNN 36.9%, steering 81%, diversity ratio 0.93, composite score 0.844) lack confidence intervals
- No standard errors reported for any metric
- Methods section (lines 247) mentions: "All metrics are reported as point estimates over the fixed evaluation set..."

**Evidence:**
```latex
% From clop_dit_biology.tex line 247
All metrics are reported as point estimates over the fixed evaluation set described above. 
...no multiple-comparison correction was applied to these exploratory comparisons.
```

**Reviewer Impact:** Reviewers will question whether results are statistically stable.

**Recommended Actions:**
1. Add bootstrap 95% confidence intervals for key metrics (KNN, steering, diversity)
2. Document bootstrap procedure in Methods (n=1000 resamples, stratified by cell type)
3. Include CI table in Supplementary Table S3
4. Add small-multiple CI visualization in benchmark figure (Fig 13)

---

### Concern 3.2: Multiple Comparisons Not Addressed
**Severity:** Medium  
**Issue:** CFG sweep across 8 scales × 2 solvers = 16 configurations without correction.

**Current Status:**
- 8 CFG scales tested: {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}
- 2 ODE solvers (Euler, Midpoint)
- Multiple step counts (10, 20, 50)
- Methods section acknowledges: "no multiple-comparison correction was applied"

**Recommended Actions:**
1. Designate primary vs. exploratory comparisons explicitly
2. State which configuration is primary (CFG=2.0 Euler-10 for accuracy, CFG=1.0 Midpoint for diversity)
3. Add a sentence: "These comparisons are exploratory; no multiple-comparison correction was applied"

---

### Concern 3.3: Sample Size and Power Justification Missing
**Severity:** Medium  
**Issue:** No justification for 100 cell types, 200 cells per group, or 80/20 split.

**Current Metrics:**
- 100 cell types evaluated
- 200 cells generated per type
- 80/20 train-validation split
- 220,304 total cells

**Recommended Actions:**
1. Add brief justification in Dataset subsection
2. Cite precedent from scVI/scGen papers for similar splits
3. Note that 100 cell types provides good coverage of the cell ontology

---

## 4. METHODS DETAIL AND SOFTWARE VERSIONS (Medium Priority)

### Concern 4.1: Text Condition Construction Insufficiently Detailed
**Severity:** Medium  
**Issue:** How 1,088 text groups were constructed needs more detail.

**Current Status (from article lines 193-194):**
```latex
Text descriptions were generated for each cell type using a structured template: 
\texttt{``\{cell type\}, tissue: \{tissue\}, organism: \{organism\}, markers: 
\{top 5 DE marker genes\}, context: \{disease/condition\}''}, where marker genes 
were identified per cell type by one-vs.-rest Wilcoxon rank-sum test on the training set.
```

**Recommended Actions:**
1. Add the exact script path: `scripts/build_text_captions.py`
2. Provide one concrete example of a full text description in the supplement
3. Document marker gene selection threshold (adjusted p-value, logFC cutoff)

---

### Concern 4.2: Software Versions Need Explicit Listing
**Severity:** Medium  
**Issue:** Key library versions not explicitly stated in Methods.

**Current Status (line 197):**
```latex
Training and evaluation were implemented in Python~3.10 using PyTorch~2.1 (CUDA~12), 
scGPT~v0.2.1, and Hugging Face Transformers~4.36 for BiomedBERT
```

**Assessment:** Adequate but could be enhanced with a formal software table.

**Recommended Actions:**
1. Consider adding a software table in Appendix or supplement
2. Reference `requirements.txt` and `configs/clop_v9.3.yaml`
3. Note BiomedBERT-large commit hash for reproducibility

---

## 5. LIMITATIONS AND GENERALIZATION (Medium Priority)

### Concern 5.1: In-Distribution Only Claims Need Clarification
**Severity:** Medium  
**Issue:** The Discussion mentions limitations but could be more explicit about OOD scope.

**Current Status (lines 428-429):**
```latex
CLOP-DiT currently generates most reliably for cell states represented in training; 
novel prompts far from the training distribution remain sensitive to CLOP projector 
extrapolation. A pilot OOD/free-form prompt run was executed...but robust quantitative 
OOD scoring is still incomplete...
```

**Assessment:** Well-addressed in the Discussion.

**Recommended Actions:**
1. Add one sentence in Abstract: "Metrics apply to in-distribution conditions only"
2. Ensure the limitation is prominent in Discussion opening

---

### Concern 5.2: Held-Out Validation Split Needs Specification
**Severity:** Medium  
**Issue:** Which 8 datasets were held out and how they were chosen.

**Current Status (line 195):**
```latex
The 8 validation datasets were selected by study (not by cell) to ensure no sample 
overlap between training and validation splits; the full list of GEO accession 
identifiers for all 80 datasets is provided in Supplementary Table~S1, and the 
validation study identifiers are in Supplementary Table~S2.
```

**Recommended Actions:**
1. Ensure Supplementary Table S2 is created and referenced correctly
2. Add a sentence on selection criteria (random vs. stratified by tissue/disease)

---

## 6. FIGURE PRESENTATION AND ORGANIZATION (Medium Priority)

### Concern 6.1: Figure Legends Contain Statistics (Policy Violation)
**Severity:** Medium  
**Issue:** LEGEND_CAPTION_POLICY.md prohibits statistical summaries in legend titles.

**Policy Reference:**
```markdown
# From LEGEND_CAPTION_POLICY.md
- Do not put statistical summaries in legend titles or entries: values such as 
Pearson r, p-values, sign agreement, CV correlation, or confidence intervals belong 
in the figure caption or in a small, non-dominant in-figure annotation
```

**Evidence Needed:**
- Review all 15 article figure producers for legend titles containing statistics
- Use VCD and contract tests to enforce this

**Recommended Actions:**
1. Run `scripts/visual_conflict_detector.py` on all figures
2. Move any "r =", "p =", "Sign =" from legend titles to captions
3. Add contract test in `tests/test_article_delivery.py`

---

### Concern 6.2: Annotation Budget and Text Density
**Severity:** Medium  
**Issue:** FIGURE_PRESENTATION_POLICY.md sets strict limits on annotation.

**Policy Requirements:**
- At most one stats box per subplot
- At most two highlighted outliers per subplot
- No more than three text elements beyond axes labels and ticks
- Article-facing panels must be print-optimized

**Recommended Actions:**
1. Audit all figures against the annotation budget
2. Verify `style.add_stat_box()` is used at most once per subplot
3. Ensure outliers are capped at 2 per subplot

---

### Concern 6.3: Colorbar Placement and Overlap
**Severity:** Low  
**Issue:** Colorbars must not overlap main content per FIGURE_PRESENTATION_POLICY.md.

**Policy Requirement:**
```markdown
- **Colorbars must not overlap main figure content.** Use `style.add_colorbar_safe()` 
for all article and report colorbars
```

**Recommended Actions:**
1. Verify all colorbars use `style.add_colorbar_safe()`
2. Check VCD reports for colorbar overlap warnings

---

## 7. ARTICLE WRITING AND STRUCTURE (Medium Priority)

### Concern 7.1: Abstract Density and Claims
**Severity:** Medium  
**Issue:** Abstract is dense; downstream metrics could be more explicit.

**Current Abstract (lines 107):**
```latex
\abstract{Generating realistic synthetic single-cell gene expression profiles... 
CLOP-DiT achieves $k$-nearest-neighbor accuracy of 36.9\% (37$\times$ random chance 
with 100 classes), steering accuracy of 81.0\% (random = 50\%), and a diversity ratio 
of 0.93...}
```

**Assessment:** Strong abstract but could mention unconditional collapse explicitly.

**Recommended Actions:**
1. Consider adding: "Unconditional generation collapses to random chance, confirming 
conditioning drives specificity"
2. Optionally add one downstream metric (e.g., "DE concordance r = 0.17")

---

### Concern 7.2: Section Transitions and Signposting
**Severity:** Low  
**Issue:** ARTICLE_SUBSECTION_POLICY.md requires specific transitions.

**Current Status:**
- Methods has organization paragraph (lines 178-183) ✅
- Results has figure-family organization (lines 255-261) ✅
- Transitions at section boundaries appear present ✅

**Assessment:** Generally well-structured per policy.

---

### Concern 7.3: Terminology Consistency
**Severity:** Low  
**Issue:** WRITING_AND_DOCS_POLICY.md defines canonical terminology.

**Canonical Terms:**
- "CLOP-DiT" (not "CLOP-DIT" or "CLOP DiT")
- "scGPT" (not "ScGPT" or "sc-gpt")
- "flow matching" (lowercase)
- "BiomedBERT" (capitalization)
- "AdaLN-Zero"
- "classifier-free guidance (CFG)"

**Recommended Actions:**
1. Run spell-check for terminology consistency
2. Verify no "CLOP-DIT" or "CLOP DiT" variations exist

---

## 8. BIOLOGICAL VALIDATION AND CLAIMS (High Priority)

### Concern 8.1: Downstream Metrics Inconsistency
**Severity:** High  
**Issue:** Different sections report conflicting downstream numbers.

**Evidence from Article:**
- Line 433: "classifier transfer reaches 30.8% accuracy (macro F1 0.247)"
- Line 286: "linear classifier accuracy of 51.1%"
- Table 1: "LinAcc = 0.511 for CFG=2.0 Euler-10"

**Discrepancy:** 51.1% vs 30.8% for classifier transfer

**Explanation:** Likely different metrics:
- 51.1% = Logistic Regression on generated cells (CFG=2.0)
- 30.8% = Transfer accuracy in downstream validation (different setting)

**Recommended Actions:**
1. Clarify which metric is which in the text
2. Ensure Conclusions references the correct metric consistently
3. Consider harmonizing to avoid confusion

---

### Concern 8.2: DE Concordance Values Vary
**Severity:** Medium  
**Issue:** DE concordance numbers differ across sections.

**Evidence:**
- Abstract: "DE concordance (logFC Pearson r = 0.17)"
- Conclusions (line 433): "DE concordance is moderate but directionally consistent 
(mean logFC Pearson r ≈ 0.39, mean sign agreement ≈ 0.94)"

**Discrepancy:** r = 0.17 vs r ≈ 0.39

**Explanation:** Likely different contrasts or aggregation methods.

**Recommended Actions:**
1. Clarify which contrast yields r = 0.17 vs r = 0.39
2. Document all three contrasts explicitly (CD8 vs CD4, TAM vs monocyte, epithelial vs fibroblast)
3. Report mean ± std across contrasts

---

## 9. BASELINE COMPARISONS (Medium Priority)

### Concern 9.1: Learned Baselines Not Fully Executed
**Severity:** Medium  
**Issue:** REVIEWER_CONCERNS notes learned baselines need to be run.

**Current Status (from REVIEWER_CONCERNS_AND_NEXT_STEPS.md):**
```markdown
**Next steps:**
19. **Run at least one learned baseline** and write its artifacts into 
`results/baselines/{method}/`...
20. **Use `scripts/run_downstream_for_baselines.py`** to generate per-method 
downstream biological validation...
```

**Evidence:**
- Scripts exist: `scripts/baselines/train_scvi_baseline.py`, `train_embedding_vae_baseline.py`
- Registry exists: `src/evaluation/baseline_registry.py`
- Status unclear if fully executed

**Recommended Actions:**
1. Execute at least scVI baseline for comparison
2. Include baseline results in Fig 12 and Fig 13
3. Document baseline limitations in Discussion

---

### Concern 9.2: Fréchet Distance Interpretation
**Severity:** Low  
**Issue:** FD is misleading for conditional generation.

**Current Handling:**
- Discussion (line 422-423) acknowledges FD limitation
- Table 1 includes FD with caveat in caption

**Assessment:** Well-handled but worth monitoring.

---

## 10. SUPPLEMENTARY MATERIALS (Medium Priority)

### Concern 10.1: Supplementary Content Incomplete
**Severity:** Medium  
**Issue:** Referenced supplementary tables may not exist.

**Referenced in Article:**
- Supplementary Table S1: GEO accession list (mentioned line 440)
- Supplementary Table S2: Validation dataset identifiers
- Supplementary Table S3: Full CFG sweep results (mentioned line 440)
- Supplementary Figure S1: per-type KNN accuracy

**Recommended Actions:**
1. Create all referenced supplementary materials
2. Verify the supplementary materials are mentioned in the Data Availability section
3. Consider adding a "Supplementary Materials" subsection listing all S1-S3 tables

---

## 11. VISUAL CONFLICT DETECTION (Low Priority)

### Concern 11.1: VCD Warnings May Exist
**Severity:** Low  
**Issue:** Visual Conflict Detection (VCD) should have 0 warnings for submission.

**Policy Target (from FIGURE_ORGANIZATION.md):**
```markdown
### Target: 0 warnings for all panels
After the layout restructuring and Panel S heatmap contrast fix...run full regeneration 
in the intended environment to confirm 0 VCD warnings.
```

**Recommended Actions:**
1. Run full figure regeneration: `bash scripts/regenerate_report.sh`
2. Check VCD output for warnings (not info-level)
3. Address any warnings before submission

---

## 12. CODE AND REPOSITORY ISSUES (Medium Priority)

### Concern 12.1: Code Availability Statement Clarity
**Severity:** Medium  
**Issue:** Code availability needs clear statement.

**Current Status (line 474):**
```latex
Model checkpoints, generated embeddings, and the figure-reproduction script...are 
available from the corresponding author upon reasonable request. All evaluation 
code is available in the project repository (available upon request pending 
institutional approval for open-source release).
```

**Recommended Actions:**
1. Ensure institutional approval is obtained or timeline stated
2. Consider temporary GitHub private repo with reviewer access
3. Provide tarball of evaluation code as supplementary material if needed

---

## SUMMARY: PRIORITY ACTION MATRIX

| Priority | Concern | Action Required | Time Estimate |
|----------|---------|-----------------|---------------|
| **Critical** | 1.1 Data Availability | Create S1, S2 tables; verify exist | 2-4 hours |
| **High** | 2.1 Metadata placeholders | Verify all filled (appears OK) | 30 min |
| **High** | 3.1 Confidence Intervals | Bootstrap CIs for key metrics | 4-6 hours |
| **Medium** | 3.2 Multiple comparisons | Clarify primary vs exploratory | 1 hour |
| **Medium** | 4.1 Text construction | Add script path, example | 1 hour |
| **Medium** | 6.1 Legend statistics | Audit figures, fix violations | 2-3 hours |
| **Medium** | 8.1 Classifier metric inconsistency | Harmonize or clarify | 1 hour |
| **Medium** | 8.2 DE concordance inconsistency | Clarify contrasts | 1 hour |
| **Medium** | 9.1 Learned baselines | Run scVI baseline | 4-8 hours |
| **Medium** | 10.1 Supplementary materials | Create S1-S3 | 3-4 hours |
| **Low** | 6.2 Annotation budget | Audit figures | 2 hours |
| **Low** | 11.1 VCD warnings | Regenerate, check | 2 hours |

---

## POSITIVE ASPECTS (Reviewer Strengths)

The article has several strengths that should be highlighted:

1. **Comprehensive evaluation framework:** Four evaluation axes (KNN, steering, diversity, downstream) provide multi-faceted validation
2. **Strong ablation evidence:** Unconditional generation collapse to random chance (1.0%) is convincing evidence for conditioning
3. **Transparent limitation discussion:** The Discussion acknowledges the in-distribution limitation and accuracy gap explicitly
4. **Reproducibility pipeline:** `regenerate_report.sh` provides one-command figure regeneration
5. **Well-organized figure structure:** Figure families map logically to evidence types
6. **Detailed methods:** Architecture specifications in Appendix with parameter counts

---

## CONCLUSION

This article presents a novel and technically sound method for text-conditioned single-cell generation. The key concerns center on:

1. **Completeness of supplementary materials** (S1, S2 tables)
2. **Statistical rigor** (confidence intervals, multiple comparisons)
3. **Consistency of reported metrics** (classifier accuracy, DE concordance)
4. **Baseline comparison completeness** (learned baseline execution)

Addressing the Critical and High priority items will significantly strengthen the submission and preempt common reviewer objections. The Medium and Low priority items are polish that would further improve the manuscript.

**Estimated time to address all concerns:** 20-30 hours  
**Recommended submission timeline:** After completing Critical, High, and key Medium items.

---

## APPENDIX: REFERENCED DOCUMENTATION

- `docs/REVIEWER_CONCERNS_AND_NEXT_STEPS.md` - Source for many concerns
- `docs/FIGURE_PRESENTATION_POLICY.md` - Figure layout rules
- `docs/LEGEND_CAPTION_POLICY.md` - Legend vs caption rules
- `docs/ARTICLE_SUBSECTION_POLICY.md` - Writing structure
- `docs/WRITING_AND_DOCS_POLICY.md` - Terminology and source of truth
- `docs/FIGURE_ORGANIZATION.md` - Figure mapping and producers
- `docs/BIOLOGICAL_CLAIM_MAP.md` - Evidence chain documentation
- `docs/SUBMISSION_CHECKLIST.md` - Pre-submission tasks

---

*Report generated from comprehensive review of CLOP-DiT article and supporting documentation.*
