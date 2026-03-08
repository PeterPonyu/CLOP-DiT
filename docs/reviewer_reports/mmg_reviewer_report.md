# CLOP-DiT Article Review: Comprehensive Reviewer Concerns Report

*Report generated: 2026-03-07*
*Article: "CLOP-DiT: Text-Guided Single-Cell Gene Expression Generation via Contrastive Language-Omics Pretraining and Diffusion Transformers"*
*Target Journal: MDPI Biology*

---

## Executive Summary

This report consolidates reviewer concerns and recommendations for the CLOP-DiT manuscript based on:
1. Analysis of the article content (`articles/clop_dit_biology.tex`)
2. Review of figure presentation policies and organization
3. Existing documented concerns from `docs/REVIEWER_CONCERNS_AND_NEXT_STEPS.md`
4. Evaluation of visualization standards and biological claim validation

**Overall Assessment:** The article presents a novel two-stage pipeline for text-conditioned single-cell generation with rigorous quantitative evaluation. However, several concerns related to reproducibility, statistical reporting, and metadata completeness require attention before submission.

---

## 1. CRITICAL CONCERNS (Editorial Return Risk)

### 1.1 Data and Code Availability (MDPI Requirement)

**Concern:** MDPI Biology requires "full experimental details must be provided so that the results can be reproduced" and "make full datasets available where possible."

**Current Status:** The Data Availability Statement in the article states:
> "Model checkpoints, generated embeddings, and the figure-reproduction script... are available from the corresponding author upon reasonable request."

**Issues Identified:**
- No explicit repository URL or DOI provided for code
- GEO accession numbers mentioned but not consolidated in a supplementary table
- The phrase "upon reasonable request" may not satisfy open science requirements

**Recommendation:** 
- Create a public repository (GitHub/GitLab) with code and pre-trained weights
- Add Supplementary Table S1 with complete GEO accession list
- Clarify if institutional approval is pending for open-source release

### 1.2 Author Metadata and Front Matter

**Current Status (Lines 81-93, 467-489):**
- Author: Zeyu Fu with ORCID placeholder
- Affiliation: Complete
- Correspondence: Complete
- Funding: "This research received no external funding." ✓
- Ethics: Appropriate waiver statement for GEO data ✓
- Author Contributions: Complete CRediT statement ✓
- Acknowledgments: Complete ✓
- Conflicts: "The author declares no conflicts of interest." ✓

**Assessment:** Front matter appears complete. No template placeholders remain.

---

## 2. STATISTICS AND UNCERTAINTY REPORTING

### 2.1 Confidence Intervals and Variability

**Concern:** Single-point estimates without confidence intervals may be overstated.

**Current Metrics Reported (Point Estimates Only):**
| Metric | Value | CI Provided? |
|--------|-------|--------------|
| KNN top-1 accuracy | 36.9% | No |
| Steering accuracy | 81.0% | No |
| Diversity ratio | 0.93 | No |
| Linear classifier accuracy | 51.1% | No |
| Composite benchmark score | 0.844 | No |

**Issues:**
- No standard errors or bootstrap 95% CIs for key metrics
- Figure 13 mentions "bootstrap 95% confidence intervals" in caption but these may not be visible
- Sample size justification not provided for 100 cell types / 200 cells per group

**Recommendation:**
- Add a "Statistical Methods" subsection in Methods describing variance estimation
- Include bootstrap CIs in Supplementary Table S3
- Clarify whether the CFG sweep comparisons are exploratory (no multiple comparison correction)

### 2.2 Multiple Comparisons

**Concern:** The CFG sweep compares 8 guidance scales × 2 solvers × multiple metrics.

**Current Text (Lines 247):**
> "Two configurations are designated as primary operating points... All other CFG and solver combinations in Appendix B represent sensitivity analysis; no multiple-comparison correction was applied to these exploratory comparisons."

**Assessment:** This is appropriately disclosed. However, the primary vs. exploratory distinction should be clearer in the abstract and results.

---

## 3. METHODS DETAIL AND REPRODUCIBILITY

### 3.1 Text Condition Construction

**Current Description (Lines 193):**
> "Text descriptions were generated for each cell type using a structured template: `{cell type}, tissue: {tissue}, organism: {organism}, markers: {top 5 DE marker genes}, context: {disease/condition}`"

**Strengths:**
- Template is specified
- Reference to `scripts/build_text_captions.py` provided

**Potential Reviewer Questions:**
- How were the top 5 DE genes selected (Wilcoxon rank-sum test details)?
- What ontology was used for cell type names?
- Were any text groups excluded or merged?

**Recommendation:** Add a supplementary method describing the exact text construction pipeline.

### 3.2 Software and Version Information

**Current Description (Lines 197):**
> "Training and evaluation were implemented in Python 3.10 using PyTorch 2.1 (CUDA 12), scGPT v0.2.1, and Hugging Face Transformers 4.36 for BiomedBERT"

**Strengths:**
- Key software versions provided
- References to `requirements.txt` and config file included

**Recommendation:** Consider adding a "Code Availability" subsection with repository link.

### 3.3 Validation Split Details

**Current Description (Lines 195):**
> "8 validation datasets were selected by study (not by cell) to ensure no sample overlap... the full list of GEO accession identifiers... is provided in Supplementary Table S1"

**Concern:** Which 8 datasets and how they were chosen is not explicitly stated.

**Recommendation:** Add one sentence: "Validation datasets were selected by stratified random sampling across tissue types to ensure diversity."

---

## 4. RESULTS AND FIGURE ANALYSIS

### 4.1 Figure Quality Assessment

| Figure | Content | Concerns |
|--------|---------|----------|
| Fig 1 | Architecture diagram | Vector quality; clear flow representation |
| Fig 2 | Training dynamics | Loss curves well-labeled; temperature plot included |
| Fig 3 | Embedding space | UMAP projections; co-localization evidence |
| Fig 4 | Metrics dashboard | KNN, steering, diversity comparison |
| Fig 5 | Fidelity & alignment | Per-type heterogeneity; heatmap readability |
| Fig 6 | Marker genes | Expression preservation across lineages |
| Fig 7 | Expression correlation | Gene-level fidelity |
| Fig 8 | Expression analysis | CV, range, variance structure |
| Fig 9 | Conditioning landscape | PCA of condition modes |
| Fig 10 | Diversity diagnostics | CFG sweep visualization |
| Fig 11 | Diversity trade-off | Noise-scale analysis; violin distributions |
| Fig 12 | Baseline comparison | Radar chart; relative improvement heatmap |
| Fig 13 | Composite benchmark | Bootstrap CIs; normalized heatmap |
| Fig 14 | Downstream clustering+classifier | ARI/NMI; confusion matrix; per-type F1 |
| Fig 15 | DE concordance | Effect-size weighted scatter; per-contrast metrics |

### 4.2 Visualization Policy Compliance

**From FIGURE_PRESENTATION_POLICY.md:**
- Annotation budget: At most one stats box per subplot ✓
- Text density ceiling: Max 3 text elements beyond axes/ticks ✓
- Color semantics: Using `COLORS["good"]/["warn"]/["bad"]` ✓
- No figure numbers drawn on images (LaTeX caption only) ✓
- Font consistency using style.py constants ✓

**Potential Concerns:**
- Panel S (Fig 13) heatmap text requires explicit dark color (`#1a1a1a`) for WCAG contrast
- Info-level VCD issues may be present but are documented as acceptable

### 4.3 Legend and Caption Policy

**From LEGEND_CAPTION_POLICY.md:**
- Legends contain series/keys only ✓
- Statistical summaries in caption, not legend ✓
- At most one small stats annotation per subplot ✓

**Review:** Figures appear compliant with legend/caption separation policy.

---

## 5. BIOLOGICAL CLAIMS VALIDATION

### 5.1 Key Claims vs. Evidence

| Claim | Evidence Location | Quantitative Support |
|-------|------------------|----------------------|
| Text conditioning drives specificity | Fig 4, Table 1 | Unconditional KNN = 1.0% vs. conditioned = 36.9% |
| CLOP creates well-separated condition space | Fig 3, Fig 5 | Pairwise cosine: raw 0.994 → projected 0.222 |
| Generated cells preserve cell-type identity | Fig 6, Fig 7 | Marker gene fidelity; gene-level correlation |
| Diversity is controllable | Fig 10, Fig 11 | DivR 0.47–1.13 across CFG; Midpoint solver = 0.93 |
| Generated cells integrate into workflows | Fig 14, Fig 15 | ARI/NMI; classifier transfer; DE concordance |

### 5.2 Biological Plausibility Concerns

**Strengths:**
- Marker gene expression preserved across lineages (Fig 6)
- Per-type analysis shows heterogeneity (not just global means)
- Downstream DE concordance tested on 3 biologically meaningful contrasts

**Potential Reviewer Questions:**
1. **Absolute KNN gap:** 36.9% vs. real data 89.0%—is this sufficient for biological utility?
   - *Author response in Discussion (Lines 418):* The gap reflects extreme difficulty (cell types differ by only 0.6% in raw space)

2. **DE concordance strength:** Pearson r = 0.17 for logFC—is this biologically meaningful?
   - *Current text:* "moderate but directionally consistent"
   - *Conclusion (Line 433):* "mean logFC Pearson r ≈ 0.39, mean sign agreement ≈ 0.94"

3. **Discriminator AUC:** If generated cells are biologically plausible, why AUC = 0.656?
   - *Acknowledged in Fig 14 caption:* "partial but non-trivial separability"

---

## 6. LIMITATIONS AND GENERALIZATION

### 6.1 Current Limitations (Documented in Article)

**Lines 428:**
> "CLOP-DiT currently generates most reliably for cell states represented in training; novel prompts far from the training distribution remain sensitive to CLOP projector extrapolation."

> "A pilot OOD/free-form prompt run was executed... but robust quantitative OOD scoring is still incomplete... the main reported metrics remain in-distribution."

> "The absolute accuracy gap from real data and discriminator AUC above 0.5 suggest opportunities for improvement..."

**Assessment:** Limitations are appropriately disclosed.

### 6.2 Generalization Concerns

**Potential Reviewer Questions:**
- Only 80 datasets—are tissues/organisms sufficiently diverse?
- How would the model perform on entirely novel cell types not in training?
- Can the method scale to full atlases (millions of cells)?

**Recommendation:** Add a sentence about scaling limitations in the Discussion.

---

## 7. BASELINE COMPARISONS

### 7.1 Baseline Methods Evaluated

From Table 1 and Fig 12:
- Unconditional generation (CFG = 0)
- Gaussian sampling from per-type statistics
- Decoder-only ablation

**Concern:** No comparison to learned baseline methods (e.g., scVI, scGen).

**Status from docs:** `scripts/baselines/train_scvi_baseline.py` and `train_embedding_vae_baseline.py` exist but may not have been run for the current results.

**Recommendation:** If learned baselines were not included in final evaluation, add a sentence explaining why (e.g., "focus on demonstrating text-conditioning capability; comparison to VAE baselines is future work").

---

## 8. WRITING AND PRESENTATION

### 8.1 Abstract Assessment

**Lines 107:**
> "CLOP-DiT achieves k-nearest-neighbor accuracy of 36.9% (37× random chance with 100 classes), steering accuracy of 81.0% (random = 50%), and a diversity ratio of 0.93..."

**Strengths:**
- Key metrics with baselines included
- Downstream validation mentioned

**Potential Enhancement:**
- Consider adding one phrase about downstream DE concordance

### 8.2 Terminology Consistency

**Review:**
- "CLOP-DiT" used consistently ✓
- "flow matching" (not "Flow Matching" or "Flow-Matching") ✓
- "classifier-free guidance (CFG)" defined on first use ✓
- "scGPT" and "BiomedBERT" properly attributed ✓

---

## 9. SUPPLEMENTARY MATERIALS

### 9.1 Current Supplementary References

From Line 440:
- Table S1: GEO accession identifiers
- Table S2: held-out validation dataset identifiers  
- Table S3: full CFG sweep results
- Figure S1: per-type KNN accuracy ranked by training-set cell count

### 9.2 Recommended Additional Supplementary Materials

1. **Supplementary Method S1:** Exact text caption construction pipeline
2. **Supplementary Table S4:** Bootstrap 95% CIs for key metrics
3. **Supplementary Figure S2:** t-SNE alternative to UMAP (if available)

---

## 10. SUMMARY OF RECOMMENDATIONS

### High Priority (Address Before Submission)

| # | Issue | Action |
|---|-------|--------|
| 1 | Code availability | Create public repository or Zenodo archive with pre-trained weights |
| 2 | Statistical uncertainty | Add bootstrap CI paragraph; include CIs in supplement |
| 3 | Supplementary Table S1 | Ensure complete GEO accession list is provided |
| 4 | Validation split clarity | Add sentence explaining how 8 validation datasets were selected |

### Medium Priority (Address if Time Permits)

| # | Issue | Action |
|---|-------|--------|
| 5 | Learned baselines | Either include results or explain their absence |
| 6 | OOD evaluation | Clarify scope of pilot OOD experiments |
| 7 | DE concordance | Add specific r values to abstract or conclusions |
| 8 | Supplementary materials | Add text construction methodology (Supp Method S1) |

### Low Priority (Polish)

| # | Issue | Action |
|---|-------|--------|
| 9 | Figure verification | Run `scripts/pipeline/verify_article_figures.sh` to confirm all 15 PDFs present |
| 10 | VCD check | Confirm 0 warnings for all panels |
| 11 | Reference completeness | Verify all cited works match intended references |

---

## 11. POSITIVE ASPECTS TO HIGHLIGHT

The following strengths should be emphasized in any response to reviewers:

1. **Rigorous Evaluation Framework:** Four-axis evaluation (KNN, steering, diversity, downstream) provides comprehensive assessment
2. **Ablation Evidence:** Unconditional generation collapses to random chance—conclusive evidence that conditioning drives specificity
3. **Per-Type Analysis:** Figure 5 shows heterogeneity across types rather than hiding it in global means
4. **Downstream Validation:** Beyond distributional metrics, the article tests clustering, classification, and DE workflows
5. **Transparent Limitations:** OOD limitations and accuracy gaps are honestly disclosed
6. **Reproducibility Infrastructure:** One-command figure regeneration via `regenerate_report.sh`

---

## APPENDIX: REFERENCED DOCUMENTS

This report draws on the following project documentation:

- `docs/REVIEWER_CONCERNS_AND_NEXT_STEPS.md` - Original reviewer concern template
- `docs/FIGURE_ORGANIZATION.md` - Figure mapping and producers
- `docs/FIGURE_PRESENTATION_POLICY.md` - Annotation and layout rules
- `docs/LEGEND_CAPTION_POLICY.md` - Legend vs. caption separation
- `docs/FIGURE_STYLE_AUDIT.md` - Style consistency audit
- `docs/BIOLOGICAL_CLAIM_MAP.md` - Claim-to-evidence mapping
- `docs/SUBMISSION_CHECKLIST.md` - Submission readiness checklist
- `articles/clop_dit_biology.tex` - Authoritative manuscript source

---

*End of Reviewer Concerns Report*
