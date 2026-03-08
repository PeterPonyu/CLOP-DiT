# CLOP-DiT Reviewer Concerns Report

*Generated: 2026-03-07*
*Manuscript: "CLOP-DiT: Text-Guided Single-Cell Gene Expression Generation via Contrastive Language-Omics Pretraining and Diffusion Transformers" — MDPI Biology submission*

---

## Executive Summary

This report consolidates concerns that a professional reviewer is likely to raise, derived from a full reading of the manuscript (`articles/clop_dit_biology.tex`), all figure presentation policies, result JSON files, ablation reports, and internal documentation. Concerns are organized from highest to lowest severity. Items marked **BLOCKING** may cause editorial return or reject; items marked **MAJOR** are likely to require revision; items marked **MINOR** are polish tasks.

---

## 1. Submission-Blocking / Editorial-Return Issues

### 1.1 [STATUS: Resolved in .tex, checklist not updated]
The submission checklist (`docs/SUBMISSION_CHECKLIST.md`) marks items 1–6 (affiliation, correspondence, funding, ethics, author contributions, acknowledgments) as unchecked, but the LaTeX file already contains correct content for all six. The checklist itself is stale. **No action needed on the .tex**, but the checklist should be verified and ticked off before submission to avoid confusion.

### 1.2 [BLOCKING] Code and Data Repository Not Linked
The Data Availability section says the code is "available upon request pending institutional approval for open-source release." MDPI Biology requires that reviewers can, at minimum, access enough to reproduce figures. The `regenerate_report.sh` script and all figure-reproduction instructions exist in `docs/QUICK_START.md`, but there is no public URL. A reviewer who cannot access the code repository has no path to reproducing Figures 2–15. **Required action:** Either provide a permanent link (GitHub, Zenodo) or provide a detailed "how to obtain" statement and confirm that the reproduction package can be supplied to reviewers within a reasonable time (MDPI standard: within 14 days of request).

### 1.3 [BLOCKING] No Supplementary Tables S1 / S2 Submitted
The manuscript references Supplementary Table S1 (80 GEO accession identifiers) and Table S2 (held-out validation datasets) in the Dataset subsection and Data Availability section. These tables do not exist as submitted files in the worktree. Without them, reviewers cannot verify the dataset composition, and the Data Availability statement is incomplete.

---

## 2. Results Inconsistencies (Internal Contradictions)

### 2.1 [MAJOR] Classifier Accuracy Reported Inconsistently
- **Abstract:** "logistic regression accuracy 51.1%"
- **Table 1:** `LinAcc = 0.511` for CFG=2.0 Euler-10
- **Conclusions:** "classifier transfer reaches 30.8% accuracy (macro F1 0.247)"

The abstract and table report 51.1%, the conclusions report 30.8%. These are two different metrics (linear classifier on generated embeddings vs. cell-type classifier on generated expressions), but the distinction is not explained at first occurrence and the word "classifier" is used for both without qualification. A reviewer will flag this as a numerical contradiction. **Required action:** Either reconcile with explicit metric names at both sites, or add a sentence clarifying that these measure different classification tasks.

### 2.2 [MAJOR] logFC Pearson r Reported Inconsistently
- **Abstract:** "differential expression concordance (logFC Pearson r = 0.17)"
- **Discussion / Conclusions:** "mean logFC Pearson r ≈ 0.39, mean sign agreement ≈ 0.94"

The abstract gives 0.17 while the conclusions give ~0.39. No explanation is provided for this discrepancy. These are likely computed over different contrasts or gene sets. **Required action:** Reconcile or explicitly note that 0.17 is the conservative single-contrast value and 0.39 is the multi-contrast mean, or vice versa.

### 2.3 [MAJOR] Composite Score Inconsistency
Internal documentation (`docs/REVIEWER_CONCERNS_AND_NEXT_STEPS.md`, written before the current manuscript draft) references a composite score of 0.668. The current manuscript states 0.844. This version mismatch suggests the documentation was not updated when the pipeline was revised. The manuscript value (0.844) should be treated as authoritative, but the inconsistency signals that internal docs lag behind the results.

---

## 3. Statistical and Methodological Concerns

### 3.1 [MAJOR] Ablation Results Are Paradoxical — May Undermine Methods Claims
The ablation report (`results/ablations/ablation_report.md`) shows that removing regularization components **improves** CLOP validation prototype accuracy substantially:
- `no_cohesion`: +10.24 pp over baseline (86.41% vs. 76.17%)
- `no_cell_noise`: +10.07 pp over baseline

This means the "full v8.2 config (all improvements enabled)" is actually **worse** than simpler configurations on the CLOP validation metric. The manuscript presents the full-config model as the final system without discussing this. A reviewer will ask: "Why are you using a worse configuration?" The ablation section is also not referenced in the article — it is only in supplementary/internal docs. **Required action:** Either (a) explain in the Discussion why the production config is preferred despite lower CLOP validation accuracy (e.g., downstream DiT generation quality is better, not CLOP validation accuracy), or (b) update the production config to use the best-found settings and re-report all results.

### 3.2 [MAJOR] Only Analytical Baselines, No Trained Generative Baselines
The manuscript compares CLOP-DiT against:
- Unconditional generation (CFG = 0)
- Gaussian per-type sampling
- Shuffled-label and mean-collapse controls
- Decoder-only ablations

The code infrastructure for scVI and Embedding VAE baselines exists (`scripts/baselines/train_scvi_baseline.py`, `models/baselines/embedding_vae.pt`), but neither is scored in the main article figures or table. Reviewers familiar with scVI, scGen, or other learned single-cell generative models will note the absence of any trained learned baseline. **Required action:** Run at least one learned baseline (e.g., scVI conditional) and include its scores in Table 1 and Figure 12. The results pipeline (`src/evaluation/model_benchmarking.py`) already supports this via the baseline registry.

### 3.3 [MAJOR] No Uncertainty Quantification Except on Panel S
KNN accuracy (36.9%), steering (81.0%), and diversity ratio (0.93) are reported as point estimates with no confidence intervals or standard errors. Bootstrap 95% CIs exist in `benchmark_report.json` and are shown in Figure 13 (Panel S), but are not propagated to the headline numbers in the abstract, Table 1, or the Results subsection headers. A computational biology reviewer will ask for uncertainty intervals on all primary endpoint numbers. **Required action:** Add bootstrap 95% CIs to at least the primary-endpoint numbers in the abstract, Table 1, and the first paragraph of Section 3.3.

### 3.4 [MAJOR] OOD Evaluation Has No Quantitative Results
`results/ood_evaluation/ood_results.json` contains 6 novel-type prompts and 8 free-form prompts, but all metric fields are empty dictionaries (`"free_form_metrics": {}, "structured_metrics": {}, "comparison": {}`). The Discussion acknowledges "a pilot OOD/free-form prompt run was executed in this workstream, but robust quantitative OOD scoring is still incomplete." This is acceptable disclosure in the Discussion, but there is a risk a reviewer will request quantitative OOD results. **Required action:** Run the OOD evaluation pipeline, populate the metrics, and either add a supplementary figure or at minimum report a summary statistic. Alternatively, remove the OOD JSON from the submission package to avoid implying incomplete results were intentionally withheld.

### 3.5 [MINOR] Rare-Cell Augmentation Shows Marginal Benefit for the Target Type
`results/rare_cell_augmentation/augmentation_results.json` shows that augmenting with 1× synthetic "Cycling cells" yields macro F1 0.9526 vs. baseline 0.9645, and 10× augmentation yields the same as 5× (0.9466). The rare-type F1 for "Cycling cells" is **lower** in all augmented conditions (0.79/0.74) compared to the unaided baseline (0.83). This suggests synthetic augmentation slightly hurts the target rare-cell type in this experiment. This result is not discussed in the manuscript; the Discussion mentions "data augmentation for rare cell types" as a promising application without caveating the actual experiment. **Required action:** Either include this result in the Discussion as a finding ("augmentation at ratio 1–10× did not improve rare-cell classifier recall in the tested configuration") or remove the augmentation experiment from the submission package if results are not yet positive enough to report.

### 3.6 [MINOR] 100 Cell Types Evaluated Out of 1,088 Training Groups
The abstract reports results across "100 cell types." The dataset has 1,088 text groups. The evaluation uses a subset of 100, but no criterion for this selection is given. **Required action:** Add one sentence in the Evaluation subsection explaining the selection criterion (e.g., "100 types with sufficient training representation, ≥ 50 cells per group").

### 3.7 [MINOR] Primary vs. Exploratory Comparisons Not Pre-Specified
The manuscript compares 8 CFG scales, 2 ODE solvers, and multiple step counts, but the two "primary operating points" (CFG=2.0 Euler-10; CFG=1.0 Midpoint-10) are not pre-declared before the CFG sweep table. A reviewer may apply multiple-comparison concerns to the table. **Required action:** The statement in the Evaluation subsection already partially addresses this: "Two configurations are designated as primary operating points." Strengthen by moving this declaration to a separate, clearly labelled sentence before presenting Table 1.

---

## 4. Methods Transparency and Reproducibility

### 4.1 [MAJOR] Text Caption Construction Recipe Not Fully Documented
The Dataset subsection states captions follow a template `"{cell type}, tissue: {tissue}, organism: {organism}, markers: {top 5 DE marker genes}, context: {disease/condition}"` and that the script is `scripts/build_text_captions.py`. However, no file named `build_text_captions.py` exists in the worktree. The script that performs this function appears to be `scripts/02_subcluster_descriptions.py` or `scripts/02b_enrich_descriptions.py`, neither of which is referenced in the article. A reviewer who attempts to reproduce the dataset processing will be immediately blocked. **Required action:** Correct the script filename in the manuscript to match the actual file, or add `build_text_captions.py` as a wrapper that calls the correct script.

### 4.2 [MINOR] ZCA Whitening Mentioned in Figure Caption Only
The architecture figure caption mentions "ZCA whitening is applied to BiomedBERT embeddings prior to projection," but the main text Methods section does not describe ZCA whitening at all. A reviewer may ask where this preprocessing step is described, how its parameters were set, and why it was applied to only one modality. **Required action:** Add one sentence in Section 2.2 (CLOP) describing ZCA whitening, its parameters, and the rationale.

### 4.3 [MINOR] Logit-Normal Timestep Distribution Not Justified
Section 2.3 states timesteps are "drawn from a logit-normal distribution" for flow matching training but gives no citation or rationale for choosing logit-normal over uniform. This is a non-standard choice that reviewers may query. **Required action:** Add a brief justification or cite the source (likely from Stable Diffusion 3 or similar).

---

## 5. Writing and Framing Issues

### 5.1 [MAJOR] "First Method" Claim Requires Verification
The Introduction states CLOP-DiT is "the first method to accept free-text biological descriptions as input for single-cell generation." Cell2Sentence (Levine et al., 2023) and similar work converts scRNA-seq data to language model inputs. Generative models conditioned on free-text in biology are an active area. If CLOP-DiT is genuinely first in this specific setup (contrastive alignment + DiT + flow matching), the claim should be scoped more precisely. **Required action:** Add a qualifier: "...to our knowledge, the first method to combine contrastive language-omics alignment with a diffusion transformer for text-conditioned single-cell generation."

### 5.2 [MINOR] Evaluation Is Strictly In-Distribution
The manuscript correctly limits primary claims to in-distribution conditions (Limitations section), but the abstract and most results sections describe the evaluation without this qualifier. A reader may not realize until the Limitations section that all 36.9% KNN, 81% steering, and 0.93 diversity numbers are measured on the same training-distribution prompts. **Required action:** Add one qualifying phrase in the abstract or in the Results opening paragraph: "...across 100 in-distribution cell types..."

### 5.3 [MINOR] Discussion "Evidence Chain" Paragraph Is Self-Promotional
The Discussion paragraph beginning "Why the biological evidence is credible" reads defensively rather than analytically. Phrases like "This structure makes it possible to inspect not only whether the model works on average, but also where it fails" are accurate but sound like promotional copy. **Required action:** Reframe as a standard discussion of evidence quality and limitations.

### 5.4 [MINOR] JBHI Article Markdown vs. LaTeX — Figure Numbering Conflict
`docs/CLOP_DiT_JBHI_Article.md` uses a different figure numbering (Fig 9 = decoder comparison / Panel O; Fig 10 = ODE step / Panel L), which conflicts with the authoritative LaTeX mapping (Fig 9 = Panel M conditioning UMAP; Fig 10 = Panel J diversity diagnostics). If this markdown is ever shared with reviewers as a preprint or supplement, the mismatched figure numbers will cause confusion. **Required action:** Either update `CLOP_DiT_JBHI_Article.md` to match the LaTeX figure ordering or add a note at the top stating it uses a legacy figure ordering.

---

## 6. Figure Presentation Concerns

### 6.1 [MAJOR] No Per-Type Failure Mode Visualization (Critical for Reviewers)
Figures 4–8 report aggregate metrics that hide catastrophic per-type failures. The enhancement roadmap (`docs/FIGURE_ENHANCEMENT_ROADMAP.md`) notes: "Inhibitory GABAergic neurons have centroid cosine of 0.39 vs 0.92+ for others." No article figure explicitly shows which cell types fail. Reviewers in single-cell biology routinely ask "which types work and which don't?" Figure 5 (fidelity-alignment) has a per-type bar chart, but it omits specific outlier type names for "readability." **Required action:** Implement Enhancement 1.1 (per-type heterogeneity scatter) or at minimum label the top 3 outlier types in Figure 5, panel A. This is a Tier 1 enhancement already planned.

### 6.2 [MAJOR] Classifier Per-Type Heatmap Missing from Figure 14
Figure 14 (`fig_downstream_pq.pdf`) shows overall classifier accuracy and a confusion matrix, but the Tier 1 enhancement (1.3, classifier per-type precision/recall/F1 heatmap) is not yet implemented. The Results section already describes this panel in the manuscript text: "The enhanced classifier panel reports a per-type precision/recall/F1 heatmap." If the figure does not yet contain this heatmap, there is a mismatch between text and figure. **Required action:** Implement the heatmap or remove the reference from the text until the panel is generated.

### 6.3 [MAJOR] DE Effect-Size Scatter Not Yet Implemented in Figure 15
Similar to 6.2, the Results section describes Figure 15 as containing "effect-size-weighted scatter...with point size proportional to average absolute effect size and colour indicating adjusted significance." This is Tier 1 Enhancement 1.4, which is not yet marked as done. If the current `panel_r_de_concordance.pdf` does not contain these features, the caption and Results text are ahead of the actual figure. **Required action:** Implement or correct the text description to match the current figure contents.

### 6.4 [MINOR] Annotation Budget Violations in Dense Panels
Figure presentation policy (`docs/FIGURE_PRESENTATION_POLICY.md`, §1–4) limits each subplot to one stats box and two highlighted outliers, with at most three text elements beyond axes labels. The descriptions of Figures 5 and 14 in the manuscript text suggest these panels carry more annotation than the policy allows (e.g., "centroid cosine is ranked across the 100 evaluated cell types" with type labels). The VCD (Visual Conflict Detector) policy in `scripts/vcd/` enforces this, but VCD operates only at save-time. **Required action:** Run `scripts/pipeline/verify_article_figures.sh` and check VCD output logs for annotation budget violations after the next regeneration.

### 6.5 [MINOR] Marker Gene Figure Uses Lineage-Color Mapping in Caption Only
Figure 6 caption states "bar colours group markers by lineage family (CD8 T, myeloid, epithelial, stromal); the lineage colour mapping is described in the caption rather than repeated as an in-panel legend." This violates LEGEND_CAPTION_POLICY (the legend should list series/keys). A reader scanning the figure without the caption will not know what the bar colors mean. **Required action:** Add a minimal legend or color key directly in the figure for the four lineage colors, even if compact.

### 6.6 [MINOR] Fig 11 Violin Panel Has Runtime Dependencies That May Be Missing
`docs/FIGURES_9-12_POLICY.md` states Fig 11 is produced only when all three components are available: panels L, K, and the violin (which requires `results/diversity_diagnostics.json` and embedding caches). If any is missing, the figure is skipped entirely. The submission package should confirm this file is present and up to date. **Required action:** Verify `results/figures/fig_diversity_tradeoff.pdf` contains the violin panel by checking its page count or visual content.

---

## 7. Broader Scientific Concerns

### 7.1 [MAJOR] 37× Random KNN Accuracy Is Impressive but the Gap to Real Data Is Large
The manuscript frames 36.9% KNN as "37× random chance" and correctly contextualizes the 0.994 pairwise cosine similarity of raw embeddings. However, the gap to real-data KNN (89.0%) is a 52 percentage-point deficit. The Discussion addresses this, but a reviewer will ask whether this gap implies the generated cells are useful in practice, given that downstream classifier transfer is also limited (30.8% accuracy, macro F1 0.247). The discriminator AUC of 0.656 further confirms non-trivial separability of real vs. generated cells. **Suggested addition:** Include a calibration statement in the Discussion or Abstract quantifying what level of KNN gap is acceptable for downstream applications (e.g., cite a use case where 30% accuracy is sufficient, such as augmenting rare-cell training data for coarse-grained classification).

### 7.2 [MINOR] Circular Design: Text Captions Are Derived from Training-Set Marker Genes
The text captions incorporate "top 5 DE marker genes" identified by one-vs.-rest Wilcoxon test on the training set. This creates a partially circular information loop: the text condition encodes training-derived marker genes, which are then evaluated by whether the generated expression preserves those same marker genes. This does not invalidate the result but reduces the strength of "biological fidelity" claims. **Suggested addition:** Add one sentence in Section 2.1 or the Discussion acknowledging this: "Because text captions incorporate training-derived marker genes, marker-gene fidelity metrics assess internal consistency of the pipeline rather than independent biological validation."

### 7.3 [MINOR] CLOP Validation Accuracy of ~2.5% Framing Requires Care
The manuscript correctly notes that 2.5% CLOP validation accuracy represents 6.4× random chance (1/256 groups). However, the "256" refers to batch size or mini-batch groups, not the full 1,088 group count. If the metric is accuracy over the full evaluation set, random chance is 1/1088 = 0.092%, making 2.5% represent ~27× random rather than 6.4×. The manuscript must be consistent about what denominator is used. **Required action:** Clarify whether the 2.5% is measured over 256 negatives (within-batch) or all 1,088 text groups; report the correct random-chance baseline for this metric.

---

## 8. Reference and Citation Gaps

### 8.1 [MINOR] scDiff Reference May Be Incorrect
The manuscript cites "Ding, J.; Regev, A. Deep generative model embedding of single-cell RNA-Seq profiles on hyperspheres and hyperbolic spaces. Nat. Commun. 2021, 12, 2554" as `ref-scdiff`. While this paper exists, it does not use the name "scDiff." If this citation is meant to represent the broader class of geometric generative models, the inline citation label `\cite{ref-scdiff}` creates a misleading impression. Verify the correct reference for the intended claim.

### 8.2 [MINOR] No Citation for Cell2Sentence or Text-Conditioned Single-Cell Methods
Given the "first method" claim (§5.1 above), a reviewer will check whether Cell2Sentence (Levine et al., 2023, Nature Methods) and similar large-language-model-based single-cell approaches are discussed and cited. They are absent from the current reference list.

### 8.3 [MINOR] Flow Matching Citation Is Conference Proceedings Only
`ref-flow-matching` (Lipman et al., ICLR 2023) is now a published paper with a broader impact; additionally, Albergo & Vanden-Eijnden (2022, "Building Normalizing Flows with Stochastic Interpolants") should be cited alongside Lipman et al. as the foundational flow-matching reference. Check whether the journal requires citing the final published version rather than conference proceedings.

---

## 9. Figure–Caption Alignment Gaps

| Article Figure | Caption Says | Current Figure Status | Issue |
|---|---|---|---|
| Fig 5 `fig_fidelity_alignment` | "Specific outlier type names are omitted from the bar panels for readability" | Bars without labels | No way to identify failing types without table |
| Fig 6 `panel_n_marker_gene_comparison` | "lineage colour mapping is described in the caption rather than repeated" | No in-figure color key | Readers must cross-reference caption to decode bar colors |
| Fig 9 `panel_m_conditioning_umap` | "Panel titles report the number of displayed cells and the mean within-type diversity" | Needs verification | Confirm panel titles are present in current PDF |
| Fig 14 `fig_downstream_pq` | "per-type precision/recall/F1 heatmap restricted to worst and best 20 cell types by F1" | Tier 1 enhancement not yet implemented | Caption ahead of figure |
| Fig 15 `panel_r_de_concordance` | "point size proportional to average absolute effect size and colour indicating adjusted significance" | Tier 1 enhancement not yet implemented | Caption ahead of figure |

---

## 10. Checklist: Open Items Before Submission

### Blocking
- [ ] Add Supplementary Table S1 (all 80 GEO accession IDs) and Table S2 (validation dataset IDs)
- [ ] Provide public or requestable code/data access URL; update Data Availability statement
- [ ] Reconcile classifier accuracy: 51.1% (abstract/Table 1) vs. 30.8% (conclusions)
- [ ] Reconcile logFC Pearson r: 0.17 (abstract) vs. ~0.39 (conclusions)
- [ ] Correct script name: `scripts/build_text_captions.py` does not exist

### Major
- [ ] Add or run learned baseline (scVI or Embedding VAE) in Table 1 and Figure 12
- [ ] Add bootstrap 95% CIs to headline numbers in abstract and Table 1
- [ ] Resolve ablation paradox: explain why production config is used despite lower validation accuracy than ablated versions
- [ ] Implement or remove references to per-type classifier heatmap (Fig 14) and DE effect-size scatter (Fig 15)
- [ ] Address rare-cell augmentation result showing no improvement for cycling cells
- [ ] Run and populate OOD evaluation metrics, or remove empty OOD JSON from submission package
- [ ] Populate the 100-type evaluation selection criterion in the Methods

### Minor
- [ ] Add ZCA whitening description to Section 2.2
- [ ] Add logit-normal timestep justification/citation to Section 2.3
- [ ] Add lineage color key to Figure 6 (marker gene comparison)
- [ ] Qualify "first method" claim in Introduction
- [ ] Add "in-distribution" qualifier to abstract/results metric statements
- [ ] Clarify CLOP validation accuracy denominator (batch-level vs. full-set)
- [ ] Add/check Cell2Sentence citation; verify scDiff reference
- [ ] Verify Fig 11 violin panel is present in current PDF
- [ ] Run `scripts/pipeline/verify_article_figures.sh` and resolve any VCD violations
- [ ] Tick completed items in `docs/SUBMISSION_CHECKLIST.md`

---

## Appendix A: Key Metrics Reference (for reviewer response)

| Metric | Value | Baseline / Random |
|---|---|---|
| KNN top-1 accuracy | 36.9% | 1.0% random; 89.0% real data |
| KNN top-5 accuracy | 55.3% | — |
| Steering accuracy | 81.0% | 50.0% random |
| Diversity ratio (CFG=2.0 Euler) | 0.513 | 1.0 ideal |
| Diversity ratio (CFG=1.0 Midpoint) | 0.929 | 1.0 ideal |
| Linear classifier accuracy (abstract) | 51.1% | — |
| Cell-type classifier transfer (conclusions) | 30.8% (macro F1 0.247) | — |
| Discriminator AUC (real vs. generated) | 0.656 | 0.5 indistinguishable |
| logFC Pearson r (abstract) | 0.17 | — |
| logFC Pearson r (conclusions) | ~0.39 | — |
| Composite benchmark score | 0.844 | — |
| CLOP pairwise cosine (projected) | 0.222 | 0.994 raw |
| CLOP validation accuracy | ~2.5% | 1/256 batch ≈ 0.39% |
| DiT val velocity cosine (EMA) | 0.976 | — |
| Rare-cell (Cycling) F1 — baseline | 0.828 | — |
| Rare-cell (Cycling) F1 — 10× augment | 0.741 | LOWER than baseline |

---

## Appendix B: Source Documents Reviewed

- `articles/clop_dit_biology.tex` — main manuscript
- `docs/REVIEWER_CONCERNS_AND_NEXT_STEPS.md` — prior reviewer concern tracker
- `docs/FIGURE_PRESENTATION_POLICY.md` — figure presentation rules
- `docs/FIGURES_9-12_POLICY.md`, `docs/FIGURES_12-15_POLICY.md` — figure-specific policies
- `docs/LEGEND_CAPTION_POLICY.md` — legend vs. caption rules
- `docs/FIGURE_STYLE_AUDIT.md` — style consistency audit
- `docs/FIGURE_ENHANCEMENT_ROADMAP.md` — planned figure enhancements
- `docs/BIOLOGICAL_CLAIM_MAP.md` — claim-to-evidence mapping
- `docs/SUBMISSION_CHECKLIST.md` — submission checklist
- `docs/REGENERATION_STATUS.md` — figure regeneration status
- `results/ablations/ablation_report.md` — ablation study results
- `results/ood_evaluation/ood_results.json` — OOD evaluation (empty metrics)
- `results/rare_cell_augmentation/augmentation_results.json` — augmentation experiment
- `todo/2026-03-06-figure-regeneration-gaps.md` — recently resolved figure gaps
