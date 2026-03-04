# Professional Reviewer Concerns and Next Steps

*Last revised: 2026-03-06*

Based on real-time MDPI Biology requirements, computational biology/single-cell generative model review norms, and common statistical/methods weaknesses, the following concerns could be raised by a professional reviewer. Each is paired with **concrete next-step tasks** for Claude (or the author) to perform.

---

## 1. Reproducibility and Data Availability (MDPI requirement)

**Concern:** Biology requires *"full experimental details must be provided so that the results can be reproduced"* and *"make full datasets available where possible"*. The manuscript does not yet provide:
- A **Data Availability Statement** specifying where GEO datasets, curated metadata, and/or generated outputs can be accessed (or a clear justification if restricted).
- A **code/software availability** statement (repository URL, version, and license) so that training and evaluation can be reproduced.

**Next steps (Claude/author):**
1. **Add a Data Availability subsection** in the article (replace placeholder): state that training data were curated from GEO (list accession numbers or a supplementary table), that validation/test splits are defined by config (e.g. `configs/clop_v9.3.yaml`), and that code/weights are available at [repository URL] or "available upon request" with a short justification if not public.
2. **Add or point to a REPRODUCIBILITY.md** (or supplement): one-page instructions (environment, `regenerate_report.sh`, config path, expected outputs) so reviewers can reproduce key figures and tables.

---

## 2. Author and Submission Metadata (template placeholders)

**Concern:** Several front/back matter fields are still template placeholders, which will cause editorial rejection or return:
- **Affiliation:** `TODO: Add affiliation here`
- **Correspondence:** `TODO@email.com`
- **Funding:** Literal "Please add: ..." text.
- **Institutional Review / Informed Consent:** Generic "Please add ..." or "In this section you should add ..." instead of a concrete statement (e.g. "Not applicable" for public GEO data only).
- **Author Contributions:** Generic CRediT placeholder not filled.
- **Acknowledgments / Conflicts of Interest:** Placeholder or empty.

**Next steps (Claude/author):**
3. **Replace all TODOs and "Please add"** in `articles/clop_dit_biology.tex`: insert real affiliation, corresponding author email, and a short funding sentence (e.g. "This research received no external funding" if true).
4. **Add explicit ethics statements**: e.g. "Ethical review and approval were waived for this study because only publicly available, de-identified GEO data were used." and "Not applicable" for informed consent, unless human subjects apply.
5. **Fill Author Contributions** with a one-sentence CRediT-style statement (e.g. "Z.F.: conceptualization, methodology, software, validation, formal analysis, investigation, data curation, writing—original draft, writing—review and editing, visualization.").
6. **Set Acknowledgments** to a short sentence (e.g. "The authors thank …" or "The authors acknowledge …") and **Conflicts of interest** to "The authors declare no conflicts of interest" if accurate.

---

## 3. Statistics and Uncertainty Reporting

**Concern:** Reviewers of computational and single-cell papers often expect:
- **Uncertainty/confidence**: No confidence intervals (e.g. bootstrap 95% CI) or standard errors for KNN accuracy (36.9%), steering (81%), diversity ratio, or composite score (0.668). Single-point estimates can be overstated if variability is high.
- **Multiple comparisons**: Many configurations (CFG, solvers, steps) are compared; no mention of correction for multiple comparisons or pre-specified primary endpoints.
- **Sample size / power**: No justification for 100 cell types, 200 cells per group, or 80/20 train–validation split (e.g. sensitivity or power considerations).

**Next steps (Claude/author):**
7. **Add one short subsection or paragraph** in Methods or Results (e.g. "Evaluation protocol and uncertainty"): state that metrics are reported as point estimates over a fixed evaluation set; optionally add that bootstrap 95% CIs (or SE) are provided in Supplementary Table SX for key metrics (then add that table if the pipeline can compute it).
8. **Clarify primary vs exploratory**: In Methods or figure captions, state which configuration is the primary one (e.g. CFG=2.0 Euler-10 and CFG=1.0 Midpoint-10) and that others are sensitivity/exploratory; mention that no multiple-comparison correction was applied for exploratory comparisons.

---

## 4. Methods Detail and Software Versions

**Concern:** Reproducibility and reviewer trust require:
- **Exact text-condition construction**: How the 1,088 text groups and captions were built (templates, marker genes, ontology?) is only partly described; a reviewer may ask for a precise recipe or a supplementary example.
- **Preprocessing**: QC filters, HVG selection (e.g. 1,890 genes), and any batch or dataset-level exclusions should be briefly stated in the main text or clearly pointed to in supplement/code.
- **Software and versions**: scGPT, BiomedBERT, PyTorch (or JAX), and key libraries (versions) used for training and evaluation should be listed (e.g. in Methods or in a "Code availability" / supplement).

**Next steps (Claude/author):**
9. **Add one short paragraph** in Dataset or Methods: "Text descriptions were generated by [template/script]: for each cell type we combined [cell type label, tissue, organism, marker genes, etc.] into a single caption; see Supplementary Method S1 / `scripts/...` for the exact schema."
10. **Add a "Software and implementation" sentence**: e.g. "Training and evaluation were implemented in Python 3.x using PyTorch x.x, scGPT [version or commit], and Hugging Face Transformers for BiomedBERT; see `requirements.txt` and `configs/clop_v9.3.yaml` for full environment."

---

## 5. Limitations and Generalization

**Concern:** The Discussion already mentions limitations (in-distribution only, accuracy gap, need for more tissues); a reviewer may still ask:
- **Held-out validation**: Which 8 datasets (or how) were held out for validation? A one-sentence clarification in Dataset or Methods would help.
- **Out-of-distribution (OOD) text**: A short explicit statement that OOD text prompts were not evaluated (or only qualitatively) and that claims are limited to in-distribution conditions.

**Next steps (Claude/author):**
11. **In Dataset subsection**: Add one sentence, e.g. "Validation datasets were selected by [random split / by study / list GEO IDs] to ensure no overlap with training; see Supplementary Table SX for the list."
12. **In Limitations**: Keep the current limitation text; optionally add one sentence: "We did not quantitatively evaluate out-of-distribution or free-form text prompts; reported metrics apply to in-distribution conditions only."

---

## 6. Figure and Reference Consistency

**Concern:** Minor issues that can trigger editorial or reviewer queries:
- **Fig 12 (baselines)** and **Fig 14 (downstream)**: The article uses `panel_o_baseline_comparison.pdf` and `fig_downstream_pq.pdf` respectively. Ensure these files are in the submission package. The legacy `fig12_downstream_composed.pdf` has been retired in favour of the separate `fig_downstream_pq.pdf` (P+Q merged) and `panel_r_de_concordance.pdf` figures.
- **References**: Verify that all cited works (scDiff, Ding & Regev, etc.) match the intended references and that no key prior art (e.g. recent single-cell generative or text-conditioned methods) is missing.

**Next steps (Claude/author):**
13. **Confirm downstream figure assets**: Ensure `fig_downstream_pq.pdf` (clustering + classifier) and `panel_r_de_concordance.pdf` (DE concordance) are included in the submission package.
14. **Quick reference pass**: Skim Introduction and Related Work for missing citations (e.g. latest scGPT, flow matching, or single-cell generation papers from 2023–2024) and fix any wrong author/year in the bibliography.

---

## 7. Abstract and Claims

**Concern:** The abstract is dense and accurate; a reviewer might still ask:
- **Quantified baseline**: "37× random" is clear; optionally add a single sentence that a baseline (e.g. unconditional or Gaussian) is at chance (1% KNN) to reinforce the claim that conditioning drives the effect.
- **Downstream numbers**: If ARI, NMI, or DE concordance values are reported in the main text or figures, ensure they are consistent and that at least one downstream number appears in the abstract or conclusions (currently "integrate with real data" is qualitative).

**Next steps (Claude/author):**
15. **Abstract**: Already states unconditional collapse; no change strictly required. Optionally add one phrase: e.g. "Downstream validation shows clustering alignment (ARI/NMI), classifier transfer, and DE concordance (e.g. Pearson r for logFC)."
16. **Conclusions**: If specific downstream numbers (e.g. ARI, DE r) are in the results, add one sentence to Conclusions with those values so the take-home is quantitative.

---

## 8. Supplementary and Reporting Guidelines

**Concern:** Biology encourages standards and clear supplementary material:
- **Reporting guideline**: If the journal or field has a preferred guideline (e.g. for computational studies or ML in biology), a one-sentence statement can strengthen the submission.
- **Supplementary materials**: A short list (Table S1: GEO accessions; Table S2: validation dataset IDs; Table S3: full CFG sweep; Figure S1: …) in the Data Availability or a "Supplementary Materials" sentence helps reviewers and reproducibility.

**Next steps (Claude/author):**
17. **Add a sentence** (e.g. in Data Availability or Methods): "Supplementary materials include: Table S1 (GEO accession list), Table S2 (validation dataset identifiers), Table S3 (full CFG sweep), and Figure S1 (…)."
18. **Create minimal supplement** if not present: at least Table S1 (dataset list) and a one-page reproducibility instruction (or point to `docs/QUICK_START.md` / `scripts/regenerate_report.sh`).

---

## Summary: Task List for Claude/Author

| # | Priority | Task |
|---|----------|------|
| 1 | High | Add Data Availability statement (GEO, code repo or "upon request", config) |
| 2 | High | Add REPRODUCIBILITY.md or supplement with reproduce instructions |
| 3 | High | Replace affiliation, correspondence, funding placeholders |
| 4 | High | Add ethics statements (IRB waiver / not applicable, informed consent) |
| 5 | High | Fill Author Contributions (CRediT-style) |
| 6 | High | Set Acknowledgments and Conflicts of interest |
| 7 | Medium | Add uncertainty/CI paragraph and optionally bootstrap table |
| 8 | Medium | Clarify primary vs exploratory comparisons |
| 9 | Medium | Add text-condition construction paragraph and software/versions |
| 10 | Medium | Clarify validation split (which 8 datasets) |
| 11 | Medium | One sentence on OOD limitation in Limitations |
| 12 | Low | Confirm Fig 12 file and caption; reference pass |
| 13 | Low | Optional: downstream numbers in abstract/conclusions; supplement list |

Implementing items **1–6** addresses the most common causes of editorial return (incomplete metadata and data/ethics statements). Items **7–11** address typical reviewer concerns on reproducibility and statistics. Items **12–13** polish the submission.
