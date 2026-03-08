# Reviewer Report: Results, Visualization Policy, Figure Presentation, and Writing

Date: 2026-03-07
Scope reviewed:
- `articles/clop_dit_biology.tex` (authoritative manuscript)
- `docs/CLOP_DiT_JBHI_Article.md` (derivative manuscript)
- `docs/FIGURE_PRESENTATION_POLICY.md`
- `docs/LEGEND_CAPTION_POLICY.md`
- `docs/WRITING_AND_DOCS_POLICY.md`
- `docs/ARTICLE_SUBSECTION_POLICY.md`
- `docs/FIGURE_ORGANIZATION.md`
- `docs/FIGURES_9-12_POLICY.md`
- `docs/FIGURES_12-15_POLICY.md`
- `docs/CLOP-DiT_Evaluation_Report.md`
- `docs/REGENERATION_STATUS.md`

## Executive Assessment

The project has strong policy coverage and clear intent for publication-quality figures and writing. However, there are reviewer-facing risks from internal metric inconsistencies, cross-document drift, and missing figure artifacts in this workspace snapshot (which blocks independent verification of actual figure presentation compliance).

## Major Concerns (Highest Priority)

1) Inconsistent key results across manuscript sections
- Concern:
  - `articles/clop_dit_biology.tex` reports conflicting downstream DE concordance values:
    - Abstract: logFC Pearson `r = 0.17`
    - Conclusions: mean logFC Pearson `r ≈ 0.39`
  - Classifier transfer wording is also inconsistent:
    - Abstract calls downstream classifier transfer "logistic regression accuracy 51.1%"
    - Conclusions report classifier transfer "30.8% (macro F1 0.247)"
- Reviewer risk: Appears as unstable or selectively reported outcomes.
- Recommended fix:
  - Harmonize all headline downstream metrics in Abstract, Results, Discussion, and Conclusions to one validated source.
  - Distinguish "core LinAcc metric" from "downstream transfer accuracy" explicitly if they are different evaluations.

2) Temperature value inconsistency in training narrative/caption
- Concern:
  - Methods define learnable temperature initialized at `0.07` with range `[0.01, 0.5]`.
  - Training text references final temperature around `0.060`.
  - Figure 2 caption states final `tau = 14.0`.
- Reviewer risk: Mathematical inconsistency (tau vs inverse temperature/logit scale) undermines confidence in implementation and reporting.
- Recommended fix:
  - Standardize notation (either `tau` or `1/tau` / logit scale) and annotate exactly what is plotted in the caption.
  - Ensure all mentions use the same variable definition.

3) Composite benchmark score mismatch across docs
- Concern:
  - `articles/clop_dit_biology.tex` states composite score `0.844`.
  - `docs/FIGURE_ORGANIZATION.md` and `docs/CLOP-DiT_Evaluation_Report.md` reference benchmark panel score `0.668`.
- Reviewer risk: Perceived inconsistency in benchmarking pipeline or stale manuscript text.
- Recommended fix:
  - Recompute and lock the benchmark value for the submission build.
  - Update manuscript + docs together using one canonical generated artifact.

4) Figure presentation cannot be independently verified in current workspace snapshot
- Concern:
  - No `results/figures/*` or `articles/figures/*` files are present here.
  - Policy docs assert regenerated article-ready figures exist, but artifacts are unavailable in this worktree.
- Reviewer risk: Submission packaging and figure-policy compliance cannot be audited from current project state.
- Recommended fix:
  - Regenerate figures in the target environment and include the 15 article figure PDFs (or validated symlinks) before submission QA.
  - Archive a verification log from `scripts/pipeline/verify_article_figures.sh`.

## Medium Concerns

5) Source-of-truth drift between authoritative `.tex` and derivative markdown article
- Concern:
  - `WRITING_AND_DOCS_POLICY.md` correctly states `.tex` is authoritative, but `docs/CLOP_DiT_JBHI_Article.md` contains different figure mapping and some differing claims/values.
- Reviewer risk: If the derivative markdown is shared externally, reviewers may detect contradictions.
- Recommended fix:
  - Either sync derivative markdown to current `.tex` or add a stronger "internal snapshot only" warning at top of markdown.

6) Uncertainty reporting remains limited for many headline metrics
- Concern:
  - Methods acknowledge point estimates and exploratory comparisons; CIs appear for selected benchmark elements but not consistently for core metrics (KNN, steering, diversity in main claims).
- Reviewer risk: Statistical rigor concern (variance not quantified for key claims).
- Recommended fix:
  - Add uncertainty estimates (bootstrap CI or repeated-seed intervals) for headline metrics in main text or supplement table.

7) Reproducibility and availability statement may still be viewed as weak
- Concern:
  - Data/code/model availability uses "available upon request/pending approval" language.
- Reviewer risk: Reproducibility and editorial strictness concerns.
- Recommended fix:
  - Provide a versioned public release (or clearer justified restriction statement plus a concrete request workflow and expected response window).

## Writing and Presentation Policy Alignment Notes

- Positive:
  - Figure policy architecture is well-defined (annotation budget, legend/caption split, style constants, VCD checks).
  - Results section includes signposting and figure-family organization, matching subsection policy intent.
- Remaining risk:
  - Some result passages remain dense and combine multiple claims in one subsection, which may dilute the "one primary claim per subsection" rule.

## Pre-Submission Action Checklist

1. Resolve all metric inconsistencies (r values, classifier transfer, benchmark score, temperature notation).
2. Regenerate all figures and re-verify article figure package.
3. Reconcile manuscript and derivative markdown or clearly deprecate derivative for reviewer use.
4. Add uncertainty for headline claims (at minimum in supplement).
5. Final consistency pass: Abstract, Results, Discussion, Conclusions, figure captions, and tables.

## Overall Recommendation

Proceed after a focused consistency-and-verification pass. The methodological narrative is strong, but current cross-document and cross-section numeric inconsistencies are likely to trigger reviewer confidence and reproducibility concerns if not resolved before submission.
