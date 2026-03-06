# Documentation Index

This index organizes all markdown documentation in the CLOP-DiT repo for quick reference.

---

## Operational (run / pipeline)


| Document                                                   | Purpose                                                                                                                                                                                      |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [QUICK_START.md](QUICK_START.md)                           | 9-step pipeline (architecture → embeddings → decode → diversity → conditioning → downstream → benchmark → visualization → verify). Use this to regenerate figures and run the full workflow. |
| [FIGURE_ORGANIZATION.md](FIGURE_ORGANIZATION.md)           | Panel list (A–S), merged figures, article figure map, canonical producers, VCD architecture, figure logic and limitations, and single-source-of-truth policy.                                |
| [REGENERATION_STATUS.md](REGENERATION_STATUS.md)           | Figure staleness, what changed in code, prerequisites, and how to run regeneration in your dev environment.                                                                                  |
| [DIRECTORY_CLEANUP_POLICY.md](DIRECTORY_CLEANUP_POLICY.md) | Directory classification, what is tracked in Git, cleanup commands, rebuild commands, and archive policy.                                                                                    |


---

## Roadmaps and plans


| Document                                                                                 | Purpose                                                                                                                                |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [BIOLOGICAL_CLAIM_MAP.md](BIOLOGICAL_CLAIM_MAP.md)                                       | Reviewer-facing map of each biological claim to the exact figures, metrics, and producing modules/scripts.                             |
| [BIOLOGICAL_VALIDATION_EXPANSION.md](BIOLOGICAL_VALIDATION_EXPANSION.md)                 | Additional datasets, DE contrasts, and downstream applications that broaden the current biological results.                            |
| [ABLATION_SUITE.md](ABLATION_SUITE.md)                                                   | Organized view of the current CLOP ablation families and reporting policy.                                                             |
| [FIGURE_ENHANCEMENT_ROADMAP.md](FIGURE_ENHANCEMENT_ROADMAP.md)                           | Prioritized enhancements (Tier 1–4): per-type heterogeneity, distribution tails, classifier heatmap, DE effect-size scatter, CIs, etc. |
| [FIGURE_DESIGN_AND_PROMPT_DIVERSITY_PLAN.md](FIGURE_DESIGN_AND_PROMPT_DIVERSITY_PLAN.md) | Design and prompt-diversity planning for figures.                                                                                      |
| [ROBUSTNESS_EXPERIMENTS.md](ROBUSTNESS_EXPERIMENTS.md)                                   | Seed, subsampling, and caption-sensitivity robustness design plus execution harness.                                                   |
| [REVIEWER_CONCERNS_AND_NEXT_STEPS.md](REVIEWER_CONCERNS_AND_NEXT_STEPS.md)               | Reviewer concerns and planned next steps.                                                                                              |
| [ENHANCEMENT_TASKS.md](ENHANCEMENT_TASKS.md)                                             | Concise actionable task list; points to FIGURE_ENHANCEMENT_ROADMAP for detail.                                                         |


---

## Audit and status


| Document                                                         | Purpose                                                                                    |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| [AUDIT_CLAUDE_COMPLETED_WORK.md](AUDIT_CLAUDE_COMPLETED_WORK.md) | Audit of completed work (author info, reviewer items, figure overlap/truncation, VCD).     |
| [DIRECTORY_CLEANUP_POLICY.md](DIRECTORY_CLEANUP_POLICY.md)       | Policy for cleaning and organizing project directories.                                    |
| [WRITING_AND_DOCS_POLICY.md](WRITING_AND_DOCS_POLICY.md)         | Manuscript source of truth, session doc placement, style, and links to figure/docs policy. |
| [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md)               | Pre-submission checklist (metadata, ethics, reproducibility, figures, references).         |


---

## Article and evaluation content


| Document                                                       | Purpose                                |
| -------------------------------------------------------------- | -------------------------------------- |
| [CLOP_DiT_JBHI_Article.md](CLOP_DiT_JBHI_Article.md)           | JBHI-style article content (markdown). |
| [CLOP-DiT_Evaluation_Report.md](CLOP-DiT_Evaluation_Report.md) | Evaluation report narrative.           |


---

## Session and AI-generated markdown (policy)

Session and AI-generated markdown live in `**docs/intermediate_analysis/**` (working outputs: caption analysis, diagnostic reports, implementation summaries, etc.) and `**docs/archive/**` (historical: session reports, cleanup summaries, experiment notes). Both directories are **gitignored**; contents are ephemeral and not required for regeneration or submission. For structure, naming conventions, and cleanup, see [DIRECTORY_CLEANUP_POLICY.md](DIRECTORY_CLEANUP_POLICY.md) and [WRITING_AND_DOCS_POLICY.md](WRITING_AND_DOCS_POLICY.md). Optional: [intermediate_analysis/README.md](intermediate_analysis/README.md) describes the intermediate folder purpose and naming.

---

## Root-level markdown (project)


| File                                           | Purpose                                          |
| ---------------------------------------------- | ------------------------------------------------ |
| [../README.md](../README.md)                   | Project overview, setup, usage.                  |
| [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md) | Reproducibility guide (data, code, environment). |


---

## Quick actions

- **Regenerate all figures:** `bash scripts/regenerate_report.sh` (see [QUICK_START.md](QUICK_START.md)).
- **Verify article figures:** `bash scripts/verify_article_figures.sh`.
- **Check figure status:** [REGENERATION_STATUS.md](REGENERATION_STATUS.md).
- **Add new panels:** [FIGURE_ENHANCEMENT_ROADMAP.md](FIGURE_ENHANCEMENT_ROADMAP.md).

