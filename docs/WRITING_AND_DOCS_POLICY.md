# Writing and Documentation Policy

This document defines the source of truth for the manuscript, how to place session and AI-generated markdown, and where to find figure-related guidance.

---

## Manuscript source of truth

- **Authoritative manuscript:** [articles/clop_dit_biology.tex](../articles/clop_dit_biology.tex) (MDPI Biology). All narrative and front-matter changes (abstract, sections, captions, author metadata) are made in the `.tex` file.
- **Derivative:** [CLOP_DiT_JBHI_Article.md](CLOP_DiT_JBHI_Article.md) is a JBHI-style markdown version for internal use or review. It can be updated in batches from the .tex or kept as a snapshot; it is not the source of truth.

---

## Submission checklist

Before submission, use the task list in [REVIEWER_CONCERNS_AND_NEXT_STEPS.md](REVIEWER_CONCERNS_AND_NEXT_STEPS.md) or the concise [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md). Items 1–6 (Data Availability, Reproducibility, metadata, ethics, contributions, acknowledgments, conflicts) address the most common causes of editorial return; items 7–11 address reviewer expectations on statistics and methods; items 12–13 are polish.

---

## Style and consistency

- **Journal:** MDPI Biology author instructions apply for format and structure.
- **Terminology:** Use consistently: CLOP-DiT, scGPT, flow matching, BiomedBERT, AdaLN-Zero, classifier-free guidance (CFG). Avoid mixing naming (e.g. "CLOP-DiT" not "CLOP-DIT" or "CLOP DiT").
- **Figure-related text:** For figure production, layout, and limitations, see [FIGURE_ORGANIZATION.md](FIGURE_ORGANIZATION.md) (including the "Figure logic and limitations" section).

---

## Session and AI-generated markdown

- **Working outputs:** Place in **`docs/intermediate_analysis/`** (e.g. caption analysis, diagnostic reports, implementation summaries). Naming: `{TOPIC}_{DESCRIPTION}.md` or `{YYYY-MM-DD}_short_name.md`. See [intermediate_analysis/README.md](intermediate_analysis/README.md) for folder purpose and naming.
- **Historical / archive:** Place in **`docs/archive/`** (e.g. session reports, cleanup summaries, experiment notes). Naming: `YYYY-MM-DD_short_name.md`. Both directories are gitignored; contents are not required for regeneration or submission. See [DIRECTORY_CLEANUP_POLICY.md](DIRECTORY_CLEANUP_POLICY.md) for cleanup and archive policy.

---

## Versioning (optional)

Key docs (e.g. FIGURE_ORGANIZATION, REGENERATION_STATUS, REVIEWER_CONCERNS_AND_NEXT_STEPS) may include a "Last revised: YYYY-MM-DD" or "Doc version" line at the top so it is clear when the plan was last updated.
