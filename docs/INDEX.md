# Documentation Index

> Complete organization of all CLOP-DiT documentation.

**Location**: `docs/`
**Last Updated**: 2025-03-08

---

## Quick Navigation

| Need | Go To |
|------|-------|
| Run the pipeline | [operational/QUICK_START.md](operational/QUICK_START.md) |
| Find any document | [operational/INDEX.md](operational/INDEX.md) |
| Plan enhancements | [roadmaps/FIGURE_ENHANCEMENT_ROADMAP.md](roadmaps/FIGURE_ENHANCEMENT_ROADMAP.md) |
| Check figure status | [operational/REGENERATION_STATUS.md](operational/REGENERATION_STATUS.md) |
| Clean workspace | [operational/DIRECTORY_CLEANUP_POLICY.md](operational/DIRECTORY_CLEANUP_POLICY.md) |

---

## Directory Structure

| Directory | Documents | Purpose |
|-----------|-----------|---------|
| [operational/](operational/INDEX.md) | 4 | Active execution guides |
| [roadmaps/](roadmaps/INDEX.md) | 15 | Planning and policy |
| [article/](article/INDEX.md) | 5 | Manuscript content |
| [audit/](audit/INDEX.md) | 3 | Work audits and checklists |
| [archive/](archive/INDEX.md) | 6 | Historical reports |
| [intermediate_analysis/](intermediate_analysis/INDEX.md) | varies | Working outputs |
| [reviewer_reports/](reviewer_reports/) | 9 | Reviewer feedback |

---

## Document Status Reference

### Current Documents (21 total)

**Operational (4)**:
- INDEX.md, QUICK_START.md, DIRECTORY_CLEANUP_POLICY.md, REGENERATION_STATUS.md

**Roadmaps & Planning (15)**:
- BIOLOGICAL_CLAIM_MAP.md, BIOLOGICAL_VALIDATION_EXPANSION.md, ABLATION_SUITE.md
- FIGURE_*.md series (8 documents)
- ROBUSTNESS_EXPERIMENTS.md, ENHANCEMENT_TASKS.md
- FIGURES_9-12_POLICY.md, FIGURES_12-15_POLICY.md

**Article (5)**:
- CLOP_DiT_JBHI_Article.md, CLOP-DiT_Evaluation_Report.md
- ARTICLE_AGENT_PLAN.md, ARTICLE_SUBSECTION_POLICY.md, WRITING_AND_DOCS_POLICY.md

**Audit (3)**:
- AUDIT_CLAUDE_COMPLETED_WORK.md, SUBMISSION_CHECKLIST.md, REVIEWER_CONCERNS_AND_NEXT_STEPS.md

### Archived Documents (15 total)

- docs/archive/ (6 historical session reports)
- docs/intermediate_analysis/ (ephemeral working outputs)
- docs/reviewer_reports/ (9 reviewer reports)

---

## Root-Level Files

| File | Purpose |
|------|---------|
| [README.md](README.md) | Documentation entry point |

---

## Common Actions

### Regenerate Figures
```bash
bash scripts/pipeline/regenerate_report.sh
```
See [operational/QUICK_START.md](operational/QUICK_START.md)

### Check Figure Status
See [operational/REGENERATION_STATUS.md](operational/REGENERATION_STATUS.md)

### Plan New Panels
See [roadmaps/FIGURE_ENHANCEMENT_ROADMAP.md](roadmaps/FIGURE_ENHANCEMENT_ROADMAP.md)

---

## See Also

- [Project README](../README.md) - Project overview
- [VERSIONS.md](../VERSIONS.md) - Version mapping
- [Scripts](../scripts/INDEX.md) - Executable scripts
