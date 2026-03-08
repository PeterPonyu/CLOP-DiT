# Documentation

Complete documentation for the CLOP-DiT project. For the version manifest, see [../VERSIONS.md](../VERSIONS.md).

---

## Quick Start

| Document | Status | Purpose |
|----------|--------|---------|
| **[INDEX.md](INDEX.md)** | Current | Complete documentation organization by category |
| **[QUICK_START.md](QUICK_START.md)** | Current | 9-step pipeline to regenerate figures |
| [DIRECTORY_CLEANUP_POLICY.md](DIRECTORY_CLEANUP_POLICY.md) | Current | Cleanup and organization guidelines |

---

## Document Status Reference

| Category | Documents | Status |
|----------|-----------|--------|
| **Operational** | QUICK_START.md, FIGURE_ORGANIZATION.md, REGENERATION_STATUS.md | Current |
| **Roadmaps** | FIGURE_ENHANCEMENT_ROADMAP.md, BIOLOGICAL_CLAIM_MAP.md, ABLATION_SUITE.md | Current |
| **Policies** | DIRECTORY_CLEANUP_POLICY.md, WRITING_AND_DOCS_POLICY.md, LEGEND_CAPTION_POLICY.md | Current |
| **Article Content** | CLOP_DiT_JBHI_Article.md, CLOP-DiT_Evaluation_Report.md | Current |
| **Archive** | See [archive/](archive/) | Historical session reports |

---

## Document Organization

All documentation is organized in **[INDEX.md](INDEX.md)** by category:

1. **Operational** - Run pipeline, generate figures, directory cleanup
2. **Roadmaps and Plans** - Enhancement roadmaps, biological claims, ablations
3. **Audit and Status** - Completed work audits, submission checklists
4. **Article Content** - Manuscript content and evaluation reports
5. **Session/AI-generated** - Intermediate analysis (intermediate_analysis/), Historical reports (archive/)

---

## Common Actions

- **Regenerate figures**: `bash scripts/regenerate_report.sh` (see [QUICK_START.md](QUICK_START.md))
- **Check figure status**: See [REGENERATION_STATUS.md](REGENERATION_STATUS.md)
- **Clean up workspace**: See [DIRECTORY_CLEANUP_POLICY.md](DIRECTORY_CLEANUP_POLICY.md)
- **Add new figure panels**: See [FIGURE_ENHANCEMENT_ROADMAP.md](FIGURE_ENHANCEMENT_ROADMAP.md)

---

## Archive Location

Old session reports and planning documents are in **[docs/archive/](archive/)**:
- Session reports (2026-03-02 era)
- Cleanup summaries
- Experiment notes
- Superseded enhancement summaries

These are gitignored and not required for regeneration or submission. See [DIRECTORY_CLEANUP_POLICY.md](DIRECTORY_CLEANUP_POLICY.md) for archive policy.
