# Article and Paper Content

> Manuscript content, article planning, and writing policy documents.

**Location**: `docs/article/`
**Documents**: 5
**Status**: Current
**Last Updated**: 2025-03-08

---

## Document Index

### Manuscript Content

| Document | Purpose | Format | Size |
|----------|---------|--------|------|
| [CLOP_DiT_JBHI_Article.md](CLOP_DiT_JBHI_Article.md) | JBHI-style article (markdown) | Markdown | 43KB |
| [CLOP-DiT_Evaluation_Report.md](CLOP-DiT_Evaluation_Report.md) | Full evaluation narrative | Markdown | 22KB |

### Article Planning

| Document | Purpose |
|----------|---------|
| [ARTICLE_AGENT_PLAN.md](ARTICLE_AGENT_PLAN.md) | AI-assisted article writing plan |
| [ARTICLE_SUBSECTION_POLICY.md](ARTICLE_SUBSECTION_POLICY.md) | Subsection organization policy |

### Writing Standards

| Document | Purpose |
|----------|---------|
| [WRITING_AND_DOCS_POLICY.md](WRITING_AND_DOCS_POLICY.md) | Source of truth, style guide, links |

---

## LaTeX Article

The primary publication is in LaTeX format:

| File | Location | Purpose |
|------|----------|---------|
| `clop_dit_biology.tex` | `articles/` | Main LaTeX source |
| `clop_dit_biology.pdf` | `articles/` | Compiled PDF |
| Supplementary | `articles/` | Tables, figures |

---

## Building the Article

```bash
# Build LaTeX article
cd articles/
pdflatex clop_dit_biology.tex
bibtex clop_dit_biology
pdflatex clop_dit_biology.tex
pdflatex clop_dit_biology.tex

# Or use the build script
bash scripts/pipeline/build_article.sh
```

---

## Content Sync

Markdown content in this directory should sync with LaTeX:
- `CLOP_DiT_JBHI_Article.md` → `articles/clop_dit_biology.tex`
- Evaluation narrative → Results sections

---

## See Also

- [LaTeX Source](../../articles/)
- [Roadmaps](../roadmaps/INDEX.md) - Figure planning
- [Operational](../operational/INDEX.md) - Execution guides
