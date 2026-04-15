# v2 Revision — Active Working Tree

This directory is a working copy seeded from `v1_prerevision/` on 2026-04-15.
All revision edits to the manuscript, figures, and cover letter happen here.

## Discipline

- **File names must stay identical to v1** so `latexdiff` can pair them.
- **Do not overwrite v1_prerevision.** It is the immutable diff base.
- **Bundle each edit with its reviewer comment ID** in the commit message
  (e.g., `manuscript: rework abstract motivation (R2.1)`).
- **Every experiment that produces a new figure or number** must also
  update the corresponding `results.md` under `revision/experiments/…`
  before the change is mirrored here.

## Diff workflow reminder

```bash
cd revision/manuscripts
latexdiff v1_prerevision/clop_dit_genes.tex v2_revision/clop_dit_genes.tex \
  > diff/clop_dit_genes.diff.tex
```

See `revision/manuscripts/README.md` for the full pdflatex pipeline.
