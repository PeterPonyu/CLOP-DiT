# Manuscripts — LaTeX diff workspace

This directory holds the two manuscript snapshots used to generate the
revision diff PDF that PeerJ will expect with the resubmission.

## Layout

```
manuscripts/
├── v1_prerevision/    🔒 frozen copy of the submitted version
│   ├── clop_dit_genes.tex
│   ├── clop_dit_genes.pdf    (compiled reference)
│   ├── clop_dit_biology.bbl
│   ├── Definitions/          journal class files
│   ├── figures/              figure PDFs as submitted
│   └── cover_letter_peerj_cs.{tex,pdf}
│
├── v2_revision/       📝 active editing target
│   └── (populated when revision editing starts)
│
└── diff/              auto-generated latexdiff outputs
    └── (populated at end of revision)
```

## Rules

- `v1_prerevision/` is **frozen**. Never edit. It is the diff base.
- All manuscript edits happen in `v2_revision/`. Start it by copying from `v1_prerevision/`:
  ```bash
  cp -r revision/manuscripts/v1_prerevision/. revision/manuscripts/v2_revision/
  ```
- Keep file names identical across `v1` and `v2` so `latexdiff` can match.

## Generating the diff

```bash
cd revision/manuscripts
latexdiff v1_prerevision/clop_dit_genes.tex v2_revision/clop_dit_genes.tex \
  > diff/clop_dit_genes.diff.tex

# Compile the diff in v2_revision/ so Definitions/ and figures/ resolve
cp diff/clop_dit_genes.diff.tex v2_revision/
cd v2_revision
pdflatex clop_dit_genes.diff.tex && bibtex clop_dit_genes.diff && \
  pdflatex clop_dit_genes.diff.tex && pdflatex clop_dit_genes.diff.tex
mv clop_dit_genes.diff.pdf ../diff/
```

## Why manuscripts are copied, not symlinked

Symlinks into `articles/` would break as soon as `articles/` is edited during
revision. A copy guarantees the diff base stays identical to the submitted
PDF regardless of what happens in `articles/`.
