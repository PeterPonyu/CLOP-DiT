# Manuscripts — LaTeX diff workspace (local-only)

This directory holds the two manuscript snapshots used to generate the
revision diff PDF for the resubmission.

**The entire `manuscripts/` subtree below this file is gitignored.**
It exists only on disk. That keeps venue-specific LaTeX class files,
cover-letter text, and any in-progress manuscript drafts out of the
public repository.

## Layout

```
manuscripts/
├── v1_prerevision/    🔒 frozen copy of the submitted version
│   ├── <main>.tex
│   ├── <main>.pdf               (compiled reference)
│   ├── Definitions/             class files
│   ├── figures/                 figure PDFs as submitted
│   └── cover_letter.tex         (local-only)
│
├── v2_revision/       📝 active editing target
│   └── (populated when revision editing starts)
│
└── diff/              auto-generated latexdiff outputs
    └── (populated at end of revision)
```

## Rules

- `v1_prerevision/` is **frozen**. Never edit. It is the diff base.
- All manuscript edits happen in `v2_revision/`. Start it by copying
  from `v1_prerevision/`:
  ```bash
  cp -r revision/manuscripts/v1_prerevision/. revision/manuscripts/v2_revision/
  ```
- Keep file names identical across `v1` and `v2` so `latexdiff` can
  match.

## Generating the diff

```bash
cd revision/manuscripts
latexdiff v1_prerevision/<main>.tex v2_revision/<main>.tex \
  > diff/<main>.diff.tex

# Compile the diff in v2_revision/ so Definitions/ and figures/ resolve
cp diff/<main>.diff.tex v2_revision/
cd v2_revision
pdflatex <main>.diff.tex && bibtex <main>.diff && \
  pdflatex <main>.diff.tex && pdflatex <main>.diff.tex
mv <main>.diff.pdf ../diff/
```

## Why these files stay off the public repository

Keeping the manuscript sources local-only prevents venue-specific
templates, cover letters, and in-progress revisions from ever reaching
the public remote. The frozen SHA-256 manifest in
`revision/prerevision_baseline/artifact_hashes.txt` can be extended
locally to cover these files if cross-machine integrity verification
becomes necessary.
