# Legacy Submission-Variant Audit — 2026-04-17

Inventory of every top-level file or directory in the CLOP-DiT workspace
that is NOT part of the revision-round submission packet. Prepared before
the cleanup commit so every move/delete is pre-authorised in writing.

## Authoritative submission packet (do NOT touch)

| Path | Role |
|---|---|
| `revision/manuscripts/v2_revision/clop_dit_genes.{tex,pdf}` | Main revised manuscript (neutral `\documentclass{article}`) |
| `revision/manuscripts/diff/clop_dit_genes.diff.{tex,pdf}` | latexdiff tracked-changes manuscript |
| `revision/manuscripts/diff/clop_dit_genes_tracked_changes.pdf` | Rebuild-target copy of the diff PDF |
| `revision/manuscripts/v2_revision/cover_letter_peerj_cs.{tex,pdf}` | Revision-round cover letter (PeerJ CS) |
| `revision/response_letter/rebuttal_letter.{tex,pdf}` | Rebuttal letter |
| `revision/response_letter/revision_cover_letter.{tex,pdf}` | Submission cover letter |
| `revision/response_letter/build.sh` | Rebuttal/cover build driver |
| `revision/SUBMISSION_CHECKSUMS.txt` | sha256 manifest for the four submission PDFs |
| `revision/manuscripts/v2_revision/figures/` + `revision/manuscripts/diff/figures/` | Figure PDFs referenced by the manuscripts |
| `revision/manuscripts/v2_revision/Definitions/` | Neutral preamble (per commit `cddd7cb`) |

## Stale variants — tracked vs. gitignored

Re-check of `.gitignore` (lines 82-121) shows that **almost every** candidate path
is already gitignored and therefore untracked. Only `figure_subpanels_archive/`
carries live git history. The cleanup therefore splits into two buckets:

### Bucket A — untracked on-disk clutter (plain `rm`, no commit content)

These paths exist only on the local working tree. They are already ignored by
`.gitignore` so `git status` will not show them as removed. Deleting them from
disk cleans the workspace without producing any diff to commit.

| Path | Why it misleads | Action |
|---|---|---|
| `articles/` | Stale `clop_dit_biology`/`clop_dit_genes` build tree from the pre-revision round; own PDFs diverge from the submission packet | `rm -rf` |
| `articles_elsevier/` | Pre-revision Elsevier build tree (venue no longer targeted) | `rm -rf` |
| `articles_elsevier copy/` | Literal `cp -r` duplicate of `articles_elsevier/` | `rm -rf` |
| `articles.zip` | Frozen archive of `articles/` from an earlier round | `rm` |
| `LaTeX_Template_for_PeerJ_Journal_Submissions/` | Raw PeerJ CS template download; the revision workspace uses the neutral preamble instead | `rm -rf` |
| `cover_letter.{md,pdf}` at repo root | Pre-revision cover letter superseded by `revision/response_letter/revision_cover_letter.*` | `rm` |
| `cover_letter_peerj_cs.{md,pdf,tex}` at repo root | Earlier PeerJ CS cover-letter draft; the live copy lives under `revision/manuscripts/v2_revision/cover_letter_peerj_cs.{tex,pdf}` | `rm` |
| `revision/manuscripts/v2_revision/clop_dit_genes.tex.mdpi_backup` | Pre-`cddd7cb` MDPI-template backup (untracked); the neutral-template `clop_dit_genes.tex` is the authoritative version | `rm` (history is recoverable via `git log -p` on the parent `.tex` and the `cddd7cb` swap commit) |

### Bucket B — tracked in git history (`git rm -r`, produces commit content)

| Path | Why it misleads | Action |
|---|---|---|
| `figure_subpanels_archive/` | 18 pre-revision figure subpanel PDFs plus `results.pdf` and `view.jpg`. No code references them (confirmed via `grep -r figure_subpanels_archive src scripts`); they duplicate the canonical PDFs under `results/figures/`. | `git rm -r figure_subpanels_archive` |

### Not touched (authoritative submission packet)

- `revision/manuscripts/v2_revision/{clop_dit_genes.tex, clop_dit_genes.pdf, cover_letter_peerj_cs.*, figures/, Definitions/}`
- `revision/manuscripts/diff/{clop_dit_genes.diff.tex, clop_dit_genes.diff.pdf, clop_dit_genes_tracked_changes.pdf, figures/}`
- `revision/response_letter/{rebuttal_letter.*, revision_cover_letter.*, build.sh}`
- `revision/SUBMISSION_CHECKSUMS.txt`

### Why not move under `archive/submission_variants/`?

An earlier plan was to `git mv` each tree into a `archive/submission_variants/`
directory with an explanatory README. The corrected analysis above shows that
introducing a new archive hierarchy would actually re-track directories that are
currently gitignored — net result: larger repo, new trees to document, no real
gain. The better answer is a clean delete. The commit message and this audit
report together leave a sufficient paper trail.

## Verdict

Inventory complete. Ready to execute in US-T02.
