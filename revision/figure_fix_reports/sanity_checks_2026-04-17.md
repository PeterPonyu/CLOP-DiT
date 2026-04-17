# Sanity Checks — 2026-04-17

## Remote vs. local

- Current HEAD: `127a0da` (pre-deepening); after this commit, will advance again.
- `origin/revision/major`: already at `127a0da` before this commit was made — no commits pending on push.
- `git fetch origin` after the fact: 0 new refs.
- Tag `submission-2026-04-17`: `612b0cc` (one anchor commit back, at the original "SHIP-READY" snapshot). Intentionally NOT moved — keeps the original anchor recoverable. If the user wants a fresh anchor after this deepening, tag `submission-2026-04-17-r2` on the new HEAD.

## Baseline integrity

- `revision/manuscripts/v1_prerevision/clop_dit_manuscript.tex` sha256: `bfc926a835d775da5603e6c22b5d613258e914575041a3d1d3602e9afbc2ae34`
- Tag `pre-revision-2026-04-15` exists on the branch; `git show pre-revision-2026-04-15 --stat` confirms the baseline commit is reachable.
- `v1_prerevision` `.tex` hash is stable across every ralph run — no accidental mutation of the baseline.

## Submission checksums

`sha256sum -c revision/SUBMISSION_CHECKSUMS.txt` returns OK on all 4 PDFs at the time of this commit. The checksum file now references:

- `revision/manuscripts/v2_revision/clop_dit_manuscript.pdf`
- `revision/manuscripts/diff/clop_dit_manuscript.diff.pdf` (canonical tracked-changes file, post-dedupe)
- `revision/response_letter/rebuttal_letter.pdf`
- `revision/response_letter/revision_cover_letter.pdf`

## Literal `??` in submission PDFs

`pdftotext <pdf> - | grep -c "??"` returns 0 on every submission PDF after this commit. The new pre-push hook (`scripts/hooks/pre-push`, symlinked into `.git/hooks/pre-push`) enforces this gate on every push; bypass requires `git push --no-verify`.
