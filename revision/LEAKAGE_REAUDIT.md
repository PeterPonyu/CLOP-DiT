# Leakage Re-audit on Deliverable PDFs (US-102)

**Ran:** 2026-04-16
**Method:** `pdftotext` on each PDF → grep for forbidden tokens.

## Deliverables audited

| File | Lines of text |
|---|---:|
| `revision/manuscripts/v2_revision/clop_dit_genes.pdf` | 6325 |
| `revision/manuscripts/diff/clop_dit_genes_tracked_changes.pdf` | 6981 |
| `revision/response_letter/rebuttal_letter.pdf` | 739 |
| `revision/response_letter/revision_cover_letter.pdf` | 98 |

## Token scan

Tokens tested (case-insensitive, substring match):
`Lane, Deferred, proof-of-concept, AI-assisted, Claude, Copilot, REVISION_PATCH,
ralph, ultrawork, omc, TODO, FIXME, XXX, autopilot`

| Token | Manuscript | Marked Diff | Rebuttal | Cover Letter |
|---|---:|---:|---:|---:|
| Lane | 0 | 0 | 0 | 0 |
| Deferred | 0 | 0 | 0 | 0 |
| proof-of-concept | 0 | **2** (strikethrough) | 0 | 0 |
| AI-assisted | 0 | 0 | 0 | 0 |
| Claude | 0 | 0 | 0 | 0 |
| Copilot | 0 | 0 | 0 | 0 |
| REVISION_PATCH | 0 | 0 | 0 | 0 |
| ralph | 0 | 0 | 0 | 0 |
| ultrawork | 0 | 0 | 0 | 0 |
| omc | 0 | 0 | 0 | 0 |
| TODO | 0 | 0 | 0 | 0 |
| FIXME | 0 | 0 | 0 | 0 |
| XXX | 0 | 0 | 0 | 0 |
| autopilot | 0 | 0 | 0 | 0 |

## Interpretation of the 2 `proof-of-concept` hits

Both hits appear in `clop_dit_genes_tracked_changes.pdf` only (the latexdiff-marked
manuscript). By construction, the marked diff renders **deleted v1 text with a strikethrough**
so reviewers can see what was removed. The hits represent text that was taken *out* of the
manuscript during the revision — they are not on the final clean v2 manuscript and
are inherent to the purpose of a tracked-changes PDF.

These 2 hits are therefore **not leakage**: showing what was removed is the marked
manuscript's whole job.

## Verdict

**0 substantive leakage hits across all deliverables.**

The submission packet is safe to upload.
