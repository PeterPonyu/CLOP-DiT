# Deep Cleanup Inventory — 2026-04-17

Depth 4-5 audit of duplicates, outdated venue markers, and intermediate
AI-generated artefacts under `revision/` and the repo root. Every path is
classified as `TRACKED` (needs `git rm`) or `UNTRACKED` (needs plain `rm`
per the active `.gitignore`).

## Authoritative submission packet — DO NOT TOUCH

| Path | Role |
|---|---|
| `revision/manuscripts/v2_revision/clop_dit_genes.{tex,pdf}` | Main revised manuscript (Array/Elsevier target) |
| `revision/manuscripts/diff/clop_dit_genes.diff.{tex,pdf}` | latexdiff tracked-changes manuscript |
| `revision/manuscripts/diff/clop_dit_genes_tracked_changes.pdf` | Reviewer-facing copy of the diff PDF |
| `revision/manuscripts/v2_revision/figures/*.pdf` | 22 article figures |
| `revision/response_letter/{rebuttal_letter,revision_cover_letter}.{tex,pdf}` | Rebuttal + cover letter |
| `revision/response_letter/build.sh` | Rebuttal build driver |
| `revision/SUBMISSION_CHECKSUMS.txt` | sha256 manifest |
| `revision/experiments/**` | Lane A/B/C/D raw analysis artefacts |
| `revision/{REVIEWER_RESPONSE_SYNTHESIS,SUBMISSION_READINESS,REVISION_LOG,LEAKAGE_REAUDIT,rebuttal_crossref_audit}.md` | Canonical audit docs |

## Bucket A — Duplicates / misplaced (TRACKED: `git rm`)

| Path | Why it is stale | Action |
|---|---|---|
| `revision/figure_fix_reports/fig25_cross_dataset.md` | Old figure numbering — `article_delivery.py` manifest uses fig01a..fig09b / figS01..figS02 / figS_lane_c only. No fig25 exists. | `git rm` |
| `revision/figure_fix_reports/fig26_expanded_de.md` | Same — no fig26 in the manifest | `git rm` |
| `revision/figure_fix_reports/fig27_ood_robustness.md` | Same | `git rm` |
| `revision/figure_fix_reports/fig28_marker_completeness.md` | Same | `git rm` |
| `revision/figure_fix_reports/fig29_embedding_augmentation.md` | Same | `git rm` |
| `revision/figure_fix_reports/fig30_validation_summary.md` | Same | `git rm` |

## Bucket B — Untracked gitignored clutter (plain `rm`)

| Path | Why it is stale | Action |
|---|---|---|
| Root `AUDIT_INDEX.md` | Outdated (Mar 24); superseded by `revision/SUBMISSION_READINESS.md` | `rm` |
| Root `IMPROVEMENT_PLAN.md` | Outdated (Mar 24); revision-planning superseded by `revision/REVISION_LOG.md` | `rm` |
| Root `SUBMISSION_AUDIT_PEERJ_CS.md` | Wrong venue — current submission targets Array/Elsevier; supersded by `revision/SUBMISSION_READINESS.md` + `revision/figure_fix_reports/submission_audit_2026-04-17.md` | `rm` |
| Root `ZENODO_WORKFLOW.md` | Pre-revision Zenodo workflow; superseded by `revision/ZENODO_RELEASE_NOTES.md` | `rm` |
| `revision/manuscripts/v1_prerevision/cover_letter_peerj_cs.{tex,pdf}` | Pre-revision PeerJ CS cover; not a submission artefact; v1 kept only as latexdiff reference | `rm` |
| `revision/manuscripts/v1_prerevision/Definitions/` | MDPI class + logos from the old MDPI template; v1 tex still uses this so we KEEP it for now (latexdiff reference builds need it) | **KEEP** |
| `revision/manuscripts/v2_revision/Definitions/` | Orphaned — v2 uses the neutral preamble; `Definitions/` refs in tex are all commented out (lines 1294, 1298) | `rm -rf` |
| `revision/manuscripts/v2_revision/clop_dit_genes.diff.{tex,pdf,aux,log,out}` | Misplaced build output — authoritative diff lives under `revision/manuscripts/diff/`; nothing in the build chain writes here on purpose | `rm` |
| `revision/manuscripts/diff/clop_dit_genes_tracked_changes.tex` | Stale copy of `clop_dit_genes.diff.tex` from an earlier round; the canonical tracked-changes source IS `clop_dit_genes.diff.tex`, the `.._tracked_changes.pdf` is a rename of the same build output | `rm` |
| `revision/manuscripts/diff/clop_dit_genes.diff.{aux,log,out}` | Intermediate build artefacts from pdflatex | `rm` (keeps `.tex` + `.pdf`) |

## Bucket C — Intermediate AI-generated artefacts

None found at depth 4-5 beyond the pdflatex intermediate files already
covered in Bucket B. The experiments under `revision/experiments/` are
data artefacts (JSON, CSV, markdown SYNTHESIS), not AI slop.

## Post-cleanup v2_revision tree (target state)

```
revision/manuscripts/v2_revision/
├── clop_dit_genes.tex       # neutral-preamble source
├── clop_dit_genes.pdf        # rebuilt output
└── figures/                  # 22 article figures (PDF only)
```

No more Definitions/, no more stale .diff.*, no more duplicate
cover-letter or backup files.

## Verdict

Inventory complete. Ready to execute in US-D02 with `git rm` for Bucket A
and plain `rm` for Bucket B.
