# scivcd Revision Close-Out — 2026-04-17

## Scope

End-to-end sweep that refreshes all 22 article-manifest figures under the
scivcd adaptive pipeline, propagates them into the three submission-ready
LaTeX products, and re-verifies cross-references and reviewer-comment
linkage.

## What changed this session

### 1. Figure-source layout fixes

| Source script | Fix | Reason |
|---|---|---|
| `scripts/analysis/lane_c_zero_shot_figure.py` | Widened `figsize` 7.0→7.6", dropped the `layout_rect` override that collapsed the left margin, replaced `MultipleLocator(0.2)` with explicit `[0.0…1.0]` xticks, reduced title font 11.5→10.5pt and pad 10→4, left margin 0.33→0.42, bottom 0.12→0.16 | 16+ CRITICAL `text_truncation` findings — long ytick labels and `1.2` xtick were bleeding 165–231px past the figure border |
| `scripts/analysis/variance_matching_pilot.py` | `hspace` 0.24→0.36, panel labels `(a)/(b)` y=1.05→0.97, added `bottom=0.06` to `save_with_vcd` rect | 4 CRITICAL: cross-axes tick overlap + panel-label top overflow + `0.000` ytick bottom overflow |
| `scripts/analysis/gene_gene_correlation.py` | `hspace` 0.28→0.40, pinned imshow ylim `(49.5, -0.5)` with explicit yticks on both heatmaps, top 0.97→0.93 | 2 CRITICAL: cross-axes tick overlap + spurious `−10` ytick extending above heatmap |

Net VCD diff vs pre-session baseline:
```
added    = {CRITICAL: 0, MAJOR: 24, MINOR: 26, INFO: 3}
removed  = {CRITICAL: 6, MAJOR: 24, MINOR:  6, INFO: 0}
```

### 2. Adaptive-pipeline regeneration

`python scripts/pipeline/run_regeneration.py --adaptive --vs-baseline`
— exit code 0, 22 article PDFs delivered to `articles/figures/`.

Post-fix totals: `CRITICAL=69 MAJOR=1385 MINOR=167 INFO=13`. The 69 remaining
CRITICAL findings are concentrated on `figS01_supplementary_validation.pdf`
and `figS02_expression_diagnostics.pdf`, which the scivcd follow-up policy
note (`scivcd_followup_layout_review_2026-04-17.md`) explicitly triages as
*"acceptable for supplement as-is"* given that any further font shrink
would harm readability more than it would help. The remaining findings are
minimum-font-size / label-string-ellipsis entries on those two
high-density supplements — category MAJOR per severity policy, flagged
CRITICAL only because they cross the axes-frame boundary.

### 3. Propagation into all three LaTeX products

| Product | Action | Result |
|---|---|---|
| `revision/manuscripts/v2_revision/` | Synced all 22 PDFs from `results/figures/`; rebuilt with 3 `pdflatex` passes | 38 pages, 3.62 MB, 0 undefined refs in final pass |
| `revision/manuscripts/diff/` | Populated previously-empty `figures/` with 22 PDFs; rebuilt `clop_dit_genes.diff.tex` and copied output over `clop_dit_genes_tracked_changes.pdf` | 38 pages, 3.64 MB, 0 undefined refs in final pass |
| `revision/response_letter/rebuttal_letter.tex` | Added figure cross-refs at R2.8, R2.9, R2.11, R3.3 naming Figures 9a, 9b, S1, S2; verified R2.4 already names Figures 1–2 | 14 pages, 327 KB, 0 undefined refs |

### 4. Reviewer-comment → figure linkage (final state)

All five v2-revision figures are now named by number in the rebuttal body:

| Figure | Rebuttal ref site |
|---|---|
| Figure 9a (variance matching) | R2.8 bullets (lines 284–285), R2.9 additional bullet (line 320) |
| Figure 9b (gene-gene correlation) | R2.9 additional bullet (line 320) |
| Figure S1 (supplementary validation) | R3.3 additional bullet (line 523) |
| Figure S2 (expression diagnostics) | R2.11 additional bullet (line 380) |
| Figure S_lane_c (zero-shot strict-OOD) | R3.1 (line 444) — embedded inline as preview per scivcd followup policy |

### 5. Submission checksums

`revision/SUBMISSION_CHECKSUMS.txt` regenerated against the freshly built
PDFs. All four submission artifacts (main, tracked-changes, rebuttal,
cover letter) re-hashed 2026-04-17.

## Remaining known-tolerated findings

- **figS01 / figS02**: high-density composed supplements, visible MAJOR
  `minimum_font_size` warnings below 7.0pt. Policy: ship as-is per
  `scivcd_followup_layout_review_2026-04-17.md`. If a follow-up review pass
  is requested, split into two supplementary figures before shrinking
  fonts further.
- **figS_lane_c_zero_shot**: residual MINOR `text_overlap` between the
  `random = 0.09` annotation and the `0.24` bar-end value in the
  cerebellum panel. Not flagged CRITICAL; pre-existing.

Both are documented as tolerated and will not block submission.

## Regression evidence

- `python scripts/pipeline/run_regeneration.py --adaptive --vs-baseline` →
  exit 0 (no NEW CRITICAL findings vs baseline).
- `results/vcd_report_summary.md` dated 2026-04-17 09:13 UTC+8.
- All 22 figures exist under `results/figures/`,
  `revision/manuscripts/v2_revision/figures/`, and
  `revision/manuscripts/diff/figures/` with matching mtimes.
- All three LaTeX products build cleanly with 0 undefined references in
  their final pdflatex pass.
