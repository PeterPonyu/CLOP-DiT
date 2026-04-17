# Submission Product Audit — 2026-04-17

Audit of the uncommitted CLOP-DiT submission-product changes produced by the
prior scivcd revision sweep (closeout at
`revision/figure_fix_reports/scivcd_revision_closeout_2026-04-17.md`). Run
before committing the changes to `revision/major`.

## Acceptance criteria (US-S01)

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | `SUBMISSION_CHECKSUMS.txt` matches live sha256 of the four submission PDFs | **PASS** | `sha256sum -c revision/SUBMISSION_CHECKSUMS.txt` → all 4 rows `OK` |
| 2 | Every new `\includegraphics` in `rebuttal_letter.tex` points to a file that exists | **PASS** | Single new `\includegraphics` at line 444 → `../manuscripts/v2_revision/figures/figS_lane_c_zero_shot.pdf` exists (22/22 figures present in delivery dir) |
| 3 | `rebuttal_crossref_audit.md` embedding-check matches what `rebuttal_letter.tex` actually references | **PASS** | Audit addendum names fig09a, fig09b, figS01, figS02, figS_lane_c; rebuttal body references Figure 9a (R2.8 / R2.9), 9b (R2.9), S1 (R3.3), S2 (R2.11), and embeds figS_lane_c (R3.1). Manuscript crossref labels `fig:var-pilot` / `fig:strict-ood` resolve (final `pdflatex` pass reported 0 undefined refs in the closeout). |
| 4 | `REVISION_LOG.md` Last-updated line names the 2026-04-17 scivcd sweep AND preserves the prior 2026-04-16 Lane C entry | **PASS** | Line 9 = new 2026-04-17 scivcd-sweep update; line 11 = prior 2026-04-16 Lane C entry kept verbatim under `Last updated (prior)` |
| 5 | Submission audit report exists with explicit PASS/FAIL per criterion | **PASS** | This file |

## Inventory of uncommitted changes being audited

### Revision-docs + rebuttal sync group (commit 1 candidate)

| File | Kind | Summary |
|---|---|---|
| `revision/REVISION_LOG.md` | edit | Added 2026-04-17 scivcd-sweep entry; kept prior 2026-04-16 Lane C entry as `Last updated (prior)` |
| `revision/SUBMISSION_CHECKSUMS.txt` | edit | Refreshed sha256 hashes for all four submission PDFs after 2026-04-17 rebuild |
| `revision/rebuttal_crossref_audit.md` | edit | Added 2026-04-17 embedding check covering manuscript + diff + rebuttal figure references |
| `revision/response_letter/rebuttal_letter.tex` | edit | Added `graphicx` include; added explicit Figure 9a/9b/S1/S2 names in R2.8/R2.9/R2.11/R3.3; embedded `figS_lane_c_zero_shot.pdf` preview under R3.1; replaced `\ref{...}` cross-manuscript references with hardcoded "Table 8 / Figure 9 / Appendix A.1" to eliminate `??` in the reply PDF |
| `revision/figure_fix_reports/scivcd_revision_closeout_2026-04-17.md` | new | End-to-end closeout for the scivcd adaptive-sweep revision run |
| `revision/figure_fix_reports/scivcd_followup_layout_review_2026-04-17.md` | new | Visual follow-up policy note (fig09a/9b/S01/S02/S_lane_c); source of the annotation-crowding rule being upstreamed to scivcd |

### Figure-script refinements group (commit 2 candidate)

| File | Kind | Summary |
|---|---|---|
| `scripts/analysis/lane_c_zero_shot_figure.py` | edit | Widened figsize 7.0→7.6"; removed collapsing layout_rect override; explicit xticks 0.0→1.0; smaller title; larger left/bottom margins — removes 16+ CRITICAL text_truncation findings |
| `scripts/analysis/variance_matching_pilot.py` | edit | hspace 0.24→0.36; panel-label y=1.05→0.97; bottom=0.06 on save_with_vcd rect — removes 4 CRITICAL cross-axes + panel-label overflows |
| `scripts/analysis/gene_gene_correlation.py` | edit | hspace 0.28→0.40; pinned imshow ylim `(49.5, -0.5)` with explicit yticks on both heatmaps; top 0.97→0.93 — removes 2 CRITICAL ytick/axes-overflow findings |
| `src/visualization/article_delivery.py` | edit | Article-delivery manifest updated to include the fresh 22-figure set for the v2_revision + diff manuscript dirs |

## Notes & minor imprecisions

- The rebuttal letter names the strict-OOD figure as "Figure 9" in the main
  text for R3.1 (line 456). The manuscript label is `fig:strict-ood`, which in
  the final built PDF is numbered by LaTeX — the final `pdflatex` pass
  confirmed 0 undefined refs, so the hardcoded numeral resolves to an actual
  figure. Equivalent statement in the main text uses the `\ref{fig:strict-ood}`
  form.
- "Figures 9a/9b" is shorthand for the two source-file stems
  (`fig09a_variance_matching.pdf`, `fig09b_gene_gene_correlation.pdf`) stacked
  inside one manuscript float labelled with both `\label{fig:var-pilot}` and
  `\label{fig:gene-gene-corr}`. This matches reviewer-facing convention and
  does not introduce build warnings.

## Verdict

**PASS.** The uncommitted changes are internally consistent, the live PDFs match
the refreshed checksums, and every referenced figure file exists in the delivery
directory. Safe to commit to `revision/major`.
