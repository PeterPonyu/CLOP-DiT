# Panel-Label Refresh Audit + Fix Report — 2026-04-17

**Status:** CLOSED (all LaTeX sources + figures regenerated; submission PDFs rebuilt).

## Scope

The user's prior ralph iteration rendered panel labels as UPPERCASE A/B/C/D via `src/visualization/style.py::add_panel_label` (which calls `.upper()` on the input).  However, the **LaTeX prose and figure captions still wrote lowercase** `(a)`, `panel a`, `panels a--d`, so on-page references did not match the rendered figure panels.  In addition, the user identified six figures that still had layout issues (bar-chart alignment, panel-label overlap, mid-panel whitespace, out-of-axes callouts, drifting gene-name labels).

This audit was conducted on 2026-04-17 and the fixes were applied in the same pass.

## Summary of changes

### 1) LaTeX source — 186 substitutions across 2 files (3 passes)

One-shot Python script (`/tmp/fix_panel_labels.py`) applied the following regex-driven transforms to the main manuscript and the diff-tracked manuscript:

| Pattern                                           | Intent                                                 | Hits |
|---------------------------------------------------|--------------------------------------------------------|------|
| `(\textbf{x})` → `\textbf{X}`                     | drop outer parens, uppercase the single letter         | 67 + 59 |
| `(\textbf{a--c})` → `\textbf{A--C}`               | drop outer parens, uppercase the range                 |  5 + 5  |
| `\textbf{(a--c)}` → `\textbf{A--C}`               | inner-paren caption range                              |  5 + 5  |
| `\textbf{(a,\,b)}` → `\textbf{A, B}`              | comma-pair caption                                     |  2 + 2  |
| `panels a--c` / `panels~a--c` → `panels A--C`     | prose ranges                                           |  4 + 4  |
| `panel a` / `panel~a` → `panel A`                 | single-letter prose                                    | 11 + 11 |
| `\ref{fig:…}a` → `\ref{fig:…}A`                   | inline figure-subpanel reference                       |  1 + 1  |

- **Main manuscript:** `revision/manuscripts/v2_revision/clop_dit_manuscript.tex` — 96 substitutions.
- **Tracked-changes manuscript:** `revision/manuscripts/diff/clop_dit_manuscript.diff.tex` — 90 substitutions.
- **Rebuttal + cover letter:** 0 substitutions (both used numeric indexing only; confirmed clean).

Post-pass verification: `grep` of all four patterns returns 0 matches in all four LaTeX sources.

### 2) Panel-label font size

`PANEL_LABEL_FONT_SIZE` in `src/visualization/style.py` bumped from **16 → 18** to address the reviewer note that panel labels read "slightly small" against the surrounding text.

### 3) Six user-flagged figure fixes

| Figure                              | Problem                                                      | Fix                                                                                                      |
|-------------------------------------|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `figS_lane_a1_knn_family_heatmap`   | right bar chart not row-aligned with heatmap                 | pinned `ax_bar.set_ylim(n-0.5, -0.5)` (matches `imshow` extents); dropped `sharey` to avoid ytick leakage |
| `figS_lane_a2_organism_stratified`  | panel labels overlapped the panel titles and p-value boxes   | moved labels to `x=-0.22, y=1.22` (further outside the axes rect)                                        |
| `figS_lane_b3_forced_scarcity`      | wide empty gap between the two panels                        | set `wspace=0.12` in gridspec + subplots_adjust; pulled baseline text labels 0.2 units inward            |
| `fig09b_gene_gene_correlation` (h)  | bottom outlier labels rendered outside the axes rectangle    | replaced offset-point callouts with interior axes-fraction slots + ConnectionPatch leader lines          |
| `fig09a_variance_matching` (d)      | bottom-right scatter labels drifted outside the axes         | same interior-slots + ConnectionPatch pattern                                                            |
| `fig05a_expression_analysis` (a, c) | panel-a gene labels drifted far from their dots; a/c overlap | switched to `annotate(..., textcoords="offset points")` with per-point direction; `hspace=0.42→0.55`     |

### 4) Pipeline verification

- `scripts/pipeline/run_regeneration.py` ran to completion (27.4s, exit 0).
- `article_delivery.py` verified all 25 article-facing PDFs and delivered them to `articles/figures/` and `revision/manuscripts/v2_revision/figures/`.
- scivcd on the 3 refreshed appendix lane figures: all PASS (source=pdf-only-backfill).
- scivcd on fig09a/fig09b/fig05a (re-run live): remaining warnings are label-density / minimum-font-size issues carried over from the broader figure population — none are introduced by this pass and none affect the 6 fixed call-outs.

### 5) Confidential-info scrub

- `revision/manuscripts/diff/clop_dit_manuscript.diff.tex` previously carried `%DIF DEL /tmp/clop-v1-neutral/...` and `%DIF ADD /home/zeyufu/Desktop/CLOP-DiT/...` LaTeX comments (leaked local filesystem paths).  These comments have been rewritten to venue-neutral text (`v1_prerevision/...`, `v2_revision/...`).
- `pdftotext | grep` across all four submission PDFs: 0 hits for `??`, 0 hits for `/home/zeyufu` or `/tmp/clop`.

### 6) Submission PDFs rebuilt + checksums refreshed

All four PDFs rebuilt from the refreshed sources in the same run; `revision/SUBMISSION_CHECKSUMS.txt` regenerated; `sha256sum -c` verifies all four.

| File                                                         | New SHA256 (prefix) |
|--------------------------------------------------------------|---------------------|
| `revision/manuscripts/v2_revision/clop_dit_manuscript.pdf`   | `1ac89c…`           |
| `revision/manuscripts/diff/clop_dit_manuscript.diff.pdf`     | `67d28b…`           |
| `revision/response_letter/rebuttal_letter.pdf`               | `2da117…`           |
| `revision/response_letter/revision_cover_letter.pdf`         | `1c65f2…`           |

## Python call-site status (informational only)

All 78 `add_panel_label(ax, "a")` / similar call-sites in `src/visualization/*.py` and `scripts/analysis/*.py` remain lowercase.  No change required — `style.py` line 313 applies `.upper()` at render time.

## Not done this pass (deferred)

- Further reducing the broad "minimum_font_size < 7pt" warning population (e.g., fig27_ood_robustness, fig28_marker_completeness, figS01_supplementary_validation) — these are not user-flagged and are driven by the intrinsic density of those figures.  A separate densification pass would be a larger undertaking.
- `remaining_work_roadmap_2026-04-17.md` items other than the language/precision pass captured in `remaining_work_progress_2026-04-17.md`.
