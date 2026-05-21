# Venue preflight checklist — CLOP-DiT revision v1.0.0

Tag: `v1.0.0-revision`
Generated: 2026-04-18

Fill in the venue column once the target journal is confirmed and re-run the checklist.
This file is gitignored.

---

## Current manuscript metrics (auto-extracted 2026-04-18)

| Metric | Value | Venue limit | OK? |
|---|---|---|---|
| Page count (typeset PDF) | 43 | `__ ` | [ ] |
| Main-text figures | 19 | `__ ` | [ ] |
| Supplementary figures | 10 | `__ ` | [ ] |
| Delivered article-figure count (manifest) | 25 | `__ ` | [ ] |
| Main-text file size | 3,748 kB | `__ ` | [ ] |
| Diff PDF file size | 3,785 kB | not checked | [ ] |

Re-extract with:
```bash
pdfinfo revision/manuscripts/v2_revision/clop_dit_manuscript.pdf | grep -E 'Pages|File size'
ls results/figures/fig0*.pdf | wc -l    # main
ls results/figures/figS*.pdf | wc -l    # supplementary
```

## 1. Manuscript structure

- [ ] **Abstract** — fits venue word limit (recount after any late edits).
- [ ] **Sections** — Introduction / Methods / Results / Discussion / Conclusion (or venue-specific order).
- [ ] **Keywords** — at least the venue minimum supplied on the title page.
- [ ] **Footnotes** — moved to Methods or an endnote section if the venue forbids footnotes.

## 2. Figures

- [ ] **Resolution** — every embedded PDF figure renders ≥ 300 dpi at printed size (effective-DPI VCD flags cleared for the main-text set).
- [ ] **Color-blind safety** — palette passes the Nature-style confusability check (VCD `colorblind_confusable` residuals are below target and have been acknowledged).
- [ ] **Font sizes** — tick labels, panel labels, and annotations render ≥ 7 pt at printed size where feasible (tracked in VCD `minimum_font_size` residuals).
- [ ] **Panel labels** — uppercase A–Z, consistent style (verified post-ralph #7).
- [ ] **No U+2026** — zero ellipsis characters in rendered PDF text (verified via pdftotext).

## 3. References and citations

- [ ] **Bibliography style** — matches venue (e.g. Nature, ACS, APA).
- [ ] **Cross-references** — `pdflatex` run shows 0 "undefined reference" warnings in the log.
- [ ] **Self-references** — pre-print/arXiv links updated to the tagged release.

## 4. Supplementary materials

- [ ] **SI numbering** — figures use `figS_` prefix only; no misnamed `supp-` or `ext-data-`.
- [ ] **SI PDF separated** — if the venue requires a separate SI PDF, split the supplementary appendix into its own file.
- [ ] **Large data tables** — uploaded as `.csv` / `.xlsx` supplements rather than inline tables if over the venue's row limit.

## 5. Declarations

- [ ] **Data availability** — statement present; DOI or accession inserted.
- [ ] **Code availability** — statement present; Zenodo DOI inserted once the v1.0.0-revision release triggers the webhook.
- [ ] **Competing interests** — declared per venue template.
- [ ] **Author contributions** — matches CRediT taxonomy if venue requires it.
- [ ] **Funding** — every grant number listed.

## 6. Identifiers

- [ ] **ORCID** — each author's ORCID present on the title page.
- [ ] **Data DOI** — recorded here: `__________________`.
- [ ] **Code DOI (Zenodo)** — recorded here: `__________________`.
- [ ] **Preprint DOI (if applicable)** — recorded here: `__________________`.

## 7. LaTeX template adherence (venue-specific)

- [ ] **Document class** — matches venue-provided `.cls`.
- [ ] **Margins / column width** — untouched from venue template.
- [ ] **Title-page block** — uses venue macros for authors and affiliations.
- [ ] **Line numbering** — `linenumbers` package active if venue requires it for review copy.

## 8. Color-figure / open-access fees

- [ ] **Color charges** — confirmed with venue; pay-or-switch-to-grayscale decision recorded.
- [ ] **Open-access fee** — plan on file.

## 9. Rebuttal + cover-letter conventions

- [ ] **Rebuttal format** — point-by-point matches venue style (itemized reviewer concerns + responses).
- [ ] **Cover letter** — editor addressed by name; changes list matches rebuttal items.
- [ ] **Word limits** — checked for rebuttal if venue imposes one.

## 10. Final upload dry run

- [ ] Uploaded the submission bundle to the venue test harness (or manuscript-tracking system staging).
- [ ] Verified every file's SHA-256 matches `revision/SUBMISSION_CHECKSUMS.txt` after the upload round-trip.
- [ ] Recorded the submission-system tracking number here: `__________________`.

---

Once every item is checked, proceed with the real submission via the venue's portal.

## 2026-04-23 audit addendum

- Audit verdict: TARGETED-FIX-WAVE
- BLOCKERs: 5 (see `revision/figure_fix_reports/AUDIT_SYNTHESIS_2026-04-23.md`)
- Preflight status: UNCHANGED on mechanical checks (cross-refs + figure files + status tags all CLEAN); content-side blockers sit in rebuttal + cover letter + two manuscript paragraphs.

### 2026-04-23 BLOCKER resolution (team fix-wave)

- All 5 BLOCKERs closed: CL-1, CL-2 (cover-letter phantom tables), REB-1 (rebuttal Table 2 off-by-one), BLOCKER-L1 (L406 fig:training cite), BLOCKER-L2 (L716 tab:encoder-comparison cite + new Table 9).
- Rebuttal reworked to use `\msref{...}` systematically with `% target: <label>` tracking comments across all 16 reviewer responses.
- Compile gates: manuscript 44 pages, rebuttal 16 pages, cover letter 3 pages — all exit 0. Cross-refs resolve (zero undefined).
- Mechanical preflight re-run: still CLEAN (refs / figures / status tags).
- **Preflight status: CLEARED for submission pending final PDF sanity-read.**
