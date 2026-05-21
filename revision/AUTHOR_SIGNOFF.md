# Author sign-off checklist — CLOP-DiT revision v1.0.0

Tag: `v1.0.0-revision`
Branch: `revision/major`
Generated: 2026-04-18

Print, sign at the bottom, and archive with the submission packet.
This file is gitignored; keep it local to your revision workstation.

---

## 1. Submission artifact manifest (against `revision/SUBMISSION_CHECKSUMS.txt`)

| Artifact | SHA-256 (current) | Author verified |
|---|---|---|
| `manuscripts/v2_revision/clop_dit_manuscript.pdf` | `9b7f20644289c691a0dd8f8c0fd49d93fd293b2b8eb6c1227870f3fc02e3524e` | [ ] |
| `manuscripts/diff/clop_dit_manuscript.diff.pdf` | `d66d246ad16fd457dcca65540c0b70dac183371563c432777bd0dedc89226b11` | [ ] |
| `response_letter/rebuttal_letter.pdf` | `0292524ad6dcf271fbe456feca9606ae53c3e7ce1c6587779a0c389c34ac677f` | [ ] |
| `response_letter/revision_cover_letter.pdf` | `f737544dce7d5f804df1fc940e2160efe3e64ca2ff7d33f3e682b642a08a18a2` | [ ] |

Re-verify with: `cd revision && sha256sum -c SUBMISSION_CHECKSUMS.txt`

## 2. Pre-submission content checks

- [ ] **Authorship** — all author names, ORCIDs, affiliations, and corresponding-author email are current on the title page of `clop_dit_manuscript.pdf`.
- [ ] **Funding** — funding acknowledgements list every grant number that supported the work.
- [ ] **Conflicts of interest** — disclosure matches each co-author's up-to-date status.
- [ ] **Data availability statement** — names the correct repository or DOI for the single-cell datasets used (scGPT, human reference).
- [ ] **Code availability statement** — points to the public `PeterPonyu/CLOP-DiT` repo at the tagged release (Zenodo DOI to be inserted once minted).
- [ ] **Ethics / consent** — if human subjects data was re-analyzed, the relevant ethics approval is cited.

## 3. Rebuttal + cover letter accuracy

- [ ] **Reviewer 2 replies (11)** — every numeric claim in `response_letter/rebuttal_letter.pdf` matches the manuscript body.
- [ ] **Reviewer 3 replies (4)** — same check.
- [ ] **Additional Gaussian concern** — response still cites Table 5 / Section 3.12.
- [ ] **Cover letter tone** — no defensive phrasing, accepts scope of revision.

## 4. Figure integrity

- [ ] **No U+2026 ellipsis in labels** — `pdftotext` scan of `results/figures/*.pdf` returns 0 hits (programmatically verified 2026-04-18).
- [ ] **Panel-label case** — all article figures use uppercase A/B/C (verified by ralph #6 + #7).
- [ ] **Caption cross-references** — each `Figure~N(X)` reference in the main text matches an actual panel letter.
- [ ] **Supplementary figure numbering** — figS01/figS02 panels not renumbered since the rebuttal was drafted.
- [ ] **Accepted VCD residuals**:
  - figS01_supplementary_validation: 4 `axes_overflow` flags on `line: Line2D` (novel-types / free-form rows). Bboxes of per-row text labels at the first / last y-position touch the axis edge; ink is *inside* the axis, no content clipped. Architect-reviewed and ruled below reviewer-visibility.
  - fig05b_conditioning_landscape: 2 `axes_overflow` flags on `line: Line2D` (bottom-left). Same bbox-edge artifact; no visible content loss.
  - Previously-flagged `panel_label_overlap` on G/H in fig05b is suppressed post-ralph-#7 via the VCD rule's spine-skip (Spine bbox is the full axes rect, so the rule was detecting the label's position above-left-of-axis against the full spine rectangle rather than against actual ink).

## 5. Final PDF approval

- [ ] **Main manuscript** — scanned cover-to-cover at 100% zoom, no truncated text, no rendering artefacts.
- [ ] **Diff PDF** — addition/deletion markup cleanly rendered; key revision boxes visible.
- [ ] **Rebuttal letter** — reviewer comments typeset correctly; all `\msref{}` links resolve.
- [ ] **Cover letter** — editor addressed by name.

## 6. Reproducibility

- [ ] **Tag exists** — `git show v1.0.0-revision` on `PeterPonyu/CLOP-DiT` resolves.
- [x] **Zenodo DOI** — GitHub release for `v1.0.0-revision` triggered the Zenodo webhook; DOI minted: **`10.5281/zenodo.19643738`** (verified 2026-04-18).
- [ ] **README updated** — citation badge or link points at the Zenodo DOI.

## 7. Leak safety (local-only, not in public repo)

- [ ] Pre-push hook `.git/hooks/pre-push` still symlinks to `scripts/pipeline/check_public_safe.sh` (verified).
- [ ] No `configs/pipeline.yaml`, `revision/manuscripts/**`, or `revision/response_letter/**` content has been accidentally pushed to `PeterPonyu/CLOP-DiT`.

---

**Author sign-off**

| Author | Signature | Date |
|---|---|---|
| Corresponding author |  |  |
| Co-author |  |  |
| Co-author |  |  |

Once every checkbox is ticked and all signatures are in place, the submission packet (`revision/submission_bundle_v1.0.0.tar.gz`) is cleared for upload.

## 2026-04-23 audit addendum

- Audit verdict: TARGETED-FIX-WAVE
- BLOCKERs: 5 (see `revision/figure_fix_reports/AUDIT_SYNTHESIS_2026-04-23.md`)
- Sign-off status: RE-OPENED pending BLOCKER resolution (sections 3 + 5)

### 2026-04-23 BLOCKER resolution (team fix-wave)

- CL-1/CL-2: phantom `Table~S6`–`S13` refs in `revision_cover_letter.tex` replaced with real `\ref{tab:zca-ablation}` / `\ref{tab:ablation}` / `\ref{tab:cond-ablation}` / `\ref{tab:strict-ood}` citations.
- REB-1: `rebuttal_letter.tex:132` off-by-one fixed — `Table~3` → `\msref{Section~3.2, Table~2}` (core-results is Table 2).
- BLOCKER-L1: `clop_dit_manuscript.tex:406` now cites `Figure~\ref{fig:training}, panel~C` (training batch-accuracy curve).
- BLOCKER-L2: `clop_dit_manuscript.tex:716` now cites `Table~\ref{tab:encoder-comparison}`; new compact Table~9 (PCA / scGPT-human / scGPT-pancancer × within-L2 / tightness) inserted at L719.
- Rebuttal letter: `\msref{}` cross-reference macro added at L77 and applied systematically across all 16 reviewer responses (per user request for accurate `\ref`/indexing); cover letter preamble amended with `\usepackage{amsmath}` for `\text{}` rendering.
- Compile verification: manuscript PDF 44 pages, rebuttal PDF 16 pages, cover letter PDF 3 pages — all exit 0, zero undefined refs.
- Tests: `pytest tests/test_article_delivery.py tests/test_visualization.py` → 33 passed, 3 skipped.
- **Sign-off status: CLEARED pending final PDF review of the three rebuilt documents.**

### 2026-04-24 font/layout refresh

- Figure A1 panel E/D collision fixed; all article-facing figure PDFs regenerated and synced to the clean manuscript and tracked-changes figure directories.
- Figure font baseline unified to Arial/Arial Bold; `pdffonts` over `results/figures`, `v2_revision/figures`, and `diff/figures` reports only `ArialMT` and `Arial-BoldMT` subsets.
- Manuscript/rebuttal/cover PDFs rebuilt: clean manuscript 45 pages, diff manuscript 45 pages, rebuttal 19 pages, cover letter 3 pages.
- `revision/SUBMISSION_CHECKSUMS.txt` refreshed to the 2026-04-24 hashes shown in section 1.
- Audit report: `revision/figure_fix_reports/font_layout_logic_audit_2026-04-24.md`.
- **Sign-off status: CLEARED pending final human PDF skim and portal upload checks.**

### 2026-04-23 BLOCKER resolution (team fix-wave)

- CL-1/CL-2: phantom `Table~S6`–`S13` refs in `revision_cover_letter.tex` replaced with real `\ref{tab:zca-ablation}` / `\ref{tab:ablation}` / `\ref{tab:cond-ablation}` / `\ref{tab:strict-ood}` citations.
- REB-1: `rebuttal_letter.tex:132` off-by-one fixed — `Table~3` → `\msref{Section~3.2, Table~2}` (core-results is Table 2).
- BLOCKER-L1: `clop_dit_manuscript.tex:406` now cites `Figure~\ref{fig:training}, panel~C` (training batch-accuracy curve).
- BLOCKER-L2: `clop_dit_manuscript.tex:716` now cites `Table~\ref{tab:encoder-comparison}`; new compact Table~9 (PCA / scGPT-human / scGPT-pancancer × within-L2 / tightness) inserted at L719.
- Rebuttal letter: `\msref{}` cross-reference macro added at L77 and applied systematically across all 16 reviewer responses (per user request for accurate `\ref`/indexing).
- Compile verification: manuscript PDF 44 pages, rebuttal PDF 16 pages, cover letter PDF 3 pages — all exit 0, zero undefined refs.
- Tests: `pytest tests/test_article_delivery.py tests/test_visualization.py` → 33 passed, 3 skipped.
- **Sign-off status: CLEARED pending final PDF review of the three rebuilt documents.**
