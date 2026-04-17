# Prose Audit — Intro/Background/Discussion Check — 2026-04-17

Reviewer comments that explicitly asked for stronger intro / background /
discussion prose, mapped against the current v2_revision manuscript at
`revision/manuscripts/v2_revision/clop_dit_genes.tex`.

## Reviewer asks vs. current coverage

| Reviewer ask | Where it should live | Pre-audit verdict | Remediation shipped this round |
|---|---|---|---|
| **R2.1** — "The abstract lacks a statement of research background and fails to explain the specific challenges currently faced in the field of single-cell generation to support the necessity of this study." | `\abstract{}` block | **FAIL** — the abstract was not being rendered at all after the MDPI → neutral-preamble swap (commit `cddd7cb`). The `\mdpiemitheader` emitter tried to call `\begin{abstract}` on a command that had been renewed as a one-argument macro, so the abstract box rendered as empty. Page 1 jumped straight from an empty `Keywords:` line into `1 Introduction`. | **PASS** — (a) Fixed the preamble emitter to render the abstract as an indented quoted paragraph; (b) restored the `\Title / \Author / \address / \corres / \abstract / \keyword` block in the body with the R2.1 background + challenge framing up front ("Single-cell RNA sequencing has transformed the study of cellular heterogeneity, yet most existing single-cell generative models condition on narrow categorical labels..."); (c) retained the rest of the paragraph intact so the rebuttal's headline metrics are still backed by the abstract. |
| **R2.2** — "Five-field template justification" | Introduction §1 (the paragraph that motivates the 5-field prompt) | **PASS** — Introduction §1 already introduces the five-field template as "structured-metadata conditioning rather than free-form language understanding" and Methods §2.1 justifies each field. | No change. |
| **R2.4** — "Figures~1 and~2 readability" | Not prose; figure-source fixes | **PASS** — fonts bumped, spacing widened in commits `ff3309b` + `19ca2c0`. | No change. |
| Reviewer general — "strengthen Discussion where reviewers asked" | Discussion §4 | **PASS** — Discussion already has eight clearly-signposted paragraphs (unified mechanistic picture, variance/covariance deficits, extended validation, ablation/reproducibility/decoder, benchmarking/operating regimes, limitations/scope, strict-OOD, encoder-side localization) synthesising every Lane-A/B/C/D finding. | No change — Discussion prose is adequate. |

## Introduction framing re-check

The Introduction now lines up cleanly with R2.1 and R2.2:

- **Paragraph 1 (lines 160-161)** — frames the inverse problem ("inverse
  problem — generating new cell states from structured biological descriptors
  — remains difficult"); names the preservation challenge ("preserve
  biologically meaningful condition-specific structure").
- **Paragraph 2 (lines 162-163)** — names the exact methodological gap
  ("most existing single-cell generative models condition on categorical
  labels, perturbation labels, or batch covariates"). This matches the
  new abstract opening so the rebuttal-referenced phrasing is consistent
  across the Abstract and the Introduction.
- **Paragraph 3 (lines 164-165)** — positions CellWhisperer as the closest
  discriminative comparator and explicitly states CLOP-DiT targets the
  *generative* setting.
- **Table 1 (lines 166-183)** — fills R2.2's request that the five-field
  template distinction from other methods be made visually explicit.
- **Paragraph 4 (line 185)** — technical ingredients (CLIP/SigLIP + DiT +
  flow matching).
- **Paragraph 5 (line 187)** — three-stage pipeline: CLOP → DiT → scGPT
  decoder with a concrete naming of the five-field template.
- **Paragraph 6 (line 189)** — four contributions.
- **Paragraph 7 (line 191)** — the "informative but not yet a drop-in
  simulator" scope statement, matching the transparency requested by
  reviewers.

Verdict: Introduction is strong enough for R2.1/R2.2 once the Abstract
is restored (which this round does).

## Discussion re-check

Discussion §4 (lines 647-676 in the current tex) enumerates, in order:

1. Unified mechanistic picture (A5 + A3 + A2 + A1 + B3 synthesis) — this
   is the top-of-funnel framing the rebuttal letter cites.
2. Variance and covariance deficits — names Figure~9a explicitly, links
   to the R2.8/R2.9 rebuttal replies.
3. Extended validation — sets up the appendix.
4. Ablation, reproducibility, decoder analysis — consolidates B2 + B4 +
   Lane D findings.
5. Benchmarking and operating regimes — R2.9 Gaussian-baseline framing.
6. Limitations and scope — R3.2 species-stratified result.
7. Strict-OOD generalization (zero-shot) — R3.1 full mechanistic
   discussion (adrenal cortex 0.875, Purkinje 0.00, sibling collapse).
8. Encoder-side localisation — Lane D encoder-trio result.
9. Modular architecture as a path forward — future work.

All eight reviewer-facing lines of evidence have a dedicated discussion
paragraph. No WEAK verdicts remain.

## Files touched this round

- `revision/manuscripts/v2_revision/clop_dit_genes.tex` — restored
  `\Title / \Author / \address / \corres / \abstract / \keyword` block
  with R2.1-strengthened abstract; fixed `\mdpiemitheader` abstract
  renderer.
- `revision/manuscripts/v2_revision/clop_dit_genes.pdf` — rebuilt.
- `revision/manuscripts/diff/clop_dit_genes.diff.{tex,pdf}` — regenerated
  via latexdiff against a neutralised v1 so the tracked-changes
  manuscript now shows the restored title/abstract block as added.
- `revision/manuscripts/diff/clop_dit_genes_tracked_changes.pdf` —
  refreshed copy of the diff PDF.

## Verdict

**PASS.** The only reviewer-facing prose blocker was R2.1 (missing
rendered abstract). That is fixed this round. Introduction and
Discussion prose is otherwise already at the depth the reviewers asked
for, and the rebuttal letter's explicit cross-references (`\msref{...}`)
continue to resolve to real manuscript sections and figures.
