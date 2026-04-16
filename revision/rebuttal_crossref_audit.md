# Rebuttal → Manuscript Crossref Audit (US-101)

**Ran:** 2026-04-16
**Rebuttal:** `revision/response_letter/rebuttal_letter.tex`
**Manuscript:** `revision/manuscripts/v2_revision/clop_dit_genes.tex`

## Method

Extracted every `\msref{...}` target from the rebuttal and verified each references
either (a) an actual section/subsection in the manuscript, or (b) an actual `\label{}`
on a table or figure, or (c) the Appendix. Ran the equivalent check on the cover
letter for cross-reference consistency.

## Verified section targets

| Reference | Manuscript anchor | Status |
|---|---|---|
| `Abstract` | `\begin{abstract}` block, lines 151–157 | PASS |
| `Section~2.1` | \subsection "Datasets and preprocessing" (line 205) — contains the 5-field template description | PASS |
| `Section~3.4` | \subsection "Per-Type Generation Quality and Text–Cell Alignment" (line 435) | PASS |
| `Section~3.5` | \subsection "Gene-Level Fidelity" (line 445) | PASS |
| `Section~3.10` | \subsection "CLOP Ablation Study" (line 537) | PASS |
| `Section~3.11` | \subsection "Rare Cell Augmentation" (line 578) | PASS |
| `Section~3.12` | \subsection "Conditioning Field Ablation and Swap-Label Permutation Test" (line 586) — swap-label evidence for R2.3 | PASS |
| `Section~3.13` | \subsection "Strict-OOD Generalization (Zero-Shot)" (line 613) | PASS |
| `Discussion` | \section{Discussion} (line 647) | PASS |
| `Results` | \section{Results} (line 352) | PASS |
| `Figures~1--2` | `\label{fig:architecture}` (line 202), `\label{fig:training}` (line 373) | PASS |
| `Appendix~\ref{app:suppl-figs}` | `\label{app:suppl-figs}` (line 1072) | PASS |

## Issues identified and remediated

| Issue | Remediation |
|---|---|
| Rebuttal cited `Supplementary Table~S6` (KNN taxonomy) — not present in manuscript | Deleted claim; evidence already inline in rebuttal's revisionupdate box |
| Rebuttal cited `Supplementary Table~S7` (organism split) — not present | Deleted claim; analysis reported inline in §3.4 |
| Rebuttal cited `Supplementary Table~S8` (abundance fidelity) — not present | Deleted claim |
| Rebuttal cited `Supplementary Table~S9` (variance baselines) — not present | Replaced with "table reproduced above" referencing the revisionupdate's inline table |
| Rebuttal cited `Supplementary Table~S10` (ZCA ablation) — manuscript has `Table~\ref{tab:zca-ablation}` in main text | Rewrote reference to cite the actual main-text table |
| Rebuttal cited `Supplementary Table~S11` (augmentation sweep) — not present | Deleted claim; table reproduced in revisionupdate box |
| Rebuttal cited `Supplementary Table~S12` (bridge) — not present | Deleted claim; table reproduced in revisionupdate box |
| Rebuttal cited `Supplementary Table~S13` (strict-OOD) — manuscript has `Table~\ref{tab:strict-ood}` in main text | Rewrote reference to cite the actual main-text table |
| Cover letter carried the same 8 phantom table references | Rewrote `Key Manuscript Changes` block to cite actual sections and tables |
| Appendix changelog table cited S6–S13 | Rewrote each row to cite actual section/table labels |

## Result

All `\msref{...}` targets in the edited rebuttal now resolve to real manuscript anchors.
Rebuttal will rebuild and present accurate, verifiable cross-references to the manuscript
under review. No supplementary-table fabrication risk remains in the cover letter either.

**Verdict:** PASS after remediation.
