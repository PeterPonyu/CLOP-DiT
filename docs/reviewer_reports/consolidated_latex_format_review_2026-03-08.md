# Consolidated LaTeX Format Review Report

**Date:** 2026-03-08
**Source:** 8 parallel Cursor worktree reviews (bbh, flz, ivl, ode, qir, swi, tpz, zzo)
**Status:** Merged and de-duplicated; actions executed in `clop_dit_biology.tex`

---

## Summary

Eight independent LaTeX formatting reviews were conducted in parallel across Cursor worktrees. All 8 reports unanimously identified the same top-priority issues. This document consolidates the unique findings and tracks which actions have been executed.

## Actions Completed

### Priority 1 — HIGH (all 8 reports agreed)

| Action | Status | Details |
|--------|--------|---------|
| Replace `[H]` with `[!htbp]` for all 17 main-body figures | **DONE** | All `\begin{figure}[H]` replaced with `\begin{figure}[!htbp]` |
| Replace `[H]` with `[!htbp]` for all 3 main-body tables | **DONE** | All `\begin{table}[H]` replaced with `\begin{table}[!htbp]` |
| Add `\raggedbottom` to preamble | **DONE** | Added after `\usepackage{longtable}` |
| Standardize appendix float specifiers | **DONE** | Appendix tables already use `[htbp]`; consistent with `[!htbp]` main body |

### Priority 2 — MEDIUM-HIGH (3–5 reports)

| Action | Status | Details |
|--------|--------|---------|
| Wrap equations in `linenomath` | **DONE** | All 6 equations wrapped for submit-mode line numbering |
| Add `\FloatBarrier` controls | **DONE** | `placeins` package added; barriers at Baseline and Downstream sections |
| Add `\usepackage{placeins}` | **DONE** | Added to preamble |

### Priority 3 — MEDIUM (2–3 reports)

| Action | Status | Details |
|--------|--------|---------|
| Fix `\headheight` warning | DEFERRED | MDPI class may normalize this in production |
| Fix hyperref PDF-string warnings | DEFERRED | Would require `\texorpdfstring` wrappers; low visual impact |
| Remove manual `\vspace{6pt}` | **DONE** | Removed before supplementary block |
| Longtable column improvement | DEFERRED | Current `p{5.5cm}` is adequate |
| Overfull `\hbox` warnings | DEFERRED | Requires line-by-line text reflow |

### Priority 4–5 — LOW (1–2 reports; polish)

| Action | Status | Notes |
|--------|--------|-------|
| `siunitx` package | DEFERRED | Not critical for submission; MDPI compatibility uncertain |
| `\cref`/`\Cref` usage | DEFERRED | Already loaded by MDPI class; manual refs are consistent |
| En-dash for ranges | DEFERRED | Minor typography polish |
| External `.bib` file | DEFERRED | Current `thebibliography` is valid for submission |
| `stfloats` package | DEFERRED | Alternative to `\raggedbottom`; not needed since raggedbottom applied |

---

## Unique Findings Per Worktree

### bbh
- Overfull `\hbox` warnings from long `\texttt{}` tokens
- `\headheight` too small warning
- Hyperref PDF-string warnings from math in metadata
- Verification checklist: rebuild twice with `latexmk -pdf`

### flz (most detailed, 578 lines)
- `linenomath` wrapping for equations (DONE)
- Non-breaking spaces for numeric content (`220{,}304~cells`)
- En-dash usage for ranges
- Table `\small` sizing inconsistency
- Graded overall formatting "B+"

### ivl
- Compile pipeline fragility: missing figure PDFs block build
- Cautious about `\raggedbottom` for MDPI submission
- Float parameter tuning values if defaults suboptimal

### ode
- `\FloatBarrier` at Results subsection ends (DONE — selective)
- Longtable spacing: `\LTpre`/`\LTpost`
- `\cref`/`\Cref` from cleveref
- `\usepackage[hyphens]{url}`
- `\bibsep` for bibliography spacing

### qir
- Three-phase diff strategy: global baseline → manual refinement → raggedbottom QA
- Section-flow control missing (floats used as workaround)
- `[!t]` for pipeline overview figure

### swi
- Complete enumeration of all 17 figure lines and 6 table lines
- `\unskip` removal after figures if switching from `[H]`
- Missing `\centering` in appendix tables S2, S3
- "When Paper is Accepted" checklist

### tpz
- `linenomath` at HIGH priority (DONE)
- `siunitx` with `S` column type for decimal alignment
- Math typography: explicit L2 norm notation
- Implementation regex patterns for find/replace

### zzo
- `stfloats` with `\fnbelowfloat` as alternative to raggedbottom
- `tabularx` X columns for longtable
- Percentage formatting consistency
- External resource links for MDPI guidelines

---

## Remaining Items for Future Passes

1. **Visual QA** — Rebuild PDF with all figure assets and inspect float placement
2. **Overfull boxes** — Address `\hbox` warnings in dense text passages
3. **Font consistency** — Audit all panels for consistent font sizes
4. **Acceptance mode** — When switching from `submit` to `accept`, remove `linenomath` wrappers
