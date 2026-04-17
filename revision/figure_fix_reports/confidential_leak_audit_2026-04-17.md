# Confidential-leak audit — submission PDFs (Array / Elsevier)

**Auditor:** pre-submission leak sweep
**Date:** 2026-04-17
**Scope:**
- `revision/manuscripts/v2_revision/clop_dit_manuscript.pdf`
- `revision/manuscripts/diff/clop_dit_manuscript_tracked_changes.pdf`
- `revision/response_letter/rebuttal_letter.pdf`
- `revision/response_letter/revision_cover_letter.pdf`

Plus the three tex sources (`clop_dit_manuscript.tex`, `rebuttal_letter.tex`, `revision_cover_letter.tex`). The corresponding-author address `fuzeyu09@gmail.com` is authorised and is **not** flagged.

## Method

1. `pdftotext <path> -` applied to each PDF (line-anchored `.txt` held in `/tmp`).
2. `rg` run against those text dumps and the three `.tex` sources for the leak categories listed in the brief: internal filesystem paths, agent/AI tool names (claude / anthropic / codex / gpt / copilot / openai / omc / ralph / autopilot / architect-agent / scivcd / oh-my-claudecode / claudecode), Git commit SHAs, TODO / FIXME / HACK / `print(` / debug artefacts, non-author emails, Zenodo / DOI references, `.aux`/`.log` tail-drops, AI co-authorship statements, the internal `revision/experiments/lane_a_analysis/...` and `revision/experiments/lane_b_retrain/...` caption paths called out explicitly in the task.

## Per-leak table

| String | Location | Severity | Suggested fix |
|---|---|---|---|
| `revision/experiments/lane_a_analysis/a1_knn_confusion/.` | `clop_dit_manuscript.pdf` p.37 (pdftotext line 6411); Fig 13 caption `Data source:` tail. Mirrors from `clop_dit_manuscript.tex` Fig 13 caption (line flagged as long-line by grep). Also present in `clop_dit_manuscript_tracked_changes.pdf` line 7300 and at tracked-changes line 7437. | **BLOCKER** | Delete the sentence `Data source: revision/experiments/lane_a_analysis/a1_knn_confusion/.` from the Fig 13 caption (the caption remains complete without it). Regenerate manuscript + tracked-changes PDF. |
| `revision/experiments/lane_a_analysis/a2_organism_split/` | `clop_dit_manuscript.pdf` p.38 (pdftotext line 6521); Fig 14 caption. Mirror in `clop_dit_manuscript_tracked_changes.pdf` line 7439. | **BLOCKER** | Delete `Data source: revision/experiments/lane_a_analysis/a2_organism_split/.` from the Fig 14 caption. Regenerate. |
| `revision/experiments/lane_b_retrain/b3_mixing_sweep/.` | `clop_dit_manuscript.pdf` p.38 (pdftotext line 6597); Fig 15 caption. Mirror in `clop_dit_manuscript_tracked_changes.pdf` line 7579. | **BLOCKER** | Delete `Data source: revision/experiments/lane_b_retrain/b3_mixing_sweep/.` from the Fig 15 caption. Regenerate. |
| `Zenodo upon acceptance of this manuscript` | `clop_dit_manuscript.pdf` p.28 (line 4166); Data Availability section. No DOI is actually cited — only a promise of a future Zenodo deposit. | ADVISORY | Acceptable as drafted (no private/unminted DOI is printed). Keep wording as "will be archived on Zenodo upon acceptance" — no change required unless Elsevier prefers a placeholder DOI. |
| `ChatGPT` (bibliographic) | `clop_dit_manuscript.pdf` line 6636 / tracked-changes line 7619; Ref [13] Chen & Zou 2025, *Nat. Biomed. Eng.*, "embedding model for single-cell biology built from ChatGPT". Matches only inside a published bibliography entry. | COSMETIC | Keep — this is a legitimate, published reference. Not a leak. |
| `LLM decoding` | `clop_dit_manuscript.tex` line 215 (Cell2Sentence comparison row, method-family column). Rendered in the comparison table on p.7. | COSMETIC | Keep — describes a competing method's decoding strategy. Not an AI-assist disclosure. |
| `\useofartificialintelligence` macro | `clop_dit_manuscript.tex` line 137 (MDPI-style shim `\newcommand{\useofartificialintelligence}[1]{\section*{Use of AI-Assisted Tools}#1}`). **Macro is defined but never invoked** — no "Use of AI-Assisted Tools" section is rendered in any of the four PDFs (confirmed by grep of every pdftotext dump: no match for `AI-Assisted` / `Use of AI` / `AI assistance` in any PDF). Acknowledgments section (line 4170 of manuscript pdftotext) only thanks scGPT / BiomedBERT / GEO communities. | COSMETIC | No action required. The macro is inert; removing the unused shim from the preamble is optional hygiene. |
| `github.com/PeterPonyu/CLOP-DiT` | `clop_dit_manuscript.pdf` line 6719, Ref [45]. Public repository link. | COSMETIC | Confirm the repo is public and carries no private branches or `Co-Authored-By:` lines in visible history before submission. No PDF-side fix needed. |
| Corresponding-author email `fuzeyu09@gmail.com` | Manuscript p.1 line 16, tracked-changes p.1 line 16, rebuttal line 4, cover line 12 + line 97. | **AUTHORISED** | No action — flagged only for completeness per brief. |

## Negative results (nothing found, scan clean)

- No `/home/zeyufu/`, `/tmp/`, `/Users/`, `/root/`, `/mnt/`, `C:\Users\` anywhere in any PDF dump or tex source.
- No occurrences of `claude`, `anthropic`, `codex`, `copilot`, `openai`, `chatgpt` (except the legitimate Ref [13] Nat. Biomed. Eng. citation), `\bomc\b`, `\bralph\b`, `autopilot`, `architect-agent`, `scivcd`, `oh-my-claude`, `claudecode`.
- No Git commit SHAs (7–40 hex-char tokens) matched outside of bibliographic identifiers.
- No `TODO`, `FIXME`, `XXX`, `HACK`, `print(`, or `debug` artefacts in any PDF or tex source.
- No non-author emails. Only `fuzeyu09@gmail.com` (authorised) appears.
- No `.aux` / `.log` tail-drops, no reference to private Zenodo DOIs, no `Co-Authored-By:` lines, no "AI-assisted" or "LLM-assisted" disclosure rendered in any PDF.
- `rebuttal_letter.pdf` and `revision_cover_letter.pdf` are fully clean on every category scanned.
- `clop_dit_manuscript_tracked_changes.pdf` carries the same three lane-path leaks as the clean manuscript and nothing additional.

## Summary

**BLOCKER count: 3** — all three are the same class of leak: internal experiment folder paths (`revision/experiments/lane_a_analysis/a1_knn_confusion/`, `.../a2_organism_split/`, `revision/experiments/lane_b_retrain/b3_mixing_sweep/`) printed verbatim in Fig 13 / Fig 14 / Fig 15 captions inside the v2 manuscript PDF and mirrored in the tracked-changes diff PDF. All three originate in the `\caption{...}` bodies of `clop_dit_manuscript.tex`. Fix is a one-line deletion per caption, followed by a regeneration of both PDFs.

No other BLOCKER-class material (agent / AI tool names, filesystem paths, commit SHAs, debug artefacts, non-author emails, AI co-authorship disclosures) is present.
