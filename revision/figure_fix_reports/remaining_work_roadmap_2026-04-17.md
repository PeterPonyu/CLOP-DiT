# Remaining-Work Roadmap — 2026-04-17

Opt-in queue of polish/prep items the user can pick from. Each row lists
description, effort (S/M/L), value tier (reviewer-facing / internal /
prep), pre-requisite state, and a "when to do this" note.

| # | Item | Effort | Value | Pre-req | When to do it |
|---|---|:---:|:---:|---|---|
| 1 | **Anonymization audit for double-blind venues** — grep manuscript, rebuttal, cover letter, and figure captions for author names, GitHub handles, institution strings, corresponding-author email, ORCID, Zenodo DOI. Produce a redacted shadow build if the target venue requires blind review. | S | reviewer-facing | Known venue policy | Only if the submission venue requires double-blind review (Array/Elsevier usually does not); skip otherwise. |
| 2 | **Reproducibility statement + Zenodo deposit rehearsal** — write a Data & Code Availability section listing the GitHub repo, the frozen tag for reproduction (`submission-2026-04-17`), the checksum manifest path, the scGPT / BiomedBERT / PanglaoDB / CellMarker versions, and push a dry-run Zenodo deposit for the revision. | M | reviewer-facing + internal | Zenodo account, final manuscript frozen | Within 48 h of submission, so the DOI can be referenced on the cover letter and in future work. |
| 3 | **Preprint (arXiv / bioRxiv) preparation** — re-render the manuscript PDF under arXiv's line-numbering + margin rules; upload to bioRxiv with the same Lane C figure and rebuttal summary for public visibility. | M | reviewer-facing | Author sign-off; submission packet frozen | Before or within 72 h of journal submission; gets the work cited even during review. |
| 4 | **Code-availability GitHub tagging** — tag `code-release-2026-04-17` on the `revision/major` head; add a README badge pointing to the tagged release; ensure CLI entry points (figure regen, eval metrics) run cleanly from a fresh clone. | S | reviewer-facing + prep | Remote is in sync with local HEAD | Immediately before journal submission. |
| 5 | **Rebuttal numeric-concordance sweep** — re-run the cross-document number audit (manuscript vs rebuttal) after the 8 THIN-reply deepenings land, to confirm every new number quoted in the rebuttal still resolves to a manuscript table/figure. | S | reviewer-facing | This commit (deepening) landed | Immediately after this commit; `pdftotext` + grep workflow. |
| 6 | **fig05a panel (a) gene-label drift fix** (deferred US-J07) — find the source script for fig05a, replace the fixed slot-offset callouts with `xytext`-offset adjacent labels, and add ≥ 2× vertical gap between panels (a) and (c). | S | reviewer-facing | None | Next figure-polish round. |
| 7 | **Language / copy-edit pass** — have a native-speaker copy-editor (or a structured prompt) read the manuscript top-to-bottom for awkward phrasing, passive-voice overuse, and mid-sentence tense shifts; apply surgical edits only. | M | reviewer-facing | Numeric concordance (item 5) complete | If the submission window allows another day. |
| 8 | **Final author sign-off circulation checklist** — email the four current submission PDFs to Y.L., J.W., S.W. (co-authors listed on the title page); include the sha256 manifest and the URL of the tagged GitHub release; capture each author's explicit sign-off in writing. | S | prep (ethics) | Items 4 & 5 complete | Last human-in-the-loop step before the journal portal upload. |
| 9 | **Pre-commit guard for internal filesystem paths** — extend the pre-push hook that blocks literal `??` to also fail on `/home/`, `revision/experiments/`, and AI-tool mentions in any submission PDF. | S | internal | pre-push hook exists | Immediately after the literal-`??` hook is in place (ships in this commit). |
| 10 | **Journal-specific formatting preflight** — once the target venue is final, re-render under the venue's class file (Elsevier Array uses `elsarticle.cls` or the Array portal's PDF-wrapped upload). Some reviewers will expect line numbers in the revision PDF; verify whether the Array portal adds them server-side or the author must enable `\usepackage{lineno}`. | M | reviewer-facing | Known venue class file | Final step before submission. |
| 11 | **Supplementary file inventory** — bundle the four submission PDFs, the 25 figure PDFs separately, the sha256 manifest, and a short README into `revision/submission_bundle/`. Elsevier Array portals usually want a zip. | S | reviewer-facing + prep | All PDFs frozen | Last step before upload. |
| 12 | **Expanded cross-dataset validation (Lane E, future work)** — if the R3.1 Lane-C result prompts a follow-up request, propose the next strict-OOD tissue panel (e.g., pancreas, liver cirrhosis) for a re-revision response. | L | reviewer-facing (conditional) | Reviewer pushback | Only if reviewers request it after seeing the current submission. |

## Suggested short-term ordering

1. **Now (this commit)**: items 5 + 9 (pre-push hook + concordance sweep) ship alongside the 8 rebuttal deepenings.
2. **Before submission**: items 4, 8, 11, 10 (tag release, author sign-off, bundle, venue preflight).
3. **Within 48 h of submission**: items 2, 3 (Zenodo DOI + preprint).
4. **Opt-in as time allows**: items 1, 6, 7, 12.

## Not recommended

- Further experiments beyond Lane A/B/C/D — six rounds of revision and three new supplementary figures have already closed the reviewer-evidence loop. Adding another lane before reviewers see the current submission is negative expected value.
- Re-tagging `submission-2026-04-17` after the rebuttal deepening — better to create `submission-2026-04-17-r2` or `submission-2026-04-18` so the original candidate anchor remains recoverable.
