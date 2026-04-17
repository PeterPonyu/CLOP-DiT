# Remaining-Work Progress — 2026-04-17 (ralph-L follow-up)

Building on `remaining_work_roadmap_2026-04-17.md`.  Captures the concrete opt-in items shipped during the second ralph pass on 2026-04-17.

## Shipped this pass

1. **Panel-label uppercase enforcement (text + figures)** — 186 LaTeX substitutions; bumped `PANEL_LABEL_FONT_SIZE` 16→18 for on-page prominence.  Status: CLOSED; see `panel_label_refresh_audit_2026-04-17.md`.

2. **Six reviewer-flagged figure layout fixes** — documented above; all six issues regenerated and re-audited with scivcd.  Status: CLOSED.

3. **Confidentiality scrub of diff.tex** — latexdiff-leaked local filesystem paths (`/tmp/clop-v1-neutral/…`, `/home/zeyufu/…`) rewritten to neutral text.  Final `pdftotext | grep` across all four submission PDFs: 0 leaks.  Status: CLOSED.

4. **Baseline / remote / checksum sanity** — `git fetch origin` then `git log origin/revision/major..HEAD` → empty (local = remote).  `sha256sum -c SUBMISSION_CHECKSUMS.txt` → 4/4 OK after refresh.  Baseline file `revision/manuscripts/v1_prerevision/clop_dit_manuscript.tex` is a curated venue-neutral snapshot, not a 1:1 copy of the `pre-revision-2026-04-15` tag payload (the tag packaged the baseline under `revision/prerevision_baseline/` instead); this is documented in `REVISION_LOG.md`.  Status: CLOSED.

## Opt-in items still deferrable

- **Anonymization audit for double-blind venues** (S).  *Array* / Elsevier is single-blind, so not required for the current submission, but worth a single `grep -i "Zeyu\|Chongqing\|Army Medical"` if re-targeting a double-blind venue.
- **Zenodo deposit rehearsal** (M).  `ZENODO_RELEASE_NOTES.md` exists.  A dry-run deposit would verify the artefact bundle before clicking "Publish".
- **arXiv / bioRxiv preprint rehearsal** (M).  The main manuscript is self-contained; preprint posting is low-risk once venue policy is confirmed.
- **Language / copy-edit pass** on the longest Results subsections (3.4, 3.5, 3.10).  Recommended if another reviewing round follows.
- **Rebuttal concordance audit** — cross-ref every numeric claim in `rebuttal_letter.tex` against the corresponding manuscript figure / table.  The prior `rebuttal_crossref_audit.md` covers this; re-running it post-rebuild would catch any drift introduced by the label-refresh pass.
- **Final author sign-off circulation** — prepare a short "changes since last author review" digest and send to the three co-authors before the final submit click.
