# Submission Gate — Pre-Upload Checklist

Treat the submission chain as a shipped **product**, not a collection of
working files. Nothing goes to the venue until every item below passes.

This gate is the counterpart of `scripts/pipeline/build_submission_bundle.sh`
— the script rebuilds everything; this checklist verifies the rebuild landed
coherently before the tarball leaves the author's machine.

---

## A. Source-reality locks (5 minutes)

Run the preflight greps to confirm cited source lines have not drifted:

```bash
grep -n 'get_position().height' src/visualization/fig09_expression_corr.py
# expect: L116 with `* 0.32`

grep -n 'abbreviate_cell_type' src/visualization/fig16_benchmark.py
# expect: L146 (=14), L181 (=16), L283 (=12) — or plan-documented deltas

grep -n "Accuracy (%)\|set_ylim(0, 105)" src/visualization/fig03_training.py
# expect: panel C `Accuracy (%)` at L166 with ylim (0, 105)
```

Any mismatch → **halt** and reconcile the plan before rebuilding the bundle.

---

## B. Figure state (2 minutes)

```bash
python scripts/pipeline/run_regeneration.py
```

Post-regeneration check:

- [ ] `results/figures/` has **25** `fig*.pdf` files, timestamps within the
      last 10 minutes.
- [ ] `articles/figures/`, `revision/manuscripts/v2_revision/figures/`, and
      `Elsevier_Array_Submission/figures/` all carry the same 25 PDFs. If
      any mirror is stale:
      ```bash
      cp results/figures/fig*.pdf Elsevier_Array_Submission/figures/
      ```
- [ ] `results/vcd_report.json` total warnings **≤** baseline total
      (`revision/figure_fix_reports/vcd_baseline_2026-04-19.json`):
      ```bash
      pytest tests/test_vcd_checks.py::test_vcd_warn_count_regression -v
      ```
- [ ] `tests/test_panel_label_offsets.py` passes:
      ```bash
      pytest tests/test_panel_label_offsets.py -v
      ```
- [ ] Documented VCD exemptions in `tests/test_vcd_checks.py` still carry
      their structural-rationale docstrings (fig02a horizontal-alignment
      conflict; fig07c I-a density excess). If the rationale no longer
      applies, eliminate the exemption rather than refresh it.

---

## C. Manuscript + diff (5 minutes)

- [ ] `revision/manuscripts/v2_revision/clop_dit_manuscript.pdf` rebuilt by
      `scripts/pipeline/build_article.sh` (exit 0; page count stable ±1
      vs. the previous build).
- [ ] `revision/manuscripts/diff/clop_dit_manuscript.diff.pdf` rebuilt by
      the latexdiff chain in `revision/manuscripts/README.md` §"Generating
      the diff". Page count matches the manuscript within ±2 pages (diff
      often adds a couple of overflow pages for change-bar annotations).
- [ ] `revision/manuscripts/v1_prerevision/` has **not** been modified
      since the baseline snapshot (git diff is empty for that subtree).

---

## D. Response letter (3 minutes)

- [ ] `revision/response_letter/rebuttal_letter.pdf` rebuilt by
      `revision/response_letter/build.sh` (exit 0).
- [ ] Page count ≤ **24** (hard cap) and ideally ≤ **20** (soft cap).
- [ ] Every reviewer comment block (`\begin{reviewercomment}`) has:
      (a) the original reviewer text quoted **verbatim**, and
      (b) at least one inlined `\begin{manuscriptexcerpt}` that copies the
      cited manuscript content — the reviewer must not need to flip back
      to the paper.
- [ ] `\msref{Figure X}` citations may skip the excerpt box (figure
      captions don't benefit from quoted paper text).
- [ ] `revision/response_letter/revision_cover_letter.pdf` rebuilt in the
      same pass.

---

## E. Checksums + bundle (2 minutes)

- [ ] `revision/SUBMISSION_CHECKSUMS.txt` is a 4-line SHA-256 manifest over
      the 4 submission PDFs, with times newer than any source `.tex` file
      it covers.
- [ ] `revision/submission_bundle_v1.0.0.tar.gz` contains the **5-entry
      PDF core** plus the **data-provenance archive**:
      ```
      manuscripts/v2_revision/clop_dit_manuscript.pdf
      manuscripts/diff/clop_dit_manuscript.diff.pdf
      response_letter/rebuttal_letter.pdf
      response_letter/revision_cover_letter.pdf
      SUBMISSION_CHECKSUMS.txt
      rebuttal_numerics/                   (9 primary JSONs + README.md)
      ```
      Audit with `tar tzf revision/submission_bundle_v1.0.0.tar.gz`.
      The `rebuttal_numerics/` subdirectory is the reviewer-auditability
      hook — every numeric claim in the rebuttal is mapped in its
      `README.md` to an upstream JSON field.
- [ ] **Author archive** (optional): `--include-author-internal` rebuilds
      a 7-entry tarball that also includes `AUTHOR_SIGNOFF.md` and
      `VENUE_PREFLIGHT.md`. This variant is for local archival only — do
      **not** upload it to the venue, since both files self-declare
      "gitignored, keep local".

---

## F. Tarball vs. checksums coherence (1 minute)

```bash
cd /tmp && mkdir -p _subverify && cd _subverify
tar xzf ../../revision/submission_bundle_v1.0.0.tar.gz
sha256sum -c SUBMISSION_CHECKSUMS.txt
cd .. && rm -rf _subverify
```

All 4 lines must print `OK`. If any line mismatches, the tarball and
checksum file disagreed — rerun `build_submission_bundle.sh`.

---

## G. Sign-off

- [ ] `revision/AUTHOR_SIGNOFF.md` dated today, with **author initials**
      under every section.
- [ ] `revision/VENUE_PREFLIGHT.md` items all checked — especially the
      venue page-limit entry if the journal has published one.
- [ ] `git log --oneline -3` shows the three-commit coherent-state chain
      (submission state #1 → #2 → infra).

---

## Quick one-command path

```bash
bash scripts/pipeline/build_submission_bundle.sh --interactive
```

The `--interactive` flag pauses before tarball repack and prints a VCD
diff + checksum preview so the author can eyeball the state before
committing. In CI or autopilot contexts, drop the flag.

## One-line upload-ready assertion

```bash
test -f revision/submission_bundle_v1.0.0.tar.gz \
  && [[ $(tar tzf revision/submission_bundle_v1.0.0.tar.gz | wc -l) == 5 ]] \
  && echo "READY FOR UPLOAD"
```
