#!/usr/bin/env bash
# build_submission_bundle.sh — single-command submission bundle builder.
#
# Rebuilds every artifact shipped to the journal and packs a venue-safe
# tarball at revision/submission_bundle_v1.0.0.tar.gz:
#
#   1. Regenerate all 25 article figures (run_regeneration.py).
#   2. Build manuscript PDF       (scripts/pipeline/build_article.sh).
#   3. Build diff PDF             (latexdiff v1 -> v2 + 3 pdflatex passes).
#   4. Build rebuttal + cover PDFs (revision/response_letter/build.sh).
#   5. Refresh SUBMISSION_CHECKSUMS.txt with fresh SHA-256.
#   6. Repack submission_bundle_v1.0.0.tar.gz with the venue-safe 5-entry
#      manifest (4 PDFs + SUBMISSION_CHECKSUMS.txt). AUTHOR_SIGNOFF.md and
#      VENUE_PREFLIGHT.md are excluded by default since they self-declare
#      "gitignored, keep local"; pass --include-author-internal to include.
#
# Flags:
#   --interactive                 Pause before overwriting the tarball and
#                                 print a diff of the VCD report vs baseline;
#                                 wait for user ACK.
#   --include-author-internal     Build a 7-entry author archive (adds
#                                 AUTHOR_SIGNOFF.md + VENUE_PREFLIGHT.md).
#   --skip-regen                  Trust the existing results/figures and
#                                 skip step 1. Useful when only re-linking
#                                 text changes to a clean figure state.
#
# Exit codes:
#   0  submission bundle rebuilt successfully
#   1  any build stage failed
#   2  v1_prerevision/ was modified (frozen-snapshot guard)
#   3  pre-flight file check failed

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

INTERACTIVE=0
INCLUDE_AUTHOR_INTERNAL=0
SKIP_REGEN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interactive)                INTERACTIVE=1; shift ;;
    --include-author-internal)    INCLUDE_AUTHOR_INTERNAL=1; shift ;;
    --skip-regen)                 SKIP_REGEN=1; shift ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

step() { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

# Frozen-snapshot guard: v1_prerevision/ must not have been modified.
V1="revision/manuscripts/v1_prerevision"
if [[ -d "$V1" ]]; then
  if find "$V1" -newer revision/figure_fix_reports/vcd_baseline_2026-04-19.json -type f -print -quit 2>/dev/null | grep -q .; then
    fail "v1_prerevision/ has files newer than the VCD baseline — frozen snapshot violated; aborting (exit 2)."
  fi
fi

# Pre-flight check: plan's preflight greps must still match source.
step "Pre-flight: grep-n verification"
grep -qn 'get_position().height \* 0.32' src/visualization/fig09_expression_corr.py \
  || fail "fig09_expression_corr.py cbar height drifted; expected '* 0.32' at ~L116"
grep -qn 'abbreviate_cell_type' src/visualization/fig16_benchmark.py \
  || fail "fig16_benchmark.py abbreviate_cell_type calls missing"

# ---- Step 1: regenerate figures ---------------------------------------
if [[ $SKIP_REGEN -eq 0 ]]; then
  step "Regenerating figures"
  python scripts/pipeline/run_regeneration.py || fail "regeneration failed"
else
  step "Skipping regen (--skip-regen)"
fi

# ---- Step 2: manuscript PDF -------------------------------------------
step "Building manuscript PDF"
bash scripts/pipeline/build_article.sh > /dev/null || fail "manuscript build failed"
test -f revision/manuscripts/v2_revision/clop_dit_manuscript.pdf \
  || fail "manuscript PDF missing after build"

# ---- Step 3: latexdiff + diff PDF -------------------------------------
step "Building diff PDF (latexdiff)"
(
  cd revision/manuscripts
  latexdiff v1_prerevision/clop_dit_manuscript.tex \
            v2_revision/clop_dit_manuscript.tex \
    > diff/clop_dit_manuscript.diff.tex
  cp diff/clop_dit_manuscript.diff.tex v2_revision/
  cd v2_revision
  pdflatex -interaction=nonstopmode clop_dit_manuscript.diff.tex > /dev/null
  bibtex clop_dit_manuscript.diff > /dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode clop_dit_manuscript.diff.tex > /dev/null
  pdflatex -interaction=nonstopmode clop_dit_manuscript.diff.tex > /dev/null
  mv -f clop_dit_manuscript.diff.pdf ../diff/clop_dit_manuscript.diff.pdf
  rm -f clop_dit_manuscript.diff.{tex,aux,log,out,blg,bbl,fdb_latexmk,fls}
  cd ../diff
  rm -f clop_dit_manuscript.diff.{aux,log,out,blg,bbl,fdb_latexmk,fls,tex}
)
test -f revision/manuscripts/diff/clop_dit_manuscript.diff.pdf \
  || fail "diff PDF missing after latexdiff + pdflatex chain"

# ---- Step 4: rebuttal + cover PDFs ------------------------------------
step "Building rebuttal + cover-letter PDFs"
( cd revision/response_letter && bash build.sh > /dev/null ) \
  || fail "rebuttal/cover build failed"

# ---- Step 5: refresh checksums ----------------------------------------
step "Refreshing SUBMISSION_CHECKSUMS.txt"
(
  cd revision
  sha256sum \
    manuscripts/v2_revision/clop_dit_manuscript.pdf \
    manuscripts/diff/clop_dit_manuscript.diff.pdf \
    response_letter/rebuttal_letter.pdf \
    response_letter/revision_cover_letter.pdf \
    > SUBMISSION_CHECKSUMS.txt
)

# ---- Step 5.5: interactive gate (optional) ----------------------------
if [[ $INTERACTIVE -eq 1 ]]; then
  step "Interactive gate: review VCD + checksums before tarball repack"
  diff --unified=0 \
    revision/figure_fix_reports/vcd_baseline_2026-04-19.json \
    results/vcd_report.json | head -40 || true
  echo
  cat revision/SUBMISSION_CHECKSUMS.txt
  echo
  read -rp "Proceed with tarball repack? [y/N] " ack
  [[ "$ack" == "y" || "$ack" == "Y" ]] || fail "user aborted"
fi

# ---- Step 6: repack tarball -------------------------------------------
step "Repacking submission_bundle_v1.0.0.tar.gz"
(
  cd revision
  MANIFEST=(
    manuscripts/v2_revision/clop_dit_manuscript.pdf
    manuscripts/diff/clop_dit_manuscript.diff.pdf
    response_letter/rebuttal_letter.pdf
    response_letter/revision_cover_letter.pdf
    SUBMISSION_CHECKSUMS.txt
  )
  # Ship data-provenance archive alongside PDFs if present. rebuttal_numerics/
  # contains the 9 analysis-output JSONs + README.md that back every numeric
  # claim in the rebuttal — see revision/rebuttal_numerics/README.md.
  if [[ -d rebuttal_numerics ]]; then
    MANIFEST+=( rebuttal_numerics/ )
  fi
  if [[ $INCLUDE_AUTHOR_INTERNAL -eq 1 ]]; then
    MANIFEST+=( AUTHOR_SIGNOFF.md VENUE_PREFLIGHT.md )
  fi
  tar czf submission_bundle_v1.0.0.tar.gz "${MANIFEST[@]}"
)

step "Submission bundle rebuilt successfully"
printf '  → %s (%s bytes)\n' \
  "revision/submission_bundle_v1.0.0.tar.gz" \
  "$(stat -c%s revision/submission_bundle_v1.0.0.tar.gz)"
printf '  Entries: %d\n' \
  "$(tar tzf revision/submission_bundle_v1.0.0.tar.gz | wc -l)"
printf '  Next step: see revision/SUBMISSION_GATE.md\n'
