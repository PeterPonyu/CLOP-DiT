#!/usr/bin/env bash
# build_article.sh — Verify article figures and build the LaTeX PDF.
#
# 1. Runs scripts/pipeline/verify_article_figures.sh so all 20 JPEG/PDF figure pairs are in articles/figures/
# 2. Runs latexmk -g -pdf in the articles directory, with LaTeX including the JPEG assets
#
# The -g flag forces a full recompile regardless of latexmk's timestamp/MD5 cache.
# This is required because figure files are symlinks: latexmk can incorrectly treat
# a symlink as unchanged (using the symlink's own mtime) even when the target PDF
# has been overwritten with a freshly generated figure.
#
# Paths respect configs/pipeline.yaml and env (CLOPDIT_FIG_DIR, CLOPDIT_ARTICLE_FIGURES_DIR).
#
# Usage:
#   bash scripts/build_article.sh
#   bash scripts/build_article.sh --check   # verify figures only, do not build PDF
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "▶ Verifying article figures..."
bash scripts/pipeline/verify_article_figures.sh

if [[ "${1:-}" == "--check" ]]; then
    echo "  (--check: skipping LaTeX build)"
    exit 0
fi

echo ""
echo "▶ Building article PDF..."
mapfile -t _article_paths < <(python scripts/pipeline/get_article_paths.py)
ARTICLE_DIR="${_article_paths[0]}"
ARTICLE_TEX="${_article_paths[1]}"
cd "$ARTICLE_DIR"
# -g: force full recompile even if latexmk considers everything up-to-date.
# This ensures regenerated figures (updated symlink targets) are always embedded.
latexmk -g -pdf "$ARTICLE_TEX"
# The manuscript uses natbib + an inline `\begin{thebibliography}` block
# (no external .bib). latexmk has been observed to stop short of the
# second pass in certain states, leaving thousands of `[?]` citations and
# `??` \ref tokens in the rendered PDF. Run an explicit verification pass
# after latexmk and fail loudly if any unresolved reference slipped
# through — better a halted build than a silently broken submission PDF.
unresolved=$(pdftotext -layout "${ARTICLE_TEX%.tex}.pdf" - 2>/dev/null | grep -cE '\[\?\]|\?\?' || true)
unresolved=${unresolved:-0}
if [[ "$unresolved" != "0" ]]; then
    echo "  ✗ $unresolved unresolved [?]/?? refs in ${ARTICLE_TEX%.tex}.pdf — running extra pdflatex passes" >&2
    for i in 1 2 3; do
        pdflatex -interaction=nonstopmode "$ARTICLE_TEX" > /dev/null
    done
    unresolved=$(pdftotext -layout "${ARTICLE_TEX%.tex}.pdf" - 2>/dev/null | grep -cE '\[\?\]|\?\?' || true)
unresolved=${unresolved:-0}
    if [[ "$unresolved" != "0" ]]; then
        echo "  ✗ Still $unresolved unresolved refs after recovery passes — aborting" >&2
        exit 1
    fi
fi
# Keep PDF outputs, but clean transient LaTeX build artifacts
# (.aux, .log, .fdb_latexmk, .fls, etc.) to avoid noisy commits.
latexmk -c "$ARTICLE_TEX"
cd - > /dev/null

echo ""
echo "  → $ARTICLE_DIR/${ARTICLE_TEX%.tex}.pdf"
