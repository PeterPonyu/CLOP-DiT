#!/usr/bin/env bash
# build_article.sh — Verify article figures and build the LaTeX PDF.
#
# 1. Runs scripts/pipeline/verify_article_figures.sh so all 17 PDFs are in articles/figures/
# 2. Runs latexmk -pdf in the articles directory
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
latexmk -pdf "$ARTICLE_TEX"
# Keep PDF outputs, but clean transient LaTeX build artifacts
# (.aux, .log, .fdb_latexmk, .fls, etc.) to avoid noisy commits.
latexmk -c "$ARTICLE_TEX"
cd - > /dev/null

echo ""
echo "  → $ARTICLE_DIR/${ARTICLE_TEX%.tex}.pdf"
