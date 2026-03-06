#!/usr/bin/env bash
# build_article.sh — Verify article figures and build the LaTeX PDF.
#
# 1. Runs verify_article_figures.sh so all 15 PDFs are in articles/figures/
# 2. Runs latexmk -pdf in the articles directory
#
# Paths respect configs/pipeline.yaml and env (CLOPDIT_FIG_DIR, CLOPDIT_ARTICLE_FIGURES_DIR).
#
# Usage:
#   bash scripts/build_article.sh
#   bash scripts/build_article.sh --check   # verify figures only, do not build PDF
set -euo pipefail

cd "$(dirname "$0")/.."

echo "▶ Verifying article figures..."
bash scripts/verify_article_figures.sh

if [[ "${1:-}" == "--check" ]]; then
    echo "  (--check: skipping LaTeX build)"
    exit 0
fi

echo ""
echo "▶ Building article PDF..."
cd articles
latexmk -pdf clop_dit_biology.tex
cd ..

echo ""
echo "  → articles/clop_dit_biology.pdf"
