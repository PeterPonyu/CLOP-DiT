#!/usr/bin/env bash
# build_article_elsevier.sh — Generate and build the Elsevier-formatted manuscript.
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${CLOPDIT_PYTHON_BIN:-python}"

export CLOPDIT_ARTICLE_DIR="articles_elsevier"
export CLOPDIT_ARTICLE_TEX="clop_dit_elsevier.tex"
export CLOPDIT_ARTICLE_FIGURES_DIR="articles_elsevier/figures"

"$PYTHON_BIN" scripts/pipeline/make_elsevier_article.py
bash scripts/pipeline/build_article.sh "$@"
