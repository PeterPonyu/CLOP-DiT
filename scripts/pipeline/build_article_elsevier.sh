#!/usr/bin/env bash
# build_article_elsevier.sh — Build the hand-maintained Elsevier-formatted manuscript.
set -euo pipefail

cd "$(dirname "$0")/../.."

export CLOPDIT_ARTICLE_DIR="articles_elsevier"
export CLOPDIT_ARTICLE_TEX="clop_dit_elsevier.tex"
export CLOPDIT_ARTICLE_FIGURES_DIR="articles_elsevier/figures"

bash scripts/pipeline/build_article.sh "$@"

# Keep the elsarticle natbib sidecar from creating noisy diffs.
: > "$CLOPDIT_ARTICLE_DIR/${CLOPDIT_ARTICLE_TEX%.tex}.spl"
