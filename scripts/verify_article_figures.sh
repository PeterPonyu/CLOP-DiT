#!/usr/bin/env bash
# verify_article_figures.sh — Verify all 15 article figures exist and (re)create symlinks.
#
# Delegates to the Python delivery module (single source of truth for the figure list).
#
# Usage:
#   bash scripts/verify_article_figures.sh            # verify + create symlinks
#   bash scripts/verify_article_figures.sh --check   # verify only (no symlink changes)
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--check" ]]; then
    exec python -m src.visualization.article_delivery --check-only
else
    exec python -m src.visualization.article_delivery
fi
