#!/usr/bin/env bash
# verify_article_figures.sh — Verify all 20 article JPEG/PDF figure pairs exist and (re)create symlinks.
#
# Delegates to the Python delivery module (single source of truth for the figure list).
#
# Usage:
#   bash scripts/pipeline/verify_article_figures.sh            # verify + create symlinks
#   bash scripts/pipeline/verify_article_figures.sh --check   # verify only (no symlink changes)
set -euo pipefail

cd "$(dirname "$0")/../.."

# Run the delivery script by path so we don't load the rest of src.visualization (e.g. torch)
if [[ "${1:-}" == "--check" ]]; then
    exec python src/visualization/article_delivery.py --check-only
else
    exec python src/visualization/article_delivery.py
fi
