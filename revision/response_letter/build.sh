#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

for f in rebuttal_letter revision_cover_letter; do
  echo "=== Compiling $f.tex ==="
  pdflatex -interaction=nonstopmode "$f.tex" 2>&1 | tail -5
  pdflatex -interaction=nonstopmode "$f.tex" 2>&1 | tail -5
  echo "=== Done: $f.pdf ==="
done

# Clean auxiliary files
rm -f *.aux *.log *.out *.toc *.fdb_latexmk *.fls 2>/dev/null || true
echo "Build complete."
