#!/usr/bin/env bash
# setup_data.sh — Download / verify external data and model weights.
#
# Prerequisites:
#   - conda environment 'dl' with PyTorch, scGPT, etc.
#   - Internet access for model downloads (first run only)
#
# This script ensures all required data directories exist and contain
# the expected files. It does NOT overwrite existing files.
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== CLOP-DiT Data Setup ==="

# ─────────────────────────────────────────────────────────────
# 1. Directory structure
# ─────────────────────────────────────────────────────────────
echo ""
echo "--- Creating directory structure ---"
for d in \
    data/cached_latents \
    data/processed_h5ad \
    models/checkpoints \
    models/scgpt_human \
    results \
    results/figures \
    results/downstream \
    logs \
    figures/v5_publication
do
    mkdir -p "$d"
    echo "  ✓ $d"
done

# ─────────────────────────────────────────────────────────────
# 2. Verify scGPT model weights
# ─────────────────────────────────────────────────────────────
echo ""
echo "--- Checking scGPT model weights ---"
SCGPT_DIR="models/scgpt_human"
SCGPT_CKPT="$SCGPT_DIR/best_model.pt"

if [ -f "$SCGPT_CKPT" ]; then
    SIZE=$(du -sh "$SCGPT_CKPT" | cut -f1)
    echo "  ✓ scGPT pancancer model found ($SIZE)"
else
    echo "  ✗ Missing: $SCGPT_CKPT"
    echo "    Download from: https://github.com/bowang-lab/scGPT"
    echo "    Place whole_human model weights in $SCGPT_DIR/"
fi

# ─────────────────────────────────────────────────────────────
# 3. Verify cached latents
# ─────────────────────────────────────────────────────────────
echo ""
echo "--- Checking cached latents ---"
CACHE_DIR="data/cached_latents"
REQUIRED_LATENTS=(
    "cell_embeddings_dedup_preprocessed.npy"
    "text_group_ids_dedup.npy"
    "projected_text.npy"
    "projected_cells.npy"
)
MISSING=0
for f in "${REQUIRED_LATENTS[@]}"; do
    if [ -f "$CACHE_DIR/$f" ]; then
        SIZE=$(du -sh "$CACHE_DIR/$f" | cut -f1)
        echo "  ✓ $f ($SIZE)"
    else
        echo "  ✗ Missing: $CACHE_DIR/$f"
        MISSING=$((MISSING + 1))
    fi
done
if [ $MISSING -gt 0 ]; then
    echo ""
    echo "  To regenerate latent cache:"
    echo "    conda run -n dl python scripts/03_cache_latents.py"
    echo "    conda run -n dl python scripts/03b_preprocess_embeddings.py"
fi

# ─────────────────────────────────────────────────────────────
# 4. Verify CLOP/DiT checkpoints
# ─────────────────────────────────────────────────────────────
echo ""
echo "--- Checking model checkpoints ---"
CKPT_DIR="models/checkpoints"
for f in clop_best.pth clop_final.pth dit_best.pth dit_final.pth; do
    if [ -f "$CKPT_DIR/$f" ]; then
        SIZE=$(du -sh "$CKPT_DIR/$f" | cut -f1)
        echo "  ✓ $f ($SIZE)"
    else
        echo "  ✗ Missing: $CKPT_DIR/$f"
        echo "    To train: bash scripts/full_retrain_pipeline.sh"
    fi
done

# ─────────────────────────────────────────────────────────────
# 5. Verify processed h5ad
# ─────────────────────────────────────────────────────────────
echo ""
echo "--- Checking processed h5ad ---"
H5AD_COUNT=$(find data/processed_h5ad -name "*.h5ad" 2>/dev/null | wc -l)
if [ "$H5AD_COUNT" -gt 0 ]; then
    echo "  ✓ $H5AD_COUNT h5ad file(s) found"
else
    echo "  ✗ No h5ad files in data/processed_h5ad/"
    echo "    Run: conda run -n dl python scripts/01_integrate_h5_datasets.py"
fi

echo ""
echo "=== Setup check complete ==="
