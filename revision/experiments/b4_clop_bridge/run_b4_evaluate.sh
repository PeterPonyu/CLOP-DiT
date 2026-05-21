#!/bin/bash
# B4 Phase 3: Generate embeddings and evaluate each DiT variant
#
# For each CLOP ablation variant with a trained DiT, this script:
# 1. Creates a symlinked cache dir with the variant's projected_text.npy
# 2. Runs generate_embeddings.py with the variant's DiT checkpoint
# 3. Collects metrics for cross-variant comparison
#
# Run AFTER run_b4_retrain.sh completes all DiT training.

set -euo pipefail

PYTHON=/home/zeyufu/miniconda3/envs/clopdit/bin/python
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

B4_DIR="revision/experiments/b4_clop_bridge"
CACHE_DIR="data/cached_latents"
VARIANTS="abl_baseline no_cohesion no_cell_noise"

echo "=== B4 Phase 3: Generate + Evaluate ==="

for V in $VARIANTS; do
    DIT_CKPT="models/revision/b4_bridge/$V/dit_best.pth"
    if [ ! -f "$DIT_CKPT" ]; then
        echo "[SKIP] $V: no DiT checkpoint at $DIT_CKPT"
        continue
    fi

    echo ""
    echo "--- Variant: $V ---"

    # Create variant-specific cache dir with symlinks
    VCACHE="$B4_DIR/cache_$V"
    mkdir -p "$VCACHE"
    # Symlink everything from real cache except projected_text.npy
    for f in "$CACHE_DIR"/*; do
        fname="$(basename "$f")"
        if [ "$fname" = "projected_text.npy" ]; then
            continue
        fi
        if [ ! -e "$VCACHE/$fname" ]; then
            ln -sf "$(realpath "$f")" "$VCACHE/$fname"
        fi
    done
    # Link variant's projected text
    ln -sf "$(realpath "$B4_DIR/projected_text/projected_text_${V}.npy")" "$VCACHE/projected_text.npy"

    # Output dir for this variant
    OUT_DIR="$B4_DIR/results_$V"
    mkdir -p "$OUT_DIR"

    # Generate embeddings (100 per type, same settings as production)
    echo "  Generating embeddings..."
    $PYTHON scripts/inference/generate_embeddings.py \
        --dit-checkpoint "$DIT_CKPT" \
        --cache-dir "$VCACHE" \
        --output-dir "$OUT_DIR" \
        --num-per-type 100 \
        --num-steps 20 \
        --cfg-scale 1.5 \
        --condition-mode condition_noise \
        --noise-scale 0.03 \
        --seed 42 \
        2>&1 | tee "$OUT_DIR/generate_log.txt"

    echo "  Metrics saved to $OUT_DIR/generation_metrics.json"
done

echo ""
echo "=== Phase 3 complete. Run analyze_b4_bridge.py for comparison ==="
