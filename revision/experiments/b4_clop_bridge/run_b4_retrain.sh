#!/bin/bash
# B4 — CLOP ablation → full pipeline bridge
# Retrains DiT on 3 CLOP ablation variants to test end-to-end transfer.
#
# Variants: abl_baseline, no_cohesion, no_cell_noise
# Step 1: Project text through each CLOP variant
# Step 2: Train DiT (300 epochs) per variant
# Step 3: Generate embeddings per variant (future)
#
# Estimated GPU time: ~6 hr total on RTX 5090

set -euo pipefail

PYTHON=/home/zeyufu/miniconda3/envs/clopdit/bin/python
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

B4_DIR="revision/experiments/b4_clop_bridge"
CACHE_DIR="data/cached_latents"

VARIANTS="abl_baseline no_cohesion no_cell_noise"

echo "=== B4 Phase 2: DiT retraining on CLOP ablation variants ==="
echo "Repository: $REPO_ROOT"
echo "Variants: $VARIANTS"
echo ""

# ── Step 1: Project text through each CLOP variant ──
echo "=== Step 1: Projecting text embeddings ==="
for V in $VARIANTS; do
    echo ""
    echo "--- Projecting: $V ---"
    OUT_DIR="$B4_DIR/projected_text"
    mkdir -p "$OUT_DIR"

    $PYTHON "$B4_DIR/project_text_variant.py" \
        --variant "$V" \
        --output "$OUT_DIR/projected_text_${V}.npy"

    echo "  Saved: $OUT_DIR/projected_text_${V}.npy"
done

echo ""
echo "=== Step 2: Training DiT per variant ==="
for V in $VARIANTS; do
    echo ""
    echo "--- Training DiT: $V ---"
    SAVE_DIR="models/revision/b4_bridge/$V"
    PROJ_TEXT="$B4_DIR/projected_text/projected_text_${V}.npy"

    mkdir -p "$SAVE_DIR"

    $PYTHON scripts/training/04b_train_dit.py \
        --config configs/dit.yaml \
        --projected_text "$PROJ_TEXT" \
        --save_dir "$SAVE_DIR" \
        --seed 42 \
        2>&1 | tee "$SAVE_DIR/train_log.txt"

    echo "  DiT trained for $V → $SAVE_DIR"
done

echo ""
echo "=== B4 Phase 2 complete ==="
echo "DiT checkpoints saved under models/revision/b4_bridge/"
echo "Next: run generation + evaluation per variant"
