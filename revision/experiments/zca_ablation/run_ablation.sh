#!/usr/bin/env bash
# ZCA ablation driver — trains CLOP under three preprocessing conditions.
#
# Conditions:
#   whiten      — ZCA whitening (production default)
#   center_norm — mean-center + L2 normalize (no decorrelation)
#   none        — raw collapsed embeddings (expected to fail)
#
# All other hyperparameters held fixed (same seed, epochs, loss, architecture).
# Estimated wall-clock: ~30-45 min total (3 × 100 epochs on RTX 5090).

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Use clopdit conda env (has CUDA 12.8 + sm_120 support for RTX 5090)
PYTHON="/home/zeyufu/miniconda3/envs/clopdit/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: clopdit env not found at $PYTHON" >&2
    exit 1
fi

HERE="revision/experiments/zca_ablation"
LOG_DIR="${HERE}/logs"
mkdir -p "$LOG_DIR"

# ── Step 1: set up symlink cache for center_norm condition ────────────
# center_norm needs its own cache dir so auto-generated *_preprocessed.npy
# files don't overwrite the production ZCA ones.

CN_CACHE="data/cached_latents_ablation_centernorm"
if [ ! -d "$CN_CACHE" ]; then
    echo "[setup] Creating symlink cache for center_norm at $CN_CACHE"
    mkdir -p "$CN_CACHE"

    # Symlink all raw (non-preprocessed) files
    for f in \
        cell_embeddings.npy \
        cell_embeddings_dedup.npy \
        text_embeddings.npy \
        text_embeddings_dedup.npy \
        text_group_ids.npy \
        text_group_ids_dedup.npy \
        sample_ids.npy \
        sample_ids_dedup.npy \
        text_strings.json \
        text_captions_deduplicated.json \
        text_caption_metadata.json \
        processed_datasets.json \
        metadata.json \
        METADATA_DEDUP.json \
        manifest.json; do
        src="data/cached_latents/$f"
        if [ -f "$src" ]; then
            ln -sf "$(realpath "$src")" "$CN_CACHE/$f"
        fi
    done
    echo "[setup] Symlink cache ready ($(ls "$CN_CACHE" | wc -l) files)"
else
    echo "[setup] Symlink cache $CN_CACHE already exists"
fi

# ── Step 2: train each condition sequentially ─────────────────────────

CONDITIONS=(whiten center_norm none)

for COND in "${CONDITIONS[@]}"; do
    echo ""
    echo "=============================================="
    echo "  ZCA ablation: condition=$COND"
    echo "=============================================="

    CONFIG="${HERE}/configs/clop_${COND}.yaml"
    SAVE_DIR="models/revision/zca_ablation/${COND}"
    LOG="${LOG_DIR}/${COND}.log"

    mkdir -p "$SAVE_DIR"

    echo "[train $COND] start $(date '+%H:%M:%S')"
    time $PYTHON "${HERE}/train_clop_ablation.py" \
        --config "$CONFIG" \
        > "$LOG" 2>&1
    echo "[train $COND] done  $(date '+%H:%M:%S')"

    # Quick check: best checkpoint exists?
    if [ -f "${SAVE_DIR}/clop_best.pth" ]; then
        echo "[train $COND] clop_best.pth saved OK"
    else
        echo "[train $COND] WARNING: clop_best.pth NOT found in ${SAVE_DIR}"
    fi
done

echo ""
echo "All ZCA ablation conditions complete."
echo "Run: python ${HERE}/analyze_ablation.py"
