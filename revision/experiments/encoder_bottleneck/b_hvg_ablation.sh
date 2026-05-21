#!/usr/bin/env bash
# B — HVG-count ablation driver.
#
# Preprocesses + scGPT-embeds the 50-dataset pipeline at n_top_genes in
# {500, 1000, 4000, 8000}. All other pipeline settings (max_cells=3000,
# subsample_seed=0, scGPT encoder, batch_size=128) are held fixed so any
# difference in latent within-type variance is attributable to the HVG
# count. The n_top_genes=2000 baseline is reused from
# data/processed_h5ad_cap3k_seed0 / data/cached_latents_cap3k_seed0.
#
# Total expected wall-clock: ~40-50 minutes.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

HVG_COUNTS=(500 1000 4000 8000)
LOG_DIR="revision/experiments/encoder_bottleneck/hvg_ablation_logs"
mkdir -p "$LOG_DIR"

for N in "${HVG_COUNTS[@]}"; do
    echo ""
    echo "=============================================="
    echo "  HVG ablation: n_top_genes=$N"
    echo "=============================================="

    H5AD_DIR="data/processed_h5ad_hvg${N}"
    CACHE_DIR="data/cached_latents_hvg${N}"

    mkdir -p "$H5AD_DIR" "$CACHE_DIR"

    echo "[preprocess] $(date '+%H:%M:%S')"
    time python scripts/data_prep/00_prepare_all_data.py \
        --output_dir "$H5AD_DIR" \
        --max_cells 3000 \
        --subsample_seed 0 \
        --n_top_genes "$N" \
        > "$LOG_DIR/preprocess_hvg${N}.log" 2>&1

    echo "[embed] $(date '+%H:%M:%S')"
    time python scripts/data_prep/03_cache_latents.py \
        --h5ad_dir "$H5AD_DIR" \
        --metadata "$H5AD_DIR/metadata_structured.json" \
        --output_dir "$CACHE_DIR" \
        --cell_encoder scgpt \
        --scgpt_dir models/scgpt_human \
        --batch_size 128 \
        > "$LOG_DIR/embed_hvg${N}.log" 2>&1

    echo "[done hvg=$N] $(date '+%H:%M:%S')"
done

echo ""
echo "All HVG variants complete."
