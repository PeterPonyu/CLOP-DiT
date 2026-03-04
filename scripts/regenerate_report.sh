#!/usr/bin/env bash
# regenerate_report.sh — One-command regeneration of all 13 panels (A–M).
#
# Prerequisites: trained CLOP (clop_best.pth) + DiT (dit_best.pth) + scGPT decoder.
# All intermediate outputs (embeddings, metrics, figures) are regenerated.
#
# Usage:
#   bash scripts/regenerate_report.sh              # full pipeline (with UMAP)
#   bash scripts/regenerate_report.sh --no-umap    # skip slow UMAP panels
#   bash scripts/regenerate_report.sh --skip-gen    # reuse existing embeddings
set -euo pipefail

cd "$(dirname "$0")/.."
echo "═══════════════════════════════════════════════════════════════"
echo "  CLOP-DiT Report Regeneration Pipeline"
echo "═══════════════════════════════════════════════════════════════"

# Parse flags
UMAP_FLAG=""
SKIP_GEN=false
for arg in "$@"; do
    case $arg in
        --no-umap) UMAP_FLAG="--no-umap" ;;
        --skip-gen) SKIP_GEN=true ;;
    esac
done

# ── Step 1: Generate embeddings (condition_noise ε=0.03, CFG=1.5) ──
if [ "$SKIP_GEN" = false ]; then
    echo ""
    echo "▶ Step 1/6: Generating cell embeddings..."
    python scripts/generate_embeddings.py \
        --condition-mode condition_noise \
        --noise-scale 0.03 \
        --cfg-scale 1.5 \
        --num-per-type 100 \
        --num-steps 20
fi

# ── Step 2: Decode gene expression ──
echo ""
echo "▶ Step 2/6: Decoding gene expression via scGPT..."
python scripts/decode_expression.py

# ── Step 3: Diversity diagnostics (Panels J + K) ──
echo ""
echo "▶ Step 3/6: Running diversity diagnostics..."
python scripts/diversity_diagnostics.py \
    --num-per-type 100 \
    --cfg-scales 1.0 1.5 2.0 3.0 5.0 7.0

# ── Step 4: Conditioning analysis (Panels L + M) ──
echo ""
echo "▶ Step 4/6: Running conditioning analysis..."
python scripts/conditioning_analysis.py \
    --num-per-type 100 \
    --cfg-scale 1.5 \
    --noise-scale 0.03

# ── Step 5: Generate Panels A–I, N, O + combined report ──
echo ""
echo "▶ Step 5/6: Generating panels A–I, N, O + combined PDF..."
python -m src.visualization.results_visualizer $UMAP_FLAG

# ── Step 6: Summary ──
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Report regeneration complete"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Outputs:"
echo "  Panels A–I:  results/figures/panel_[a-i]_*.png"
echo "  Panels J–K:  results/figures/panel_j_diversity_diagnostics.png"
echo "               results/figures/panel_k_expression_diversity.png"
echo "  Panels L–M:  results/figures/panel_l_noise_tradeoff.png"
echo "               results/figures/panel_m_conditioning_umap.png"
echo "  Panel N:     results/figures/panel_n_marker_gene_comparison.png"
echo "  Panel O:     results/figures/panel_o_baseline_comparison.png"
echo "  Combined:    results/figures/clop_dit_full_report.pdf"
echo ""
echo "  Metrics:     results/generation_metrics.json"
echo "               results/generation_metadata.json"
echo "               results/diversity_diagnostics.json"
echo "               results/expression_metrics.json"
echo ""
ls -lh results/figures/clop_dit_full_report.pdf 2>/dev/null || true
