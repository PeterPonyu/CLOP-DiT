#!/usr/bin/env bash
# regenerate_report.sh — One-command regeneration of all 19 panels (A–S) + architecture figure.
#
# Prerequisites: trained CLOP (clop_best.pth) + DiT (dit_best.pth) + scGPT decoder.
# All intermediate outputs (embeddings, metrics, figures) are regenerated.
# After step 7 (visualization), step 8 verifies all 17 article figures and
# creates symlinks in articles/figures/ so the LaTeX article builds correctly.
#
# Paths come from configs/pipeline.yaml and src.utils.paths; override via env:
#   CLOPDIT_CACHE_DIR, CLOPDIT_RESULTS_DIR, CLOPDIT_FIG_DIR,
#   CLOPDIT_ARTICLE_FIGURES_DIR, CLOPDIT_CKPT_DIR
#
# Usage:
#   bash scripts/regenerate_report.sh              # full pipeline (with UMAP)
#   bash scripts/regenerate_report.sh --no-umap    # skip slow UMAP panels
#   bash scripts/regenerate_report.sh --skip-gen    # reuse existing embeddings
set -euo pipefail

cd "$(dirname "$0")/../.."
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

# ── Step 0: Generate architecture figure ──
echo ""
echo "▶ Step 0/8: Generating architecture figure..."
python scripts/analysis/generate_architecture_figure.py

# ── Step 1: Generate embeddings (condition_noise ε=0.03, CFG=1.5) ──
if [ "$SKIP_GEN" = false ]; then
    echo ""
    echo "▶ Step 1/8: Generating cell embeddings..."
    python scripts/inference/generate_embeddings.py \
        --condition-mode condition_noise \
        --noise-scale 0.03 \
        --cfg-scale 1.5 \
        --num-per-type 100 \
        --num-steps 20
fi

# ── Step 2: Decode gene expression ──
echo ""
echo "▶ Step 2/8: Decoding gene expression via scGPT..."
python scripts/analysis/decode_expression.py

# ── Step 3: Diversity diagnostics (Panels J + K) ──
echo ""
echo "▶ Step 3/8: Running diversity diagnostics..."
python scripts/analysis/diversity_diagnostics.py \
    --num-per-type 100 \
    --cfg-scales 1.0 1.5 2.0 3.0 5.0 7.0

# ── Step 4: Conditioning analysis (Panels L + M) ──
echo ""
echo "▶ Step 4/8: Running conditioning analysis..."
python scripts/analysis/conditioning_analysis.py \
    --num-per-type 100 \
    --cfg-scale 1.5 \
    --noise-scale 0.03

# ── Step 5: Downstream biology (clustering, classifier, DE → P/Q/R data) ──
echo ""
echo "▶ Step 5/8: Running downstream biology analysis..."
python -m src.evaluation.downstream_biology \
    --output-dir results/downstream

# ── Step 6: Model benchmarking (CLOP-DiT vs 4 baselines) ──
echo ""
echo "▶ Step 6/8: Running model benchmarking..."
python -m src.evaluation.model_benchmarking

# ── Step 7: Generate Panels A–S + combined report ──
echo ""
echo "▶ Step 7/8: Generating panels A–S + combined PDF..."
python -m src.visualization.results_visualizer $UMAP_FLAG

# ── Step 8: Verify article figures + create symlinks ──
echo ""
echo "▶ Step 8/8: Verifying article figures + creating symlinks..."
bash scripts/pipeline/verify_article_figures.sh

# ── Summary ──
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Report regeneration complete"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Outputs:"
echo "  Architecture: results/figures/fig_architecture.{png,pdf}"
echo "  Panels A–I:  results/figures/panel_[a-i]_*.{png,pdf}"
echo "  Panels J–K:  results/figures/panel_j_diversity_diagnostics.{png,pdf}"
echo "               results/figures/panel_k_expression_diversity.{png,pdf}"
echo "  Panels L–M:  results/figures/panel_l_noise_tradeoff.{png,pdf}"
echo "               results/figures/panel_m_conditioning_umap.{png,pdf}"
echo "  Panel N:     results/figures/panel_n_marker_gene_comparison.{png,pdf}"
echo "  Panel O:     results/figures/panel_o_baseline_comparison.{png,pdf}"
echo "  Panel P:     results/figures/panel_p_clustering_mixing.{png,pdf}"
echo "  Panel Q:     results/figures/panel_q_classifier_alignment.{png,pdf}"
echo "  Panel R:     results/figures/panel_r_de_concordance.{png,pdf}"
echo "  Panel S:     results/figures/panel_s_benchmark.{png,pdf}"
echo ""
echo "  Merged (article):"
echo "    results/figures/fig_training_dynamics.{png,pdf}     (A+C)"
echo "    results/figures/fig_embedding_space.{png,pdf}       (B+E)"
echo "    results/figures/fig_fidelity_alignment.{png,pdf}    (G+F)"
echo "    results/figures/fig_diversity_tradeoff.{png,pdf}    (L+K)"
echo "    results/figures/fig_downstream_pq.{png,pdf}         (P+Q)"
echo ""
echo "  Combined:    results/figures/clop_dit_full_report.pdf"
echo "  Symlinks:    articles/figures/ (17 PDFs → results/figures/)"
echo ""
echo "  Metrics:     results/generation_metrics.json"
echo "               results/generation_metadata.json"
echo "               results/diversity_diagnostics.json"
echo "               results/expression_metrics.json"
echo "               results/benchmark_report.json"
echo ""
ls -lh results/figures/clop_dit_full_report.pdf 2>/dev/null || true
