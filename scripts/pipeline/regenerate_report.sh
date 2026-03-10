#!/usr/bin/env bash
# regenerate_report.sh — One-command regeneration of all 20 article figures.
#
# Prerequisites: trained CLOP (clop_best.pth) + DiT (dit_best.pth) + scGPT decoder.
# All intermediate outputs (embeddings, metrics, figures) are regenerated.
# After step 7 (visualization), step 8 verifies all 20 article figures and
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

# ── Step 7: Generate Figs 3–18 + combined report ──
echo ""
echo "▶ Step 7/10: Generating Figs 3–18 + combined PDF..."
python -m src.visualization.results_visualizer $UMAP_FLAG

# ── Step 8: External figures (Figs 19, 20) ──
echo ""
echo "▶ Step 8/10: Generating external figures (Figs 19, 20)..."
python scripts/analysis/variance_matching_pilot.py
python scripts/analysis/gene_gene_correlation.py

# ── Step 9: Verify article figures + create symlinks ──
echo ""
echo "▶ Step 9/10: Verifying article figures + creating symlinks..."
bash scripts/pipeline/verify_article_figures.sh

# ── Summary ──
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Report regeneration complete"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Outputs (20 article figures):"
echo "  Fig 1:   results/figures/fig_architecture.{png,pdf}"
echo "  Fig 2:   results/figures/fig_evaluation_pipeline.{png,pdf}"
echo "  Fig 3:   results/figures/fig03_training_dynamics.{png,pdf}"
echo "  Fig 4:   results/figures/fig04_embedding_space.{png,pdf}"
echo "  Fig 5:   results/figures/fig05_metrics_summary.{png,pdf}"
echo "  Fig 6:   results/figures/fig06_per_type_fidelity.{png,pdf}"
echo "  Fig 7:   results/figures/fig07_text_cell_alignment.{png,pdf}"
echo "  Fig 8:   results/figures/fig08_marker_genes.{png,pdf}"
echo "  Fig 9:   results/figures/fig09_expression_correlation.{png,pdf}"
echo "  Fig 10:  results/figures/fig10_expression_analysis.{png,pdf}"
echo "  Fig 11:  results/figures/fig11_conditioning_umap.{png,pdf}"
echo "  Fig 12:  results/figures/fig12_diversity_diagnostics.{png,pdf}"
echo "  Fig 13:  results/figures/fig13_noise_tradeoff.{png,pdf}"
echo "  Fig 14:  results/figures/fig14_expression_diversity.{png,pdf}"
echo "  Fig 15:  results/figures/fig15_baseline_comparison.{png,pdf}"
echo "  Fig 16:  results/figures/fig16_benchmark.{png,pdf}"
echo "  Fig 17:  results/figures/fig17_downstream_pq.{png,pdf}"
echo "  Fig 18:  results/figures/fig18_de_concordance.{png,pdf}"
echo "  Fig 19:  results/figures/fig19_variance_matching_pilot.{png,pdf}"
echo "  Fig 20:  results/figures/fig20_gene_gene_correlation.{png,pdf}"
echo ""
echo "  Combined:    results/figures/clop_dit_full_report.pdf"
echo "  Symlinks:    articles/figures/ (20 PDFs → results/figures/)"
echo ""
ls -lh results/figures/clop_dit_full_report.pdf 2>/dev/null || true
