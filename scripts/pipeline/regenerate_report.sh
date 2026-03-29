#!/usr/bin/env bash
# regenerate_report.sh — One-command regeneration of all article-facing figures.
#
# Prerequisites: trained CLOP (clop_best.pth) + DiT (dit_best.pth) + scGPT decoder.
# All intermediate outputs (embeddings, metrics, figures) are regenerated.
# After the figure-generation steps, article_delivery verifies all 21
# article-facing figure assets and creates symlinks in articles/figures/ so
# the LaTeX article builds correctly.
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

# ── Step 0: Generate architecture figure + evaluation pipeline ──
echo ""
echo "▶ Step 0/10: Generating architecture figure (Fig 1)..."
python scripts/analysis/generate_architecture_figure.py

echo ""
echo "▶ Step 0b/10: Generating evaluation pipeline figure (Fig 2)..."
python scripts/analysis/evaluation_pipeline_figure.py

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

# ── Step 8b: Python-composed supplementary appendix figures ──
echo ""
echo "▶ Step 8b/10: Generating Python-composed supplementary figures..."
python -m src.visualization.figS01_supplementary_validation
python -m src.visualization.figS02_expression_diagnostics

# ── Step 9: Verify article figures + create symlinks ──
echo ""
echo "▶ Step 9/10: Verifying article figures + creating symlinks..."
bash scripts/pipeline/verify_article_figures.sh

# ── Step 10: Rebuild LaTeX article PDF ──
echo ""
echo "▶ Step 10/10: Rebuilding LaTeX article PDF..."
bash scripts/pipeline/build_article.sh

# ── Summary ──
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Report regeneration complete"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Outputs (21 article-facing components + supporting diagnostics):"
echo "  Fig 1:   results/figures/fig01a_architecture.{png,pdf} + fig01b_evaluation_pipeline.{png,pdf}"
echo "  Fig 2:   results/figures/fig02a_training_dynamics.{png,pdf} + fig02b_embedding_space.{png,pdf}"
echo "  Fig 3:   results/figures/fig03a_metrics_summary.{png,pdf} + fig03b_per_type_fidelity.{png,pdf} + fig03c_text_cell_alignment.{png,pdf}"
echo "  Fig 4:   results/figures/fig04a_marker_genes.{png,pdf} + fig04b_expression_correlation.{png,pdf}"
echo "  Fig 5:   results/figures/fig05a_expression_analysis.{png,pdf} + fig05b_conditioning_landscape.{png,pdf}"
echo "  Fig 6:   results/figures/fig06_diversity_diagnostics.{png,pdf}"
echo "  Fig 7:   results/figures/fig07a_expression_diversity.{png,pdf} + fig07b_baseline_comparison.{png,pdf} + fig07c_benchmark.{png,pdf}"
echo "  Fig 8:   results/figures/fig08a_downstream_validation.{png,pdf} + fig08b_de_concordance.{png,pdf}"
echo "  Fig 9:   results/figures/fig09a_variance_matching.{png,pdf} + fig09b_gene_gene_correlation.{png,pdf}"
echo "  Fig S1:  results/figures/figS01_supplementary_validation.{png,pdf}"
echo "  Fig S2:  results/figures/figS02_expression_diagnostics.{png,pdf}"
echo "  Support: results/figures/fig13_noise_tradeoff.{png,pdf}, fig21_*.{png,pdf} … fig31_*.{png,pdf}"
echo ""
echo "  Combined:    results/figures/clop_dit_full_report.pdf"
echo "  Delivery:    articles/figures/ (21 PDFs → results/figures/)"
echo ""
ls -lh results/figures/clop_dit_full_report.pdf 2>/dev/null || true
