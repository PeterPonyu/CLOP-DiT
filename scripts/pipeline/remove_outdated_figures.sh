#!/usr/bin/env bash
# remove_outdated_figures.sh — Remove legacy figure outputs that are not produced by the current pipeline.
#
# The canonical pipeline is regenerate_report.sh. It produces:
#   fig_architecture, fig_training_dynamics, fig_embedding_space, fig_fidelity_alignment,
#   fig_diversity_tradeoff, fig_downstream_pq, panel_a_* … panel_s_*, clop_dit_full_report.pdf
#
# This script removes only known-outdated files (from 10_full_pipeline.py or older scripts)
# so that current report figures are left intact.
#
# Usage: bash scripts/pipeline/remove_outdated_figures.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIG_DIR="${CLOPDIT_FIG_DIR:-$REPO_ROOT/results/figures}"
cd "$REPO_ROOT"

if [[ ! -d "$FIG_DIR" ]]; then
    echo "No figures directory at $FIG_DIR; nothing to clean."
    exit 0
fi

REMOVED=0
for base in fig1_training_dynamics fig2_embedding_space fig3_metrics_dashboard fig4_biological_validation fig5_dimension_sampling; do
    for ext in png pdf; do
        if [[ -f "$FIG_DIR/${base}.${ext}" ]]; then
            rm -f "$FIG_DIR/${base}.${ext}"
            echo "  Removed (outdated): ${base}.${ext}"
            ((REMOVED++)) || true
        fi
    done
done
if [[ -f "$FIG_DIR/visual_conflict_report.json" ]]; then
    rm -f "$FIG_DIR/visual_conflict_report.json"
    echo "  Removed (outdated): visual_conflict_report.json"
    ((REMOVED++)) || true
fi

if [[ $REMOVED -eq 0 ]]; then
    echo "No outdated figure files found in $FIG_DIR."
else
    echo "Removed $REMOVED outdated file(s). Current figures unchanged."
fi
