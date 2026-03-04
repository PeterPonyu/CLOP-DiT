#!/usr/bin/env bash
# cleanup_heavy.sh — Delete regenerable outputs PLUS intermediate checkpoints.
#
# ⚠  Keeps only *_best.pth and *_final.pth model weights.
# ⚠  Also removes archive directories and old experiment results.
#
# To recover: re-train CLOP and DiT from scratch (04a → 04b).
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== CLOP-DiT Heavy Cleanup ==="
echo ""

# --- 1. Light cleanup first ---
echo "--- Phase 1: Light cleanup ---"
bash scripts/cleanup_light.sh
echo ""

# --- 2. Intermediate model checkpoints ---
echo "--- Phase 2: Intermediate checkpoints ---"
SAVED=0
REMOVED=0
for f in models/checkpoints/*_epoch_*.pth; do
    [ -f "$f" ] || continue
    rm -f "$f"
    REMOVED=$((REMOVED + 1))
done
echo "  ✓ Removed $REMOVED intermediate checkpoint(s)"
echo "  ♻ Kept: *_best.pth, *_final.pth, *_history.json"

# --- 3. Archive directories ---
echo ""
echo "--- Phase 3: Archive directories ---"
for d in results/archive configs/archive docs/archive docs/intermediate_analysis \
         scripts/archive models/checkpoints/archive data/cached_latents_v5.2/archive; do
    if [ -d "$d" ]; then
        SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
        rm -rf "$d"
        echo "  ✓ $d ($SIZE)"
    fi
done

# --- 4. Old experiment run directories ---
echo ""
echo "--- Phase 4: Old experiment runs ---"
for d in results/clop_v*/; do
    [ -d "$d" ] || continue
    SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
    rm -rf "$d"
    echo "  ✓ $d ($SIZE)"
done

# --- 5. Generated embeddings / expression (rebuilt by inference) ---
echo ""
echo "--- Phase 5: Generated data (rebuilt by inference) ---"
rm -f results/generated_embeddings.npy results/generated_labels.npy
rm -f results/generated_expression.npy results/generated_expression_labels.npy
rm -f results/real_expression.npy results/real_expression_labels.npy
echo "  ✓ results/generated_*.npy, results/real_expression*.npy"

# --- 6. Downstream results (rebuilt by evaluation) ---
rm -rf results/downstream/
echo "  ✓ results/downstream/"

# --- 7. Expression/generation metrics (rebuilt by evaluation) ---
rm -f results/expression_metrics.json results/generation_metrics.json
rm -f results/expression_gene_names.json results/generation_metadata.json
echo "  ✓ results/*_metrics.json, generation_metadata.json"

echo ""
TOTAL=$(du -sh . 2>/dev/null | cut -f1)
echo "=== Done. Repo size: $TOTAL ==="
echo "To fully rebuild: bash scripts/full_retrain_pipeline.sh"
echo "To rebuild outputs only: bash scripts/regenerate_report.sh"
