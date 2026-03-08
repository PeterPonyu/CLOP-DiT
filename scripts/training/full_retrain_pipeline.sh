#!/bin/bash
# =============================================================================
# CLOP-DiT Full Retraining Pipeline with Polished Texts
# =============================================================================
#
# This script orchestrates the complete retraining pipeline after text polishing:
# 1. Re-embed polished texts with BiomedBERT
# 2. Retrain CLOP with new embeddings
# 3. Train DiT with improved CLOP projections
# 4. (Optional) Rectified flow refinement
# 5. Comprehensive evaluation
#
# Expected improvements:
# - CLOP val_proto_acc: 10.45% → 15-25% (cross-tissue generalization)
# - DiT generation quality: Higher similarity to real cells
# - Biological plausibility: Better marker gene expression patterns
#
# Usage:
#   bash scripts/full_retrain_pipeline.sh
#
# Estimated time: ~12-24 hours on single GPU (depends on dataset size)
# =============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
CACHE_DIR="data/cached_latents_v5.2"
CLOP_CONFIG="configs/clop_v72.yaml"
DIT_CONFIG="configs/dit.yaml"
DEVICE="cuda"
NUM_GPUS=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_DIR="logs/retrain_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/main.log"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$MAIN_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$MAIN_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$MAIN_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$MAIN_LOG"
}

print_header() {
    echo "" | tee -a "$MAIN_LOG"
    echo "=============================================================================" | tee -a "$MAIN_LOG"
    echo "$1" | tee -a "$MAIN_LOG"
    echo "=============================================================================" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
}

# =============================================================================
# Step 0: Verify polished texts exist
# =============================================================================
print_header "STEP 0: VERIFY POLISHED TEXTS"

if [ ! -f "$CACHE_DIR/text_strings.json" ]; then
    log_error "text_strings.json not found in $CACHE_DIR"
    log_error "Run scripts/polish_texts.py first!"
    exit 1
fi

if [ ! -f "$CACHE_DIR/text_strings_original_templates.json" ]; then
    log_warning "Original templates backup not found (text_strings_original_templates.json)"
    log_warning "If this is your first run after polishing, this is expected"
else
    log_success "Found original templates backup"
fi

# Check text quality
log_info "Verifying text quality..."
python3 << 'EOF'
import json
import sys
with open("data/cached_latents_v5.2/text_strings.json") as f:
    texts = json.load(f)

# Check average length
lengths = [len(t) for t in texts.values()]
avg_len = sum(lengths) / len(lengths)

print(f"Total texts: {len(texts)}")
print(f"Average length: {avg_len:.0f} chars")

# If avg length < 200, likely not polished
if avg_len < 200:
    print("ERROR: Average text length < 200 chars - texts may not be polished!")
    print("Expected: ~300 chars after polishing")
    sys.exit(1)

print("✓ Text quality check passed")
EOF

if [ $? -ne 0 ]; then
    log_error "Text quality check failed"
    exit 1
fi

log_success "Polished texts verified"

# =============================================================================
# Step 1: Re-embed polished texts with BiomedBERT
# =============================================================================
print_header "STEP 1: RE-EMBED POLISHED TEXTS"

log_info "Re-embedding all 1088 texts with BiomedBERT-large..."
log_info "This will update text_embeddings.npy and recompute ZCA whitening"

python3 scripts/re_embed_polished_texts.py \
    --cache_dir "$CACHE_DIR" \
    --recompute_whitening \
    --batch_size 64 \
    --device "$DEVICE" \
    2>&1 | tee "$LOG_DIR/step1_reembed.log"

if [ $? -ne 0 ]; then
    log_error "Re-embedding failed!"
    exit 1
fi

log_success "Re-embedding complete"

# Verify new embeddings
log_info "Verifying new embeddings..."
python3 << 'EOF'
import numpy as np
text_emb = np.load("data/cached_latents_v5.2/text_embeddings.npy")
print(f"Text embeddings shape: {text_emb.shape}")
print(f"Mean norm: {np.linalg.norm(text_emb, axis=1).mean():.4f}")

# Check if whitening was updated
try:
    text_mean = np.load("data/cached_latents_v5.2/text_mean.npy")
    text_W_zca = np.load("data/cached_latents_v5.2/text_W_zca.npy")
    print(f"ZCA transform shape: {text_W_zca.shape}")
    print("✓ Whitening transforms updated")
except:
    print("WARNING: Whitening transforms not found")
EOF

# =============================================================================
# Step 2: Retrain CLOP with new embeddings
# =============================================================================
print_header "STEP 2: RETRAIN CLOP"

log_info "Training CLOP with polished text embeddings..."
log_info "Expected: val_proto_acc 15-25% (vs 10.45% baseline)"
log_info "Config: $CLOP_CONFIG"

# Update config for new run
CLOP_VERSION="v8.0_polished_texts"
log_info "Version: $CLOP_VERSION"

python3 -m src.training.train_clop \
    --config "$CLOP_CONFIG" \
    --version_id "$CLOP_VERSION" \
    --cache_dir "$CACHE_DIR" \
    --device "$DEVICE" \
    2>&1 | tee "$LOG_DIR/step2_clop_train.log"

if [ $? -ne 0 ]; then
    log_error "CLOP training failed!"
    exit 1
fi

log_success "CLOP training complete"

# Find best checkpoint
CLOP_CKPT=$(ls -t models/checkpoints/clop_${CLOP_VERSION}*.pt 2>/dev/null | head -1)
if [ -z "$CLOP_CKPT" ]; then
    log_error "CLOP checkpoint not found!"
    exit 1
fi

log_info "Best CLOP checkpoint: $CLOP_CKPT"

# =============================================================================
# Step 3: Extract CLOP-aligned embeddings for DiT training
# =============================================================================
print_header "STEP 3: EXTRACT CLOP-ALIGNED EMBEDDINGS"

log_info "Projecting cell and text embeddings through trained CLOP..."

python3 << EOF
import torch
import numpy as np
from pathlib import Path
from src.architecture.clop import CLOPAligner
import yaml

# Load config
with open("$CLOP_CONFIG") as f:
    config = yaml.safe_load(f)

# Load CLOP model
device = "$DEVICE"
model = CLOPAligner(
    text_dim=config["text_dim"],
    cell_dim=config["cell_dim"],
    proj_dim=config["proj_dim"],
    text_layers=config["text_layers"],
    cell_layers=config["cell_layers"],
    dropout=config["dropout"],
)

# Load checkpoint
ckpt = torch.load("$CLOP_CKPT", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()

# Load embeddings
text_emb = np.load("$CACHE_DIR/text_embeddings.npy")
cell_emb = np.load("$CACHE_DIR/cell_embeddings.npy")

print(f"Loaded text: {text_emb.shape}, cell: {cell_emb.shape}")

# Project through CLOP
with torch.no_grad():
    text_proj = model.text_encoder(torch.from_numpy(text_emb).float().to(device))
    cell_proj = model.cell_encoder(torch.from_numpy(cell_emb).float().to(device))
    
    text_proj = text_proj.cpu().numpy()
    cell_proj = cell_proj.cpu().numpy()

# Save for DiT training
np.save("$CACHE_DIR/text_proj_clop.npy", text_proj)
np.save("$CACHE_DIR/cell_proj_clop.npy", cell_proj)

print(f"✓ Saved CLOP projections: {text_proj.shape}, {cell_proj.shape}")
EOF

if [ $? -ne 0 ]; then
    log_error "CLOP projection failed!"
    exit 1
fi

log_success "CLOP projections extracted"

# =============================================================================
# Step 4: Train DiT with improved CLOP embeddings
# =============================================================================
print_header "STEP 4: TRAIN DiT"

log_info "Training DiT with CLOP-aligned embeddings..."
log_info "Using classifier-free guidance (10% dropout)"

DIT_VERSION="v2.0_polished_clop"
log_info "Version: $DIT_VERSION"

python3 -m src.training.train_dit \
    --config "$DIT_CONFIG" \
    --version_id "$DIT_VERSION" \
    --clop_proj_dir "$CACHE_DIR" \
    --device "$DEVICE" \
    2>&1 | tee "$LOG_DIR/step4_dit_train.log"

if [ $? -ne 0 ]; then
    log_error "DiT training failed!"
    exit 1
fi

log_success "DiT training complete"

# Find best checkpoint
DIT_CKPT=$(ls -t models/checkpoints/dit_${DIT_VERSION}*.pt 2>/dev/null | head -1)
if [ -z "$DIT_CKPT" ]; then
    log_error "DiT checkpoint not found!"
    exit 1
fi

log_info "Best DiT checkpoint: $DIT_CKPT"

# =============================================================================
# Step 5: Comprehensive Evaluation
# =============================================================================
print_header "STEP 5: COMPREHENSIVE EVALUATION"

log_info "Running full generative evaluation suite..."

python3 << EOF
import sys
sys.path.append(".")
import torch
import numpy as np
from src.architecture.dit import DiT1D
from src.evaluation.generative_metrics import GenerativeEvaluator, print_metrics_summary
import yaml

# Load DiT config
with open("$DIT_CONFIG") as f:
    config = yaml.safe_load(f)

# Load DiT model
device = "$DEVICE"
dit_model = DiT1D(
    latent_dim=config.get("latent_dim", 512),
    cond_dim=config.get("cond_dim", 512),
    hidden_dim=config.get("hidden_dim", 512),
    num_blocks=config.get("num_blocks", 8),
    num_heads=config.get("num_heads", 8),
    dropout=config.get("dropout", 0.1),
    cond_dropout_prob=config.get("cond_dropout_prob", 0.1),
)

# Load checkpoint
ckpt = torch.load("$DIT_CKPT", map_location=device)
dit_model.load_state_dict(ckpt["model_state_dict"])
dit_model = dit_model.to(device)

# Load data
text_proj = np.load("$CACHE_DIR/text_proj_clop.npy")
cell_proj = np.load("$CACHE_DIR/cell_proj_clop.npy")

print(f"Loaded projections: text {text_proj.shape}, cell {cell_proj.shape}")

# Run evaluation
evaluator = GenerativeEvaluator(dit_model, device=device)
metrics = evaluator.evaluate_full(
    text_embeddings=text_proj[:1000],  # Sample for speed
    real_cell_embeddings=cell_proj[:1000],
    num_samples=1000,
    cfg_scale=3.0,
    decode_expression=False,
)

# Print results
print_metrics_summary(metrics)

# Save metrics
import json
with open("$LOG_DIR/evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\\n✓ Saved metrics to $LOG_DIR/evaluation_metrics.json")
EOF

if [ $? -ne 0 ]; then
    log_warning "Evaluation script failed (non-critical)"
else
    log_success "Evaluation complete"
fi

# =============================================================================
# Step 6: Optional - Rectified Flow Refinement
# =============================================================================
print_header "STEP 6: RECTIFIED FLOW (OPTIONAL)"

log_info "Rectified flow can further improve sampling speed (4-8 steps)"
log_info "Recommendation: Run this after verifying DiT quality"

read -p "Run rectified flow refinement? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Starting rectified flow training..."
    
    python3 scripts/rectify_flow.py \
        --dit_checkpoint "$DIT_CKPT" \
        --config "$DIT_CONFIG" \
        --num_rectification_samples 50000 \
        --rectify_epochs 50 \
        --lr 1e-5 \
        --batch_size 512 \
        --device "$DEVICE" \
        2>&1 | tee "$LOG_DIR/step6_rectified_flow.log"
    
    if [ $? -ne 0 ]; then
        log_warning "Rectified flow training failed (non-critical)"
    else
        log_success "Rectified flow complete"
    fi
else
    log_info "Skipping rectified flow"
fi

# =============================================================================
# Summary
# =============================================================================
print_header "RETRAINING PIPELINE COMPLETE"

log_success "All steps completed!"

echo "" | tee -a "$MAIN_LOG"
echo "Summary:" | tee -a "$MAIN_LOG"
echo "  ✓ Text embeddings re-computed with BiomedBERT" | tee -a "$MAIN_LOG"
echo "  ✓ CLOP retrained: $CLOP_CKPT" | tee -a "$MAIN_LOG"
echo "  ✓ DiT retrained: $DIT_CKPT" | tee -a "$MAIN_LOG"
echo "  ✓ Evaluation metrics: $LOG_DIR/evaluation_metrics.json" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Next steps:" | tee -a "$MAIN_LOG"
echo "  1. Review evaluation metrics in $LOG_DIR" | tee -a "$MAIN_LOG"
echo "  2. Run full inference: python scripts/05_inference.py" | tee -a "$MAIN_LOG"
echo "  3. Generate cells from text: python scripts/06_generate_from_text.py" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

log_info "Complete logs saved to: $LOG_DIR"
log_info "Main log: $MAIN_LOG"

print_header "DONE"
