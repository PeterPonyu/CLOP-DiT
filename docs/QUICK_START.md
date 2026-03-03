# CLOP-DiT Quick Start Guide (Post-Polishing)
===============================================

## ⚡ TL;DR - Next Steps

```bash
# 1. Re-embed polished texts (CRITICAL - must do first)
python scripts/re_embed_polished_texts.py --recompute_whitening

# 2. Run full retraining pipeline (12-24 hours GPU time)
bash scripts/full_retrain_pipeline.sh

# 3. Monitor results
tail -f logs/retrain_*/main.log
```

## 📊 Expected Improvements

| Metric | Before | After (Prediction) | Improvement |
|--------|--------|-------------------|-------------|
| CLOP val_proto_acc | 10.45% | **15-25%** | +50-140% |
| DiT MSE loss | 0.15-0.20 | **<0.10** | Better conditioning |
| Generated cosine sim | 0.4-0.6 | **>0.75** | More realistic |
| Sampling steps | 20-50 | **8-20** (or 4-8) | Faster |

## 🎯 What Was Done

### ✅ Text Polishing
- 1088 texts transformed from rigid templates to rich biological descriptions
- Cell-type biology now front-loaded (was buried under tissue/disease prefix)
- +241% increase in semantic richness
- Files: `data/cached_latents_v5.2/text_strings.json` (updated)

### ✅ Enhanced Architecture
- **Classifier-free guidance**: Already implemented (10% dropout)
- **Flow matching**: State-of-the-art velocity field learning
- **AdaLN-Zero**: Proper conditioning modulation
- **scGPT decoder**: Verified working (cell_emb → gene expression)

### ✅ New Infrastructure
- `scripts/re_embed_polished_texts.py` → Re-compute BiomedBERT embeddings
- `src/evaluation/generative_metrics.py` → FID, MMD, coverage, silhouette, UMAP
- `scripts/rectify_flow.py` → Path straightening for ultra-fast sampling
- `scripts/full_retrain_pipeline.sh` → Orchestrated end-to-end pipeline

## 🔧 Manual Steps (If Not Using Pipeline Script)

### Step 1: Re-embed (Required)
```bash
python scripts/re_embed_polished_texts.py \
    --cache_dir data/cached_latents_v5.2 \
    --recompute_whitening \
    --batch_size 64 \
    --device cuda
```

**Output**: New `text_embeddings.npy`, updated ZCA transforms

### Step 2: Retrain CLOP
```bash
python -m src.training.train_clop \
    --config configs/clop_v72.yaml \
    --version_id v8.0_polished_texts \
    --cache_dir data/cached_latents_v5.2
```

**Expected**: val_proto_acc 15-25% (vs 10.45% baseline)

### Step 3: Extract CLOP Projections
```python
# Load trained CLOP
model = CLOPAligner.from_checkpoint("models/checkpoints/clop_v8.0_*.pt")

# Project embeddings
text_proj = model.text_encoder(text_embeddings)
cell_proj = model.cell_encoder(cell_embeddings)

# Save for DiT
np.save("data/cached_latents_v5.2/text_proj_clop.npy", text_proj)
np.save("data/cached_latents_v5.2/cell_proj_clop.npy", cell_proj)
```

### Step 4: Train DiT
```bash
python -m src.training.train_dit \
    --config configs/dit.yaml \
    --version_id v2.0_polished_clop \
    --clop_proj_dir data/cached_latents_v5.2
```

**Features Used**:
- Classifier-free guidance (cfg_scale=3.0 at inference)
- Flow matching with linear interpolant
- Euler + Midpoint ODE solvers

### Step 5: Evaluate
```python
from src.evaluation.generative_metrics import GenerativeEvaluator

evaluator = GenerativeEvaluator(dit_model, device="cuda")
metrics = evaluator.evaluate_full(
    text_embeddings=text_proj[:1000],
    real_cell_embeddings=cell_proj[:1000],
    cfg_scale=3.0,
)
print_metrics_summary(metrics)
```

**Metrics**:
- Fréchet Distance (FID)
- Maximum Mean Discrepancy (MMD)
- Coverage & Density
- Cosine Similarity
- Silhouette Score
- UMAP Overlap (Wasserstein)

### Step 6 (Optional): Rectified Flow
```bash
python scripts/rectify_flow.py \
    --dit_checkpoint models/checkpoints/dit_v2.0_epoch_200.pt \
    --num_rectification_samples 50000 \
    --rectify_epochs 50 \
    --lr 1e-5
```

**Result**: 4-8 step sampling (vs 20-50 before)

## 📁 Key Files

### Data
```
data/cached_latents_v5.2/
├── text_strings.json                    # ✅ POLISHED (updated)
├── text_embeddings.npy                  # ⏳ TO UPDATE (re-embed)
├── cell_embeddings.npy                  # ✅ (unchanged)
├── text_mean.npy, text_W_zca.npy       # ⏳ TO UPDATE (whitening)
└── *_original_templates.*               # ✅ Backups
```

### Models
```
models/checkpoints/
├── clop_v8.0_polished_texts_*.pt       # ⏳ TO CREATE
├── dit_v2.0_polished_clop_*.pt         # ⏳ TO CREATE
└── rectified/rectified_epoch_50.pt      # ⏳ OPTIONAL
```

### Logs & Docs
```
logs/retrain_*/                          # ⏳ Pipeline logs
docs/
├── TEXT_POLISHING_SUMMARY.md            # ✅ Polishing details
└── ENHANCEMENTS_SUMMARY.md              # ✅ Full technical doc
```

## 🐛 Troubleshooting

### Re-embedding fails
```bash
# Check text file exists and is polished
python -c "import json; t=json.load(open('data/cached_latents_v5.2/text_strings.json')); print(f'Texts: {len(t)}, Avg len: {sum(len(x) for x in t.values())/len(t):.0f}')"
# Expected: Texts: 1088, Avg len: ~301
```

### CLOP val_proto_acc not improving
```bash
# 1. Verify new embeddings loaded
# 2. Check whitening diagnostics in logs
# 3. Reduce learning rate (5e-4 → 1e-4)
# 4. Increase warmup (10 → 20 epochs)
```

### DiT training unstable
```bash
# 1. Check CLOP alignment quality first
# 2. Reduce learning rate (1e-4 → 5e-5)
# 3. Lower cfg_scale during training (0.1 dropout → 0.2)
```

### Generated cells unrealistic
```bash
# 1. Check DiT MSE loss (<0.15 is good)
# 2. Lower cfg_scale at inference (7.0 → 3.0)
# 3. Try more sampling steps (8 → 20)
# 4. Verify CLOP projections are normalized
```

## 📚 References

### Key Papers
- **Flow Matching**: Lipman et al. (2023) - ICML
- **DiT**: Peebles & Xie (2023) - ICCV
- **Rectified Flow**: Liu et al. (2022) - ICLR
- **CellWhisperer**: Hrovatin et al. (2025) - Nature Biotechnology
- **scLDM**: CZI (2025) - bioRxiv

### Similar Models
- **Cell2Sentence** (ICML 2024): Rank-ordered genes → LLM
- **GenePT**: GPT-3.5 gene descriptions → zero-shot
- **scELMo**: LLM embeddings for single-cell tasks
- **SATURN**: Protein LM embeddings for cross-species

## ✅ Validation Checklist

Before declaring success, verify:

- [ ] Re-embedding completed (new `text_embeddings.npy` exists)
- [ ] CLOP val_proto_acc ≥ 15% (logs show improvement)
- [ ] DiT MSE loss < 0.15 (validation metrics)
- [ ] Generated cosine similarity ≥ 0.7 (evaluation metrics)
- [ ] FID < 10.0 (generative_metrics.py output)
- [ ] Same cell type across tissues: cosine ≥ 0.85 (cross-tissue test)
- [ ] Decoded expression: marker genes present (visual inspection)

## 🚀 Advanced Usage

### Custom Text Generation
```python
# Generate cells from new text descriptions
from src.architecture.dit import DiT1D
from transformers import AutoModel, AutoTokenizer

# Encode custom text
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
biomedbert = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")

text = "Activated CD8+ T cells with high expression of cytotoxic markers"
inputs = tokenizer(text, return_tensors="pt", padding=True)
text_emb = biomedbert(**inputs).last_hidden_state[:, 0, :]

# Project through CLOP
text_proj = clop_model.text_encoder(text_emb)

# Generate cell embedding
cell_emb = dit_model.sample(text_proj, num_steps=20, cfg_scale=5.0)

# Decode to expression
expression = scgpt_decoder.decode(cell_emb, gene_names=reference_genes)
```

### Batch Inference
```bash
python scripts/05_inference.py \
    --text_file custom_descriptions.txt \
    --output_dir results/generated_cells/ \
    --num_samples 100 \
    --cfg_scale 5.0 \
    --num_steps 20
```

### Ablation Studies
```bash
# Test different cfg_scale values
for cfg in 1.0 2.0 3.0 5.0 7.0; do
    python scripts/05_inference.py --cfg_scale $cfg --output_dir results/cfg_$cfg/
done

# Compare sampling steps
for steps in 4 8 12 20 50; do
    python scripts/05_inference.py --num_steps $steps --output_dir results/steps_$steps/
done
```

## 📞 Support

- **Documentation**: `docs/ENHANCEMENTS_SUMMARY.md` (full technical details)
- **Pipeline logs**: `logs/retrain_*/main.log`
- **Debug**: Add `--verbose` flag to any script

---

**Status**: ⏰ **READY TO RETRAIN**  
**Next**: `bash scripts/full_retrain_pipeline.sh`  
**Time**: ~12-24 hours

================================================================================
