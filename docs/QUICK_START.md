# CLOP-DiT Quick Start Guide
===============================================

## TL;DR - Regenerate the Full Report

```bash
# Generate all 19 panels (A-S) + combined PDF + article symlinks
bash scripts/pipeline/regenerate_report.sh

# Or skip embedding generation if results already exist
bash scripts/pipeline/regenerate_report.sh --skip-gen
```

To build the article PDF after figures are ready:
```bash
bash scripts/pipeline/build_article.sh
```

For a single entry point that can run from data through to article (or from a given stage), use:
```bash
python scripts/pipeline/run_pipeline.py --from generate   # report only (like regenerate_report.sh)
python scripts/pipeline/run_pipeline.py --stage all --build-article   # full pipeline + PDF
```

Paths are controlled by `configs/pipeline.yaml` and env vars (`CLOPDIT_CACHE_DIR`, etc.).

The pipeline automatically verifies all 17 article figures and creates symlinks in `articles/figures/`. To verify without regenerating:

```bash
bash scripts/pipeline/verify_article_figures.sh --check   # verify only
bash scripts/pipeline/verify_article_figures.sh            # verify + recreate symlinks
```

## Current Pipeline (9 Steps)

`regenerate_report.sh` orchestrates:

0. **Architecture figure** → `scripts/analysis/generate_architecture_figure.py`
1. **Generate embeddings** → `scripts/inference/generate_embeddings.py`
2. **Decode gene expression** → `scripts/analysis/decode_expression.py`
3. **Diversity diagnostics** (Panels J+K) → `scripts/analysis/diversity_diagnostics.py`
4. **Conditioning analysis** (Panels L+M) → `scripts/analysis/conditioning_analysis.py`
5. **Downstream biology** (Panels P/Q/R) → `python -m src.evaluation.downstream_biology`
6. **Model benchmarking** (Panel S) → `python -m src.evaluation.model_benchmarking`
7. **Visualization** (all 19 panels + 5 merged figures) → `python -m src.visualization.results_visualizer`
8. **Verify + symlink** (article figures) → `scripts/pipeline/verify_article_figures.sh`

**Output:** `results/figures/panel_a_*.png` through `panel_s_*.png`, 5 merged `fig_*.pdf`, `clop_dit_full_report.pdf`, and 17 symlinks in `articles/figures/`.

## Manual Steps (If Not Using Pipeline Script)

### Step 1: Retrain CLOP
```bash
python -m src.training.train_clop \
    --config configs/clop_v9.3.yaml \
    --cache_dir data/cached_latents_v5.2
```

**Expected**: val_proto_acc 15-25% (vs 10.45% baseline)

### Step 2: Extract CLOP Projections
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

### Step 3: Train DiT
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

### Step 4: Evaluate
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

## Key Files

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
logs/retrain_*/                          # Pipeline logs
docs/
├── CLOP-DiT_Evaluation_Report.md       # Evaluation details
└── DIRECTORY_CLEANUP_POLICY.md          # Cleanup and rebuild commands
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
python scripts/inference/05_inference.py \
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
    python scripts/inference/05_inference.py --cfg_scale $cfg --output_dir results/cfg_$cfg/
done

# Compare sampling steps
for steps in 4 8 12 20 50; do
    python scripts/inference/05_inference.py --num_steps $steps --output_dir results/steps_$steps/
done
```

## Support

- **Documentation**: `docs/CLOP-DiT_Evaluation_Report.md` (evaluation details)
- **Cleanup policy**: `docs/DIRECTORY_CLEANUP_POLICY.md`
- **Pipeline logs**: `logs/retrain_*/main.log`
- **Debug**: Add `--verbose` flag to any script

---

**Status**: Full 19-panel evaluation pipeline operational.
**Regenerate**: `bash scripts/pipeline/regenerate_report.sh`
