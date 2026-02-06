# CLOP-DiT: Contrastive Language-Omics Pre-training + Diffusion Transformer

> **Text-conditioned generation of single-cell gene expression profiles via Flow Matching**

---

## Architecture Overview

```
User Text ─→ PubMedBERT ─→ CLOP Projector ─→ Condition c
                                                   ↓
                         z₀ ~ N(0,I) ─→ DiT(z_t, t, c) ─→ ODE Integrate ─→ z₁ (Cell Embedding)
                                                                                    ↓
                                                                      scGPT Decoder / Linear Decoder
                                                                                    ↓
                                                                     Gene Expression Matrix (Genes × Cells)
```

## Enhanced Action Plan (v2.0)

### What's improved over the initial draft

| Aspect | Initial Draft | Enhanced Plan (This Version) |
|--------|--------------|------------------------------|
| **Cell Encoder** | scGPT only | scGPT + PCA fallback for rapid prototyping |
| **DiT Architecture** | Flat vector input | **Pseudo-token decomposition** — reshapes embedding into a token sequence for proper attention |
| **Conditioning** | Simple concatenation | **AdaLN-Zero** with zero-init gates (proven in DiT, Flux) |
| **Training stability** | Basic Adam | AdamW + Cosine warmup + EMA model averaging + AMP |
| **Guidance** | None | **Classifier-Free Guidance** with condition dropout |
| **ODE Solver** | Euler only | Euler + **Midpoint** (2nd-order, more accurate) |
| **CLOP Loss** | Basic InfoNCE | InfoNCE + **learnable temperature** + label smoothing + optional EMA targets |
| **Evaluation** | Visual only | **Fréchet Distance** + MMD + Coverage/Density + per-dim KL + retrieval R@K |
| **Data pipeline** | Manual | **GEOparse automated fetching** + SFT batch processing + latent caching |
| **Decoder** | scGPT only | scGPT + **LinearDecoder fallback** (trainable MLP for HVGs) |
| **Scalability** | Not addressed | Memory-mapped .npy cache, batch processing, gradient checkpointing ready |

### Phase 0: Environment Setup (✅ Done)

```bash
conda activate clopdit
cd /path/to/CLOP-DiT
```

### Phase 1: Data Acquisition & Cleaning

**Goal**: Build `(text_description, cell_matrix)` paired dataset.

```bash
# 1a. Fetch GEO metadata (start with 10 lung cancer datasets)
python scripts/01_fetch_data.py --gse_ids GSE131907 GSE148071

# 1b. Clean text via SFT (requires ~8GB VRAM for 4-bit Llama-3)
python scripts/02_clean_text.py \
    --input data/processed_h5ad/metadata_raw.json \
    --output data/processed_h5ad/metadata_structured.json
```

**Key design decision**: Even without Llama-3 SFT, you can manually write JSON metadata for 10 pilot datasets. The SFT step is for *scaling* to thousands of datasets later.

### Phase 2: Latent Caching (Critical Optimization)

**Goal**: Eliminate heavy models from training loops.

```bash
# 2. Pre-compute embeddings (PCA fallback for rapid prototyping)
python scripts/03_cache_latents.py \
    --h5ad_dir data/processed_h5ad \
    --metadata data/processed_h5ad/metadata_structured.json \
    --cell_encoder pca \
    --cell_dim 512

# When scGPT weights are available:
python scripts/03_cache_latents.py --cell_encoder scgpt --scgpt_dir models/scgpt_weights
```

**Output**: `data/cached_latents/` with `.npy` files (~2-5GB total for 100K cells).

### Phase 3: CLOP Contrastive Alignment

**Goal**: Align text and cell embeddings in shared space.

```bash
python scripts/04a_train_clop.py \
    --cache_dir data/cached_latents \
    --batch_size 256 \
    --epochs 100 \
    --lr 3e-4 \
    --proj_dim 256
```

**Monitoring**:
- Loss should drop rapidly in first 10 epochs
- `acc_t2c` and `acc_c2t` should reach >0.8 for small datasets
- Temperature should stabilize around 0.03-0.1
- If loss oscillates: increase batch size (minimum 64)

### Phase 4: DiT Flow Matching Training

**Goal**: Learn the velocity field for conditional generation.

```bash
python scripts/04b_train_dit.py \
    --cache_dir data/cached_latents \
    --batch_size 512 \
    --epochs 200 \
    --hidden_dim 384 \
    --num_blocks 8 \
    --lr 1e-4
```

**Model sizing guide**:

| Config | Blocks | Hidden | Heads | Tokens | Params | VRAM |
|--------|--------|--------|-------|--------|--------|------|
| Small  | 4      | 256    | 4     | 16     | ~15M   | ~4GB |
| Base   | 8      | 384    | 6     | 16     | ~55M   | ~8GB |
| Large  | 12     | 512    | 8     | 16     | ~120M  | ~16GB |

**Monitoring**:
- MSE loss will drop to very low values (0.01-0.1) since we're fitting linear interpolations
- Cosine similarity between predicted and target velocities should approach 0.95+
- Check generation quality every 10 epochs via UMAP visualization

### Phase 5: Inference

```bash
python scripts/05_inference.py \
    --prompt "Lung adenocarcinoma treated with cisplatin from Homo sapiens" \
    --num_cells 500 \
    --cfg_scale 3.0 \
    --num_steps 4 \
    --solver euler \
    --output generated_luad_cisplatin.h5ad
```

---

## Project Structure

```
CLOP-DiT/
├── configs/
│   ├── clop.yaml              # CLOP alignment config
│   └── dit.yaml               # DiT training config
├── data/
│   ├── raw_geo/               # Raw GEO downloads
│   ├── processed_h5ad/        # Preprocessed h5ad + metadata JSON
│   └── cached_latents/        # Pre-computed .npy embeddings
│       ├── cell_embeddings.npy
│       ├── text_embeddings.npy
│       ├── projected_text.npy  # (After CLOP training)
│       ├── sample_ids.npy
│       └── manifest.json
├── models/
│   ├── checkpoints/           # Trained model weights
│   │   ├── clop_best.pth
│   │   └── dit_best.pth
│   ├── scgpt_weights/         # Downloaded scGPT weights
│   └── sft_adapter/           # Fine-tuned Llama-3 LoRA
├── scripts/
│   ├── 01_fetch_data.py
│   ├── 02_clean_text.py
│   ├── 03_cache_latents.py
│   ├── 04a_train_clop.py
│   ├── 04b_train_dit.py
│   └── 05_inference.py
├── src/
│   ├── architecture/
│   │   ├── dit.py             # 1D DiT with AdaLN-Zero
│   │   ├── clop.py            # CLOP aligner (projectors + InfoNCE)
│   │   └── decoder.py         # scGPT wrapper + Linear fallback
│   ├── data_pipeline/
│   │   ├── geo_fetcher.py     # GEO data acquisition
│   │   ├── text_cleaner.py    # SFT text cleaning
│   │   ├── cache_builder.py   # Latent pre-computation
│   │   └── dataset.py         # PyTorch Dataset classes
│   ├── training/
│   │   ├── train_clop.py      # CLOP trainer
│   │   ├── train_dit.py       # DiT trainer
│   │   └── schedulers.py      # LR schedulers
│   ├── evaluation/
│   │   ├── metrics.py         # FD, MMD, Coverage, KL, R@K
│   │   └── visualizer.py      # UMAP, training curves, distributions
│   └── utils/
│       ├── helpers.py         # Seed, device, parameter counting
│       └── logging_config.py  # Structured logging
├── notebooks/                 # Jupyter exploration notebooks
├── requirements.txt
├── setup.py
└── README.md
```

---

## Technical Details

### DiT Architecture: Pseudo-Token Decomposition

Unlike image DiTs that operate on spatial patch tokens, our 1D-DiT reshapes the flat cell embedding into a short pseudo-token sequence:

$$z \in \mathbb{R}^{512} \rightarrow \text{reshape} \rightarrow X \in \mathbb{R}^{16 \times 32} \rightarrow \text{project} \rightarrow H \in \mathbb{R}^{16 \times 384}$$

This allows the self-attention mechanism to model **inter-dimensional correlations** within the embedding — crucial for capturing gene regulatory relationships.

### Flow Matching Formulation

Given paired data $(z_0, z_1)$ where $z_0 \sim \mathcal{N}(0, I)$ and $z_1$ is a real cell embedding:

$$z_t = (1-t)z_0 + tz_1, \quad t \in [0, 1]$$

$$v_{\text{target}} = z_1 - z_0$$

$$\mathcal{L} = \mathbb{E}_{t, z_0, z_1} \left[ \| v_\theta(z_t, t, c) - v_{\text{target}} \|^2 \right]$$

At inference, integrate from $t=0$ to $t=1$:

$$z_{t+\Delta t} = z_t + v_\theta(z_t, t, c) \cdot \Delta t$$

### Classifier-Free Guidance

During training, the condition is randomly dropped with probability $p = 0.1$, replaced by a learnable null token $c_\varnothing$.

At inference:

$$v_{\text{guided}} = v_\theta(z_t, t, c_\varnothing) + s \cdot \left[ v_\theta(z_t, t, c) - v_\theta(z_t, t, c_\varnothing) \right]$$

where $s$ is the guidance scale (default 3.0).

---

## Evaluation Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **Fréchet Distance** | Distribution similarity (mean + covariance) | Lower is better |
| **MMD** | Kernel-based distribution discrepancy | Lower is better |
| **Coverage** | Fraction of real modes covered by generated | Higher (→1.0) |
| **Density** | Fidelity of generated samples | ~1.0 (too high = mode collapse) |
| **Per-dim KL** | Marginal distribution matching per embedding dimension | Lower is better |
| **R@K** | CLOP retrieval accuracy (text↔cell) | Higher is better |

---

## External Resources Required

1. **PubMedBERT** — auto-downloaded by HuggingFace `transformers`
2. **scGPT weights** — download from [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT), place in `models/scgpt_weights/`
3. **Llama-3-8B** (optional) — auto-downloaded by Unsloth (requires `pip install unsloth`)

---

## Citation

```bibtex
@software{clopdit2026,
  author = {Fu, Zeyu},
  title = {CLOP-DiT: Contrastive Language-Omics Pre-training + Diffusion Transformer},
  year = {2026},
  url = {https://github.com/PeterPonyu/CLOP-DiT}
}
```

## License

MIT License
