# CLOP-DiT: Contrastive Language-Omics Pre-training + Diffusion Transformer

> **Text-conditioned generation of single-cell gene expression profiles via Flow Matching**

---

## Architecture Overview

```
User Text ─→ BiomedBERT-large (1024-d) ─→ CLOP Text Projector ─→ Condition c (256-d)
                                                                       ↓
                              z₀ ~ N(0,I) ─→ DiT(z_t, t, c) ─→ ODE Integrate ─→ z₁ (Cell Embedding, 512-d)
                                                                                         ↓
                                                                          scGPT generate() Decoder
                                                                          [CLS] injection → Transformer → ExprDecoder
                                                                                         ↓
                                                                          Gene Expression Matrix (Genes × Cells)
```

### Pipeline Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  55 h5ad files   │ ──→ │ 00_prepare_all_  │ ──→ │ processed_h5ad/    │
│ (Cancer + Dev)   │     │    data.py       │     │ ~50 × ~3K cells    │
│ CancerDatasets/  │     │ Curated bio text │     │ + metadata JSON    │
│ CancerDatasets2/ │     │ QC + HVG + norm  │     │ ~50 unique texts   │
│ DevelopmentData/ │     │ Skip invalid:    │     │ (5 datasets        │
│ DevelopmentData2/│     │  normalized-only │     │  filtered out)     │
└─────────────────┘     │  Ensembl IDs     │     └────────┬───────────┘
                         └──────────────────┘              │
                                                           ▼
┌──────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ scGPT pan-cancer │ ──→ │ 03_cache_latents │ ──→ │ cached_latents/    │
│ (51.9M, 5.7M     │     │     .py          │     │ cell_emb (N×512)   │
│ cancer cells)    │     │ scGPT + BiomedBERT│    │ text_emb (N×1024)  │
│                  │     │      -large       │     │ sample_ids, meta   │
│ BiomedBERT-large │     └──────────────────┘     └────────┬───────────┘
│ text encoder     │                                       │
│ (1024-d)         │                                       ▼
└──────────────────┘     ┌──────────────────┐     ┌────────────────────┐
                         │ 04a_train_clop   │ ──→ │ CLOP Aligner       │
                         │     .py          │     │ proj_text (N×256)  │
                         │ InfoNCE + temp   │     │ clop_best.pth      │
                         │ 200 epochs       │     └────────┬───────────┘
                         └──────────────────┘              │
                         ┌──────────────────┐     ┌────────▼───────────┐
                         │ 04b_train_dit    │ ──→ │ DiT (22.1M)        │
                         │     .py          │     │ dit_best.pth (EMA) │
                         │ Flow Matching    │     │ 500 epochs         │
                         │ 500 epochs       │     └────────┬───────────┘
                         └──────────────────┘              │
                         ┌──────────────────┐     ┌────────▼───────────┐
                         │ 05_inference.py  │ ──→ │ Generated cells    │
                         │ Text → CLOP → DiT│     │ (N × 512) → scGPT │
                         │ → scGPT decode  │     │ → gene expression  │
                         └──────────────────┘     └────────┬───────────┘
                                                           │
                         ┌──────────────────┐     ┌────────▼───────────┐
                         │ 06_evaluate.py   │ ──→ │ FD, MMD, R@K       │
                         │ Metrics + Viz    │     │ Gene correlation   │
                         │ + Expression eval│     │ UMAP figures       │
                         └──────────────────┘     └────────────────────┘
```

---

## Changelog

### v0.2.0 → v0.3.0

| Aspect | v0.2.0 | v0.3.0 |
|--------|--------|--------|
| **Datasets** | 43 datasets (3 dirs), 81K cells | **~50 datasets** (4 dirs, 55 total − 5 filtered), ~150K cells |
| **Data filtering** | None (some invalid data) | Auto-skip normalized-only (2) + Ensembl ID (3) datasets |
| **Cell encoder** | scGPT whole-human (33M cells) | **scGPT pan-cancer** (5.7M cancer cells, tumor-specialized) |
| **Text encoder** | BiomedBERT-base (768-d, 110M) | **BiomedBERT-large** (1024-d, 340M) |
| **Decoder** | Broken `model.decoder(batch)` — just ExprDecoder head on raw embedding | **scGPT `generate()`**: inject cell_emb at [CLS] → full transformer → ExprDecoder per gene |
| **CLOP training** | 50 epochs | **200 epochs** |
| **DiT training** | 100 epochs | **500 epochs** |
| **Output** | 512-d embeddings only | **Gene expression matrix** via scGPT decoding + embeddings |
| **Evaluation** | FD, MMD, R@K on embeddings | + **Gene expression correlation** (Pearson, Spearman per-gene/per-cell) |
| **Figures** | 5 figures (embedding space) | + **Expression heatmap**, **gene correlation scatter** |

### v0.1.0 → v0.2.0

| Aspect | v0.1.0 | v0.2.0 |
|--------|--------|--------|
| **Datasets** | 3 cancer datasets, 5,833 cells | **43 datasets** (28 cancer + 15 development), **81,225 cells** |
| **Text descriptions** | 3 auto-generated from filenames | **43 curated biological descriptions** |
| **Cell encoder** | PCA fallback as default | **scGPT mandatory** (51.9M, 512-d) |
| **Text encoder** | BiomedBERT loaded but trivial texts | **BiomedBERT with meaningful biomedical text** (768-d) |
| **CLOP training** | 3 unique negatives | **43 unique text classes** |
| **Evaluation pipeline** | metrics.py existed but never called | **06_evaluate.py** — full evaluation with 5 figures |

### Key Design Decisions (v0.3)

1. **scGPT pan-cancer over whole-human**: The pan-cancer checkpoint was pretrained on 5.7M cancer cells specifically. Since 28/50 of our datasets are cancer, this specialized model provides better cancer cell embeddings while still generalizing to development datasets.

2. **scGPT generate() for decoding**: The v0.2 decoder was fundamentally broken — it applied ExprDecoder directly to the 512-d embedding as if it were a single transformer output token. The correct approach re-injects the cell embedding at the [CLS] position, re-runs the full 12-layer transformer with all gene tokens, and then applies ExprDecoder to each gene position to predict per-gene expression. This is the native scGPT reconstruction pathway.

3. **BiomedBERT-large (1024-d)**: The base model (768-d, 110M params) may not capture fine-grained distinctions between similar cancer types. The large model (1024-d, 340M params) has 3× more parameters and a richer representation space for discriminating between subtle biological descriptions.

4. **Dataset filtering**: scGPT requires raw integer counts for its rank-based binning (51 bins). Datasets with only normalized data (no raw counts) or Ensembl IDs (not in scGPT's gene symbol vocabulary) are automatically filtered.

---

## v0.2.0 Results

### CLOP Alignment
| Metric | Value |
|--------|-------|
| Text→Cell R@1 | 0.9535 |
| Cell→Text R@1 | 1.0000 |
| R@3 (both) | 1.0000 |
| Mean cosine similarity | 0.6397 ± 0.1187 |

### DiT Generation Quality
| Metric | Value | Direction |
|--------|-------|-----------|
| Fréchet Distance | 123.84 | Lower ↓ |
| MMD (RBF) | 0.262 | Lower ↓ |
| Coverage | 0.024 | Higher ↑ |
| Density | 0.062 | ~1.0 |
| Mean KL divergence | 2.133 | Lower ↓ |
| Paired cosine similarity | 0.855 | Higher ↑ |
| Val loss (flow matching) | 0.662 | Lower ↓ |
| Val cosine similarity | 0.818 | Higher ↑ |

### Training Configuration
| Component | Setting |
|-----------|---------|
| Total cells | 81,225 |
| Datasets | 43 (28 cancer + 15 development) |
| Cell encoder | scGPT (512-d, 51.9M params) |
| Text encoder | BiomedBERT (768-d) |
| CLOP | 2M params, proj_dim=256, 50 epochs, batch=512 |
| DiT | 22.1M params, 8 blocks, hidden=384, 100 epochs, batch=512, EMA=0.9999 |
| Hardware | NVIDIA RTX 5090 Laptop GPU (25.1 GB VRAM) |

---

## Quick Start

### Environment Setup

```bash
conda activate clopdit
cd CLOP-DiT
```

### Step 1: Prepare Data (~50 datasets from 4 directories)

```bash
python scripts/00_prepare_all_data.py \
    --output_dir data/processed_h5ad \
    --max_cells 3000
```

### Step 2: Cache Latent Embeddings (scGPT pan-cancer + BiomedBERT-large)

```bash
python scripts/03_cache_latents.py \
    --h5ad_dir data/processed_h5ad \
    --metadata data/processed_h5ad/metadata_structured.json \
    --output_dir data/cached_latents \
    --cell_encoder scgpt \
    --scgpt_dir models/scgpt_pancancer \
    --text_encoder microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract
```

### Step 3: Train CLOP Alignment (200 epochs)

```bash
python scripts/04a_train_clop.py \
    --config configs/clop.yaml
```

### Step 4: Train DiT (500 epochs)

```bash
python scripts/04b_train_dit.py \
    --config configs/dit.yaml
```

### Step 5: Generate Cells (with gene expression decoding)

```bash
python scripts/05_inference.py \
    --prompt "CD8+ cytotoxic T cells from human lung adenocarcinoma tumor microenvironment" \
    --num_cells 500 \
    --decode_expression \
    --reference_h5ad data/processed_h5ad/GSE123902_LungAdreHmCancer_processed.h5ad \
    --output generated_cells.h5ad
```

### Step 6: Evaluate (including gene expression quality)

```bash
python scripts/06_evaluate.py \
    --output_dir figures \
    --decode_expression \
    --reference_h5ad data/processed_h5ad/GSE123902_LungAdreHmCancer_processed.h5ad
```

---

## Project Structure

```
CLOP-DiT/
├── configs/
│   ├── clop.yaml              # CLOP config (text_dim=1024, 200 epochs)
│   └── dit.yaml               # DiT config (500 epochs)
├── data/
│   ├── processed_h5ad/        # ~50 preprocessed h5ad + metadata JSON
│   └── cached_latents/        # Pre-computed .npy embeddings
│       ├── cell_embeddings.npy   # (N, 512) scGPT pan-cancer
│       ├── text_embeddings.npy   # (N, 1024) BiomedBERT-large
│       ├── projected_text.npy    # (N, 256) CLOP-projected
│       ├── sample_ids.npy        # (N,) dataset IDs
│       ├── metadata.json         # ID → text description mapping
│       └── manifest.json         # Cache statistics
├── figures/                   # Evaluation outputs
│   ├── clop_alignment_umap.png
│   ├── generation_comparison_umap.png
│   ├── dim_distributions.png
│   ├── expression_comparison.png    # v0.3: gene expression heatmap
│   ├── gene_correlation.png         # v0.3: per-gene correlation
│   ├── clop_training_curves.png
│   ├── dit_training_curves.png
│   └── evaluation_metrics.json
├── models/
│   ├── checkpoints/           # Trained weights
│   │   ├── clop_best.pth
│   │   ├── dit_best.pth (EMA)
│   │   ├── clop_history.json
│   │   └── dit_history.json
│   ├── scgpt_pancancer/       # v0.3: scGPT pan-cancer pretrained (5.7M cancer cells)
│   │   ├── best_model.pt (205MB)
│   │   ├── vocab.json (60K+ genes)
│   │   └── args.json
│   └── scgpt_human/           # scGPT whole-human pretrained (33M cells)
│       ├── best_model.pt (196MB)
│       ├── vocab.json (60697 genes)
│       └── args.json
├── scripts/
│   ├── 00_prepare_all_data.py  # v0.3: 55 datasets, 4 dirs, auto-filter
│   ├── 03_cache_latents.py     # scGPT pan-cancer + BiomedBERT-large
│   ├── 04a_train_clop.py
│   ├── 04b_train_dit.py
│   ├── 05_inference.py         # v0.3: scGPT generate() decoding
│   └── 06_evaluate.py          # v0.3: gene expression evaluation
├── src/
│   ├── architecture/
│   │   ├── dit.py             # 1D DiT with AdaLN-Zero (22.1M)
│   │   ├── clop.py            # CLOP aligner
│   │   ├── scgpt_embed.py     # Standalone scGPT encoder + generate() decoder
│   │   └── decoder.py         # v0.3: scGPT generate() decoder wrapper
│   ├── data_pipeline/
│   │   ├── cache_builder.py   # BiomedBERT-large default (1024-d)
│   │   ├── dataset.py         # CLOPDataset, DiTDataset
│   │   ├── geo_fetcher.py     # GEO data acquisition
│   │   └── text_cleaner.py    # SFT text cleaning
│   ├── training/
│   │   ├── train_clop.py      # CLOPTrainer (saves text_dim/cell_dim in ckpt)
│   │   ├── train_dit.py       # DiTTrainer
│   │   └── schedulers.py      # CosineWarmupScheduler
│   ├── evaluation/
│   │   ├── metrics.py         # FD, MMD, Coverage, Density, KL, R@K
│   │   └── visualizer.py      # UMAP, tSNE, training curves
│   └── utils/
│       ├── helpers.py         # Seed, device, parameter counting
│       └── logging_config.py  # Structured logging
├── requirements.txt
├── setup.py
└── README.md
```

---

## Technical Details

### DiT Architecture: Pseudo-Token Decomposition

The 1D-DiT reshapes the flat cell embedding into a pseudo-token sequence for self-attention:

$$z \in \mathbb{R}^{512} \rightarrow \text{reshape} \rightarrow X \in \mathbb{R}^{16 \times 32} \rightarrow \text{project} \rightarrow H \in \mathbb{R}^{16 \times 384}$$

This enables the self-attention mechanism to model **inter-dimensional correlations** within the embedding — crucial for capturing gene regulatory relationships encoded by scGPT.

### Flow Matching Formulation

Given paired data $(z_0, z_1)$ where $z_0 \sim \mathcal{N}(0, I)$ and $z_1$ is a real scGPT cell embedding:

$$z_t = (1-t)z_0 + tz_1, \quad t \in [0, 1]$$
$$v_{\text{target}} = z_1 - z_0$$
$$\mathcal{L} = \mathbb{E}_{t, z_0, z_1} \left[ \| v_\theta(z_t, t, c) - v_{\text{target}} \|^2 \right]$$

### Classifier-Free Guidance

During training, condition dropout $p = 0.1$. At inference:

$$v_{\text{guided}} = v_\theta(z_t, t, c_\varnothing) + s \cdot \left[ v_\theta(z_t, t, c) - v_\theta(z_t, t, c_\varnothing) \right]$$

Default guidance scale $s = 3.0$.

### Why scGPT (Not PCA, Not Geneformer)

scGPT is a 51.9M-parameter transformer pretrained on 33M+ human cells from the CellxGene census. Its 512-d embeddings encode cell state in a **universal latent space** that generalizes across:
- Different sequencing platforms (10x, Smart-seq2, Drop-seq)
- Different tissues and organs
- Different diseases and conditions

**Why not PCA?** PCA embeddings are dataset-specific: a PCA basis learned on lung cancer data is meaningless for a brain metastasis dataset. This defeats the entire purpose of text-conditioned cross-dataset generation.

**Why not Geneformer V2?** Geneformer (316M params, 104M cells, including cancer-tuned variant) was evaluated but rejected because it **lacks a decoder**. CLOP-DiT needs to decode generated 512-d embeddings back to gene expression for biological validation. scGPT's `generate()` method provides this native decoding pathway: inject cell_emb at [CLS] → full transformer → ExprDecoder → per-gene expression.

**Why pan-cancer over whole-human?** The pan-cancer checkpoint was pretrained on 5.7M cancer cells specifically. Since 28/50 datasets are cancer types, this provides better cancer cell embeddings. The architecture (embsize=512, 12 layers, 8 heads) is identical to whole-human, so it generalizes to development datasets too.

### Why BiomedBERT-large (Not Base, Not Generic BERT)

BiomedBERT is pretrained on PubMed abstracts and understands biomedical terminology. The **large** model (1024-d, 340M params) has 3× more parameters than base (768-d, 110M params), providing richer representations for discriminating between:
- "acute myeloid leukemia" vs "acute lymphoblastic leukemia"
- "tumor-infiltrating T cells" vs "circulating NK cells"
- "basal cell carcinoma" vs "squamous cell carcinoma"

A generic BERT would treat these as similar cancer-related text, collapsing the condition space.

### scGPT Decoding: generate() Method (v0.3)

The key insight is that scGPT can **decode** cell embeddings back to gene expression, not just encode. The `generate()` method:

1. **Gene token encoding**: Each gene in the vocabulary gets an embedding via `GeneEncoder` → (seq_len, 512)
2. **Cell embedding injection**: The 512-d cell embedding is placed at position 0 (the [CLS] token position)
3. **Full transformer pass**: The 12-layer transformer processes the combined sequence, allowing the cell embedding to attend to all gene tokens
4. **Per-gene decoding**: `ExprDecoder` (3-layer MLP: 512→512→512→1) predicts a scalar expression value for each gene

$$\text{cell\_emb} \xrightarrow{\text{inject at [CLS]}} \text{gene\_tokens} \xrightarrow{\text{Transformer}_{12L}} \text{output} \xrightarrow{\text{ExprDecoder}} \hat{x} \in \mathbb{R}^{G}$$

This is fundamentally different from the v0.2 approach which just applied ExprDecoder directly to the raw 512-d embedding — that treated the cell embedding as a single gene token, producing a single scalar instead of per-gene expression values.

---

## External Resources

1. **BiomedBERT-large** (`microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract`) — auto-downloaded by HuggingFace `transformers`
2. **scGPT pan-cancer weights** — downloaded to `models/scgpt_pancancer/` from [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) (pretrained on 5.7M cancer cells)
3. **scGPT whole-human weights** — `models/scgpt_human/` (pretrained on 33M+ cells, used in v0.1-v0.2)

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
