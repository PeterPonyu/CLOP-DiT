# CLOP-DiT: Contrastive Language-Omics Pre-training + Diffusion Transformer

> **Text-conditioned generation of single-cell gene expression profiles via Flow Matching**

---

## Architecture Overview

```
User Text ─→ BiomedBERT (768-d) ─→ CLOP Text Projector ─→ Condition c (256-d)
                                                                ↓
                              z₀ ~ N(0,I) ─→ DiT(z_t, t, c) ─→ ODE Integrate ─→ z₁ (Cell Embedding, 512-d)
                                                                                        ↓
                                                                          scGPT Decoder / Linear Decoder
                                                                                        ↓
                                                                         Gene Expression Matrix (Genes × Cells)
```

### Pipeline Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  43 h5ad files   │ ──→ │ 00_prepare_all_  │ ──→ │ processed_h5ad_v2/ │
│ (Cancer + Dev)   │     │    data.py       │     │ 43 × ~2K cells     │
│ CancerDatasets/  │     │ Curated bio text │     │ + metadata JSON    │
│ CancerDatasets2/ │     │ QC + HVG + norm  │     │ 43 unique texts    │
│ DevelopmentData/ │     └──────────────────┘     └────────┬───────────┘
└─────────────────┘                                        │
                                                           ▼
┌──────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ scGPT (51.9M)    │ ──→ │ 03_cache_latents │ ──→ │ cached_latents_v2/ │
│ pretrained cell  │     │     .py          │     │ cell_emb (81K×512) │
│ encoder (512-d)  │     │ scGPT + BiomedBERT│    │ text_emb (81K×768) │
│                  │     └──────────────────┘     │ sample_ids, meta   │
│ BiomedBERT       │                              └────────┬───────────┘
│ text encoder     │                                       │
│ (768-d)          │                                       ▼
└──────────────────┘     ┌──────────────────┐     ┌────────────────────┐
                         │ 04a_train_clop   │ ──→ │ CLOP Aligner (2M)  │
                         │     .py          │     │ proj_text (81K×256)│
                         │ InfoNCE + temp   │     │ clop_best.pth      │
                         └──────────────────┘     └────────┬───────────┘
                                                           │
                         ┌──────────────────┐     ┌────────▼───────────┐
                         │ 04b_train_dit    │ ──→ │ DiT (22.1M)        │
                         │     .py          │     │ dit_best.pth (EMA) │
                         │ Flow Matching    │     │ val_cos_sim=0.82   │
                         └──────────────────┘     └────────┬───────────┘
                                                           │
                         ┌──────────────────┐     ┌────────▼───────────┐
                         │ 05_inference.py  │ ──→ │ Generated cells    │
                         │ Text → CLOP → DiT│     │ (N × 512) .h5ad   │
                         └──────────────────┘     └────────┬───────────┘
                                                           │
                         ┌──────────────────┐     ┌────────▼───────────┐
                         │ 06_evaluate.py   │ ──→ │ FD, MMD, R@K       │
                         │ Metrics + Viz    │     │ UMAP figures       │
                         └──────────────────┘     └────────────────────┘
```

---

## Changelog: v0.1.0 → v0.2.0

| Aspect | v0.1.0 | v0.2.0 |
|--------|--------|--------|
| **Datasets** | 3 cancer datasets, 5,833 cells | **43 datasets** (28 cancer + 15 development), **81,225 cells** |
| **Text descriptions** | 3 auto-generated from filenames (e.g., "Single-cell RNA sequencing with cancer from human (GSE123813).") | **43 curated biological descriptions** with tissue, disease, cell type, and study context |
| **Cell encoder** | PCA fallback as default | **scGPT mandatory** (51.9M pretrained, 512-d universal embeddings) |
| **Text encoder** | BiomedBERT loaded but trivial texts | **BiomedBERT with meaningful biomedical text** (768-d) |
| **CLOP training** | 3 unique negatives → trivial contrastive task | **43 unique text classes** → meaningful cross-modal alignment |
| **CLOP retrieval** | Not measured | **R@1: 0.95 (t2c), 1.00 (c2t)** |
| **DiT evaluation** | Basic MSE + cosine sim only | **Full GenerationMetrics**: FD, MMD, Coverage, Density, KL, R@K |
| **Evaluation pipeline** | metrics.py + visualizer.py existed but **never called** | **06_evaluate.py** — full evaluation with 5 visualization types |
| **Figures** | None generated | **5 publication-quality figures** (UMAP alignment, generation comparison, training curves, dimension distributions) |
| **Cache builder** | `build_cache()` defaulted to PCA | Defaults to `scgpt`, PCA deprecated with warning |

### Key Design Decisions

1. **scGPT as mandatory cell encoder**: PCA produces dataset-specific embeddings that don't generalize across datasets. scGPT provides a universal pretrained latent space learned from 33M+ cells, enabling cross-dataset generation — which is the entire point of CLOP-DiT.

2. **Curated biological descriptions**: Auto-generated text like "Single-cell RNA sequencing with cancer from human" is too generic for BiomedBERT to produce discriminative embeddings. Each dataset now has 2-sentence descriptions mentioning specific cancer types, cell populations, tissue compartments, and biological context.

3. **43 datasets (vs 3)**: With only 3 unique text descriptions, CLOP contrastive learning has only 3 negative classes per batch — essentially memorizing 3 mappings. With 43 diverse datasets spanning cancer, development, aging, and immune biology, the model must learn genuine text↔cell alignment.

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

### Step 1: Prepare Data (43 datasets)

```bash
python scripts/00_prepare_all_data.py \
    --output_dir data/processed_h5ad_v2 \
    --max_cells 2000
```

### Step 2: Cache Latent Embeddings (scGPT + BiomedBERT)

```bash
python scripts/03_cache_latents.py \
    --h5ad_dir data/processed_h5ad_v2 \
    --metadata data/processed_h5ad_v2/metadata_structured.json \
    --output_dir data/cached_latents_v2 \
    --cell_encoder scgpt \
    --scgpt_dir models/scgpt_human
```

### Step 3: Train CLOP Alignment

```bash
python scripts/04a_train_clop.py \
    --cache_dir data/cached_latents_v2 \
    --save_dir models/checkpoints_v2 \
    --epochs 50 --batch_size 512
```

### Step 4: Train DiT

```bash
python scripts/04b_train_dit.py \
    --cache_dir data/cached_latents_v2 \
    --projected_text data/cached_latents_v2/projected_text.npy \
    --save_dir models/checkpoints_v2 \
    --epochs 100 --batch_size 512
```

### Step 5: Generate Cells

```bash
python scripts/05_inference.py \
    --prompt "Lung adenocarcinoma with tumor-infiltrating T cells and macrophages" \
    --num_cells 500 \
    --dit_checkpoint models/checkpoints_v2/dit_best.pth \
    --clop_checkpoint models/checkpoints_v2/clop_best.pth \
    --output generated_cells.h5ad
```

### Step 6: Evaluate

```bash
python scripts/06_evaluate.py \
    --cache_dir data/cached_latents_v2 \
    --clop_checkpoint models/checkpoints_v2/clop_best.pth \
    --dit_checkpoint models/checkpoints_v2/dit_best.pth \
    --output_dir figures_v2
```

---

## Project Structure

```
CLOP-DiT/
├── configs/
│   ├── clop.yaml              # CLOP alignment config
│   └── dit.yaml               # DiT training config
├── data/
│   ├── processed_h5ad_v2/     # 43 preprocessed h5ad + metadata JSON
│   └── cached_latents_v2/     # Pre-computed .npy embeddings
│       ├── cell_embeddings.npy   # (81225, 512) scGPT
│       ├── text_embeddings.npy   # (81225, 768) BiomedBERT
│       ├── projected_text.npy    # (81225, 256) CLOP-projected
│       ├── sample_ids.npy        # (81225,) dataset IDs
│       ├── metadata.json         # ID → text description mapping
│       └── manifest.json         # Cache statistics
├── figures_v2/                # Evaluation outputs
│   ├── clop_alignment_umap.png
│   ├── generation_comparison_umap.png
│   ├── dim_distributions.png
│   ├── clop_training_curves.png
│   ├── dit_training_curves.png
│   └── evaluation_metrics.json
├── models/
│   ├── checkpoints_v2/        # v0.2 trained weights
│   │   ├── clop_best.pth
│   │   ├── dit_best.pth (EMA)
│   │   ├── clop_history.json
│   │   └── dit_history.json
│   └── scgpt_human/           # scGPT pretrained weights
│       ├── best_model.pt (196MB)
│       ├── vocab.json (60697 genes)
│       └── args.json
├── scripts/
│   ├── 00_prepare_all_data.py  # NEW: 43-dataset prep with curated text
│   ├── 00_prepare_local_data.py # v0.1 (deprecated)
│   ├── 03_cache_latents.py     # scGPT default, PCA deprecated
│   ├── 04a_train_clop.py
│   ├── 04b_train_dit.py
│   ├── 05_inference.py
│   └── 06_evaluate.py          # NEW: full metrics + visualization
├── src/
│   ├── architecture/
│   │   ├── dit.py             # 1D DiT with AdaLN-Zero (22.1M)
│   │   ├── clop.py            # CLOP aligner (2M)
│   │   ├── scgpt_embed.py     # Standalone scGPT encoder (51.9M)
│   │   └── decoder.py         # scGPT/Linear decoder
│   ├── data_pipeline/
│   │   ├── cache_builder.py   # scGPT default, graceful fallback
│   │   ├── dataset.py         # CLOPDataset, DiTDataset
│   │   ├── geo_fetcher.py     # GEO data acquisition
│   │   └── text_cleaner.py    # SFT text cleaning
│   ├── training/
│   │   ├── train_clop.py      # CLOPTrainer
│   │   ├── train_dit.py       # DiTTrainer (now with GenerationMetrics)
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

### Why scGPT (Not PCA)

scGPT is a 51.9M-parameter transformer pretrained on 33M+ human cells from the CellxGene census. Its 512-d embeddings encode cell state in a **universal latent space** that generalizes across:
- Different sequencing platforms (10x, Smart-seq2, Drop-seq)
- Different tissues and organs
- Different diseases and conditions

PCA embeddings are dataset-specific: a PCA basis learned on lung cancer data is meaningless for a brain metastasis dataset. This defeats the entire purpose of text-conditioned cross-dataset generation.

### Why BiomedBERT (Not Generic BERT)

BiomedBERT is pretrained on PubMed abstracts and understands biomedical terminology. It can discriminate between:
- "acute myeloid leukemia" vs "acute lymphoblastic leukemia"
- "tumor-infiltrating T cells" vs "circulating NK cells"
- "basal cell carcinoma" vs "squamous cell carcinoma"

A generic BERT would treat these as similar cancer-related text, collapsing the condition space.

---

## External Resources

1. **BiomedBERT** (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`) — auto-downloaded by HuggingFace `transformers`
2. **scGPT weights** — downloaded to `models/scgpt_human/` via `gdown` from [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)

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
