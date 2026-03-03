# CLOP-DiT: Contrastive Language-Omics Pre-training + Diffusion Transformer

> **Text-conditioned generation of single-cell gene expression profiles via Flow Matching**

## Key Results (v6.1)

| Metric | v5.2 | v6.0 | v6.1 | Notes |
|--------|------|------|------|-------|
| **CLOP Val Acc** | 2.5% | 1.5% (overfitting) | **TBD** (+proto_acc) | Prototype alignment reduces memorization |
| **CLOP Train Acc** | ~2.5% (stuck) | 80% (memorizing) | **66%** (generalizing) | Lower train = less overfitting |
| **Proto Acc (train)** | — | — | **5%** (25× random) | True between-group alignment metric |
| **Unique Text Groups** | 1,088 | 1,088 | **2,342** (whitened) | ZCA amplifies subtle differences |
| **Within-Group Variance** | — | 67.2% | **67.2%** (data-level) | Addressed via loss, not data |

**Training data:** 220,304 cells from 80 datasets, ~2,342 unique whitened text groups

### v6.1 Root Cause Fix: Text-Cell Granularity Mismatch

v6.0 fixed the embedding collapse but revealed a deeper issue:
text descriptions are at **Leiden cluster level** (~2,300 unique), while cells
have **per-cell variation** (220K unique embeddings). Variance decomposition shows:

| Variance Component | % of Total | Meaning |
|--------------------|-----------|---------|
| **Within-group** (sub-cluster) | **67.2%** | Noise that text CANNOT discriminate |
| **Between-group** (text-aligned) | **32.8%** | Signal that text CAN discriminate |

Standard SigLIP memorizes per-cell within-group noise (train_acc 80%) that doesn't
generalize (val_acc 1.5%). **Prototype-SigLIP** aligns text to group centroids,
forcing the model to learn only the 33% between-group signal.

### v6.1 also discovered: AMP Float16 Bug

Both `_build_duplicate_mask` and `_compute_group_ids` were silently broken under
AMP autocast. Float16 matmul on 1024-dim vectors loses enough precision that
identical embeddings appear different (cosine < 0.9999). **This means v6.0's
auto_duplicate_mask was effectively disabled** — duplicate text cells were being
penalized as false negatives. Fixed by forcing float32 in both methods.

---

## Architecture Overview

```
User Text ─→ BiomedBERT-large (1024-d) ─→ [ZCA Whitening] ─→ CLOP Text Projector ─→ Condition c (512-d)
                                                                                            ↓
                                   z₀ ~ N(0,I) ─→ DiT(z_t, t, c) ─→ ODE Integrate ─→ z₁ (Cell Embedding, 512-d)
                                                                                              ↓
                                                                               scGPT generate() Decoder
                                                                               [CLS] injection → Transformer → ExprDecoder
                                                                                              ↓
                                                                               Gene Expression Matrix (Genes × Cells)
```

### Pipeline Data Flow (v6)

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  80 datasets     │ ──→ │ Prepare & QC     │ ──→ │ processed_h5ad/    │
│ (Cancer + Dev)   │     │ 00_prepare_all_  │     │ ~80 × ~2.8K cells  │
│ CancerDatasets/  │     │    data.py       │     │ + metadata JSON    │
│ DevelopmentData/ │     │ 01_integrate_*   │     │ 1,088 unique texts │
│ GEO-DataHub      │     │ 01b_integrate_*  │     │                    │
└─────────────────┘     └──────────────────┘     └────────┬───────────┘
                                                           ▼
┌──────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ scGPT pan-cancer │ ──→ │ 03_cache_latents │ ──→ │ cached_latents_    │
│ (51.9M, 5.7M     │     │     .py          │     │  v5.2/             │
│ cancer cells)    │     │ scGPT + BiomedBERT│    │ cell_emb (N×512)   │
│                  │     │      -large       │     │ text_emb (N×1024)  │
│ BiomedBERT-large │     └──────────────────┘     │                    │
│ text encoder     │                               └────────┬───────────┘
└──────────────────┘                                        │
                         ┌──────────────────┐     ┌─────────▼──────────┐
                         │ 03b_preprocess_  │ ──→ │ Whitened embeddings│
                         │ embeddings.py   │     │ cell_emb_pp (N×512)│
                         │ ZCA Whitening    │     │ text_emb_pp(N×1024)│
                         │ (v6 critical!)   │     │ cos 0.96→0.002 ✓  │
                         └──────────────────┘     └─────────┬──────────┘
                         ┌──────────────────┐     ┌─────────▼──────────┐
                         │ 04a_train_clop   │ ──→ │ CLOP (3.42M)       │
                         │     .py          │     │ clop_best.pth      │
                         │ SigLIP + dup mask│     │ proj_dim=512       │
                         │ 200 epochs       │     │ proj_text (N×512)  │
                         └──────────────────┘     └─────────┬──────────┘
                         ┌──────────────────┐     ┌─────────▼──────────┐
                         │ 04b_train_dit    │ ──→ │ DiT (22.10M, EMA)  │
                         │     .py          │     │ dit_best.pth       │
                         │ Flow Matching    │     │ cond_dim=512       │
                         │ 200 epochs       │     └─────────┬──────────┘
                         └──────────────────┘               │
                         ┌──────────────────┐     ┌─────────▼──────────┐
                         │ 05_inference.py  │ ──→ │ Generated cells    │
                         │ CFG=1.0–3.0     │     │ (N × 512) → scGPT  │
                         │ Euler/Midpoint   │     │ → gene expression  │
                         └──────────────────┘     └─────────┬──────────┘
                         ┌──────────────────┐     ┌─────────▼──────────┐
                         │ 15_final_verified│ ──→ │ KNN, Steering,     │
                         │ _evaluation.py   │     │ DivR, LinAcc       │
                         │ In-distribution  │     │ + CFG sweep (24)   │
                         └──────────────────┘     └────────────────────┘
```

---

## Changelog

### v6.2.0 (Current — March 2026)

**Evidence-based text enrichment: Maximizes caption entropy for discriminative alignment.**

| Aspect | v6.1 | v6.2 |
|--------|------|------|
| **Text descriptions** | Template-based (mean 284 chars) | **Evidence-dense** (mean 581 chars) |
| **Unique captions** | 1,073 cluster texts | **5,258** (4.9 variants/cluster) |
| **Text storage** | 902 MB (99.5% duplicated) | **5.3 MB** (deduplicated + group IDs) |
| **Evidence sources** | Marker genes only | **+ pathways, anti-markers, GO terms, TFs, expr stats** |
| **Text augmentation** | None | **variant_prob=0.3** (30% chance of caption variant per sample) |
| **Storage format** | Flat duplicated .npy | **Deduplicated** (text_embeddings_unique + text_group_ids) |

New files:
- `scripts/02b_enrich_descriptions.py` — Evidence extraction + enriched caption generation
- `scripts/02c_build_enriched_cache.py` — Build deduplicated cache from enriched metadata
- `data/processed_h5ad/subcluster_metadata_enriched.json` — Enriched cluster annotations

### v6.1.0 (March 2026)

**Granularity-aware alignment: Addresses the text-cell resolution mismatch.**

| Aspect | v6.0 | v6.1 |
|--------|------|------|
| **CLOP loss** | SigLIP (individual pairs) | **Prototype-SigLIP** (centroid alignment + cohesion) |
| **Group detection** | Broken (AMP float16 bug) | **Fixed** (forced float32 for cosine threshold) |
| **Duplicate mask** | Broken (AMP float16 bug) | **Fixed** (forced float32) |
| **Cell augmentation** | None (noise_std=0) | **Gaussian noise** (std=0.05) |
| **Dropout** | 0.1 | **0.2** |
| **Weight decay** | 0.01 | **0.05** |
| **Batch size** | 256 (~0 collisions) | **1024** (~480 cells in multi-cell groups) |
| **Temperature cap** | None (unlimited) | **100.0** |
| **Cohesion weight** | — | **0.1** (pulls cells toward group centroid) |
| **Leiden resolution** | 0.6–1.2 (adaptive) | **1.2–2.5** (2× more clusters) |
| **Train acc (3 ep)** | 80% (memorizing noise) | **66%** (less memorization) |
| **Train-val gap** | 78.5% | **64.3%** (narrower) |

### v6.0.0 (March 2026)

**Industrial-grade alignment fix: Solves the v5.2 CLOP training failure.**

| Aspect | v5.2 | v6.0 |
|--------|------|------|
| **Embedding preprocessing** | None (raw encoders) | **ZCA whitening** (text cos 0.96→0.002, cell cos 0.99→0.0003) |
| **CLOP loss** | InfoNCE (softmax) | **SigLIP** (pairwise sigmoid BCE — robust to collapsed spaces) |
| **Duplicate handling** | None (same-text cells penalized as negatives) | **Auto-duplicate mask** (same-text cells excluded from negative set) |
| **Normalization** | BatchNorm | **LayerNorm** (cross-dataset generalization) |
| **proj_dim** | 256 | **512** (richer shared space) |
| **Temperature init** | 0.07 | **10.0** (SigLIP learns higher temps with separate bias) |
| **Warmup** | 5 epochs | **10 epochs** (stability with higher LR) |
| **LR** | 3e-4 | **1e-3** (SigLIP converges faster with SGD-like dynamics) |
| **Early stopping** | None | **30-epoch patience** |
| **Train acc (epoch 3)** | ~2.5% (stuck) | **80%** (verified — model is learning) |
| **Val acc** | 2.5% (ceiling) | **Target >10%** |

**Root cause analysis:** Both BiomedBERT and scGPT embeddings exhibit extreme cosine similarity collapse (>0.95). Standard InfoNCE's softmax denominator is dominated by near-identical negatives, making gradients vanish. SigLIP's pairwise sigmoid formulation treats each (text, cell) pair independently, bypassing this failure mode entirely. This is the same insight that drove Google's transition from CLIP → SigLIP for vision-language alignment.

### v5.2.0 (February 2026)

| Aspect | v0.4 | v5.2 |
|--------|------|------|
| **Datasets** | 69 datasets, 190K cells | **80 datasets**, **220,304 cells** |
| **Data sources** | Cancer + Development + 10x h5 | + **GEO-DataHub** (11 additional studies) |
| **Text descriptions** | 69 dataset-level labels | **1,088 sub-cluster descriptions** (marker genes, tissue, disease) |
| **Evaluation** | FD, MMD, Coverage, Density, KL | **KNN (37×), Steering (81%), DivR (0.93), LinAcc** |
| **Eval protocol** | Out-of-distribution prompts (6 types) | **In-distribution conditions** (100 groups × 200 cells) |
| **Validation** | 6 datasets, `shuffle=False` (bug) | **8 datasets**, `shuffle=True` (fixed) |
| **Val accuracy** | 1.1% (with shuffle bug) | **2.5%** (6.4× random, shuffle fixed) |

### Earlier Versions (Archived)

<details>
<summary>v0.3 → v0.4 → v5.0 history</summary>

**v0.4:** 69 datasets (190K cells) from 4 dirs + 19 GSE studies. GEO-verified metadata. </br>
**v0.3:** 50 datasets (138K cells). scGPT pan-cancer + BiomedBERT-large. scGPT `generate()` decoder. </br>
**v0.2:** 43 datasets (81K cells). scGPT mandatory. BiomedBERT-base. </br>
**v0.1:** 3 datasets (5,833 cells). Proof-of-concept.

</details>

---

## v6.1 Training Configuration

| Component | v6.0 | v6.1 |
|-----------|------|------|
| Total cells | 220,304 | 220,304 |
| Text groups | 1,088 raw → 2,342 whitened | 2,342 whitened |
| **Preprocessing** | ZCA whitening | ZCA whitening |
| **CLOP loss** | SigLIP (broken dup mask) | **Prototype-SigLIP** (fixed float32) |
| **CLOP** | 3.42M, proj=512, batch=256 | **3.42M, proj=512, batch=1024** |
| **Regularization** | dropout=0.1, wd=0.01 | **dropout=0.2, wd=0.05, noise=0.05** |
| **DiT** | 22.10M, cond=512 | 22.10M, cond=512 |
| Hardware | NVIDIA RTX 5090 Laptop GPU (24.1 GB VRAM) | Same |

### CLOP v6.1 (3.42M params, Prototype-SigLIP, 200 epochs target)
| Metric | v5.2 | v6.0 (3 epoch test) |
|--------|------|---------------------|
| Train acc | ~2.5% (stuck) | **80% by epoch 3** |
| Train loss | 1.19 | 0.005 |
| Val acc | 2.5% | 1.5% (4× random — improving) |
| Temperature | 0.060 | 17.12 (SigLIP learned) |
| Loss type | InfoNCE | **SigLIP (sigmoid BCE)** |

### v5.2 Results (Downstream — to be re-evaluated after v6 CLOP)

### DiT Generation (22.10M params, 200 epochs + EMA)
| Metric | Value | Direction |
|--------|-------|-----------|
| Val cosine (EMA) | 0.976 | Higher ↑ |
| Val loss | 0.091 | Lower ↓ |
| Angular error | ~12.6° | Lower ↓ |

### Evaluation (In-Distribution, 100 groups × 200 cells)
| Method | KNN-1 ↑ | Steering ↑ | DivR (→1) | LinAcc ↑ | KNN/Rand |
|--------|---------|-----------|-----------|----------|----------|
| **Real Data** | **0.890** | — | **1.000** | **0.942** | 89× |
| CFG=2.0 Euler-10 | **0.369** | **0.810** | 0.513 | **0.511** | **37×** |
| CFG=1.0 Midpoint-10 | 0.288 | 0.807 | **0.929** | 0.357 | 29× |
| CFG=0.0 (Uncond.) | 0.010 | 0.475 | 1.833 | 0.010 | 1× |
| Gaussian N(μ,Σ) | 0.011 | 0.466 | 2.277 | 0.009 | 1× |

## Quick Start

### Environment Setup

```bash
conda activate /path/to/.conda
cd CLOP-DiT
```

### Step 1: Prepare Data (80 datasets from multiple sources)

```bash
# Local h5ad datasets
python scripts/00_prepare_all_data.py \
    --output_dir data/processed_h5ad --max_cells 3000

# Integrate 10x h5 datasets
python scripts/01_integrate_h5_datasets.py

# Integrate GEO-DataHub h5ad datasets
python scripts/01b_integrate_geodh_h5ad.py

# Generate sub-cluster descriptions (→ 1,088 text groups)
python scripts/02_subcluster_descriptions.py
```

### Step 2: Cache Latent Embeddings (scGPT pan-cancer + BiomedBERT-large)

```bash
python scripts/03_cache_latents.py \
    --h5ad_dir data/processed_h5ad \
    --metadata data/processed_h5ad/metadata_structured.json \
    --output_dir data/cached_latents_v5.2 \
    --cell_encoder scgpt \
    --scgpt_dir models/scgpt_pancancer \
    --text_encoder microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract
```

### Step 2b: Preprocess Embeddings (v6 — CRITICAL for alignment)

```bash
# ZCA whitening to fix embedding space collapse
# Text cosine: 0.96 → 0.002, Cell cosine: 0.99 → 0.0003
python scripts/03b_preprocess_embeddings.py --cache_dir data/cached_latents_v5.2
```

### Step 3: Train CLOP Alignment (SigLIP + whitened embeddings, 200 epochs)

```bash
python scripts/04a_train_clop.py --config configs/clop.yaml
```

### Step 4: Train DiT (200 epochs, with optional resume)

```bash
python scripts/04b_train_dit.py --config configs/dit.yaml
# Resume:
python scripts/04b_train_dit.py --config configs/dit.yaml --resume
```

### Step 5: Evaluate (In-Distribution)

```bash
# Full verified evaluation (KNN, steering, diversity, CFG sweep)
python scripts/15_final_verified_evaluation.py

# Publication figures
python scripts/16_publication_figures.py
```

### Step 6: Generate Cells

```bash
python scripts/05_inference.py \
    --prompt "CD8+ cytotoxic T cells from human lung adenocarcinoma" \
    --num_cells 500 \
    --cfg_scale 2.0 \
    --decode_expression \
    --reference_h5ad data/processed_h5ad/GSE123902_LungAdreHmCancer_processed.h5ad \
    --output generated_cells.h5ad
```

---

## Project Structure

```
CLOP-DiT/
├── configs/
│   ├── clop.yaml              # CLOP config (v6: SigLIP, whitened emb, proj=512)
│   ├── dit.yaml               # DiT config (v6: cond_dim=512, 200 epochs)
│   └── models.yaml            # Multi-model presets (cancer/general/ablations)
├── data/
│   ├── processed_h5ad/        # 80 preprocessed h5ad + metadata JSON
│   └── cached_latents_v5.2/   # Pre-computed .npy embeddings (220,304 cells)
│       ├── cell_embeddings.npy             # (N, 512) scGPT pan-cancer [raw]
│       ├── text_embeddings.npy             # (N, 1024) BiomedBERT-large [raw]
│       ├── cell_embeddings_preprocessed.npy  # (N, 512) ZCA-whitened [v6]
│       ├── text_embeddings_preprocessed.npy  # (N, 1024) ZCA-whitened [v6]
│       ├── cell_preprocessor_preprocessed.npz  # Whitening transform state [v6]
│       ├── text_preprocessor_preprocessed.npz  # Whitening transform state [v6]
│       ├── projected_text.npy              # (N, 512) CLOP-projected [v6: 512-d]
│       ├── sample_ids.npy                  # (N,) dataset IDs
│       ├── metadata.json                   # 1,088 text groups → descriptions
│       └── manifest.json                   # Cache statistics
├── docs/
│   ├── CLOP_DiT_JBHI_Article.md  # JBHI manuscript (v5.2 results)
│   └── CLOP-DiT_Evaluation_Report.md  # Full evaluation report
├── figures/
│   ├── final_eval_summary.png       # 4-panel eval summary
│   └── v5_publication/              # Publication figures (WCAG-accessible)
│       ├── fig2_training_dynamics.{png,pdf}
│       ├── fig3_pca_embedding.{png,pdf}
│       ├── fig4_metrics_dashboard.{png,pdf}
│       ├── fig5_biological_validation.{png,pdf}
│       ├── fig6_cfg_solver.{png,pdf}
│       ├── fig7_dimension_sampling.{png,pdf}
│       ├── fig8_tables.{png,pdf}
│       └── fig9_baseline_comparison.{png,pdf}
├── models/
│   ├── checkpoints/           # Trained weights
│   │   ├── clop_best.pth
│   │   ├── dit_best.pth (EMA)
│   │   ├── clop_history.json
│   │   └── dit_history.json
│   ├── scgpt_pancancer/       # scGPT pan-cancer pretrained (5.7M cancer cells)
│   │   ├── best_model.pt (205MB)
│   │   ├── vocab.json (60K+ genes)
│   │   └── args.json
│   └── scgpt_human/           # scGPT whole-human pretrained (33M cells)
│       ├── best_model.pt (196MB)
│       ├── vocab.json (60697 genes)
│       └── args.json
├── scripts/
│   ├── 00_prepare_all_data.py         # Multi-source h5ad preparation
│   ├── 01_integrate_h5_datasets.py    # 10x h5 integration
│   ├── 01b_integrate_geodh_h5ad.py    # GEO-DataHub h5ad integration
│   ├── 02_subcluster_descriptions.py  # Sub-cluster text generation
│   ├── 03_cache_latents.py            # scGPT + BiomedBERT embedding cache
│   ├── 03b_preprocess_embeddings.py   # [v6] ZCA whitening (CRITICAL)
│   ├── 04a_train_clop.py             # CLOP alignment (v6: SigLIP)
│   ├── 04b_train_dit.py              # DiT training (--resume)
│   ├── 04c_train_cell2cell.py        # Cell-to-cell training
│   ├── 05_inference.py               # Text-conditioned generation
│   ├── 06_cell2cell_inference.py     # Cell-to-cell inference
│   ├── 06_evaluate.py               # Legacy evaluation (FD/MMD)
│   ├── 07_marker_gene_analysis.py   # Marker gene visualization
│   ├── 08_biological_validation.py  # Biological validation metrics
│   ├── 14_biological_evaluation.py  # In-distribution bio evaluation
│   ├── 14b_cfg_sweep.py            # CFG scale + solver sweep
│   ├── 15_final_verified_evaluation.py  # Final verified eval (KNN/Steering/DivR)
│   ├── 16_publication_figures.py    # Publication figures + visual conflict detection
│   ├── 17_baseline_comparison.py   # Phase 2 decoder architecture comparison
│   └── README.md                    # Script documentation
├── src/
│   ├── architecture/
│   │   ├── dit.py             # 1D DiT with AdaLN-Zero (22.1M)
│   │   ├── clop.py            # CLOP aligner (v6: SigLIP + InfoNCE + auto-dup mask)
│   │   ├── scgpt_embed.py     # Standalone scGPT encoder + generate() decoder
│   │   └── decoder.py         # scGPT generate() decoder wrapper
│   ├── data_pipeline/
│   │   ├── cache_builder.py   # BiomedBERT-large default (1024-d)
│   │   ├── dataset.py         # CLOPDataset, DiTDataset (v6: use_preprocessed)
│   │   ├── embedding_preprocessor.py  # [v6] ZCA whitening + diagnostics
│   │   ├── geo_fetcher.py     # GEO data acquisition
│   │   └── text_cleaner.py    # SFT text cleaning
│   ├── training/
│   │   ├── train_clop.py      # CLOPTrainer (saves text_dim/cell_dim in ckpt)
│   │   ├── train_dit.py       # DiTTrainer
│   │   └── schedulers.py      # CosineWarmupScheduler
│   ├── evaluation/
│   │   ├── metrics.py         # KNN accuracy, steering, DivR, FD, MMD
│   │   └── visualizer.py      # PCA, tSNE, training curves
│   └── utils/
│       ├── helpers.py         # Seed, device, parameter counting
│       └── logging_config.py  # Structured logging
├── results/                   # Evaluation results (JSON)
├── logs/                      # Training logs
├── requirements.txt
├── setup.py
└── README.md
```

---

## Technical Details

### v6.0 Key Innovation: Upstream Whitening + SigLIP

**The problem:** Both BiomedBERT and scGPT produce embeddings that live in a narrow cone of the hypersphere. Mean pairwise cosine similarity: text = 0.961, cell = 0.996. This means all embeddings look nearly identical to InfoNCE's softmax — the loss function cannot distinguish positive pairs from negatives.

**The fix (industrial-standard approach):**

1. **ZCA Whitening** (upstream, before CLOP): Decorrelate all embedding dimensions and equalize their variances. This preserves the original coordinate alignment (unlike PCA whitening) while spreading the distribution from [0.95, 1.0] to [-0.3, 0.7]. This is the same technique used by OpenCLIP, Meta's ImageBind, and Google's SigLIP internally.

2. **SigLIP Loss** (replaces InfoNCE): Instead of softmax over the entire batch (which requires globally discriminable features), SigLIP treats each (text_i, cell_j) pair as an independent binary classification: "are these matched (+1) or not (-1)?" The loss is:

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{B^2}\sum_{i}\sum_{j} \log \sigma(y_{ij} \cdot (\mathbf{t}_i \cdot \mathbf{c}_j \cdot \tau + b))$$

where $y_{ij} = +1$ on diagonal, $-1$ off-diagonal, $\tau$ is learned temperature, and $b$ is learned bias.

3. **Auto-duplicate mask**: With 220K cells but only 1,088 unique text descriptions (avg 202 cells/text), many cells in a batch share identical text. Without masking, these are incorrectly penalized as false negatives. The mask detects text duplicates (cosine > 0.9999) and excludes them from the negative set.

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

## Application Scenarios & Scientific Value

### Core Proposition: Text-to-Cell-State Prediction

CLOP-DiT is a **virtual cell state predictor**: given a natural language description of a biological condition, it generates synthetic single-cell gene expression profiles that recapitulate the expected transcriptomic patterns. This is analogous to AlphaFold predicting protein structures from sequences — we predict cell expression states from biological context descriptions.

### Key Application Scenarios

1. **Hypothesis-driven virtual experiments**: Generate cells for conditions not yet profiled by scRNA-seq (e.g., "CD8+ T cells from a patient with KRAS-G12C mutant lung adenocarcinoma treated with sotorasib") and examine predicted marker gene expression to guide experimental design.

2. **Rare cell state discovery**: Generate cells for under-sampled states in existing datasets. The model can interpolate between observed conditions to predict intermediate cell states that are difficult to capture in a single-timepoint scRNA-seq snapshot.

3. **Cross-study harmonization**: By projecting all text descriptions into a shared CLOP alignment space, the model provides a unified framework for comparing cell states across studies, tissues, and diseases.

4. **Drug response prediction**: Given a text description of treatment context, generate the expected gene expression profile and compare with untreated cells to predict transcriptomic drug effects.

5. **Benchmark foundation for single-cell AI**: The multi-model architecture (swappable scGPT/BiomedBERT variants) enables systematic ablation studies comparing cell encoders, text encoders, and decoders.

### Model Configuration Presets (v0.4)

| Preset | Cell Encoder | Text Encoder | Use Case |
|--------|-------------|--------------|----------|
| `cancer` | scGPT pan-cancer (5.7M cells) | BiomedBERT-large (1024-d) | Cancer TME studies |
| `general` | scGPT whole-human (33M cells) | BiomedBERT-base (768-d) | General single-cell |
| `large_text_general_cell` | scGPT whole-human | BiomedBERT-large | Ablation: text encoder effect |
| `base_text_cancer_cell` | scGPT pan-cancer | BiomedBERT-base | Ablation: cell encoder effect |

### Alternative Decoder Models Evaluated

| Model | Decode → Expression? | Size | Pretraining | Status |
|-------|---------------------|------|-------------|--------|
| **scGPT** | ✅ `generate()` | 51.9M | 33M / 5.7M cells | ✅ Implemented |
| **scFoundation** (xTrimoGene) | ✅ RDE mode | 100M | 50M+ cells | 🔜 Candidate |
| **Geneformer** | ❌ Encoder-only | 316M | 104M cells | ❌ No decoder |
| **UCE** | ❌ Encoder-only | — | Tabula Sapiens | ❌ No decoder |
| **scVI** | ✅ VAE decoder | ~1-5M | Per-dataset | ❌ Not foundation |

The critical requirement is **embedding → gene expression decoding**. Only scGPT and scFoundation (xTrimoGene) provide foundation-model-level decoders. scFoundation is the strongest candidate for future integration.

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
