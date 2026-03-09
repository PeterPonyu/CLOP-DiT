# CLOP-DiT Scripts

## Directory Organization

Scripts are organized by pipeline stage in subdirectories.

### `data_prep/` -- Data Preparation (Steps 00-03)

| Script | Purpose |
|--------|---------|
| `00_prepare_all_data.py` | Prepare all local h5ad datasets |
| `01_integrate_h5_datasets.py` | Integrate 10x Genomics H5 datasets |
| `01b_integrate_geodh_h5ad.py` | Integrate GEO-DataHub h5ad datasets |
| `02_subcluster_descriptions.py` | Generate sub-cluster text descriptions |
| `02b_enrich_descriptions.py` | Evidence-based caption enrichment |
| `03_cache_latents.py` | Cache scGPT + BiomedBERT-large embeddings |
| `03b_preprocess_embeddings.py` | ZCA whitening |
| `03c_build_dedup_cache.py` | Build deduplicated cache |

### `training/` -- Model Training (Steps 04a-c)

| Script | Purpose |
|--------|---------|
| `04a_train_clop.py` | Train CLOP alignment |
| `04b_train_dit.py` | Train DiT flow matching |
| `04c_train_cell2cell.py` | Train Cell2Cell model |
| `run_5fold_cv.py` | 5-fold cross-validation |
| `run_ablation_study.py` | Ablation study runner |
| `run_robustness_experiments.py` | Robustness experiments |

### `inference/` -- Generation & Evaluation (Steps 05-08)

| Script | Purpose |
|--------|---------|
| `05_inference.py` | Text-conditioned cell generation |
| `06_evaluate.py` | Evaluation (FD/MMD/Coverage) |
| `08_biological_validation.py` | Biological validation metrics |
| `generate_embeddings.py` | Generate latent embeddings from DiT |

### `analysis/` -- Post-hoc Analysis

| Script | Purpose |
|--------|---------|
| `decode_expression.py` | Decode latents to gene expression via scGPT |
| `diversity_diagnostics.py` | Diversity and CFG sweep analysis |
| `conditioning_analysis.py` | Conditioning comparison and noise trade-off |
| `07_marker_gene_analysis.py` | Marker gene visualization |
| `generate_architecture_figure.py` | Architecture diagram (Fig 1) |
| `compute_bootstrap_cis.py` | Bootstrap confidence intervals |
| `run_downstream_for_baselines.py` | Downstream biology for baselines |

### `pipeline/` -- Orchestration

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | Full pipeline orchestrator |
| `run_regeneration.py` | Figure regeneration + VCD |
| `regenerate_report.sh` | One-command figure regeneration |
| `build_article.sh` | LaTeX article build |

### `baselines/` -- Baseline Methods

| Script | Purpose |
|--------|---------|
| `train_embedding_vae_baseline.py` | Embedding-VAE baseline |
| `train_scvi_baseline.py` | scVI latent baseline |

### `vcd/` -- Visual Conflict Detector

Automated 33-pass visual quality assurance for matplotlib figures. See `scripts/vcd/__init__.py` for API.

## One-Command Figure Regeneration

```bash
bash scripts/pipeline/regenerate_report.sh
```
