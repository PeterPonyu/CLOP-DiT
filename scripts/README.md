# CLOP-DiT Scripts

## Directory Organization

### Phase 1: Data Preparation
| Script | Purpose |
|---|---|
| `00_prepare_all_data.py` | Prepare all local h5ad datasets (multi-source, max 3000 cells/dataset) |
| `01_integrate_h5_datasets.py` | Integrate 10x Genomics H5 datasets into processed h5ad |
| `01b_integrate_geodh_h5ad.py` | Integrate GEO-DataHub h5ad datasets with equivalent preprocessing |
| `02_subcluster_descriptions.py` | Generate sub-cluster text descriptions (→ 1,088 text groups) |

### Phase 2: Embedding & Training
| Script | Purpose |
|---|---|
| `03_cache_latents.py` | Cache scGPT + BiomedBERT-large latent embeddings |
| `04a_train_clop.py` | Train CLOP contrastive alignment (200 epochs) |
| `04b_train_dit.py` | Train DiT flow matching (200 epochs, `--resume`) |
| `04c_train_cell2cell.py` | Train Cell2Cell translation model |

### Phase 3: Inference & Legacy Evaluation
| Script | Purpose |
|---|---|
| `05_inference.py` | Text-conditioned cell generation with scGPT decoding |
| `06_evaluate.py` | Legacy evaluation (FD/MMD/Coverage) |
| `06_cell2cell_inference.py` | Cell2Cell inference |
| `07_marker_gene_analysis.py` | Marker gene visualization |
| `08_biological_validation.py` | Biological validation metrics |

### Phase 4: v5.2 Evaluation & Publication
| Script | Purpose |
|---|---|
| `14_biological_evaluation.py` | In-distribution biological evaluation |
| `14b_cfg_sweep.py` | CFG scale + ODE solver sweep |
| `15_final_verified_evaluation.py` | Final verified evaluation (KNN accuracy, steering, DivR) |
| `16_publication_figures.py` | Publication figures with WCAG visual conflict detection |
| `17_baseline_comparison.py` | Phase 2 decoder architecture comparison (6 decoders, 10 metrics) |

### Auxiliary
| Script | Purpose |
|---|---|
| `10_full_pipeline.py` | Full pipeline orchestrator (data → train → eval) |
| `11_jbhi_enhanced_eval.py` | JBHI article enhanced evaluation |
| `12_compose_and_check.py` | Compose article and consistency checks |
| `13_tier_comparison_fixed.py` | Tier comparison (fixed version) |
| `run_5fold_cv.py` | 5-fold cross-validation runner |

## v5.2 Key Results

| Metric | Value | Significance |
|--------|-------|-------------|
| KNN Accuracy | 36.9% (37× random) | Strong text-cell alignment |
| Steering Accuracy | 81% | Semantic control over generation |
| Diversity Ratio | 0.93 | Non-collapsed diverse outputs |
| Datasets | 80 | Multi-source integration |
| Cells | 220,304 | Large-scale training |
| Text Groups | 1,088 | Fine-grained descriptions |

