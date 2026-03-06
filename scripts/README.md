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
| `run_robustness_experiments.py` | Seed/subsample/prompt robustness experiment planner/executor |
| `run_downstream_for_baselines.py` | Re-run downstream biology for every baseline with expression artifacts |

### Phase 4: Evaluation & Publication Pipeline (Current)

The canonical evaluation and figure pipeline is:

```bash
bash scripts/regenerate_report.sh     # Steps 0–7: generate → evaluate → visualize
```

This orchestrates:

| Step | Script / Module | Purpose |
|------|----------------|---------|
| 0 | `generate_architecture_figure.py` | Architecture diagram (Fig 1) |
| 1 | `generate_embeddings.py` | Generate latent embeddings from DiT |
| 2 | `decode_expression.py` | Decode latents → gene expression via scGPT |
| 3 | `diversity_diagnostics.py` | Panels J + K (diversity, CFG sweep) |
| 4 | `conditioning_analysis.py` | Panels L + M (noise trade-off, conditioning UMAP) |
| 5 | `src.evaluation.downstream_biology` | Panels P, Q, R (clustering, classifier, DE) |
| 6 | `src.evaluation.model_benchmarking` | Panel S (composite benchmark) |
| 7 | `src.visualization.results_visualizer` | All panels A–S + 5 merged figures + PDF report |

### Baselines

Learned and external baseline adapters live in `scripts/baselines/`.

| Script | Purpose |
|--------|---------|
| `baselines/train_embedding_vae_baseline.py` | Train a lightweight learned baseline in the shared scGPT embedding space |
| `baselines/train_scvi_baseline.py` | Prepare an scVI latent baseline and write metadata/artifact contract files |

### Legacy / Archived Scripts

Scripts in `scripts/archive/` are from earlier development phases and are **not** part of the current pipeline:

| Script | Status |
|--------|--------|
| `15_final_verified_evaluation.py` | Superseded by `regenerate_report.sh` step 7 |
| `16_publication_figures.py` | Superseded by `results_visualizer` |
| `17_baseline_comparison.py` | Superseded by `model_benchmarking` |

**Figure cleanup:** `bash scripts/remove_outdated_figures.sh` removes legacy outputs (fig1_*–fig5_*, visual_conflict_report.json) from `results/figures/` if present. See docs/FIGURE_ORGANIZATION.md.

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

