# Biological Claim Map

*Last revised: 2026-03-06*

This document answers the reviewer question "where is the biological meaning shown?" by mapping each major claim to the exact figures, metrics, and code modules that produce the evidence.

## Claim-to-Evidence Map

| Claim | Reviewer question answered | Primary article figures | Quantitative signals | Canonical code path | What failure would look like |
|------|------|------|------|------|------|
| Text conditioning is learned rather than ignored | Does the model actually use the text prompt? | Fig 2, Fig 3, Fig 4 | CLOP validation accuracy, pairwise cosine separation, KNN, steering, unconditional collapse | `scripts/04a_train_clop.py`, `scripts/04b_train_dit.py`, `src/visualization/panels_training.py`, `src/visualization/panels_merged.py`, `src/evaluation/run_metrics.py` | Unconditional and conditioned generation would have similar KNN/steering, or text and cell embeddings would not co-localize |
| CLOP builds a biologically useful condition space | Are the text and cell modalities aligned in a meaningful way? | Fig 3, Fig 5 | Joint UMAP overlap, centroid cosine, diagonal dominance in text-cell similarity heatmap | `src/visualization/panels_embedding.py`, `src/visualization/panels_heatmaps.py`, `src/visualization/panels_merged.py` | Text clusters would drift away from their target cell clusters and off-diagonal similarities would dominate |
| Generated cells preserve cell-type identity | Are the outputs type-specific or generic averages? | Fig 4, Fig 5, Fig 6 | KNN top-1/top-5, centroid cosine, per-type FD, marker specificity, marker expression fidelity | `src/evaluation/run_metrics.py`, `src/evaluation/biological_validation/figures.py`, `src/visualization/panels_heatmaps.py`, `src/visualization/panels_expression.py` | Marker genes would flatten across types, per-type cosine would collapse, and KNN would approach chance |
| Generated expression is biologically plausible at gene level | Are latent samples decoded into realistic expression programs? | Fig 6, Fig 7, Fig 8 | Gene mean Pearson, gene mean R2, variance correlation, marker-specificity index, norm matching | `scripts/08_biological_validation.py`, `src/evaluation/biological_validation/run.py`, `src/evaluation/biological_validation/figures.py`, `src/visualization/panels_expression.py` | Real vs generated gene means would decorrelate, variance would be distorted, or marker enrichment would disappear |
| Diversity is preserved and controllable | Is the model robust, or does guidance collapse diversity? | Fig 10, Fig 11 | Diversity ratio, within-type cosine distributions, CFG sweep, condition sensitivity | `scripts/diversity_diagnostics.py`, `scripts/conditioning_analysis.py`, `src/visualization/panels_diversity.py`, `src/visualization/panels_conditioning.py` | Diversity would collapse uniformly across CFG, or condition noise would have no measurable effect |
| Biological meaning is not limited to a few easy types | Are failures concentrated in specific cell types or widespread? | Fig 5, Fig 10, Fig 14, Fig 15 | Per-type centroid cosine, per-type FD, per-type mixing, per-type classifier accuracy, contrast-specific DE concordance | `src/visualization/panels_heatmaps.py`, `src/visualization/downstream_panels.py`, `src/visualization/panels_de_concordance.py`, `src/evaluation/downstream_biology.py` | A few aggregate means would hide large per-type failures or a single easy contrast would dominate the conclusion |
| Generated cells remain useful in downstream analysis workflows | Can synthetic cells integrate into standard single-cell pipelines? | Fig 14, Fig 15 | ARI, NMI, cluster purity, kNN mixing, classifier transfer accuracy/F1, discriminator AUC, DE concordance metrics | `src/evaluation/downstream_biology.py`, `src/visualization/downstream_panels.py`, `src/visualization/panels_classifier.py`, `src/visualization/panels_clustering.py`, `src/visualization/panels_de_concordance.py` | Generated cells would segregate from real cells, classifier transfer would fail, or DE signs/top genes would disagree strongly |
| CLOP-DiT outperforms non-text and learned comparison baselines | Is benchmarking sufficient, or is the method only better than weak controls? | Fig 12, Fig 13 | FD, MMD, coverage, density, centroid cosine, diversity ratio, composite score, bootstrap CIs, optional downstream and DE summaries | `src/evaluation/baseline_registry.py`, `src/evaluation/model_benchmarking.py`, `src/visualization/baseline_panels.py`, `src/visualization/benchmark_panels.py` | Baselines would match CLOP-DiT on the same biologically grounded metrics or the ranking would be unstable across CIs |

## Figure-by-Figure Localization

| Article figure | Biological question | Producer |
|------|------|------|
| Fig 1 `fig_architecture` | Where in the pipeline is biological information introduced, transformed, and decoded? | `scripts/generate_architecture_figure.py` |
| Fig 2 `fig_training_dynamics` | Is the conditioning pipeline trained stably enough for biological interpretation? | `src/visualization/panels_training.py` |
| Fig 3 `fig_embedding_space` | Do text and cell embeddings align, and do generated latents remain on-manifold? | `src/visualization/panels_merged.py` |
| Fig 4 `panel_d_metrics_summary` | Does conditioning outperform chance and unconditional generation on core type-aware metrics? | `src/visualization/panels_metrics.py` |
| Fig 5 `fig_fidelity_alignment` | Which cell types work well, which fail, and how well do text and cell centroids align? | `src/visualization/panels_merged.py` |
| Fig 6 `panel_n_marker_gene_comparison` | Are canonical lineage markers preserved in generated expression? | `src/visualization/panels_expression.py` |
| Fig 7 `panel_h_expression_correlation` | Are gene-wise expression shifts faithful rather than visually similar only in embedding space? | `src/visualization/panels_expression.py` |
| Fig 8 `panel_i_expression_analysis` | Are higher-order expression statistics preserved? | `src/visualization/panels_expression.py` |
| Fig 9 `panel_m_conditioning_umap` | Does the conditioning space separate controllable regimes that map to distinct cell clouds? | `scripts/conditioning_analysis.py` |
| Fig 10 `panel_j_diversity_diagnostics` | Where does diversity compress, and under what guidance settings? | `scripts/diversity_diagnostics.py` |
| Fig 11 `fig_diversity_tradeoff` | Is diversity loss global or concentrated in a few cell-type tails? | `src/visualization/results_visualizer.py` plus panels from `panels_diversity.py` and `panels_conditioning.py` |
| Fig 12 `panel_o_baseline_comparison` | Do simple non-text baselines explain the apparent gains? | `src/visualization/baseline_panels.py` |
| Fig 13 `panel_s_benchmark` | Is the benchmark broad, uncertainty-aware, and stable across metrics? | `src/evaluation/baseline_registry.py` + `src/evaluation/model_benchmarking.py` + `src/visualization/benchmark_panels.py` |
| Fig 14 `fig_downstream_pq` | Do generated cells integrate with real cells in clustering and classifier transfer workflows? | `src/evaluation/downstream_biology.py` + `src/visualization/downstream_panels.py` |
| Fig 15 `panel_r_de_concordance` | Are biologically meaningful contrasts preserved at the DE level? | `src/evaluation/downstream_biology.py` + `src/visualization/panels_de_concordance.py` |

## How To Answer Reviewers Quickly

- "Where is biological meaning shown?"
  - Point to Fig 6-8 for gene-level fidelity, Fig 14-15 for downstream biological usefulness, and Fig 5 for per-type heterogeneity rather than only global means.
- "How do you know the method is robust?"
  - Point to Fig 10-11 for CFG and diversity sensitivity, Fig 13 for bootstrap confidence intervals, and the robustness harness in `scripts/training/run_robustness_experiments.py`.
- "How do you know text matters?"
  - Point to Fig 3-5 and the unconditional-collapse control in Fig 4.
- "How do you know this is not only an embedding trick?"
  - Point to gene-expression and DE figures: Fig 6-8 and Fig 15.
- "How are baselines compared fairly?"
  - Point to `src/evaluation/baseline_registry.py`, the artifact contract under `results/baselines/{method}/`, and the per-method downstream wrapper `scripts/analysis/run_downstream_for_baselines.py`.
