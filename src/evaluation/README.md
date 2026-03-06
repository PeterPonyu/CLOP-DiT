# src/evaluation/

Evaluation package for CLOP-DiT. Covers three complementary tracks plus a multi-method benchmarking engine.

## Evaluation Tracks

### 1. Embedding-Space Metrics
Core generative quality metrics computed on 512-d cell embeddings.

- `metrics.py` -- `GenerationMetrics`: Frechet distance, centroid cosine similarity, per-type fidelity, diversity ratio, memorization distance.
- `embedding_quality.py` -- Embedding space health checks (isotropy, effective rank, uniformity).
- `generative_metrics.py` -- Extended generation quality metrics.
- `visualizer.py` -- `EmbeddingVisualizer`: UMAP plotting for embeddings.

### 2. Biological Validation (`biological_validation/`)
End-to-end text-to-cell generation and biological plausibility analysis.

| Module | Role |
|--------|------|
| `constants.py` | Default prompts, marker gene sets (CD8+ T, macrophage, epithelial, NK), and style. |
| `generation.py` | `generate_cells_text2cell()`: runs CLOP + DiT + scGPT decode for given prompts. |
| `io.py` | Loaders for prompt files, subcluster metadata, and dataset specs. |
| `metrics.py` | Marker enrichment, DE analysis, FID on expression, PCA overlap, Spearman correlations. |
| `figures.py` | Publication figures 1--4 for the biological validation section. |
| `run.py` | Main entry point (`python -m src.evaluation.biological_validation`). |

### 3. Downstream Biology
`downstream_biology.py` constructs matched real/generated AnnData objects and runs:
- Clustering alignment (Leiden ARI/NMI)
- Classifier alignment (LogReg on real, evaluated on generated)
- DE concordance (logFC correlation for configured contrasts)
- kNN mixing score (cross-origin neighbour fraction)

Results are saved to `results/downstream/` and visualized in Panels P, Q, R.

## Benchmarking Infrastructure

- `baseline_registry.py` -- `MethodSpec` dataclass defining per-method metadata, artifact paths, and capabilities. Registered methods: `clop_dit` (primary), `embedding_vae`, `scvi_latent`, plus built-in synthetic controls (`gaussian`, `shuffled_labels`, `random_normal`, `mean_only`).
- `model_benchmarking.py` -- Benchmark engine that evaluates all registered methods under a common metric suite and produces `benchmark_report.json` (consumed by Panel S).
- `run_metrics.py` -- Shared entry point for loading trained CLOP/DiT models and running embedding-space metrics. Used by `scripts/06_evaluate.py`, `scripts/10_full_pipeline.py`, and `scripts/run_5fold_cv.py`.

## Artifact Contract

Each baseline method writes to `results/baselines/{method}/`:

| File | Required |
|------|----------|
| `embeddings.npy` | Yes |
| `labels.npy` | Yes |
| `metadata.json` | Yes |
| `expression.npy` | Optional |
| `expression_labels.npy` | Optional |
| `expression_metrics.json` | Optional |
