# models/baselines/

Trained checkpoint files for learned benchmark baselines evaluated against CLOP-DiT.

## Expected Checkpoints

| File | Method | Training Script | Config |
|------|--------|----------------|--------|
| `embedding_vae.pt` | EmbeddingVAE | `scripts/baselines/train_embedding_vae.py` | `configs/baselines/embedding_vae.yaml` |
| `scvi_latent.pt` | scVI Latent | `scripts/baselines/train_scvi.py` | `configs/baselines/scvi.yaml` |

These checkpoints are produced by the baseline training scripts under `scripts/baselines/` and consumed during benchmark evaluation (`src/evaluation/model_benchmarking.py`).

## How They Connect to the Pipeline

1. **Train**: Run the baseline training script, which reads data from `data/cached_latents_v5.2/` (EmbeddingVAE) or `results/real_expression.npy` (scVI) and saves the checkpoint here.
2. **Generate**: The training script also writes evaluation artifacts (embeddings, labels, metadata) to `results/baselines/{method}/`.
3. **Benchmark**: `src/evaluation/model_benchmarking.py` reads artifacts from `results/baselines/` (not the checkpoints directly) to compute metrics across all registered methods.

## Related Directories

- Baseline configs: `configs/baselines/`
- External source metadata: `external/baselines/`
- Evaluation artifacts: `results/baselines/{method}/`
- CLOP-DiT checkpoints: `models/checkpoints/`
