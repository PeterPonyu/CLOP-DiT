# external/baselines/

External baseline source code and metadata for benchmark comparison against CLOP-DiT.

## Current Status

No external baselines are pinned yet. The baseline registry in `src/evaluation/baseline_registry.py` defines two learned baseline methods that will be tracked here once external sources are integrated:

| Method | Slug | Source | Notes |
|--------|------|--------|-------|
| EmbeddingVAE | `embedding_vae` | Project-internal | VAE trained on cached scGPT embeddings. Config: `configs/baselines/embedding_vae.yaml`. No external dependency. |
| scVI Latent | `scvi_latent` | scvi-tools | Variational inference model trained on expression matrices. Config: `configs/baselines/scvi.yaml`. Depends on the `scvi-tools` pip package. |

Additionally, four built-in synthetic controls (`gaussian`, `shuffled_labels`, `random_normal`, `mean_only`) are generated at evaluation time and require no external code.

## Expected Layout (When Populated)

When an external baseline is pinned, create a subdirectory here with:

```
external/baselines/<method>/
    SOURCE.txt          # upstream URL, commit/tag, fetch date
    LICENSE             # original license
    PATCH_NOTES.md      # any modifications made for compatibility
```

## Related Directories

- Training configs: `configs/baselines/`
- Trained checkpoints: `models/baselines/`
- Evaluation artifacts: `results/baselines/{method}/`
- Registry code: `src/evaluation/baseline_registry.py`
