# src/

Main Python package for CLOP-DiT: text-conditioned single-cell generation via contrastive alignment and flow-matching diffusion.

## Subpackages

### `architecture/`
Neural network definitions: CLOP contrastive aligner, DiT diffusion transformer, scGPT decoder, Cell2Cell translation module, and standalone scGPT embedding extractor. See `src/architecture/README.md`.

### `data_pipeline/`
Data loading and preprocessing:

- `dataset.py` -- `CLOPDataset` / `DiTDataset` PyTorch datasets over cached latents.
- `cache_builder.py` -- Builds the cached latent directory from raw h5ad files.
- `embedding_preprocessor.py` -- ZCA whitening of cell/text embeddings (fixes collapse).
- `geo_fetcher.py` -- GEO metadata download and processing.
- `group_aware_sampler.py` -- Batch sampler ensuring prototype collisions for Prototype-SigLIP.
- `text_cleaner.py` -- Template-anchored text normalization (strips dataset-specific suffixes).
- `subcluster_annotation.py` -- Subcluster description generation.

### `evaluation/`
Three evaluation tracks: embedding-space metrics, biological validation, and downstream biology. Includes the baseline registry and multi-method benchmarking engine. See `src/evaluation/README.md`.

### `training/`
Training loops and schedulers:

- `train_clop.py` -- CLOP alignment training loop (Prototype-SigLIP, EMA, early stopping).
- `train_dit.py` -- DiT flow-matching training loop (logit-normal time sampling, CFG).
- `train_cell2cell.py` -- Cell2Cell conditional flow-matching training.
- `schedulers.py` -- Cosine-annealing with warmup LR scheduler.

### `utils/`
Shared utilities:

- `paths.py` -- Canonical path constants derived from `configs/pipeline.yaml`.
- `helpers.py` -- Seed setting, device selection, YAML loading.
- `logging_config.py` -- Structured logging setup.
- `experiment_tracker.py` -- Run metadata, config snapshots, metric history.

### `visualization/`
Publication figure generation with 19 manuscript panels (A through S), a central orchestrator, and shared style system. See `src/visualization/README.md`.
