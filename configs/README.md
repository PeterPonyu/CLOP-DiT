# configs/

Configuration files for CLOP-DiT training, inference, and evaluation.
All paths inside YAML files are relative to the project root unless absolute.

## Core Configs

| File | Purpose |
|------|---------|
| `clop.yaml` | CLOP contrastive alignment training (Prototype-SigLIP loss, projector architecture, batch size, LR, etc.) |
| `dit.yaml` | DiT flow-matching training (transformer blocks, guidance dropout, time sampling, inference steps) |
| `pipeline.yaml` | Central path registry consumed by all scripts (`cache_dir`, `checkpoint_dir`, `results_dir`, `fig_dir`, `article_figures_dir`). Override any path via `CLOPDIT_*` environment variables. |
| `models.yaml` | Model registry with preset encoder/decoder combinations (cell encoder: scGPT variants, text encoder: BiomedBERT/BioLinkBERT/SciBERT/BioBERT, decoder: scGPT). Defines the active preset and ablation matrix. |

## Baseline Configs

`baselines/` contains YAML configs for benchmark baseline methods:

- `embedding_vae.yaml` -- VAE trained on cached scGPT cell embeddings (hidden 512, latent 64).
- `scvi.yaml` -- scVI latent model trained on real expression matrices (latent 32, 200 epochs).

These are consumed by `scripts/baselines/` training scripts and write artifacts to `results/baselines/{method}/`.

## Supplementary Files

| File | Purpose |
|------|---------|
| `downstream_contrasts_extended.json` | Cell-type pairs for DE concordance analysis (e.g., CD8+ T vs CD4+ T, macrophages vs monocytes). |
| `robustness_prompt_variants.example.json` | Example config for robustness experiments pointing to prompt variant files. |

The directory `prompts/` contains JSON files used for caption-sensitivity robustness runs: `default_prompts.json`, `marker_dropped_prompts.json`, and `context_shuffled_prompts.json`. Each file maps cell type names to prompt text (see `scripts/08_biological_validation.py --prompt_file`). The example config references these paths; do not remove or rename them if you use the default robustness workflow.

## Archive

`archive/` stores historical CLOP configs (v6.41 through v9.3) used during hyperparameter sweeps and ablation experiments. These are kept for reproducibility but are not used by current training scripts.
