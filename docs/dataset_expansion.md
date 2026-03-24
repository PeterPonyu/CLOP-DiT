# Adding New Datasets to CLOP-DiT

This guide describes how to add new scRNA-seq datasets to the CLOP-DiT pipeline.

## Prerequisites

- A processed `.h5ad` file with annotated cell types
- The CLOP-DiT environment installed (`pip install -e .`)

## Step-by-Step Workflow

### 1. Prepare the h5ad File

Place your preprocessed AnnData (`.h5ad`) file in `data/processed_h5ad/`. The file
should contain:

- **X**: Gene expression matrix (raw counts or log-normalized)
- **obs['cell_type']**: Cell-type annotations
- **var**: Gene names as the index

Naming convention: `{GEO_ACCESSION}_processed.h5ad` (e.g., `GSE999999_processed.h5ad`).

If starting from raw GEO data, use the data-prep scripts:

```bash
# Download and preprocess from GEO
python scripts/data_prep/00_prepare_all_data.py --accessions GSE999999

# Or integrate existing h5/h5ad files
python scripts/data_prep/01_integrate_h5_datasets.py --input your_data.h5 --output data/processed_h5ad/GSE999999_processed.h5ad
```

### 2. Generate Structured Text Descriptions

Each dataset needs structured metadata describing its cell types. Generate
sub-cluster descriptions for the new dataset:

```bash
python scripts/data_prep/02_subcluster_descriptions.py \
    --h5ad data/processed_h5ad/GSE999999_processed.h5ad \
    --output data/cache/subcluster_metadata.json
```

Optionally enrich descriptions with pathway annotations:

```bash
python scripts/data_prep/02b_enrich_descriptions.py \
    --metadata data/cache/subcluster_metadata.json
```

### 3. Build the Latent Cache

Use `03_cache_latents.py` to encode cells and text into embeddings:

```bash
python scripts/data_prep/03_cache_latents.py \
    --h5ad_dir data/processed_h5ad/ \
    --metadata data/cache/subcluster_metadata.json \
    --scgpt_dir models/scgpt_human \
    --output data/cached_latents/
```

**Incremental mode**: If you already have a cache and are adding new datasets,
use `--incremental` to skip datasets that haven't changed:

```bash
python scripts/data_prep/03_cache_latents.py \
    --h5ad_dir data/processed_h5ad/ \
    --metadata data/cache/subcluster_metadata.json \
    --scgpt_dir models/scgpt_human \
    --output data/cached_latents/ \
    --incremental
```

### 4. Generate Caption Variants (Optional)

For improved robustness, generate paraphrased text variants:

```bash
python scripts/data_prep/generate_caption_variants.py \
    --metadata data/cache/subcluster_metadata.json
```

### 5. Rebuild the Deduplicated Cache

After adding new datasets, rebuild the deduplicated evaluation cache:

```bash
python scripts/data_prep/03c_build_dedup_cache.py
```

### 6. Retrain (if needed)

If you want the model to learn from the new data, retrain CLOP and DiT:

```bash
python scripts/pipeline/run_pipeline.py --stage all
```

## Data Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Cells per dataset | 100 | 500+ |
| Cell types per dataset | 1 | 3+ |
| Gene count | 2,000 HVGs | 2,000 HVGs |
| Organisms | Human or Mouse | Either |

## File Structure After Adding Data

```
data/
  processed_h5ad/
    GSE999999_processed.h5ad    ← Your new file
  cached_latents/
    cell_embeddings.npy         ← Updated with new cells
    text_embeddings_unique.npy  ← Updated with new texts
    processed_datasets.json     ← Tracks processed datasets
    manifest.json               ← Updated cache stats
```

## Notes

- All paths are resolved through `src/utils/paths.py`; override with environment
  variables if needed (e.g., `CLOPDIT_CACHE_DIR`)
- The scGPT encoder expects the same gene vocabulary used during pretraining;
  mismatched genes are silently dropped
- Run `pytest tests/` after adding data to verify pipeline integrity
