# Cap3k vs cap10k preprocessing comparison

**Ran on:** 2026-04-15
**Seed:** `--subsample_seed 0` (deterministic)
**Output:** `data/processed_h5ad_cap10k/` (isolated, 3.7 GB, 50 h5ad files)

## Scope

50 datasets from `DATASET_DIRS` (CancerDatasets, CancerDatasets2,
DevelopmentDatasets, DevelopmentDatasets2, datasets/). 5 incompatible
datasets skipped (same as cap3k baseline): `GSE120575_melanomaHmCancer`,
`GSE148215_hESCHSPCD8Hm`, `GSE225948_bloodMmStrokeDev`,
`GSE247719_PanSci_05_Muscle_adata`, `GSE247719_PanSci_T_cell_adata`.

Geodh (11 datasets) and scRNA25100 (~12 datasets) not yet
preprocessed at cap10k — they require separate source-dir plumbing
and will be handled in a follow-up if the downstream pipeline
benefits justify the effort.

## Headline

- Total cells across the 50 shared datasets:
  - cap3k: **138,477**
  - cap10k: **432,354**
  - ratio: **3.12 x**
- Wall-clock: **5 min 2 s** (single CPU, 55 datasets scanned)

## Per-dataset cell-count gains

For datasets with raw count > 30,000, cap10k gives the expected
~3.35 x gain (subsample-bound). For small datasets already below
cap3k, no change or modest gain.

```
Top 15 by cap10k / cap3k ratio
  dataset                                        3k    10k     x
  GSE115571_LPSMmDev                            442   1505  3.40
  GSE228499_breastHmCancer                     2448   8239  3.37
  GSE213740_ADHm                               2704   9093  3.36
  GSE225600_breast_CancerHm                    2333   7825  3.35
  ...
Bottom 5 (no gain expected)
  endo                                         2531   2531  1.00
  GSE98638_TcellLiverHmCancer                  3000   5063  1.69
```

Note: `GSE115571_LPSMmDev` cap3k had 442 cells because the dataset
is tiny and fell below the 3000 cap after QC. At cap10k the raw
sample is larger, yielding 1505 cells after the same QC filters.

## Next stage (not yet run)

1. **scGPT embedding** at cap10k (`03_cache_latents.py`): generates
   `cell_embeddings.npy` at 432k cells x 512-d. Heavy GPU step.
2. **CLOP alignment training** (`04a_*.py`): needs the embeddings.
3. **DiT training** (`04b_*.py`): standard DiT training loop.
4. **Generation + evaluation**: compare to cap3k baseline on per-type
   within-type heterogeneity metrics (the whole reason we did this).

All downstream artifacts should be written to cap10k-suffixed paths
so the baseline remains untouched and both can be compared
head-to-head.

## Reproducibility

```
python scripts/data_prep/00_prepare_all_data.py \
  --output_dir data/processed_h5ad_cap10k \
  --max_cells 10000 \
  --subsample_seed 0
```

Commit: `c33e90e` (seed-fix) + `85d64e6` (smoke test).
