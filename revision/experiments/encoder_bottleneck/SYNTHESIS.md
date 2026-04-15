# Encoder bottleneck — three-experiment synthesis

**Ran:** 2026-04-15
**Anchor tag:** `pre-cap-increase-2026-04-15`
**Goal:** Determine whether the within-type latent under-dispersion
flagged by reviewer R2.11 originates upstream (frozen scGPT
encoder) or downstream (DiT generator), and whether any input-side
lever can expand the latent manifold.

## Three experiments, one verdict

| experiment | input lever | range tested | within-type latent variance change |
|---|---|---|---:|
| Cap-increase (cells / dataset) | `max_cells` | 3 000 → 10 000 (3.12 x) | median ratio **1.008** (essentially zero) |
| HVG-count ablation | `n_top_genes` | 500 → 8 000 (16 x) | median range **11.2 → 8.5** then back to 8.8 (curve flattens at >= 2000) |
| Raw-vs-latent compression | (no lever; diagnostic) | – | latent within-L2 = **0.168 x** raw within-L2 (encoder compresses 6 x) |

All three independently say the same thing: **the scGPT encoder
absorbs whatever input you throw at it into the same compact
latent manifold**. Adding cells does nothing; adding genes makes
clusters slightly tighter; raw-space heterogeneity is collapsed by
~6 x at encoder stage.

## Numerical detail

### Cap-increase (commit `de378ea`, `2e3d870`)

```
seed-aligned cap10k vs cap3k_seed0 (50 datasets, 432k vs 138k cells):
  per-dataset variance ratio   median 1.008  IQR [0.984, 1.035]
  within-cluster (475 pairs)   median 0.976  IQR [0.895, 1.079]
  bimodal tails of first-pass diagnostic = 100 % seed drift artefact
```

### HVG-count ablation (commit `e72124b`, `addd49f`)

```
n_top_genes  median_var  ratio_to_2000   shape
   500          11.178       1.217       compressed signal -> wider cluster
  1000          10.126       1.102
  2000           9.185       1.000       baseline
  4000           8.486       0.924       curve minimum
  8000           8.782       0.956       slight rebound
```

scGPT latent within-cluster variance is essentially flat (±8 %)
across the 2000–8000 HVG range. The 500 / 1000 elevation is the
expected "fewer genes -> noisier embedding" effect; once enough
HVGs are present the encoder saturates.

### Raw-vs-latent tightness (commit `0b73b71`)

```
Per (dataset, cell_type) cluster:
  within-type L2 ratio  (latent / raw)  median 0.168  IQR [0.132, 0.214]
  between-type L2 ratio (latent / raw)  median 0.274  IQR [0.237, 0.299]
  tightness ratio (within/between)      median 0.647  IQR [0.583, 0.731]
```

The encoder compresses within-type variation **6 x more aggressively
than the raw HVG signal would imply**, while between-type centroids
shrink only 3.7 x. Net effect: clusters are MORE separable in latent
space (overall tightness drops 35 %) but the within-type fine
structure that R2.11 cares about is disproportionately lost.

## Interpretation

scGPT's pipeline (raw counts -> rank-binning -> frozen 12-layer
transformer -> 512-d cell embedding) has a hard upper bound on the
within-type variance it can represent. Adding more cells of the same
type, or feeding more HVGs per cell, hits that bound; the manifold
does not grow. This is a property of the frozen encoder, not of the
CLOP aligner or the DiT generator downstream.

## Implications for the revision

1. **The within-type heterogeneity gap reviewer R2.11 flags is a
   property of the latent space we generate into, not a flaw in the
   DiT.** This sharpens the existing A5 + B1 diagnosis.
2. **Three independent input-side levers fail to move the latent
   variance** (cap-cells, n_top_genes, and seed) — so any "more data"
   suggestion (R3.3-style) is empirically refuted in this regime.
3. **The productive next levers are encoder-side** (replace scGPT,
   fine-tune scGPT, or add a variance-preserving CLOP loss) **or
   generator-side** (variance-preserving DiT objective, latent
   noise injection). Both are scoped for follow-up; we now have the
   evidence to motivate them.

## Paste-ready Discussion paragraph

> Three independent experiments converge on encoder-stage latent
> compression as the limiting factor for within-type heterogeneity in
> our pipeline. Increasing the per-dataset cell cap from 3 000 to
> 10 000 (3.12 x more cells, fixed seed) leaves median per-dataset
> latent variance unchanged (ratio 1.008) and median within-cluster
> variance slightly reduced (0.976) across 475 matched (dataset, cell-
> type) pairs. Sweeping the HVG count from 500 to 8 000 produces only
> a 24 % spread in latent variance, which saturates above 2 000 HVGs
> and slightly rebounds at 8 000. Direct comparison of raw HVG-space
> against scGPT latent-space tightness shows the encoder compresses
> within-type L2 distance by ~6 x (median ratio 0.168) while between-
> type centroid distance only by ~3.7 x (median ratio 0.274), so
> labels remain well-separated in latent space but the within-type
> fine structure is disproportionately lost. The within-type
> heterogeneity gap is therefore a property of the frozen scGPT
> manifold, not of the CLOP aligner or the DiT generator we train on
> top of it; productive future levers are encoder replacement or a
> variance-preserving loss in the CLOP / DiT stack.

## Artefacts (gitignored heavy data; small JSONs versioned)

- `data/processed_h5ad_hvg{500,1000,4000,8000}/`
- `data/cached_latents_hvg{500,1000,4000,8000}/`
- `data/cached_latents_cap{3k_seed0,10k}/`
- `revision/experiments/encoder_bottleneck/`:
  - `raw_vs_latent_variance.json` + script `a_raw_vs_latent_variance.py`
  - `hvg_ablation_summary.json` + driver `b_hvg_ablation.sh` + analyzer `b_analyze_hvg.py`
- `revision/experiments/cap_increase/SYNTHESIS.md` (cap-increase result)
