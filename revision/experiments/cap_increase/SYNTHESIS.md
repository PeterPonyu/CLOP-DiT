# Cap-increase experiment — synthesis

**Ran:** 2026-04-15
**Anchor tag:** `pre-cap-increase-2026-04-15`
**Scope:** 50 non-geodh datasets re-run end-to-end at `max_cells=10000`
with `seed=0`, plus a matched `max_cells=3000` re-run with the same
seed so the two caps differ by nothing except the cap itself.

## Pipelines compared

| tier | preprocess | embed | subcluster |
|---|---|---|---|
| cap3k_seed0 | `data/processed_h5ad_cap3k_seed0/` | `data/cached_latents_cap3k_seed0/` | 987 clusters / 815 annotated |
| cap10k | `data/processed_h5ad_cap10k/` | `data/cached_latents_cap10k/` | 1212 clusters / 1004 annotated |

## Headline: **the cap is not the bottleneck**

With fixed seed, cap10k yields **3.12 x** more cells per dataset but
only **~1 %** more latent variance overall and **–2.4 % (median)**
within-cluster variance.

### Per-dataset comparison (50 paired datasets)

```
scGPT latent space, pre-CLOP
  cells     : 138 477 (cap3k_seed0) → 432 354 (cap10k), 3.12 x
  median variance ratio cap10k / cap3k = 1.008
  mean                                = 1.010
  IQR                                 = [0.984, 1.035]
  > 1    : 29 / 50
  > 1.5  : 0 / 50
```

### Within-cluster comparison (475 paired (dataset, cell-type) pairs)

```
scGPT latent space, per subcluster
  median variance ratio cap10k / cap3k = 0.976
  mean                                = 0.987
  IQR                                 = [0.895, 1.079]
  > 1    : 202 / 475
  > 1.5  :   7 / 475
  robust mean-L2 ratio: median 0.988
```

### What changed between the first-pass and the clean comparison

The first cap3k-vs-cap10k diagnostic reported `median = 0.989` and a
bimodal per-dataset distribution (gains up to 5 x, losses down to
0.16 x). That was **entirely a seed artefact** — the cap3k baseline
cache had been produced without a fixed `random_state` in
`sc.pp.subsample`, whereas cap10k used seed 0. Running cap3k again
under the same seed collapses the IQR to `[0.984, 1.035]`, reveals
the per-dataset effect is essentially null, and exposes the bimodal
tails as noise.

## Mechanistic read

Within-type variance in **scGPT latent space saturates long before
3 000 cells**. The scGPT encoder's raw-count → rank-binning → frozen
transformer path has limited capacity to represent the fine-grained
variation that reviewer R2.11 flags as missing. Feeding 3 x more
cells does not rescue it because the encoder's output manifold
already contains what it is going to contain at modest input sizes.

This is **consistent with the A5 + B1 Lane-A / Lane-B diagnosis**
that latent-stage under-dispersion is upstream of the DiT and is not
a pipeline-hyperparameter problem. A cap-increase cannot fix an
encoder-side bottleneck.

## Implications for the revision

1. **Do not advertise cap-increase as a heterogeneity fix in the
   manuscript.** The seed-aligned comparison disproves it at the
   scGPT latent level.
2. **The pilot experiment itself is a valuable negative result.**
   Adding a Discussion paragraph explicitly stating "we tested
   whether larger per-dataset caps would enlarge within-type latent
   variance, and found no effect" strengthens the existing story
   that the bottleneck is encoder-side.
3. **Do not retrain CLOP / DiT on the cap10k cache.** There is no
   downstream latent signal to train on — the CLOP aligner operates
   on scGPT outputs whose within-type distribution is identical.
4. **The next productive direction is generator-side, not data-side:**
   heterogeneity-preserving objectives (e.g. variance-preserving
   loss, moment-matching regularisation) during DiT training, which
   can affect the learned distribution independently of the fixed
   encoder manifold.

## Rebuttal-ready addendum (paste into R2.11 / R3.3 response)

> At the reviewer's suggestion, we also tested whether relaxing the
> per-dataset cell cap from 3 000 to 10 000 would expose additional
> within-type latent variance. Preprocessing and scGPT encoding were
> re-run on 50 datasets at both caps with a fixed `sc.pp.subsample`
> seed so only the cap itself differed. The median per-dataset total
> variance ratio cap10k / cap3k in scGPT latent space is 1.008 (IQR
> [0.984, 1.035]), and the median within-cluster variance ratio,
> pooled over 475 matched (dataset, cell-type) pairs, is 0.976 (IQR
> [0.895, 1.079]). 3.12 x more input cells therefore produce no
> practically-meaningful increase in latent heterogeneity. This
> further corroborates the A5 + B1 diagnosis that the bottleneck is
> at the generator / encoder stage upstream of the DiT, not in the
> data volume.

## Artefacts

- `data/processed_h5ad_cap3k_seed0/` (50 × processed h5ad + subcluster
  metadata, 987 clusters)
- `data/processed_h5ad_cap10k/` (50 × processed h5ad + subcluster
  metadata, 1212 clusters)
- `data/cached_latents_cap3k_seed0/` (138 477 × 512 scGPT latents)
- `data/cached_latents_cap10k/` (432 354 × 512 scGPT latents)
- `revision/experiments/cap_increase/seed0_variance_comparison.json`
- `revision/experiments/cap_increase/within_cluster_variance.json`

Safety anchor (rollback reference): tag `pre-cap-increase-2026-04-15`.
