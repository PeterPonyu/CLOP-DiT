# ZCA whitening ablation — synthesis

**Ran:** 2026-04-15
**Goal:** Determine whether ZCA whitening in the CLOP preprocessing
pipeline is a critical component or a marginal refinement, and
quantify its contribution to alignment quality.

## Experimental design

Three CLOP training runs with identical architecture, loss
(PrototypeSigLIP), and hyperparameters. Only the embedding
preprocessing differs:

| condition | text preprocessing | cell preprocessing | effect on embedding space |
|---|---|---|---|
| **whiten** (ZCA) | mean-center + ZCA + L2 | mean-center + ZCA + L2 | text cos 0.94 → −0.01, cell cos 0.99 → 0.43 |
| **center_norm** | mean-center + L2 | mean-center + L2 | text cos 0.94 → 0.02, cell cos 0.99 → 0.87 |
| **none** | raw | raw | text cos ≈ 0.94, cell cos ≈ 0.99 (collapsed) |

Each condition trains 100 epochs, seed 42, batch 1024, lr 5e-4,
on the production `data/cached_latents` (150 550 train / 16 695 val
cells, 69 types). Checkpoints saved to
`models/revision/zca_ablation/{condition}/`.

## Results

| condition | quality | proto_acc | tc_align | separation | cos_sim | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| **whiten (ZCA)** | **0.9656** | 0.9991 | 0.9783 | 1.0108 | **0.8512** | 98 |
| center_norm | 0.9600 | 0.9995 | 0.9839 | 1.0102 | 0.8148 | 96 |
| none | 0.9562 | 0.9982 | 0.9769 | 1.0099 | 0.8085 | 99 |

Quality score = 0.3 × proto_acc + 0.3 × tc_align + 0.2 × separation
+ 0.2 × cos_sim.

**Ratios vs whiten baseline:**

| condition | quality ratio | cos_sim ratio |
|---|---:|---:|
| center_norm | 0.994 (−0.6 %) | 0.957 (−4.3 %) |
| none | 0.990 (−1.0 %) | 0.950 (−5.0 %) |

## Interpretation

1. **ZCA whitening is a modest refinement, not a critical component.**
   Composite quality drops only 1 % when ZCA is removed entirely.
   Prototype accuracy (the coarse type-level alignment metric) is
   near-perfect (≥ 0.998) across all conditions — the 3-layer MLP
   projectors + PrototypeSigLIP loss are powerful enough to learn
   through the raw embedding collapse (pairwise cosine 0.94 / 0.99).

2. **ZCA's main contribution is to fine-grained cosine structure.**
   The largest gap between conditions is in `cos_sim` (0.851 vs
   0.809, a 5 % difference). ZCA decorrelates the cell embedding
   dimensions (cos 0.99 → 0.43), giving the projection heads a
   richer input signal to work with. Without decorrelation, the
   projector must learn to extract the same information from a much
   more correlated input — it mostly succeeds, but the fine-grained
   positive-pair cosine similarity is modestly lower.

3. **center_norm captures most of the benefit.** Mean-centering +
   L2 normalization alone closes ~40 % of the gap between `none`
   and `whiten`. The remaining benefit of ZCA is the eigenvalue
   equalisation step that decorrelates cell dimensions.

4. **The finding is consistent with the encoder-bottleneck diagnosis.**
   Since the scGPT encoder already compresses within-type variance by
   ~6 × (Section 8 of REVISION_LOG), the CLOP aligner operates in a
   regime where most of the useful signal is in the between-type
   centroid structure, which is well-preserved even without
   preprocessing. ZCA's marginal 5 % cos_sim gain reflects its
   ability to surface within-type fine structure that the encoder
   partially retains.

## Implications for the revision

- **ZCA whitening should be kept as a best-practice default** (it
  helps and costs nothing at inference time), but it should NOT be
  presented as a critical architectural choice. The aligner is
  robust to its removal.
- **The reviewer question "does ZCA matter?" has a clear answer:**
  aggregate alignment quality is stable (−1 %), fine-grained cosine
  similarity drops modestly (−5 %), type-level prototype accuracy
  is unaffected (≥ 0.998).
- **A downstream DiT retrain** (feeding ZCA-free CLOP projections
  into the diffusion model) is the next step to confirm whether the
  5 % cos_sim gap propagates to generated expression quality. This
  is scoped for B4 if reviewer follow-up requires it.

## Paste-ready text

> We ablated ZCA whitening in the CLOP preprocessing pipeline by
> training identical aligners under three conditions: full ZCA
> whitening (production default), mean-centre + L2 normalisation only,
> and no preprocessing (raw collapsed embeddings). Composite alignment
> quality (0.3 × prototype accuracy + 0.3 × text–cell alignment + 0.2
> × inter-type separation + 0.2 × positive-pair cosine similarity) was
> 0.966, 0.960, and 0.956 respectively — a spread of only 1 %.
> Prototype accuracy was near-perfect (≥ 0.998) in all conditions,
> confirming that the three-layer MLP projectors and PrototypeSigLIP
> loss are robust to input-space collapse. The primary contribution of
> ZCA is a 5 % improvement in positive-pair cosine similarity (0.851
> vs 0.809), reflecting decorrelation of the 512-d scGPT cell
> embedding dimensions. ZCA is a useful default but not a critical
> architectural choice.

## Artefacts

- Configs: `revision/experiments/zca_ablation/configs/clop_{whiten,center_norm,none}.yaml`
- Driver: `revision/experiments/zca_ablation/run_ablation.sh`
- Trainer wrapper: `revision/experiments/zca_ablation/train_clop_ablation.py`
- Analysis: `revision/experiments/zca_ablation/analyze_ablation.py`
- Checkpoints (gitignored): `models/revision/zca_ablation/{whiten,center_norm,none}/`
- Symlink cache (gitignored): `data/cached_latents_ablation_centernorm/`
- Results: `revision/experiments/zca_ablation/zca_ablation_summary.json`
