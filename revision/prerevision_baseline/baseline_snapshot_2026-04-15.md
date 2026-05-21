# Performance Baseline Snapshot — 2026-04-15

This file records the current baseline before further revision experiments, so future model or data changes can be compared against a fixed reference rather than against memory.

## Snapshot metadata

- **Date**: 2026-04-15
- **Branch**: `github-ready`
- **Commit**: `3b6f38a67e8d2237e1c3e9b51a96be1d4a5c2c6b`
- **Purpose**: Preserve a stable pre-revision reference point for tracking metric changes during reviewer-driven modifications.

## Recommended source-of-truth files

- `articles/clop_dit_genes.tex`
- `results/comprehensive_summary.json`
- `results/ood_evaluation/ood_marker_analysis.json`
- `results/rare_cell_augmentation/augmentation_results.json`
- `results/conditioning_ablation/field_ablation_results.json`
- `revision/reviewer_response_draft.md`

## Headline manuscript metrics

### Core operating regimes

| Regime / Method | KNN-1 | KNN-5 | Steering | DivR | LinAcc | FD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real data | 0.890 | 0.993 | — | 1.000 | 0.942 | 0.00 |
| High-fidelity (`CFG=2.0`, Euler-10) | 0.369 | 0.553 | 0.810 | 0.513 | 0.511 | 3.17 |
| High-diversity (`CFG=1.0`, Midpoint-10) | 0.288 | 0.461 | 0.807 | 0.929 | 0.357 | 2.64 |
| Unconditional (`CFG=0.0`) | 0.010 | 0.051 | 0.475 | 1.833 | 0.010 | 0.58 |
| Gaussian baseline | 0.011 | 0.048 | 0.466 | 2.277 | 0.009 | 0.03 |

### Bootstrap reference point

These come from the bootstrap summary configuration already used in the manuscript.

| Metric | Point estimate | 95% CI |
| --- | ---: | --- |
| KNN-1 | 0.509 | [0.421, 0.586] |
| KNN-5 | 0.728 | [0.658, 0.794] |
| Steering | 1.000 | [1.000, 1.000] |
| DivR | 0.969 | [0.963, 0.974] |
| Linear accuracy | 0.583 | [0.518, 0.646] |
| Centroid cosine | 0.898 | [0.875, 0.913] |

## Cross-dataset / expression-fidelity baseline

### Decoder-reconstructed cross-dataset table

- `r_mean > 0.999` across all 5 datasets
- `R^2 > 0.998` across all 5 datasets
- `r_var` range: **-0.019 to +0.013**
- `FD_embed` range: **62.1 to 155.4**
- Current interpretation: mean expression is preserved much better than gene-wise variance structure

### Stricter tissue-level summary from `results/comprehensive_summary.json`

| Tissue | Pearson r | Spearman rho | PCA overlap |
| --- | ---: | ---: | ---: |
| Lung | 0.559 | 0.576 | 0.327 |
| Gastric | 0.494 | 0.508 | 0.346 |
| Skin | 0.445 | 0.401 | 0.289 |
| Liver | 0.233 | 0.190 | 0.255 |
| Blood | 0.418 | 0.384 | 0.251 |

## Conditioning / semantics baseline

### Field ablation

| Variant | KNN-1 | Steering | Mean centroid cosine |
| --- | ---: | ---: | ---: |
| Full | 0.0287 | 0.998 | 0.0501 |
| No markers | 0.0226 | 0.980 | 0.0349 |
| Shuffled markers | 0.0281 | 0.979 | 0.0352 |
| Metadata only | 0.0151 | 0.624 | 0.0250 |
| External markers (15 types) | 0.1707 | — | 0.7023 vs full |

### OOD / free-form prompt summary

- Novel prompt count: **6**
- Novel prompt mean marker-hit rate: **0.167**
- Novel prompts with any hit: **1 / 6**
- Free-form vs structured expression cosine: **1.0** for all tested pairs
- Conditioning-vector cosine range: **0.73–0.94**
- DiT latent cosine range: **0.39–0.54**
- Current interpretation: upstream conditioning differentiates prompts, but the frozen decoder collapses those differences at expression level

## Rare-cell augmentation baseline

### Cycling-cell pilot

| Setting | Cycling-cell F1 | Macro-F1 |
| --- | ---: | ---: |
| Baseline | 0.828 | 0.964 |
| +1x synthetic | 0.786 | 0.953 |
| +5x synthetic | 0.741 | 0.947 |
| +10x synthetic | 0.741 | 0.947 |

Current interpretation: generated profiles are not catastrophic, but they do not add enough within-type heterogeneity to improve rare-class decision boundaries.

## Why this backup matters

This snapshot should be treated as the **pre-revision baseline**. Any future experiment should be compared against it directly, not against a moving target.

In practice, every future modification should record:

1. **What changed** (e.g. new text encoder, extra mouse datasets, OOD holdout split, ZCA ablation)
2. **Which reviewer concern it targets**
3. **Whether the change improved, worsened, or traded off**:
   - high-fidelity KNN / Steering / DivR
   - high-diversity KNN / Steering / DivR
   - `r_var`
   - OOD marker-hit rate
   - rare-cell F1
   - organism-stratified performance (if added)
4. **Whether the gain is likely upstream (conditioning), midstream (latent generator), or downstream (decoder/interface)**

## Suggested comparison rule

A modification should not be considered a clear improvement unless it satisfies at least one of the following:

- Improves the target reviewer-facing metric **without** substantially damaging the main operating-regime metrics, or
- Improves a reviewer-critical weakness (e.g. OOD, organism stratification, rare augmentation, variance fidelity) while leaving the main operating-regime metrics approximately stable, or
- Produces a clearly interpretable trade-off that can be defended explicitly in the revision.

## Best practice

Keep this file unchanged. For each new experiment, create a separate comparison note using the companion template in:

- `revision/performance_tracking/comparison_template.md`
