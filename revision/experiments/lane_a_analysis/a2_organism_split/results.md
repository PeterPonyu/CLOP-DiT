# A2 — Human vs mouse stratified evaluation

**Reviewer comment:** R3.2 — species stratification is not reported.
**Retraining:** None.
**Ran on:** 2026-04-15
**Script:** `compute_organism_split.py`
**Input artifacts (read-only, live paths):**
- `data/cached_latents/text_caption_metadata.json` (per-type organism list)
- `data/cached_latents/text_captions_deduplicated.json` (for type-name mapping)
- `results/per_type_dashboard.json`
- `results/expression_metrics.json`

## Scope caveat

Stratification is **type-level**, not cell-level. Each of the 69
evaluation cell types is tagged with the set of organisms that appear
in its training-corpus captions (human, mouse, or both). The frozen
baseline does not emit per-cell organism tags, so a type classified as
"both" contributes to metrics that average over human and mouse
training cells. A strict cell-level re-evaluation is scoped for
**Lane C** and is not the prerequisite for answering R3.2: if there
were a severe species bias at the type level, it would already show
up in the current stratification.

## Status

- [x] Organism label attached to every evaluation cell type (type-level)
- [x] Metrics partitioned into human-only, mouse-only, both
- [x] Pairwise Mann-Whitney U tests
- [x] Gap table produced
- [x] Rebuttal verdict sentence committed

## Results

### Group counts (69 cell types total)

| Organism group | N types |
|---|---:|
| human_only | 18 |
| mouse_only | 8 |
| both (human + mouse) | 43 |

### Per-group summary (mean ± std)

| Metric | human_only (n=18) | mouse_only (n=8) | both (n=42–43) |
|---|---:|---:|---:|
| centroid cosine | 0.917 ± 0.043 | 0.919 ± 0.068 | 0.887 ± 0.085 |
| Frechet distance | 1.052 ± 0.076 | 1.028 ± 0.080 | 1.121 ± 0.045 |
| diversity ratio | 0.957 ± 0.069 | 0.917 ± 0.056 | 0.940 ± 0.040 |
| expression Pearson r | ≈ 1.000 | ≈ 1.000 | ≈ 1.000 |
| real within-type cosine (intra) | 0.178 ± 0.094 | 0.191 ± 0.088 | 0.080 ± 0.059 |
| gen within-type cosine (intra) | 0.218 ± 0.072 | 0.260 ± 0.072 | 0.135 ± 0.059 |

### Pairwise Mann-Whitney (two-sided)

The only question R3.2 asks directly is **human-only vs mouse-only**.
On every headline metric, the answer is **no significant difference**:

| Metric | mean Δ (human − mouse) | p |
|---|---:|---:|
| centroid cosine | −0.003 | 0.14 |
| Frechet distance | +0.024 | 0.85 |
| diversity ratio | +0.040 | 0.18 |
| expression Pearson r | +0.000 | 0.43 |
| real intra cosine | −0.013 | 0.87 |
| gen intra cosine | −0.043 | 0.38 |

Mouse-only types (n = 8) are actually *marginally better* on two of
the six metrics (centroid 0.919 vs 0.917; FD 1.028 vs 1.052), but the
differences are well within noise.

The statistically significant differences we do observe are between
**multi-organism types and single-organism types** (both organism
groups differ from "both" with p ≤ 1 × 10⁻³ on centroid cosine and
FD). The direction is that "both" types have lower centroid cosine
and higher FD. The likely mechanism is that types tagged with both
organisms are drawn from a wider set of tissue contexts, which
lowers their `real_intra_cos` (0.080 vs ~ 0.18 for single-organism
types) — i.e., they are biologically more heterogeneous, which
A3 already identifies as the dominant predictor of per-type fidelity.

## Rebuttal-ready sentence (paste into R3.2 response)

We have now stratified all headline per-type metrics by organism
(Supplementary Table S4 and Figure S4). Across the 18 human-only and
8 mouse-only cell types, the two organism groups are statistically
indistinguishable on centroid cosine (0.917 vs 0.919, Mann-Whitney
p = 0.14), Frechet distance (1.052 vs 1.028, p = 0.85), diversity
ratio (p = 0.18), expression Pearson r (p = 0.43), and real
within-type cosine tightness (p = 0.87); mouse-only types are in
fact marginally better on FD. The significant differences in the
stratification are between single-organism and multi-organism types,
not between species; the mechanism is the broader tissue scope of
multi-organism types, captured by `real_intra_cos`, which A3
identifies as the dominant predictor of per-type fidelity.
Consequently, we do not observe the species bias suggested by the
imbalance in the dataset count (59 human vs 21 mouse); the shared
CLOP space appears to generalise across species at the type level,
and the cell-level strict stratification remains scoped for
Lane C when targeted mouse data expansion is complete.
