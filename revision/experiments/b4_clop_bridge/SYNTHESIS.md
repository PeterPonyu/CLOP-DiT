# B4 — CLOP ablation → full pipeline bridge

## Phase 1: Projected text space diagnostic (2026-04-15)

**Script:** `revision/experiments/b4_clop_bridge/diagnose_projected_text.py`
**Question:** Before committing 6+ hours of DiT retraining, do CLOP
ablation variants produce functionally equivalent conditioning vectors?
If so, ablation conclusions transfer trivially; if not, full DiT
retraining per variant is needed.

### Method

Loaded each CLOP ablation checkpoint, projected 1,088 unique text
embeddings, and compared: (a) absolute per-group cosine similarity,
(b) structural similarity (Pearson r of the 1088 x 1088 inter-group
cosine similarity matrix).

Variants: production (`clop_best.pth`), ablation baseline, no_cohesion,
no_cell_noise, fixed_temperature.

### Result: structurally different

**Production vs all ablation variants:** structural Pearson r = 0.04–0.06.
The production CLOP learned a completely different inter-type similarity
geometry than every ablation variant, including the ablation baseline
(identical config).

**Among ablation variants:** structural Pearson r = 0.36–0.76.
The most similar pair is abl_baseline ↔ no_cell_noise (r = 0.76);
the most divergent is no_cohesion ↔ fixed_temperature (r = 0.36).

| pair | cos_sim (mean) | structure (Pearson r) |
|---|---:|---:|
| production vs abl_baseline | 0.267 | 0.053 |
| production vs no_cohesion | 0.347 | 0.064 |
| production vs no_cell_noise | 0.292 | 0.036 |
| production vs fixed_temperature | 0.070 | 0.040 |
| abl_baseline vs no_cohesion | 0.778 | 0.549 |
| abl_baseline vs no_cell_noise | 0.884 | 0.756 |
| abl_baseline vs fixed_temperature | 0.646 | 0.738 |
| no_cohesion vs no_cell_noise | 0.821 | 0.553 |
| no_cohesion vs fixed_temperature | 0.262 | 0.365 |
| no_cell_noise vs fixed_temperature | 0.510 | 0.585 |

### Interpretation

1. **Contrastive training produces stochastic embedding geometries.**
   Two CLOP runs with identical config and seed but different
   initialisation-order dynamics learn different rotations of the
   embedding space. This is a known property of contrastive methods
   (the loss is invariant to global rotations of the shared space).

2. **The ablation suite's internal rankings are valid.** Within the
   ablation runs (which share the same data split and init trajectory),
   the structural correlations are 0.36–0.76 — moderate but
   meaningfully different from zero. The hyperparameter changes
   (no_cohesion, no_cell_noise, fixed_temperature) DO change the
   embedding geometry, and these changes WOULD propagate to DiT.

3. **Production → ablation transfer cannot be assumed.** The near-zero
   structural correlation means the production DiT's conditioning
   space is geometrically unrelated to the ablation projected texts.
   Ablation conclusions about _which hyperparameter matters_ are
   internally valid but the magnitude of effect in the full production
   pipeline is unknown without DiT retraining.

4. **Full DiT retraining IS needed for B4.** There is no shortcut:
   the projected text spaces are structurally different, so a DiT
   trained on one variant's projections cannot be evaluated with
   another's.

### Implications for the revision

- **The ablation study is valid as a Stage-1 component analysis.** The
  relative rankings (no_cohesion ≈ no_cell_noise >> fixed_temperature
  in proto_acc) are internally consistent.
- **The ablation study should not be over-claimed as end-to-end.** We
  should acknowledge the contrastive-stochasticity caveat and scope
  the ablation as "Stage-1 sensitivity analysis" rather than "full
  pipeline ablation".
- **Phase 2 (DiT retraining) is gated on GPU budget.** 3 variants ×
  300 epochs × ~2 hr each = ~6 hr GPU. This will produce the
  end-to-end bridge numbers the reviewer requests.

### Artefacts

- Diagnostic JSON: `projected_text_diagnostic.json`
- Preview: `projected_text_diagnostic_preview.txt`
- Script: `diagnose_projected_text.py`
