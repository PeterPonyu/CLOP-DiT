# A1 — KNN error taxonomy

**Reviewer comment:** R2.6 — "KNN only 36%/55%. Where do the other 2/3 errors go?"
**Retraining:** None.
**Input artifacts (frozen baseline):**
- `revision/prerevision_baseline/metrics_frozen/generation_metrics.json`
- `results/generated_embeddings.npy`, `results/generated_labels.npy` (read-only)

## Question

When the 1-NN classifier misassigns a generated cell, is the error
within a biologically related lineage (graceful) or across unrelated
lineages (catastrophic)?

## Deliverables

1. **Family-level confusion matrix** — 69 cell types grouped into 6–10
   biologically interpretable families (lymphoid, myeloid, epithelial,
   endothelial, stromal, progenitor/cycling, stress-response,
   erythroid, …). Report within-family vs cross-family error rates.
2. **Top-10 confused type pairs** — e.g., CD4 T vs CD8 T, macrophage
   vs monocyte, with per-pair error counts and direction.
3. **One-sentence verdict** for the rebuttal: "Of the 64% mis-KNN
   cells, X% were within the same immediate biological family."

## Plan

- Define the family taxonomy in `family_taxonomy.yaml` (commit with the
  analysis so it is reviewable).
- Compute KNN on the frozen generated embeddings against real cells.
- Emit `confusion_family.csv` + `confusion_top_pairs.csv` + a small
  heatmap PDF.

## Status

- [ ] Family taxonomy defined
- [ ] KNN run against frozen embeddings
- [ ] Family-level confusion computed
- [ ] Top-10 pair table computed
- [ ] Figure drafted
- [ ] Rebuttal verdict sentence committed

## Results

_(fill in after the analysis runs)_

## Rebuttal-ready sentence

_(2–3 sentences that can be pasted into the R2.6 response)_
