# Lane C — Strict-OOD Candidate Inventory

This inventory is a **baseline-overlap audit**, not a curated execution manifest.
It helps decide which target tissues are still plausible first-wave strict-OOD candidates.

| Target tissue | Candidate source | Candidate label | Baseline overlap count | Recommendation |
|---|---|---|---:|---|
| kidney | MULTI_TISSUE_PILOT + feasibility note | GSE131685 / GSE140989 / HCA kidney v2 | 0 | preferred_zero_overlap_candidate |
| testis | feasibility note | GTEx testis snRNA-seq | 0 | preferred_zero_overlap_candidate |
| intestine | feasibility note | HCA gut v2 | 5 | manual_overlap_audit_required |
| cerebellum | feasibility note | Allen cerebellum snRNA-seq | 0 | preferred_zero_overlap_candidate |
| distal airway | feasibility note | HCA lung upper airway | 2 | manual_overlap_audit_required |
| merkel-like | feasibility note | NCBI GEO Merkel-cell carcinoma scRNA-seq | 2 | manual_overlap_audit_required |

## Interpretation

- `preferred_zero_overlap_candidate` means the current `metadata_structured.json` text audit found no obvious baseline mention of that tissue string family.
- `manual_overlap_audit_required` means at least one current baseline dataset text already matches the target alias family, so the tissue is risky as a first strict-OOD choice unless the scope is narrowed more carefully.

## Baseline overlap examples

- **kidney**: none
- **testis**: none
- **intestine**: GSE225857_liverColonMetasisHmCancer, GSE307774_scRNA25100, GSE236565_scRNA25100, GSE280847_scRNA25100, GSE279781_geodh
- **cerebellum**: none
- **distal airway**: lung, GSE130148_LungHmDev
- **merkel-like**: GSE117988_MCCPBMCCancer, GSE117988_MCCTumorCancer
