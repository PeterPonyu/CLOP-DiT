# Lane C — Feasibility, data plan, and blocker-to-action synthesis

**Author:** worker-1 (team clop-dit-revision-next-step-ex)
**Ran on:** 2026-04-15
**Upstream inputs (read-only):**
- `revision/experiments/lane_c_data/README.md` (scope + evaluation contract)
- `revision/experiments/lane_c_data/results.md` (status + gating read)
- `revision/experiments/lane_a_analysis/a3_abundance_fidelity/results.md`
- `revision/experiments/lane_a_analysis/a2_organism_split/results.md`
- `revision/experiments/lane_b_retrain/b3_mixing_sweep/` (forced-scarcity)
- `revision/experiments/b4_clop_bridge/b4_bridge_comparison.json`
- `data/processed_h5ad/` (current 80-dataset baseline corpus)

This note turns the gated Lane-C stub into an executable plan. It
keeps every choice traceable to an upstream finding so a reviewer can
audit *why* each data addition is in scope and *why* each blocker
decomposes the way it does.

## 1. What Lane C must still answer

After Lanes A and B, the reviewer-facing gaps that only Lane C can
close are strictly narrower than the original Lane-C README implied:

| Reviewer comment | Residual risk after A+B | Lane-C commitment |
|---|---|---|
| R3.1 strict OOD | **Fully open.** No held-out tissue evaluation exists in A/B. | Required. |
| R3.2 cross-species | Type-level answered by A2 (no species gap, p = 0.14–0.87). Cell-level held-out mouse still pending. | Required, but scope is *confirmatory*, not exploratory — A2 tells us we expect no species gap. |
| R2.7 low-abundance | A3 shows abundance is **not** the dominant axis; `real_intra_cos` is. B3-forced-scarcity shows augmentation lifts rare-F1 by +0.11–0.36 under genuine headroom. | **De-scoped** from Lane C. Generic data-scaling for abundance alone is no longer the best lever; the heterogeneity lever is. |

**Implication for priorities:** the README's Priority-1 (mouse-heavy)
becomes confirmatory rather than exploratory; Priority-2 (strict-OOD)
is the only unavoidable new-data commitment; Priority-3 (rare /
transitional) is rewritten around *heterogeneous* cell types, not
merely *rare-by-count* ones.

## 2. Feasibility read on the three priorities

### Priority 1 — Mouse-heavy expansion (confirmatory)

- **Size needed.** A2 has 8 mouse-only types out of 69. To bring
  mouse-only representation to ≥ 20 % at the cell level (vs the
  current ~26 % of datasets but unknown cell fraction), we need
  roughly 3–4 additional mouse-only datasets covering 10–15 new
  mouse-only or mouse-dominant cell types.
- **Candidate sources (public, CC0/CC-BY):** Tabula Muris Senis
  (mouse multi-tissue), Mouse Cell Atlas, and existing `GSE145929_*`
  mouse datasets already in `data/processed_h5ad/` (not yet used as
  mouse-primary). No new licences required.
- **Curation cost.** ~2 engineer-days per dataset if we keep the
  existing h5ad schema (gene symbol, `cell_type` label, `organism`
  column). Expect 6–8 engineer-days total.
- **Retraining cost.** Folded into the single Lane-C retrain below —
  no extra GPU budget if scheduled with Priorities 2 and 3.

### Priority 2 — Strict-OOD tissues (unavoidable)

- **Target tissues.** Kidney, testis, intestine, cerebellum, distal
  airway, Merkel-like — per README. Each must be **excluded entirely**
  from the training corpus (both CLOP and DiT) so the evaluation is
  pure extrapolation.
- **Operational definition of "held out".** A tissue is strictly
  held-out if (i) no cell with `tissue == T` appears in the training
  corpus, (ii) no cell type is labelled uniquely with `T` as its
  primary tissue in captions, and (iii) evaluation uses only `T`'s
  test partition. A leakage-check script (`check_strict_ood.py`) must
  emit PASS/FAIL for each tissue before training starts.
- **Size needed.** 1–2 datasets per tissue, ≥ 2 000 cells each, with
  cell-type labels that resolve to the existing 69-type vocabulary or
  an explicit "novel" bucket.
- **Candidate sources.** HCA kidney v2, GTEx testis snRNA-seq, HCA
  gut v2 (intestine), Allen cerebellum snRNA-seq, HCA lung upper
  airway, a Merkel-cell carcinoma scRNA-seq study from NCBI GEO.
- **Curation cost.** ~3 engineer-days per tissue (heavier than
  Priority 1 because label vocabulary reconciliation is harder).
  Expect 18 engineer-days total.
- **Retraining cost.** Same single retrain as Priority 1.

### Priority 3 — Heterogeneity-targeted states (redefined)

- **Redefinition.** A3 demoted "rare by training count" as a failure
  axis. Replace Priority 3 with cell types in the bottom quartile of
  `real_intra_cos` (cycling, stress-response, progenitor, transitional
  epithelial). These are the types A3 identifies as the true
  fidelity bottleneck.
- **Size needed.** 3–5 datasets adding ≥ 500 cells each for the
  targeted heterogeneous types already in the 69-type vocabulary.
- **Curation cost.** ~1.5 engineer-days per dataset (labels align
  with existing vocabulary). Expect 4.5–7.5 engineer-days total.

### Aggregate feasibility budget

| Item | Estimate |
|---|---|
| Data curation engineering | 28–34 engineer-days |
| CLOP retrain (single run, extended corpus) | ~10 GPU-hours (RTX 5090, scaled from current baseline) |
| DiT retrain (single run, 300 epochs) | ~2 GPU-hours (ref. B4 phase-2 timing) |
| Five-slice evaluation sweep | ~4 GPU-hours |
| Total GPU | ~16 GPU-hours for the Lane-C run itself |
| Elapsed time | ~3 weeks wall-clock if curation is single-threaded; ~1.5 weeks with two curators in parallel |

This budget is the primary input to the go / no-go decision; it is
modest on GPU but non-trivial on engineer-days.

## 3. Blocker-to-action synthesis

`results.md` lists the practical blocker as "cost of fresh data
curation plus a full CLOP + DiT retrain". Decomposed:

| # | Blocker | Action | Owner axis | Exit criterion |
|---|---|---|---|---|
| B-1 | Raw data curation labour | Adopt a **dataset-ingest manifest** with one YAML per dataset (source URL, licence, organism, tissue, expected cell-type vocabulary). Curator fills YAML; an ingest script converts to h5ad, validates schema, emits a row in `data/processed_h5ad_revision/MANIFEST.csv`. | engineering | ≥ 6 manifests merged; manifest validator CI green. |
| B-2 | Strict-OOD leakage discipline | Write `scripts/check_strict_ood.py` that takes the held-out tissue list and the training-corpus index, emits PASS/FAIL per tissue; wire into the CLOP+DiT training entrypoints as a precondition. | engineering | Script + unit test committed; training refuses to start on FAIL. |
| B-3 | Label-vocabulary reconciliation across datasets | Publish a **vocabulary bridge** (`data/processed_h5ad_revision/label_bridge.csv`) that maps each dataset's labels to the existing 69-type vocabulary or to an explicit "novel" tag. Two reviewers must sign off per row. | domain-expert | Bridge covers 100 % of Priority-2 cells; spot-audit ≥ 30 rows. |
| B-4 | Full CLOP + DiT retrain cost | Piggy-back on the B4 retrain scaffolding (`revision/experiments/b4_clop_bridge/`). Single run, not per-variant, using the extended corpus. | compute | One CLOP checkpoint + one DiT checkpoint on `processed_h5ad_revision`. |
| B-5 | Five-slice metric discipline (README contract) | Factor the A2 / A3 reporting code into a shared `revision/experiments/lane_c_data/compute_five_slice.py`; run once on the baseline and once on the Lane-C model; commit a diff table. | engineering | Diff table committed with overall + 4 slices, no silent overall-only reporting. |
| B-6 | Regression risk on overall metrics | **Stop rule:** if overall centroid cosine drops > 0.01 or FD rises > 0.05 vs frozen baseline, Lane C is rejected and a Limitations paragraph is drafted instead. | reviewer-defence | Stop-rule decision committed before retrain; no silent acceptance of regressions. |
| B-7 | Go / no-go gating | Historical feasibility estimate only. This row is **superseded** by `go_no_go_decision.md`: full Lane C is now no-go for this revision round, and only a narrowed Priority-2-only run may proceed if it can be staffed at `<= 18 engineer-days` with B-2/B-3 owners assigned. | PM | See `go_no_go_decision.md` and `conditional_go_staffing_checklist.md` for the active gate. |

**B-6 and B-7 are the most important.** Without them the lane has a
failure mode where partial new data regresses overall metrics while
only marginally improving the five slices — which would be the worst
possible reviewer outcome.

## 4. Recommended decision for this revision round

A2, A3 and B3-forced-scarcity together already discharge R2.7 and the
type-level part of R3.2. R3.1 (strict OOD) is the only reviewer
comment that **requires** Lane C. If the engineer-day budget for
curation (B-1 + B-3) cannot be secured inside the revision window:

- Narrow Lane C to **Priority 2 only** (strict-OOD tissues).
- Keep the five-slice contract unchanged; for slices not newly improved
  by the narrowed run, report the frozen-baseline values and explicitly
  cross-reference A2 / A3 for the existing mouse-only and low-abundance
  interpretation.
- Retain B-6 stop rule unchanged.

This narrowed plan costs ~18 engineer-days and ~8 GPU-hours and
answers R3.1 cleanly, while R3.2 and R2.7 rely on the already-committed
A2 / A3 / B3-forced-scarcity evidence.

If even the narrowed plan is infeasible in the revision window, the
honest outcome is a Limitations paragraph citing A2 (no type-level
species gap), A3 (heterogeneity, not abundance, is the dominant axis)
and B4 (Stage-1 rankings do not reliably transfer) as the reasons we
did not attempt strict-OOD retraining in this round, and committing to
it for the next release.

## 5. Status transitions this note implies

In `revision/experiments/lane_c_data/results.md` the checklist should
be updated (separate commit) to reflect:

- Priority-3 redefined around `real_intra_cos` bottom quartile.
- Gate B-7 (engineer-day cap) added before "curated" items.
- Stop rule B-6 added before "Full CLOP + DiT retrain complete".
- R2.7 moved out of Lane-C scope (discharged by A3 + B3-forced).

No code or model artifacts are produced by this note; it is pure
planning output and is consumed by the next Lane-C execution task.
