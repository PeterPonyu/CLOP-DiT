# SUBMISSION_READINESS — CLOP-DiT major revision

**Date:** 2026-04-16
**Branch:** `revision/major` (30+ commits ahead of `origin/revision/major` at time of writing)
**Baseline tag:** `pre-revision-2026-04-15`

This document is the pre-submission checkpoint. It enumerates every
reviewer comment with a one-line status + pointer to evidence, and
states whether we ship this revision round or hold.

## Comment-by-comment status

| # | Topic | Status | Evidence |
|---|---|---|---|
| **R2.1** | Stage-1 CLOP success definition | Text revision only | `reviewer_response_draft.md` §R2.1 · `rebuttal_letter.tex` §R2.1 |
| **R2.2** | Five-field template justification | Text revision only — existing per-field ablation sufficient once "optimal" is reworded as "sufficient" | `reviewer_response_draft.md` §R2.2 |
| **R2.3** | Hyperparameter table | Text revision only | `reviewer_response_draft.md` §R2.3 |
| **R2.4** | Figure readability | Text/figure revision — Fig 1a/1b fonts bumped, spacing widened (commit `ff3309b`) | `scripts/analysis/{generate_architecture_figure,evaluation_pipeline_figure}.py` · `src/visualization/style.py` |
| **R2.5** | Training details | Text revision only | `reviewer_response_draft.md` §R2.5 |
| **R2.6** | KNN error structure | **Additional analysis completed (Lane A1)** — 49.2% of KNN mis-typings are within-family; 9/15 top-confused pairs within-family | `REVISION_LOG.md` §2 · `revision/experiments/lane_a_analysis/a1_knn_confusion/` |
| **R2.7** | Low-abundance fidelity | **Additional analysis completed (Lane A3)** — abundance ρ=+0.57, heterogeneity ρ=−0.77 (p=3×10⁻¹³); "biologically difficult" retired in favour of "broad intrinsic heterogeneity" | `REVISION_LOG.md` §2 · `revision/experiments/lane_a_analysis/a3_abundance_fidelity/` |
| **R2.8** | Variance deficit r_var | Text revision only — A4 quantified all three slices (pooled 0.988, within-type 0.85, cross-dataset 0.01) for clarification | `REVISION_LOG.md` §2 · `revision/experiments/lane_a_analysis/a4_hvg_variance/` |
| **R2.9** | Gaussian benchmark | **Additional analysis completed (Lane B1)** — CLOP-DiT r_var=+0.201 sits strictly between pooled-Gaussian floor (+0.014) and type-aware oracle (+0.813) | `REVISION_LOG.md` §5 · `revision/experiments/lane_b_retrain/b1_gaussian_baseline/` |
| **R2.10** | CLOP ablation study | **Additional experiment completed (Lane B2 + Lane B4)** — B2 ZCA ablation shows 1.0% quality drop; B4 bridge: no_cohesion Stage-1 +10.2pp but Stage-2 FD +31%, coverage −45% — PARTIAL_REVERSAL verdict | `REVISION_LOG.md` §9, §11 · `revision/experiments/{zca_ablation,b4_clop_bridge}/` |
| **R2.11** | Rare-cell mechanism | **Additional analysis completed (Lane A5 + encoder trio)** — 13/18 rare types show upstream latent compression (30–66%) with expansive decoder (47–190%) | `REVISION_LOG.md` §2, §8 · `revision/experiments/{lane_a_analysis/a5_rare_mechanism,encoder_bottleneck,cap_increase}/` |
| **Gaussian addendum (R2)** | Diffusion vs Gaussian benchmark | Text revision only | `reviewer_response_draft.md` §Additional Reviewer #2 Concern |
| **R3.1** | Strict OOD experiment | **BLOCKED — limitations fallback delivered this round.** Technical gate passes (kidney/testis/cerebellum); staffing gate fails (no approved ≤18 engineer-days budget). Paste-ready limitations paragraph cites A2/A3/B3-forced/B4 | `REVISION_LOG.md` §13–19 · `revision/experiments/lane_c_data/limitations_fallback_paragraph.md` · `scripts/check_{strict_ood,lane_c_staffing_gate}.py` |
| **R3.2** | Cross-species | **Additional analysis completed (Lane A2)** — no statistically significant human-only vs mouse-only gap on any headline metric (all p > 0.14) | `REVISION_LOG.md` §2 · `revision/experiments/lane_a_analysis/a2_organism_stratification/` |
| **R3.3** | Rare-cell mixing strategies | **Additional experiment completed (Lane B3 + forced-scarcity)** — natural scarcity is ceiling-driven; forced-scarcity positive control shows F1 0.50 → 0.78–0.87 with oversampling and hybrid CLOP-DiT at 10× | `REVISION_LOG.md` §12 · `revision/experiments/lane_b_retrain/b3_mixing_sweep/` |
| **R3.4** | ZCA preprocessing ablation | **Additional experiment completed (Lane B2)** — formal 3-way comparison (ZCA vs center+L2 vs raw); ZCA ~1% quality improvement | `REVISION_LOG.md` §9 · `revision/experiments/zca_ablation/` |

**Tally:** 15/16 concerns closed this round (9 with new experimental evidence: R2.6/2.7/2.9/2.10/2.11/3.2/3.3/3.4 + Lane D woven into R2.11; 6 with text revision supported by existing ablations or data: R2.1/2.2/2.3/2.4/2.5/2.8 + Gaussian addendum). 1/16 (R3.1) explicitly bounded as limitations with four independent supporting experiments (A2/A3/B3-forced/B4).

## Ship verdict

**Decision: SHIP THIS ROUND.**

Rationale:

1. **15/16 reviewer concerns** closed this round — 9 with new experimental evidence (Lane A1–A5, B1–B4, Lane D) and 6 with targeted text revision whose claims are backed by existing ablation evidence already in the manuscript (notably R2.2 and R2.8).
2. **R2.10** (CLOP ablation study, the most experimentally expensive reviewer ask after R3.1) delivered a two-pronged response: B2 formal ZCA ablation + B4 Stage-1→full-pipeline bridge. The B4 PARTIAL_REVERSAL verdict actually sharpens the manuscript's self-awareness — we can now make the narrower, defensible claim instead of over-generalising.
3. **R3.1** (strict OOD) is the only remaining gap. It is (a) explicitly bounded by a documented staffing-gate failure rather than silently omitted, (b) technically scaffolded (kidney/testis/cerebellum shortlist, leakage checker, staffing-gate validator, all committed), and (c) flanked by four independent experiments (A2, A3, B3-forced, B4) that reduce what strict-OOD can uniquely claim. The limitations fallback paragraph is paste-ready in both manuscript and response-letter forms.
4. **Infrastructure is clean:** zero uncommitted changes, rebuttal PDF builds (12 pages, 297KB), 52 Lane-C tests pass, all figure-generation scripts bumped for R2.4 readability. Baseline SHA-256 manifest still valid against frozen `pre-revision-2026-04-15` tag.

Holding another round to close R3.1 would require securing ≤18 engineer-days of curation plus 8 GPU-hours for retraining, and would likely not change the manuscript's scientific conclusions given the current evidence base. The honest move is to ship with the limitations paragraph and commit to R3.1 as future work.

## What-if-reviewer-pushes-back triage

### On R3.1 ("why didn't you just run the strict-OOD experiment?")

- Point to `revision/experiments/lane_c_data/go_no_go_decision.md` (costed plan, explicit no-go).
- Point to `revision/experiments/lane_c_data/conditional_go_staffing_assessment.md` (staffing gate failed on ≤18 engineer-day budget + missing owners).
- Point to `scripts/check_strict_ood.py` (technical gate *passes* for the kidney/testis/cerebellum shortlist — the question is staffing, not capability).
- Emphasise the four independent bracketing experiments (A2/A3/B3-forced/B4) and commit to R3.1 as an explicit follow-up release.

### On R2.10 ("the CLOP ablation results now contradict the Stage-2 results — what does that mean?")

- Lean into the partial-reversal finding: it is itself the interesting result. The cohesion loss, which hurt Stage-1 prototype accuracy, is essential for the inter-type geometric structure the DiT needs. This is a genuine mechanistic insight, not a failed experiment.
- Point to B4 bridge metrics table: `no_cohesion` FD=0.230 vs baseline 0.175 (+31%), coverage 0.079 vs 0.144 (−45%), centroid cosine 0.854 vs 0.929 (−8.1%).
- Frame as "Stage-1 ablation remains valid as a loss-landscape characterisation, but should not be over-interpreted as end-to-end predictive". This is stated explicitly in the revised Results §3.12.

### On R2.11 ("if the failure is upstream, why not fix the generator?")

- Lane D encoder comparison shows within-type compression is scGPT-specific (median ratio 0.14 vs PCA 0.90). The compression lives in the frozen encoder.
- A5 shows the decoder is already expansive (47–190% of real expression variance). So the bottleneck is genuinely at the encoder→generator interface.
- This motivates the future-work framing of "heterogeneity-preserving training objectives" rather than a quick-fix post-hoc augmentation strategy.

## Submission packet inventory

- Manuscript: `revision/manuscripts/v2_revision/clop_dit_genes.tex` (1325 lines, compiles clean, A1–A5 + B1–B4 + D all have either integrated prose or paste-ready `\iffalse` fragments at the revision patch block).
- Rebuttal letter: `revision/response_letter/rebuttal_letter.pdf` (12 pages, builds via `build.sh`).
- Cover letter: `revision/response_letter/revision_cover_letter.pdf` (3 pages).
- Response draft (long-form): `revision/reviewer_response_draft.md` (all 16 sections with explicit Status lines).
- Synthesis reference: `revision/REVIEWER_RESPONSE_SYNTHESIS.md` (11 paste-ready blocks, one per completed analysis).
- Canonical revision log: `revision/REVISION_LOG.md` (§§1–19; §0 summary table at top).

## Anchor commitments

- Before uploading: `git push origin revision/major` and verify `origin/revision/major` matches.
- Re-run `sha256sum -c revision/prerevision_baseline/artifact_hashes.txt | grep -v ': OK$' | head` — must return empty.
- Re-run `bash revision/response_letter/build.sh` on a clean checkout to confirm PDFs regenerate deterministically.
- Confirm the manuscript `.tex` used for upload matches the local gitignored copy (checksum + visual spot-check on Figures 1a/1b).
