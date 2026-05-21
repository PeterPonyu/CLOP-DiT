# Lane C — Data expansion (planned, not started)

**Reviewer comments:** R3.1 (strict OOD), R3.2 (species stratification).
R2.7 is already discharged by A3 + B3-forced-scarcity and is retained
here only as a reporting slice, not as an active Lane C science target.
**Retraining:** Full pipeline.
**Current gates:** A3 is already complete; before any curation or retrain
work starts, Lane C now requires the B-7 engineer-day cap sign-off and
the B-6 regression stop-rule decision documented in
`go_no_go_decision.md`. The older budget estimates in
`feasibility_and_blockers.md` are now historical inputs, not the active
gate.

See `revision/experiments/lane_c_data/README.md` for priority-ordered
data additions and the five-slice evaluation contract.

## Question

Can targeted data expansion (mouse-heavy, strict-OOD tissues, rare /
transitional states) answer the still-open R3.1 / R3.2 questions while
preserving the five reviewer-facing reporting slices — overall,
low-abundance, mouse, strict-OOD, rare-classification — without
regressing overall metrics?

## Status

- [x] A3 gate cleared
- [x] Feasibility + blocker-to-action plan committed
      (`feasibility_and_blockers.md`)
- [x] B-7 decision committed
      (`go_no_go_decision.md`): no-go for full Lane C; conditional go
      only for Priority-2 strict-OOD at `<= 18 engineer-days`
- [ ] Priority-1 mouse datasets curated (confirmatory; A2 already
      answers R3.2 at type level)
- [ ] Priority-2 OOD tissue datasets curated + held out
      (unavoidable for R3.1)
- [ ] Priority-3 heterogeneity-targeted (bottom-quartile
      `real_intra_cos`) datasets curated — redefined per A3
- [ ] Strict-OOD leakage check script committed (B-2)
- [ ] Label-vocabulary bridge reviewed (B-3)
- [x] Regression stop-rule decision committed (B-6)
- [x] Conditional-go staffing checklist committed
      (`conditional_go_staffing_checklist.md`)
- [x] Conditional-go staffing assessment committed
      (`conditional_go_staffing_assessment.md`)
- [ ] Conditional-go staffing gate passed (`<= 18 engineer-days`)
- [x] NO-GO retained for the current revision window
- [ ] Full CLOP + DiT retrain complete
- [ ] Five-slice metric comparison committed
- [x] Rebuttal / limitations fallback paragraph drafted
      (`limitations_fallback_paragraph.md`)

## Results

No new-data run has been started in this branch yet.

Current gating read:

1. The original A3 dependency is resolved.
2. A3 also showed that abundance alone is not the dominant predictor of
   failure, which lowers the expected value of generic data-scaling.
3. Even so, Lane C remains the only remaining revision lane that can
   directly answer strict-OOD generalisation (R3.1) and upgrade the
   current type-level cross-species answer into a stronger cell-level
   / held-out-tissue evaluation.
4. The full Lane C plan is now a documented **no-go** for this revision
   round; only a narrowed Priority-2-only strict-OOD run remains
   conditionally open.
5. The practical blocker is therefore no longer ambiguity about scope.
   The staffing assessment now records a concrete **failed conditional-go
   gate**: no approved `<= 18 engineer-days` staffing decision and no
   populated/owned revision-side execution pack, even though provisional
   owner slots plus the ingest directory, leakage-check script,
   staffing-plan template, staffing-gate validator, and proposal-level
   manifest / label-bridge rows now exist.
6. Because that gate failed, the repo now carries an explicit
   limitations/rebuttal fallback paragraph rather than pretending that a
   strict-OOD run is still merely "about to start".
7. A baseline-overlap candidate inventory now narrows the cleanest
   first-wave strict-OOD options to **kidney, testis, and cerebellum**;
   intestine, distal-airway, and merkel-like proposals still require
   extra overlap audit before they could be used honestly.
8. The technical strict-OOD gate for that shortlist now passes at the
   **proposal** level (`MANIFEST.csv` contains held-out rows and
   `check_strict_ood.py` reports zero leakage), but the staffing gate
   still fails because the plan remains draft even though the label
   bridge now has provisional reviewed/novel rows.

## Rebuttal-ready sentence

Pending new-data execution. If this lane is started, the rebuttal must
explicitly report which of the five required slices improved, stayed
flat, or regressed relative to the frozen baseline.
