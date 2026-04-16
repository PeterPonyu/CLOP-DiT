# Lane C — Conditional-Go Staffing Assessment

**Author:** leader follow-up after the committed B-7 / B-6 decision  
**Ran on:** 2026-04-16  
**Decision scope:** Decide whether the narrowed **Priority-2-only strict-OOD**
Lane C path can actually open in the current revision window.

## Outcome

**Conditional go is not approved. NO-GO is retained for this revision round.**

The narrowed Lane-C path remains *conceptually available*, but the
required staffing/readiness gate is not met in the current repository
state, so no new-data curation or retraining should start yet.

## Why the gate fails right now

The committed decision in `go_no_go_decision.md` says the narrowed run
may start only if all of the following are true:

1. the plan is explicitly limited to **Priority-2 strict-OOD tissues**
2. the curation budget is reconfirmed at `<= 18 engineer-days`
3. a named owner exists for **B-2 strict-OOD leakage checking**
4. a named owner exists for **B-3 label-vocabulary bridge review**
5. the five-slice reporting contract remains unchanged

Current repo evidence shows that only the scope-control requirement is
clearly satisfied. The staffing/readiness requirements are still unmet.

## Evidence snapshot

### Conditions that are satisfied

- The narrowed plan is already scoped to **Priority-2 strict-OOD only**
  in `go_no_go_decision.md`.
- The five-slice reporting contract is still preserved in
  `revision/experiments/lane_c_data/README.md` and `results.md`.
- The B-6 stop rule is documented and committed.
- A baseline-overlap candidate inventory now exists and suggests that
  **kidney, testis, and cerebellum** are the cleanest first-wave
  strict-OOD candidates, while intestine / distal-airway / merkel-like
  options still need extra overlap review.
- The revision-side manifest now contains three **proposal-level held-out
  rows** for kidney, testis, and cerebellum, and the strict-OOD checker
  passes on that shortlist with zero leakage.

### Conditions that are still unmet

- No committed reconfirmation of a real `<= 18 engineer-days` staffing
  budget exists beyond the planning-side conditional estimate.
- The staffing plan now contains **provisional single-author owner slots**
  for B-2, B-3, CLOP, DiT, and five-slice reporting, but these are
  still draft placeholders rather than an approved staffing decision.
- Revision-side technical scaffolding now exists, but remains only
  **scaffolding**:
  - `data/processed_h5ad_revision/` now contains proposal rows, not a curated execution pack
  - `scripts/check_strict_ood.py` now passes for the proposal shortlist, but this is still a proposal audit rather than a launched run
  - `label_bridge.csv` now contains provisional single-review rows (`reviewed` / `novel`), but not a fully approved bridge
  - `conditional_go_staffing_plan.json` and `scripts/check_lane_c_staffing_gate.py`
    now exist, but they still fail until the plan is approved

## Decision matrix

| Gate item | Current state | Evidence |
|---|---|---|
| Priority-2-only scope | PASS | `go_no_go_decision.md` |
| `<= 18 engineer-days` staffing reconfirmed | FAIL | draft estimate only; no approved staffing decision |
| B-2 owner assigned | PARTIAL | provisional single-author slot exists in staffing plan, but not yet approved |
| B-3 owner assigned | PARTIAL | provisional single-author slot exists in staffing plan, but not yet approved |
| Five-slice contract preserved | PASS | `README.md`, `results.md` |
| Strict-OOD shortlist leak check | PASS | proposal manifest rows for kidney/testis/cerebellum pass `check_strict_ood.py` with zero leakage |
| Label-bridge proposal review | PARTIAL | one exact reviewed mapping plus two novel-bucket rows now exist, but only as provisional single-review entries |
| Ingest/readiness scaffolding present | PARTIAL | proposal rows and gate scaffolding now exist, but no approved/populated revision-side execution pack is committed |

## Practical implication

Lane C remains **planned but not started**.

The honest next move is still the fallback described in
`go_no_go_decision.md`:

- keep the full lane closed
- keep the narrowed lane closed until staffing/readiness artifacts exist
- continue to lean on A2, A3, B3-forced-scarcity, and B4 in the current
  rebuttal/limitations path

## What would reopen the conditional-go path

The narrowed Lane-C run should be reconsidered only after all of the
following are committed:

1. a named-owner staffing note for B-2, B-3, CLOP, DiT, and five-slice reporting
2. a concrete `<= 18 engineer-days` estimate tied to those owners
3. a revision-side ingest plan under `data/processed_h5ad_revision/`
4. a committed `check_strict_ood.py` owner/implementation path
5. a committed label-bridge review workflow
