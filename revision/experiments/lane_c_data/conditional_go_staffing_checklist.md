# Lane C — Conditional-Go Staffing Checklist

Use this checklist only for the narrowed **Priority-2-only strict-OOD**
Lane C path defined in `go_no_go_decision.md`.

Current assessment: see `conditional_go_staffing_assessment.md`.
Machine-checkable gate inputs:

- `conditional_go_staffing_plan.json`
- `scripts/check_lane_c_staffing_gate.py`

The staffing plan currently contains **provisional single-author owner
slots** only. Checked items below should therefore be interpreted as
"role slots exist in the draft plan", not "the conditional-go path is
approved".

## Decision gate

Lane C may start only if every item below is marked complete.

## Staffing and ownership

- [ ] Total curation estimate reconfirmed at `<= 18 engineer-days`
- [x] B-2 owner assigned: strict-OOD leakage checker
- [x] B-3 owner assigned: label-vocabulary bridge review
- [x] CLOP retrain owner assigned
- [x] DiT retrain owner assigned
- [x] Five-slice evaluation/reporting owner assigned

## Scope control

- [x] Scope explicitly limited to **Priority-2 strict-OOD tissues only**
- [x] No generic Lane-C expansion beyond Priority 2 is included
- [x] Five-slice reporting contract preserved unchanged
- [ ] B-6 stop-rule acknowledged by all assigned owners

## Launch readiness

- [x] `data/processed_h5ad_revision/` ingest plan defined
- [x] proposal-level strict-OOD candidate rows committed in `MANIFEST.csv`
- [x] proposal-level label-bridge rows committed in `label_bridge.csv`
- [ ] `scripts/check_strict_ood.py` implementation owner named
- [x] `label_bridge.csv` review workflow defined
- [x] Baseline comparison source pinned to frozen prerevision metrics

## Outcome

- [ ] CONDITIONAL GO approved
- [x] NO-GO retained; use limitations / rebuttal fallback instead

Exactly one of the last two boxes should be checked.
