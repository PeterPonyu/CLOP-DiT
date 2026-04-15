# Lane C — Conditional-Go Staffing Checklist

Use this checklist only for the narrowed **Priority-2-only strict-OOD**
Lane C path defined in `go_no_go_decision.md`.

## Decision gate

Lane C may start only if every item below is marked complete.

## Staffing and ownership

- [ ] Total curation estimate reconfirmed at `<= 18 engineer-days`
- [ ] B-2 owner assigned: strict-OOD leakage checker
- [ ] B-3 owner assigned: label-vocabulary bridge review
- [ ] CLOP retrain owner assigned
- [ ] DiT retrain owner assigned
- [ ] Five-slice evaluation/reporting owner assigned

## Scope control

- [ ] Scope explicitly limited to **Priority-2 strict-OOD tissues only**
- [ ] No generic Lane-C expansion beyond Priority 2 is included
- [ ] Five-slice reporting contract preserved unchanged
- [ ] B-6 stop-rule acknowledged by all assigned owners

## Launch readiness

- [ ] `data/processed_h5ad_revision/` ingest plan defined
- [ ] `scripts/check_strict_ood.py` implementation owner named
- [ ] `label_bridge.csv` review workflow defined
- [ ] Baseline comparison source pinned to frozen prerevision metrics

## Outcome

- [ ] CONDITIONAL GO approved
- [ ] NO-GO retained; use limitations / rebuttal fallback instead

Exactly one of the last two boxes should be checked.
