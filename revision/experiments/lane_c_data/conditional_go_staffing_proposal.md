# Lane C — Conditional-Go Staffing Proposal

**Author:** leader follow-up after the machine-checkable staffing gate scaffolding  
**Ran on:** 2026-04-16  
**Status:** draft, not approved

This note turns the machine-readable
`conditional_go_staffing_plan.json` into a human-readable staffing
proposal for the narrowed **Priority-2-only strict-OOD** Lane C path.

## Current posture

Lane C is still **closed** for this revision round.

This file does **not** reopen the lane. It only makes the current draft
assumptions legible:

- the total draft staffing cap stays at `<= 18 engineer-days`
- all owner slots currently collapse to a **single-author path**
- approval is still blocked by empty manifest / label-bridge inputs and
  by the lack of a real accepted staffing decision

## Draft line-item estimate

| Work item | Days | Notes |
|---|---:|---|
| strict-OOD dataset ingest and manifest updates | 6 | curate held-out tissue datasets into `data/processed_h5ad_revision/` |
| label-bridge review and reconciliation | 4 | map external labels onto the 69-type vocabulary or a novel bucket |
| strict-OOD leakage checking and launch validation | 3 | run / iterate `check_strict_ood.py` before training |
| CLOP + DiT retrain execution oversight | 3 | one CLOP + one DiT run, no broad variant sweep |
| five-slice evaluation and reporting integration | 2 | baseline diff + reviewer-facing summary |
| **Total** | **18** | draft ceiling only; not yet approved |

## Provisional owner slots

These are the current draft owner identifiers already reflected in
`conditional_go_staffing_plan.json`:

| Gate / responsibility | Draft owner slot |
|---|---|
| B-2 strict-OOD leakage | `leader_single_author__strict_ood_gate` |
| B-3 label bridge review | `leader_single_author__label_bridge_review` |
| CLOP retrain | `leader_single_author__clop_retrain` |
| DiT retrain | `leader_single_author__dit_retrain` |
| Five-slice reporting | `leader_single_author__five_slice_reporting` |

## Why this still does not pass the gate

Even with these provisional owner slots, the gate remains closed because:

1. the plan status is still `draft_not_approved`
2. `data/processed_h5ad_revision/MANIFEST.csv` still has zero dataset rows
3. `data/processed_h5ad_revision/label_bridge.csv` still has zero review rows
4. no one has explicitly accepted the B-6 stop rule in an execution-ready artifact

## Minimal next move if we still want to pursue Reviewer 3.1

The smallest honest reopening step is:

1. add at least one real strict-OOD candidate row to `MANIFEST.csv`
2. add at least one reviewed label mapping row to `label_bridge.csv`
3. explicitly promote this staffing proposal from draft to approved
4. rerun:
   - `python scripts/check_strict_ood.py ...`
   - `python scripts/check_lane_c_staffing_gate.py --json`

Until those four happen, Lane C should continue to be described as
**planned, scoped, and technically scaffolded, but not approved to
start**.
