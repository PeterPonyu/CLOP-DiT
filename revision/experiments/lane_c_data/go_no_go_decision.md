# Lane C — Go / No-Go and Stop-Rule Decision

**Author:** leader follow-up after validated Lane C feasibility planning
**Ran on:** 2026-04-16
**Decision scope:** Convert the B-7 engineer-day gate and the B-6
regression stop-rule into an executable revision decision.

## Decision summary

1. **No-go for the full Lane C plan in this revision round.**
   The full plan remains priced at ~28-34 engineer-days and ~16 GPU-hours,
   which is too expensive relative to the remaining reviewer delta.
2. **Conditional go for a narrowed Priority-2-only Lane C run.**
   We should proceed only if the team can staff a strict-OOD-only lane
   at `<= 18 engineer-days` with explicit owners for B-2 and B-3.
3. **Default fallback if the conditional go fails:**
   do not start new-data curation or retraining; instead, ship the
   limitations/rebuttal path backed by A2, A3, B3-forced-scarcity, and B4.

## Why this is the right decision

- **R3.1 strict OOD** is the only reviewer question that still requires
  new data. This justifies a narrow Lane C, not the original full lane.
- **R3.2 cross-species** is already answered at the type level by A2 and
  only needs confirmatory strengthening, not an exploratory corpus build.
- **R2.7 low abundance** is already discharged by A3 + B3-forced-scarcity
  and should not keep driving Lane C scope.

## Conditional-go requirements

All of the following must be true before any curation starts:

1. The narrowed plan is explicitly limited to **Priority 2 strict-OOD
   tissues**.
2. The curation budget is re-estimated at `<= 18 engineer-days`.
3. An owner is assigned for **B-2 strict-OOD leakage checking**.
4. An owner is assigned for **B-3 label-vocabulary bridge review**.
5. The five-slice reporting contract remains unchanged.

If any of these fail, Lane C remains **no-go** for this revision round.

## B-6 regression stop-rule (committed)

If the narrowed Lane C run is approved and executed, it must be rejected
immediately if either of the following is observed against the frozen
baseline:

- overall centroid cosine drops by more than `0.01` (`> 0.01`)
- Frechet distance rises by more than `0.05` (`> 0.05`)

On stop-rule trigger:

1. Abort further Lane C scaling work for this revision round.
2. Draft a Limitations / rebuttal paragraph instead of accepting the
   regression.
3. Report the five slices honestly, including the failed overall guardrail.

## Next executable step

Do not start curation or retraining yet.

The next real execution task is **not** full Lane C startup. It is:

1. confirm whether the narrowed Priority-2-only plan can be staffed at
   `<= 18 engineer-days`
2. assign owners for B-2 and B-3
3. only then decide whether to open `data/processed_h5ad_revision/`

Until those conditions are met, Lane C remains planned but not started.
