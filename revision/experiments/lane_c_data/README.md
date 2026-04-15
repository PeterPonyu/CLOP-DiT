# Lane C — Targeted data expansion

**Gate:** Do not start this lane until Lane A's abundance-fidelity analysis
(A3) has finished and quantified the long-tail gap. If A3 shows the
long-tail gap is not the dominant failure mode, de-prioritize this lane
in favour of Lane D.

## Unique value proposition

This is the only lane that can address **3.1 (strict OOD)** and
meaningfully strengthen the evidence behind **3.2 (cross-species)**. All
other reviewer comments are better addressed by Lane A or B.

## Planned data additions (priority-ordered)

1. **Mouse-heavy datasets** — rebalance the 59 human / 21 mouse split.
   Directly strengthens the answer to R3.2.
2. **Strict-OOD tissues** — kidney, testis, intestine, cerebellum,
   distal airway, Merkel-like. Must be **held out entirely** from
   training so the evaluation is extrapolation, not interpolation.
3. **Rare / transitional states** — cycling, stress-response,
   progenitor. Targeted to improve A3-identified long-tail gaps.

## Evaluation discipline

Every new-data run must report metrics on **five slices**, not just
overall, or the contribution gets diluted:

- overall
- low-abundance subset (bottom quartile by training count)
- mouse-only subset
- strict-OOD hold-out subset
- rare-cell classification subset (tie-in to B3)

## Ground rules

- New raw data goes under `data/processed_h5ad_revision/` (create at
  first ingest) to preserve the baseline `data/processed_h5ad/` integrity.
- Retraining is a full-pipeline operation: new CLOP + new DiT + fresh
  baseline comparison.
- The `results.md` in this lane must include an explicit statement of
  which subsets improved, stayed flat, or regressed relative to the
  frozen baseline.
