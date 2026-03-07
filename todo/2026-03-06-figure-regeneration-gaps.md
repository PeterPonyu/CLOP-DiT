# Figure Regeneration Gaps After Commit `a078341`

Date: 2026-03-06
Status: **RESOLVED** (2026-03-07, commit `190da0e`)

## Resolution

All gaps identified below were fixed in commit `190da0e`:

1. **Figure 2 (`fig_training_dynamics.pdf`)**: Regenerated (2026-03-07 14:12).
   Training histories found in `models/checkpoints/{clop,dit}_history.json`.

2. **Figure 3 (`fig_embedding_space.pdf`)**: Regenerated (2026-03-07 14:12).
   Array dimension mismatch fixed: use `text_group_ids.npy` (220k entries)
   instead of `text_group_ids_dedup.npy` (167k) when indexing `projected_text.npy`.

3. All 15 article figures regenerated, symlinked, and PDF recompiled (21 pages).

## Original Issue (for reference)

- Figures 2 and 3 were stale after commit `a078341` due to missing cache artifacts
- Figure 2 needed training history JSONs (found in models/checkpoints/)
- Figure 3 had array dimension mismatch (220k text vs 167k dedup indices)
