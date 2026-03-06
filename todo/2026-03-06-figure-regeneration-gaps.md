# Figure Regeneration Gaps After Commit `a078341`

Date: 2026-03-06
Commit checked: `a0783416f002a39b42574a64b884a99f1dafff6a`
Commit message: `feat(visualization): unify figure policy for article and bio-validation`

## Summary

The article PDF was recompiled after the latest visualization commit, but not all article figures were regenerated from the new code before compilation.

- Regenerated successfully from latest code:
  - Figure 1: `results/figures/fig_architecture.pdf`
  - Figure 4: `results/figures/panel_d_metrics_summary.pdf`
- Still stale relative to commit `a078341`:
  - Figure 2: `results/figures/fig_training_dynamics.pdf`
  - Figure 3: `results/figures/fig_embedding_space.pdf`

## Failure Records

### 1. Figure 2 regeneration failed

- Target figure: `fig_training_dynamics.pdf`
- Producer: `src/visualization/panels_training.py`
- Commit touched producer: yes
- Existing figure timestamp: `2026-03-06 19:37:23 +0800`
- Commit timestamp: `2026-03-06 20:16:37 +0800`
- Failure mode: figure was generated before the commit and could not be regenerated during verification.

#### Missing information / missing artifacts

- `results/clop_training_history.json` is missing
- `results/dit_training_history.json` is missing

#### What lacked information

The merged training-dynamics producer requires both `clop_hist` and `dit_hist` dictionaries. Those histories are not present in the workspace as reusable result files, so the figure cannot be regenerated directly from the current repository state.

#### Experiment / pipeline step that is incomplete

- Missing training-history export or preservation for the CLOP training experiment
- Missing training-history export or preservation for the DiT training experiment
- Current artifact management is insufficient for post-commit figure refresh of Figure 2

#### Impact

- The current article PDF still embeds a pre-commit Figure 2
- Any cosmetic or policy updates in `src/visualization/panels_training.py` after `19:37` are not reflected in the PDF

### 2. Figure 3 regeneration failed

- Target figure: `fig_embedding_space.pdf`
- Producer: `src/visualization/panels_merged.py`
- Commit touched producer: yes
- Existing figure timestamp: `2026-03-06 19:37:52 +0800`
- Commit timestamp: `2026-03-06 20:16:37 +0800`
- Failure mode: figure was generated before the commit and could not be regenerated during verification.

#### Missing information / missing artifacts

- `data/cache/text_group_ids_dedup.npy` is missing
- `data/cache/projected_text.npy` is missing
- `data/cache/projected_cells.npy` is missing
- Fallback `data/cache/cell_embeddings_dedup_preprocessed.npy` is also not available under `data/cache/`

#### What lacked information

The embedding-space merged figure requires cached CLOP-space artifacts for row B, and real-cell embedding cache for row E. Generated embeddings exist, but the corresponding CLOP projection cache is absent from the expected cache directory, so the merged article figure cannot be rebuilt from current files.

#### Experiment / pipeline step that is incomplete

- Missing cached CLOP projection export for text/cell embedding experiment
- Missing reusable cache packaging for the embedding-space visualization experiment
- Cache location and retention are incomplete for Figure 3 regeneration

#### Impact

- The current article PDF still embeds a pre-commit Figure 3
- Any layout/style updates in `src/visualization/panels_merged.py` after `19:37` are not reflected in the PDF

## Successful refreshes

### Figure 1

- File: `results/figures/fig_architecture.pdf`
- Regenerated timestamp: `2026-03-06 20:24:32 +0800`
- Included in refreshed article PDF

### Figure 4

- File: `results/figures/panel_d_metrics_summary.pdf`
- Regenerated timestamp: `2026-03-06 20:24:43 +0800`
- Included in refreshed article PDF

## Current PDF status

- PDF file: `articles/clop_dit_biology.pdf`
- Recompiled timestamp: `2026-03-06 20:25:43 +0800`
- Sync state:
  - Figure 1: current
  - Figure 2: stale
  - Figure 3: stale
  - Figure 4: current

## Next Steps

### Required to complete figure synchronization

1. Recover or regenerate CLOP and DiT training histories
   - Required outputs:
     - `results/clop_training_history.json`
     - `results/dit_training_history.json`
   - Then rerun the Figure 2 producer in `src/visualization/panels_training.py`

2. Recover or regenerate embedding-space cache artifacts
   - Required outputs:
     - `data/cache/text_group_ids_dedup.npy`
     - `data/cache/projected_text.npy`
     - `data/cache/projected_cells.npy`
     - or the expected fallback real-cell cache file in the same cache directory
   - Then rerun the Figure 3 producer in `src/visualization/panels_merged.py`

3. Recompile the LaTeX article after Figures 2 and 3 are regenerated
   - Target: `articles/clop_dit_biology.pdf`

### Recommended pipeline-level fix

1. Make training histories first-class saved outputs of the report pipeline
2. Make CLOP projection caches first-class saved outputs of the report pipeline
3. Add a preflight validation step before article compilation to fail if article figures are older than the last figure-related commit or older than their producers
4. Ensure `scripts/regenerate_report.sh` can rebuild Figures 2 and 3 from repository-local artifacts without requiring hidden cache state

## Practical rerun path

If the goal is full synchronization after the latest visualization commit, the most reliable next action is:

1. restore the missing training and cache artifacts, or rerun the upstream generation steps that create them
2. rerun the relevant figure producers for Figures 2 and 3
3. rerun `bash scripts/verify_article_figures.sh`
4. rerun LaTeX compilation for `articles/clop_dit_biology.tex`