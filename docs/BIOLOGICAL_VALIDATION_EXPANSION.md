# Biological Validation Expansion

*Last revised: 2026-03-06*

This note tracks concrete ways to broaden the CLOP-DiT biological results beyond the current generation-only narrative.

## Dataset expansion targets

- Add at least one primarily developmental dataset to the `08_biological_validation.py` dataset manifest.
- Add at least one immune-focused dataset with clear condition labels so differential-expression and prompt-sensitivity analyses are not limited to lineage identity alone.
- Preserve per-dataset exports (`text2cell_per_dataset.json/.csv`, `celltypist_per_dataset.json/.csv`) so weaker and stronger datasets can be summarized transparently.

## Additional DE contrasts

The downstream biology pipeline now accepts a contrasts JSON:

```bash
python -m src.evaluation.downstream_biology \
  --contrasts-json configs/downstream_contrasts_extended.json
```

This should be used to test:

- canonical lineage contrasts
- myeloid-state contrasts
- perturbation-style contrasts where metadata permit them

If a contrast is absent in a dataset, the current code safely skips it.

## Adjacent applications already supported by the codebase

- **Rare-cell augmentation**: use generated cells to fill low-count classes, then re-evaluate clustering and classifier transfer.
- **Cell-state editing**: use the Cell2Cell pathway and compare edit strength versus source-identity preservation.
- **Perturbation-style analysis**: reuse DE concordance on condition labels rather than only on cell-type labels.
- **Atlas completion**: assess whether generated cells improve coverage of sparse cell-type neighborhoods without harming mixing scores.

## Suggested reporting policy

- Keep the main article focused on the strongest, most interpretable biological tasks.
- Put broader dataset tables, extra contrasts, and application-specific stress tests in supplement or the internal evaluation report.
- Report both wins and non-wins for each added task so the broadened results remain credible.
