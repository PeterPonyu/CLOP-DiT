# Ablation Suite

*Last revised: 2026-03-06*

The CLOP ablation runner in `scripts/training/run_ablation_study.py` is now organized by ablation family rather than as a flat list.

## Families

| Family | Purpose | Example ablations |
|------|---------|-------------------|
| `reference` | Full reference configuration | `baseline` |
| `preprocessing` | Test sensitivity to embedding preprocessing | `no_whitening` |
| `regularization` | Test which regularizers matter | `no_variants`, `no_mixup`, `no_rdrop`, `no_cell_noise`, `no_cohesion`, `lighter_dropout`, `no_regularization` |
| `conditioning` | Test text/group conditioning aids | `no_duplicate_mask` |
| `architecture` | Test representational capacity | `wider_proj`, `smaller_proj` |
| `optimization` | Test training dynamics and temperature behavior | `low_temperature_cap`, `no_label_smoothing`, `fixed_temperature`, `small_batch` |

## Current scope

- This runner is **CLOP-stage specific**. It does not yet cover full DiT or end-to-end baseline ablations.
- Results are aggregated into `results/ablations/ablation_report.md` with category labels, which makes it easier to summarize which family of changes matters most.

## Usage

```bash
python scripts/training/run_ablation_study.py --list
python scripts/training/run_ablation_study.py --base_config configs/clop.yaml --report
```

## Reporting guidance

- Use the category labels when summarizing ablations in internal docs or supplements.
- Keep any manuscript-facing prose neutral and focused on the scientific interpretation of the ablation, not on anticipated reviewer reactions.
