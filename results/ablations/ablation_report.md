# CLOP Ablation Study Results

Generated: 2026-03-07 14:59

## Results Table

| Rank | Experiment | Category | Val Proto Acc | Top-5 | Top-10 | Best Epoch | Gap | Description |
|------|-----------|----------|--------------|-------|--------|------------|-----|-------------|
| 1 |  **no_cohesion** | regularization | 86.41% | 99.7% | 99.9% | 13 | 0.9x | Disable prototype cohesion regularization |
| 2 | no_cell_noise | regularization | 86.23% | 99.7% | 99.9% | 15 | 1.0x | Disable cell embedding noise augmentation |
| 3 | lighter_dropout | regularization | 77.32% | 99.4% | 99.8% | 16 | 0.9x | Lower dropout (0.15 vs 0.30) |
| 4 | no_variants | regularization | 76.86% | 99.3% | 99.8% | 18 | 0.9x | Disable caption variant augmentation |
| 5 | low_temperature_cap | optimization | 76.78% | 99.3% | 99.8% | 18 | 0.9x | Temperature cap at 20 (v8.1 level) vs 50 |
| 6 | no_rdrop | regularization | 76.77% | 99.3% | 99.8% | 18 | 0.9x | Disable R-Drop consistency regularization |
| 7 | no_label_smoothing | optimization | 76.74% | 99.3% | 99.8% | 18 | 0.9x | Disable label smoothing |
| 8 | no_duplicate_mask | conditioning | 76.73% | 99.3% | 99.8% | 18 | 0.9x | Disable automatic duplicate-text masking |
| 9 | baseline | reference | 76.17% | 99.4% | 99.8% | 17 | 0.9x | Full v8.2 config (all improvements enabled) |
| 10 | smaller_proj | architecture | 76.15% | 99.4% | 99.8% | 18 | 0.9x | Reduce projection head to 256 dimensions |
| 11 | no_regularization | regularization | 76.10% | 99.4% | 99.8% | 17 | 0.9x | Remove all new regularization (variants, MixUp, R- |
| 12 | no_mixup | regularization | 76.03% | 99.4% | 99.8% | 17 | 0.9x | Disable embedding MixUp regularization |
| 13 | wider_proj | architecture | 75.91% | 99.3% | 99.8% | 18 | 0.9x | Wider projection head (768-dim vs 512-dim) |
| 14 | fixed_temperature | optimization | 71.55% | 99.1% | 99.8% | 23 | 0.0x | Freeze temperature instead of learning it |
| 15 | high_variant_prob | regularization | 70.14% | 96.7% | 98.6% | 28 | 0.8x | Higher variant probability (0.5 vs 0.3) |
| 16 | [hist] v6.4.1 | historical | 10.45% | 0.0% | 0.0% | 102 | 3.5x |  |
| 17 | [hist] v6.3 | historical | 10.22% | 0.0% | 0.0% | 32 | 0.0x |  |
| 18 | [hist] v7.0 | historical | 9.99% | 27.3% | 38.9% | 107 | 3.5x |  |
| 19 | [hist] v8.1_groupfix | historical | 9.72% | 26.2% | 36.2% | 113 | 3.5x | v8.1 with explicit group_id propagation. +0.84pp v |
| 20 | [hist] v7.2_improved_baseline | historical | 9.34% | 26.5% | 38.2% | 111 | 3.4x | Regressed -1.1pp vs v6.4.1 due to missing temp_lr_ |
| 21 | [hist] v8.1_rerun_120ep | historical | 8.88% | 0.0% | 0.0% | 105 | 3.9x | v8.1 re-run with fixed config (120 epochs, broken  |
| 22 | [hist] clop_v8.1_1772467137 | historical | 6.66% | 24.9% | 37.4% | 68 | 4.3x |  |
| 23 | [hist] v6.4 | historical | 5.65% | 0.0% | 0.0% | 0 | 0.0x | checkpoint/history were overwritten by later runs; |
| 24 | [hist] v8.3_variants_only | historical | 5.32% | 18.4% | 26.4% | 95 | 3.3x | Caption variant augmentation (prob=0.3) HURTS: -4. |
| 25 | [hist] v8.2_all_improvements | historical | 1.50% | 0.0% | 0.0% | 3 | 0.2x | FAILED — Combined regularization (variants+MixUp+R |
| 26 | [hist] v7.0_temp_reg | historical | 1.35% | 0.0% | 0.0% | 4 | 0.0x |  |
| 27 | [hist] v7.1_light_temp_reg | historical | 1.31% | 0.0% | 0.0% | 45 | 0.0x |  |

## Ablation Impact Analysis

| Ablation | Category | Δ Val Proto Acc | Δ Top-10 | Impact |
|----------|----------|----------------|----------|--------|
| no_cohesion | regularization | +10.24pp | +0.1pp | positive |
| no_cell_noise | regularization | +10.07pp | +0.1pp | positive |
| lighter_dropout | regularization | +1.15pp | +0.0pp | positive |
| no_variants | regularization | +0.69pp | +0.0pp | positive |
| low_temperature_cap | optimization | +0.61pp | +0.0pp | positive |
| no_rdrop | regularization | +0.60pp | +0.0pp | positive |
| no_label_smoothing | optimization | +0.57pp | +0.0pp | positive |
| no_duplicate_mask | conditioning | +0.56pp | +0.0pp | positive |
| smaller_proj | architecture | -0.01pp | +0.0pp | neutral |
| no_regularization | regularization | -0.07pp | +0.0pp | neutral |
| no_mixup | regularization | -0.14pp | +0.0pp | neutral |
| wider_proj | architecture | -0.26pp | +0.0pp | neutral |
| fixed_temperature | optimization | -4.61pp | +0.0pp | negative |
| high_variant_prob | regularization | -6.03pp | -1.2pp | negative |

## Key Findings

- **Best configuration**: no_cohesion (86.41%)
- **Worst configuration**: high_variant_prob (70.14%)
- **Baseline (v8.2 full)**: 76.17%
- **Ablation families**:
  - architecture: best = smaller_proj (76.15%)
  - conditioning: best = no_duplicate_mask (76.73%)
  - optimization: best = low_temperature_cap (76.78%)
  - reference: best = baseline (76.17%)
  - regularization: best = no_cohesion (86.41%)