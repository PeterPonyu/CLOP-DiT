# Robustness Experiments

*Last revised: 2026-03-06*

This document defines the robustness experiments used to broaden the CLOP-DiT results without overloading the main article.

## Primary robustness axes

| Axis | Purpose | Recommended placement |
|------|---------|-----------------------|
| Seed stability | Show that key metrics do not depend on a single random seed | Supplementary table or figure |
| Subsampling robustness | Show how performance changes as available cells per type decrease | Main text summary plus supplementary detail |
| Caption sensitivity | Show whether marker-dropping or paraphrased prompts materially change biological fidelity | Supplementary |

## Recommended reporting split

- **Main article**:
  - A short paragraph summarizing whether key conclusions remain stable across seeds and moderate subsampling.
  - One compact supplementary pointer rather than a full extra figure in the main text.
- **Supplementary material**:
  - Per-seed metric table for KNN, steering, diversity ratio, centroid cosine, and at least one downstream metric.
  - Subsampling table or line plot across 100%, 50%, and 25% cells-per-type.
  - Caption-sensitivity comparison across default prompts, auto-generated prompts, and at least one marker-dropped prompt variant.

## Execution harness

**Prerequisites:** The dataset manifest `configs/biovalidation_datasets.json` must exist. For caption-sensitivity experiments, prompt variant files are required; the repo ships defaults so the workflow runs without extra setup.

- **Prompt variants (caption sensitivity):** The example config `configs/robustness_prompt_variants.example.json` points to three JSON files under `configs/prompts/`: `default_prompts.json`, `marker_dropped_prompts.json`, and `context_shuffled_prompts.json`. These files are included in the repo. Each file must be a JSON object mapping cell type names to prompt strings (same format as expected by `scripts/08_biological_validation.py --prompt_file`). To use custom prompt sets, create your own JSON files and either copy the example config to a new file and edit the paths, or pass your config via `--prompt-variants-json`.

Use:

```bash
python scripts/run_robustness_experiments.py \
  --dataset-manifest configs/biovalidation_datasets.json \
  --output-dir results/robustness
```

To include caption-sensitivity experiments (default, marker-dropped, and context-shuffled prompts), pass the example prompt-variants config:

```bash
python scripts/run_robustness_experiments.py \
  --dataset-manifest configs/biovalidation_datasets.json \
  --output-dir results/robustness \
  --prompt-variants-json configs/robustness_prompt_variants.example.json
```

By default the script writes `results/robustness/experiment_plan.json` without launching expensive runs. Add `--execute` to run the planned commands.

## Output policy

Each experiment writes into `results/robustness/<experiment_id>/` and should preserve:

- the exact command
- the seed or subsample fraction
- the prompt variant name, if applicable
- the raw metrics JSON produced by `scripts/08_biological_validation.py`

## Interpretation guidance

- Prefer reporting **ranges or confidence intervals** rather than cherry-picked best seeds.
- Treat caption-sensitivity runs as **stress tests**, not as the primary evaluation setting.
- If a method is strong on average but unstable under subsampling, say so explicitly in the supplement and discussion.
