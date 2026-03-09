# Contributing to CLOP-DiT

Thank you for your interest in contributing.

## Development Setup

```bash
conda create -n clopdit python=3.10
conda activate clopdit
pip install -e ".[dev]"
```

## Code Style

- Follow existing conventions in `src/` (type hints, docstrings for public functions)
- Use `black` for formatting if available
- Keep imports organized: stdlib, third-party, local

## Running Tests

```bash
pytest tests/ -v
```

Some tests require GPU or data files and will be automatically skipped in CI.

## Adding a Baseline Method

1. Create a training script in `scripts/baselines/`
2. Register the method in `src/evaluation/baseline_registry.py`
3. Export artifacts to `results/baselines/{method_name}/` following the contract in `REPRODUCIBILITY.md`
4. Add a config file in `configs/baselines/`

## Pull Requests

- Create a feature branch from `main`
- Include a clear description of changes
- Ensure `pytest tests/` passes
- Keep PRs focused on a single feature or fix
