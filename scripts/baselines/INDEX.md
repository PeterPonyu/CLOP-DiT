# Baseline Methods

> Training scripts for baseline comparison methods.

**Location**: `scripts/baselines/`
**Scripts**: 2
**Version**: v9.3+

---

## Scripts

| Script | Method | Description |
|--------|--------|-------------|
| `train_embedding_vae_baseline.py` | Embedding VAE | Variational autoencoder baseline |
| `train_scvi_baseline.py` | scVI | Single-cell variational inference |

---

## Usage

```bash
# Train Embedding VAE baseline
python scripts/baselines/train_embedding_vae_baseline.py \
    --config configs/baselines/embedding_vae.yaml

# Train scVI baseline (dry run)
python scripts/baselines/train_scvi_baseline.py \
    --config configs/baselines/scvi.yaml \
    --dry-run
```

---

## Output

Baseline results are written to:
- `results/baselines/{method}/`
- See [baseline registry](../../src/evaluation/baseline_registry.py) for details

---

## See Also

- [Evaluation: Baseline Registry](../../src/evaluation/baseline_registry.py)
- [Results: Baselines](../../results/baselines/)
