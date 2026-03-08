# Analysis & Diagnostics Scripts

> Scripts for post-hoc analysis, diagnostics, and specialized utilities.

**Location**: `scripts/analysis/`
**Version Compatibility**: v6.0+
**Last Updated**: 2025-03-08

---

## Script Index

### Marker Gene Analysis

| Script | Version | Size | Description |
|--------|---------|------|-------------|
| `07_marker_gene_analysis.py` | v9.3 | 27KB | Marker gene detection and analysis |
| `generate_architecture_figure.py` | v6.0+ | 20KB | Generate architecture diagram |

### Data Quality & Audit

| Script | Version | Size | Description |
|--------|---------|------|-------------|
| `audit_data.py` | v6.0+ | 9.8KB | Data quality auditing |
| `deduplicate_captions.py` | v6.3+ | 11KB | Caption deduplication |
| `fix_dedup_captions.py` | v6.3+ | 6.0KB | Fix deduplication issues |
| `decode_expression.py` | v6.0+ | 16KB | Decode embeddings to expression |

### Evaluation & Metrics

| Script | Version | Size | Description |
|--------|---------|------|-------------|
| `compute_bootstrap_cis.py` | v9.3 | 11KB | Bootstrap confidence intervals |
| `conditioning_analysis.py` | v9.3 | 11KB | Text conditioning analysis |
| `diversity_diagnostics.py` | v9.3 | 28KB | Diversity metric diagnostics |

---

## Common Analysis Workflows

### Post-Generation Analysis

```bash
# Analyze generated cells
python scripts/analysis/07_marker_gene_analysis.py \
    --input results/generated_cd4_t.h5ad

# Compute diversity metrics
python scripts/analysis/diversity_diagnostics.py \
    --embeddings results/generated_embeddings.npy
```

### Data Auditing

```bash
# Audit cache quality
python scripts/analysis/audit_data.py \
    --cache_dir data/cached_latents_v5.2

# Deduplicate if needed
python scripts/analysis/deduplicate_captions.py \
    --input data/processed_h5ad/ \
    --output data/deduplicated/
```

---

## Output Locations

| Script Type | Output Directory |
|-------------|------------------|
| Marker analysis | `results/downstream/` |
| Diversity diagnostics | `results/diversity_diagnostics.json` |
| Bootstrap CIs | `results/bootstrap_cis.json` |
| Audit reports | `logs/audit_*.log` |

---

## See Also

- [Inference Scripts](../inference/INDEX.md) - Generate data to analyze
- [VCD System](../vcd/) - Visual conflict detection
- [Results Index](../../results/INDEX.md) - Output structure
