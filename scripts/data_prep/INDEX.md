# Data Preparation Scripts (Phase 0-3)

> Scripts for data acquisition, integration, text generation, and embedding caching.

**Location**: `scripts/data_prep/`
**Version Compatibility**: v5.2+
**Last Updated**: 2025-03-08

---

## Script Index

| Script | Version | Size | Description | Dependencies |
|--------|---------|------|-------------|--------------|
| `00_prepare_all_data.py` | v5.2+ | 27KB | Initial data preparation from raw sources | None |
| `01_integrate_h5_datasets.py` | v5.2+ | 24KB | 10x h5 dataset integration | `00_prepare_all_data.py` |
| `01b_integrate_geodh_h5ad.py` | v6.0+ | 16KB | GEO-DataHub integration | `00_prepare_all_data.py` |
| `02_subcluster_descriptions.py` | v6.0+ | 2.3KB | Generate subcluster text descriptions | `01*_integrate_*.py` |
| `02b_enrich_descriptions.py` | v6.2+ | 32KB | Evidence-based text enrichment | `02_subcluster_descriptions.py` |
| `02d_rebuild_v63_cache.py` | v6.3+ | 19KB | Rebuild cache for v6.3 template anchors | `02b_enrich_descriptions.py` |
| `03_cache_latents.py` | v5.2+ | 3.4KB | Cache scGPT + BiomedBERT embeddings | `02*_descriptions.py` |
| `03b_preprocess_embeddings.py` | v6.0+ | 2.7KB | ZCA whitening preprocessing | `03_cache_latents.py` |
| `03c_build_dedup_cache.py` | v6.3+ | 6.5KB | Build deduplicated cache | `03b_preprocess_embeddings.py` |

---

## Execution Order

```bash
# Phase 0: Initial preparation
python scripts/data_prep/00_prepare_all_data.py

# Phase 1: Dataset integration
python scripts/data_prep/01_integrate_h5_datasets.py
python scripts/data_prep/01b_integrate_geodh_h5ad.py

# Phase 2: Text generation
python scripts/data_prep/02_subcluster_descriptions.py
python scripts/data_prep/02b_enrich_descriptions.py

# Phase 3: Embedding caching
python scripts/data_prep/03_cache_latents.py
python scripts/data_prep/03b_preprocess_embeddings.py
python scripts/data_prep/03c_build_dedup_cache.py
```

---

## Common Issues

- **Memory**: `03_cache_latents.py` requires ~8GB RAM for large datasets
- **Time**: Full phase 0-3 takes ~4-6 hours on standard hardware
- **Storage**: Cache outputs ~4GB in `data/cached_latents_v5.2/`

---

## See Also

- [Training Scripts](../training/INDEX.md) - Phase 4 training
- [Full Pipeline](../pipeline/INDEX.md) - Orchestration
- [Docs: QUICK_START.md](../../docs/operational/QUICK_START.md) - Complete workflow
