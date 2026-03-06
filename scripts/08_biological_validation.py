#!/usr/bin/env python3
# 08_biological_validation.py — Publication-ready biological validation (v2)
"""
Focused biological validation for CLOP-DiT producing 3 core figures:

  Figure 1 — Text2Cell Biological Fidelity (multi-dataset, N×3 panel)
  Figure 2 — Cell2Cell Editing Quality (1×3 panel)
  Figure 3 — External Cell Identity Validation (CellTypist, 1×2 panel)
  Figure 4 — Multi-dataset Summary

+ metrics_summary.json with all quantitative metrics.

Usage:
  python scripts/08_biological_validation.py \\
      --reference_h5ad data/processed_h5ad/GSE123902_LungAdreHmCancer_processed.h5ad \\
      --dataset_key GSE123902_LungAdreHmCancer \\
      --output_dir figures/biovalidation

  python scripts/08_biological_validation.py \\
      --dataset_manifest configs/biovalidation_datasets.json \\
      --output_dir figures/biovalidation
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.biological_validation import main

if __name__ == "__main__":
    main()
