# experiments/__init__.py — Experiment modules for CLOP-DiT master plan
"""
Experiment modules for CLOP-DiT evaluation and application studies.

Modules
-------
ood_evaluation
    Out-of-distribution prompt robustness evaluation.
rare_cell_augmentation
    Data augmentation utility study for rare cell types.
"""

from . import ood_evaluation
from . import rare_cell_augmentation

__all__ = [
    "ood_evaluation",
    "rare_cell_augmentation",
]
