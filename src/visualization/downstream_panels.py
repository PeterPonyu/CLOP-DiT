"""
downstream_panels.py — Backward-compatibility shim.

All implementation has moved to :mod:`fig17_downstream`.
Panel R (DE concordance) is re-exported from :mod:`panels_de_concordance`.
"""

from .fig17_downstream import (  # noqa: F401
    generate_downstream_panels,
    plot_classifier_panel,
    plot_clustering_and_classifier_merged,
    plot_clustering_panel,
)
from .panels_de_concordance import plot_de_concordance_panel  # noqa: F401

__all__ = [
    "plot_clustering_panel",
    "plot_classifier_panel",
    "plot_de_concordance_panel",
    "plot_clustering_and_classifier_merged",
    "generate_downstream_panels",
]
