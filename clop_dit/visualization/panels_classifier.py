"""
panels_classifier.py — Backward-compatibility shim.

Panel Q implementation has moved to :mod:`fig17_downstream`.
"""

from .fig17_downstream import (  # noqa: F401
    _compute_classifier_summary,
    _plot_classifier_metric_heatmap,
    plot_classifier_panel,
)

__all__ = [
    "plot_classifier_panel",
    "_compute_classifier_summary",
    "_plot_classifier_metric_heatmap",
]
