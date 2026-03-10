"""
panels_diversity.py — Backward-compatibility shim.

The implementations have moved to dedicated per-figure modules:
  * plot_diagnostics               ->  fig12_diversity.py      (Article Fig 12)
  * plot_expression_diversity_panel ->  fig14_expr_diversity.py (Article Fig 14)

This module re-exports both functions so existing call-sites keep working.
"""

from .fig12_diversity import plot_diagnostics  # noqa: F401
from .fig14_expr_diversity import plot_expression_diversity_panel  # noqa: F401

__all__ = ["plot_diagnostics", "plot_expression_diversity_panel"]
