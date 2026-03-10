"""
panels_expression.py -- Backward-compatibility shim.

The actual implementations have been split into dedicated per-figure modules:

  * :pymod:`fig08_markers`   -- ``plot_marker_gene_comparison()``  (Article Fig 8)
  * :pymod:`fig09_expression_corr` -- ``plot_expression_correlation()`` (Article Fig 9)
  * :pymod:`fig10_expression_analysis` -- ``plot_expression_analysis()`` (Article Fig 10)

This module re-exports every public name so that existing ``from
src.visualization.panels_expression import ...`` statements continue to work
without modification.
"""

from .fig08_markers import (  # noqa: F401 -- re-export
    MARKER_PANEL_GENES,
    MARKER_PANEL_TYPES,
    plot_marker_gene_comparison,
)
from .fig09_expression_corr import plot_expression_correlation  # noqa: F401
from .fig10_expression_analysis import plot_expression_analysis  # noqa: F401

__all__ = [
    "MARKER_PANEL_GENES",
    "MARKER_PANEL_TYPES",
    "plot_marker_gene_comparison",
    "plot_expression_correlation",
    "plot_expression_analysis",
]
