"""
panels_heatmaps.py -- Backward-compatibility shim.

The actual implementations have been split into dedicated per-figure modules:

  * :pymod:`fig06_fidelity`  -- ``plot_per_type_generation()`` (Article Fig 6)
  * :pymod:`fig07_alignment` -- ``plot_text_cell_heatmap()``   (Article Fig 7)

This module re-exports every public name so that existing ``from
src.visualization.panels_heatmaps import ...`` statements continue to work
without modification.
"""

from .fig06_fidelity import plot_per_type_generation  # noqa: F401
from .fig07_alignment import plot_text_cell_heatmap  # noqa: F401

__all__ = [
    "plot_text_cell_heatmap",
    "plot_per_type_generation",
]
