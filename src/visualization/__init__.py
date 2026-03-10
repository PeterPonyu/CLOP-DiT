from .style import (
    VIS_STYLE, TYPE_PALETTE, COLORS, METHOD_COLORS,
    apply_style, style_axes, save_panel,
    add_panel_label, add_panel_labels_to_axes,
)
from ._plot_helpers import plot_umap_overlay, plot_confusion_matrix, plot_roc_curve

try:
    from .results_visualizer import ResultsVisualizer
except Exception:  # pragma: no cover - optional dependency chain may be missing in lightweight envs
    ResultsVisualizer = None

__all__ = [
    "ResultsVisualizer",
    "VIS_STYLE",
    "TYPE_PALETTE",
    "COLORS",
    "METHOD_COLORS",
    "apply_style",
    "style_axes",
    "save_panel",
    "add_panel_label",
    "add_panel_labels_to_axes",
    "plot_umap_overlay",
    "plot_confusion_matrix",
    "plot_roc_curve",
]
