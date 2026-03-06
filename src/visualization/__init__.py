from .style import VIS_STYLE, TYPE_PALETTE, COLORS, apply_style, style_axes, save_panel

try:
    from .results_visualizer import ResultsVisualizer
except Exception:  # pragma: no cover - optional dependency chain may be missing in lightweight envs
    ResultsVisualizer = None

__all__ = [
    "ResultsVisualizer",
    "VIS_STYLE",
    "TYPE_PALETTE",
    "COLORS",
    "apply_style",
    "style_axes",
    "save_panel",
]
