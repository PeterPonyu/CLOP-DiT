"""
Centralized visualization style for CLOP-DiT publication figures.

Provides:
  - VIS_STYLE: matplotlib rcParams dict (Nature/Cell conventions)
  - TYPE_PALETTE: 69 deterministic, colorblind-friendly colours
  - COLORS: named semantic colours for real/gen/baseline
  - apply_style(): activate rcParams globally
  - style_axes(ax, kind): per-subplot typography & spine cleanup
  - save_panel(fig, path, dpi): save PNG + PDF in one call
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────
# Publication rcParams — Nature/Cell conventions
# ──────────────────────────────────────────────────────────────
VIS_STYLE: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.edgecolor": "0.8",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "figure.constrained_layout.use": True,
    "figure.facecolor": "white",
}

# ──────────────────────────────────────────────────────────────
# Semantic colour palette
# ──────────────────────────────────────────────────────────────
COLORS = {
    "real": "#1976D2",
    "generated": "#FF7043",
    "baseline_gauss": "#4CAF50",
    "baseline_shuffle": "#9C27B0",
    "good": "#4CAF50",
    "warn": "#FF9800",
    "bad": "#F44336",
    "neutral": "#78909C",
    "accent": "#FFC107",
    "bg_light": "#F5F5F5",
}

# ──────────────────────────────────────────────────────────────
# 69-type deterministic palette (colourblind-friendly via HSL spacing)
# ──────────────────────────────────────────────────────────────

def _build_type_palette(n: int = 69) -> np.ndarray:
    """Generate *n* distinct colours via HSL spacing."""
    cmap = matplotlib.colormaps.get_cmap("gist_ncar").resampled(n + 4)
    colours = cmap(np.linspace(0.02, 0.95, n))
    rng = np.random.default_rng(42)
    order = rng.permutation(n)
    return colours[order]


TYPE_PALETTE = _build_type_palette(69)


# ──────────────────────────────────────────────────────────────
# Style helpers
# ──────────────────────────────────────────────────────────────

def apply_style() -> None:
    """Activate VIS_STYLE globally via ``matplotlib.rcParams``."""
    matplotlib.rcParams.update(VIS_STYLE)


def style_axes(
    ax: plt.Axes,
    kind: str = "default",
    *,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    hide_top_right: bool = True,
) -> plt.Axes:
    """Apply consistent typography and spine styling to *ax*.

    Parameters
    ----------
    kind : {"default", "bar", "heatmap", "scatter", "umap", "polar"}
        Adjusts font sizes and grid visibility to suit the subplot type.
    """
    style_map = {
        "default":  {"title": 13, "label": 11, "tick": 9,  "grid": True},
        "bar":      {"title": 13, "label": 11, "tick": 9,  "grid": True},
        "heatmap":  {"title": 13, "label": 11, "tick": 8,  "grid": False},
        "scatter":  {"title": 13, "label": 11, "tick": 9,  "grid": True},
        "umap":     {"title": 13, "label": 11, "tick": 9,  "grid": False},
        "polar":    {"title": 13, "label": 10, "tick": 9,  "grid": True},
        "table":    {"title": 13, "label": 11, "tick": 9,  "grid": False},
    }
    s = style_map.get(kind, style_map["default"])

    if title:
        ax.set_title(title, fontsize=s["title"], fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=s["label"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=s["label"])
    ax.tick_params(labelsize=s["tick"])
    if s["grid"]:
        ax.grid(True, alpha=0.25, linewidth=0.5)
    else:
        ax.grid(False)

    if hide_top_right and kind not in ("polar", "heatmap"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return ax


def save_panel(
    fig: plt.Figure,
    path: Path | str,
    dpi: int = 300,
    *,
    close: bool = False,
) -> Path:
    """Save *fig* as both PNG and PDF, return the PNG path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi)
    if close:
        plt.close(fig)
    return path


def quality_color(value: float, thresholds: tuple = (0.8, 0.5)) -> str:
    """Return good/warn/bad colour based on *value* vs *thresholds*."""
    hi, lo = thresholds
    if value >= hi:
        return COLORS["good"]
    elif value >= lo:
        return COLORS["warn"]
    return COLORS["bad"]


# ──────────────────────────────────────────────────────────────
# Layout and dense-label helpers (avoid font overlap)
# ──────────────────────────────────────────────────────────────

# Tighter subplot spacing for condensed figures
GRIDSPEC_TIGHT = {"wspace": 0.18, "hspace": 0.22}
GRIDSPEC_DEFAULT = {"wspace": 0.25, "hspace": 0.28}


def set_dense_tick_labels(
    ax: plt.Axes,
    axis: str = "both",
    *,
    max_labels: int = 25,
    fontsize: int = 6,
    rotation: int = 45,
    ha: str = "right",
) -> None:
    """Reduce tick label density to avoid overlap when many categories.

    If axis has more than *max_labels* ticks, show every Nth label.
    """
    for a in (["x", "y"] if axis == "both" else [axis]):
        ticks = ax.get_xticklabels() if a == "x" else ax.get_yticklabels()
        n = len(ticks)
        if n > max_labels:
            step = max(1, n // max_labels)
            for i, t in enumerate(ticks):
                if i % step != 0:
                    t.set_visible(False)
        for t in ticks:
            if t.get_visible():
                t.set_fontsize(fontsize)
                if a == "x":
                    t.set_rotation(rotation)
                    t.set_ha(ha)
                else:
                    t.set_rotation(0)
                    t.set_ha("right")
