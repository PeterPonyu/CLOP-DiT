"""
Centralized visualization style for CLOP-DiT publication figures.

Provides:
  - VIS_STYLE: matplotlib rcParams dict (Nature/Cell conventions)
  - TYPE_PALETTE: 69 deterministic, colorblind-friendly colours
  - COLORS: named semantic colours for real/gen/baseline
  - apply_style(): activate rcParams globally
  - style_axes(ax, kind): per-subplot typography & spine cleanup
  - save_panel(fig, path, dpi): save PNG + PDF in one call
  - FONT_LEGEND: standard legend font size
  - add_colorbar_safe(): colorbar helper with consistent defaults
  - set_figure_suptitle(): suptitle helper using SUPTITLE_Y
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# ──────────────────────────────────────────────────────────────
# Publication rcParams — Nature/Cell conventions
# ──────────────────────────────────────────────────────────────
# Calibrated for MDPI column: half-width panels at figsize=(4.5,3.2) scale ~0.71x
# at 0.48\linewidth (3.21" print on A4 170mm text width).
# With composed_scale=0.70, sizes must satisfy: size * 0.70 >= 7pt.
# → min body text ~10pt, titles ~12pt, ticks ~10pt, legends ~10pt.
VIS_STYLE: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.titlepad": 8,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "legend.edgecolor": "0.8",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "figure.constrained_layout.use": False,
    "figure.facecolor": "white",
}

# ──────────────────────────────────────────────────────────────
# Semantic colour palette
# ──────────────────────────────────────────────────────────────
COLORS = {
    "real": "#0D47A1",
    "generated": "#BF360C",
    "baseline_gauss": "#1B5E20",
    "baseline_shuffle": "#4A148C",
    "good": "#1B5E20",
    "warn": "#E65100",
    "bad": "#B71C1C",
    "neutral": "#455A64",
    "accent": "#FF8F00",
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

# Consistent suptitle vertical position — keeps title close to axes
SUPTITLE_Y = 0.98
# Standard legend font size (matches VIS_STYLE legend.fontsize)
FONT_LEGEND = 10
_FONTS_REGISTERED = False


# ──────────────────────────────────────────────────────────────
# Style helpers
# ──────────────────────────────────────────────────────────────

def apply_style() -> None:
    """Activate VIS_STYLE globally via ``matplotlib.rcParams``."""
    global _FONTS_REGISTERED
    if not _FONTS_REGISTERED:
        register_project_fonts()
        _FONTS_REGISTERED = True
    matplotlib.rcParams.update(VIS_STYLE)


def register_project_fonts(font_dir: Optional[Path | str] = None) -> list[str]:
    """Register local font files so Arial can resolve on clean systems."""
    candidates = []
    if font_dir is not None:
        candidates.append(Path(font_dir))
    else:
        root = Path(__file__).resolve().parents[2]
        candidates.extend([
            root / "fonts",
            root / "assets" / "fonts",
            root / "articles" / "fonts",
        ])

    registered: list[str] = []
    for base in candidates:
        if not base.exists() or not base.is_dir():
            continue
        for ext in ("*.ttf", "*.otf", "*.ttc"):
            for fpath in sorted(base.glob(ext)):
                try:
                    fm.fontManager.addfont(str(fpath))
                    registered.append(str(fpath))
                except Exception:
                    continue
    return registered


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
        "default":  {"title": 11, "label": 10, "tick": 10, "grid": True},
        "bar":      {"title": 11, "label": 10, "tick": 10, "grid": True},
        "heatmap":  {"title": 11, "label": 10, "tick": 8,  "grid": False},
        "scatter":  {"title": 11, "label": 10, "tick": 10, "grid": True},
        "umap":     {"title": 11, "label": 10, "tick": 10, "grid": False},
        "polar":    {"title": 11, "label": 10, "tick": 10, "grid": True},
        "table":    {"title": 11, "label": 10, "tick": 10, "grid": False},
    }
    s = style_map.get(kind, style_map["default"])

    if title:
        ax.set_title(title, fontsize=s["title"])
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

    ax._clop_styled = True
    return ax


def set_figure_suptitle(
    fig: plt.Figure,
    title: str,
    fontsize: int = 11,
    **kwargs,
) -> None:
    """Set a figure suptitle at the canonical vertical position."""
    fig.suptitle(title, fontsize=fontsize, y=SUPTITLE_Y, **kwargs)


def add_colorbar_safe(
    mappable,
    *,
    ax: plt.Axes,
    label: Optional[str] = None,
    shrink: float = 0.6,
    pad: float = 0.08,
    orientation: str = "vertical",
    aspect: int = 20,
    **kwargs,
):
    """Add a colorbar with sensible defaults and guard against missing mappable."""
    fig = ax.get_figure()
    cbar = fig.colorbar(
        mappable, ax=ax, shrink=shrink, pad=pad,
        orientation=orientation, aspect=aspect, **kwargs,
    )
    if label:
        cbar.set_label(label, fontsize=VIS_STYLE.get("axes.labelsize", 10))
    cbar.ax.tick_params(labelsize=VIS_STYLE.get("xtick.labelsize", 10))
    return cbar


def save_with_vcd(
    fig: plt.Figure,
    path: Path | str,
    dpi: int = 300,
    *,
    close: bool = False,
    run_vcd: bool = True,
) -> Path:
    """Canonical save: tight_layout, margins, VCD check, PNG + PDF.

    This is the **single** save path for all CLOP-DiT figures.
    It avoids mixing constrained_layout with tight_layout and uses
    consistent ``bbox_inches="tight"`` with ``pad_inches=0.08``.

    Parameters
    ----------
    fig : Figure
    path : output path (PNG; PDF is saved alongside)
    dpi : resolution
    close : whether to ``plt.close(fig)`` after saving
    run_vcd : whether to run visual conflict detection before save
    """
    import logging as _logging

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    basename = path.stem

    # 1) Apply style_axes to all axes (if not already styled by caller)
    for ax in fig.get_axes():
        if not ax.axison:
            continue
        if getattr(ax, "_clop_styled", False):
            continue
        if hasattr(ax, "name") and ax.name == "polar":
            style_axes(ax, kind="polar")
        elif ax.images:
            style_axes(ax, kind="heatmap")
        else:
            style_axes(ax, kind="default")

    # 2) tight_layout — single call with generous rect to leave room for suptitle
    #    and avoid labels being clipped.  Do NOT follow this with subplots_adjust,
    #    which would fight the layout engine and produce inconsistent spacing.
    try:
        fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.94], pad=0.8)
    except Exception:
        pass  # fall back gracefully

    # 3) Run VCD (strict mode)
    if run_vcd:
        try:
            import sys
            _scripts = Path(__file__).resolve().parent.parent.parent / "scripts"
            if str(_scripts) not in sys.path:
                sys.path.insert(0, str(_scripts))
            from visual_conflict_detector import detect_all_conflicts
            issues = detect_all_conflicts(fig, label=basename, verbose=True)
            if issues:
                n_warn = sum(1 for x in issues if x.get("severity") == "warning")
                if n_warn > 0:
                    try:
                        from vcd.vcd_actions import diagnose
                        actions = diagnose(issues)
                        top_actions = ", ".join(a.action_type for a in actions[:4])
                    except Exception:
                        top_actions = ""
                    _logging.getLogger(__name__).warning(
                        "%s: %d visual conflict warning(s)%s",
                        basename,
                        n_warn,
                        f" | suggested actions: {top_actions}" if top_actions else "",
                    )
        except Exception:
            pass

    # 4) Save PNG + PDF with consistent settings
    save_kw = dict(dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(path, **save_kw)
    fig.savefig(path.with_suffix(".pdf"), **save_kw)

    if close:
        plt.close(fig)
    return path


# Backward-compatible alias
save_panel = save_with_vcd


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

# Subplot spacing presets.
# hspace / wspace: fraction of subplot size used as inter-subplot gap.
# Recommended range for dense figures: hspace 0.30–0.55, wspace 0.30–0.55.
GRIDSPEC_TIGHT   = {"wspace": 0.40, "hspace": 0.45}   # compact multi-row panels
GRIDSPEC_DEFAULT = {"wspace": 0.50, "hspace": 0.50}   # standard multi-row panels
GRIDSPEC_1ROW    = {"wspace": 0.40}                   # single-row panels (no hspace needed)


def set_dense_tick_labels(
    ax: plt.Axes,
    axis: str = "both",
    *,
    max_labels: int = 25,
    fontsize: int = 10,
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


def set_scientific_tickformat(
    ax: plt.Axes,
    axis: str = "y",
    *,
    scilimits: tuple[int, int] = (-2, 3),
) -> None:
    """Apply scientific notation formatting for large/small magnitudes."""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(scilimits)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        fmt_y = ScalarFormatter(useMathText=True)
        fmt_y.set_scientific(True)
        fmt_y.set_powerlimits(scilimits)
        ax.yaxis.set_major_formatter(fmt_y)
