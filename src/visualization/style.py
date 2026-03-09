"""
Centralized visualization style for CLOP-DiT publication figures.

Provides:
  - VIS_STYLE: matplotlib rcParams dict (Nature/Cell conventions)
  - TYPE_PALETTE: 69 deterministic, colorblind-friendly colours
  - COLORS: named semantic colours for real/gen/baseline
  - apply_style(): activate rcParams globally
  - style_axes(ax, kind): per-subplot typography & spine cleanup
  - save_panel(fig, path, dpi): save PNG + PDF in one call
    - FONT_LEGEND / FONT_LEGEND_DENSE: standard and dense legend font sizes
  - add_colorbar_safe(): colorbar helper with consistent defaults
  - set_figure_suptitle(): suptitle helper using SUPTITLE_Y
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
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
    "good": "#1565C0",
    "warn": "#F9A825",
    "bad": "#D84315",
    "neutral": "#455A64",
    "accent": "#FF8F00",
    "bg_light": "#F5F5F5",
    # Annotation and UI colors (centralized from panel files)
    "annotation_dark": "#333333",
    "annotation_medium": "#424242",
    "annotation_light": "#222222",
    "median_dark": "#1E1E1E",
    "border_light": "#DDDDDD",
    "border_medium": "#bdbdbd",
    "bg_gauge": "#E0E0E0",
    "bg_infobox": "#E8F0FE",
    "bg_yellow": "#FFFDE7",
    "border_amber": "#FBC02D",
    "heatmap_purple": "#6A1B9A",
    "error_red": "#D32F2F",
    "trend_dark": "#263238",
    "confusion_marker": "#00E5FF",  # Cyan — colorblind-safe on blue-red heatmaps
}

# Marker-gene category colors for expression panels
MARKER_CATEGORY_COLORS = {
    "CD8+ T": "#1565C0",
    "Myeloid": "#C62828",
    "Epithelial": "#2E7D32",
    "Stromal": "#6A1B9A",
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
SUPTITLE_Y = 0.96
SUPTITLE_Y_CLOSE = 0.96  # Multi-row figures: keeps suptitle closer to axes
# Standard legend font size (matches VIS_STYLE legend.fontsize)
FONT_LEGEND = 10
# Dense multi-panel figures where 10pt legends would crowd the layout
FONT_LEGEND_DENSE = 8
# Architecture diagram (Fig 1) — diagram-specific labels (min 5.5pt per VCD)
FONT_ARCH_LABEL = 7
FONT_ARCH_SUBLABEL = 7
# Centralized font sizes for publication figures
FONT_SUPTITLE = 11
FONT_TITLE = 11
FONT_LABEL = 10
FONT_TICK = 10
FONT_TICK_DENSE = 8
FONT_ANNOTATION = 8
FONT_SMALL = 7
# Minimum-size fonts for dense contexts (replaces illegal sub-7pt values)
FONT_HEATMAP_CELL = 7      # Heatmap cell annotations (was 5-6.5pt)
FONT_DENSE_YTICK = 7       # Dense y-axis tick labels (was 6pt)
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
    y = kwargs.pop("y", SUPTITLE_Y)
    fig.suptitle(title, fontsize=fontsize, y=y, **kwargs)


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.10,
    y: float = 1.05,
    *,
    fontsize: int = 14,
    fontweight: str = "bold",
    color: str = "black",
    stroke_linewidth: float = 3.0,
    stroke_foreground: str = "white",
    **kwargs,
) -> None:
    """Add a panel label (a, b, c, etc.) outside the top-left corner of a subplot.

    The label is placed outside the axes border (default x=-0.10, y=1.05
    in axes coordinates) so it never overlaps with plot content.  A white
    outline stroke (path_effects) ensures readability over any background.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the label to.
    label : str
        The label text (e.g., 'a', 'b', 'c').
    x, y : float
        Position in axes coordinates.  Defaults place the label just
        outside the top-left corner of the axes.
    fontsize : int
        Font size for the label.
    fontweight : str
        Font weight (e.g., 'bold', 'normal').
    color : str
        Text color.
    stroke_linewidth : float
        Width of the white outline stroke for readability.
    stroke_foreground : str
        Color of the outline stroke.
    **kwargs
        Additional keyword arguments passed to ``ax.text()``.
    """
    ax.text(
        x, y, f"({label})",
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        va="bottom",
        ha="left",
        zorder=100,
        path_effects=[
            pe.withStroke(linewidth=stroke_linewidth, foreground=stroke_foreground),
            pe.Normal(),
        ],
        **kwargs,
    )


def add_panel_labels_to_axes(
    axes: list[plt.Axes],
    labels: Optional[list[str]] = None,
    **kwargs,
) -> None:
    """Add sequential panel labels to a list of axes.

    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
        List of axes to label.
    labels : list of str, optional
        Custom labels. If None, uses 'a', 'b', 'c', ...
    **kwargs
        Passed to add_panel_label.
    """
    if labels is None:
        labels = [chr(ord('a') + i) for i in range(len(axes))]

    for ax, label in zip(axes, labels):
        add_panel_label(ax, label, **kwargs)


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
    if getattr(cbar, "solids", None) is not None:
        try:
            cbar.solids.set_edgecolor("face")
            cbar.solids.set_rasterized(True)
        except Exception:
            pass
    if getattr(cbar, "outline", None) is not None:
        cbar.outline.set_linewidth(0.6)
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
    layout_rect: tuple[float, float, float, float] | None = None,
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
    layout_rect : optional (left, bottom, right, top) in figure coords; if given,
        passed to tight_layout(rect=layout_rect) instead of default
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
        if layout_rect is not None:
            rect = list(layout_rect)
        else:
            rect = list(getattr(fig, "_clop_layout_rect", None) or [0.02, 0.03, 0.98, 0.95])
        fig.tight_layout(rect=rect, pad=0.8)
    except Exception:
        pass  # fall back gracefully

    # 3) Run VCD (strict mode)
    if run_vcd:
        try:
            import sys
            _scripts = Path(__file__).resolve().parent.parent.parent / "scripts"
            if str(_scripts) not in sys.path:
                sys.path.insert(0, str(_scripts))
            from vcd import detect_all_conflicts
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


def run_vcd_check(fig: plt.Figure, label: str) -> None:
    """Run visual conflict detection on a figure without saving. Used before PIL composition."""
    import logging as _logging

    try:
        import sys
        _scripts = Path(__file__).resolve().parent.parent.parent / "scripts"
        if str(_scripts) not in sys.path:
            sys.path.insert(0, str(_scripts))
        from vcd import detect_all_conflicts
        issues = detect_all_conflicts(fig, label=label, verbose=True)
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
                    label,
                    n_warn,
                    f" | suggested actions: {top_actions}" if top_actions else "",
                )
    except Exception:
        pass


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


def set_adaptive_ytick_labels(
    ax: plt.Axes,
    labels: list[str],
    *,
    min_visible: int = 5,
    max_visible: int = 14,
    fontsize: int = 8,
    preserve_ends: bool = True,
) -> None:
    """Show an adaptive number of y-tick labels, avoiding overlap.

    Replaces hard-coded ``i % 7 == 0`` thinning with a formula
    that guarantees at least *min_visible* labels are shown.
    """
    n = len(labels)
    ax.set_yticks(range(n))
    if n <= max_visible:
        ax.set_yticklabels(labels, fontsize=fontsize, ha="right")
        return
    step = max(1, int(np.ceil(n / max_visible)))
    thinned = [labels[i] if i % step == 0 else "" for i in range(n)]
    if preserve_ends and n > 0:
        thinned[0] = labels[0]
        thinned[-1] = labels[-1]
    ax.set_yticklabels(thinned, fontsize=fontsize, ha="right")


def abbreviate_cell_type(name: str, max_len: int = 20) -> str:
    """Intelligently abbreviate cell type names for figure labels.

    Uses biology-aware abbreviations before falling back to truncation.
    Replaces hard-coded ``[:20]`` slicing throughout the codebase.
    """
    if len(name) <= max_len:
        return name
    abbrevs = [
        ("lymphocytes", "lymph."),
        ("macrophages", "mac."),
        ("fibroblasts", "fibro."),
        ("progenitors", "prog."),
        ("endothelial", "endo."),
        ("mesenchymal", "mesen."),
        ("epithelial", "epith."),
        ("regulatory", "reg."),
        ("inflammatory", "inflam."),
        (" cells", ""),
        (" cell", ""),
    ]
    result = name
    for full, short in abbrevs:
        if len(result) <= max_len:
            break
        result = result.replace(full, short)
    if len(result) > max_len:
        result = result[:max_len - 1] + "\u2026"
    return result
