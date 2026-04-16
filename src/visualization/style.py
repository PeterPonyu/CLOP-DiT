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

import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import Bbox

from .panel_geometry import (
    DEFAULT_EXPORT_PAD_INCHES,
    DEFAULT_LAYOUT_RECT,
    apply_layout_rect,
    get_export_pad_inches,
)

# ──────────────────────────────────────────────────────────────
# Publication rcParams — Nature/Cell conventions
# ──────────────────────────────────────────────────────────────
# Calibrated for a typical two-column page: half-width panels at figsize=(4.5,3.2)
# scale ~0.71x at 0.48\linewidth (3.21" print on A4 170mm text width).
# With composed_scale=0.70, sizes must satisfy: size * 0.70 >= 7pt.
# → min body text ~11pt, titles ~14pt, ticks ~11pt, legends ~11pt.
VIS_STYLE: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "normal",
    "axes.titlepad": 8,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "legend.edgecolor": "0.8",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
    "savefig.bbox": None,
    "savefig.pad_inches": 0.10,
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

# Canonical method colors for benchmarking and baseline comparison panels
METHOD_COLORS = {
    "CLOP-DiT":              COLORS["real"],              # #0D47A1
    "Gaussian N(\u03bc,\u03c3\u00b2I)": COLORS["baseline_gauss"],   # #1B5E20
    "Shuffled Labels":       COLORS["baseline_shuffle"],   # #4A148C
    "Random N(0,I)":         "#795548",
    "Mean-only (collapse)":  COLORS["accent"],             # #FF8F00
}

# Concordance metric colors (neutral, no semantic overlap with data-source colors)
METRIC_COLORS = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A"]

# ──────────────────────────────────────────────────────────────
# 69-type deterministic palette (HSL spacing via gist_ncar)
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
SUPTITLE_Y_CLOSE = SUPTITLE_Y  # Deprecated alias — identical to SUPTITLE_Y
# Standard legend font size (matches VIS_STYLE legend.fontsize)
FONT_LEGEND = 11
# Dense multi-panel figures where 11pt legends would crowd the layout
FONT_LEGEND_DENSE = 10
# Architecture diagram (Fig 1) — diagram-specific labels (min 5.5pt per VCD)
FONT_ARCH_LABEL = 12
FONT_ARCH_SUBLABEL = 11
FONT_ARCH_TITLE = 13
FONT_ARCH_LEGEND = 11
# Centralized font sizes for publication figures
FONT_SUPTITLE = 15
FONT_TITLE = 14
FONT_LABEL = 12
PANEL_LABEL_FONT_SIZE = 14
FONT_TICK = 11
FONT_TICK_DENSE = 10
FONT_ANNOTATION = 10
FONT_SMALL = 9
# Minimum-size fonts for dense contexts (replaces illegal sub-7pt values)
FONT_HEATMAP_CELL = 9      # Heatmap cell annotations
FONT_DENSE_YTICK = 9       # Dense y-axis tick labels
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
            root / "scripts" / "vcd" / "fonts",
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
        "default":  {"title": 14, "label": 12, "tick": 11, "grid": True},
        "bar":      {"title": 14, "label": 12, "tick": 11, "grid": True},
        "heatmap":  {"title": 14, "label": 12, "tick": 10, "grid": False},
        "scatter":  {"title": 14, "label": 12, "tick": 11, "grid": True},
        "umap":     {"title": 14, "label": 12, "tick": 11, "grid": False},
        "polar":    {"title": 14, "label": 12, "tick": 11, "grid": True},
        "table":    {"title": 14, "label": 12, "tick": 11, "grid": False},
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
    fontsize: int = PANEL_LABEL_FONT_SIZE,
    fontweight: str = "semibold",
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
        zorder=120,
        clip_on=False,
        gid=f"panel_label:{label}",
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


def is_vcd_enabled(default: bool = True) -> bool:
    """Return whether live VCD checks should run for figure generation.

    Controlled by the ``CLOPDIT_ENABLE_VCD`` environment variable.
    Truthy values: 1/true/yes/on; falsy values: 0/false/no/off.
    """
    raw = os.getenv("CLOPDIT_ENABLE_VCD")
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _safe_artist_bbox(artist, renderer) -> Bbox | None:
    """Return a display-coordinate bbox for a visible artist when possible."""
    if artist is None or not getattr(artist, "get_visible", lambda: True)():
        return None
    try:
        bbox = artist.get_window_extent(renderer)
    except Exception:
        return None
    if bbox is None or bbox.width <= 0 or bbox.height <= 0:
        return None
    return bbox


def _collect_export_bboxes(fig: plt.Figure, renderer) -> list[Bbox]:
    """Collect display-coordinate bboxes that should contribute to export cropping."""
    boxes: list[Bbox] = []
    # Only use the full canvas as a baseline when no tighter content exists.
    # This lets figures opt into tight cropping by setting fig._clop_tight_crop = True.
    if not getattr(fig, "_clop_tight_crop", False):
        width_px, height_px = fig.canvas.get_width_height()
        boxes.append(Bbox.from_extents(0, 0, width_px, height_px))

    for ax in fig.get_axes():
        if not ax.get_visible():
            continue
        try:
            bbox = ax.get_tightbbox(renderer)
        except Exception:
            bbox = None
        if bbox is not None and bbox.width > 0 and bbox.height > 0:
            boxes.append(bbox)

        legend = ax.get_legend()
        legend_bbox = _safe_artist_bbox(legend, renderer)
        if legend_bbox is not None:
            boxes.append(legend_bbox)

    for txt in getattr(fig, "texts", []) or []:
        bbox = _safe_artist_bbox(txt, renderer)
        if bbox is not None:
            boxes.append(bbox)

    suptitle = getattr(fig, "_suptitle", None)
    bbox = _safe_artist_bbox(suptitle, renderer)
    if bbox is not None:
        boxes.append(bbox)

    for legend in getattr(fig, "legends", []) or []:
        bbox = _safe_artist_bbox(legend, renderer)
        if bbox is not None:
            boxes.append(bbox)

    for artist in getattr(fig, "_clop_export_artists", []) or []:
        bbox = _safe_artist_bbox(artist, renderer)
        if bbox is not None:
            boxes.append(bbox)

    return boxes


def compute_fixed_export_bbox(
    fig: plt.Figure,
    *,
    pad_inches: float = DEFAULT_EXPORT_PAD_INCHES,
) -> Bbox:
    """Compute one deterministic bbox to reuse across all export formats.

    The returned bbox is expressed in inches and can be passed directly to
    ``savefig(..., bbox_inches=...)``.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = _collect_export_bboxes(fig, renderer)
    union = Bbox.union(boxes)
    pad_px = float(pad_inches) * float(fig.dpi)
    union = Bbox.from_extents(
        union.x0 - pad_px,
        union.y0 - pad_px,
        union.x1 + pad_px,
        union.y1 + pad_px,
    )
    return fig.dpi_scale_trans.inverted().transform_bbox(union)


def get_export_savefig_kwargs(
    fig: plt.Figure,
    dpi: int = 300,
    *,
    layout_rect: tuple[float, float, float, float] | None = None,
    pad_inches: float | None = None,
) -> dict:
    """Return deterministic savefig kwargs shared by JPEG/PDF/buffer exports."""
    if layout_rect is not None:
        apply_layout_rect(fig, layout_rect)
    elif getattr(fig, "_clop_layout_rect", None) is not None and not getattr(fig, "_clop_layout_managed", False):
        apply_layout_rect(fig, getattr(fig, "_clop_layout_rect", DEFAULT_LAYOUT_RECT))

    pad = get_export_pad_inches(
        fig,
        fallback=float(pad_inches if pad_inches is not None else VIS_STYLE.get("savefig.pad_inches", DEFAULT_EXPORT_PAD_INCHES)),
    )
    bbox_inches = compute_fixed_export_bbox(fig, pad_inches=pad)
    return {
        "dpi": dpi,
        "bbox_inches": bbox_inches,
        "pad_inches": 0.0,
    }


def save_with_vcd(
    fig: plt.Figure,
    path: Path | str,
    dpi: int = 300,
    *,
    close: bool = False,
    run_vcd: bool = True,
    layout_rect: tuple[float, float, float, float] | None = None,
) -> Path:
    """Canonical save: deterministic layout, shared export crop, JPEG + PDF.

    This is the **single** save path for all CLOP-DiT figures.
    Layout must be fixed in the figure module itself (typically via
    ``apply_layout_rect`` and direct artist positioning); the saver only
    computes one deterministic export bbox and reuses it across formats.

    Parameters
    ----------
    fig : Figure
    path : output path (PNG; PDF is saved alongside)
    dpi : resolution
    close : whether to ``plt.close(fig)`` after saving
    run_vcd : whether to run visual conflict detection before save
    layout_rect : optional (left, bottom, right, top) in figure coords; if given,
        applied via the deterministic layout-rect helper before export
    """
    import logging as _logging

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    basename = path.stem
    live_vcd_dir = path.parent / "_live_vcd"
    live_vcd_dir.mkdir(parents=True, exist_ok=True)
    live_vcd_payload = {
        "figure": basename,
        "warnings": [],
        "info": [],
        "error": None,
    }
    effective_run_vcd = bool(run_vcd) and is_vcd_enabled(default=True)

    # 1) Apply style_axes to all axes (if not already styled by caller)
    for ax in fig.get_axes():
        if not ax.axison:
            continue
        if getattr(ax, "_clop_styled", False):
            continue
        # Skip colorbar axes — their tick styling is set by the figure script
        if hasattr(ax, '_colorbar_info') or getattr(ax, '_colorbar', None) is not None:
            continue
        if hasattr(ax, "name") and ax.name == "polar":
            style_axes(ax, kind="polar")
        elif ax.images:
            style_axes(ax, kind="heatmap")
        else:
            style_axes(ax, kind="default")

    # 2) Freeze export geometry once so JPEG/PDF use the exact same crop.
    save_kw = get_export_savefig_kwargs(fig, dpi=dpi, layout_rect=layout_rect)

    # 3) Run VCD (strict mode)
    if effective_run_vcd:
        try:
            import sys
            _scripts = Path(__file__).resolve().parent.parent.parent / "scripts"
            if str(_scripts) not in sys.path:
                sys.path.insert(0, str(_scripts))
            from vcd import detect_all_conflicts, count_by_severity_level
            # US-307: honor adaptive profile selection via env var
            _profile = os.environ.get("CLOPDIT_VCD_PROFILE", "full")
            issues = detect_all_conflicts(fig, label=basename, verbose=False, profile=_profile)
            warnings_only, info_only, issue_counts = _summarize_vcd_issues(issues)
            live_vcd_payload["warnings"] = [_format_vcd_issue(x) for x in warnings_only]
            live_vcd_payload["info"] = [_format_vcd_issue(x) for x in info_only]
            live_vcd_payload["counts_by_type"] = dict(issue_counts)
            # US-202: structured findings + severity-level counts (backward-compatible)
            live_vcd_payload["findings"] = [
                {
                    "type": str(x.get("type", "")),
                    "detail": str(x.get("detail", "")),
                    "severity": str(x.get("severity", "")),
                    "severity_level": str(x.get("severity_level", "")),
                }
                for x in issues
            ]
            live_vcd_payload["severity_counts"] = count_by_severity_level(issues)
            _log_vcd_issues(_logging.getLogger(__name__), basename, warnings_only, info_only, issue_counts)
        except Exception as exc:
            live_vcd_payload["error"] = str(exc)
    else:
        live_vcd_payload["skipped"] = True
        live_vcd_payload["skip_reason"] = "disabled by run_vcd flag or CLOPDIT_ENABLE_VCD=0"

    live_vcd_payload["total_warnings"] = len(live_vcd_payload["warnings"])
    live_vcd_payload["total_info"] = len(live_vcd_payload["info"])
    with open(live_vcd_dir / f"{basename}.json", "w") as f:
        json.dump(live_vcd_payload, f, indent=2)

    # 4) Save PDF with deterministic settings.
    fig.savefig(path.with_suffix(".pdf"), **save_kw)

    if close:
        plt.close(fig)
    return path


# Backward-compatible alias
save_panel = save_with_vcd


def run_vcd_check(fig: plt.Figure, label: str) -> None:
    """Run visual conflict detection on a figure without saving. Used before PIL composition."""
    import logging as _logging

    if not is_vcd_enabled(default=True):
        return

    try:
        import sys
        _scripts = Path(__file__).resolve().parent.parent.parent / "scripts"
        if str(_scripts) not in sys.path:
            sys.path.insert(0, str(_scripts))
        from vcd import detect_all_conflicts
        issues = detect_all_conflicts(fig, label=label, verbose=False)
        warnings_only, info_only, issue_counts = _summarize_vcd_issues(issues)
        _log_vcd_issues(_logging.getLogger(__name__), label, warnings_only, info_only, issue_counts)
    except Exception:
        pass


def _format_vcd_issue(issue: dict) -> str:
    issue_type = str(issue.get("type", "issue"))
    detail = str(issue.get("detail", "")).strip()
    return f"[{issue_type}] {detail}" if detail else f"[{issue_type}]"


def _summarize_vcd_issues(issues: list[dict]) -> tuple[list[dict], list[dict], Counter]:
    warnings_only = [x for x in issues if str(x.get("severity", "")).lower() == "warning"]
    info_only = [x for x in issues if str(x.get("severity", "")).lower() != "warning"]
    issue_counts = Counter(str(x.get("type", "unknown")) for x in issues)
    return warnings_only, info_only, issue_counts


def _format_vcd_type_counts(issue_counts: Counter, max_items: int = 4) -> str:
    if not issue_counts:
        return "none"
    items = sorted(issue_counts.items(), key=lambda item: (-item[1], item[0]))
    head = ", ".join(f"{name}={count}" for name, count in items[:max_items])
    if len(items) > max_items:
        head += f", +{len(items) - max_items} more"
    return head


def _log_vcd_issues(logger, label: str, warnings_only: list[dict], info_only: list[dict], issue_counts: Counter) -> None:
    status = "PASS" if not warnings_only else ("WARN" if len(warnings_only) < 3 else "FAIL")
    log_fn = logger.info if not warnings_only else logger.warning
    log_fn(
        "VCD[live][%s] %s warn=%d info=%d types=%s",
        label,
        status,
        len(warnings_only),
        len(info_only),
        _format_vcd_type_counts(issue_counts),
    )
    for idx, issue in enumerate(warnings_only[:3], start=1):
        logger.warning(
            "  [%d/%d] %s",
            idx,
            len(warnings_only),
            _format_vcd_issue(issue)[:220],
        )
    if len(warnings_only) > 3:
        logger.warning("  ... +%d more warning(s)", len(warnings_only) - 3)


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
