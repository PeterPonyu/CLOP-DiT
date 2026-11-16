"""Direct artist and axes positioning helpers for publication figures."""

from __future__ import annotations

from typing import Sequence


def get_axes_rect(ax) -> tuple[float, float, float, float]:
    """Return the axes rectangle in figure coordinates."""
    pos = ax.get_position()
    return float(pos.x0), float(pos.y0), float(pos.width), float(pos.height)


def set_axes_rect(ax, rect: Sequence[float]):
    """Set the axes rectangle in figure coordinates."""
    if len(rect) != 4:
        raise ValueError(f"axes rect must contain 4 floats, got {rect!r}")
    ax.set_position(tuple(map(float, rect)))
    return ax


def move_axes(ax, *, dx: float = 0.0, dy: float = 0.0, dw: float = 0.0, dh: float = 0.0):
    """Move and/or resize an axes in figure coordinates."""
    left, bottom, width, height = get_axes_rect(ax)
    return set_axes_rect(ax, (left + dx, bottom + dy, width + dw, height + dh))


def rect_next_to_axes(
    ax,
    *,
    side: str = "right",
    width: float = 0.012,
    height: float = 0.25,
    pad: float = 0.01,
    align: str = "center",
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> tuple[float, float, float, float]:
    """Compute a rectangle adjacent to an existing axes.

    Parameters are interpreted in figure coordinates.
    """
    left, bottom, ax_width, ax_height = get_axes_rect(ax)
    if side not in {"right", "left", "top", "bottom"}:
        raise ValueError(f"unsupported side {side!r}")

    if side in {"right", "left"}:
        cbar_height = min(height, ax_height)
        if align == "top":
            y = bottom + ax_height - cbar_height
        elif align == "bottom":
            y = bottom
        else:
            y = bottom + (ax_height - cbar_height) / 2.0
        x = left + ax_width + pad if side == "right" else left - pad - width
        return x + x_offset, y + y_offset, width, cbar_height

    cbar_width = min(width, ax_width)
    if align == "left":
        x = left
    elif align == "right":
        x = left + ax_width - cbar_width
    else:
        x = left + (ax_width - cbar_width) / 2.0
    y = bottom + ax_height + pad if side == "top" else bottom - pad - height
    return x + x_offset, y + y_offset, cbar_width, height


def add_axes_next_to(fig, ax, **kwargs):
    """Create a new axes adjacent to ``ax`` using :func:`rect_next_to_axes`."""
    return fig.add_axes(rect_next_to_axes(ax, **kwargs))


def add_shared_legend_axes(fig, rect: Sequence[float]):
    """Create an invisible axes dedicated to shared legends or annotations."""
    legend_ax = fig.add_axes(tuple(map(float, rect)))
    legend_ax._is_legend_cell = True
    legend_ax.set_axis_off()
    legend_ax.patch.set_alpha(0.0)
    return legend_ax


def union_axes_rect(axes) -> tuple[float, float, float, float]:
    """Return the union rectangle of multiple axes in figure coordinates."""
    rects = [get_axes_rect(ax) for ax in axes]
    left = min(r[0] for r in rects)
    bottom = min(r[1] for r in rects)
    right = max(r[0] + r[2] for r in rects)
    top = max(r[1] + r[3] for r in rects)
    return left, bottom, right - left, top - bottom


def layout_axes_row(
    axes,
    *,
    widths: Sequence[float] | None = None,
    gaps: float | Sequence[float] = 0.02,
    rect: Sequence[float] | None = None,
):
    """Lay out a sequence of axes in one explicit horizontal row.

    Parameters are interpreted in figure coordinates. When ``rect`` is omitted,
    the current union rectangle of ``axes`` is reused.
    """
    axes = list(axes)
    if not axes:
        return []

    if widths is None:
        widths = [1.0] * len(axes)
    if len(widths) != len(axes):
        raise ValueError("widths must match number of axes")

    if isinstance(gaps, (int, float)):
        gaps = [float(gaps)] * max(len(axes) - 1, 0)
    else:
        gaps = list(map(float, gaps))
    if len(gaps) != max(len(axes) - 1, 0):
        raise ValueError("gaps must contain len(axes) - 1 values")

    left, bottom, width, height = union_axes_rect(axes) if rect is None else tuple(map(float, rect))
    total_gap = sum(gaps)
    usable_width = width - total_gap
    scale = usable_width / sum(map(float, widths))

    x = left
    rects = []
    for idx, (ax, rel_width) in enumerate(zip(axes, widths)):
        ax_width = float(rel_width) * scale
        ax_rect = (x, bottom, ax_width, height)
        set_axes_rect(ax, ax_rect)
        rects.append(ax_rect)
        if idx < len(gaps):
            x += ax_width + gaps[idx]
    return rects