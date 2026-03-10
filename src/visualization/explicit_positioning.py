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
    legend_ax.set_axis_off()
    legend_ax.patch.set_alpha(0.0)
    return legend_ax