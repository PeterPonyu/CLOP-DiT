"""
Shared I/O helpers for panel modules.

Provides save_to_dir() so panel modules can persist figures to an output_dir
with optional custom save_panel_fn (e.g. for VCD-integrated saving) without
duplicating path handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt


def save_to_dir(
    fig: plt.Figure,
    basename: str,
    output_dir: str | Path,
    dpi: int = 300,
    save_panel_fn: Optional[Callable[..., Path]] = None,
) -> Path:
    """Persist *fig* to output_dir/basename.png (and .pdf) via save_panel_fn or default save_panel.

    Parameters
    ----------
    fig : matplotlib Figure to save
    basename : stem for filename (no extension)
    output_dir : directory to write into (created if needed)
    dpi : resolution
    save_panel_fn : optional callable(fig, path, dpi) -> Path; if None, uses style.save_panel

    Returns
    -------
    Path to the saved PNG file.
    """
    from .style import save_panel as _default_save_panel

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{basename}.png"
    if save_panel_fn is not None:
        return save_panel_fn(fig, path, dpi)
    return _default_save_panel(fig, path, dpi)
