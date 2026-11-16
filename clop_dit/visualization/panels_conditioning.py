"""
panels_conditioning.py — Backward-compatibility shim.

The implementations have moved to dedicated per-figure modules:
  * plot_panel_m  ->  fig11_conditioning.py   (Article Fig 11)
  * plot_panel_l  ->  fig13_noise_tradeoff.py (Article Fig 13)

This module re-exports both functions so existing call-sites keep working.
"""

from .fig11_conditioning import plot_panel_m  # noqa: F401
from .fig13_noise_tradeoff import plot_panel_l  # noqa: F401

__all__ = ["plot_panel_l", "plot_panel_m"]
