"""
panels_clustering.py — Backward-compatibility shim.

Panel P implementation has moved to :mod:`fig17_downstream`.
"""

from .fig17_downstream import plot_clustering_panel  # noqa: F401

__all__ = ["plot_clustering_panel"]
