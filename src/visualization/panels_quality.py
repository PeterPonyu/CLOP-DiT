"""
panels_quality.py — Facade re-exporting quality panels (D, E, F, G) and merged figures.

Implementation lives in:
  - panels_metrics: plot_metrics_summary (D), plot_diversity_distributions_violin
  - panels_umap_quality: plot_real_vs_generated (E)
  - panels_heatmaps: plot_text_cell_heatmap (F), plot_per_type_generation (G)
  - panels_merged: plot_embedding_space_merged, plot_fidelity_and_alignment_merged
"""

from __future__ import annotations

from .panels_heatmaps import (
    plot_per_type_generation,
    plot_text_cell_heatmap,
)
from .panels_merged import (
    plot_embedding_space_merged,
    plot_fidelity_and_alignment_merged,
)
from .panels_metrics import (
    plot_diversity_distributions_violin,
    plot_metrics_summary,
)
from .panels_umap_quality import plot_real_vs_generated

__all__ = [
    "plot_diversity_distributions_violin",
    "plot_embedding_space_merged",
    "plot_fidelity_and_alignment_merged",
    "plot_metrics_summary",
    "plot_per_type_generation",
    "plot_real_vs_generated",
    "plot_text_cell_heatmap",
]
