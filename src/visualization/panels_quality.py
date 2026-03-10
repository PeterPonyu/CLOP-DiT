"""panels_quality.py — Backward-compat shim; re-exports from new fig{NN} modules."""
from __future__ import annotations

from .fig04_embedding import plot_embedding_space_merged, plot_real_vs_generated, plot_fidelity_and_alignment_merged
from .fig05_metrics import plot_diversity_distributions_violin, plot_metrics_summary
from .fig06_fidelity import plot_per_type_generation
from .fig07_alignment import plot_text_cell_heatmap

__all__ = [
    "plot_diversity_distributions_violin",
    "plot_embedding_space_merged",
    "plot_fidelity_and_alignment_merged",
    "plot_metrics_summary",
    "plot_per_type_generation",
    "plot_real_vs_generated",
    "plot_text_cell_heatmap",
]
