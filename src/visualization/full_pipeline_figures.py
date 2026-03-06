"""
full_pipeline_figures.py — DEPRECATED.

The legacy 5-figure pipeline (fig1_*–fig5_*) has been superseded by
the panel-based system in results_visualizer.py (Panels A–S → Figures 1–15).
"""

def generate_all_figures(*args, **kwargs):
    """No-op stub — use ResultsVisualizer.generate_full_report() instead."""
    import logging
    logging.getLogger(__name__).warning(
        "full_pipeline_figures.generate_all_figures() is deprecated; "
        "use ResultsVisualizer.generate_full_report() instead."
    )
    return []
