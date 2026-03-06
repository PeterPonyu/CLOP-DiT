"""
full_pipeline_figures.py — DEPRECATED (kept for backward import compatibility).

This module intentionally does not generate or save any figures.
The legacy 5-figure outputs (`fig1_*`–`fig5_*`) were retired to avoid
filename conflicts with canonical article/report figures.

Use `ResultsVisualizer.generate_full_report()` instead.
"""

def generate_all_figures(*args, **kwargs):
    """No-op stub — use ResultsVisualizer.generate_full_report() instead."""
    import logging
    logging.getLogger(__name__).warning(
        "full_pipeline_figures.generate_all_figures() is deprecated; "
        "legacy fig1_*–fig5_* outputs are disabled; use "
        "ResultsVisualizer.generate_full_report() instead."
    )
    return []
