"""
report.py — Orchestrates full report generation across all panel modules.

Used by the ResultsVisualizer façade to generate the combined multi-page PDF.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from . import baseline_panels, downstream_panels

logger = logging.getLogger(__name__)


def combine_panels_pdf(
    saved_paths: List[Path],
    output_dir: Path,
    filename: str = "clop_dit_full_report.pdf",
) -> Path:
    """Combine all saved panel PNGs into a single multi-page PDF."""
    from matplotlib.backends.backend_pdf import PdfPages

    combined_path = output_dir / filename
    existing = [p for p in saved_paths if p.exists()]
    if not existing:
        logger.warning("No panels to combine")
        return combined_path

    with PdfPages(combined_path) as pdf:
        for panel_path in existing:
            png_path = panel_path.with_suffix(".png")
            if not png_path.exists():
                continue
            img = plt.imread(str(png_path))
            fig_tmp, ax_tmp = plt.subplots(
                figsize=(img.shape[1] / 100, img.shape[0] / 100)
            )
            ax_tmp.imshow(img)
            ax_tmp.axis("off")
            pdf.savefig(fig_tmp, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig_tmp)

    logger.info(f"Combined report → {combined_path}")
    return combined_path
