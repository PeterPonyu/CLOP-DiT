"""Composition infrastructure for single-producer article figures.

This module defines the primitives that the single-producer migration plan
(`.omc/plans/single-producer-architecture-2026-04-20.md`) depends on:

1.  **Composite basename convention** — every merged article figure publishes
    a new PDF under `fig0X_composed.pdf` alongside the legacy per-slice PDFs
    (dual-publish during the revision window).
2.  **Composite VCD registry** — maps each composite's PDF basename to the
    list of legacy slice basenames it replaces. `test_composite_vcd_coverage`
    in `tests/test_vcd_checks.py` consumes this registry to enforce the
    cross-slice acceptance gate ``warn(composite) <= sum(warn(slices))``.
3.  **AE pixel-tolerance calibration** — `compute_ae_pixel_tolerance`
    shells out to ImageMagick ``compare -metric AE`` and returns the pixel
    count that differs between two PDFs rasterised at the same DPI. The
    calibration protocol is: run a ``fig0X → fig0X`` self-comparison
    against the current canonical PDF to measure the inherent rasteriser
    jitter, then pin ``AE_PIXEL_TOLERANCE = max(AE_self * 3, 500)`` so the
    per-migration acceptance check (plan §5.6 bullet 5) is grounded in the
    observed noise floor rather than a hand-picked threshold.

The registry is empty at Step 0 (plan pilot is Fig 8 at Step 1). Every
subsequent migration appends exactly one entry: ``{"fig0X_composed.pdf":
[<slice_1>.pdf, <slice_2>.pdf, ...]}``.

Public API:
    * ``COMPOSITE_BASENAME_FMT``       — format string, e.g. ``"fig0{n}_composed"``
    * ``composite_basename(n)``        — returns ``f"fig0{n}_composed"``
    * ``COMPOSITE_VCD_REGISTRY``       — ``dict[str, list[str]]``
    * ``AE_PIXEL_TOLERANCE``           — int, max allowed pixel drift
    * ``compute_ae_pixel_tolerance(a, b, dpi=150)`` — ImageMagick AE count
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------
# Composite basename convention
# --------------------------------------------------------------------------

COMPOSITE_BASENAME_FMT: str = "fig0{n}_composed"
"""Format string for composite PDF basenames. Used by migrated article
figures (Fig 3, 4, 7a+b, 8 per the plan) to emit a single-producer PDF
alongside the legacy per-slice PDFs during the revision window."""


def composite_basename(n: int) -> str:
    """Return the canonical composite basename for article figure ``n``.

    >>> composite_basename(8)
    'fig08_composed'
    >>> composite_basename(3)
    'fig03_composed'
    """
    if not isinstance(n, int) or n < 0 or n > 99:
        raise ValueError(f"article figure index must be 0..99, got {n!r}")
    return COMPOSITE_BASENAME_FMT.format(n=n)


# --------------------------------------------------------------------------
# Composite VCD registry (populated per-migration)
# --------------------------------------------------------------------------

COMPOSITE_VCD_REGISTRY: Dict[str, List[str]] = {
    # Step 1 pilot (plan §3 Step 1): Fig 8 composite replaces
    # fig08a_downstream_validation.pdf + fig08b_de_concordance.pdf. Produced
    # by src/visualization/fig08_composed.py alongside the legacy slices
    # (dual-publish during the revision window).
    "fig08_composed.pdf": [
        "fig08a_downstream_validation.pdf",
        "fig08b_de_concordance.pdf",
    ],
    # Step 2 (plan §3 Step 2): Fig 3 composite replaces
    # fig03a_metrics_summary.pdf + fig03b_per_type_fidelity.pdf +
    # fig03c_text_cell_alignment.pdf. Produced by
    # src/visualization/fig03_composed.py alongside the legacy slices
    # (dual-publish during the revision window).
    "fig03_composed.pdf": [
        "fig03a_metrics_summary.pdf",
        "fig03b_per_type_fidelity.pdf",
        "fig03c_text_cell_alignment.pdf",
    ],
    # Step 3 (plan §3 Step 3): Fig 4 composite replaces
    # fig04a_marker_genes.pdf + fig04b_expression_correlation.pdf. Produced
    # by src/visualization/fig04_composed.py alongside the legacy slices
    # (dual-publish during the revision window). Note the source-file naming:
    # fig08_markers.py is the article-Fig-4-A source and fig09_expression_corr.py
    # is the article-Fig-4-B source (source numbering diverged pre-revision;
    # documented in plan §3 Step 3).
    "fig04_composed.pdf": [
        "fig04a_marker_genes.pdf",
        "fig04b_expression_correlation.pdf",
    ],
}
"""Maps composite PDF basename -> list of legacy slice PDF basenames.

Populated incrementally:
    Step 1 (Fig 8):      {"fig08_composed.pdf": ["fig08a_downstream_validation.pdf",
                                                  "fig08b_de_concordance.pdf"]}
    Step 2 (Fig 3):      {... , "fig03_composed.pdf": ["fig03a_metrics_summary.pdf",
                                                        "fig03b_per_type_fidelity.pdf",
                                                        "fig03c_text_cell_alignment.pdf"]}
    Step 3 (Fig 4):      adds "fig04_composed.pdf" -> [fig04a, fig04b]
    Step 4 (Fig 7a+b):   adds "fig07_composed.pdf" -> [fig07a, fig07b]
                         (fig07c_benchmark.pdf stays separate — Option I-a exemption)

Consumer: ``tests/test_vcd_checks.py::test_composite_vcd_coverage`` iterates
every (composite, slices) pair and asserts
``len(current[composite]["warnings"]) <= sum(len(current[s]["warnings"]) for s in slices)``.

Vacuously green while the registry is empty (Step 0 state)."""


# --------------------------------------------------------------------------
# AE pixel-tolerance calibration
# --------------------------------------------------------------------------

AE_PIXEL_TOLERANCE: int = 500
"""Maximum pixel count that may differ between a composite's slice-crop and
the corresponding legacy per-slice PDF (plan §5.6 bullet 5 acceptance gate).

Default ``500`` is a conservative floor for anti-aliased Nature/Cell-scale
figures at 150 DPI. For per-figure calibration, run
``compute_ae_pixel_tolerance(fig0X.pdf, fig0X.pdf)`` against the current
canonical PDF to observe the rasteriser's inherent jitter, then pin
``AE_PIXEL_TOLERANCE = max(AE_self * 3, 500)`` (3x the self-compare noise
floor, with the 500 pixel minimum for figures where ``AE_self`` is zero)."""


def compute_ae_pixel_tolerance(pdf_a: Path | str,
                                pdf_b: Path | str,
                                dpi: int = 150) -> int:
    """Return the ImageMagick ``-metric AE`` pixel-count between two PDFs.

    Rasterises both PDFs at ``dpi`` and compares pixel-by-pixel. Used both
    for calibration (``fig0X → fig0X`` self-comparison to measure noise) and
    for per-migration acceptance checks (composite slice-crop vs legacy
    per-slice PDF).

    Returns the integer pixel count from ``compare -metric AE``. Raises
    ``RuntimeError`` if the ImageMagick/ghostscript toolchain is unavailable
    or if ``compare`` produces unparseable output.
    """
    pdf_a, pdf_b = Path(pdf_a), Path(pdf_b)
    for p in (pdf_a, pdf_b):
        if not p.exists():
            raise FileNotFoundError(p)

    # ``compare -metric AE`` writes the pixel count to stderr and exits
    # non-zero when the inputs differ. ``-fuzz 0%`` keeps the comparison
    # strict; the caller is expected to have pinned DPI + anti-aliasing on
    # both sides. ``NULL:`` discards the diff image.
    cmd = [
        "compare",
        "-metric", "AE",
        "-fuzz", "0%",
        "-density", str(dpi),
        f"{pdf_a}[0]",
        f"{pdf_b}[0]",
        "NULL:",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # compare emits the AE count on stderr
    stderr = proc.stderr.strip()
    if not stderr:
        raise RuntimeError(
            f"`compare` returned empty stderr (stdout={proc.stdout!r}, "
            f"rc={proc.returncode}); check that ImageMagick + ghostscript "
            f"are installed and that both PDFs are readable."
        )
    # Typical output is just the integer count, e.g. "12345". Older
    # ImageMagick may append "@ x,y" coordinates of the first differing
    # pixel; take the first whitespace-separated token.
    first_token = stderr.split()[0]
    try:
        return int(float(first_token))
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse AE pixel count from `compare` output: {stderr!r}"
        ) from exc


__all__ = [
    "COMPOSITE_BASENAME_FMT",
    "composite_basename",
    "COMPOSITE_VCD_REGISTRY",
    "AE_PIXEL_TOLERANCE",
    "compute_ae_pixel_tolerance",
]
