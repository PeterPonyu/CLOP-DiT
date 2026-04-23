"""scivcd — scientific visualization conflict detection for matplotlib.

Public surface for the rewrite of the legacy ``scripts/vcd`` pipeline.
The package exposes three layers:

* Core types and registry (re-exported from :mod:`scivcd.core`).
* Exemption helpers :func:`exempt`, :func:`ignore`, :func:`is_exempt`
  for silencing checks on a per-artist basis.
* The programmatic API :func:`check`, :func:`install`, :func:`uninstall`
  and the :class:`Report` dataclass.

Typical usage::

    import scivcd
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "note", **scivcd.ignore("non_panel_bold_text"))

    report = scivcd.check(fig)
    if report:
        print(report.summary())
"""

from __future__ import annotations

__version__ = "0.1.0"

from scivcd.core import (
    Category,
    CheckSpec,
    Finding,
    ScivcdConfig,
    Severity,
    Stage,
    iter_checks,
    register,
    unregister,
)
from scivcd.exemptions import exempt, ignore, is_exempt
from scivcd.api import Report, audit_export, check, install, uninstall
from scivcd import checks as _checks  # noqa: F401  # side-effect import: register built-in checks

__all__ = [
    "__version__",
    # core enums + dataclasses
    "Severity",
    "Category",
    "Stage",
    "Finding",
    "CheckSpec",
    "ScivcdConfig",
    # registry
    "register",
    "unregister",
    "iter_checks",
    # exemptions
    "exempt",
    "ignore",
    "is_exempt",
    # api
    "check",
    "audit_export",
    "install",
    "uninstall",
    "Report",
]
