"""Explicit-enumeration import smoke test for src/visualization/ subdirs.

Plan `.omc/plans/viz-cleanup-2026-04-20.md` §1.4 Pre-mortem Scenario 2
mandated this check to catch silent-swallow import failures that slip past
the lazy-import pattern in results_visualizer.py (where relocations can
silently break producers whose PDFs are expected regeneration targets).
"""

import importlib
import pkgutil

import pytest


@pytest.mark.parametrize("subpackage_name", ["experimental", "supplementary"])
def test_subpackage_modules_importable(subpackage_name: str) -> None:
    """Every module under `src/visualization/<subpackage_name>/` must import cleanly.

    Parameterized over the two subdirs created by Track A of the viz-cleanup
    plan. Any future subdir relocation should add its name to the parametrize
    list so the same guard applies.
    """
    pkg = importlib.import_module(f"src.visualization.{subpackage_name}")
    failures = []
    for _, mod_name, _ in pkgutil.iter_modules(pkg.__path__):
        qual = f"{pkg.__name__}.{mod_name}"
        try:
            importlib.import_module(qual)
        except Exception as exc:  # pragma: no cover — exercised on regression
            failures.append(f"{qual}: {type(exc).__name__}: {exc}")
    assert not failures, (
        f"{subpackage_name}/ modules must import without error. "
        f"{len(failures)} failure(s):\n  " + "\n  ".join(failures)
    )
