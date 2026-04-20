"""Explicit-enumeration import smoke test for src/visualization/ subdirs.

Plan `.omc/plans/viz-cleanup-2026-04-20.md` §1.4 Pre-mortem Scenario 2
mandated this check to catch silent-swallow import failures that slip past
the lazy-import pattern in results_visualizer.py (where relocations can
silently break producers whose PDFs are expected regeneration targets).
"""

import importlib
import pkgutil

import pytest


def _iter_package_modules(package):
    return [
        f"{package.__name__}.{mod_name}"
        for _, mod_name, _ in pkgutil.iter_modules(package.__path__)
    ]


class TestExperimentalImports:
    """Every module in src/visualization/experimental/ must import cleanly."""

    def test_all_experimental_modules_importable(self):
        import src.visualization.experimental as exp_pkg
        failures = []
        for mod_name in _iter_package_modules(exp_pkg):
            try:
                importlib.import_module(mod_name)
            except Exception as exc:
                failures.append(f"{mod_name}: {type(exc).__name__}: {exc}")
        assert not failures, (
            "Experimental subdir modules must import without error. "
            f"{len(failures)} failure(s):\n  " + "\n  ".join(failures)
        )


class TestSupplementaryImports:
    """Every module in src/visualization/supplementary/ must import cleanly."""

    def test_all_supplementary_modules_importable(self):
        import src.visualization.supplementary as sup_pkg
        failures = []
        for mod_name in _iter_package_modules(sup_pkg):
            try:
                importlib.import_module(mod_name)
            except Exception as exc:
                failures.append(f"{mod_name}: {type(exc).__name__}: {exc}")
        assert not failures, (
            "Supplementary subdir modules must import without error. "
            f"{len(failures)} failure(s):\n  " + "\n  ".join(failures)
        )
