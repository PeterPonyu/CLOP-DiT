"""Regression checks for the SciVCD sandbox article-figure policy.

The sandbox lives under the gitignored `experiments/scivcd-lifetime/` workspace,
so this test module skips cleanly on any environment (CI, fresh clone) that
does not have the sandbox checked out locally.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
SANDBOX_POLICY_PATH = (
    REPO_ROOT
    / "experiments"
    / "scivcd-lifetime"
    / "src"
    / "visualization"
    / "figure_asset_policy.py"
)

pytestmark = pytest.mark.skipif(
    not SANDBOX_POLICY_PATH.exists(),
    reason="scivcd-lifetime sandbox not present (experiments/ is gitignored)",
)
SANDBOX_POLICY_SHIM_PATH = (
    REPO_ROOT
    / "experiments"
    / "scivcd-lifetime"
    / "src"
    / "visualization"
    / "article_figure_policy.py"
)
SANDBOX_COMPOSE_PATH = (
    REPO_ROOT
    / "experiments"
    / "scivcd-lifetime"
    / "src"
    / "visualization"
    / "composed"
    / "compose_figures.py"
)
SANDBOX_UNIFIED_RUNNER_PATH = (
    REPO_ROOT
    / "experiments"
    / "scivcd-lifetime"
    / "src"
    / "visualization"
    / "unified"
    / "run_unified.py"
)
SANDBOX_UNIFIED_DIR = SANDBOX_UNIFIED_RUNNER_PATH.parent
SANDBOX_RUN_EXPERIMENT_PATH = (
    REPO_ROOT
    / "experiments"
    / "scivcd-lifetime"
    / "run_experiment.py"
)
SANDBOX_README_PATH = (
    REPO_ROOT
    / "experiments"
    / "scivcd-lifetime"
    / "README.md"
)


def _load_sandbox_policy():
    spec = importlib.util.spec_from_file_location(
        "sandbox_article_figure_policy",
        SANDBOX_POLICY_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load sandbox policy from {SANDBOX_POLICY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestSandboxArticleFigurePolicy:
    def test_policy_components_match_main_article_manifest(self):
        from src.visualization.article_delivery import ARTICLE_FIGURE_BASENAMES

        policy = _load_sandbox_policy()
        assert policy.flattened_article_component_basenames() == tuple(
            ARTICLE_FIGURE_BASENAMES
        )

    def test_unified_producer_specs_have_matching_modules_and_builders(self):
        policy = _load_sandbox_policy()
        for output_stem, module_name, builder_name in policy.unified_producer_specs():
            module_path = SANDBOX_UNIFIED_DIR / f"{module_name}.py"
            assert module_path.exists(), f"Missing unified module for {output_stem}: {module_path}"
            text = module_path.read_text()
            assert f"def {builder_name}(" in text, (
                f"Missing builder {builder_name} in {module_path}"
            )

    def test_compose_and_unified_runners_use_shared_policy_helpers(self):
        compose_text = SANDBOX_COMPOSE_PATH.read_text()
        unified_text = SANDBOX_UNIFIED_RUNNER_PATH.read_text()
        shim_text = SANDBOX_POLICY_SHIM_PATH.read_text()

        assert "raster_composition_policy" in compose_text
        assert "unified_producer_specs" in unified_text
        assert "figure_asset_policy" in shim_text

    def test_run_experiment_exposes_unified_modes(self):
        text = SANDBOX_RUN_EXPERIMENT_PATH.read_text()
        assert "unified" in text
        assert "unified-lifecycle" in text
        assert "cmd_unified" in text
        assert "cmd_unified_lifecycle" in text

