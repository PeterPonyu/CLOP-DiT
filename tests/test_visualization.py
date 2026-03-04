"""Tests for visualization module imports and basic functionality."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVisualizationImports:
    """Verify all visualization modules import without error."""

    def test_import_style(self):
        from src.visualization.style import apply_style, save_with_vcd, style_axes

    def test_import_panels_training(self):
        from src.visualization.panels_training import (
            plot_clop_training,
            plot_dit_training,
            plot_training_dynamics_combined,
        )

    def test_import_panels_quality(self):
        from src.visualization.panels_quality import (
            plot_metrics_summary,
            plot_real_vs_generated,
            plot_text_cell_heatmap,
            plot_per_type_generation,
            plot_embedding_space_merged,
            plot_fidelity_and_alignment_merged,
        )

    def test_import_panels_expression(self):
        from src.visualization.panels_expression import (
            plot_expression_correlation,
            plot_expression_analysis,
            plot_marker_gene_comparison,
        )

    def test_import_baseline_panels(self):
        from src.visualization.baseline_panels import plot_baseline_comparison

    def test_import_downstream_panels(self):
        from src.visualization.downstream_panels import (
            plot_clustering_panel,
            plot_classifier_panel,
            plot_de_concordance_panel,
            plot_clustering_and_classifier_merged,
        )

    def test_import_results_visualizer(self):
        from src.visualization.results_visualizer import ResultsVisualizer


class TestPathsModule:
    """Verify centralized paths resolve correctly."""

    def test_import_paths(self):
        from src.utils.paths import (
            PROJECT_ROOT, CACHE_DIR, RESULTS_DIR,
            FIG_DIR, CHECKPOINT_DIR, CONFIG_DIR,
        )

    def test_project_root_exists(self):
        from src.utils.paths import PROJECT_ROOT
        assert PROJECT_ROOT.exists()
        assert (PROJECT_ROOT / "README.md").exists()

    def test_config_dir_has_yamls(self):
        from src.utils.paths import CONFIG_DIR
        yamls = list(CONFIG_DIR.glob("*.yaml"))
        assert len(yamls) >= 2, "Expected at least clop and dit configs"


class TestStyleConstants:
    """Verify style module constants are set correctly."""

    def test_legend_frameon_false(self):
        from src.visualization.style import VIS_STYLE
        assert VIS_STYLE.get("legend.frameon") is False

    def test_suptitle_y(self):
        from src.visualization.style import SUPTITLE_Y
        assert SUPTITLE_Y == 0.98
