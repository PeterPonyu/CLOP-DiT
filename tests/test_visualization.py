"""Tests for visualization module imports and basic functionality."""

import pytest
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        assert SUPTITLE_Y == 0.96

    def test_panel_label_defaults_are_publication_scale(self):
        from src.visualization.style import PANEL_LABEL_FONT_SIZE
        assert PANEL_LABEL_FONT_SIZE == 20


class TestPanelLabelHelper:
    """Verify the shared panel-label helper contract."""

    def test_add_panel_label_renders_uppercase_gid_and_defaults(self):
        from src.visualization.style import PANEL_LABEL_FONT_SIZE, add_panel_label

        fig, ax = plt.subplots()
        add_panel_label(ax, "a")
        fig.canvas.draw()

        assert len(ax.texts) == 1
        label = ax.texts[0]
        assert label.get_text() == "A"
        assert label.get_gid() == "panel_label:A"
        assert label.get_fontsize() == PANEL_LABEL_FONT_SIZE
        assert label.get_position() == (-0.12, 1.08)
        assert label.get_clip_on() is False
        plt.close(fig)


class TestMigratedFigureLabels:
    """Stage 2 gate: figures that migrated to ``safe_tick_labels`` must not
    emit the U+2026 HORIZONTAL ELLIPSIS codepoint anywhere in rendered text.

    Enabled via ``CLOPDIT_ENFORCE_ELLIPSIS_BAN=<space-separated basenames>``.
    Without the env var the test is skipped so Stage 0 can land before any
    figure is migrated. Requires ``pdfplumber``; skipped when absent.
    """

    def test_no_ellipsis_in_migrated_figure_labels(self):
        import os

        migrated = os.environ.get("CLOPDIT_ENFORCE_ELLIPSIS_BAN", "").split()
        if not migrated:
            pytest.skip(
                "set CLOPDIT_ENFORCE_ELLIPSIS_BAN to a space-separated list of "
                "figure basenames (without .pdf) to enforce the Stage 2 gate"
            )

        try:
            import pdfplumber
        except ImportError:
            pytest.skip("pdfplumber not installed")

        from src.utils.paths import FIG_DIR

        offenders: list[tuple[str, str]] = []
        for base in migrated:
            pdf_path = FIG_DIR / f"{base}.pdf"
            if not pdf_path.exists():
                offenders.append((base, "missing PDF"))
                continue
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if "\u2026" in text:
                        offenders.append((base, "contains U+2026"))
                        break
        assert not offenders, (
            "Migrated figures must not contain ellipsis characters in labels: "
            + "; ".join(f"{b}: {why}" for b, why in offenders)
        )
