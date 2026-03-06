"""Tests for article figure manifest and delivery (src.visualization.article_delivery)."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestArticleFigureManifest:
    """Canonical list of 15 article figures."""

    def test_manifest_length(self):
        from src.visualization.article_delivery import ARTICLE_FIGURE_BASENAMES
        assert len(ARTICLE_FIGURE_BASENAMES) == 15

    def test_manifest_contains_expected_basenames(self):
        from src.visualization.article_delivery import ARTICLE_FIGURE_BASENAMES
        expected = {
            "fig_architecture",
            "fig_training_dynamics",
            "fig_embedding_space",
            "fig_fidelity_alignment",
            "fig_diversity_tradeoff",
            "fig_downstream_pq",
            "panel_d_metrics_summary",
            "panel_n_marker_gene_comparison",
            "panel_j_diversity_diagnostics",
            "panel_s_benchmark",
            "panel_r_de_concordance",
        }
        for name in expected:
            assert name in ARTICLE_FIGURE_BASENAMES, f"Missing basename: {name}"
        assert len(set(ARTICLE_FIGURE_BASENAMES)) == len(ARTICLE_FIGURE_BASENAMES), "Duplicate basenames"


class TestDeliverFigures:
    """deliver_figures() with temp dirs and dummy PDFs."""

    def test_check_only_all_present(self, tmp_path):
        from src.visualization.article_delivery import (
            ARTICLE_FIGURE_BASENAMES,
            deliver_figures,
        )
        for base in ARTICLE_FIGURE_BASENAMES:
            (tmp_path / f"{base}.pdf").write_bytes(b"%PDF-1.0 dummy\n")
        ok = deliver_figures(tmp_path, tmp_path / "out", symlink=True, check_only=True)
        assert ok is True
        assert not (tmp_path / "out").exists()

    def test_check_only_missing_returns_false(self, tmp_path):
        from src.visualization.article_delivery import (
            ARTICLE_FIGURE_BASENAMES,
            deliver_figures,
        )
        # Create only 14 of 15
        for base in ARTICLE_FIGURE_BASENAMES[:-1]:
            (tmp_path / f"{base}.pdf").write_bytes(b"%PDF-1.0 dummy\n")
        ok = deliver_figures(tmp_path, tmp_path / "out", symlink=True, check_only=True)
        assert ok is False

    def test_deliver_symlink(self, tmp_path):
        from src.visualization.article_delivery import (
            ARTICLE_FIGURE_BASENAMES,
            deliver_figures,
        )
        for base in ARTICLE_FIGURE_BASENAMES:
            (tmp_path / f"{base}.pdf").write_bytes(b"%PDF-1.0 dummy\n")
        target = tmp_path / "article_figures"
        ok = deliver_figures(tmp_path, target, symlink=True, check_only=False)
        assert ok is True
        assert target.is_dir()
        for base in ARTICLE_FIGURE_BASENAMES:
            link = target / f"{base}.pdf"
            assert link.exists(), f"Missing link: {link}"
            assert link.is_symlink(), f"Not a symlink: {link}"
            assert link.resolve().exists(), f"Broken symlink: {link}"

    def test_deliver_copy(self, tmp_path):
        from src.visualization.article_delivery import (
            ARTICLE_FIGURE_BASENAMES,
            deliver_figures,
        )
        for base in ARTICLE_FIGURE_BASENAMES:
            (tmp_path / f"{base}.pdf").write_bytes(b"%PDF-1.0 dummy\n")
        target = tmp_path / "article_figures"
        ok = deliver_figures(tmp_path, target, symlink=False, check_only=False)
        assert ok is True
        for base in ARTICLE_FIGURE_BASENAMES:
            f = target / f"{base}.pdf"
            assert f.is_file(), f"Missing file: {f}"
            assert not f.is_symlink(), f"Expected copy, got symlink: {f}"
            assert f.read_bytes() == b"%PDF-1.0 dummy\n"