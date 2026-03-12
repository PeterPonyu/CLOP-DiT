"""Tests for article figure manifest and delivery (src.visualization.article_delivery)."""

import re
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestArticleFigureManifest:
    """Canonical list of 20 article figures."""

    @staticmethod
    def _article_tex_basenames() -> set[str]:
        tex_path = Path(__file__).parent.parent / "articles" / "clop_dit_biology.tex"
        tex = tex_path.read_text()
        matches = re.findall(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\.pdf\}", tex)
        return set(matches)

    def test_manifest_length(self):
        from src.visualization.article_delivery import ARTICLE_FIGURE_BASENAMES
        assert len(ARTICLE_FIGURE_BASENAMES) == 20

    def test_manifest_contains_expected_basenames(self):
        from src.visualization.article_delivery import ARTICLE_FIGURE_BASENAMES
        expected = {
            "fig01_architecture",
            "fig02_evaluation_pipeline",
            "fig03_training_dynamics",
            "fig04_embedding_space",
            "fig05_metrics_summary",
            "fig06_per_type_fidelity",
            "fig07_text_cell_alignment",
            "fig08_marker_genes",
            "fig09_expression_correlation",
            "fig10_expression_analysis",
            "fig11_conditioning_umap",
            "fig12_diversity_diagnostics",
            "fig13_noise_tradeoff",
            "fig14_expression_diversity",
            "fig15_baseline_comparison",
            "fig16_benchmark",
            "fig17_downstream_pq",
            "fig18_de_concordance",
            "fig19_variance_matching_pilot",
            "fig20_gene_gene_correlation",
        }
        for name in expected:
            assert name in ARTICLE_FIGURE_BASENAMES, f"Missing basename: {name}"
        assert len(set(ARTICLE_FIGURE_BASENAMES)) == len(ARTICLE_FIGURE_BASENAMES), "Duplicate basenames"

    def test_manifest_matches_article_tex(self):
        from src.visualization.article_delivery import ARTICLE_FIGURE_BASENAMES

        manifest = set(ARTICLE_FIGURE_BASENAMES)
        tex_basenames = self._article_tex_basenames()
        assert manifest == tex_basenames, (
            "Article manifest and LaTeX includes diverged. "
            f"Manifest-only: {sorted(manifest - tex_basenames)}; "
            f"TeX-only: {sorted(tex_basenames - manifest)}"
        )


class TestDeliverFigures:
    """deliver_figures() with temp dirs and dummy JPEG/PDF figure pairs."""

    @staticmethod
    def _write_dummy_pair(root: Path, base: str) -> None:
        (root / f"{base}.pdf").write_bytes(b"%PDF-1.0 dummy\n")

    def test_check_only_all_present(self, tmp_path):
        from src.visualization.article_delivery import (
            _SOURCE_BASENAMES,
            deliver_figures,
        )
        for base in _SOURCE_BASENAMES:
            self._write_dummy_pair(tmp_path, base)
        ok = deliver_figures(tmp_path, tmp_path / "out", symlink=False, check_only=True)
        assert ok is True
        assert not (tmp_path / "out").exists()

    def test_check_only_missing_returns_false(self, tmp_path):
        from src.visualization.article_delivery import (
            _SOURCE_BASENAMES,
            deliver_figures,
        )
        # Create only 19 of 20 PDF figures
        for base in _SOURCE_BASENAMES[:-1]:
            self._write_dummy_pair(tmp_path, base)
        ok = deliver_figures(tmp_path, tmp_path / "out", symlink=False, check_only=True)
        assert ok is False

    def test_deliver_symlink(self, tmp_path):
        from src.visualization.article_delivery import (
            ARTICLE_FIGURE_BASENAMES,
            _SOURCE_BASENAMES,
            deliver_figures,
        )
        for base in _SOURCE_BASENAMES:
            self._write_dummy_pair(tmp_path, base)
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
            _SOURCE_BASENAMES,
            deliver_figures,
        )
        for base in _SOURCE_BASENAMES:
            self._write_dummy_pair(tmp_path, base)
        target = tmp_path / "article_figures"
        ok = deliver_figures(tmp_path, target, symlink=False, check_only=False)
        assert ok is True
        for base in ARTICLE_FIGURE_BASENAMES:
            pdf_file = target / f"{base}.pdf"
            assert pdf_file.is_file(), f"Missing file: {pdf_file}"
            assert not pdf_file.is_symlink(), f"Expected copy, got symlink: {pdf_file}"
            assert pdf_file.read_bytes() == b"%PDF-1.0 dummy\n"


class TestArticlePresentationPolicy:
    """Presentation policy: article-facing panels must not put statistics in legend titles."""

    # Patterns that indicate a statistic value (policy: put these in caption or add_stat_box, not legend title)
    # Descriptive phrases like "95% Bootstrap CI Comparison" are allowed; "CI = [0.1, 0.2]" or "r = 0.95" are not.
    STAT_IN_LEGEND_TITLE = re.compile(
        r"\b(r|p|R)\s*=\s*|"
        r"Sign\s*=\s*|"
        r"CV\s*corr\s*=|"
        r"CI\s*[=:]|"
        r"confidence\s*[=:]|"
        r"Pearson\s*r?\s*=\s*|"
        r"Spearman\s*=\s*|"
        r"AUC\s*=\s*|"
        r"p\s*[-<]\s*0\.|"
        r"n\s*=\s*\d",
        re.IGNORECASE,
    )

    def test_article_panels_no_statistics_in_legend_title(self):
        """Article figure producer source must not use legend(title=...) with statistical content."""
        from src.visualization.article_delivery import ARTICLE_FIGURE_PRODUCERS

        repo_root = Path(__file__).parent.parent
        violations = []
        chunk = 1200  # chars after .legend( to look for title=
        for basename, rel_path in ARTICLE_FIGURE_PRODUCERS:
            path = repo_root / rel_path
            if not path.exists():
                violations.append((basename, rel_path, f"Producer file not found: {path}"))
                continue
            text = path.read_text()
            pos = 0
            while True:
                idx = text.find(".legend(", pos)
                if idx < 0:
                    break
                snippet = text[idx : idx + chunk]
                for m in re.finditer(
                    r"title\s*=\s*([\"'])([^\"']*)\1",
                    snippet,
                ):
                    title_content = m.group(2)
                    if self.STAT_IN_LEGEND_TITLE.search(title_content):
                        violations.append(
                            (basename, rel_path, f"Legend title contains statistic: {title_content[:60]!r}")
                        )
                pos = idx + 1
        assert not violations, (
            "Article panels must not put statistics in legend titles (use caption or add_stat_box). "
            "Violations: " + "; ".join(f"{b} ({p}): {msg}" for b, p, msg in violations)
        )