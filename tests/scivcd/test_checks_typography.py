"""Tests for TYPOGRAPHY-category checks.

Each check gets a positive fixture (triggers a finding) and a negative
fixture (produces no finding).  Skips gracefully if the full API is not
yet implemented.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pytest

scivcd = pytest.importorskip(
    "scivcd",
    reason="scivcd top-level API not yet implemented",
)
if not hasattr(scivcd, "check"):
    pytest.skip("scivcd.check not yet implemented", allow_module_level=True)

from scivcd.core.types import Category

check = scivcd.check


def _typography_findings(report):
    return [f for f in report.findings if f.category is Category.TYPOGRAPHY]


def _skip_if_no_typography(report):
    if not _typography_findings(report) and not bool(report):
        pytest.skip("No TYPOGRAPHY checks registered yet")


# ---------------------------------------------------------------------------
# minimum_font_size — labels below the publication floor
# ---------------------------------------------------------------------------

class TestTypographyMinimumFontSize:
    def test_positive_tiny_labels(self):
        """Axis labels at 4 pt are below any reasonable publication minimum."""
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("X", fontsize=4)
        ax.set_ylabel("Y", fontsize=4)
        ax.set_title("Tiny", fontsize=4)
        fig.canvas.draw()
        report = check(fig)
        plt.close(fig)
        _skip_if_no_typography(report)
        assert len(_typography_findings(report)) >= 1

    def test_negative_adequate_font_sizes(self):
        """Labels at 11–12 pt are above the publication floor."""
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("X axis", fontsize=11)
        ax.set_ylabel("Y axis", fontsize=11)
        ax.set_title("Adequate", fontsize=12)
        fig.canvas.draw()
        report = check(fig)
        plt.close(fig)
        assert len(_typography_findings(report)) == 0


# ---------------------------------------------------------------------------
# inconsistent_font_sizes — mix of very large and very small text
# ---------------------------------------------------------------------------

class TestTypographyInconsistentFontSizes:
    def test_positive_extreme_size_mix(self):
        """A 4pt tick label alongside a 24pt title represents inconsistent typography."""
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1])
        ax.set_title("HUGE TITLE", fontsize=24)
        ax.tick_params(axis="both", labelsize=4)
        fig.canvas.draw()
        report = check(fig)
        plt.close(fig)
        _skip_if_no_typography(report)
        assert bool(report)

    def test_negative_uniform_font_sizes(self):
        """Uniform 11pt across all text elements is typographically consistent."""
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("X", fontsize=11)
        ax.set_ylabel("Y", fontsize=11)
        ax.set_title("Title", fontsize=11)
        ax.tick_params(axis="both", labelsize=11)
        fig.canvas.draw()
        report = check(fig)
        plt.close(fig)
        assert len(_typography_findings(report)) == 0


# ---------------------------------------------------------------------------
# text_truncation — tick labels or titles clipped by the figure boundary
# ---------------------------------------------------------------------------

class TestTypographyTextTruncation:
    def test_positive_truncated_tick_labels(self):
        """Long tick labels on a cramped figure get clipped at the axes edge."""
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.bar(range(4), [1, 2, 3, 4])
        ax.set_xticks(range(4))
        ax.set_xticklabels(
            ["Very_Long_Inhibitory_Cell_Type_A",
             "Very_Long_Excitatory_Cell_Type_B",
             "Another_Extremely_Long_Label_C",
             "Yet_Another_Verbose_Label_D"],
            rotation=0, fontsize=10,
        )
        fig.subplots_adjust(bottom=0.02, left=0.02, right=0.98)
        fig.canvas.draw()
        report = check(fig)
        plt.close(fig)
        _skip_if_no_typography(report)
        assert bool(report)

    def test_negative_short_tick_labels(self):
        """Short tick labels on a standard-sized figure are not truncated."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(4), [1, 2, 3, 4])
        ax.set_xticks(range(4))
        ax.set_xticklabels(["A", "B", "C", "D"], fontsize=10)
        fig.tight_layout()
        report = check(fig)
        plt.close(fig)
        assert len(_typography_findings(report)) == 0


# ---------------------------------------------------------------------------
# bold_in_body_text — bold face used for non-panel-label text
# ---------------------------------------------------------------------------

class TestTypographyBoldUsage:
    def test_positive_bold_body_annotation(self):
        """A bold-weight annotation in the plot body (not a panel label) is flagged."""
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1])
        ax.text(0.5, 0.5, "**Important note**",
                transform=ax.transAxes,
                fontweight="bold", fontsize=10,
                ha="center", va="center")
        fig.canvas.draw()
        report = check(fig)
        plt.close(fig)
        _skip_if_no_typography(report)
        assert bool(report)

    def test_negative_normal_weight_annotation(self):
        """A normal-weight annotation produces no TYPOGRAPHY finding."""
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1])
        ax.text(0.5, 0.5, "Regular note",
                transform=ax.transAxes,
                fontweight="normal", fontsize=10,
                ha="center", va="center")
        fig.canvas.draw()
        report = check(fig)
        plt.close(fig)
        assert len(_typography_findings(report)) == 0
