"""Unit tests for VCD check modules (US-206).

Covers three check modules: publication, policy (severity mapping), and
baseline (diff against a pinned snapshot). Each test constructs a minimal
matplotlib figure and asserts on the check output shape.

The ``scripts/vcd/`` package is gitignored (internal-tool policy from
commit d5782e2). When it is absent — for example on a fresh CI checkout —
the whole module is skipped rather than erroring out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

if not (_REPO / "scripts" / "vcd" / "__init__.py").exists():
    pytest.skip(
        "scripts/vcd/ package not present (intentionally gitignored internal tool)",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Severity mapping (US-202)
# ---------------------------------------------------------------------------

def test_severity_maps_truncation_to_critical():
    from vcd.vcd_policy import severity_level_for
    issue = {"type": "text_truncation", "severity": "warning"}
    assert severity_level_for(issue) == "CRITICAL"


def test_severity_maps_cross_axes_overlap_to_critical():
    from vcd.vcd_policy import severity_level_for
    issue = {"type": "cross_axes_text_overlap", "severity": "warning"}
    assert severity_level_for(issue) == "CRITICAL"


def test_severity_maps_legend_masking_to_minor():
    from vcd.vcd_policy import severity_level_for
    issue = {"type": "legend_artist_masking", "severity": "warning"}
    assert severity_level_for(issue) == "MINOR"


def test_severity_count_aggregates_levels():
    from vcd.vcd_policy import annotate_severity_levels, count_by_severity_level
    issues = [
        {"type": "text_truncation", "severity": "warning"},
        {"type": "text_overlap", "severity": "warning"},
        {"type": "legend_artist_masking", "severity": "warning"},
        {"type": "bold_usage", "severity": "info"},
    ]
    annotate_severity_levels(issues)
    counts = count_by_severity_level(issues)
    assert counts == {"CRITICAL": 1, "MAJOR": 1, "MINOR": 1, "INFO": 1}


# ---------------------------------------------------------------------------
# Publication-quality checks (US-204)
# ---------------------------------------------------------------------------

def test_check_minimum_font_size_flags_small_label():
    from vcd.vcd_checks_publication import check_minimum_font_size
    # 14-inch figure rendered at 7-inch width -> scale 0.5;
    # a 10pt label renders at 5pt, below the 7pt threshold.
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot([0, 1], [0, 1])
    ax.set_title("Tiny", fontsize=10)
    issues = check_minimum_font_size(fig, min_pt=7.0)
    plt.close(fig)
    assert any(i["type"] == "minimum_font_size" for i in issues)


def test_check_minimum_font_size_passes_large_label():
    from vcd.vcd_checks_publication import check_minimum_font_size
    # 7-inch figure at 7-inch include width -> scale 1.0; 12pt passes.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1])
    ax.set_title("Large Enough", fontsize=12)
    issues = check_minimum_font_size(fig, min_pt=7.0)
    plt.close(fig)
    assert not any(i["type"] == "minimum_font_size" for i in issues)


def test_check_effective_dpi_flags_low_dpi():
    from vcd.vcd_checks_publication import check_effective_dpi
    # 14-inch figure at 300 dpi rendered at 7-inch width -> effective DPI = 600
    fig = plt.figure(figsize=(14, 8), dpi=300)
    ok = check_effective_dpi(fig, min_effective_dpi=800)
    plt.close(fig)
    assert any(i["type"] == "effective_dpi_low" for i in ok)


def test_check_effective_dpi_passes_high_dpi():
    from vcd.vcd_checks_publication import check_effective_dpi
    fig = plt.figure(figsize=(14, 8), dpi=300)
    ok = check_effective_dpi(fig, min_effective_dpi=300)
    plt.close(fig)
    assert ok == []


def test_check_colorblind_safety_flags_red_orange():
    from vcd.vcd_checks_publication import check_colorblind_safety
    # Muted red vs orange — the canonical deuteranopia confusion pair.
    issues = check_colorblind_safety(
        fig=plt.figure(),
        palette=[(0.8, 0.3, 0.3), (0.8, 0.5, 0.3)],
        min_delta_e=15.0,
    )
    plt.close("all")
    assert any(i["type"] == "colorblind_confusable" for i in issues)


def test_check_colorblind_safety_passes_orange_blue():
    from vcd.vcd_checks_publication import check_colorblind_safety
    # Orange / blue are CVD-safe.
    issues = check_colorblind_safety(
        fig=plt.figure(),
        palette=[(1.0, 0.5, 0.0), (0.0, 0.45, 0.8)],
        min_delta_e=10.0,
    )
    plt.close("all")
    assert not any(i["type"] == "colorblind_confusable" for i in issues)


# ---------------------------------------------------------------------------
# Baseline diff (US-203)
# ---------------------------------------------------------------------------

def test_baseline_diff_empty_when_matched():
    from vcd.vcd_baseline import snapshot_from_vcd_report, diff_against_baseline
    vcd = {
        "fig01.pdf": {
            "severity_counts": {"CRITICAL": 0, "MAJOR": 1, "MINOR": 0, "INFO": 0},
            "findings": [
                {"type": "text_overlap", "severity_level": "MAJOR", "detail": "a vs b"},
            ],
        }
    }
    snap = snapshot_from_vcd_report(vcd)
    report = diff_against_baseline(snap, snap)
    assert report.has_new_critical is False
    assert report.totals_added == {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0, "INFO": 0}


def test_baseline_diff_flags_new_critical():
    from vcd.vcd_baseline import diff_against_baseline
    baseline = {"figures": {"fig01.pdf": {"severity_counts": {}, "finding_keys": []}}}
    current = {
        "figures": {
            "fig01.pdf": {
                "severity_counts": {"CRITICAL": 1},
                "finding_keys": [["text_truncation", "CRITICAL", "xtick 150 truncated"]],
            }
        }
    }
    report = diff_against_baseline(current, baseline)
    assert report.has_new_critical is True
    assert report.totals_added["CRITICAL"] == 1


def test_baseline_diff_reports_removed_findings():
    from vcd.vcd_baseline import diff_against_baseline
    baseline = {
        "figures": {
            "fig01.pdf": {
                "severity_counts": {"MAJOR": 1},
                "finding_keys": [["text_overlap", "MAJOR", "A vs B"]],
            }
        }
    }
    current = {"figures": {"fig01.pdf": {"severity_counts": {}, "finding_keys": []}}}
    report = diff_against_baseline(current, baseline)
    assert report.totals_removed["MAJOR"] == 1
    assert report.has_new_critical is False
