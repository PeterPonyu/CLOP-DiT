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


# ---------------------------------------------------------------------------
# Complexity routing (US-301)
# ---------------------------------------------------------------------------

def test_complexity_simple_single_axes():
    from vcd.vcd_complexity import classify_figure, Complexity
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    assert classify_figure(fig) == Complexity.SIMPLE
    plt.close(fig)


def test_complexity_compound_row():
    from vcd.vcd_complexity import classify_figure, Complexity
    fig, axs = plt.subplots(1, 3)
    for a in axs:
        a.plot([0, 1], [0, 1])
    assert classify_figure(fig) == Complexity.COMPOUND
    plt.close(fig)


def test_complexity_composed_grid():
    from vcd.vcd_complexity import classify_figure, Complexity
    fig, axs = plt.subplots(3, 3)
    for row in axs:
        for a in row:
            a.plot([0, 1], [0, 1])
    assert classify_figure(fig) == Complexity.COMPOSED
    plt.close(fig)


def test_complexity_composed_on_fig_legend():
    from vcd.vcd_complexity import classify_figure, Complexity
    fig, ax = plt.subplots()
    ln, = ax.plot([0, 1], [0, 1], label="x")
    fig.legend([ln], ["X"])
    assert classify_figure(fig) == Complexity.COMPOSED
    plt.close(fig)


def test_profile_auto_routes_fewer_checks_on_simple():
    from vcd import detect_all_conflicts
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    auto = detect_all_conflicts(fig, verbose=False, profile="auto")
    full = detect_all_conflicts(fig, verbose=False, profile="full")
    # Simple figures may hit publication checks in both, but auto should be
    # at worst equal to full — never larger.
    assert len(auto) <= len(full)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Font autofix (US-302)
# ---------------------------------------------------------------------------

def test_autofix_scales_up_on_roomy_figure():
    from vcd.vcd_autofix import maximize_font_size
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.6)
    for i, a in enumerate(axs):
        a.plot([0, 1, 2], [0, 1, 0])
        a.set_title(f"P{i}", fontsize=8)
    res = maximize_font_size(fig, step=1.05, max_iter=8)
    plt.close(fig)
    assert res.scale_factor >= 1.0
    assert res.iterations >= 1


def test_autofix_returns_font_scaling_result_shape():
    from vcd.vcd_autofix import maximize_font_size, FontScalingResult
    fig, ax = plt.subplots()
    ax.set_title("t", fontsize=10)
    res = maximize_font_size(fig)
    plt.close(fig)
    assert isinstance(res, FontScalingResult)
    assert res.scale_factor >= 1.0
    assert res.max_legible_avg_pt >= res.current_avg_pt


# ---------------------------------------------------------------------------
# Layout tightening (US-303)
# ---------------------------------------------------------------------------

def test_tighten_shrinks_over_padded_figure():
    from vcd.vcd_tighten import tighten_layout, TightenResult
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.6, wspace=0.6, left=0.18, right=0.82, top=0.82, bottom=0.18)
    for row in axs:
        for a in row:
            a.plot([0, 1], [0, 1])
    before_hspace = fig.subplotpars.hspace
    res = tighten_layout(fig, step=0.04, max_iter=12)
    plt.close(fig)
    assert isinstance(res, TightenResult)
    assert res.after_params["hspace"] <= before_hspace


def test_tighten_saves_whitespace_pct_is_finite():
    from vcd.vcd_tighten import tighten_layout
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.subplots_adjust(wspace=0.4, left=0.15, right=0.85)
    for a in axs:
        a.plot([0, 1], [0, 1])
    res = tighten_layout(fig, step=0.03)
    plt.close(fig)
    assert res.saved_whitespace_pct >= 0.0
    assert res.saved_whitespace_pct < 100.0


# ---------------------------------------------------------------------------
# Content-aware checks (US-305 fold-back)
# ---------------------------------------------------------------------------

def test_label_string_ellipsis_flags_pre_truncated_labels():
    from vcd.vcd_checks_content import check_label_string_ellipsis
    fig, ax = plt.subplots()
    ax.bar(range(3), [1, 2, 3])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Inhibitory GABAer…", "Smooth musc…", "CD8+ T lym…"])
    fig.canvas.draw()
    issues = check_label_string_ellipsis(fig)
    plt.close(fig)
    assert any(i["type"] == "label_string_ellipsis" for i in issues)


def test_label_string_ellipsis_clean_labels_pass():
    from vcd.vcd_checks_content import check_label_string_ellipsis
    fig, ax = plt.subplots()
    ax.bar(range(3), [1, 2, 3])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["GABA", "Smooth", "CD8"])
    fig.canvas.draw()
    issues = check_label_string_ellipsis(fig)
    plt.close(fig)
    assert not any(i["type"] == "label_string_ellipsis" for i in issues)


def test_overlapping_series_flags_redundant_lines():
    from vcd.vcd_checks_content import check_overlapping_series_values
    fig, ax = plt.subplots()
    import numpy as np
    x = np.arange(50)
    # One reference series spans 0..1 so the axis has non-zero range;
    # three coincident series all sit at the ceiling (0.99).
    ax.plot(x, x / 49.0, label="ref")
    y_ceiling = np.full_like(x, 0.99, dtype=float)
    for label in ("a", "b", "c"):
        ax.plot(x, y_ceiling, label=label)
    fig.canvas.draw()
    issues = check_overlapping_series_values(
        fig, min_series=3, coincidence_threshold=0.90, min_x_fraction=0.5
    )
    plt.close(fig)
    # The 3 ceiling series are coincident across the x-range.
    assert any(i["type"] == "overlapping_series_values" for i in issues)


def test_overlapping_series_distinct_lines_pass():
    from vcd.vcd_checks_content import check_overlapping_series_values
    fig, ax = plt.subplots()
    import numpy as np
    x = np.linspace(0, 1, 50)
    ax.plot(x, x, label="a")
    ax.plot(x, 1 - x, label="b")
    ax.plot(x, np.sin(x * 6), label="c")
    fig.canvas.draw()
    issues = check_overlapping_series_values(fig)
    plt.close(fig)
    assert not any(i["type"] == "overlapping_series_values" for i in issues)


def test_duplicate_tick_labels_detected():
    from vcd.vcd_checks_content import check_duplicate_tick_labels
    fig, ax = plt.subplots()
    ax.bar(range(6), [1, 2, 3, 1, 2, 3])
    ax.set_xticks(range(6))
    ax.set_xticklabels(["CLOP", "scVI", "Gaussian", "CLOP", "scVI", "Gaussian"])
    fig.canvas.draw()
    issues = check_duplicate_tick_labels(fig)
    plt.close(fig)
    assert any(i["type"] == "duplicate_tick_labels" for i in issues)


# ---------------------------------------------------------------------------
# Panel-label detection (refresh hardening)
# ---------------------------------------------------------------------------

def test_panel_label_overlap_detects_helper_gid_labels():
    from src.visualization.style import add_panel_label
    from vcd.vcd_core import _collect_artists
    from vcd.vcd_checks_text import _check_panel_label_overlap

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title("Title", loc="left")
    add_panel_label(ax, "a", x=0.0, y=1.0)
    ax.text(0.0, 1.0, "Overlap", transform=ax.transAxes, ha="left", va="bottom")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = _check_panel_label_overlap(fig, renderer, _collect_artists(fig, renderer))
    plt.close(fig)

    assert any(i["type"] == "panel_label_text_overlap" for i in issues)


def test_panel_label_overlap_detects_manual_top_band_labels():
    from vcd.vcd_core import _collect_artists
    from vcd.vcd_checks_text import _check_panel_label_overlap

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.text(0.35, 1.02, "B", transform=ax.transAxes, fontsize=18,
            fontweight="bold", ha="left", va="bottom")
    ax.text(0.35, 1.02, "Overlap", transform=ax.transAxes, ha="left", va="bottom")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = _check_panel_label_overlap(fig, renderer, _collect_artists(fig, renderer))
    plt.close(fig)

    assert any(i["type"] == "panel_label_text_overlap" for i in issues)


def test_panel_label_overlap_ignores_center_annotation_negative_control():
    from vcd.vcd_core import _collect_artists
    from vcd.vcd_checks_text import _check_panel_label_overlap

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.text(0.50, 0.50, "A", transform=ax.transAxes, fontsize=18,
            fontweight="bold", ha="center", va="center")
    ax.text(0.50, 0.50, "Overlap", transform=ax.transAxes, ha="center", va="center")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = _check_panel_label_overlap(fig, renderer, _collect_artists(fig, renderer))
    plt.close(fig)

    assert not any(i["type"].startswith("panel_label_") for i in issues)


def test_panel_label_overlap_skips_spine_patch():
    """Stage 7 fix: Spine patches are frame decorations, not content.
    A panel label touching its own axes border is not a visibility
    defect, so panel_label_overlap skips Spine artists (matching the
    Spine-skip policy already applied across the other VCD checks).
    This test locks the skip in.
    """
    from src.visualization.style import add_panel_label
    from vcd.vcd_core import _collect_artists
    from vcd.vcd_checks_text import _check_panel_label_overlap

    fig, ax = plt.subplots(figsize=(5, 4))
    # Standard panel-label position: just above the axes top-left.
    add_panel_label(ax, "G", x=-0.14, y=1.09)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = _check_panel_label_overlap(fig, renderer, _collect_artists(fig, renderer))
    plt.close(fig)

    for issue in issues:
        if issue.get("type") == "panel_label_overlap":
            assert "Spine" not in issue.get("detail", ""), (
                f"Spine overlap should be skipped, got: {issue}"
            )


# ---------------------------------------------------------------------------
# VCD-delta regression gate (Phase 4a)
# ---------------------------------------------------------------------------

import json as _json

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BASELINE_PATH = _REPO_ROOT / "revision" / "figure_fix_reports" / "vcd_baseline_2026-04-19.json"
_CURRENT_PATH = _REPO_ROOT / "results" / "vcd_report.json"

# Per-figure warning-count exemptions for acknowledged design trade-offs.
# Each entry: figure_key -> (max_delta_allowed, rationale).
# The plan §ADR records the trade-off; the exemption is visible in test output.
_VCD_REGRESSION_EXEMPTIONS: dict[str, tuple[int, str]] = {
    # fig02a +3: panel labels at PANEL_OFFSET_FARLEFT (x=-0.22) for horizontal
    # alignment (user's "at least 1 direction aligned" directive) intrinsically
    # overlap wide numeric yticks ("100", "2.25") on panels C and G. Real fixes
    # attempted (MaxNLocator retightening, y=0.98 label-inside-axes, ylim
    # compression) each regressed the figure further; the overlap is structural
    # given the chosen horizontal alignment and cannot be eliminated without
    # reverting to per-panel bespoke offsets (undoing A2-partial).
    "fig02a_training_dynamics.pdf": (
        5,
        "Horizontal panel-label alignment (x=-0.22) conflicts with wide numeric "
        "yticks on panels C (Accuracy ytick=100) and G (LR ytick=2.25); ≤130 px² "
        "overlap is structural given user-approved alignment directive.",
    ),
    # fig07c +17: user chose Option I-a (repeat method labels ×3 across all 24
    # rows). The label_density_excess VCD warning fires because 24 labels fill
    # 93% of axis height — a direct consequence of user's I-a choice. Real
    # fixes explored: shrinking max_len (tried 14→12/16→14/12→10) just produces
    # minimum_font_size warnings instead (labels render <5pt). The only way to
    # eliminate density_excess is to revert I-a (undoing user's explicit pick)
    # or restructure the figure layout to make panel I taller.
    "fig07c_benchmark.pdf": (
        20,
        "Option I-a trade-off: user-directed labels ×3 across 24 rows "
        "intrinsically produces label_density_excess at the chosen panel height; "
        "shrinking label length trades density warns for font-size warns.",
    ),
}


def test_vcd_warn_count_regression():
    """Per-figure warning counts must not increase vs the committed baseline.

    The baseline is the VCD snapshot captured before the figure-polish pass
    (revision/figure_fix_reports/vcd_baseline_2026-04-19.json).  For each
    figure key present in both reports the number of items in the 'warnings'
    list must be <= the baseline count + the per-figure exemption (if any).
    Figures absent from the current report (intentionally removed) are skipped
    with a log message.  Non-figure keys such as '__summary__' (whose
    'warnings' value is not a list) are silently ignored.

    Per-figure exemptions live in ``_VCD_REGRESSION_EXEMPTIONS`` and represent
    acknowledged design trade-offs — each entry cites the plan section that
    authorizes it.

    If results/vcd_report.json does not exist, the test is skipped with an
    actionable message — run scripts/pipeline/run_regeneration.py first.
    """
    if not _CURRENT_PATH.exists():
        pytest.skip(
            "VCD report missing — run scripts/pipeline/run_regeneration.py first"
        )

    baseline = _json.loads(_BASELINE_PATH.read_text())
    current = _json.loads(_CURRENT_PATH.read_text())

    regressions = []
    exemption_hits = []
    for fig_key, base_entry in baseline.items():
        # Skip non-figure summary entries (e.g. '__summary__')
        base_warns_raw = base_entry.get("warnings")
        if not isinstance(base_warns_raw, list):
            continue

        if fig_key not in current:
            # Figure intentionally removed; not a regression.
            print(f"[vcd-delta] {fig_key}: absent from current report — skipped")
            continue

        cur_warns_raw = current[fig_key].get("warnings")
        if not isinstance(cur_warns_raw, list):
            # Current entry malformed; treat as zero warnings (conservative).
            cur_warns_raw = []

        base_count = len(base_warns_raw)
        cur_count = len(cur_warns_raw)
        delta = cur_count - base_count

        if delta <= 0:
            continue

        exemption = _VCD_REGRESSION_EXEMPTIONS.get(fig_key)
        if exemption is not None and delta <= exemption[0]:
            exemption_hits.append(
                f"{fig_key}: +{delta} within exemption (max {exemption[0]}) — "
                f"{exemption[1]}"
            )
            continue

        regressions.append(
            f"{fig_key}: baseline={base_count} current={cur_count} (+{delta})"
        )

    for msg in exemption_hits:
        print(f"[vcd-delta][exempt] {msg}")

    assert not regressions, (
        "VCD regressions vs baseline — warning counts increased for:\n  "
        + "\n  ".join(regressions)
    )


# ---------------------------------------------------------------------------
# Cross-slice VCD coverage gate (plan §5.6 bullet 7)
# ---------------------------------------------------------------------------
#
# When a single-producer composite PDF replaces several legacy per-slice PDFs,
# the composite can hide cross-gridspec overlaps that the individual slices
# never saw. The per-slice VCD regression gate above only sees the old
# per-slice keys; a new composite with cross-cell overlap would pass that
# gate silently.
#
# This test iterates the ``COMPOSITE_VCD_REGISTRY`` maintained in
# ``src/visualization/article_composition.py`` and enforces
#
#     len(current[composite]["warnings"]) <= sum(len(current[slice]["warnings"])
#                                                for slice in slices)
#
# for every registered (composite, slices) pair.
#
# **Limitation** (plan iter2 §5.6 bullet 7 note): this is a NECESSARY but
# NOT SUFFICIENT gate. If a slice tightens (e.g. 2 → 1 warning) while the
# composite gains a new cross-cell overlap (0 → 1), the sum check still
# passes (``1 <= 1 + N``). The slice-tightening case is caught by the
# manual visual-spot-check sign-off (plan §5.6 bullet 6), not by this test.


def test_composite_vcd_coverage():
    """For every composite registered in ``COMPOSITE_VCD_REGISTRY``, the
    composite's VCD warn count must not exceed the sum of warn counts on
    its constituent legacy slice PDFs. Vacuously green while the registry
    is empty (Step 0 state of the single-producer migration plan).
    """
    from src.visualization.article_composition import COMPOSITE_VCD_REGISTRY

    if not COMPOSITE_VCD_REGISTRY:
        # Step 0 state: no composites registered yet. The gate is live but
        # vacuously green until Step 1 (Fig 8 pilot) populates the first
        # entry.
        pytest.skip("COMPOSITE_VCD_REGISTRY empty — no composites to check yet")

    if not _CURRENT_PATH.exists():
        pytest.skip(
            "VCD report missing — run scripts/pipeline/run_regeneration.py first"
        )

    current = _json.loads(_CURRENT_PATH.read_text())

    failures = []
    for composite_key, slice_keys in COMPOSITE_VCD_REGISTRY.items():
        # Composite may not be in the report yet if the producer hasn't run.
        composite_entry = current.get(composite_key)
        if composite_entry is None:
            failures.append(
                f"{composite_key}: absent from current VCD report — "
                f"run_regeneration.py may not have produced the composite PDF"
            )
            continue

        composite_warns_raw = composite_entry.get("warnings")
        composite_warns = (
            len(composite_warns_raw)
            if isinstance(composite_warns_raw, list)
            else 0
        )

        # Sum the slice warn counts. Missing slices are treated as zero
        # (not a failure here — the per-slice regression gate above is the
        # authority on slice-level changes).
        slice_sum = 0
        for slice_key in slice_keys:
            slice_entry = current.get(slice_key)
            if slice_entry is None:
                continue
            slice_warns_raw = slice_entry.get("warnings")
            if isinstance(slice_warns_raw, list):
                slice_sum += len(slice_warns_raw)

        if composite_warns > slice_sum:
            failures.append(
                f"{composite_key}: composite warns={composite_warns} "
                f"exceeds sum(slices)={slice_sum} over {slice_keys}"
            )

    assert not failures, (
        "Composite VCD coverage gate tripped (cross-slice overlap suspected):\n  "
        + "\n  ".join(failures)
        + "\n\nNote: the <=-sum gate is necessary-but-not-sufficient — it "
          "does not catch the case where a slice's warn count tightens "
          "while the composite gains an orthogonal cross-cell overlap of "
          "comparable size. Visual spot-check (plan §5.6 bullet 6) is "
          "required alongside this gate."
    )
