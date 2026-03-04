"""Auto-refinement engine: generate -> detect -> adjust loop."""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import matplotlib.pyplot as plt

from .panel_config import PanelConfig, AxisConfig

logger = logging.getLogger(__name__)

# Ensure VCD package is importable
_scripts = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))


def _count_warnings(issues: list[dict]) -> int:
    """Count warning-severity issues."""
    return sum(1 for i in issues if i.get("severity") == "warning")


# ═══════════════════════════════════════════════════════════════════════════════
# Weighted issue scoring
# ═══════════════════════════════════════════════════════════════════════════════

# Issue weights reflect publication-readiness priorities:
#   High:   truncation, log-scale misuse, label density  → broken/misleading
#   Medium: contrast, font size, text overlap             → readability
#   Low:    overplotting, legend occlusion, precision      → nice-to-have
_ISSUE_WEIGHTS: dict[str, float] = {
    # Integrity (data correctness)
    "text_truncation":          10.0,
    "patch_truncation":         10.0,
    "log_scale_nonpositive":    10.0,
    "floating_significance":     8.0,
    # Readability
    "label_density_excess":      8.0,
    "fontsize_too_small":        7.0,
    "low_contrast_text":         7.0,
    "text_overlap":              6.0,
    "tick_spine_overlap":        6.0,
    "font_family_violation":     5.0,
    # Clarity
    "overplotted_scatter":       3.0,
    "legend_data_occlusion":     3.0,
    "legend_spillover":          3.0,
    "legend_truncation":         3.0,
    "cross_panel_spillover":     3.0,
    "cbar_tick_overlap":         3.0,
    "cbar_tick_truncation":      3.0,
    "colorblind_confusable":     2.0,
    "precision_excess":          2.0,
    "scale_inconsistency":       2.0,
    # Minor
    "artist_overlap":            1.0,
    "axes_overflow":             1.0,
    "log_scale_unlabelled":      1.0,
    "low_contrast_line":         1.0,
    "bold_usage":                0.5,
    "errorbar_invisible":        1.0,
    "legend_text_crowding":      1.0,
    # Complexity / compaction (info-level, lower weight)
    "panel_complexity_excess":   2.0,
    "whitespace_excess":         0.5,
}

# Default weight for unrecognized issue types
_DEFAULT_WEIGHT: float = 2.0

# Severity multipliers
_SEVERITY_MUL: dict[str, float] = {
    "warning": 1.0,
    "info":    0.3,
}


def weighted_score(issues: list[dict]) -> float:
    """Compute a weighted quality score for a list of VCD issues.

    Lower is better.  Score = sum(issue_weight * severity_multiplier).
    """
    total = 0.0
    for iss in issues:
        itype = iss.get("type", "")
        severity = iss.get("severity", "info")
        w = _ISSUE_WEIGHTS.get(itype, _DEFAULT_WEIGHT)
        m = _SEVERITY_MUL.get(severity, 0.3)
        total += w * m
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# Category-specific action appliers
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_figure_actions(config: PanelConfig, action) -> tuple[PanelConfig, bool]:
    """Apply figure-level layout actions (margins, figsize, spacing)."""
    at = action.action_type
    changed = False

    if at == "increase_figsize_height":
        config = config.increase_figsize(dw=0.0, dh=0.5)
        changed = True
    elif at == "increase_figsize_width":
        config = config.increase_figsize(dw=0.5, dh=0.0)
        changed = True
    elif at == "increase_figsize":
        config = config.increase_figsize(dw=0.5, dh=0.5)
        changed = True
    elif at == "increase_wspace":
        config = config.increase_spacing(dw=0.05, dh=0.0)
        changed = True
    elif at == "increase_hspace":
        config = config.increase_spacing(dw=0.0, dh=0.05)
        changed = True
    elif at == "increase_bottom_margin":
        bm = min(config.bottom_margin + 0.03, 0.15)
        config = config.with_updates(bottom_margin=bm)
        changed = True
    elif at == "increase_right_margin":
        rm = max(config.right_margin - 0.02, 0.85)
        config = config.with_updates(right_margin=rm)
        changed = True
    elif at == "increase_top_margin":
        tm = max(config.top_margin - 0.02, 0.85)
        config = config.with_updates(top_margin=tm)
        changed = True
    elif at == "increase_margins":
        # General margin widening
        bm = min(config.bottom_margin + 0.03, 0.15)
        config = config.with_updates(bottom_margin=bm)
        changed = True
    elif at == "reduce_hspace":
        # Compaction: reduce vertical subplot spacing
        delta = action.params.get("delta", 0.05)
        min_h = action.params.get("min_hspace", 0.20)
        config = config.reduce_spacing(dh=delta, min_hspace=min_h)
        changed = True
    elif at == "reduce_figsize_height":
        # Compaction: shrink figure height towards target ratio
        delta = action.params.get("delta_height", 0.3)
        min_h = action.params.get("min_height", 4.0)
        config = config.reduce_figsize(dh=delta, min_h=min_h)
        changed = True

    return config, changed


def _apply_axis_density_actions(config: PanelConfig, action) -> tuple[PanelConfig, bool]:
    """Apply tick-label, rotation, and annotation density actions."""
    at = action.action_type
    changed = False

    if at == "reduce_tick_labels":
        target_axis = action.params.get("axis_name", "default")
        ax_cfg = config.get_axis(target_axis)
        ax_cfg.max_tick_labels = max(5, ax_cfg.max_tick_labels - 5)
        changed = True
    elif at == "rotate_labels":
        target_axis = action.params.get("axis_name", "default")
        ax_cfg = config.get_axis(target_axis)
        ax_cfg.label_rotation = min(90, ax_cfg.label_rotation + 15)
        changed = True
    elif at == "reduce_annotations":
        target_axis = action.params.get("axis_name", "default")
        ax_cfg = config.get_axis(target_axis)
        ax_cfg.max_annotations = max(0, ax_cfg.max_annotations - 2)
        changed = True

    return config, changed


def _apply_legend_actions(config: PanelConfig, action) -> tuple[PanelConfig, bool]:
    """Apply legend repositioning, font shrink, and entry reduction actions."""
    at = action.action_type
    changed = False

    if at in ("move_legend", "move_legend_inside"):
        target_axis = action.params.get("axis_name", "default")
        ax_cfg = config.get_axis(target_axis)
        cycle = ["best", "upper left", "upper right", "lower left", "lower right"]
        try:
            idx = cycle.index(ax_cfg.legend_loc)
            ax_cfg.legend_loc = cycle[(idx + 1) % len(cycle)]
        except ValueError:
            ax_cfg.legend_loc = "best"
        changed = True
    elif at == "shrink_legend_font":
        target_axis = action.params.get("axis_name", "default")
        ax_cfg = config.get_axis(target_axis)
        ax_cfg.legend_fontsize = max(7, ax_cfg.legend_fontsize - 1)
        changed = True

    return config, changed


def _apply_structural_actions(config: PanelConfig, action) -> tuple[PanelConfig, bool]:
    """Apply structural layout changes (subplot grid, panel splitting)."""
    at = action.action_type
    changed = False

    if at == "reduce_subplots_per_row":
        preferred = action.params.get("preferred_max_cols", 2)
        if config.ncols > preferred:
            config = config.reduce_cols(preferred)
            changed = True

    return config, changed


def _apply_perceptual_actions(config: PanelConfig, action) -> tuple[PanelConfig, bool]:
    """Apply perceptual/semantic adjustments to AxisConfig knobs."""
    at = action.action_type
    changed = False

    if at == "reduce_alpha":
        target_axis = action.params.get("axis_name", "default")
        ax_cfg = config.get_axis(target_axis)
        new_alpha = action.params.get("alpha", 0.3)
        ax_cfg.alpha = new_alpha
        changed = True
    elif at == "reduce_label_precision":
        target_axis = action.params.get("axis_name", "default")
        ax_cfg = config.get_axis(target_axis)
        ax_cfg.numeric_precision = action.params.get("max_decimals", 4)
        changed = True
    elif at == "fix_cvd_palette":
        target_axis = action.params.get("axis_name", "default")
        ax_cfg = config.get_axis(target_axis)
        ax_cfg.palette_name = "colorblind_safe"
        changed = True
    elif at == "use_density_viz":
        target_axis = action.params.get("axis_name", "default")
        ax_cfg = config.get_axis(target_axis)
        ax_cfg.max_points = 2000  # signal to panel code to subsample
        changed = True

    return config, changed


def _apply_complexity_actions(config: PanelConfig, action) -> tuple[PanelConfig, bool]:
    """Apply complexity-reduction actions (Pass 31).

    Currently records the intent in AxisConfig so panel drawing code can
    respect it.  Actions ``drop_secondary_bar_labels`` and
    ``suggest_panel_split`` cannot be fully auto-applied (they require panel
    code cooperation), but we lower ``max_annotations`` as a signal.
    """
    at = action.action_type
    changed = False

    if at == "drop_secondary_bar_labels":
        # Signal every axis: reduce max_annotations to keep only top/bottom 3
        n = action.params.get("n", 3)
        for ax_cfg in config.axes.values():
            if ax_cfg.max_annotations > n * 2:
                ax_cfg.max_annotations = n * 2
                changed = True
        if not config.axes:
            # No axes yet; prime the default axis
            ax_cfg = config.get_axis("default")
            ax_cfg.max_annotations = n * 2
            changed = True

    elif at == "reduce_legend_entries":
        # Signal every axis: cap legend entries
        max_e = action.params.get("max_entries", 8)
        for ax_cfg in config.axes.values():
            ax_cfg.legend_ncol = max(1, ax_cfg.legend_ncol)
        # Mark in config description; panel code must check this
        config = config.with_updates(
            layout_mode="compact" if config.layout_mode == "auto" else config.layout_mode
        )
        changed = True

    return config, changed


# ── Compaction helper ────────────────────────────────────────────────────────

def _compact_layout(
    config: PanelConfig,
    *,
    target_ratio: float = 0.50,
    hspace_compact: float = 0.35,
    hspace_min: float = 0.20,
    height_min: float = 4.0,
) -> tuple[PanelConfig, bool]:
    """Attempt one step of layout compaction.

    Reduces hspace towards *hspace_compact*, then reduces figure height
    towards *target_ratio* × width.  Each call changes at most one
    parameter so the caller can re-detect after every step.

    Returns ``(new_config, changed)``.
    """
    changed = False

    # Step 1: reduce hspace if above target
    if config.hspace > hspace_compact + 0.01:
        new_hspace = max(config.hspace - 0.05, hspace_compact)
        config = config.with_updates(hspace=new_hspace)
        changed = True

    # Step 2: reduce height if h/w ratio exceeds target
    w, h = config.figsize
    if w > 0 and h / w > target_ratio + 0.02:
        new_h = max(h - 0.3, max(height_min, w * target_ratio))
        if new_h < h - 0.01:
            config = config.with_updates(figsize=(w, new_h))
            changed = True

    return config, changed


def _apply_actions(config: PanelConfig, issues: list[dict]) -> PanelConfig:
    """Apply rule-based adjustments to config based on VCD issues.

    Delegates to category-specific appliers for maintainability.
    Returns a new PanelConfig with adjustments, or the same config if
    no changes were needed.
    """
    from vcd.vcd_actions import diagnose

    actions = diagnose(issues)
    if not actions:
        return config

    new_config = config
    changed = False

    for action in actions:
        # Try each category applier in priority order
        for applier in (
            _apply_figure_actions,
            _apply_axis_density_actions,
            _apply_legend_actions,
            _apply_structural_actions,
            _apply_perceptual_actions,
            _apply_complexity_actions,
        ):
            new_config, did_change = applier(new_config, action)
            if did_change:
                changed = True
                break  # action handled, move to next

    if not changed:
        return config
    return new_config


def refine_panel(
    make_figure_fn: Callable[[PanelConfig], Optional[plt.Figure]],
    config: PanelConfig,
    *,
    max_passes: int = 3,
    label: str = "",
    verbose: bool = True,
) -> Tuple[Optional[plt.Figure], PanelConfig, list[dict]]:
    """Run generate->detect->adjust loop with weighted scoring.

    Parameters
    ----------
    make_figure_fn : callable
        Takes a PanelConfig, returns a Figure (or None).
    config : PanelConfig
        Initial configuration.
    max_passes : int
        Maximum refinement iterations.
    label : str
        Label for VCD output.
    verbose : bool
        Print progress.

    Returns
    -------
    (fig, final_config, final_issues) :
        The best figure (lowest weighted score), the config that
        produced it, and remaining issues.
    """
    from vcd import detect_all_conflicts

    current_config = config
    best_fig = None
    best_issues: list[dict] = []
    best_score = float("inf")
    best_config = config
    action_history: set[str] = set()  # track applied actions to avoid loops

    for pass_num in range(1, max_passes + 1):
        # Generate
        fig = make_figure_fn(current_config)
        if fig is None:
            logger.warning(f"[auto_refine] {label} pass {pass_num}: make_figure_fn returned None")
            return None, current_config, []

        # Detect
        issues = detect_all_conflicts(fig, label=f"{label}_pass{pass_num}", verbose=verbose)
        n_warn = _count_warnings(issues)
        score = weighted_score(issues)

        if verbose:
            logger.info(
                f"[auto_refine] {label} pass {pass_num}/{max_passes}: "
                f"{n_warn} warnings, score={score:.1f}"
            )

        # Track best by weighted score
        if score < best_score:
            if best_fig is not None:
                plt.close(best_fig)
            best_fig = fig
            best_issues = issues
            best_score = score
            best_config = current_config
        else:
            plt.close(fig)
            # Score didn't improve — bail early
            if verbose:
                logger.info(
                    f"[auto_refine] {label}: score did not improve "
                    f"({score:.1f} >= {best_score:.1f}), stopping"
                )
            break

        # Done if clean
        if n_warn == 0:
            break

        # Last pass — no more adjustments
        if pass_num == max_passes:
            break

        # Adjust
        new_config = _apply_actions(current_config, issues)

        # Check for action loops
        config_key = repr(new_config)
        if config_key in action_history or new_config is current_config:
            if verbose:
                logger.info(f"[auto_refine] {label}: no further adjustments available")
            break
        action_history.add(config_key)
        current_config = new_config

    # ── Compaction phase ──────────────────────────────────────────────────
    # Only run when the main loop has resolved all serious integrity issues
    # (truncation, log-scale misuse, label density).  Integrity issues carry
    # weight ≥ 7.0; if the best weighted score is still large the compaction
    # phase is skipped so we do not mask unsolved problems.
    _COMPACTION_SCORE_GATE = 20.0  # skip compaction above this total score
    _MAX_COMPACT_PASSES = 6

    if best_fig is not None and best_score < _COMPACTION_SCORE_GATE:
        if verbose:
            logger.info(
                f"[auto_refine] {label}: entering compaction phase "
                f"(best_score={best_score:.1f})"
            )
        compact_config = best_config
        for cp in range(1, _MAX_COMPACT_PASSES + 1):
            candidate_config, did_compact = _compact_layout(compact_config)
            if not did_compact:
                if verbose:
                    logger.info(
                        f"[auto_refine] {label}: layout at compaction target, "
                        f"stopping after {cp - 1} compaction passes"
                    )
                break

            cfig = make_figure_fn(candidate_config)
            if cfig is None:
                break

            c_issues = detect_all_conflicts(
                cfig,
                label=f"{label}_compact{cp}",
                verbose=verbose,
            )
            c_score = weighted_score(c_issues)
            c_warns = _count_warnings(c_issues)

            # Accept compaction only when it introduces no new warnings and
            # does not meaningfully worsen the score
            if c_warns == 0 and c_score <= best_score + 1.0:
                plt.close(best_fig)
                best_fig = cfig
                best_issues = c_issues
                best_score = c_score
                best_config = candidate_config
                compact_config = candidate_config
                if verbose:
                    logger.info(
                        f"[auto_refine] {label}: compaction pass {cp} accepted "
                        f"(score={c_score:.1f}, "
                        f"hspace={candidate_config.hspace:.2f}, "
                        f"h={candidate_config.figsize[1]:.1f}in)"
                    )
            else:
                plt.close(cfig)
                if verbose:
                    logger.info(
                        f"[auto_refine] {label}: compaction pass {cp} rejected "
                        f"({c_warns} new warnings, score={c_score:.1f})"
                    )
                break

    return best_fig, best_config, best_issues
