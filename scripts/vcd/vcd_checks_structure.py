"""VCD structural detection passes (16, 19-22) + per-axes summary."""

from __future__ import annotations

import re
from matplotlib.text import Text

from .vcd_core import _ArtistInfo, _safe_bbox, _shrink, _fig_bbox, _overlap_area, _sides_outside


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 16: Significance bracket annotation issues
# ═══════════════════════════════════════════════════════════════════════════════

def _check_significance_brackets(fig, renderer, border_tol_px=5.0):
    """Pass 16: Detect significance bracket annotation issues.

    Checks:
      a) Bracket star/ns text truncated at figure border.
      b) Bracket star text overlapping other text labels.
      c) Bracket lines (drawn with clip_on=False) extending
         beyond figure bounds.

    Significance brackets are drawn by ``significance_brackets.py``
    using ``clip_on=False``, ``zorder=10-11``, fontweight=bold, and
    text content matching ``*``, ``**``, ``***``, or ``ns``.
    """
    issues: list[dict] = []
    fig_bb = _fig_bbox(fig)
    star_pattern = re.compile(r'^(\*{1,3}|ns)$')

    # Collect all significance annotation texts and bracket lines
    star_texts = []
    bracket_lines = []

    for ax in fig.get_axes():
        for child in ax.get_children():
            if hasattr(child, 'get_text'):
                txt = child.get_text().strip()
                if star_pattern.match(txt) and child.get_visible():
                    bb = _safe_bbox(child, renderer)
                    if bb:
                        star_texts.append((txt, bb, child))

            # Lines with zorder >= 10 and clip_on=False are likely brackets
            if hasattr(child, 'get_xdata') and hasattr(child, 'get_zorder'):
                if child.get_zorder() >= 10 and not child.get_clip_on():
                    bb = _safe_bbox(child, renderer)
                    if bb:
                        bracket_lines.append(("bracket_line", bb, child))

    # a) Check star texts for truncation at figure border
    for txt, bb, artist in star_texts:
        sides = _sides_outside(bb, fig_bb, border_tol_px)
        if sides:
            issues.append({
                "type": "bracket_text_truncation",
                "severity": "warning",
                "detail": (
                    f"Significance text '{txt}' extends beyond "
                    f"figure border ({', '.join(sides)})"
                ),
                "elements": [f"sig_text:{txt}"],
            })

    # b) Check star texts overlapping other (non-bracket) texts
    all_texts = []
    for ax in fig.get_axes():
        for child in ax.get_children():
            if hasattr(child, 'get_text') and child.get_visible():
                other_txt = child.get_text().strip()
                if other_txt and not star_pattern.match(other_txt):
                    bb = _safe_bbox(child, renderer)
                    if bb:
                        all_texts.append((other_txt[:25], bb))

    for sig_txt, sig_bb, _ in star_texts:
        for other_txt, other_bb in all_texts:
            shrunk_sig = _shrink(sig_bb, 1.0)
            shrunk_other = _shrink(other_bb, 1.0)
            if shrunk_sig and shrunk_other and shrunk_sig.overlaps(shrunk_other):
                area = _overlap_area(sig_bb, other_bb)
                if area > 10:
                    issues.append({
                        "type": "bracket_text_overlap",
                        "severity": "warning",
                        "detail": (
                            f"Significance '{sig_txt}' overlaps "
                            f"label '{other_txt}' ({area:.0f} px\u00b2)"
                        ),
                        "elements": [f"sig_text:{sig_txt}",
                                     f"label:{other_txt}"],
                    })

    # c) Check bracket lines for figure-border truncation
    for tag, bb, artist in bracket_lines:
        sides = _sides_outside(bb, fig_bb, border_tol_px)
        if sides:
            issues.append({
                "type": "bracket_line_truncation",
                "severity": "warning",
                "detail": (
                    f"Bracket line extends beyond "
                    f"figure border ({', '.join(sides)})"
                ),
                "elements": [tag],
            })

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Per-axes summary helper
# ═══════════════════════════════════════════════════════════════════════════════

def _per_axes_summary(fig, renderer, infos):
    """Per-subplot conflict summary (Layer 1).

    Returns a dict mapping axes title -> list of issues in that subplot,
    useful for pinpointing which panels still have internal conflicts.
    """
    per_ax: dict[str, list[dict]] = {}

    for ax in fig.get_axes():
        if getattr(ax, '_is_legend_cell', False):
            continue
        title = ax.get_title() or f"ax@{id(ax):#x}"
        aid = id(ax)
        ax_issues: list[dict] = []

        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            per_ax[title] = ax_issues
            continue

        leg_bb = _safe_bbox(legend, renderer)
        if leg_bb is None:
            per_ax[title] = ax_issues
            continue

        local_data = [a for a in infos
                      if a.ax_id == aid
                      and a.kind in ("collection", "line", "image", "patch")
                      and not any(s in a.tag for s in ("Spine", "Wedge", "FancyBbox"))]

        for da in local_data:
            area = _overlap_area(leg_bb, da.bbox)
            if area > 30:
                frac = area / (da.bbox.width * da.bbox.height + 1e-8)
                if frac > 0.03:
                    # Lines span the full axes; legend overlap is expected.
                    # Rectangle (bar) patches in grouped bar charts also
                    # extend across the full axes, so legend overlap is
                    # unavoidable — downgrade to info like lines.
                    # FillBetween polys span the full axes like lines.
                    if (da.kind == "line" or da.kind == "patch"
                            or "Rectangle" in da.tag
                            or "FillBetween" in da.tag):
                        sev = "info"
                    else:
                        sev = "warning"
                    ax_issues.append({
                        "type": "subplot_legend_overlap",
                        "severity": sev,
                        "detail": (
                            f"[{title}] Legend occludes '{da.tag}' "
                            f"({frac:.0%}, {area:.0f} px\u00b2)"
                        ),
                    })
        per_ax[title] = ax_issues
    return per_ax


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 19: Font-size adequacy detection
# ═══════════════════════════════════════════════════════════════════════════════

def _check_fontsize_adequacy(
    fig,
    renderer,
    infos: list[_ArtistInfo],
    min_pt: float = 6.0,
    composed_scale: float = 0.5,
    dense_label_min_pt: float = 5.5,
) -> list[dict]:
    """Detect text whose effective print size falls below *min_pt*.

    The *composed_scale* parameter approximates the downscaling that
    occurs when the subplot PNG is placed inside the Next.js compositor
    (e.g. a 3-column grid in a 2-panel layout -> each subplot rendered
    at ~50% of its original size).  The effective font size is::

        effective_pt = artist.get_fontsize() * composed_scale

    Any text below *min_pt* after scaling is flagged.  This prevents
    tiny illegible labels from reaching the final PDF.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    renderer : matplotlib renderer
    infos : list of _ArtistInfo
    min_pt : float
        Minimum acceptable point size in the composed figure (default 6).
    composed_scale : float
        Approximate scale factor from subplot PNG to composed screenshot
        (default 0.5 for 2-panel layout with 3-column grid).

    Returns
    -------
    list[dict]
        Issues with ``type='fontsize_too_small'``.
    """
    issues: list[dict] = []
    seen_sizes: dict[float, list[str]] = {}  # group by fontsize for summary

    for info in infos:
        if info.kind != "text":
            continue
        artist = info.artist
        if not isinstance(artist, Text):
            continue
        text_str = artist.get_text().strip()
        if not text_str:
            continue
        # Skip internal matplotlib artists (underscore prefix)
        label = getattr(artist, '_label', '') or ''
        if label.startswith('_'):
            continue

        fs = artist.get_fontsize()
        effective = fs * composed_scale
        # Dense labels (tagged via gid or very small tick labels) use a
        # relaxed threshold to avoid false alarms on intentionally compact
        # heatmap / dense bar-chart ticks.
        gid = getattr(artist, '_gid', None) or ''
        is_dense = 'dense_label' in str(gid) or info.tag.startswith(('xtick', 'ytick', 'cbar_tick'))
        threshold = dense_label_min_pt if is_dense else min_pt
        if effective < threshold:
            seen_sizes.setdefault(fs, []).append(text_str[:30])

    # Emit one warning per distinct fontsize (avoids 50+ duplicate warnings)
    for fs, examples in sorted(seen_sizes.items()):
        effective = fs * composed_scale
        n = len(examples)
        sample = examples[:3]
        sample_str = ", ".join(f"'{s}'" for s in sample)
        if n > 3:
            sample_str += f" ... +{n - 3} more"
        issues.append({
            "type": "fontsize_too_small",
            "severity": "warning",
            "detail": (
                f"{n} text(s) at {fs:.1f}pt \u2192 effective {effective:.1f}pt "
                f"(min {min_pt}pt): {sample_str}"
            ),
            "elements": examples,
        })

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 20: Tick-spine overlap detection
# ═══════════════════════════════════════════════════════════════════════════════

def _check_tick_spine_overlap(fig, renderer, tol_px=2.0):
    """Pass 20: Detect tick labels rendered *inside* the data area.

    Tick labels normally sit outside the axes.  This check flags labels
    whose bbox is **entirely** inside the axes area (i.e. the label is
    rendered on top of data content rather than in the margin).

    Normal adjacent-to-spine positioning is NOT flagged — only labels
    whose full extent is within the plot rectangle.
    """
    issues = []
    for ax in fig.get_axes():
        if getattr(ax, 'name', None) == 'polar':
            continue
        ax_bb = _safe_bbox(ax, renderer)
        if ax_bb is None:
            continue

        # X-tick labels: flag only if the ENTIRE label is above the bottom spine
        for tl in ax.get_xticklabels():
            txt = tl.get_text().strip()
            if not txt:
                continue
            bb = _safe_bbox(tl, renderer)
            if bb is None:
                continue
            # Label is "inside" if its bottom edge is above the spine
            if bb.y0 > ax_bb.y0 + tol_px and bb.y1 < ax_bb.y1 - tol_px:
                issues.append({
                    "type": "tick_spine_overlap",
                    "severity": "warning",
                    "detail": (
                        f"X-tick '{txt[:20]}' rendered inside plot area"
                    ),
                    "elements": [f"xtick:{txt[:20]}"],
                })

        # Y-tick labels: flag only if the ENTIRE label is to the right of the left spine
        for tl in ax.get_yticklabels():
            txt = tl.get_text().strip()
            if not txt:
                continue
            bb = _safe_bbox(tl, renderer)
            if bb is None:
                continue
            # Label is "inside" if its left edge is past the left spine
            if bb.x0 > ax_bb.x0 + tol_px and bb.x1 < ax_bb.x1 - tol_px:
                issues.append({
                    "type": "tick_spine_overlap",
                    "severity": "warning",
                    "detail": (
                        f"Y-tick '{txt[:20]}' rendered inside plot area"
                    ),
                    "elements": [f"ytick:{txt[:20]}"],
                })
    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 21: Global font consistency check
# ═══════════════════════════════════════════════════════════════════════════════

def _check_font_policy(fig, renderer, infos,
                        allowed_families=None,
                        max_title_label_diff=2.0):
    """Pass 21: Enforce global font consistency.

    Checks:
      a) Font family must be in allowed_families (Arial/Helvetica/DejaVu Sans).
      b) Title vs label font sizes should differ by at most max_title_label_diff pt.
      c) Flag any fontweight='bold' usage (except in explicit whitelist).
    """
    if allowed_families is None:
        allowed_families = {"Arial", "Helvetica", "DejaVu Sans", "sans-serif"}

    issues = []
    bold_count = 0
    wrong_family_examples = []
    title_sizes = []
    label_sizes = []

    for info in infos:
        if info.kind != "text":
            continue
        artist = info.artist
        if not isinstance(artist, Text):
            continue
        txt = artist.get_text().strip()
        if not txt:
            continue

        # Check font family
        family = artist.get_fontfamily()
        if family:
            top_family = family[0] if isinstance(family, list) else family
            if top_family not in allowed_families:
                wrong_family_examples.append(f"'{txt[:25]}' uses {top_family}")

        # Collect title vs label sizes
        if info.tag.startswith(("title", "suptitle")):
            title_sizes.append(artist.get_fontsize())
        elif info.tag.startswith(("xlabel", "ylabel")):
            label_sizes.append(artist.get_fontsize())

        # Check bold usage
        weight = artist.get_fontweight()
        if weight in ("bold", "heavy", "extra bold", 700, 800, 900):
            bold_count += 1

    # Report wrong families
    if wrong_family_examples:
        sample = wrong_family_examples[:3]
        issues.append({
            "type": "font_family_violation",
            "severity": "warning",
            "detail": (
                f"{len(wrong_family_examples)} text(s) use non-standard font: "
                f"{'; '.join(sample)}"
                + (f" ... +{len(wrong_family_examples) - 3} more"
                   if len(wrong_family_examples) > 3 else "")
            ),
            "elements": wrong_family_examples[:5],
        })

    # Report title/label size mismatch
    if title_sizes and label_sizes:
        mean_title = sum(title_sizes) / len(title_sizes)
        mean_label = sum(label_sizes) / len(label_sizes)
        diff = abs(mean_title - mean_label)
        if diff > max_title_label_diff:
            issues.append({
                "type": "title_label_size_mismatch",
                "severity": "info",
                "detail": (
                    f"Title avg {mean_title:.1f}pt vs label avg "
                    f"{mean_label:.1f}pt (diff={diff:.1f}pt, "
                    f"max={max_title_label_diff}pt)"
                ),
                "elements": [],
            })

    # Report bold usage
    if bold_count > 0:
        issues.append({
            "type": "bold_usage",
            "severity": "info",
            "detail": f"{bold_count} text(s) use bold fontweight",
            "elements": [],
        })

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 22: Label density excess detection
# ═══════════════════════════════════════════════════════════════════════════════

def _check_label_density(fig, renderer, infos, density_threshold=0.70):
    """Pass 22: Detect axes where tick labels consume too much of the axis width/height.

    For each axes, estimates the total display width of visible tick labels
    and compares it to the available axis dimension. If the total label
    footprint exceeds *density_threshold* (fraction) of the axis extent,
    emits a ``label_density_excess`` warning.

    This check enables the auto-refine loop to make *structural* layout
    changes (fewer subplots per row, rotate labels) rather than just
    growing margins.
    """
    issues: list[dict] = []

    for ax in fig.get_axes():
        if getattr(ax, 'name', None) == 'polar':
            continue
        ax_bb = _safe_bbox(ax, renderer)
        if ax_bb is None or ax_bb.width < 1 or ax_bb.height < 1:
            continue

        title = ax.get_title() or f"ax@{id(ax):#x}"

        # -- X-tick label density --
        xtick_widths = []
        max_xtick_len = 0
        for tl in ax.get_xticklabels():
            txt = tl.get_text().strip()
            if not txt:
                continue
            max_xtick_len = max(max_xtick_len, len(txt))
            bb = _safe_bbox(tl, renderer)
            if bb:
                xtick_widths.append(bb.width)

        if xtick_widths:
            total_w = sum(xtick_widths)
            ratio_x = total_w / ax_bb.width
            if ratio_x > density_threshold:
                issues.append({
                    "type": "label_density_excess",
                    "severity": "warning",
                    "detail": (
                        f"X-tick labels in '{title}' fill {ratio_x:.0%} of axis width "
                        f"({len(xtick_widths)} labels, max_len={max_xtick_len} chars)"
                    ),
                    "elements": [f"xtick_density:{title}"],
                    "axis_kind": "xtick",
                    "axes_title": title,
                    "num_labels": len(xtick_widths),
                    "max_label_length": max_xtick_len,
                    "density_ratio": ratio_x,
                })

        # -- Y-tick label density --
        ytick_heights = []
        max_ytick_len = 0
        for tl in ax.get_yticklabels():
            txt = tl.get_text().strip()
            if not txt:
                continue
            max_ytick_len = max(max_ytick_len, len(txt))
            bb = _safe_bbox(tl, renderer)
            if bb:
                ytick_heights.append(bb.height)

        if ytick_heights:
            total_h = sum(ytick_heights)
            ratio_y = total_h / ax_bb.height
            if ratio_y > density_threshold:
                issues.append({
                    "type": "label_density_excess",
                    "severity": "warning",
                    "detail": (
                        f"Y-tick labels in '{title}' fill {ratio_y:.0%} of axis height "
                        f"({len(ytick_heights)} labels, max_len={max_ytick_len} chars)"
                    ),
                    "elements": [f"ytick_density:{title}"],
                    "axis_kind": "ytick",
                    "axes_title": title,
                    "num_labels": len(ytick_heights),
                    "max_label_length": max_ytick_len,
                    "density_ratio": ratio_y,
                })

    return issues
