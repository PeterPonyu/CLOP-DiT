# Figure Presentation Policy

*Last revised: 2026-03-07*

This document defines presentation rules for CLOP-DiT figures so panels stay readable and consistent. It complements [LEGEND_CAPTION_POLICY.md](LEGEND_CAPTION_POLICY.md) and is referenced by [FIGURE_ORGANIZATION.md](FIGURE_ORGANIZATION.md). Enforce these rules in code review.

## 1. Annotation budget

- **At most one stats box per subplot.** Short statistics (e.g. "r = 0.95") belong in a single small text annotation per subplot, not multiple boxes. If more is needed, move narrative to the caption.
- **At most two highlighted outliers per subplot** unless scientifically necessary (e.g. a specific gene or type must be called out). Avoid turning annotations into a second caption.

## 2. Primary message per panel

- **Each subplot should answer one question only.** If a panel needs title + legend + stats box + multiple callouts to be understood, it is overloaded.
- **Split or simplify:** either break into more panels or reduce annotation so the primary message is clear without clutter.

## 3. Shared semantics (colors and thresholds)

Use consistent visual language across panels so readers are not taught different semantics in different figures.

- **Good / warn / bad:** Use the semantic colours in `src/visualization/style.py`: `COLORS["good"]` (green), `COLORS["warn"]` (orange), `COLORS["bad"]` (red). Do not hardcode `#4CAF50` / `#FF9800` / `#F44336` for good/warn/bad; use `quality_color(value, thresholds)` or the shared helpers.
- **Default thresholds for quality_color:** The standard visual bands use `(0.8, 0.5)` (high, low). Document in code when a different tuple is **scientific** (e.g. method-specific cut-off) vs **visual** (policy band); visual bands should use the shared helper and shared thresholds where possible.
- **Diverging colormaps:** Use diverging colormaps (e.g. RdBu_r) only when the quantity has a natural midpoint (e.g. correlation, difference). Do not use them for one-sided metrics.
- **Absolute thresholds (0.9, 0.5, etc.):** When they denote a scientific cut-off (e.g. memorization 0.01, ratio=1.0), keep them as named constants or parameters and document. When they are only for visual good/warn/bad bands, use the shared threshold-band helper and standard semantics.

## 4. Text density ceiling

- **No subplot should require more than three text elements beyond axes labels and ticks.** This includes: subplot title, one stats box, and at most a small number of highlight labels (capped by the annotation budget). If you need more, move content to the caption or split the panel.

## 5. Standalone vs article

- **Article-facing panels** (the 15 figures in `ARTICLE_FIGURE_BASENAMES`): Optimize for print clarity and minimal annotation. They appear in the LaTeX manuscript; keep stats in caption or one short annotation, avoid report-style density.
- **Report-only panels** (legacy standalone panels used only in the full report PDF): May carry more diagnostics (e.g. extra callouts, secondary metrics). Do not let this density leak into the code paths that produce article figures; merged and article standalone panels must follow the annotation budget and text density ceiling above.

## 6. Figure title position and line wrapping

- **Use `style.set_figure_suptitle(fig, text, ...)`** instead of raw `fig.suptitle()` so that (1) the vertical position is consistent (`SUPTITLE_Y = 0.98`) and (2) long titles are wrapped at word boundaries (`SUPTITLE_MAX_CHARS_PER_LINE = 48`) to avoid overflow and overlap.
- **Long text in stat boxes or labels:** Use `style.wrap_text()` when a single line could exceed the panel width, or pass `wrap_chars` to `add_stat_box()` for inline wrapping.
- **Long axis or tick labels:** Prefer short labels, abbreviations, or symbols in the figure; document full terms or lists in the figure caption (e.g. "Types A–D: see caption") so the panel stays readable.

## 7. Colorbar placement

- **Colorbars must not overlap main figure content.** Use `style.add_colorbar_safe()` for all article and report colorbars so placement is consistent: `pad` (gap from axes) and `shrink` keep the bar clear of plot area. Defaults: vertical `pad=0.10`, horizontal `pad=0.12` or `0.14`, `shrink=0.65`.

## 8. Font and tick consistency

- **Use style constants for typography:** `style.py` defines `FONT_SUPTITLE`, `FONT_TITLE`, `FONT_LABEL`, `FONT_TICK`, `FONT_TICK_DENSE`, `FONT_LEGEND`, `TICK_PAD_PT`. Use these (or the global `VIS_STYLE` defaults) so font sizes are consistent across all figures. Do not use fontsize=6 or arbitrary values; prefer `FONT_TICK_DENSE` (8) for dense axes and `FONT_TICK` (10) elsewhere.
- **Tick labels must not be truncated:** `xtick.major.pad` and `ytick.major.pad` are set to `TICK_PAD_PT` (4pt). Save uses `pad_inches=0.10` so rotated or long tick labels have room. If a panel still clips labels, increase pad or use `set_dense_tick_labels()` to reduce label count.

## 9. Panel labels (a), (b), (c), ... for article figures

- **All article-facing multi-panel figures MUST include panel labels** drawn directly on the figure image using `style.add_panel_label(ax, 'a')` or the batch helper `style.add_panel_labels_to_axes(axes)`.
- Panel labels use bold lowercase letters in parentheses: (a), (b), (c), etc., positioned at the top-left corner of each subplot (default x=0.02, y=0.98 in axes coordinates).
- Labels have a semi-transparent white background (alpha=0.85) to ensure readability over any plot content.
- Panel labels must not be clipped by tight_layout or figure boundaries.
- The LaTeX caption references panels using matching bold letters: `(\textbf{a})`, `(\textbf{b})`, etc.
- Figure-level suptitles are **not used** for article figures; figure-level information belongs in the LaTeX `\caption{}`. Subplot titles (e.g., "Loss Convergence") are retained.
- **Internal pipeline panel codes (A–S) and article Figure numbers (1–15)** appear only in code comments and LaTeX captions, never drawn on the figure image.

## Code and review checklist

- Use `style.set_figure_suptitle()` for all figure-level titles; do not call `fig.suptitle()` directly so position and wrapping stay consistent.
- Use `style.add_stat_box()` (or equivalent) for at most one stats box per subplot; use `style.add_highlight_labels(..., max_labels=2)` for outlier callouts unless scientifically necessary to show more.
- Use `style.add_threshold_bands()` and `quality_color()` for good/warn/bad bands and colours instead of ad hoc hex and thresholds.
- Use `style.add_colorbar_safe()` for all colorbars so they do not overlap main content (see §7).
- Use style font constants (FONT_*, TICK_PAD_PT) for consistency; avoid fontsize=6 or 12 (see §8).
- Use `style.add_panel_label(ax, label)` to add panel letters to every article-facing subplot (see §9).
- Do not use `set_figure_suptitle()` for article figures; use it only if needed for internal report-only figures.
- Do not draw "Figure N" on the figure; captions provide numbering (see §9).
- Review: each subplot has a single primary message; no subplot has more than three text elements beyond axes labels/ticks.
- Review: article-facing producers do not use legend titles containing statistics (see LEGEND_CAPTION_POLICY.md); contract test enforces this.
