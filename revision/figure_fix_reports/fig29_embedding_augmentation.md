# Figure Fix Report — `fig29_embedding_augmentation`

**Producer:** `src/visualization/fig29_embedding_augmentation.py`
**Findings:** CRITICAL=1, MAJOR=58, MINOR=1, INFO=0

## CRITICAL (1)

### `text_truncation` × 1
- 'ytick: 0.0020' extends beyond figure border (top, 3px)
- **Fix:** Increase axis margins (`plt.subplots_adjust(top=..., bottom=...)`) or tighten the tick locator via `ax.xaxis.set_major_locator(MaxNLocator(nbins=N, prune='upper'))`.

## MAJOR (58)

### `minimum_font_size` × 54
- text '+0.0006' renders at 5.0pt (nominal 10.0pt × scale 0.50), below 7.0pt
- text '+0.0007' renders at 5.0pt (nominal 10.0pt × scale 0.50), below 7.0pt
- text '+0.0007' renders at 5.0pt (nominal 10.0pt × scale 0.50), below 7.0pt
- text '+0.0009' renders at 5.0pt (nominal 10.0pt × scale 0.50), below 7.0pt
- … +50 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `text_overlap` × 2
- 'ytick: 0.00125' overlaps 'annotation: (c)'
- 'ytick: 0.866' overlaps 'annotation: (a)'
- **Fix:** Raise panel-label Y (e.g. `add_panel_label(ax, 'b', y=1.12)`) or drop the last annotation so the axis tick below can breathe.

### `colorblind_confusable` × 1
- colors 'line: _nolegend_' and 'patch: _nolegend_' collapse under deuteranopia (ΔE₇₆ = 11.0 < 12.0)
- **Fix:** Replace the offending color pair. Safe defaults: orange #E69F00, blue #0072B2, vermillion #D55E00, bluish-green #009E73 (Wong 2011 palette).

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 200 (fig_dpi=100.0, fig_width=14.0in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.

## MINOR (1)

### `precision_excess` × 1
- 11 label(s) with >4 decimal places: '−0.00125' (5 decimals); '−0.00100' (5 decimals); '−0.00075' (5 decimals); '−0.00050' (5 decimals) ... +7 more
- **Fix:** Investigate the offending artist and relocate or resize.
