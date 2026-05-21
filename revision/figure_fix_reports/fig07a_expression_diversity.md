# Figure Fix Report — `fig07a_expression_diversity`

**Producer:** `src/visualization/fig14_expr_diversity.py`
**Findings:** CRITICAL=1, MAJOR=29, MINOR=0, INFO=0

## CRITICAL (1)

### `text_truncation` × 1
- 'ytick: 1.0' extends beyond figure border (top, 8px)
- **Fix:** Increase axis margins (`plt.subplots_adjust(top=..., bottom=...)`) or tighten the tick locator via `ax.xaxis.set_major_locator(MaxNLocator(nbins=N, prune='upper'))`.

## MAJOR (29)

### `minimum_font_size` × 27
- text '(a)' renders at 6.4pt (nominal 14.0pt × scale 0.46), below 7.0pt
- text '(b)' renders at 6.4pt (nominal 14.0pt × scale 0.46), below 7.0pt
- text 'Expression Variability Summary' renders at 5.5pt (nominal 12.0pt × scale 0.46), below 7.0pt
- text 'Gene Std Ratio (gen / real)' renders at 5.1pt (nominal 11.0pt × scale 0.46), below 7.0pt
- … +23 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `colorblind_confusable` × 1
- colors 'patch: _nolegend_' and 'patch: _nolegend_' collapse under deuteranopia (ΔE₇₆ = 11.0 < 12.0)
- **Fix:** Replace the offending color pair. Safe defaults: orange #E69F00, blue #0072B2, vermillion #D55E00, bluish-green #009E73 (Wong 2011 palette).

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 217 (fig_dpi=100.0, fig_width=15.2in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.
