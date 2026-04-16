# Figure Fix Report — `fig25_cross_dataset`

**Producer:** `src/visualization/fig25_cross_dataset.py`
**Findings:** CRITICAL=0, MAJOR=49, MINOR=0, INFO=0

## MAJOR (49)

### `minimum_font_size` × 47
- text '(a)' renders at 6.8pt (nominal 14.0pt × scale 0.48), below 7.0pt
- text '(b)' renders at 6.8pt (nominal 14.0pt × scale 0.48), below 7.0pt
- text '(c)' renders at 6.8pt (nominal 14.0pt × scale 0.48), below 7.0pt
- text 'Correlation' renders at 5.8pt (nominal 12.0pt × scale 0.48), below 7.0pt
- … +43 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `colorblind_confusable` × 1
- colors 'patch: _nolegend_' and 'patch: _nolegend_' collapse under deuteranopia (ΔE₇₆ = 11.0 < 12.0)
- **Fix:** Replace the offending color pair. Safe defaults: orange #E69F00, blue #0072B2, vermillion #D55E00, bluish-green #009E73 (Wong 2011 palette).

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 207 (fig_dpi=100.0, fig_width=14.5in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.
