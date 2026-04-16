# Figure Fix Report — `fig27_ood_robustness`

**Producer:** `src/visualization/fig27_ood_robustness.py`
**Findings:** CRITICAL=0, MAJOR=51, MINOR=0, INFO=0

## MAJOR (51)

### `minimum_font_size` × 50
- text '(a)' renders at 6.5pt (nominal 14.0pt × scale 0.47), below 7.0pt
- text '(b)' renders at 6.5pt (nominal 14.0pt × scale 0.47), below 7.0pt
- text '(c)' renders at 6.5pt (nominal 14.0pt × scale 0.47), below 7.0pt
- text '0.17 (1/6)' renders at 4.7pt (nominal 10.0pt × scale 0.47), below 7.0pt
- … +46 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 214 (fig_dpi=100.0, fig_width=15.0in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.
