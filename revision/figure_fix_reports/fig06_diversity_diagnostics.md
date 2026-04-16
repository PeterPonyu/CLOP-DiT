# Figure Fix Report — `fig06_diversity_diagnostics`

**Producer:** `src/visualization/fig12_diversity.py`
**Findings:** CRITICAL=0, MAJOR=92, MINOR=3, INFO=0

## MAJOR (92)

### `minimum_font_size` × 86
- text 'CFG Scale' renders at 6.0pt (nominal 12.0pt × scale 0.50), below 7.0pt
- text 'Cell Type' renders at 6.0pt (nominal 12.0pt × scale 0.50), below 7.0pt
- text 'Cosine / ratio' renders at 6.0pt (nominal 12.0pt × scale 0.50), below 7.0pt
- text 'Cosine Distance to Nearest Real Cell' renders at 6.0pt (nominal 12.0pt × scale 0.50), below 7.0pt
- … +82 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `colorblind_confusable` × 5
- colors 'line: ratio=0.5 (collapse)' and 'patch: _nolegend_' collapse under protanopia (ΔE₇₆ = 1.7 < 12.0)
- colors 'line: ratio=0.5 (collapse)' and 'patch: _nolegend_' collapse under protanopia (ΔE₇₆ = 6.0 < 12.0)
- colors 'patch: _nolegend_' and 'line: Centroid cos ↑' collapse under protanopia (ΔE₇₆ = 8.6 < 12.0)
- colors 'patch: _nolegend_' and 'line: Production ε=0.03' collapse under deuteranopia (ΔE₇₆ = 7.3 < 12.0)
- … +1 more
- **Fix:** Replace the offending color pair. Safe defaults: orange #E69F00, blue #0072B2, vermillion #D55E00, bluish-green #009E73 (Wong 2011 palette).

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 200 (fig_dpi=100.0, fig_width=14.0in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.

## MINOR (3)

### `label_string_ellipsis` × 2
- axes 'Centroid vs Noisy Conditioning' has 5 pre-truncated labels (e.g. Tissue-resi…, Erythroid l…, Pituitary t…)
- axes 'Intra-Type Diversity Ratio' has 7 pre-truncated labels (e.g. Dorsal horn s…, Capillary end…, Excitatory gl…)
- **Fix:** Drop the pre-truncated label and let matplotlib auto-wrap, OR abbreviate to a valid short name. Ellipsis hides biological identifiers from reviewers.

### `annotation_data_overlap` × 1
- Text 'annotation: ■ Norm' overlaps content 'line: Line2D(Div.)' (171 px²)
- **Fix:** Reposition or drop redundant annotations; the data artist is hidden behind text.
