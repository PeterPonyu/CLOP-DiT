# Figure Fix Report — `fig28_marker_completeness`

**Producer:** `src/visualization/fig28_marker_completeness.py`
**Findings:** CRITICAL=0, MAJOR=163, MINOR=4, INFO=1

## MAJOR (163)

### `minimum_font_size` × 153
- text '(a)' renders at 5.9pt (nominal 14.0pt × scale 0.42), below 7.0pt
- text '(b)' renders at 5.9pt (nominal 14.0pt × scale 0.42), below 7.0pt
- text '(c)' renders at 5.9pt (nominal 14.0pt × scale 0.42), below 7.0pt
- text '-0.0003' renders at 3.8pt (nominal 9.0pt × scale 0.42), below 7.0pt
- … +149 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `colorblind_confusable` × 8
- colors 'line: Antibody-se…' and 'patch: _nolegend_' collapse under deuteranopia (ΔE₇₆ = 9.7 < 12.0)
- colors 'line: B lymph.' and 'line: Antibody-se…' collapse under deuteranopia (ΔE₇₆ = 7.3 < 12.0)
- colors 'line: CD4+ helper…' and 'line: B lymph.' collapse under protanopia (ΔE₇₆ = 4.6 < 12.0)
- colors 'line: CD4+ helper…' and 'line: Generic epi…' collapse under deuteranopia (ΔE₇₆ = 7.2 < 12.0)
- … +4 more
- **Fix:** Replace the offending color pair. Safe defaults: orange #E69F00, blue #0072B2, vermillion #D55E00, bluish-green #009E73 (Wong 2011 palette).

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 236 (fig_dpi=100.0, fig_width=16.5in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.

### `text_overlap` × 1
- 'ytick: Circulating…' overlaps 'annotation: -0.0004'
- **Fix:** Raise panel-label Y (e.g. `add_panel_label(ax, 'b', y=1.12)`) or drop the last annotation so the axis tick below can breathe.

## MINOR (4)

### `label_string_ellipsis` × 3
- axes 'Canonical Marker Recall@K' has 9 pre-truncated labels (e.g. CD8+ cytoto…, CD4+ helper…, Antibody-se…)
- axes 'Marker Enrichment' has 9 pre-truncated labels (e.g. CD8+ cytoto…, CD4+ helper…, Antibody-se…)
- axes 'Marker Recovery (Canonical + Extended)' has 9 pre-truncated labels (e.g. CD8+ cytoto…, CD4+ helper…, Antibody-se…)
- **Fix:** Drop the pre-truncated label and let matplotlib auto-wrap, OR abbreviate to a valid short name. Ellipsis hides biological identifiers from reviewers.

### `panel_complexity_excess` × 1
- Panel 'Marker Recovery (Canonical + Extended)' complexity score 43.3 (threshold 15.0): 80 numeric labels (>30); 81 text elements (>20). Consider simplifying or splitting.
- **Fix:** Investigate the offending artist and relocate or resize.

## INFO (1)

### `overlapping_series_values` × 1
- 4 lines are coincident (>= 80% of x-range): B lymph., Antibody-se…, Generic epi…, Vascular en…
- **Fix:** Drop the redundant series, or explicitly mark a ceiling (e.g. `ax.axhline(100, ls='--', label='ceiling')`).
