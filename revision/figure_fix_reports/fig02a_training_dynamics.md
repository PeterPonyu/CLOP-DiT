# Figure Fix Report — `fig02a_training_dynamics`

**Producer:** `src/visualization/fig03_training.py`
**Findings:** CRITICAL=0, MAJOR=122, MINOR=11, INFO=2

## MAJOR (122)

### `minimum_font_size` × 101
- text '(a)' renders at 6.6pt (nominal 14.0pt × scale 0.47), below 7.0pt
- text '(b)' renders at 6.6pt (nominal 14.0pt × scale 0.47), below 7.0pt
- text '(c)' renders at 6.6pt (nominal 14.0pt × scale 0.47), below 7.0pt
- text '(d)' renders at 6.6pt (nominal 14.0pt × scale 0.47), below 7.0pt
- … +97 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `text_artist_overlap` × 10
- Text 'legend_text: 90% converged' overlaps content 'line: Line2D(CLOP val loss)' (826 px²)
- Text 'legend_text: 90% converged' overlaps content 'line: Line2D(DiT val loss)' (826 px²)
- Text 'legend_text: CLOP val loss' overlaps content 'line: Line2D(CLOP val loss)' (746 px²)
- Text 'legend_text: CLOP val loss' overlaps content 'line: Line2D(DiT val loss)' (746 px²)
- … +6 more
- **Fix:** Move annotations off the data region via `bbox_to_anchor`, or reduce the annotation count if the panel is already dense.

### `legend_data_occlusion` × 4
- Legend in same panel occludes 'line: Line2D(CLOP val loss)' (11%, 8812 px²)
- Legend in same panel occludes 'line: Line2D(DiT val loss)' (11%, 8812 px²)
- Legend in same panel occludes 'line: Line2D(Mean Cos)' (14%, 7160 px²)
- Legend in same panel occludes 'line: Line2D(Text↔Cell)' (9%, 2785 px²)
- **Fix:** Move the legend outside the axes or to a less-populated quadrant.

### `colorblind_confusable` × 3
- colors 'line: Train' and 'line: Val Acc' collapse under deuteranopia (ΔE₇₆ = 7.3 < 12.0)
- colors 'line: Val' and 'patch: unlabeled' collapse under protanopia (ΔE₇₆ = 8.6 < 12.0)
- colors 'patch: unlabeled' and 'line: Inter-sep' collapse under deuteranopia (ΔE₇₆ = 5.6 < 12.0)
- **Fix:** Replace the offending color pair. Safe defaults: orange #E69F00, blue #0072B2, vermillion #D55E00, bluish-green #009E73 (Wong 2011 palette).

### `text_overlap` × 3
- 'ytick: 1.2' overlaps 'annotation: (b)'
- 'ytick: 1.2' overlaps 'annotation: (d)'
- 'ytick: 120' overlaps 'annotation: (c)'
- **Fix:** Raise panel-label Y (e.g. `add_panel_label(ax, 'b', y=1.12)`) or drop the last annotation so the axis tick below can breathe.

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 211 (fig_dpi=100.0, fig_width=14.8in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.

## MINOR (11)

### `annotation_data_overlap` × 7
- Text 'annotation: Phase I: rapid
Phase II: refine
Phase III: converg' overlaps content 'line: Line2D(Train MSE)' (2556 px²)
- Text 'annotation: Phase I: rapid
Phase II: refine
Phase III: converg' overlaps content 'line: Line2D(Train)' (2507 px²)
- Text 'annotation: Phase I: rapid
Phase II: refine
Phase III: converg' overlaps content 'line: Line2D(Val MSE)' (2556 px²)
- Text 'annotation: Phase I: rapid
Phase II: refine
Phase III: converg' overlaps content 'patch: Rectangle' (1863 px²)
- … +3 more
- **Fix:** Reposition or drop redundant annotations; the data artist is hidden behind text.

### `legend_artist_masking` × 4
- Legend masks 'line: Line2D(CLOP val loss)' (11%, 8812 px²)
- Legend masks 'line: Line2D(DiT val loss)' (11%, 8812 px²)
- Legend masks 'line: Line2D(Mean Cos)' (14%, 7160 px²)
- Legend masks 'line: Line2D(Text↔Cell)' (9%, 2785 px²)
- **Fix:** Switch legend `loc=` to a corner free of data, or set `frameon=False` and use `bbox_to_anchor` to move it outside the axes.

## INFO (2)

### `log_scale_unlabelled` × 1
- Y-axis in 'DiT Loss' uses log scale but label 'flow-matching loss' does not indicate this
- **Fix:** Investigate the offending artist and relocate or resize.

### `overlapping_series_values` × 1
- 4 lines are coincident (>= 80% of x-range): Val Acc, Train Acc, Top-5, Top-10
- **Fix:** Drop the redundant series, or explicitly mark a ceiling (e.g. `ax.axhline(100, ls='--', label='ceiling')`).
