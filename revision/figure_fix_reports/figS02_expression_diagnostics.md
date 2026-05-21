# Figure Fix Report — `figS02_expression_diagnostics`

**Producer:** `src/visualization/figS02_expression_diagnostics.py`
**Findings:** CRITICAL=33, MAJOR=186, MINOR=12, INFO=2

## CRITICAL (33)

### `text_truncation` × 32
- 'annotation: (a)' extends beyond figure border (top, 15px)
- 'annotation: (b)' extends beyond figure border (top, 15px)
- 'annotation: (c)' extends beyond figure border (top, 15px)
- 'annotation: (d)' extends beyond figure border (top, 15px)
- … +28 more
- **Fix:** Increase axis margins (`plt.subplots_adjust(top=..., bottom=...)`) or tighten the tick locator via `ax.xaxis.set_major_locator(MaxNLocator(nbins=N, prune='upper'))`.

### `cross_axes_text_overlap` × 1
- xtick '1.0' (axes 0) overlaps xtick '0.99985' (axes 1) -- 158 px²; consider increasing hspace/wspace
- **Fix:** Increase `hspace`/`wspace` in the gridspec, or prune the offending tick with `prune='upper'`/`'lower'` to remove the colliding label.

## MAJOR (186)

### `minimum_font_size` × 164
- text '% discrim. wt.' renders at 6.5pt (nominal 12.0pt × scale 0.54), below 7.0pt
- text '0.012' renders at 5.4pt (nominal 10.0pt × scale 0.54), below 7.0pt
- text '0.014' renders at 5.4pt (nominal 10.0pt × scale 0.54), below 7.0pt
- text '0.015' renders at 5.4pt (nominal 10.0pt × scale 0.54), below 7.0pt
- … +160 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `legend_data_occlusion` × 8
- Legend in same panel occludes 'patch: Rectangle' (71%, 514 px²)
- Legend in same panel occludes 'patch: Rectangle' (71%, 514 px²)
- Legend in same panel occludes 'patch: Rectangle' (71%, 514 px²)
- Legend in same panel occludes 'patch: Rectangle' (71%, 514 px²)
- … +4 more
- **Fix:** Move the legend outside the axes or to a less-populated quadrant.

### `colorblind_confusable` × 3
- colors 'line: Mean = 1.000' and 'patch: _nolegend_' collapse under deuteranopia (ΔE₇₆ = 5.6 < 12.0)
- colors 'line: Median = 1.000' and 'patch: _nolegend_' collapse under protanopia (ΔE₇₆ = 8.6 < 12.0)
- colors 'line: r = 0.8' and 'patch: _nolegend_' collapse under deuteranopia (ΔE₇₆ = 11.0 < 12.0)
- **Fix:** Replace the offending color pair. Safe defaults: orange #E69F00, blue #0072B2, vermillion #D55E00, bluish-green #009E73 (Wong 2011 palette).

### `text_overlap` × 3
- 'xlabel: Training cells (log scale)' overlaps 'xtick: $\mathdefault{1.2\times10^{4}}$'
- 'xtick: 1.0' overlaps 'xtick: 0.99985'
- 'xtick: 1.5' overlaps 'xtick: 0.99995'
- **Fix:** Raise panel-label Y (e.g. `add_panel_label(ax, 'b', y=1.12)`) or drop the last annotation so the axis tick below can breathe.

### `legend_text_truncation` × 2
- Legend text 'Chance (0.5)' extends beyond figure border (bottom)
- Legend text 'Mean = 0.924' extends beyond figure border (bottom)
- **Fix:** Investigate the offending artist and relocate or resize.

### `legend_truncation` × 2
- 'legend_box' extends beyond figure border (bottom, 35px)
- Legend in axes extends beyond figure (bottom)
- **Fix:** Investigate the offending artist and relocate or resize.

### `text_artist_overlap` × 2
- Text 'legend_text: Median = 1.000' overlaps content 'patch: Rectangle' (154 px²)
- Text 'legend_text: Median = 1.000' overlaps content 'patch: Rectangle' (328 px²)
- **Fix:** Move annotations off the data region via `bbox_to_anchor`, or reduce the annotation count if the panel is already dense.

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 186 (fig_dpi=100.0, fig_width=13.0in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.

### `label_density_excess` × 1
- X-tick labels in 'Pseudobulk Correlation Dist.' fill 153% of axis width (4 labels, max_len=7 chars)
- **Fix:** Investigate the offending artist and relocate or resize.

## MINOR (12)

### `legend_artist_masking` × 8
- Legend masks 'patch: Rectangle' (71%, 514 px²)
- Legend masks 'patch: Rectangle' (71%, 514 px²)
- Legend masks 'patch: Rectangle' (71%, 514 px²)
- Legend masks 'patch: Rectangle' (71%, 514 px²)
- … +4 more
- **Fix:** Switch legend `loc=` to a corner free of data, or set `frameon=False` and use `bbox_to_anchor` to move it outside the axes.

### `label_string_ellipsis` × 2
- axes 'Per-Type Pseudobulk Corr.' has 10 pre-truncated labels (e.g. Multiciliated…, Tissue-reside…, Pituitary gon…)
- axes 'Per-Type Separability' has 11 pre-truncated labels (e.g. Pluripotent s…, Pancreatic be…, Urothelial (t…)
- **Fix:** Drop the pre-truncated label and let matplotlib auto-wrap, OR abbreviate to a valid short name. Ellipsis hides biological identifiers from reviewers.

### `annotation_data_overlap` × 1
- Text 'annotation: n = 69 types
r > 0.9: 100%
r > 0.8: 100%' overlaps content 'patch: Rectangle' (227 px²)
- **Fix:** Reposition or drop redundant annotations; the data artist is hidden behind text.

### `precision_excess` × 1
- 4 label(s) with >4 decimal places: '0.99985' (5 decimals); '0.99990' (5 decimals); '0.99995' (5 decimals); '1.00000' (5 decimals)
- **Fix:** Investigate the offending artist and relocate or resize.

## INFO (2)

### `bold_usage` × 1
- 1 text(s) use bold fontweight
- **Fix:** Investigate the offending artist and relocate or resize.

### `scale_inconsistency` × 1
- X-axis label 'Pearson r' shared by 2 panels with 11113.4× range spread: Per-Type Pseudobulk Corr., Pseudobulk Correlation Di
- **Fix:** Investigate the offending artist and relocate or resize.
