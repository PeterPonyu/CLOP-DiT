# Figure Fix Report — `fig09a_variance_matching`

**Producer:** `scripts/analysis/variance_matching_pilot.py`
**Findings:** CRITICAL=4, MAJOR=96, MINOR=6, INFO=0

## CRITICAL (4)

### `text_truncation` × 3
- 'annotation: (a)' extends beyond figure border (top, 4px)
- 'annotation: (b)' extends beyond figure border (top, 4px)
- 'ytick: 0.000' extends beyond figure border (bottom, 4px)
- **Fix:** Increase axis margins (`plt.subplots_adjust(top=..., bottom=...)`) or tighten the tick locator via `ax.xaxis.set_major_locator(MaxNLocator(nbins=N, prune='upper'))`.

### `cross_axes_text_overlap` × 1
- xtick '0.3' (axes 1) overlaps ytick '0.030' (axes 3) -- 169 px²; consider increasing hspace/wspace
- **Fix:** Increase `hspace`/`wspace` in the gridspec, or prune the offending tick with `prune='upper'`/`'lower'` to remove the colliding label.

## MAJOR (96)

### `minimum_font_size` × 86
- text 'Cumulative Proportion' renders at 6.0pt (nominal 12.0pt × scale 0.50), below 7.0pt
- text 'Dimension-Wise Variance Correlation' renders at 6.0pt (nominal 12.0pt × scale 0.50), below 7.0pt
- text 'Dorsal horn senso…' renders at 3.5pt (nominal 7.0pt × scale 0.50), below 7.0pt
- text 'Inhibitory GABAer…' renders at 3.5pt (nominal 7.0pt × scale 0.50), below 7.0pt
- … +82 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `text_overlap` × 5
- 'annotation: Neuroendocrine ar…' overlaps 'annotation: Dorsal horn senso…'
- 'annotation: n=2,004' overlaps 'annotation: n=935'
- 'annotation: n=656' overlaps 'annotation: n=727'
- 'annotation: n=727' overlaps 'annotation: n=853'
- … +1 more
- **Fix:** Raise panel-label Y (e.g. `add_panel_label(ax, 'b', y=1.12)`) or drop the last annotation so the axis tick below can breathe.

### `text_artist_overlap` × 2
- Text 'legend_text: Median = 0.174' overlaps content 'line: Line2D(Observed ECDF)' (819 px²)
- Text 'legend_text: Observed ECDF' overlaps content 'line: Line2D(Observed ECDF)' (873 px²)
- **Fix:** Move annotations off the data region via `bbox_to_anchor`, or reduce the annotation count if the panel is already dense.

### `colorblind_confusable` × 1
- colors 'patch: _nolegend_' and 'patch: 0.9–1.1 band' collapse under deuteranopia (ΔE₇₆ = 11.0 < 12.0)
- **Fix:** Replace the offending color pair. Safe defaults: orange #E69F00, blue #0072B2, vermillion #D55E00, bluish-green #009E73 (Wong 2011 palette).

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 200 (fig_dpi=100.0, fig_width=14.0in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.

### `label_density_excess` × 1
- Y-tick labels in 'Latent SWD per Cell Type' fill 94% of axis height (24 labels, max_len=18 chars)
- **Fix:** Investigate the offending artist and relocate or resize.

## MINOR (6)

### `annotation_data_overlap` × 2
- Text 'annotation: Mean = 0.201
Median = 0.174
100% positive' overlaps content 'line: Line2D(Observed ECDF)' (2663 px²)
- Text 'annotation: Median = 0.703
IQR = [0.565, 0.817]' overlaps content 'patch: Rectangle(0.9–1.1 band)' (2582 px²)
- **Fix:** Reposition or drop redundant annotations; the data artist is hidden behind text.

### `label_string_ellipsis` × 2
- axes 'Latent SWD per Cell Type' has 24 pre-truncated labels (e.g. Inhibitory GABAer…, Mast are tissue-r…, Multiciliated epi…)
- axes 'SWD vs. Training Cell Count' has 3 pre-truncated labels (e.g. Neuroendocrine ar…, Dorsal horn senso…, Inhibitory GABAer…)
- **Fix:** Drop the pre-truncated label and let matplotlib auto-wrap, OR abbreviate to a valid short name. Ellipsis hides biological identifiers from reviewers.

### `legend_artist_masking` × 2
- Legend masks 'annotation: Inhibitory GABAer…' (9%, 194 px²)
- Legend masks 'line: Line2D(Observed ECDF)' (4%, 4303 px²)
- **Fix:** Switch legend `loc=` to a corner free of data, or set `frameon=False` and use `bbox_to_anchor` to move it outside the axes.
