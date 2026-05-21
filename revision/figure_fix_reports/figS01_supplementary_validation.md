# Figure Fix Report — `figS01_supplementary_validation`

**Producer:** `src/visualization/figS01_supplementary_validation.py`
**Findings:** CRITICAL=33, MAJOR=265, MINOR=12, INFO=3

## CRITICAL (33)

### `text_truncation` × 25
- 'annotation: (a)' extends beyond figure border (left, top, 38px)
- 'annotation: (b)' extends beyond figure border (left, 67px)
- 'legend_text: Antibody-…' extends beyond figure border (bottom, 55px)
- 'legend_text: B lymph.' extends beyond figure border (bottom, 34px)
- … +21 more
- **Fix:** Increase axis margins (`plt.subplots_adjust(top=..., bottom=...)`) or tighten the tick locator via `ax.xaxis.set_major_locator(MaxNLocator(nbins=N, prune='upper'))`.

### `axes_overflow` × 5
- 'line: Line2D' extends beyond axes border (bottom)
- 'line: Line2D' extends beyond axes border (bottom)
- 'line: Line2D' extends beyond axes border (top)
- 'line: Line2D' extends beyond axes border (top)
- … +1 more
- **Fix:** Investigate the offending artist and relocate or resize.

### `cross_axes_text_overlap` × 3
- xtick '$\mathdefault{10^{1}' (axes 8) overlaps xtick '0.0' (axes 9) -- 224 px²; consider increasing hspace/wspace
- xtick '1.0' (axes 7) overlaps xtick '$\mathdefault{10^{-7' (axes 8) -- 261 px²; consider increasing hspace/wspace
- ytick '8' (axes 3) overlaps ytick '0.8' (axes 5) -- 112 px²; consider increasing hspace/wspace
- **Fix:** Increase `hspace`/`wspace` in the gridspec, or prune the offending tick with `prune='upper'`/`'lower'` to remove the colliding label.

## MAJOR (265)

### `minimum_font_size` × 207
- text '$\Delta$ F1' renders at 6.0pt (nominal 12.0pt × scale 0.50), below 7.0pt
- text '$\log_2$(var$_{\rm gen}$ / var$_{\rm rea' renders at 6.0pt (nominal 12.0pt × scale 0.50), below 7.0pt
- text '3.1' renders at 5.0pt (nominal 10.0pt × scale 0.50), below 7.0pt
- text '>2-fold deficit: 0.0%
of 1790 genes' renders at 5.5pt (nominal 11.0pt × scale 0.50), below 7.0pt
- … +203 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `text_overlap` × 19
- 'annotation: Interrogative' overlaps 'ytick: $\mathdefault{10^{1}}$'
- 'title: Variance Ratio Distribution' overlaps 'ytick: 80'
- 'xlabel: $\log_2$(var$_{\rm gen}$ / var$_{\rm real}$)' overlaps 'legend_text: Perfect match'
- 'xtick: $\mathdefault{10^{1}}$' overlaps 'xtick: 0.0'
- … +15 more
- **Fix:** Raise panel-label Y (e.g. `add_panel_label(ax, 'b', y=1.12)`) or drop the last annotation so the axis tick below can breathe.

### `text_artist_overlap` × 13
- Text 'cbarylabel: Normalized (higher = better)' overlaps content 'patch: Rectangle' (1049 px²)
- Text 'title: Multi-Seed Robustness' overlaps content 'line: Line2D' (2556 px²)
- Text 'title: Multi-Seed Robustness' overlaps content 'line: Line2D' (2556 px²)
- Text 'xtick: Best Epoch' overlaps content 'line: Line2D' (593 px²)
- … +9 more
- **Fix:** Move annotations off the data region via `bbox_to_anchor`, or reduce the annotation count if the panel is already dense.

### `legend_text_truncation` × 9
- Legend text 'Antibody-…' extends beyond figure border (bottom)
- Legend text 'B lymph.' extends beyond figure border (bottom)
- Legend text 'CD4+ help…' extends beyond figure border (bottom)
- Legend text 'CD8+ cyto…' extends beyond figure border (bottom)
- … +5 more
- **Fix:** Investigate the offending artist and relocate or resize.

### `legend_truncation` × 6
- 'legend_box' extends beyond figure border (bottom, 22px)
- 'legend_box' extends beyond figure border (bottom, 60px)
- 'legend_box' extends beyond figure border (bottom, right, 63px)
- Legend in axes extends beyond figure (bottom)
- … +2 more
- **Fix:** Investigate the offending artist and relocate or resize.

### `colorblind_confusable` × 5
- colors 'line: B lymph.' and 'line: Antibody-…' collapse under deuteranopia (ΔE₇₆ = 7.6 < 12.0)
- colors 'line: _child1' and 'patch: _nolegend_' collapse under deuteranopia (ΔE₇₆ = 11.0 < 12.0)
- colors 'line: _child4' and 'patch: _nolegend_' collapse under deuteranopia (ΔE₇₆ = 6.2 < 12.0)
- colors 'line: _nolegend_' and 'line: B lymph.' collapse under protanopia (ΔE₇₆ = 12.0 < 12.0)
- … +1 more
- **Fix:** Replace the offending color pair. Safe defaults: orange #E69F00, blue #0072B2, vermillion #D55E00, bluish-green #009E73 (Wong 2011 palette).

### `label_density_excess` × 4
- X-tick labels in 'Cross-Dataset' fill 127% of axis width (5 labels, max_len=7 chars)
- X-tick labels in 'DE Concord.' fill 167% of axis width (4 labels, max_len=10 chars)
- X-tick labels in 'Decoder Comparison' fill 138% of axis width (3 labels, max_len=14 chars)
- X-tick labels in 'Variance Scatter' fill 121% of axis width (6 labels, max_len=23 chars)
- **Fix:** Investigate the offending artist and relocate or resize.

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 200 (fig_dpi=100.0, fig_width=14.0in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.

### `low_contrast_text` × 1
- 1 text(s) with low contrast: '3.1' contrast=1.0:1
- **Fix:** Investigate the offending artist and relocate or resize.

## MINOR (12)

### `legend_artist_masking` × 9
- Legend masks 'annotation: 3.1' (23%, 62 px²)
- Legend masks 'xlabel: $\log_2$(var$_{\rm gen}$ / var$_{\rm real}$)' (52%, 1948 px²)
- Legend masks 'xlabel: Hit Rate' (31%, 275 px²)
- Legend masks 'xtick: MLP (trained)' (28%, 1281 px²)
- … +5 more
- **Fix:** Switch legend `loc=` to a corner free of data, or set `frameon=False` and use `bbox_to_anchor` to move it outside the axes.

### `label_string_ellipsis` × 3
- axes 'DE Concord.' has 5 pre-truncated labels (e.g. CD8+… v CD4+…, Tiss… v Circ…, Gene… v Fibr…)
- axes 'Marker Recall@K' has 3 pre-truncated labels (e.g. CD8+ cyto…, CD4+ help…, Antibody-…)
- axes 'Novel Cell Types' has 6 pre-truncated labels (e.g. Mucus-secreting goblet cells from huma…, Purkinje neurons from human cerebellar…, Glomerular podocytes from human kidney…)
- **Fix:** Drop the pre-truncated label and let matplotlib auto-wrap, OR abbreviate to a valid short name. Ellipsis hides biological identifiers from reviewers.

## INFO (3)

### `log_scale_unlabelled` × 2
- X-axis in 'Variance Scatter' uses log scale but label 'real variance' does not indicate this
- Y-axis in 'Variance Scatter' uses log scale but label 'gen. variance' does not indicate this
- **Fix:** Investigate the offending artist and relocate or resize.

### `bold_usage` × 1
- 1 text(s) use bold fontweight
- **Fix:** Investigate the offending artist and relocate or resize.
