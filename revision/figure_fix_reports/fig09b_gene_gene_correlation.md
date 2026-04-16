# Figure Fix Report — `fig09b_gene_gene_correlation`

**Producer:** `scripts/analysis/gene_gene_correlation.py`
**Findings:** CRITICAL=2, MAJOR=90, MINOR=3, INFO=0

## CRITICAL (2)

### `cross_axes_text_overlap` × 1
- xtick '−0.2' (axes 0) overlaps ytick '−10' (axes 3) -- 126 px²; consider increasing hspace/wspace
- **Fix:** Increase `hspace`/`wspace` in the gridspec, or prune the offending tick with `prune='upper'`/`'lower'` to remove the colliding label.

### `text_truncation` × 1
- 'ytick: −10' extends beyond figure border (top, 9px)
- **Fix:** Increase axis margins (`plt.subplots_adjust(top=..., bottom=...)`) or tighten the tick locator via `ax.xaxis.set_major_locator(MaxNLocator(nbins=N, prune='upper'))`.

## MAJOR (90)

### `minimum_font_size` × 88
- text 'Alveolar type 2 (…' renders at 4.5pt (nominal 9.0pt × scale 0.50), below 7.0pt
- text 'Density' renders at 6.0pt (nominal 12.0pt × scale 0.50), below 7.0pt
- text 'Gene index (top 50 HVG)' renders at 6.0pt (nominal 12.0pt × scale 0.50), below 7.0pt
- text 'Gene index (top 50 HVG)' renders at 6.0pt (nominal 12.0pt × scale 0.50), below 7.0pt
- … +84 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 200 (fig_dpi=100.0, fig_width=14.0in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.

### `text_overlap` × 1
- 'annotation: Inhibitory GABAer…' overlaps 'annotation: Hepatocytes are t…'
- **Fix:** Raise panel-label Y (e.g. `add_panel_label(ax, 'b', y=1.12)`) or drop the last annotation so the axis tick below can breathe.

## MINOR (3)

### `annotation_data_overlap` × 2
- Text 'annotation: Mean 95% CI: [0.624, 0.667]
Null mean: -0.000' overlaps content 'patch: Rectangle' (295 px²)
- Text 'annotation: Spearman ρ = 0.381
Pearson r = 0.365' overlaps content 'line: Line2D(OLS fit)' (2145 px²)
- **Fix:** Reposition or drop redundant annotations; the data artist is hidden behind text.

### `label_string_ellipsis` × 1
- axes 'Preservation vs. Expression Heterogeneit' has 3 pre-truncated labels (e.g. Alveolar type 2 (…, Inhibitory GABAer…, Hepatocytes are t…)
- **Fix:** Drop the pre-truncated label and let matplotlib auto-wrap, OR abbreviate to a valid short name. Ellipsis hides biological identifiers from reviewers.
