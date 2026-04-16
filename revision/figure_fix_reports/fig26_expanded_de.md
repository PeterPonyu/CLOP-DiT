# Figure Fix Report — `fig26_expanded_de`

**Producer:** `src/visualization/fig26_expanded_de.py`
**Findings:** CRITICAL=1, MAJOR=80, MINOR=2, INFO=1

## CRITICAL (1)

### `text_truncation` × 1
- 'ytick: 0.002' extends beyond figure border (top, 11px)
- **Fix:** Increase axis margins (`plt.subplots_adjust(top=..., bottom=...)`) or tighten the tick locator via `ax.xaxis.set_major_locator(MaxNLocator(nbins=N, prune='upper'))`.

## MAJOR (80)

### `minimum_font_size` × 79
- text '(a)' renders at 5.9pt (nominal 14.0pt × scale 0.42), below 7.0pt
- text '(b)' renders at 5.9pt (nominal 14.0pt × scale 0.42), below 7.0pt
- text '(c)' renders at 5.9pt (nominal 14.0pt × scale 0.42), below 7.0pt
- text '0.08' renders at 3.4pt (nominal 8.0pt × scale 0.42), below 7.0pt
- … +75 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 236 (fig_dpi=100.0, fig_width=16.5in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.

## MINOR (2)

### `label_string_ellipsis` × 2
- axes 'DE Concordance Metrics' has 5 pre-truncated labels (e.g. CD8+ cyto…
vs CD4+ help…, Tissue-re…
vs Circulati…, Generic e…
vs Fibroblas…)
- axes 'Top-K Overlap' has 5 pre-truncated labels (e.g. CD8+ cyt… vs CD4+ hel…, Tissue-r… vs Circulat…, Generic … vs Fibrobla…)
- **Fix:** Drop the pre-truncated label and let matplotlib auto-wrap, OR abbreviate to a valid short name. Ellipsis hides biological identifiers from reviewers.

## INFO (1)

### `bold_usage` × 1
- 2 text(s) use bold fontweight
- **Fix:** Investigate the offending artist and relocate or resize.
