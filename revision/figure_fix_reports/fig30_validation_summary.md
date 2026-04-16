# Figure Fix Report — `fig30_validation_summary`

**Producer:** `src/visualization/fig30_validation_summary.py`
**Findings:** CRITICAL=0, MAJOR=64, MINOR=12, INFO=1

## MAJOR (64)

### `minimum_font_size` × 61
- text '(a)' renders at 6.1pt (nominal 14.0pt × scale 0.44), below 7.0pt
- text '(b)' renders at 6.1pt (nominal 14.0pt × scale 0.44), below 7.0pt
- text '(c)' renders at 6.1pt (nominal 14.0pt × scale 0.44), below 7.0pt
- text '0.123' renders at 4.4pt (nominal 10.0pt × scale 0.44), below 7.0pt
- … +57 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `colorblind_confusable` × 2
- colors 'line: CLOP-DiT' and 'patch: _nolegend_' collapse under deuteranopia (ΔE₇₆ = 11.0 < 12.0)
- In 'Experiment Scorecard': 1 colour pair(s) may be confusable under deuteranopia: 'patch' vs 'patch' (ΔE=3.5)
- **Fix:** Replace the offending color pair. Safe defaults: orange #E69F00, blue #0072B2, vermillion #D55E00, bluish-green #009E73 (Wong 2011 palette).

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 229 (fig_dpi=100.0, fig_width=16.0in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.

## MINOR (12)

### `annotation_data_overlap` × 12
- Text 'annotation: Canonical recall@50 = 0.123' overlaps content 'patch: FancyBboxPatch' (1095 px²)
- Text 'annotation: Core Metrics' overlaps content 'patch: FancyBboxPatch' (714 px²)
- Text 'annotation: Cross-Dataset' overlaps content 'patch: FancyBboxPatch' (807 px²)
- Text 'annotation: Emb. Augmentation' overlaps content 'patch: FancyBboxPatch' (1134 px²)
- … +8 more
- **Fix:** Reposition or drop redundant annotations; the data artist is hidden behind text.

## INFO (1)

### `bold_usage` × 1
- 6 text(s) use bold fontweight
- **Fix:** Investigate the offending artist and relocate or resize.
