# Figure Fix Report — `fig01a_architecture`

**Producer:** `scripts/analysis/generate_architecture_figure.py`
**Findings:** CRITICAL=0, MAJOR=17, MINOR=78, INFO=1

## MAJOR (17)

### `colorblind_confusable` × 8
- colors 'patch: unlabeled' and 'patch: unlabeled' collapse under deuteranopia (ΔE₇₆ = 10.2 < 12.0)
- colors 'patch: unlabeled' and 'patch: unlabeled' collapse under deuteranopia (ΔE₇₆ = 10.9 < 12.0)
- colors 'patch: unlabeled' and 'patch: unlabeled' collapse under deuteranopia (ΔE₇₆ = 3.6 < 12.0)
- colors 'patch: unlabeled' and 'patch: unlabeled' collapse under deuteranopia (ΔE₇₆ = 7.0 < 12.0)
- … +4 more
- **Fix:** Replace the offending color pair. Safe defaults: orange #E69F00, blue #0072B2, vermillion #D55E00, bluish-green #009E73 (Wong 2011 palette).

### `minimum_font_size` × 8
- text '512-d' renders at 6.3pt (nominal 9.0pt × scale 0.70), below 7.0pt
- text '512→512' renders at 6.3pt (nominal 9.0pt × scale 0.70), below 7.0pt
- text 'Euler/Mid' renders at 6.3pt (nominal 9.0pt × scale 0.70), below 7.0pt
- text 'MLP 1024→512' renders at 6.3pt (nominal 9.0pt × scale 0.70), below 7.0pt
- … +4 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 143 (fig_dpi=100.0, fig_width=10.0in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.

## MINOR (78)

### `annotation_data_overlap` × 78
- Text 'annotation: $\mathbf{v} = v_{\rm unc} + s(v_{\rm cond} - v_{\r' overlaps content 'patch: FancyBboxPatch' (3054 px²)
- Text 'annotation: $\sim\mathcal{N}(0,I)$' overlaps content 'patch: FancyArrowPatch' (529 px²)
- Text 'annotation: $\sim\mathcal{N}(0,I)$' overlaps content 'patch: FancyBboxPatch' (291 px²)
- Text 'annotation: $\sim\mathcal{N}(0,I)$' overlaps content 'patch: FancyBboxPatch' (817 px²)
- … +74 more
- **Fix:** Reposition or drop redundant annotations; the data artist is hidden behind text.

## INFO (1)

### `bold_usage` × 1
- 3 text(s) use bold fontweight
- **Fix:** Investigate the offending artist and relocate or resize.
