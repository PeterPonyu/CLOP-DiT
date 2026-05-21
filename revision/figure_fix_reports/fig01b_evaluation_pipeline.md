# Figure Fix Report — `fig01b_evaluation_pipeline`

**Producer:** `scripts/analysis/evaluation_pipeline_figure.py`
**Findings:** CRITICAL=0, MAJOR=28, MINOR=17, INFO=1

## MAJOR (28)

### `minimum_font_size` × 23
- text '80 GEO Datasets
(220,304 cells, 1,088 gr' renders at 6.3pt (nominal 13.0pt × scale 0.49), below 7.0pt
- text 'Bootstrap
95% CI
(B=1000)' renders at 5.8pt (nominal 12.0pt × scale 0.49), below 7.0pt
- text 'Cell-type Stratified Split
72 train / 8 ' renders at 6.3pt (nominal 13.0pt × scale 0.49), below 7.0pt
- text 'Common-Metrics Composite (9)
PRIMARY BEN' renders at 6.3pt (nominal 13.0pt × scale 0.49), below 7.0pt
- … +19 more
- **Fix:** Increase the nominal font size (>= 10pt for single-column, >= 14pt for double-width figures) so the rendered size at the article's `\includegraphics` width passes 7pt.

### `colorblind_confusable` × 4
- colors 'patch: unlabeled' and 'patch: unlabeled' collapse under deuteranopia (ΔE₇₆ = 11.1 < 12.0)
- colors 'patch: unlabeled' and 'patch: unlabeled' collapse under deuteranopia (ΔE₇₆ = 2.8 < 12.0)
- colors 'patch: unlabeled' and 'patch: unlabeled' collapse under protanopia (ΔE₇₆ = 11.8 < 12.0)
- colors 'patch: unlabeled' and 'patch: unlabeled' collapse under protanopia (ΔE₇₆ = 9.3 < 12.0)
- **Fix:** Replace the offending color pair. Safe defaults: orange #E69F00, blue #0072B2, vermillion #D55E00, bluish-green #009E73 (Wong 2011 palette).

### `effective_dpi_low` × 1
- effective DPI at rendered width 7.0in is 206 (fig_dpi=100.0, fig_width=14.4in); below 300
- **Fix:** Raise `dpi=300` at save time OR reduce the generated figsize so the effective DPI at the rendered width is >= 300.

## MINOR (17)

### `annotation_data_overlap` × 17
- Text 'annotation: 80 GEO Datasets
(220,304 cells, 1,088 groups)' overlaps content 'patch: FancyBboxPatch' (8074 px²)
- Text 'annotation: Cell-type Stratified Split
72 train / 8 held-out' overlaps content 'patch: FancyBboxPatch' (6194 px²)
- Text 'annotation: Common-Metrics Composite (9)
PRIMARY BENCHMARK' overlaps content 'patch: FancyBboxPatch' (9033 px²)
- Text 'annotation: Deduplication
1,088 → 69 types' overlaps content 'patch: FancyBboxPatch' (4605 px²)
- … +13 more
- **Fix:** Reposition or drop redundant annotations; the data artist is hidden behind text.

## INFO (1)

### `bold_usage` × 1
- 11 text(s) use bold fontweight
- **Fix:** Investigate the offending artist and relocate or resize.
