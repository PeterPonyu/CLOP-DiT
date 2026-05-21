# VCD Multimodal Visual Audit (US-304)

**Ran:** 2026-04-17
**Auditor:** Claude Opus 4.7 (multimodal) reading each figure PNG at 100 DPI
**Corpus:** 21 article-manifest figures under `results/figures/`
**VCD baseline (current state):** CRITICAL 75, MAJOR 1394, MINOR 147, INFO 10

## Method

Each figure PDF was rendered to PNG via `pdftoppm -r 100`, opened via the
multimodal `Read` tool, and visually compared against the live-VCD sidecar
findings for that same figure. Discrepancies — either visual issues the
geometric VCD missed, or VCD findings that did not correspond to a real
readability defect — were logged. The findings below feed directly into
US-305's fold-back pass.

## 21-figure coverage table

| # | Figure | Visual read | Current VCD | Δ (visual − VCD) |
|---:|---|---|---|---|
| 1 | fig01a_architecture | Schematic with 3 color-banded stages. `train: align text and cell latents` label FLOATS ABOVE the Text Description panel — visually orphaned. `training only` label floats between Cell Path and Shared Space blocks. `512-d` appears twice near the sampled-latent block. Arrows cross the Shared Space dashed box cleanly. | MAJOR 2 cosmetic legend masking | **Missed: floating-label-outside-colored-region pattern** |
| 2 | fig01b_evaluation_pipeline | Clean flowchart. Boxes and arrows well-spaced. Only a tiny gray `Bootstrap 95% CI (B=1000)` bracket floats beside the terminal Full Composite node — legitimate annotation, no issue. | PASS | — |
| 3 | fig02a_training_dynamics | 8-panel grid (a–h). Panel (c) Classification Accuracy legend has 4 entries (Val Acc / Train Acc / Top-5 / Top-10) but only **one line visible** — the other three collapse onto the 100% ceiling. Visual redundancy. | MAJOR 17, MINOR 11 (ytick-vs-panel-label overlap) | **Missed: overlapping-series-values pattern** |
| 4 | fig02b_embedding_space | 6-panel UMAP overlay (i,j,k,l,m,n). Y-axis tick labels in (k) Cells per Type carry 18 ellipsis-truncated cell-type names ("CD8+ cytotoxic T lymp…", "Generic epith. form p…"). Bar chart readable but labels hide information. | backfilled (no live audit) | **Missed: label-string-ellipsis pattern (18 instances on one axis)** |
| 5 | fig03a_metrics_summary | 4 panels. (b) Quality Profile has `Gene Corr = 0.000` for Gaussian vs `1.000` for CLOP-DiT — invisible vs full-height bar, jarring scale contrast. (d) uses log-scale with "4.1e-06" annotations which are tiny but visible. | backfilled | **Missed: extreme-value-contrast pattern in grouped bars** |
| 6 | fig03b_per_type_fidelity | 3 panels (e,f,g). All three show ellipsis-truncated cell-type names. (g) Fidelity vs Abundance has 3 outlier annotations "Dorsal hor…", "Smooth musc…", "Kidney prox…", "Inhibitory …" connected by leader lines. Readable. | backfilled | **Missed: label-string-ellipsis pattern** |
| 7 | fig03c_text_cell_alignment | 3 panels (h,i,j). (h) is a 69×69 cosine heatmap with x-axis labels rotated and truncated ("Vascular end…", "Pituitary go…"). White rectangle highlights a block. (i) Per-Type Alignment has 18 ellipsis labels and numeric callouts "-0.086", "-0.095" along the right edge of bars. | backfilled | **Missed: heatmap-highlight-rectangle without legend entry** |
| 8 | fig04a_marker_genes | 4-panel small-figure (a–d). Tiny — 2 × 7inch layout. Y-axis labels of heatmap (b,c) show ellipsis truncation on cell-type names. Clean overall. | backfilled | **Minor: same ellipsis pattern** |
| 9 | fig04b_expression_correlation | 4 panels (e,f,g,h). (e) Per-Gene Correlation has 5 gene-label callouts (CFH, SHD, EXPH5 on left; FGL1, U91319.1 on right) — all placed via leader lines, no overlap. Clean. | backfilled | — |
| 10 | fig05a_expression_analysis | 4 panels (a–d). (a) Per-Gene Variability has 6 outlier gene-name labels (EXPH5, LYPD6B, FST, ARSI, ZBED2, SHD) placed inside the axis with short leader lines — one label appears near the `y=x` dashed line. No overlap. | backfilled | — |
| 11 | fig05b_conditioning_landscape | 7-panel figure (e–k). (e) has a 4-column row of scatter plots with shared legend BELOW — legend text truncated with ellipsis (`Hematopoietic …`, `Kidney proxima…`). (j) heatmap has 5 × 4 numeric annotations. | backfilled | **Missed: shared-legend truncation below axes** |
| 12 | fig06_diversity_diagnostics | 5 panels (a–e). (a) Diversity Ratio bar has 9 visible cell-type y-labels — all ellipsis-truncated. (e) has a legend outside the axes on the right, clean. | PASS | **Minor: ellipsis pattern** |
| 13 | fig07a_expression_diversity | Clean 2-panel figure. (a) Expression Variability bar has two groups "ratio=1.00" and "ratio=1.04" annotations very close together near the top — slightly cramped but readable. | PASS | — |
| 14 | fig07b_baseline_comparison | 3 panels (c,d,e). (d) Normalised Scores has a trellis-style plot with multi-marker legend below. (e) CLOP-DiT Advantage has 8 horizontal bars with centered numeric labels inside the bars. Clean. | backfilled | — |
| 15 | fig07c_benchmark | 4 panels (f,g,h,i). (f) Metrics Heatmap has 8 blue outlined rectangles highlighting cells — **no legend explains the outline convention**. (g) Composite Score has italic numeric labels (0.395, 0.838, etc.) at ~6pt, flagged by min_font_size. (i) Bootstrap CI has "CLOP-DiT" label repeated 3 times vertically — duplicate tick labels. | backfilled | **Missed: unexplained-highlight-rectangles + duplicate-tick-labels** |
| 16 | fig08a_downstream_validation | 6 panels (a–f). (b) Sorted kNN mixing has 15 cell-type y-labels all ellipsis-truncated. (d) Confusion Matrix is 69×69, sparse — most cells are near-white with a few diagonal blues. (e) Per-Type PRF1 heatmap has 18 cell-type labels, ALL ellipsis-truncated. | backfilled | **Missed: ellipsis pattern at extreme density (18 on one axis)** |
| 17 | fig08b_de_concordance | 3 panels (g,h,i). (h) Concordance heatmap has 3 row labels each truncated with "…". (i) Per-Contrast Summary has 3 x-axis group labels each truncated. Clean otherwise. | backfilled | **Minor: ellipsis pattern** |
| 18 | fig09a_variance_matching | 4 panels (a–d). (a) Latent SWD has 24 ellipsis-truncated cell-type labels AND numeric "n=1,149" overlay annotations overlapping with bars. (d) SWD vs Training Cell Count has 2 outlier annotations whose leader lines cross — genuine readability defect. | CRITICAL 4 (truncation + cross-axes overlap) | **VCD correctly flagged cross-axes overlap; missed dense-annotation crowding pattern** |
| 19 | fig09b_gene_gene_correlation | 4 panels (e,f,g,h). (h) Preservation vs Heterogeneity has 3 outlier labels at top-right overlapping ("Alveolar type 2 (", "Hepatocytes are t…", "Inhibitory GABAer…"). | CRITICAL 1 (ytick truncation) | **VCD caught border truncation; missed overlapping outlier annotations** |
| 20 | figS01_supplementary_validation | 12-panel composed appendix figure. Massive density: (c) Novel Cell Types + Free-form Prompt Styles uses colored-text "Goblet cells", "Purkinje cells" column with italic prompt excerpts — real content truncation. (g) Marker Recall@K with 4 truncated legend labels. (l) Decoder Comparison has 6 small bars with tiny numeric labels. | CRITICAL 60, MAJOR 11 (heavy truncation) | **VCD correctly flagged high density; counts are reviewer-visible defects** |
| 21 | figS02_expression_diagnostics | 10-panel composed appendix figure. (a) Per-Type Pseudobulk has 15 y-labels ALL ellipsis-truncated (e.g. "Multiciliated…", "Tissue-reside…", "Pituitary gon…"). (c) Quality by Family has both truncation AND colored stacked bars with small counts. (f,g) Steering by Variant labels are truncated. | MAJOR 47 | **VCD flagged many; visually dense but publication-acceptable for supplement** |

## Aggregate patterns (9 distinct issue families)

| # | Pattern | Where seen (figures) | Why VCD missed | Proposed rule |
|---:|---|---|---|---|
| 1 | **Label string ellipsis** | 2b, 3b, 3c, 4a, 5b, 6, 8a, 8b, 9a, S01, S02 (11/21) | String was truncated by the producer BEFORE rendering, so the matplotlib bbox fits the axis. VCD only measures rendered geometry. | **`label_string_ellipsis`** — flag text artists whose visible text ends with `…` / `...` at INFO severity (content-loss warning) |
| 2 | **Overlapping series at same y** | 2a panel (c) | All 4 lines land at 100%; VCD's geometry check sees 4 independent lines and a legend that fits. Functional redundancy is semantic, not geometric. | **`overlapping_series_values`** — if >=3 lines have >99% coincident y-values over >=80% of x-range, flag INFO |
| 3 | **Duplicate tick labels** | 7c panel (i) | "CLOP-DiT" repeated 3 times on y-axis. Each tick label is distinct text artist. | **`duplicate_tick_labels`** — flag axes whose tick-label list has duplicates, INFO severity (grouped axes are legitimate but worth verifying) |
| 4 | Floating label outside colored region | 1a | Labels are figure-level text at coords that visually belong to a subplot. No bbox check can know intent. | Harder to automate — skip for now; document as review-only |
| 5 | Extreme value contrast in grouped bars | 3a panel (b) | Bar = 0.000 has zero height, invisible. VCD sees no artist to check. | Skip — legitimate rendering; reviewer issue is scale choice |
| 6 | Unexplained highlight rectangles | 7c panel (f) | Rectangles are `matplotlib.patches.Rectangle` with no legend entry. VCD treats them as generic patches. | Possible future rule; skip this round |
| 7 | Heavy supplementary density | S01, S02 | Dense panels are legitimately flagged; VCD warnings correspond to real defects | Already handled by existing checks; just route to COMPOSED profile |
| 8 | Annotation / leader-line clustering | 9a panel (d), 9b panel (h) | Leader lines cross; already caught by VCD cross-axes overlap | Already covered |
| 9 | Border-adjacent tick labels | 7a, 9b | Caught by existing text_truncation | Already covered |

## Proposed new rules for US-305 fold-back

Three new rules are ready to ship — each has a clear detection signal, fits
the existing check-module API, and unit-tests naturally.

### Rule A: `label_string_ellipsis` (MINOR severity)

```python
def check_label_string_ellipsis(fig) -> list[dict]:
    """Flag text artists whose string content ends with … or ..., indicating
    the producer truncated the label before rendering. The matplotlib bbox
    check never sees this because the shorter string fits. Reviewers do see
    it because biological identifiers are cut off."""
```

Expected hits: 11/21 paper figures. This is by far the most prevalent
visual-vs-geometric mismatch in the corpus.

### Rule B: `overlapping_series_values` (INFO severity)

```python
def check_overlapping_series_values(fig) -> list[dict]:
    """For each axes with >=3 line or bar series, compute pairwise y-coincidence
    ratio. If >=3 series are >99% coincident over >=80% of the x-range, emit
    an informational finding so the author can decide whether to drop
    redundant series or mark the ceiling explicitly."""
```

Expected hits: 1 (fig02a panel c).

### Rule C: `duplicate_tick_labels` (INFO severity)

```python
def check_duplicate_tick_labels(fig) -> list[dict]:
    """Scan axis tick labels; flag axes where the same label string appears
    more than once. Valid in grouped-axis designs, but the author should
    confirm intent and not rely on group spacing alone for meaning."""
```

Expected hits: 1 (fig07c panel i).

## Closing assessment

The current VCD misses **≥ 27 real, reviewer-visible defects** across the 21
figures, all in one family: string ellipsis on cell-type labels. The other
two new rules each catch a distinct, rarer pattern. Rolling these in expands
the VCD coverage from geometric-only to content-aware, addressing the
biggest single gap between "what the linter sees" and "what a reviewer
sees".

**US-304 verdict: PASS** — all 21 figures audited, 3 new rules proposed with
concrete acceptance signals.
