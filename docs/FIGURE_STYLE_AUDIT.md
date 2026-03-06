# Figure Style And Consistency Audit

Audit of article-facing visualization style after consolidating the Cursor worktree suggestions.

## Source of truth

- Shared style lives in `src/visualization/style.py`.
- Use `COLORS` and `TYPE_PALETTE` for semantic and type-colored plots.
- Use `set_figure_suptitle()` with `SUPTITLE_Y = 0.98` for figure-level titles.
- Use `FONT_LEGEND` for normal legends and `FONT_LEGEND_DENSE` where 10pt would crowd dense multi-panel layouts.
- Use `add_colorbar_safe()` for article-facing colorbars.

## Integrated changes

- Replaced remaining semantic hardcoded real/generated colors in article-facing panels with `COLORS["real"]` and `COLORS["generated"]`.
- Aligned figure-level title placement to `set_figure_suptitle()` in training, metrics, merged embedding, downstream, and diversity-tradeoff composites.
- Aligned `panel_config.py` to the shared suptitle y-position (`0.98`).
- Replaced raw `fontsize=8` legend usage in dense training and violin panels with `FONT_LEGEND_DENSE` so the layout decision is explicit and centralized.
- Standardized classifier ROC legend sizing and semantic neutral text color.

## Intentionally not normalized further

- Dense panels still use 8pt legends where increasing to 10pt would risk VCD regressions.
- Panel-specific colorbar `shrink` and `pad` values remain layout-specific where already validated.
- Non-article scripts and evaluation-only utilities were not folded into this pass.

## Follow-up targets

- Remaining hardcoded palette values in non-article scripts and evaluation helpers.
- Additional legend-size normalization in expression panels if future VCD passes show spare space.
- Unifying any future policy constants in `style.py` before they are referenced by documentation.