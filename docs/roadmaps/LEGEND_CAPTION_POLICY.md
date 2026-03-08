# Legend and Caption Policy

*Last revised: 2026-03-06*

This document defines what belongs in figure **legends** vs. **captions** (or in-figure annotations) for CLOP-DiT publication figures. Following this policy keeps legends compact and avoids truncation or clutter.

## Legend content

- **Legends should contain only series/keys**: what each curve, color, or marker represents (e.g. "Real", "Generated", method names, cell type labels).
- **Do not put statistical summaries in legend titles or entries**: values such as Pearson r, p-values, sign agreement, CV correlation, or confidence intervals belong in the figure caption or in a small, non-dominant in-figure annotation (e.g. a short text box in a corner).
- **When there are many legend entries** (e.g. >6–8): prefer `ncol` and smaller font, or show a subset of keys with a note like "see caption for full list" in the article caption. Use `style.truncate_legend_for_caption()` to cap entries and add a "… (see caption)" line.
- **Keep legend labels brief**: avoid long phrases or statistics in legend entries; put numeric values and long lists in the figure caption.
- **Prefer axes-level legends over figure-level legends** unless a shared legend clearly reduces duplication across subplots. Figure-level legends are harder to place robustly and should be reserved for genuinely shared keys.

## Caption content

- **Figure captions** (in LaTeX) provide: figure labelling (e.g. "Figure 2: …"), panel descriptions, and **statistical summaries** (r, p, CIs, sign agreement, etc.) when they are not shown as a short annotation in the figure.
- The article caption is the single place for narrative interpretation; avoid duplicating long statistical text in the legend.

## In-figure annotations

- **Short stats** (e.g. "r = 0.95", "CV corr = 0.88") may be placed as a small text annotation in a corner of the relevant panel, with a light background so they remain readable.
- Keep such annotations brief (one line or two) so they do not overlap titles or spill into adjacent panels.
- Keep the annotation budget low: one small statistics box per subplot is usually enough. If a panel needs multiple text boxes to explain itself, move the narrative back to the caption.

## Code and review checklist

- Panel code: use `ax.legend(...)` without `title=` when the title would be a statistic; put the statistic in an `ax.text(..., transform=ax.transAxes)` or in the LaTeX caption.
- Review: ensure no legend title contains patterns like "r =", "p =", "Sign =", "CV corr =" etc.; move those to caption or annotation.

## See also

[FIGURE_PRESENTATION_POLICY.md](FIGURE_PRESENTATION_POLICY.md) defines annotation budget, primary message per panel, shared semantics (colors and thresholds), text density ceiling, and article vs report distinction.
