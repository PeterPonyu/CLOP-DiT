# Figures 9–12 Policy

*Last revised: 2026-03-06*

This document defines the maintenance and regeneration policy for article Figures 9–12 (conditioning landscape, diversity diagnostics, diversity trade-off, baselines). It complements [FIGURE_ORGANIZATION.md](FIGURE_ORGANIZATION.md), [FIGURE_PRESENTATION_POLICY.md](FIGURE_PRESENTATION_POLICY.md), and [LEGEND_CAPTION_POLICY.md](LEGEND_CAPTION_POLICY.md).

---

## Canonical mapping

| Article Fig | Panel(s)   | Output file                         | Producer                                                   | Pipeline step |
| ----------- | ---------- | ----------------------------------- | ---------------------------------------------------------- | ------------- |
| **Fig 9**   | M          | `panel_m_conditioning_umap.pdf`     | `scripts/conditioning_analysis.py` → `panels_conditioning.plot_panel_m()` | Step 4        |
| **Fig 10**  | J          | `panel_j_diversity_diagnostics.pdf` | `scripts/diversity_diagnostics.py` → `panels_diversity.plot_diagnostics()` | Step 3        |
| **Fig 11**  | L+K merged | `fig_diversity_tradeoff.pdf`        | `ResultsVisualizer._compose_diversity_tradeoff()` (tiles L+K + violin)    | Step 7        |
| **Fig 12**  | O          | `panel_o_baseline_comparison.pdf`   | `baseline_panels.plot_baseline_comparison()`               | Step 7        |

---

## Regeneration

Any edit to the following requires regeneration and verification:

- `scripts/conditioning_analysis.py`
- `scripts/diversity_diagnostics.py`
- `src/visualization/panels_conditioning.py`
- `src/visualization/panels_diversity.py`
- `src/visualization/results_visualizer.py` (`_compose_diversity_tradeoff`)
- `src/visualization/baseline_panels.py` (`plot_baseline_comparison`)
- `src/visualization/style.py` (shared style)

**Required steps:**

1. Run the pipeline at least steps 3, 4, 7 so Figs 9–12 outputs are refreshed:
   ```bash
   bash scripts/pipeline/regenerate_report.sh
   # or minimal: steps that produce M, J, L, K, O
   ```
2. Run `bash scripts/pipeline/verify_article_figures.sh`
3. Rebuild the LaTeX article: `cd articles && latexmk -pdf clop_dit_biology.tex`

---

## Presentation

- **Annotation budget:** At most one stats box per subplot; at most two highlighted outliers per subplot (FIGURE_PRESENTATION_POLICY §1–2, §4).
- **Primary message:** Each subplot answers one question only; text density ceiling: at most three text elements beyond axes/ticks (§4).
- **Legends vs captions:** Legends = series/keys only; no r, p, CIs, or numeric summaries in legend titles. Statistics go in LaTeX caption or one short in-figure annotation (LEGEND_CAPTION_POLICY).
- **No figure numbers on image:** "Figure 9" … "Figure 12" and "(A)"–"(S)" appear only in the LaTeX caption, not drawn on the figure.

---

## Article–caption alignment

LaTeX captions for Figs 9–12 are in `articles/clop_dit_biology.tex`. When adding a new visual element (e.g. violins in Fig 11), update the corresponding `\caption{...}` so the narrative matches the figure.

---

## JBHI markdown

`docs/CLOP_DiT_JBHI_Article.md` historically used a different figure mapping (Fig 9 = decoder comparison / panel O, Fig 10 = ODE step / panel L). The **LaTeX article** (`articles/clop_dit_biology.tex`) and [FIGURE_ORGANIZATION.md](FIGURE_ORGANIZATION.md) are authoritative. The JBHI markdown may be updated to match LaTeX or the discrepancy may be documented there.

---

## Fig 11 dependencies (violin panel)

The Fig 11 bottom panel (violin comparison of pairwise cosine distributions) requires:

- `results/diversity_diagnostics.json` (from `scripts/diversity_diagnostics.py`, Step 3)
- Dedup cell/group caches (e.g. `data/cache/` or equivalent)
- Generated embeddings and labels

**Fig 11 is produced only when all three components are available:** panels L and K (from Steps 4 and 3) and the violin. If any is missing, the merged figure is skipped (no partial figure with blanks or fallback text). Run the full pipeline so Steps 3–4 and embedding generation complete before Step 7.

See [REGENERATION_STATUS.md](REGENERATION_STATUS.md) for prerequisites and [todo/2026-03-06-figure-regeneration-gaps.md](../todo/2026-03-06-figure-regeneration-gaps.md) for known missing artifacts.

---

## See also

- [FIGURE_ORGANIZATION.md](FIGURE_ORGANIZATION.md) — Panel list, canonical producers, save policy
- [FIGURE_PRESENTATION_POLICY.md](FIGURE_PRESENTATION_POLICY.md) — Annotation budget, primary message, shared semantics
- [LEGEND_CAPTION_POLICY.md](LEGEND_CAPTION_POLICY.md) — Legends vs captions
- [REGENERATION_STATUS.md](REGENERATION_STATUS.md) — Staleness, prerequisites, regeneration command
- [ENHANCEMENT_TASKS.md](ENHANCEMENT_TASKS.md) — Tier 1/2 enhancements, including Figs 10/11 diversity violins
- [FIGURES_12-15_POLICY.md](FIGURES_12-15_POLICY.md) — Figs 12–15 (benchmark, downstream, DE concordance)
