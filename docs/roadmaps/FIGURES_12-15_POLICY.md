# Figures 12–15 Policy

*Last revised: 2026-03-06*

This document defines the maintenance, regeneration, presentation, and enhancement policy for article Figures 12–15 (baselines, benchmark, downstream P+Q, DE concordance). It complements [FIGURE_ORGANIZATION.md](FIGURE_ORGANIZATION.md), [FIGURE_PRESENTATION_POLICY.md](FIGURE_PRESENTATION_POLICY.md), [LEGEND_CAPTION_POLICY.md](LEGEND_CAPTION_POLICY.md), and extends [FIGURES_9-12_POLICY.md](FIGURES_9-12_POLICY.md) for Figs 13–15.

---

## Canonical mapping

| Article Fig | Panel(s)   | Output file                         | Producer                                                                 | Pipeline step |
| ----------- | ---------- | ----------------------------------- | ------------------------------------------------------------------------ | ------------- |
| **Fig 12**  | O          | `panel_o_baseline_comparison.pdf`   | `baseline_panels.plot_baseline_comparison()`                             | Step 7        |
| **Fig 13**  | S          | `panel_s_benchmark.pdf`             | `benchmark_panels.plot_benchmark_panel()`                                | Step 7        |
| **Fig 14**  | P+Q merged | `fig_downstream_pq.pdf`             | `downstream_panels.plot_clustering_and_classifier_merged()`              | Step 7        |
| **Fig 15**  | R          | `panel_r_de_concordance.pdf`        | `panels_de_concordance.plot_de_concordance_panel()` (via downstream_panels) | Step 7        |

Reference: [article_delivery.py](../src/visualization/article_delivery.py) (`ARTICLE_FIGURE_BASENAMES`, `ARTICLE_FIGURE_PRODUCERS`). LaTeX includes these in `articles/clop_dit_biology.tex`.

---

## Regeneration and dependencies

**Files that require regeneration when changed:**

- `src/visualization/baseline_panels.py`
- `src/visualization/benchmark_panels.py`
- `src/visualization/downstream_panels.py`
- `src/visualization/panels_de_concordance.py`
- `src/visualization/results_visualizer.py` (calls Fig 12/14/15)
- `src/visualization/style.py`

**Pipeline steps:** Step 6 (benchmark/downstream data) and Step 7. Figs 14–15 require `clustering_alignment.json`, `classifier_alignment.json`, `de_concordance.json` in `results/downstream/`. Fig 12 requires `generation_metrics.json` and either precomputed `baseline_metrics.json` or cache for on-the-fly `_compute_baselines()`; if missing, Panel O is skipped.

**Commands:**

```bash
bash scripts/regenerate_report.sh
bash scripts/verify_article_figures.sh
cd articles && latexmk -pdf clop_dit_biology.tex
```

---

## Presentation

Same rules as [FIGURE_PRESENTATION_POLICY.md](FIGURE_PRESENTATION_POLICY.md):

- **Annotation budget:** At most one stats box per subplot; at most two highlighted outliers per subplot; text density ceiling: at most three text elements beyond axes/ticks.
- **Primary message:** Each subplot answers one question only.
- **Legends vs captions:** Legends = series/keys only; no r, p, CIs, or numeric summaries in legend titles. Statistics go in LaTeX caption or one short in-figure annotation. See [LEGEND_CAPTION_POLICY.md](LEGEND_CAPTION_POLICY.md).
- **No figure numbers on image:** "Figure 12" … "Figure 15" and subpanel labels appear only in the LaTeX caption.
- **Suptitle:** Use `style.set_figure_suptitle()` for figure-level titles (consistent position and wrapping).
- **Colorbars:** Use `style.add_colorbar_safe()` for all colorbars.
- **Fonts:** Use style constants (`FONT_TITLE`, `FONT_LABEL`, `FONT_TICK`, `FONT_TICK_DENSE`, `FONT_LEGEND`).

---

## Article–caption alignment

LaTeX captions for Figs 12–15 are in `articles/clop_dit_biology.tex`. When adding or changing a visual element (e.g. classifier heatmap, CIs, mixing vs abundance), update the corresponding `\caption{...}` so the narrative matches the figure.

---

## Submission

Before submission, confirm the package includes:

- `panel_o_baseline_comparison.pdf` (Fig 12)
- `panel_s_benchmark.pdf` (Fig 13)
- `fig_downstream_pq.pdf` (Fig 14)
- `panel_r_de_concordance.pdf` (Fig 15)

**Legacy:** `fig12_downstream_composed.pdf` is **retired**. Do not ship it. The article uses `fig_downstream_pq.pdf` (Fig 14) and `panel_r_de_concordance.pdf` (Fig 15) instead. See [REVIEWER_CONCERNS_AND_NEXT_STEPS.md](REVIEWER_CONCERNS_AND_NEXT_STEPS.md).

---

## Known limitations

- **Panel S (Fig 13):** Heatmap text uses explicit dark color (`#1a1a1a`) for WCAG contrast. VCD may report info-level issues; document as accepted.
- **Figs 14–15:** Can be stale after code changes until regeneration. See [REGENERATION_STATUS.md](REGENERATION_STATUS.md).

---

## Enhancement policy

In-scope roadmap items for Figs 12–15 (from [FIGURE_ENHANCEMENT_ROADMAP.md](FIGURE_ENHANCEMENT_ROADMAP.md)):

| Figure | Enhancement | Tier |
|--------|-------------|------|
| Fig 12 | Optional: bootstrap CIs for baselines (O1/O2) | Optional |
| Fig 13 | Optional: amplify CI visibility in S4 | Tier 2 |
| Fig 14 | Classifier per-type heatmap; Wilson CI for classifier; mixing vs abundance scatter | Tier 1, 2 |
| Fig 15 | Effect-size weighted scatter (marginals); CIs for concordance metrics; per-contrast hierarchy; DE dendrogram | Tier 1, 2, 3/4 |

**Rule:** New panels or subplots from the roadmap must follow the same presentation policy (annotation budget, legend/caption, `save_with_vcd()`, style constants) and be documented in [FIGURE_ORGANIZATION.md](FIGURE_ORGANIZATION.md) if added to the article.

---

## See also

- [FIGURE_ORGANIZATION.md](FIGURE_ORGANIZATION.md) — Panel list, canonical producers, save policy
- [FIGURE_PRESENTATION_POLICY.md](FIGURE_PRESENTATION_POLICY.md) — Annotation budget, primary message, shared semantics
- [LEGEND_CAPTION_POLICY.md](LEGEND_CAPTION_POLICY.md) — Legends vs captions
- [FIGURE_ENHANCEMENT_ROADMAP.md](FIGURE_ENHANCEMENT_ROADMAP.md) — Full enhancement roadmap
- [ENHANCEMENT_TASKS.md](ENHANCEMENT_TASKS.md) — Tier 1/2 task list
- [REGENERATION_STATUS.md](REGENERATION_STATUS.md) — Prerequisites, staleness, regeneration command
- [FIGURES_9-12_POLICY.md](FIGURES_9-12_POLICY.md) — Figs 9–12 (Fig 12 also covered there)
