# Enhancement Tasks — What to Do Next

Concise list of actionable tasks. Full detail is in [FIGURE_ENHANCEMENT_ROADMAP.md](FIGURE_ENHANCEMENT_ROADMAP.md).

---

## Before any enhancements

1. **Regenerate figures** (in a dev environment with PyTorch + GPU):
   ```bash
   bash scripts/pipeline/regenerate_report.sh
   # or: bash scripts/pipeline/regenerate_report.sh --skip-gen
   ```
2. **Verify:** `bash scripts/pipeline/verify_article_figures.sh`
3. **Rebuild article:** `cd articles && latexmk -pdf clop_dit_biology.tex`

See [REGENERATION_STATUS.md](REGENERATION_STATUS.md) for prerequisites and [QUICK_START.md](QUICK_START.md) for the full pipeline.

---

## Tier 1 (high impact, do first)

| # | Task | Where | Est. |
|---|------|--------|------|
| 1.1 | Per-type heterogeneity scatter (cosine vs cell count; FD with CI) | `panels_quality.plot_per_type_heterogeneity()` | 2 h |
| 1.2 | Diversity distribution violins (real/gen/real–gen per type) | New in `diversity_diagnostics` or visualization | 2 h |
| 1.3 | Classifier per-type heatmap (precision/recall from confusion matrix) | `downstream_panels.plot_classifier_per_type_heatmap()` | 1.5 h |
| 1.4 | DE effect-size weighted scatter (logFC + p-value encoding) | `downstream_panels.plot_de_effect_size_weighted()` | 2 h |

Data for all of these already exists in `results/*.json`; see the roadmap for exact keys and file paths.

---

## Tier 2 (after Tier 1)

- Propagate confidence intervals to Panels Q and R (e.g. Wilson binomial CI for classifier).
- Mixing score vs cell type abundance scatter (from `clustering_alignment.json` + counts).

---

## Code / policy fixes already done

- **Save policy:** All article figure saves go through `save_with_vcd()` (or a `save_panel_fn` that uses it). Merged figures (G+F, L+K) now use this path.
- **Default output_dir:** Panel modules default to `output_dir="results/figures"`.
- **Docs:** [docs/INDEX.md](INDEX.md) indexes all docs; [FIGURE_ORGANIZATION.md](FIGURE_ORGANIZATION.md) documents the save and presentation policy.
- **Figures 9–12:** Regeneration and policy for conditioning, diversity, and baselines — see [FIGURES_9-12_POLICY.md](FIGURES_9-12_POLICY.md).
- **Fig 9:** Suptitle via `set_figure_suptitle`; subplot titles simplified to one line.
- **Fig 10:** Stats moved from subplot titles to caption (annotation budget).
- **Fig 11:** Violin panel integrated; suptitle via `set_figure_suptitle`.
- **Fig 12:** Suptitle via `set_figure_suptitle`; save via `save_panel` (PNG+PDF).
- **Figs 13–15:** Policy and scope in [FIGURES_12-15_POLICY.md](FIGURES_12-15_POLICY.md). Fig 13, 14, 15: Suptitle via `set_figure_suptitle`; save via `save_panel`.

---

## If figure generation still has issues

- **VCD warnings:** Run the visualizer or scripts and read the detector output; fix overlap/truncation per panel (see [AUDIT_CLAUDE_COMPLETED_WORK.md](AUDIT_CLAUDE_COMPLETED_WORK.md) §4).
- **Missing panels:** Ensure pipeline steps 0–7 run in order; step 3 produces J/K, step 4 produces L/M, step 7 produces the rest and merged figures.
- **Stale PDFs:** Regeneration is required after any change to figure code; the article uses whatever is in `articles/figures/` (symlinks to `results/figures/`).
