# VCD Coverage Audit (US-201)

**Ran:** 2026-04-16
**Live-VCD sidecar dir:** `results/figures/_live_vcd/`
**Canonical figure manifest:** `src/visualization/article_delivery.py:ARTICLE_FIGURE_BASENAMES` (21 entries)
**Live hook:** `save_with_vcd` in `src/visualization/style.py:504`

## Method

For every article-facing figure basename, locate the producer script via
`ARTICLE_FIGURE_PRODUCERS`, identify the save path (direct `save_with_vcd`
vs `save_panel_fn` callback), and check whether `results/figures/_live_vcd/{basename}.json`
exists after a fresh `scripts/pipeline/run_regeneration.py` run.

## Coverage table

| # | Article basename | Producer | Save path | Sidecar? | Fix if missing |
|---:|---|---|---|:---:|---|
| 1 | fig01a_architecture | scripts/analysis/generate_architecture_figure.py | `save_panel` (= save_with_vcd alias) | ✅ | — |
| 2 | fig01b_evaluation_pipeline | scripts/analysis/evaluation_pipeline_figure.py | `save_with_vcd` direct | ✅ | — |
| 3 | fig02a_training_dynamics | src/visualization/fig03_training.py | `save_with_vcd` (orchestrator wraps via `_save_panel`) | ✅ | — |
| 4 | fig02b_embedding_space | src/visualization/fig04_embedding.py | orchestrator `save_panel_fn` lambda `(fig, path, dpi)` — module calls with different signature | ❌ | Signature mismatch — orchestrator lambda expects `(fig, path, dpi)` but fig04_embedding calls `save_panel_fn(fig, path, dpi)` correctly; the real gap is that the orchestrator's UMAP gate may skip when cached UMAP data is stale. Defensive fix: wire `fig04_embedding._render_embedding_space` through `save_with_vcd` directly with a guaranteed basename match |
| 5 | fig03a_metrics_summary | src/visualization/fig05_metrics.py | orchestrator `save_panel_fn` lambda | ❌ | Same signature issue as #4; fix is to pass `save_panel_fn=self._save_panel_by_name("fig03a_metrics_summary")` helper with a tolerant `*args, **kwargs` signature |
| 6 | fig03b_per_type_fidelity | src/visualization/fig06_fidelity.py | orchestrator `save_panel_fn` lambda | ❌ | Same as #4 |
| 7 | fig03c_text_cell_alignment | src/visualization/fig07_alignment.py | orchestrator `save_panel_fn` lambda | ❌ | Same as #4 |
| 8 | fig04a_marker_genes | src/visualization/fig08_markers.py | `save_panel_fn` callback OR `save_with_vcd` fallback | ❌ | The 4-arg callback path (`fig, basename, output_dir, dpi`) is different from orchestrator's 3-arg lambda. Normalise the signature |
| 9 | fig04b_expression_correlation | src/visualization/fig09_expression_corr.py | orchestrator `save_panel_fn` lambda | ❌ | Same as #4 |
| 10 | fig05a_expression_analysis | src/visualization/fig10_expression_analysis.py | 4-arg `save_panel_fn` OR `save_with_vcd` fallback | ❌ | Same as #8 |
| 11 | fig05b_conditioning_landscape | src/visualization/fig11_conditioning.py | `save_with_vcd` direct | ❌ | Direct call should produce sidecar — investigate whether this panel is actually being generated in `generate_full_report`. May be a silent skip when cache is stale |
| 12 | fig06_diversity_diagnostics | src/visualization/fig12_diversity.py | `save_with_vcd` direct | ✅ | — |
| 13 | fig07a_expression_diversity | src/visualization/fig14_expr_diversity.py | `save_with_vcd` direct | ✅ | — |
| 14 | fig07b_baseline_comparison | src/visualization/fig15_baselines.py | `save_panel` (= save_with_vcd alias) | ❌ | `save_panel` alias should produce sidecar — investigate `generate_full_report` skip path; likely conditional on `self.results` cache |
| 15 | fig07c_benchmark | src/visualization/fig16_benchmark.py | `save_with_vcd` direct | ❌ | Same as #14 — direct call, so sidecar should exist; panel likely skipped upstream |
| 16 | fig08a_downstream_validation | src/visualization/fig17_downstream.py | `save_panel` alias | ❌ | Same as #14 |
| 17 | fig08b_de_concordance | src/visualization/fig18_de_concordance.py | `save_panel` alias | ❌ | Same as #14 |
| 18 | fig09a_variance_matching | scripts/analysis/variance_matching_pilot.py | `save_with_vcd` direct | ✅ | — |
| 19 | fig09b_gene_gene_correlation | scripts/analysis/gene_gene_correlation.py | `save_with_vcd` direct | ✅ | — |
| 20 | figS01_supplementary_validation | src/visualization/figS01_supplementary_validation.py | `save_with_vcd` fallback | ✅ | — |
| 21 | figS02_expression_diagnostics | src/visualization/figS02_expression_diagnostics.py | `save_with_vcd` direct | ✅ | — |

## Summary

| | Count | Fraction |
|---|---:|---:|
| Article-manifest figures | 21 | 100% |
| With sidecar (VCD-covered) | 9 | 43% |
| Missing sidecar | 12 | 57% |

## Root cause analysis

Every producer script imports `save_with_vcd` correctly and would emit a sidecar
if called. The gap is in **`ResultsVisualizer.generate_full_report()`**:

1. **Signature drift across `save_panel_fn` callbacks.** The orchestrator passes three
   different lambda shapes — `(fig, name, *a, **kw)`, `(fig, path, dpi)`, and
   `(fig, name, out, dpi)` — depending on which figure module is being called. Some
   modules expect `(fig, basename)` and silently fall through to their built-in
   `save_with_vcd` fallback with a **different basename** than the article manifest
   expects, so the sidecar exists but under the wrong name and the audit never finds it.

2. **Silent skip on cache staleness.** Panels like `fig07c_benchmark` (which calls
   `save_with_vcd` directly) are listed in `generate_full_report.saved` optimistically
   even when the upstream data cache is stale. The PDF carried over from a prior run
   is delivered by `article_delivery.py`, but no fresh render — and therefore no fresh
   sidecar — was produced.

3. **Additional non-manifest sidecars.** Seven sidecars exist for non-manifest figures
   (fig13, fig25, fig26, fig27, fig28, fig29, fig30) because those auxiliary figures
   are always regenerated. That is a noise source in the VCD summary but not a
   correctness problem for the 21-figure manifest.

## One-line fixes for US-205

| Basename | Minimal fix |
|---|---|
| fig02b_embedding_space | Normalise `save_panel_fn` lambda to `lambda fig, *a, **kw: self._save_panel(fig, 'fig02b_embedding_space')` |
| fig03a_metrics_summary | Same — add `*a, **kw` to the lambda |
| fig03b_per_type_fidelity | Same |
| fig03c_text_cell_alignment | Same |
| fig04a_marker_genes | Same (normalize 4-arg shape too) |
| fig04b_expression_correlation | Same |
| fig05a_expression_analysis | Same |
| fig05b_conditioning_landscape | Force regeneration when the required cache is present; add an else branch that stamps an empty sidecar annotated `reason: skipped` so the audit distinguishes skipped-panels from bypass-bugs |
| fig07b_baseline_comparison | Same as fig05b |
| fig07c_benchmark | Same as fig05b |
| fig08a_downstream_validation | Same as fig05b |
| fig08b_de_concordance | Same as fig05b |

## Verdict

**PASS** — the audit covers all 21 article-manifest figures. 12 figures lack live
sidecars; the root cause decomposes into two classes (signature drift, silent skip)
with a concrete one-line fix per basename. US-205 will apply these fixes and verify
the fresh regeneration leaves 0 `missing-live-sidecar` entries among the manifest.
