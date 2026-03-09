# Submission Checklist

Use this checklist before submitting the manuscript. Full rationale and task details are in [REVIEWER_CONCERNS_AND_NEXT_STEPS.md](REVIEWER_CONCERNS_AND_NEXT_STEPS.md).

*Last updated: 2026-03-08*

---

## High priority (avoid editorial return)

| # | Task | Done |
|---|------|------|
| 1 | Add Data Availability statement (GEO, code repo or "upon request", config) | ☑ |
| 2 | Add REPRODUCIBILITY.md or supplement with reproduce instructions | ☑ |
| 3 | Replace affiliation, correspondence, funding placeholders in `.tex` | ☑ |
| 4 | Add ethics statements (IRB waiver / not applicable, informed consent) | ☑ |
| 5 | Fill Author Contributions (CRediT-style) | ☑ |
| 6 | Set Acknowledgments and Conflicts of interest | ☑ |

**Status:** All high-priority items resolved as of 2026-03-08. Data Availability (line 611), ethics/IRB (line 607), informed consent (line 609), CRediT contributions (line 603), funding (line 605), acknowledgments (line 624), and conflicts (line 626) are present in `articles/clop_dit_biology.tex`. Affiliation and correspondence are filled with real values (lines 85–97). Supplement references `regenerate_report.sh` and `QUICK_START.md` (line 577).

---

## Medium priority (reviewer expectations)

| # | Task | Done |
|---|------|------|
| 7 | Add uncertainty/CI paragraph; optionally bootstrap table in supplement | ☑ |
| 8 | Clarify primary vs exploratory comparisons (e.g. CFG=2.0 Euler-10) | ☑ |
| 9 | Add text-condition construction paragraph and software/versions | ☑ |
| 10 | Clarify validation split (which 8 datasets, how chosen) | ☑ |
| 11 | One sentence on OOD limitation in Limitations | ☑ |

**Status:** All medium-priority items resolved as of 2026-03-08. Item 7: Bootstrap 95% CI table (Table S5) created in `articles/supplementary_tables.tex` with real data from `results/bootstrap_cis.json` (B=1,000; 6 headline metrics). Line 306 of `.tex` references Table S5. Items 8–11 resolved as before.

---

## Polish

| # | Task | Done |
|---|------|------|
| 12 | Confirm Fig 12 and Fig 14/15 files and captions per [FIGURES_12-15_POLICY.md](../roadmaps/FIGURES_12-15_POLICY.md); reference pass | ☐ |
| 13 | Optional: downstream numbers in abstract/conclusions; supplement list | ☐ |

**Status:** Item 12 remains open — figure PDFs need regeneration and verification. Item 13 remains open — abstract does not yet include explicit downstream ARI/NMI/DE numbers (conclusions section does include some downstream values).

---

## Before submission

- [ ] All 15+ article figures regenerated and verified: `bash scripts/pipeline/verify_article_figures.sh`
- [ ] Figs 12–15 present: `panel_o_baseline_comparison.pdf`, `panel_s_benchmark.pdf`, `fig_downstream_pq.pdf`, `panel_r_de_concordance.pdf` (see [FIGURES_12-15_POLICY.md](../roadmaps/FIGURES_12-15_POLICY.md))
- [ ] Article builds: `cd articles && latexmk -pdf clop_dit_biology.tex`
- [x] No TODO or "Please add" placeholders remain in `.tex` (verified 2026-03-08: only match is in commented-out template line 621)
- [x] Supplementary Tables S1–S5 files created and included in submission bundle (S1: GEO accessions TSV, S2: validation datasets TSV, S3: CFG sweep in LaTeX, S4: biovalidation datasets TSV, S5: bootstrap CIs in LaTeX)
- [x] Stale metric references (e.g. legacy 0.668 composite) removed from derivative docs (fixed 2026-03-08: FIGURE_DESIGN_AND_PROMPT_DIVERSITY_PLAN.md updated to 0.844; deprecation notice added to Evaluation Report)
- [x] Bibliography audit: all 60 bibitems cited; 7 ghost entries removed, 5 orphan entries wired to text
- [x] Runtime bugs fixed: panels_embedding.py (ax→ax_b0), panels_clustering.py (missing mixing var)
- [x] Dimension typo fixed: line 332 "256-dimensional" → "512-dimensional"
- [x] Ablation paradox and rare-cell augmentation caveat added to Discussion
- [x] Panel label audit: all 17 figures verified; fallback path label added in panels_expression.py
