# Submission Checklist

Use this checklist before submitting the manuscript. Full rationale and task details are in [REVIEWER_CONCERNS_AND_NEXT_STEPS.md](REVIEWER_CONCERNS_AND_NEXT_STEPS.md).

---

## High priority (avoid editorial return)

| # | Task | Done |
|---|------|------|
| 1 | Add Data Availability statement (GEO, code repo or "upon request", config) | ☐ |
| 2 | Add REPRODUCIBILITY.md or supplement with reproduce instructions | ☐ |
| 3 | Replace affiliation, correspondence, funding placeholders in `.tex` | ☐ |
| 4 | Add ethics statements (IRB waiver / not applicable, informed consent) | ☐ |
| 5 | Fill Author Contributions (CRediT-style) | ☐ |
| 6 | Set Acknowledgments and Conflicts of interest | ☐ |

---

## Medium priority (reviewer expectations)

| # | Task | Done |
|---|------|------|
| 7 | Add uncertainty/CI paragraph; optionally bootstrap table in supplement | ☐ |
| 8 | Clarify primary vs exploratory comparisons (e.g. CFG=2.0 Euler-10) | ☐ |
| 9 | Add text-condition construction paragraph and software/versions | ☐ |
| 10 | Clarify validation split (which 8 datasets, how chosen) | ☐ |
| 11 | One sentence on OOD limitation in Limitations | ☐ |

---

## Polish

| # | Task | Done |
|---|------|------|
| 12 | Confirm Fig 12 and Fig 14/15 files and captions per [FIGURES_12-15_POLICY.md](FIGURES_12-15_POLICY.md); reference pass | ☐ |
| 13 | Optional: downstream numbers in abstract/conclusions; supplement list | ☐ |

---

## Before submission

- [ ] All 15 article figures regenerated and verified: `bash scripts/verify_article_figures.sh`
- [ ] Figs 12–15 present: `panel_o_baseline_comparison.pdf`, `panel_s_benchmark.pdf`, `fig_downstream_pq.pdf`, `panel_r_de_concordance.pdf` (see [FIGURES_12-15_POLICY.md](FIGURES_12-15_POLICY.md))
- [ ] Article builds: `cd articles && latexmk -pdf clop_dit_biology.tex`
- [ ] No TODO or "Please add" placeholders remain in `.tex`
