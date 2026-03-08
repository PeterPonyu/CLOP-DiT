# Article Multi-Agent Improvement Plan

*Last revised: 2026-03-06*

This document defines the article "trunk," the 2-agent trunk-optimization team, and the 8-agent full-article improvement plan. Use with [ARTICLE_SUBSECTION_POLICY.md](ARTICLE_SUBSECTION_POLICY.md) for subsection-level writing rules.

---

## 1. Article trunk definition

The **trunk** is the core scientific narrative that must be internally consistent and reproducible:

- **Materials and Methods** (full section): Dataset, CLOP, DiT, Decoding, Evaluation Framework.
- **Results** (full section): All subsections from Training Dynamics through Downstream Biological Validation.

**Excluded from trunk**: Introduction, Discussion, Conclusions, Appendices, front/back matter (abstract, author contributions, data availability, etc.). Those are handled by the full 8-agent plan.

---

## 2. Two trunk agents

| Agent | Name | Scope | Responsibilities |
|-------|------|-------|------------------|
| **Trunk Agent 1** | Methods coherence and reproducibility | Materials and Methods (all subsections) | Check: dataset description and splits, CLOP/DiT equations and parameters, decoding and evaluation metrics; alignment with scripts and configs (e.g. `configs/clop_v9.3.yaml`, `scripts/regenerate_report.sh`); consistent notation and terminology; no missing steps for reproduction. |
| **Trunk Agent 2** | Results and figure-text alignment | Results (all subsections) | Check: each Results subsection references the correct figure(s); narrative matches figure panels and captions; numbers in text match figure/caption; one clear message per subsection; smooth transitions between subsections. |

### Trunk workflow

- Run Trunk Agent 1 on Methods, then Trunk Agent 2 on Results (order matters: Methods first so Results can assume methods are fixed).
- Output: a short report per agent (subsection-level issues, suggested edits, reproducibility gaps). Integrate edits into the .tex in a single pass to avoid conflicts.

---

## 3. Eight agents for the whole article

The whole article is partitioned so that each of 8 agents has a well-defined scope. Together they cover the full manuscript.

| Agent | Scope (sections/subsections or theme) | Focus |
|-------|--------------------------------------|-------|
| **Agent 1** | Introduction | Positioning, gap, contribution; first mention of CLOP-DiT and Figure 1; consistency with abstract and conclusions. |
| **Agent 2** | Materials and Methods — Dataset + CLOP | Dataset subsection and CLOP subsection: data source, splits, text template, CLOP loss and architecture; reproducibility. |
| **Agent 3** | Materials and Methods — DiT + Decoding + Evaluation | DiT subsection, Decoding, Evaluation Framework: equations, CFG, metrics, primary operating points; alignment with code. |
| **Agent 4** | Results — Training + Embedding + Core metrics | Training Dynamics, CLOP Embedding Space, Core Evaluation Metrics (including Table 1 and Figure 4); figure-text alignment. |
| **Agent 5** | Results — Per-type, markers, expression | Per-Type Generation Quality and Text-Cell Alignment, Marker Gene Fidelity, Expression Correlation and Dimension Analysis (Figs 5-8). |
| **Agent 6** | Results — Conditioning, diversity, baselines, downstream | Conditioning Landscape and Diversity Trade-Offs, Baseline Comparison and Composite Benchmark, Downstream Biological Validation (Figs 9-15). |
| **Agent 7** | Discussion | Interpretation (KNN gap, two regimes, FD caveat, biological evidence, applications, limitations); consistency with Results. |
| **Agent 8** | Conclusions + Abstract + front/back matter | Conclusions section; Abstract; optional pass on abbreviations, author contributions, data availability, supplementary statement; consistency with main text. |

### Coverage

- **Trunk Agent 1** = Agents 2 + 3 (Methods)
- **Trunk Agent 2** = Agents 4 + 5 + 6 (Results)

---

## 4. Execution order (recommended)

1. **Agent 2** (Methods: Dataset + CLOP)
2. **Agent 3** (Methods: DiT + Eval)
3. **Agent 1** (Introduction) — after methods stable
4. **Agent 4** (Results: Training, Embedding, Core metrics)
5. **Agent 5** (Results: Per-type, markers, expression)
6. **Agent 6** (Results: Conditioning, baselines, downstream)
7. **Agent 7** (Discussion) — after all Results
8. **Agent 8** (Conclusions, Abstract, front/back) — last

---

## 5. Deliverables per agent

- **Checklist**: Subsection-level checklist derived from [ARTICLE_SUBSECTION_POLICY.md](ARTICLE_SUBSECTION_POLICY.md) (opening, figure ref, one claim, length, terminology, numbers).
- **Report**: List of issues (with line or subsection reference), suggested edits, and a short summary.
- **Edits**: Optional: patch or suggested .tex diffs for the assigned scope (to be applied by maintainer or in a single integration step).

---

## See also

- [ARTICLE_SUBSECTION_POLICY.md](ARTICLE_SUBSECTION_POLICY.md) — Subsection writing rules
- [WRITING_AND_DOCS_POLICY.md](WRITING_AND_DOCS_POLICY.md) — Manuscript source of truth
- [FIGURE_ORGANIZATION.md](FIGURE_ORGANIZATION.md) — Figure logic and caption policy