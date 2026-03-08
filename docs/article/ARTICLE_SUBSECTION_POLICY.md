# Article Subsection Writing Policy

*Last revised: 2026-03-07*

This document defines the subsection-level writing improvement policy for the MDPI Biology article, [articles/clop_dit_biology.tex](../articles/clop_dit_biology.tex). Extends [WRITING_AND_DOCS_POLICY.md](WRITING_AND_DOCS_POLICY.md). Human and agent edits to the article should follow these rules.

---

## 1. Scope of a subsection

- Each `\subsection{...}` (and the block up to the next `\section` or `\subsection`) is one **subsection unit**.
- Policy applies to: Dataset, CLOP, DiT, Decoding, Evaluation Framework, and all Results subsections (Training Dynamics, CLOP Embedding Space, Core Evaluation Metrics, Per-Type Generation Quality, Marker Gene Fidelity, Expression Correlation, Conditioning Landscape, Baseline Comparison, Downstream Biological Validation), plus Discussion and Conclusions as section-level units.

---

## 2. Structural rules per subsection

- **Opening**: First sentence states the purpose or main message of the subsection (e.g. what the reader will learn or what the figure shows).
- **Figure/table reference**: If the subsection is tied to a figure or table, the first or second paragraph must reference it explicitly (`Figure~\ref{...}` or `Table~\ref{...}`) and state the takeaway, not only "Figure X shows…".
- **One primary claim per subsection**: Avoid packing multiple unrelated claims; if needed, split or add a short subheading (e.g. bold run-in).
- **Length**: Methods subsections: sufficient for reproducibility (datasets, splits, metrics, scripts). Results subsections: 1-3 short paragraphs typical; avoid long prose that duplicates caption text.
- **Closing**: Optional one-sentence takeaway or bridge to the next subsection where it helps flow.

---

## 3. Style and consistency

- **Terminology**: Use the canonical list from WRITING_AND_DOCS_POLICY (CLOP-DiT, scGPT, flow matching, BiomedBERT, AdaLN-Zero, CFG). No mixing of naming (e.g. no "CLOP-DIT" or "CLOP DiT").
- **Numbers and metrics**: Report with consistent precision (e.g. one decimal for percentages, two for ratios). Use `\ref` and `\cref` for figures/tables; avoid "above"/"below."
- **Voice**: Prefer active where natural ("We curated…", "CLOP aligns…"); passive acceptable for methods. Keep consistent within a subsection.
- **Abbreviations**: Define at first use in the main text; abbreviations list in back matter must match.

---

## 4. Figure-caption-subsection alignment

- Subsection narrative must match what the figure and its caption say (metrics, panel order, interpretation).
- Do not introduce in the subsection a metric or result that the figure does not support.
- Caption carries detailed panel description; subsection carries high-level message and interpretation.

---

## 5. Subsection checklist (for agents and reviewers)

When editing or auditing a subsection, verify:

- [ ] Opening sentence states purpose or main message
- [ ] Figure/table reference is explicit and states takeaway
- [ ] One primary claim (or split with bold run-in)
- [ ] Length appropriate (Methods: sufficient for reproducibility; Results: 1-3 paragraphs)
- [ ] Terminology matches canonical list
- [ ] Numbers and metrics use consistent precision
- [ ] Narrative matches figure/caption content
- [ ] No unsupported metrics or results in text

**Section-level signposting (for major sections only)**:
- [ ] Methods has organization paragraph with \ref targets
- [ ] Results has figure-family organization paragraph
- [ ] Section transitions present at Introduction→Methods, Methods→Results, Results→Discussion

---

## 6. Section-level transitions and signposting

### 6.1 Required transitions
Major sections must include explicit bridges:
- **Introduction ending**: Transition sentence mapping paper organization (\ref{sec:dataset}--\ref{sec:conclusions})
- **Methods ending**: Bridge sentence connecting evaluation framework to results
- **Results ending**: Mini-conclusion summarizing main findings before Discussion
- **Discussion opening**: Brief sentence linking results to interpretation

### 6.2 Signposting paragraphs
- **Methods opening**: Must include organization paragraph listing subsections with \ref targets
- **Results opening**: Must include figure-family organization paragraph mapping evidence types to figure ranges
- Format: "We organize X in N parts: (1) ... (N) ..."

### 6.3 Figure-family transitions
Between thematic figure groups in Results, add explicit transition sentences:
- Format: "Having established [prior finding], we next examine [new focus]."
- Purpose: Guide reader through evidence chain without assuming implicit connections

---

## See also

- [WRITING_AND_DOCS_POLICY.md](WRITING_AND_DOCS_POLICY.md) — Manuscript source of truth, terminology, figure text
- [ARTICLE_AGENT_PLAN.md](ARTICLE_AGENT_PLAN.md) — Trunk definition, 2-agent and 8-agent improvement plan
- [FIGURE_ORGANIZATION.md](FIGURE_ORGANIZATION.md) — Figure logic and caption policy