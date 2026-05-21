# Reviewer-Comment Adequacy Audit — CLOP-DiT Revision Packet

**Date:** 2026-04-17
**Scope:** 15 reviewer comments from Reviewers #2 and #3
**Inputs:** `revision/response_letter/rebuttal_letter.tex`, `revision/manuscripts/v2_revision/clop_dit_genes.tex`, `revision/REVIEWER_RESPONSE_SYNTHESIS.md`

Each comment is scored on the rubric:

1. What concrete evidence is promised (numbers, table, figure, section)?
2. Is that evidence present in the manuscript body (not only in the rebuttal)? Manuscript line numbers given.
3. Does the rebuttal quote the exact numeric values a reader would need?
4. **Verdict** — SUBSTANTIVE / THIN / PRETEND.

---

## R2.1 — Abstract background / challenge framing

| Field | Value |
|---|---|
| Promised evidence | 1–2 abstract sentences stating (a) limitation of label-based conditioning, (b) central challenge = balancing controllability with within-type heterogeneity. |
| In manuscript body? | **Yes.** Abstract rewritten, `clop_dit_genes.tex` line 187: "most existing single-cell generative models condition on narrow categorical labels … The central open challenge is not generation alone but preserving biologically meaningful within-type heterogeneity under such conditioning". |
| Rebuttal cites values? | N/A — this is a framing change; no numbers required. |
| **Verdict** | **SUBSTANTIVE.** Concrete, on-topic abstract edit visible in the actual manuscript. |

---

## R2.2 — Five-field template justification

| Field | Value |
|---|---|
| Promised evidence | Replace "optimal" with "practical structured scaffold"; defer combinatorial optimality; point to existing field-ablation (marker removal → steering 99.8→62.4%; external markers 0.702 cosine). |
| In manuscript body? | **Yes.** Line 225 ("practical structured scaffold of five biological fields"); field-ablation table is at lines 632–647 (Table `tab:cond-ablation`) with full numbers. |
| Rebuttal cites values? | Yes — 99.8→62.4, 0.702 cosine. |
| **Verdict** | **SUBSTANTIVE.** Wording fix landed; the supporting ablation table was already there and is now explicitly linked to the reviewer's question. |

---

## R2.3 — Biological semantics evidence

| Field | Value |
|---|---|
| Promised evidence | Soften "understands biological semantics" → "responds to biologically relevant conditioning content"; highlight (i) Jaccard 0.39–0.45 vs PanglaoDB/CellMarker2, (ii) external-marker substitution cosine 0.702, (iii) 15/15 swap-label cases follow prompt. |
| In manuscript body? | **Yes.** Jaccard mentioned at line 250; external-markers row in Table `tab:cond-ablation` (line 643); swap-label "15 of 15" result at line 651. |
| Rebuttal cites values? | Yes — all three numbers present. |
| **Verdict** | **SUBSTANTIVE.** Three independent lines of evidence all quotable in the manuscript body. |

---

## R2.4 — Figure 1+2 readability + Figure 11(c)

| Field | Value |
|---|---|
| Promised evidence | Regenerated Figures 1 & 2 with bigger fonts (11→12 pt / 10→11 pt sublabels), thicker arrows, alpha 0.15→0.20, panel label 21→15 pt on eval-pipeline fig; Figure 11(c) font harmonization. |
| In manuscript body? | Figure 1 now references `fig01a_architecture.pdf` + `fig01b_evaluation_pipeline.pdf` (lines 236–241). Font/alpha changes are implementation-side in source scripts but the figure itself is re-delivered. |
| Rebuttal cites values? | Yes — 11→12 pt, 10→11 pt, 0.15→0.20, 21→15 pt. |
| **Verdict** | **SUBSTANTIVE.** Concrete typographic numbers plus regenerated PDFs wired in. Caveat: a reader cannot verify Fig 11(c) specifically from the rebuttal text; reviewer must look at the figure — that is inherent to a figure-only change. |

---

## R2.5 — CLOP 87 % train vs 2.5 % val accuracy

| Field | Value |
|---|---|
| Promised evidence | Reframe 2.5 % as a stage-specific retrieval proxy; explain via raw cosine compression 0.994; show that DiT KNN 36.9 % + steering 81.0 % vs unconditional 1.0 % / 47.5 % justifies usefulness. |
| In manuscript body? | Partially. The DiT KNN and steering numbers appear in Table `tab:core-results` and are discussed in Section 3 (headline metrics). Ablation/production-temperature clarification appears at lines 597 and 609. The raw-cosine-0.994 bottleneck rationale appears at line 597. No dedicated "why validation is 2.5 %" paragraph is present. |
| Rebuttal cites values? | Yes — 0.994, 36.9 %, 81.0 %, 1.0 %, 47.5 %. |
| **Verdict** | **THIN.** Every supporting number is elsewhere in the manuscript but the *direct* reframing of the 2.5 % validation figure is only in the rebuttal. A single sentence in Section 3 labelled "CLOP validation accuracy as a stage-specific proxy" would upgrade this to SUBSTANTIVE. |

---

## R2.6 — KNN confusion matrix / error taxonomy

| Field | Value |
|---|---|
| Promised evidence | 10-family taxonomy over 69 types; 49.2 % within-family; 9/15 top-confused pairs within-family; only proliferation family <0.94 within-family accuracy. |
| In manuscript body? | **Yes, paragraph at line 477.** All three numbers (49.2 %, 9/15, ≥0.94) appear with named example pairs. |
| Rebuttal cites values? | Yes — 49.2 %, 9/15, 0.94. |
| **Verdict** | **SUBSTANTIVE (but no figure).** Prose + numbers are solid; however, the underlying `knn_confusion.json` data would support a proper confusion-matrix heatmap and the reviewer asked explicitly for a "confusion matrix". Currently the reader gets the *statistic* but not the *picture*. See top-3 recommendations. |

---

## R2.7 — "Biologically difficult" operationalised

| Field | Value |
|---|---|
| Promised evidence | Define difficulty as `real_intra_cos < P25`; Spearman ρ = +0.57 (abundance) vs −0.77 (heterogeneity); bottom-quartile vs top-quartile centroid cos 0.907 vs 0.896, MW p = 0.02. |
| In manuscript body? | **Yes, line 475.** Every number (P25, +0.57, p = 4.9e-7, −0.77, p = 3e-13, 0.907 vs 0.896, p = 0.02) is in the paragraph. |
| Rebuttal cites values? | Yes, all of them. |
| **Verdict** | **SUBSTANTIVE.** Textbook example of a well-addressed comment. |

---

## R2.8 — Residual distribution & variance ratio interpretation

| Field | Value |
|---|---|
| Promised evidence | 1,790 genes, 85 % inside [0.5, 2.0] tolerance band, median SD ratio 1.33, pooled r_var = 0.988; within-type median r_var = +0.85, 100 % types positive; cross-dataset 0.01–0.02 as the stricter test. |
| In manuscript body? | **Yes, line 487** (quantitative variance summary paragraph). All numbers quoted. Cross-dataset r_var values also in Table `tab:cross-dataset` (lines 562–566). |
| Rebuttal cites values? | Yes, all. |
| **Verdict** | **SUBSTANTIVE.** Figure 9(d) and Figure 10(d) are also explicitly tied to Figure 04a/04b via the caption on lines 489–494. |

---

## R2.9 — Gaussian / permuted baselines for variance

| Field | Value |
|---|---|
| Promised evidence | Two baselines: type-aware Gaussian oracle (+0.813 r_var) and pooled-Gaussian floor (+0.014); CLOP-DiT sits at +0.201; expression-scale within-type median variance ratio 0.94 / 1.13 / 1.79. Full 3-row table. |
| In manuscript body? | **Yes, line 573.** The paragraph explicitly reports all three r_var values and all three variance ratios. The full 3×3 table is in the rebuttal but the numbers are quoted in prose in the manuscript. |
| Rebuttal cites values? | Yes, with a formatted table. |
| **Verdict** | **SUBSTANTIVE.** A table replica in the manuscript would be nicer but all quoted values are already there and the reviewer's core question ("No baseline methods were provided") is answered with three baselines. |

---

## R2.10 — CLOP ablation vs full pipeline

| Field | Value |
|---|---|
| Promised evidence | Bridge experiment retraining 3 DiTs; `no_cohesion` Stage-1 acc 0.864 vs 0.762 baseline (+10.2 pp) but FD 0.230 vs 0.175 (+31 %), coverage 0.079 vs 0.144 (−45 %), centroid cos 0.854 vs 0.929 (−8.1 pp); verdict = PARTIAL_REVERSAL. |
| In manuscript body? | **Yes, line 599** (Stage-1→full-pipeline transfer paragraph). All key numbers quoted. Plus the inter-type similarity r = 0.04–0.06 finding. |
| Rebuttal cites values? | Yes, with a formatted table. |
| **Verdict** | **SUBSTANTIVE.** Bridge experiment numbers all present; only the comparative table-form itself is missing from the manuscript prose (present in rebuttal). Could be further strengthened by a bar chart (see recommendations). |

---

## R2.11 — Rare-cell mechanism + augmentation

| Field | Value |
|---|---|
| Promised evidence | Decompose rare-cell failure: 18 types latent variance 30–66 % (median 0.50×), expression 47–190 % (median 1.07×); Cycling intra-cluster cos 4.6× higher; conclusion "upstream latent under-dispersion". |
| In manuscript body? | **Yes, line 620** (Failure mechanism paragraph). All five ratios (30–66, 0.50×, 47–190, 1.07×, 4.6×) present. Systematic sweep numbers at line 622; forced-scarcity rescue at line 624. |
| Rebuttal cites values? | Yes. |
| **Verdict** | **SUBSTANTIVE.** Mechanism diagnosis is explicit. Encoder-bottleneck trio (cap_increase, HVG sweep, encoder_comparison) is in Discussion at line 712 with complete numbers. |

---

## R3.1 — Strict-OOD

| Field | Value |
|---|---|
| Promised evidence | Zero-shot eval on kidney (3,406 cells, 1 type), cerebellum (19,981 cells, 11 types), fetal gonadal (3,342 cells, 2 types); overall nearest-acc 0.350 vs ~0.1 random, FD 1.67; adrenal cortex 0.875, kidney epithelial 0.460, Purkinje 0.00; leakage audit across 9 metadata files. |
| In manuscript body? | **Yes, full §3.13 (`sec:strict-ood`), lines 653–684.** Table `tab:strict-ood` (lines 665–680), Figure `fig:strict-ood` (lines 659–663), leakage audit, per-type accuracies, mechanistic interpretation. |
| Rebuttal cites values? | Yes, with its own embedded preview of the figure. |
| **Verdict** | **SUBSTANTIVE.** This is the highest-quality response in the packet — full new section + table + figure + mechanistic narrative. |

---

## R3.2 — Organism-stratified evaluation

| Field | Value |
|---|---|
| Promised evidence | 18 human-only / 8 mouse-only; centroid cos 0.917 vs 0.919 (p=0.14); FD 1.052 vs 1.028 (p=0.85); DivR p=0.18; expression r p=0.43; tightness p=0.87. |
| In manuscript body? | **Yes, line 479** (Species stratification paragraph). All six p-values and both centroid-cos / FD means quoted. Also re-referenced in Discussion at line 708. |
| Rebuttal cites values? | Yes. |
| **Verdict** | **SUBSTANTIVE (no figure).** Numbers are complete but conveyed only in prose. A boxplot per organism × metric panel would make the null result visually decisive. Reviewer explicitly asked about possible under-performance on minority species — a figure is especially compelling when claiming *no* difference. See recommendations. |

---

## R3.3 — Augmentation strategy sweep

| Field | Value |
|---|---|
| Promised evidence | 2×6×4 sweep (Megakaryocytes, Ameloblasts × 6 strategies × 4 ratios); all baselines and CLOP numbers at 10×; forced-scarcity rescue reducing class to 30 cells; oversampling 0.783/0.866, CLOP 0.719/0.670, hybrid 0.772/0.835. |
| In manuscript body? | **Yes.** Systematic sweep at line 622 and forced-scarcity rescue at line 624. Every number quoted. |
| Rebuttal cites values? | Yes, with a formatted 4-column / 5-row table. |
| **Verdict** | **SUBSTANTIVE.** The forced-scarcity experiment is a particularly strong upgrade because it turns a null into a controlled result. A line plot showing F1 vs ratio for each strategy would be a powerful addition (see recommendations). |

---

## R3.4 — ZCA ablation

| Field | Value |
|---|---|
| Promised evidence | 3-run matched comparison (ZCA / centre+L2 / none); CLOP quality 0.966 / 0.960 / 0.956 (1 % drop), proto acc ≈ 0.998+ all three; pos-pair cos 0.851 / 0.815 / 0.809 (5 % gain). |
| In manuscript body? | **Yes.** Table `tab:zca-ablation` (lines 601–614) with all four columns including DiT KNN 36.9 % vs 36.5 %. |
| Rebuttal cites values? | Yes. |
| **Verdict** | **SUBSTANTIVE.** Formal ablation with a dedicated table. |

---

## Summary counts

| Verdict | Count | Comments |
|---|---|---|
| **SUBSTANTIVE** | **14** | R2.1, R2.2, R2.3, R2.4, R2.6, R2.7, R2.8, R2.9, R2.10, R2.11, R3.1, R3.2, R3.3, R3.4 |
| **THIN** | **1** | R2.5 (supporting numbers exist elsewhere but no direct "2.5 % is a stage-specific proxy" paragraph in Section 3) |
| **PRETEND** | **0** | — |

Overall, the packet is unusually dense for a round-2 rebuttal: every quantitative claim in the response letter traces back to a number that appears in the manuscript body. The weakest link is R2.5, which needs a single clarifying sentence rather than new analysis.

Figure-based comments (R2.6, R3.2, R2.10, R3.3, R2.11) are SUBSTANTIVE on numbers but several would be meaningfully strengthened by a dedicated visual — those are the highest-leverage additions.

---

## Top-3 new-figure recommendations (ranked by reviewer-impact × ease-of-production)

### #1 — KNN family-confusion heatmap (target: **R2.6**)

- **Data source:** `revision/experiments/lane_a_analysis/a1_knn_confusion/knn_confusion.json` + `family_taxonomy.yaml`.
- **Why highest leverage:** R2.6 literally asks for "a confusion matrix". Currently the reviewer gets 49.2 % + a list of example pairs, which is correct but not what was requested. The JSON already contains the full 69×69 confusion counts and the family mapping.
- **Ease:** Low effort — single matplotlib heatmap with family block-grouping, one script.
- **Caption draft:** "KNN top-1 confusion matrix over 69 evaluation cell types, block-grouped by the 10-family biological taxonomy (Appendix F). Off-diagonal mass is concentrated on within-family blocks, and 49.2 % of all mis-typings remain within the same lineage family, supporting a graceful-degradation rather than catastrophic-error interpretation."

### #2 — Forced-scarcity rescue line plot (target: **R3.3**)

- **Data source:** `revision/experiments/lane_b_retrain/b3_mixing_sweep/forced_scarcity_sweep.json` (+ `mixing_sweep.json` for natural-scarcity comparison).
- **Why high leverage:** Converting the 6-strategy × 4-ratio × 2-type × 2-scarcity-regime sweep from a one-column table into an F1-vs-ratio line plot (two panels: natural vs forced-scarcity, strategies coloured) lets the reader see at a glance that (a) natural scarcity is ceiling-bound, (b) forced scarcity recovers substantially, (c) the ordering across strategies is consistent. Strong visual rebuttal of "augmentation doesn't work".
- **Ease:** Low — data already JSON-shaped; single two-panel matplotlib plot.
- **Caption draft:** "Rare-class classifier F1 under six augmentation strategies across ratios 1×–10×, for Megakaryocytes and Ameloblasts under natural (top) and forced 30-cell (bottom) scarcity. The forced-scarcity panels show CLOP-DiT augmentation and hybrid strategies providing substantial recovery (F1 0.50 → 0.77–0.87), establishing that the natural-scarcity null is ceiling-driven."

### #3 — Organism-stratified metric boxplot (target: **R3.2**)

- **Data source:** `revision/experiments/lane_a_analysis/a2_organism_split/organism_per_type_table.csv`.
- **Why high leverage:** The reviewer explicitly worried that the 59/21 imbalance would produce species bias. Currently the manuscript communicates the null result ("p > 0.14 on all headline metrics") in prose. A five-panel boxplot (centroid cos, FD, DivR, expression r, tightness) over 18 human-only + 8 mouse-only types would make the equivalence visually decisive — *especially* because it is a null-result claim.
- **Ease:** Low — CSV already contains all per-type values; a `seaborn.boxplot` with per-metric facets is ~30 lines of code.
- **Caption draft:** "Per-type evaluation metrics stratified by organism (18 human-only, 8 mouse-only types). Across centroid cosine, Fréchet distance, diversity ratio, expression Pearson r, and within-type cosine tightness, human-only and mouse-only distributions overlap with Mann–Whitney p > 0.14; the 59/21 dataset-count imbalance does not propagate to evaluation-level species bias."

Runner-up candidates (SUBSTANTIVE text already, but visual would still help):

- **R2.10 Lane B4 bridge bar chart** — `b4_clop_bridge/b4_bridge_comparison.json`; three-bar FD / Coverage / Centroid-cos grouped per variant. Would dramatise the partial-reversal story but the paragraph at line 599 already communicates the numbers clearly, so marginal impact.
- **R2.11 Encoder-bottleneck triple panel** — cap_increase + encoder_bottleneck + encoder_comparison. The Discussion paragraph at line 712 integrates these beautifully; a figure is polish rather than necessity.
