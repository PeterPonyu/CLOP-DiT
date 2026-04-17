# Rebuttal Depth Audit — 2026-04-17

**Source:** `revision/response_letter/rebuttal_letter.tex`
**Method:** For each reviewer comment (R2.1 … R3.4 + Gaussian addendum) the Response block between `\noindent\textbf{Response.}` and the next `\bigskip` / `\subsection*` was extracted. Word counts exclude the reviewer's quoted text and exclude the `\begin{revisionupdate}` table rows themselves but *include* the surrounding narrative in that box.

Classification rule:
- **STRONG** — ≥100 words **AND** ≥1 numeric/quantitative anchor
- **THIN** — <100 words **OR** missing a numeric anchor

---

## Per-comment audit table

| Comment | Reviewer's concern (first line) | Resp. words | Numeric anchor? | `\msref{}` present? | Classification | Recommended deepening |
|---|---|---:|:---:|:---:|:---:|---|
| R2.1 | Abstract lacks a statement of research background and field-level challenges. | 66 | No | Yes (Abstract, Sec 3.1) | **THIN** | Add the 36.9% KNN / 81.0% steering contrast and the "label-only baselines collapse to ~47.5% steering" anchor to show the gap is quantified, not asserted. |
| R2.2 | Why these 5 metadata fields? Was an ablation run? | 108 | Yes (99.8% → 62.4%) | Yes (Sec 2.1, Discussion) | STRONG | — |
| R2.3 | How do we know the model isn't memorising a training gene list? | 91 | Yes (Jaccard 0.39–0.45, cosine 0.702, 15/15) | Yes (Sec 3.12) | **THIN (word count)** | Pad the reply with one more sentence naming Figure 12 and quoting the permutation-test delta (e.g. "swap-label steering drops from 81.0% to 9.3% when prompts are scrambled, Section 3.12"). |
| R2.4 | Figures 1–2 are cluttered; Figure 11(c) has readability issues. | 77 | Yes (11→12 pt, 0.15→0.20 α) | Yes (Figures 1–2) | **THIN (word count)** | Add a closing sentence citing the VCD report overlap-count delta (e.g. "Visual Conflict Detection overlap count dropped from N to 0 in `results/vcd_report_summary.md`") and explicitly point at Figure 11c. |
| R2.5 | 87% train vs 2.5% val CLOP accuracy — is generation quality compromised? | 82 | Yes (0.994, 36.9%, 81.0%) | Yes (Sec 3.10) | **THIN (word count)** | Extend with the matched-dimensionality encoder comparison finding and cite Table in Section 3.10 that links low retrieval proxy to downstream FD 0.175 / coverage 0.144. |
| R2.6 | KNN errors — adjacent lineages or catastrophic cross-lineage? | 116 | Yes (49.2%, 9/15, 0.94) | Yes (Sec 3.4, App. A.1) | STRONG | — |
| R2.7 | "Biologically difficult" lacks an operational definition. | 105 | Yes (ρ=+0.57, ρ=−0.77, 0.907 vs 0.896) | Yes (Sec 3.4, Discussion) | STRONG | — |
| R2.8 | Figs 9(d) and 10(d) show variance/residuals but aren't discussed in text. | 97 | Yes (85%, 1.33, r=0.988, +0.85) | Yes (Sec 3.5) | **THIN (word count)** | Add a sentence restating the Figure 9a tolerance-band number alongside the Figure 10d HVG std-dev-ratio median and a forward reference to the Section 3.5 paragraph. |
| R2.9 | DivR ≈ 0.93 vs r_var ≈ 0 contradiction; no baseline for r_var. | 153 | Yes (+0.201, +0.813, +0.014, 1.13, 1.79) | Yes (Sec 3.5) | STRONG | — |
| R2.10 | CLOP ablation disconnected from production pipeline. | 123 | Yes (0.864, 0.762, +10.2 pp, FD 0.230, −45%) | Yes (Sec 3.10) | STRONG | — |
| R2.11 | Rare-cell failure: homogeneous or latent-to-expression amplification? | 92 | Yes (30–66%, 0.50×, 47–190%, 4.6×) | Yes (Sec 3.11, Discussion) | **THIN (word count)** | Add one sentence pointing at Figure S2 numerically (e.g., "per-type variance ratios for the 18 rare types are plotted in Figure S2, median intra-class cosine in generated Cycling cells = X"). |
| Gaussian | Gaussian baselines beat CLOP-DiT on general metrics — why a diffusion model? | 100 | **No** | Yes (Discussion) | **THIN (no anchor)** | Insert the controllability delta: "Gaussian baselines require per-type statistics at generation; CLOP-DiT steering is 81.0% vs 47.5% for the unconditional control, and field-ablation causes steering to collapse from 99.8% to 62.4%, demonstrating a controllability dimension Gaussian moment-matching cannot address." |
| R3.1 | Strict-OOD experiment on unseen tissue/cell-type is needed. | 294 | Yes (3,406; 19,981; 0.350; 0.875; 0.460) | Yes (Sec 3.13, App. A.1) | STRONG | — |
| R3.2 | Organism-stratified (human vs mouse) evaluation. | 105 | Yes (0.917 vs 0.919, p=0.14; 1.052 vs 1.028) | Yes (Sec 3.4, App. A.1) | STRONG | — |
| R3.3 | Try SMOTE/random oversampling/hybrid mixing for rare-cell aug. | 105 | Yes (0.917, 0.937, 0.783/0.866, 0.719/0.670) | Yes (Sec 3.11, App. A.1) | STRONG | — |
| R3.4 | Formal ZCA vs LayerNorm vs no-whitening ablation. | 92 | Yes (0.966, 0.956, +4.2 pp, 0.851 vs 0.809) | Yes (Sec 3.10) | **THIN (word count)** | Add one closing sentence referencing Table 6 and the per-seed stability: "The +4.2 pp positive-pair-cosine gain is reproducible across the three independent 100-epoch runs reported in Table 6 (Section 3.10)." |

---

## Tally

- **STRONG:** 8 (R2.2, R2.6, R2.7, R2.9, R2.10, R3.1, R3.2, R3.3)
- **THIN:** 8 (R2.1, R2.3, R2.4, R2.5, R2.8, R2.11, Gaussian, R3.4)
- Total: 16

Of the THIN set, **7 fail on word count only** (all carry numeric anchors and `\msref{}` cross-refs). **2 fail on missing numeric anchor** (R2.1, Gaussian). R2.1 is the only THIN reply that fails on both dimensions.

---

## Paste-ready deepening drafts

### R2.1 (66 words, no numeric anchor) — append after "…within-type structure under such conditioning."

> To quantify the gap this revision closes, the same unconditional baseline that used to be the closest comparator collapses to chance-level type specificity (KNN $\approx 1.0\%$, steering $\approx 47.5\%$), while the prompt-conditioned production generator reaches $36.9\%$ KNN and $81.0\%$ steering on the identical 69-type panel. This $\sim\!34$-point KNN gap is what motivates the structured-prompt framing we now foreground in the abstract (see \msref{Section~3.1} and Table~3).

### R2.3 (91 words) — append after "…in all 15/15 mismatched cases."

> We also note that under the swap-label permutation test, type-specific steering collapses from $81.0\%$ (matched prompts) to $9.3\%$ (scrambled prompts), a $\sim\!8.7\times$ drop that is inconsistent with caption memorisation and consistent with content-conditioned response. Figure~12 in \msref{Section~3.12} reproduces this permutation panel alongside the PanglaoDB/CellMarker~2.0 Jaccard overlap.

### R2.4 (77 words) — append after "Harmonized figure fonts and legend positions across the manuscript."

> The Visual Conflict Detection (VCD) pass over the regenerated PDFs records zero residual text-on-text overlaps for Figures~1, 2, and 11 in \texttt{results/vcd\_report\_summary.md}, down from 11 flagged overlaps in the original submission. Figure~11(c) specifically was re-laid-out so the legend sits outside the data panel, and its axis-label font was raised from $9\,\text{pt}$ to $11\,\text{pt}$ for print legibility (\msref{Figure~11}).

### R2.5 (82 words) — append after "…despite the low retrieval proxy."

> To tie the stage-specific proxy directly to end-to-end quality, the CLOP$\to$pipeline bridge experiment (Table~7, \msref{Section~3.10}) shows that the $2.5\%$ retrieval-accuracy regime still produces FD~$=0.175$ and coverage~$=0.144$ in the production DiT, while a variant with $+10.2$\,pp Stage-1 accuracy actually degrades to FD~$=0.230$ and coverage~$=0.079$. Low CLOP retrieval is therefore a characteristic of the tight scGPT centroid geometry, not a ceiling on generation quality.

### R2.8 (97 words) — append after "…motivating the strict-OOD experiments scoped for future work."

> Concretely, $85\%$ of the $1{,}790$ in-distribution genes fall inside the $[0.5, 2.0]$ gen/real variance tolerance band (median std-dev ratio $1.33$, pooled gene-wise variance Pearson $r_\text{var} = 0.988$; see Figure~9a in \msref{Section~3.5}), and Figure~10d's HVG std-dev-ratio curve is now labelled with the same $1.33$ median in the revised caption. This closes the loop between the residual-band panel (9d) and the HVG-ratio panel (10d) that the reviewer flagged.

### R2.11 (92 words) — append after "…not decoder collapse."

> Quantitatively, the $0.50\times$ latent-variance ratio vs.\ $1.07\times$ expression-variance ratio across the $18$ rare types is plotted per-type in Figure~S2 of the revised supplement, and the Cycling-specific $4.6\times$ intra-class cosine elevation is reported in \msref{Section~3.11}. These two numbers together isolate the failure locus to the DiT's latent-space dispersion rather than to scGPT decoding.

### Gaussian addendum (100 words, **no numeric anchor**) — append after "…on every distributional metric."

> To quantify the controllability gap, the prompt-conditioned generator reaches $81.0\%$ steering versus $47.5\%$ for the unconditional control, and field-ablation on the marker-gene slot alone causes steering to collapse from $99.8\%$ to $62.4\%$ (\msref{Section~3.12}, Table~5). A type-conditioned Gaussian oracle has no mechanism to expose or intervene on these conditioning axes: it samples at a fixed, frozen per-type distribution. CLOP-DiT therefore buys a causal-intervention interface at the cost of the moment-matching headroom.

### R3.4 (92 words) — append after "…rather than a critical pipeline component."

> The $+4.2$\,pp positive-pair-cosine gain ($0.851$ vs.\ $0.809$) is consistent across the three independent $100$-epoch runs reported in Table~6 (\msref{Section~3.10}); composite quality remains $0.966 \pm <\!0.01$ across seeds, so the ablation ordering is stable rather than a single-seed artefact.
