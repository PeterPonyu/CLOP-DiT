# Supplement Citation Audit — Lane A/B/C/D + bonus experiments

**Date:** 2026-04-17
**Manuscript audited:** `revision/manuscripts/v2_revision/clop_dit_genes.tex`
(Introduction through Conclusions, lines 160–724).
**Experiments audited:** 15 completed experiments under `revision/experiments/`
(Lane A ×6, Lane B ×4, Lane C ×1, Lane D ×1, plus 3 bonus mechanism experiments
and 1 ZCA duplicate alias; `lane_b_retrain/b2_zca_ablation/` and
`experiments/zca_ablation/` share identical numeric content).
**Method:** Read each experiment's `results.md`/`SYNTHESIS.md`/preview file,
then grepped the manuscript for headline numbers and phrases. Classification
is **CITED** if the manuscript carries the experiment's fresh numbers or
a reviewer-comment cross-reference, **GAP** if the experiment is not
surfaced (even if an earlier baseline number on the same topic exists).

---

## 1. Summary table

| # | Experiment | Reviewer | Manuscript location | Status | Evidence |
|---|---|---|---|---|---|
| 1 | `a1_knn_confusion` (Lane A1) | R2.6 | Results §Per-Type Generation Quality (line 476) | **CITED** | "49.2% of all KNN mis-typings stay within the same cell-lineage family … 9 of top-15 within-family … pancreatic δ ↔ neuroendocrine, macrophages ↔ MHC-II+ APCs, fibroblasts ↔ mesenchymal stem cells … every family ≥ 0.94 except proliferation/stress/pluripotent." Lifted verbatim from A1 rebuttal. |
| 2 | `a2_organism_split` (Lane A2) | R3.2 | Results §Per-Type (line 478); Discussion §Limitations (line 705) | **CITED** | "18 human-only, 8 mouse-only, 43 dual-organism … centroid cosine 0.917 vs 0.919, p=0.14 … FD 1.052 vs 1.028, p=0.85 … DivR p=0.18 … p=0.43 … p=0.87." All six headline numbers landed. |
| 3 | `a3_abundance_fidelity` (Lane A3) | R2.7 | Results §Per-Type (line 474) | **CITED** | "Operational definition 'broad intrinsic within-type heterogeneity', `real_intra_cos` below 25th percentile … Spearman ρ=0.57, p=4.9×10⁻⁷ … ρ=−0.77, p=3×10⁻¹³ … bottom-quartile centroid cosine 0.907 vs 0.896, p=0.02." |
| 4 | `a4_hvg_variance` (Lane A4) | R2.8 | Results §Gene-Level Fidelity (line 486) | **CITED** | "1,790 genes … 85% inside [0.5, 2.0] band … median SD ratio 1.33 … pooled r_var=0.988 … within-type median r_var=+0.85 … 100% positive … 75% in tolerance … 8.7% >2× deficit … cross-dataset median ratio 0.01–0.02." |
| 5 | `a5_rare_failure_mechanism` (Lane A5) | R2.11 | Results §Rare Cell Augmentation (line 619); Discussion (lines 688, 707) | **CITED** | "18 lowest-abundance types … latents 30–66% of real (median 0.50×) … expression 47–190% (median 1.07×) … Cycling within 5–15% but intra-cosine 4.6× higher." Also called out as "A5" in Discussion. |
| 6 | `a6_annotation_confidence` (Lane A6, **bonus**) | n/a (internal QC for marker-curation audit) | *none* | **GAP** | No mention of overall median Jaccard 0.50, IQR [0.275, 0.80], frac<0.30=0.293, frac<0.15=0.103, or the bottom-confidence types (Unknown, Collecting duct, Granule cells, etc.). The manuscript does cite PanglaoDB/CellMarker Jaccard 0.39–0.45 (line 249) but at the corpus-level only; the per-type confidence breakdown is absent. |
| 7 | `b1_gaussian_rvar_baseline` (Lane B1) | R2.9 | Results §In-Distribution Expression-Level Validation — "Baselines for gene-wise variance recovery" (line 572) | **CITED** | "+0.201 mean across-type r_var … pooled-Gaussian floor +0.014 (60.9%) … type-aware oracle +0.813 … within-type pooled median 0.94 / 1.13 / 1.79." Both latent and expression scales present. |
| 8 | `b2_zca_ablation` (Lane B2) + duplicate `experiments/zca_ablation/` | R3.4 | Results §CLOP Ablation Study (line 598) + Table~\ref{tab:zca-ablation} | **CITED (partial)** | Table 5 reports ZCA vs no-ZCA at 76.17%/75.42% proto-acc (Δ=−0.75) and KNN 36.9/36.5 (Δ=−0.4). **But the rebuttal-ready numbers (quality 0.966 / 0.960 / 0.956; cos_sim 0.851 / 0.815 / 0.809; 3-condition ablation `whiten/center_norm/none`) are NOT in the manuscript.** The center_norm arm is completely missing. See GAP #A below. |
| 9 | `b3_mixing_sweep` (Lane B3) | R3.3 | Results §Rare Cell Augmentation — "Systematic augmentation-strategy sweep" (line 621) | **CITED** | "2×6×4 sweep … Megakaryocytes 0.917, Ameloblasts 0.937 … random oversampling →0.807/0.859 at 10× … SMOTE →0.833/0.879 … CLOP-DiT flat 0.917/0.929." |
| 10 | `b3_mixing_sweep/forced_scarcity` (B3 extension) | R3.3 addendum | *none* | **GAP** | The forced-scarcity rescue (artificially reduce rare-class training to 30 cells, baseline F1 drops to ~0.50, oversampling recovers to 0.783/0.866 at 10×, CLOP-DiT to 0.719/0.670, hybrid to 0.772/0.835) is absent. Nothing in the manuscript rebuts the "the null result is ceiling-driven" reading. |
| 11 | `b4_clop_to_full_bridge` + `b4_clop_bridge/` (Lane B4) | R2.10 | Results §CLOP Ablation — "Stage-1 to full-pipeline transfer (partial-reversal caveat)" (line 598); Discussion (§strict-OOD, line 707: "B4") | **CITED (partial)** | The partial-reversal narrative is present and the three-variant retraining is described, **but the concrete headline numbers — `no_cohesion` FD 0.2303 vs 0.175 baseline (+31%), coverage 0.079 vs 0.144 (−45%), centroid cosine 0.854 vs 0.929 (−8.1%), `no_cell_noise` FD 0.145 — are NOT reported.** See GAP #B below. The Phase-1 projected-text structural-similarity diagnostic (Pearson r=0.04–0.06 production vs ablation) is also absent. |
| 12 | `lane_c_data` zero-shot (Lane C) | R3.1, R3.2 (cell-level) | Results §Strict-OOD Generalization (lines 650–681); Discussion §Strict out-of-distribution generalization (line 707) | **CITED** | Entirely new subsection. Kidney 0.460, cerebellum 0.029, testis 0.562, overall 0.350, adrenal cortex type I 0.875, Purkinje 0.00, granule 0.045, Leydig collapse. All Lane-C numeric deliverables landed. |
| 13 | `lane_d_encoder` / `encoder_comparison` (Lane D / bonus) | R2.2/R2.3/R2.5 + R2.11 mechanism | Discussion §Where the bottleneck lives (line 709) | **CITED** | "PCA median 0.90 … scGPT-human 0.14 (~7×) … scGPT-pancancer 0.07 (~15×) … tightness ratio ~0.65." All three encoder headline numbers landed. |
| 14 | `cap_increase` (bonus, R2.11 mechanism) | R2.11 addendum | *none* | **GAP** | The seed-aligned cap3k→cap10k null (3.12× more input cells; median per-dataset variance ratio 1.008, IQR [0.984, 1.035]; within-cluster median 0.976 over 475 pairs) is completely absent. This is the single most material "we tested the obvious 'add more data' fix and it doesn't help" evidence the revision produced, and it currently lives only in SYNTHESIS files. See GAP #C below. |
| 15 | `encoder_bottleneck` (bonus, R2.11 mechanism trio) | R2.11 addendum | *partial* — one leg cited via Lane D | **GAP** | The HVG-count ablation (`n_top_genes` 500→8000, within-cluster variance 11.18→8.49→8.78, saturates at ≥2000) and the raw-vs-latent compression diagnostic (within-L2 latent/raw ratio 0.168, IQR [0.132, 0.214]; between-type 0.274) are missing. Only the encoder-comparison leg is cited (in §"Where the bottleneck lives"). The "three complementary experiments converge" Discussion beat proposed in SYNTHESIS.md is therefore only one-third complete in the current manuscript. See GAP #D below. |

**Raw tally (pre-dedup):** 15 rows, counting `b2_zca_ablation` + `zca_ablation/` as one experiment and `b4_clop_to_full_bridge` + `b4_clop_bridge/` as one experiment.

---

## 2. Per-GAP patch plan

The six GAPs below range from "single sentence" fixes (A6, ZCA center_norm)
to a full paragraph (cap-increase + HVG + raw-vs-latent). Suggested
paste-in text is given verbatim; section labels refer to existing
`\label{...}` anchors in `clop_dit_genes.tex`.

### GAP #A6 — Annotation-confidence audit (Lane A6)

**Where to drop it:** §Datasets and preprocessing, immediately after the
Jaccard-0.39–0.45 sentence on line 249 (i.e., before the "The exact caption
template …" sentence). This keeps the marker-curation audit chain in one
place rather than splitting it across Methods and Supplementary.

**Reviewer comment answered:** Strengthens the R2.1 scope/limitations
framing and the marker-circularity defense (R2.4-adjacent) without
inventing a new reviewer thread.

**Paste-ready sentence:**

> As a second check on caption quality, we audited per-cluster annotation
> confidence by computing the Jaccard overlap between each sub-cluster's
> data-derived marker set and the external PanglaoDB / CellMarker~2.0
> reference panels. Across the 1{,}073 sub-clusters (spanning 89 raw
> cell-type labels before deduplication), median confidence was 0.50 with
> IQR [0.28, 0.80]; 29.3\% of clusters fell below a 0.30 threshold and
> 10.3\% below 0.15. The bottom-confidence bucket is dominated by
> ``Unknown'' sub-clusters and by rare-tissue types (collecting duct,
> alveolar macrophages, Sertoli, mesothelial, adipocytes) where the
> external references are themselves thin, whereas high-confidence types
> (Somatotrophs, Mast cells, MHC-II-high APCs, Cycling cells) exceed
> median Jaccard 0.83. We retain all clusters above the 0.15 floor to
> preserve evaluation breadth; Supplementary Table~S\textasteriskcentered{}
> reports the full per-type confidence distribution.

---

### GAP #B (= ZCA center_norm arm) — Lane B2 missing middle condition

**Where to drop it:** Results §CLOP Ablation Study, inside
Table~\ref{tab:zca-ablation} (line 602–613). Replace the 2-row table
with a 3-row version that preserves the existing ZCA / no-ZCA rows and
adds the `center_norm` row. Also rewrite the surrounding sentence on
line 598 to name the three conditions.

**Reviewer comment answered:** R3.4 (ZCA claim is anecdotal; formal
ablation required). The current manuscript shows a 2-way diff; the
rebuttal synthesis pre-registers a 3-way. Reviewers who read the
response letter will expect the `center_norm` column to appear.

**Paste-ready table replacement (`tab:zca-ablation`):**

| Configuration | CLOP Proto Acc (%) | CLOP Quality | Pos-pair cos | DiT KNN Acc (%) |
|---|---:|---:|---:|---:|
| With ZCA (production) | 99.91 | 0.9656 | 0.851 | 36.9 |
| Center + L2 only | 99.95 | 0.9600 | 0.815 | — |
| Without preprocessing | 99.82 | 0.9562 | 0.809 | 36.5 |

**Paste-ready sentence (replaces the last sentence of the ablation
paragraph on line 598):**

> The ZCA ablation (Table~\ref{tab:zca-ablation}) now spans three matched
> CLOP training runs — full ZCA, mean-centering + L2 normalization only,
> and no preprocessing at all. Prototype accuracy remains near-perfect
> ($\geq 99.82\%$) in every condition; composite alignment quality
> drops by only $1.0\%$ (0.966 $\to$ 0.956) across the full ablation.
> ZCA's principal measurable contribution is a $+5\%$ improvement in
> positive-pair cosine similarity (0.851 vs.\ 0.809), reflecting
> decorrelation of the 512-d scGPT dimensions rather than a structural
> alignment gain. ZCA is therefore retained as a useful default but
> should not be interpreted as a critical architectural component.

---

### GAP #B4 — Lane B4 bridge headline numbers

**Where to drop it:** Results §CLOP Ablation Study, inside the
"Stage-1 to full-pipeline transfer (partial-reversal caveat)"
subparagraph on line 598. Either append 2 sentences in-line, or add
a new `tab:b4-bridge` table.

**Reviewer comment answered:** R2.10 (CLOP ablations run under a
different configuration from the production model). The current
manuscript narrates the partial-reversal story but hides the numbers,
which leaves reviewers no way to verify the claim independently.

**Paste-ready insert (append to the "partial-reversal" sentence):**

> Concretely, retraining the full DiT on top of three CLOP variants
> (ablation-baseline, $-$cohesion, $-$cell-noise) and regenerating 6{,}900
> embeddings per variant reveals that the strongest Stage-1 performer
> becomes the weakest end-to-end generator: $-$cohesion reaches
> prototype accuracy 0.864 (vs.\ 0.762 baseline, $+10.2$\,pp) but
> degrades downstream Fr\'echet distance to 0.230 (vs.\ 0.175, $+31\%$),
> coverage to 0.079 (vs.\ 0.144, $-45\%$), and centroid cosine to 0.854
> (vs.\ 0.929, $-8.1$\,pp). The $-$cell-noise variant transfers only
> partially (FD improves to 0.145, but centroid cosine drops marginally
> to 0.923). A Stage-1 projected-text diagnostic further confirms that
> production and ablation CLOPs learn structurally unrelated geometries
> (inter-type-similarity matrix Pearson $r = 0.04$--$0.06$), so CLOP
> ablation rankings should be read as a characterisation of the
> aligner's loss landscape rather than as a predictor of end-to-end
> generation quality.

---

### GAP #3.3 (forced-scarcity) — Lane B3 extension

**Where to drop it:** Results §Rare Cell Augmentation, immediately
after the "Systematic augmentation-strategy sweep" paragraph
(after line 621, before "Conditioning Field Ablation" on line 623).

**Reviewer comment answered:** R3.3 addendum. The existing B3 paragraph
concludes with a null result and an assertion that future work should
target generator-side objectives. Without forced-scarcity, a reviewer
can reasonably push back with "your null result might be ceiling-driven,
not a principled failure." The forced-scarcity rescue neutralises that
read: at genuine scarcity, augmentation *does* work — just not CLOP-DiT
alone.

**Paste-ready paragraph:**

> \textbf{Forced-scarcity rescue.} To verify that the B3 null result
> is not merely a ceiling artefact (baseline F1 $\geq 0.92$ leaves
> little headroom for any augmentation to exploit), we repeated the
> sweep after artificially reducing each rare class to 30 training
> cells, which drops baseline F1 to $\approx 0.50$ on both
> Megakaryocytes and Ameloblasts. Under genuine scarcity, classical
> augmentation recovers substantially: random oversampling reaches
> 0.783 / 0.866 at $10\times$, SMOTE reaches 0.742 / 0.823, CLOP-DiT-only
> reaches 0.719 / 0.670, and the hybrid CLOP-DiT + oversampling strategy
> reaches 0.772 / 0.835. Oversampling is strongest at the embedding
> level because duplicating real points reinforces the correct centroid
> for a logistic classifier; CLOP-DiT-only is weakest because generated
> latents introduce additional within-class variance that the
> embedding-space classifier treats as noise. This rescue experiment
> therefore confirms that the natural-scarcity null is ceiling-driven
> and that the future work target is not more aggressive mixing but a
> generator-side heterogeneity objective that preserves within-type
> structure at gene-expression level, where CLOP-DiT's synthetic
> diversity is expected to dominate simple duplication.

---

### GAP #C (cap-increase) — Encoder-bottleneck cap-increase leg

**Where to drop it:** Discussion §"Where the bottleneck lives
(encoder-side localization)", line 709. The section currently cites only
the `encoder_comparison` scGPT-vs-PCA leg; it should open with the
three-experiment convergence story the SYNTHESIS file already drafts.

**Reviewer comment answered:** R2.11 addendum and R3.3 secondary
response. This is the single most persuasive "we tested the obvious
'more data' fix" evidence in the revision branch; leaving it out
weakens the entire Discussion story.

**Paste-ready insert (prepend to the start of the "Where the bottleneck
lives" paragraph, before "A matched-dimensionality encoder comparison
…"):**

> Three independent input-side levers all fail to move within-type
> latent variance in the scGPT embedding space. (i) Relaxing the
> per-dataset cell cap from 3{,}000 to 10{,}000 cells ($3.12\times$
> more input cells, with a fixed \texttt{sc.pp.subsample} seed on both
> sides) leaves the median per-dataset total-variance ratio at 1.008
> (IQR [0.984, 1.035]) and the median within-cluster variance ratio at
> 0.976 across 475 matched (dataset, cell-type) pairs. (ii) Sweeping
> the HVG count from 500 to 8{,}000 genes produces less than a
> $5\%$ change in latent within-cluster variance once the HVG count
> exceeds 2{,}000; the curve saturates rather than scaling with gene
> budget. (iii) A direct raw-vs-latent compression diagnostic shows
> that the encoder compresses within-type L2 distance by a median
> factor of $6\times$ (ratio 0.168, IQR [0.132, 0.214]) while
> shrinking between-type centroid distance only $3.7\times$ (ratio
> 0.274), so clusters become more separable in latent space but the
> within-type fine structure R2.11 flags is disproportionately lost.
> These three results establish that within-type heterogeneity is
> bounded by the frozen scGPT representation rather than by data
> scaling or preprocessing choices.

(The existing paragraph — "A matched-dimensionality encoder comparison
on eight representative datasets …" — then follows unchanged and acts
as the architecture-specificity confirmation leg.)

---

### GAP #D (HVG + raw-vs-latent) — same patch as #C

Legs (ii) and (iii) of the paragraph above cover these two experiments,
so no separate patch is needed. Listing them as a distinct GAP keeps
the per-experiment tally honest.

---

## 3. Final tally

| Count | Quantity |
|---|---:|
| Experiments audited (post-dedup) | **15** |
| **CITED** (fully) | **9** |
| **CITED (partial)** — narrative present, numbers missing | **2** (B2 ZCA centre\_norm arm; B4 bridge headlines) |
| **GAP** (experiment not cited in manuscript) | **4** (A6 annotation confidence; B3 forced-scarcity; cap-increase; HVG + raw-vs-latent) |
| Discrete patches proposed | **5** (one per paste-ready block: A6, ZCA-3way, B4-numbers, forced-scarcity, encoder-bottleneck-trio which covers both cap-increase and HVG+raw-vs-latent) |

**Priority ranking for this revision round** (most-to-least important):

1. **Encoder-bottleneck trio (cap-increase + HVG + raw-vs-latent).**
   This is the strongest new mechanism claim in the whole revision and
   currently sits in SYNTHESIS files only. High-impact, ~1 paragraph fix.
2. **B4 bridge headline numbers.** Without the FD / coverage / centroid-
   cosine deltas the partial-reversal claim is unfalsifiable. Medium-
   effort (2 sentences or 1 mini-table).
3. **B3 forced-scarcity.** Directly blunts the likely reviewer pushback
   that the B3 null is ceiling-driven. Low-effort (one paragraph).
4. **ZCA centre\_norm arm (B2).** Small but important for rebuttal
   consistency: the response letter promises a 3-way ablation. Low-effort
   (one table row + one sentence rewrite).
5. **A6 annotation confidence.** Bonus experiment; adds breadth to the
   marker-curation audit story. Lowest priority but cheapest patch.

---

## Appendix — audit protocol

- Manuscript read: `revision/manuscripts/v2_revision/clop_dit_genes.tex`
  lines 1–724 (preamble + Abstract through Conclusions). Appendix /
  Supplementary LaTeX (lines 725–1318) is out of scope per the task
  brief; all five Lane-C limitations paragraphs, multi-seed table, and
  hyperparameter table were already read as part of the pass.
- Experiment artefacts read: all `results.md`, `SYNTHESIS.md`, and
  `*_preview.txt` files under `revision/experiments/` for the 15
  experiments listed.
- Numbers cross-referenced: 49.2, +0.201, +0.813, +0.014, +0.183, +0.85,
  +0.988, 0.917, 0.937, 0.875, 0.460, 0.562, 0.350, 0.168, 0.899, 0.144,
  0.066, 0.50 (median latent ratio), 1.008, 0.976, 1.326, 76.17, 75.42,
  86.41, 0.9656, 0.9562, 0.8512, ρ=-0.77, ρ=+0.57, p=0.14, p=0.85, p=0.43,
  p=0.87, Mann-Whitney p=0.02. Every number that DID land in the manuscript
  is logged in the "Evidence" column of §1; every number that did NOT
  land is the justification for a GAP row.
- Cross-reference corpus: `REVIEWER_RESPONSE_SYNTHESIS.md`,
  `REVISION_LOG.md`, `reviewer_response_draft.md`.
