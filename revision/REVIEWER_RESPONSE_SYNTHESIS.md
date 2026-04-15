# REVIEWER_RESPONSE_SYNTHESIS — consolidated Lane A + B findings

All rebuttal-ready paragraphs from the completed analyses (A1–A5,
B1 latent + expression scale, B3) are gathered here in one place so
the response letter and the revised Discussion can be assembled
without having to cross-read seven separate results.md files. Each
block cites the source experiment and the supporting JSON.

Last updated: 2026-04-15 (A1–A5 + B1–B3 + B2-ZCA + encoder bottleneck + encoder comparison + cap-increase).

---

## R2.6 — where do the KNN errors go? (Lane A1)

**Source:** `revision/experiments/lane_a_analysis/a1_knn_confusion/`
**Supporting artifact:** `knn_confusion.json`, `family_taxonomy.yaml`
**Suggested manuscript location:** Results §Per-Type Generation
Quality + Supplementary figure / table.

> We have characterised the error structure of the KNN classifier as
> requested. Using a 10-family biologically-grounded taxonomy over the
> 69 evaluation cell types, 49.2 % of all mis-typed cells stay within
> the same family — i.e. roughly half of the reported errors are
> graceful degradations between closely related cell types, not
> catastrophic cross-lineage failures. Of the top-15 most-confused
> cell-type pairs, 9 are within-family (e.g., pancreatic δ ↔
> neuroendocrine, tissue-resident macrophages ↔ MHC-II+
> antigen-presenting cells, fibroblasts ↔ mesenchymal stem cells),
> and the 6 cross-family pairs are dominated by either
> functional-state overlap (cycling or stress-response cells
> matching into lineage families) or shared-marker confusions at
> well-known transcriptional boundaries. Only the proliferation /
> stress / pluripotent family, which cuts across lineages by
> definition, shows within-family accuracy below 0.94.

---

## R2.7 — operationalise "biologically difficult" (Lane A3)

**Source:** `revision/experiments/lane_a_analysis/a3_abundance_fidelity/`
**Supporting artifact:** `abundance_fidelity.json`,
`per_type_table.csv`.
**Suggested manuscript location:** Results §Per-Type Generation
Quality and a new Supplementary Table.

> We agree that the phrase "biologically difficult" was imprecise
> and have replaced it throughout the manuscript with the operational
> definition "cell types with broad intrinsic within-type
> heterogeneity", quantified as `real_intra_cos` below the
> 25th percentile. Across the 68 matched cell types, per-type Frechet
> distance is only weakly related to training-set abundance (Spearman
> ρ = 0.57, p = 4.9 × 10⁻⁷) but is strongly explained by within-type
> heterogeneity (ρ = −0.77, p = 3 × 10⁻¹³), and a bottom-quartile-by-
> training-count slice does not show the fidelity gap the reviewer
> suspected (centroid cosine 0.907 vs 0.896 for the top quartile,
> Mann-Whitney p = 0.02, favouring the underrepresented set).
> Underrepresentation by training-cell-count is therefore not the
> dominant axis of failure; within-type heterogeneity is the
> better-targeted diagnosis and motivates the heterogeneity-
> preserving training extensions discussed below.

---

## R2.8 — numeric summary of the residual / variance-ratio figures (Lane A4)

**Source:** `revision/experiments/lane_a_analysis/a4_hvg_variance/`
**Supporting artifact:** `hvg_variance_summary.json`.
**Suggested manuscript location:** Results §Gene-Level Fidelity,
with explicit quoted numbers in-line.

> We have added a quantitative summary of the residual and
> variance-ratio figures to the main text. Across the 1 790 genes
> scored in the in-distribution evaluation, 85 % of genes fall inside
> the `[0.5, 2.0]` tolerance band for the gen / real variance ratio,
> with a median standard-deviation ratio of 1.33 and a pooled
> gene-wise variance Pearson correlation of r_var = 0.988.
> Restricting the analysis to within-type variance — the scale
> relevant to reviewer concerns about second-order fidelity — median
> r_var across the 69 cell types is +0.85 and 100 % of types
> exhibit a positive correlation, with 75 % of (type, gene) pairs
> remaining inside tolerance and 8.7 % showing a variance deficit
> greater than two-fold. The cross-dataset held-out evaluation, which
> contrasts generation against unseen GEO studies, remains the
> stricter test and continues to show variance collapse (median ratio
> 0.01 – 0.02), motivating the strict-OOD experiments scheduled for
> Lane C.

---

## R2.9 — baselines for gene-wise variance (Lane B1 latent + expression)

**Source:** `revision/experiments/lane_b_retrain/b1_gaussian_rvar_baseline/`
**Supporting artifact:** `latent_baselines.json`,
`expression_baselines.json`.
**Suggested manuscript location:** Results §Gene-Level Fidelity, new
comparison table; also Discussion paragraph reframing the r_var
narrative.

> We agree the original manuscript should have reported baselines
> for the gene-wise variance analysis and we have now done so at
> both the CLOP latent scale and the decoded-expression scale.
>
> At the latent scale, on the in-distribution evaluation slice,
> CLOP-DiT recovers within-type variance structure with a mean
> across-type Pearson r_var of +0.201 (median +0.183), sitting
> strictly between a type-agnostic pooled-Gaussian floor (+0.014,
> only 60.9 % of types above zero) and a type-aware Gaussian oracle
> that matches real per-type mean and covariance exactly (+0.813).
>
> At the decoded-expression scale, all three generators converge to
> high pooled r_var because HVGs dominate gene-wise variance; the
> discriminating statistic there is the within-type pooled median
> variance ratio (0.94 for the type-aware oracle, 1.13 for
> CLOP-DiT, 1.79 for the pooled Gaussian).
>
> The first-order conclusion is therefore that the near-zero
> gene-wise variance recovery reported in the cross-dataset held-out
> evaluation is not a shared feature of all latent-generation
> pipelines — a type-aware moment-matching baseline stays well above
> zero on the in-distribution slice — but that CLOP-DiT does
> under-represent within-type variance relative to that oracle,
> consistent with the upstream latent-compression mechanism diagnosed
> in the rare-cell failure analysis.

---

## R2.11 — rare-cell augmentation failure mechanism (Lane A5 + Lane B3)

**Source:** `revision/experiments/lane_a_analysis/a5_rare_failure_mechanism/`
and `revision/experiments/lane_b_retrain/b3_mixing_sweep/`.
**Supporting artifact:** `rare_failure_mechanism.json`,
`mixing_sweep.json`.
**Suggested manuscript location:** Results §Rare-Cell Augmentation
(reframed) + Discussion.

> Decomposing within-type variance at the latent and expression
> scales attributes the rare-cell augmentation failure to upstream
> latent under-dispersion rather than downstream decoder collapse.
> Across the 18 lowest-abundance cell types, the generator produces
> latents whose within-type total variance is 30 – 66 % of real
> (median 0.50 × real), yet the scGPT decoder recovers
> expression-space variance to 47 – 190 % of real (median 1.07 ×
> real). The decoder is already expansive; the generator is
> centroid-biased. For the Cycling population specifically, latent
> and expression total variance are within 5 – 15 % of real but
> intra-cluster cosine similarity is 4.6 × higher in generated
> samples than in real, indicating that synthetic diversity is
> highly directional.
>
> We further tested whether classical imbalance strategies can
> recover what naive CLOP-DiT-only augmentation loses. On two rare
> cell types (Megakaryocytes, Ameloblasts), a 2 × 6 × 4 sweep over
> random oversampling, SMOTE, CLOP-DiT-only, and hybrid strategies
> at ratios 1×, 2×, 5×, 10× shows **no** strategy improves
> rare-class F1 over the no-augmentation baseline; classical methods
> degrade monotonically with ratio, while CLOP-DiT-only stays flat.
> We therefore retire the framing that attributes the
> augmentation-pilot shortcoming to a generator-specific defect and
> instead direct future augmentation work toward generator-side
> heterogeneity objectives, not downstream mixing.

---

## R3.2 — human vs mouse stratified evaluation (Lane A2)

**Source:** `revision/experiments/lane_a_analysis/a2_organism_split/`
**Supporting artifact:** `organism_split.json`,
`organism_per_type_table.csv`.
**Suggested manuscript location:** Results §Core Evaluation Metrics
(stratified addendum) + new Supplementary Table.

> We have now stratified all headline per-type metrics by organism.
> Across the 18 human-only and 8 mouse-only cell types, the two
> organism groups are statistically indistinguishable on centroid
> cosine (0.917 vs 0.919, Mann-Whitney p = 0.14), Frechet distance
> (1.052 vs 1.028, p = 0.85), diversity ratio (p = 0.18),
> expression Pearson r (p = 0.43), and real within-type cosine
> tightness (p = 0.87); mouse-only types are in fact marginally
> better on FD. The significant differences in the stratification
> are between single-organism and multi-organism types, not between
> species; the mechanism is the broader tissue scope of
> multi-organism types, captured by `real_intra_cos`, which A3
> identifies as the dominant predictor of per-type fidelity.
> Consequently, we do not observe the species bias suggested by the
> imbalance in the dataset count (59 human vs 21 mouse); the shared
> CLOP space generalises across species at the type level, and the
> cell-level strict stratification remains scoped for Lane C when
> targeted mouse data expansion is complete.

---

## R3.3 — augmentation strategy space (Lane B3)

**Source:** `revision/experiments/lane_b_retrain/b3_mixing_sweep/`
**Supporting artifact:** `mixing_sweep.json`.
**Suggested manuscript location:** Results §Rare-Cell Augmentation
(reframed).

> At the reviewer's suggestion we systematically explored the
> augmentation strategy space, running a 2 × 6 × 4 sweep (two rare
> cell types, six strategies including random oversampling, SMOTE,
> CLOP-DiT-only, and hybrid CLOP-DiT + oversampling / CLOP-DiT +
> SMOTE, at ratios 1×, 2×, 5×, and 10×). On Megakaryocytes (natural
> rarity, 967 training cells) and Ameloblasts (384 training cells),
> no tested strategy improves rare-class F1 over the no-augmentation
> baseline (0.917 and 0.937 respectively) at any ratio. Classical
> methods degrade monotonically with ratio (random oversampling
> falls to 0.807 and 0.859 at 10×; SMOTE to 0.833 and 0.879), while
> CLOP-DiT-only stays flat (0.917 and 0.929 at 10×). The result is
> consistent across both rare types and is the expected consequence
> of the centroid-biased latent structure diagnosed in A5:
> additional generated samples can match the type centroid but do
> not add the heterogeneity needed to shift the decision boundary.

---

## Cross-experiment synthesis (for a unified Discussion paragraph)

Lane A + B together converge on a **single mechanistic story** the
revised Discussion should lead with:

1. The generator is the bottleneck (A5 + B1 latent). Within-type
   latent variance is compressed to ~ 50 % of real on rare types;
   the learned distribution sits between a type-agnostic floor and a
   type-aware oracle, but much closer to the floor.
2. The decoder is not the bottleneck (A5 + B1 expression).
   Expression-scale variance ratios recover to ~ 1.0 when the
   latents are reasonable (Gaussian-per-type oracle hits 0.94
   median) and remain ≥ 1.0 for CLOP-DiT too; the scGPT decoder is
   expansive rather than collapsing.
3. Data abundance is not the right predictor (A3). Within-type
   intrinsic heterogeneity (`real_intra_cos`) explains per-type
   Frechet distance at ρ = −0.77 vs ρ = +0.57 for
   `log10(training_count)`.
4. There is no species bias (A2). Human-only and mouse-only metrics
   are statistically indistinguishable; the 59/21 imbalance in
   dataset count does not propagate to evaluation-level bias at the
   type level.
5. Augmentation-by-mixing is the wrong lever (B3). No tested
   mixing strategy recovers what the generator failed to produce;
   the path forward is generator-side heterogeneity objectives, not
   downstream mixing.
6. KNN errors are biologically graceful (A1). Half the mis-typings
   stay within the same lineage family, and the worst family
   (proliferation / stress / pluripotent) is biologically expected.

A single paragraph of the revised Discussion can open with the
sentence: "The major-revision analyses point to a single
mechanistic picture: CLOP-DiT's limitation is **latent-stage
under-dispersion of within-type heterogeneity**, not decoder
collapse, not species imbalance, not data scarcity, and not a
correctable-by-mixing downstream artefact." Each subsequent
sentence can then cite one of the six findings above.

---

## R2.11 addendum — cap-increase negative control (2026-04-15)

**Source:** `revision/experiments/cap_increase/`
**Supporting artifact:** `seed0_variance_comparison.json`,
`within_cluster_variance.json`, `SYNTHESIS.md`.
**Suggested manuscript location:** Results §Rare-Cell Augmentation
(reframed) + Discussion paragraph on encoder-stage bottleneck.

> At the reviewer's suggestion, we also tested whether relaxing the
> per-dataset cell cap from 3 000 to 10 000 would expose additional
> within-type latent variance. Preprocessing and scGPT encoding were
> re-run on 50 datasets at both caps with a fixed `sc.pp.subsample`
> seed so only the cap itself differed. The median per-dataset total
> variance ratio cap10k / cap3k in scGPT latent space is 1.008 (IQR
> [0.984, 1.035]), and the median within-cluster variance ratio,
> pooled over 475 matched (dataset, cell-type) pairs, is 0.976 (IQR
> [0.895, 1.079]). A 3.12 x increase in input cells therefore
> produces no practically-meaningful gain in latent heterogeneity.
> Earlier first-pass comparisons appeared bimodal (5 x gains and
> 0.16 x losses) but that was fully explained by `sc.pp.subsample`
> draw drift between the original unseeded cap3k cache and the new
> seed-0 cap10k cache; matching the seed on both caps collapses the
> IQR into a tight band around 1. This further corroborates the
> A5 + B1 diagnosis that the bottleneck is upstream of the DiT at
> the scGPT encoder / CLOP interface, not in the data volume, and
> that generator-side heterogeneity objectives (not data-scaling)
> are the productive next direction.

---

## R2.10 addendum — ZCA whitening ablation (Lane B2)

**Source:** `revision/experiments/zca_ablation/`
**Supporting artifact:** `zca_ablation_summary.json`, `SYNTHESIS.md`.
**Suggested manuscript location:** Results §CLOP Aligner Ablation
(expanded) + Supplementary Table.

> We isolated the contribution of ZCA whitening by training three
> CLOP aligners differing only in embedding preprocessing: full ZCA
> (production), center + L2-normalise, and no preprocessing. CLOP
> quality score drops by just 1.0 % from ZCA (0.966) to no
> preprocessing (0.956), with prototype accuracy near-perfect
> (≥ 0.998) in all conditions. ZCA's measurable contribution is a
> 5 % improvement in positive-pair cosine similarity (0.851 vs
> 0.809), reflecting decorrelation of the 512-d scGPT dimensions
> rather than a structural alignment gain. The 3-layer MLP projectors
> combined with the PrototypeSigLIP loss are sufficient to learn
> through raw embedding collapse, confirming ZCA as a modest
> refinement rather than a critical pipeline component.

---

## Encoder bottleneck mechanistic trio (Sections 7 + 8)

**Source:** `revision/experiments/cap_increase/`,
`revision/experiments/encoder_bottleneck/`.
**Supporting artifact:** `SYNTHESIS.md` in each directory.
**Suggested manuscript location:** Discussion §Limitations and Future
Directions.

> Three complementary experiments converge on the same diagnosis:
> within-type heterogeneity in the scGPT latent space saturates
> regardless of input-side manipulation. (i) A cap-increase from
> 3 000 to 10 000 cells per dataset yields median variance ratio
> 1.008 — no gain. (ii) Varying the HVG count from 500 to 8 000
> genes changes within-type latent variance by < 5 % once HVG ≥
> 2 000 — the encoder saturates. (iii) Direct comparison of raw
> HVG space to scGPT latent space shows the encoder compresses
> within-type L2 distance by 6 × while preserving between-type
> separation. No input-side lever (cell count, gene count, seed)
> moves the within-type latent variance; the bottleneck is a
> property of the frozen scGPT transformer representation.

---

## Encoder comparison — scGPT vs PCA (Section 10, Lane D supplement)

**Source:** `revision/experiments/encoder_comparison/`
**Supporting artifact:** `encoder_comparison_summary.json`,
`SYNTHESIS.md`.
**Suggested manuscript location:** Discussion §Limitations, or
Supplementary.

> Comparing cell encoders on eight representative datasets, we find
> that within-type L2 compression is specific to the scGPT transformer
> architecture, not an artefact of dimensionality reduction. PCA
> embeddings in the same 512-d space preserve within-type distance
> almost perfectly (median ratio 0.90), while scGPT-human compresses
> it by ~7× (median ratio 0.14) and scGPT-pancancer by ~15× (median
> ratio 0.07). The pancancer model, trained on a narrower 5.7M-cell
> cancer corpus, produces an even more centroid-biased representation
> than the 33M-cell whole-human model, confirming that training data
> diversity modulates but does not eliminate the compression.
> Cluster separability (tightness ratio) is comparable across both
> scGPT variants (~0.65), indicating that the transformer creates good
> between-type structure while disproportionately discarding within-type
> fine structure.
