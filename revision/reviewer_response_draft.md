# Draft Response to Reviewers #2 and #3

Prepared on 2026-04-14.

This draft is written as a formal rebuttal in English so it can be adapted directly into a revision response letter. The reviewer comments below are preserved in full, followed by suggested responses, planned manuscript revisions, and a clear marker indicating whether additional analysis or experiments are recommended.

## Legend

| Status | Meaning |
| --- | --- |
| **Text revision only** | Can be addressed by clarifying scope, interpretation, wording, or figure presentation. |
| **Additional analysis recommended** | Can likely be addressed by stratifying or re-analyzing existing outputs without retraining the full pipeline. |
| **Additional experiment recommended** | Would benefit from new runs, new evaluation splits, or substantial new computation. |

## Overall Response Framing

We thank both reviewers for their thoughtful and constructive assessments. We especially appreciate the shared recognition that CLOP-DiT is best interpreted as an early but meaningful demonstration of structured-metadata-conditioned single-cell generation rather than a finished high-fidelity simulator. We agree that the central scientific gap is the preservation of within-type biological heterogeneity and second-order structure. In the revision, we will sharpen this framing throughout the manuscript.

More specifically, we will clarify that the main contribution of the current work is **methodological feasibility and conditional controllability**: the model learns a usable text-cell conditioning space, generates above-chance type-specific latents, and exposes an interpretable fidelity-diversity trade-off. At the same time, we will make even more explicit that **coarse cell identity is better preserved than within-type heterogeneity**, that the held-out dataset split is still an interpolation setting, and that downstream biological realism remains incomplete.

We will also strengthen the discussion of **current application scenarios** that are appropriate at this stage, including: (i) structured prompt-to-latent controllability studies, (ii) coarse in silico exploration of cell-state tendencies within the observed training distribution, and (iii) a modular testbed for improving metadata-conditioned single-cell generation. We will avoid presenting the model as a drop-in simulator for augmentation or biological decision-making.

---

## Reviewer #2

### Comment 2.1

> The abstract lacks a statement of research background and fails to explain the specific challenges currently faced in the field of single-cell generation to support the necessity of this study.

**Response.**  
We agree. The current abstract states the task and the proposed framework, but it does not sufficiently foreground the scientific gap that motivates the work. In the revision, we will add brief context explaining that most existing single-cell generative models condition on categorical labels, perturbation labels, or batch covariates rather than on richer structured biological descriptions, and that the central challenge is not only generation itself but preserving biologically meaningful within-type structure under such conditioning. This will make the necessity of the study clearer from the outset.

**Planned manuscript revision.**

- Add 1--2 sentences to the abstract stating the current limitation of label-based conditioning in single-cell generation.
- Explicitly state that the main challenge is balancing conditional controllability with preservation of within-type heterogeneity.

**Status.** **Text revision only**

---

### Comment 2.2

> The author proposes using a template with five fields: cell type, tissue, species, marker genes, and disease background. Although these fields cover the main biological dimensions, there is a lack of justification: why choose these five? Has an ablation experiment been conducted to prove that the combination of these five fields is optimal?

**Response.**  
We appreciate this point and agree that the manuscript should better justify the five-field template. Our intent was not to claim that this combination is mathematically or biologically optimal. Rather, these five fields were chosen because they form a practical, interpretable, and consistently extractable scaffold across the 80-GEO-dataset corpus: **cell type** captures identity, **tissue** captures anatomical context, **organism** captures species background, **marker genes** provide molecular specificity, and **disease context** captures pathological state. In other words, the design criterion was portability and semantic coverage across heterogeneous public datasets, not optimality in a combinatorial sense.

We do have conditioning-field ablation evidence showing that the fields are not interchangeable and do contribute unequally to the conditioning signal. In the current manuscript, removing marker genes reduces centroid cosine, and retaining only metadata causes steering to drop from 99.8% to 62.4%. Supplementary field-retention analysis further indicates that the cell-type field carries the dominant identity anchor, while additional fields modulate conditioning quality. However, we agree that these results demonstrate **usefulness and relative contribution**, not **global optimality**.

We will therefore revise the manuscript to avoid any implication that the five-field template is optimal. Instead, we will present it as a biologically motivated minimal structured template that works well enough to support the proof-of-concept task.

**Planned manuscript revision.**

- Clarify that the five fields were selected for biological interpretability, portability across datasets, and consistent availability.
- Replace any wording implying “optimality” with “practical structured scaffold” or equivalent phrasing.
- Expand the Discussion to note that future work could formalize field-wise or subset-wise prompt optimization.

**Recommended additional work.**  
A formal all-subset field ablation would strengthen the claim if space and compute allow, but we do not think such an experiment is strictly necessary if the manuscript is reworded to claim sufficiency rather than optimality.

**Status.** **Additional analysis recommended**

---

### Comment 2.3

> The author used Wilcoxon rank sum test on the training set to extract marker genes. How to prove that the model understands biological semantics during the generation process, rather than just "remembering" a specific gene list in the training set.

**Response.**  
We agree that the manuscript should be careful not to overstate “understanding.” We do not claim that the model has human-like biological understanding. The scientifically defensible claim is narrower: the model learns a conditioning pathway that responds to biologically meaningful marker-bearing prompts rather than merely replaying fixed training captions.

Three pieces of current evidence support this narrower claim. First, the training-split Wilcoxon markers show substantial overlap with **independent external resources** (PanglaoDB and CellMarker 2.0), with Jaccard overlap in the 0.39--0.45 range, far above random expectation. Second, when we substitute **external marker sets** from those databases, the resulting generations remain close to the full-prompt variant (pairwise cosine similarity 0.702 on the 15-type subset; KNN on the subset is also similar between the external-marker and full-prompt variants). Third, in the **swap-label permutation test**, the generated latents follow the prompt semantics rather than the evaluation label in all 15/15 mismatched cases. Taken together, these results argue against a trivial explanation based solely on memorizing a fixed prompt string.

In the revision, we will soften the language from “understands biological semantics” to “responds to biologically relevant conditioning content” or “captures biologically meaningful prompt sensitivity.” We think this is both more precise and more scientifically appropriate.

**Planned manuscript revision.**

- Replace strong “understanding” language with narrower causal-conditioning language.
- Emphasize the independent-marker substitution and swap-label evidence more clearly in the Results and Discussion.

**Status.** **Text revision only**

---

### Comment 2.4

> The quality of Figures 1 and 2 needs improvement, as the text, arrows, and rectangles appear very cluttered and have low readability. The font color and size in the experimental results also need to be modified to ensure the clarity and consistency of the font, and not to obscure the data (Figure 11 (c)).

**Response.**  
We agree. This is a presentation issue rather than a conceptual disagreement, and the reviewer’s point is well taken. In the revision, we will simplify the schematic layout in Figures 1 and 2, reduce text density, increase whitespace between blocks, and standardize font size, color, and line weights across all figure panels. We will also adjust legend placement and annotation density in the result figures so that labels no longer obscure the plotted data.

**Planned manuscript revision.**

- Regenerate Figures 1 and 2 with simplified geometry, larger spacing, and more consistent typography.
- Revise the cluttered result panels (including the reviewer-cited panel) to prevent text overlap and masking of data.
- Harmonize figure fonts, annotation styles, and legend positions across the manuscript.

**Status.** **Text/figure revision only**

---

### Comment 2.5

> The training accuracy mentioned in Section 3.1 is 87%, but the validation accuracy is only 2.5%. Although significantly higher than random (0.39%), the absolute level is still very low. This means that in most cases, the model cannot correctly match text and corresponding cells from a small batch. It is not yet clear whether this low accuracy will affect the quality of the generated data.

**Response.**  
This is an important point, and we agree that the manuscript should explain more clearly what this metric does and does not imply. The training/validation accuracy reported in the CLOP stage is a **stage-specific retrieval-style proxy** measured in a highly difficult contrastive setting, not the final generation metric. The reviewer is correct that the absolute validation value is low. However, its meaning should be interpreted in the context of the geometry of the input space and the difficulty of the task.

In our setting, the raw scGPT type centroids are extremely compressed (mean pairwise cosine ≈ 0.994), so even moderate improvement in prototype separation can be meaningful for downstream conditioning. The role of CLOP is therefore not to solve a high-accuracy text-cell retrieval task in isolation, but to construct a more separable condition space for the generator. This is why we interpret the strong geometry change (raw centroid cosine collapse relieved after projection) and the downstream conditional-generation evidence as more important than the absolute prototype-accuracy number alone. Indeed, the unconditional control collapses to chance-level type specificity (KNN ≈ 1.0%, steering ≈ 47.5%), whereas the conditional generator reaches 36.9% KNN and 81.0% steering in the high-fidelity operating regime. This indicates that the learned condition space is functionally useful for generation, even though the retrieval-style validation proxy remains difficult.

We will revise the manuscript to make this relationship explicit: low validation prototype accuracy indicates that instance-level retrieval remains hard, but it does not by itself imply that the conditional generation stage is ineffective.

**Planned manuscript revision.**

- Clarify that CLOP validation accuracy is a stage-specific proxy, not a direct generation-quality metric.
- Explicitly explain why downstream generation quality should be judged primarily by conditional-generation endpoints (KNN, steering, DivR), not by CLOP retrieval accuracy alone.

**Status.** **Text revision only**

---

### Comment 2.6

> The indicator values of KNN-1 and KNN-5 mentioned in sections 3.3 and 3.4 are approximately 36% and 55%, respectively. This means that about 2/3 of the generating cells are assigned to the "wrong" cell type. But to what extent is the 'error'? Are they classified into similar types (such as CD4+T vs CD8+T) or completely unrelated types (such as T cells vs epithelial cells)? The author did not provide a confusion matrix or finer grained analysis.

**Response.**  
We agree. The reviewer is correct that the biological severity of a KNN error depends on *where* the error goes. A confusion between adjacent immune states is not equivalent to a confusion across major lineages. The current manuscript notes that the scGPT latent space is tightly packed, so small embedding perturbations can affect KNN decisions, but that does not replace an explicit error taxonomy.

We already provide a downstream classifier confusion matrix in the joint expression-space analysis, but we agree that this is not the same as a **latent-space KNN confusion analysis** for the headline metric itself. We will therefore add a finer-grained analysis that distinguishes whether KNN failures are concentrated within related immune or stromal neighborhoods versus across broad cell families.

**Planned manuscript revision.**

- Add a confusion-style summary or lineage-aware error analysis for the KNN evaluation.
- Explicitly discuss whether most KNN failures are local (within related states) or gross (across unrelated lineages).

**Recommended additional work.**  
This is best handled as an additional analysis on the existing evaluation outputs and does not require retraining.

**Status.** **Additional analysis recommended**

---

### Comment 2.7

> The conclusion in Section 3.5 that failure modes are concentrated in a few biologically difficult or underrepresented types lacks quantitative support. The authors did not define what "biologically difficult" or "underrepresented" means, nor did they provide statistical evidence (such as whether there is a significant difference in centroid cosine between low abundance and high abundance types). The abundance fidelity scatter plot in Figure 6 (c) may suggest this relationship, but the results of the correlation analysis are not described in the main text.

**Response.**  
We agree in part. The reviewer is right that “biologically difficult” is too vague as currently written and should either be operationalized or removed. The current data do support an abundance-related effect more clearly than a generic “difficulty” claim: the supplementary failure-analysis panel reports a positive correlation between quality score and training sample size (Pearson $r = 0.43$), which is consistent with underrepresented types being more challenging. However, we agree that this result should be described explicitly in the main text, and that the term “biologically difficult” should not remain undefined.

In the revision, we will narrow the wording. Rather than claiming failure is concentrated in “biologically difficult” types, we will state that poorer performance is enriched among **lower-abundance and/or more heterogeneous states**, and we will report the relevant abundance-quality correlation directly in the Results.

**Planned manuscript revision.**

- Remove or redefine the phrase “biologically difficult.”
- Add the abundance-quality correlation statistics to the main text.
- Define “underrepresented” explicitly (for example, by training-count threshold or abundance quantile).

**Recommended additional work.**  
A modest additional analysis on existing per-type outputs would strengthen this section: e.g., centroid-cosine vs. cell count, low-vs-high abundance group comparison, or abundance-stratified confidence intervals.

**Status.** **Additional analysis recommended**

---

### Comment 2.8

> Section 3.7, Figure 9 (d) clearly shows the "residual distribution of each gene" and "percentage within the tolerance band", while Figure 10 (d) displays the "highly variable genes sorted by standard deviation ratio". These graphs involve variance and residuals, but there is no explanation or analysis of the results of these graphs in the main text.

**Response.**  
We agree. These panels were intended to support the same core interpretation—namely, that mean-level fidelity is better preserved than variance-level fidelity—but we did not explain them adequately in the main text. In the revision, we will explicitly interpret these plots.

The intended interpretation is as follows: the residual-distribution and tolerance-band plots indicate that many genes have relatively small mean-expression errors, which is consistent with the strong mean-level correlations. By contrast, the standard-deviation-ratio ranking of highly variable genes highlights where the model fails most clearly: variance preservation is weakest in the genes that carry the strongest cell-to-cell heterogeneity signal. In short, the current model preserves **average expression patterns** much better than **gene-specific variability structure**.

**Planned manuscript revision.**

- Add explicit text explaining the residual-distribution and tolerance-band panels.
- Add explicit text explaining that the ranked HVG standard-deviation ratios localize the variance deficit to highly variable genes.
- Connect these panels directly to the broader “first-order vs. second-order fidelity” interpretation.

**Status.** **Text revision only**

---

### Comment 2.9

> Table 4 mentions that "however, per gene variation correlation (r_var) is near zero or slightly negative across all datasets (range: -0.019 to+0.013), indicating that while the model reproduces mean expression accuracy, it does not faithfully preserve the gene level variability structure." In the high diversity mode, the overall diversity ratio (DivR) reaches 0.93, which seems close to the ideal value, but the variance correlation of each gene is almost zero (-0.019 to+0.013). The paper did not explain why the overall variance is close to reality, while the variance of individual genes has no linear relationship with reality. This contradiction weakens the credibility of DivR as a quality indicator. No baseline methods (such as unconditional generation, Gaussian baseline) were provided for the variance correlation of each gene. It is impossible to determine whether the defect is unique to CLOP DiT or a common issue in all latent variable generation models.

**Response.**  
We appreciate this comment and agree that the manuscript should explain the distinction more clearly. We do not view DivR and $r_{\text{var}}$ as contradictory; rather, they measure two different aspects of variability.

DivR is a **group-level aggregate scalar**: it asks whether the overall magnitude of within-type spread is approximately correct. By contrast, $r_{\text{var}}$ is a **gene-wise correspondence measure**: it asks whether the *same genes* that are variable in the real data are also the ones that are variable in the generated data. A model can therefore recover the *amount* of spread at the group level while still allocating that spread to the wrong genes, resulting in DivR near 1 but gene-wise variance correlation near 0. This is precisely the pattern we observe: the generator can approximately restore overall within-type dispersion in the latent space, but it does not faithfully preserve the per-gene second-order structure after decoding.

We agree, however, that the manuscript should make this distinction explicit and that it would be valuable to provide gene-wise variance-correlation baselines for comparison. The reviewer is right that without unconditional or Gaussian reference values, it is difficult to determine whether near-zero $r_{\text{var}}$ is specific to CLOP-DiT or common across latent generators under the same decoder.

**Planned manuscript revision.**

- Add an explicit explanation distinguishing DivR (aggregate spread magnitude) from $r_{\text{var}}$ (gene-wise variance allocation fidelity).
- Clarify that the shared frozen decoder and many-to-one mapping further weaken the interpretability of gene-level variance correspondence.

**Recommended additional work.**  
We recommend adding gene-wise variance-correlation values for at least the unconditional DiT control and the Gaussian baseline on the same five-dataset validation suite.

**Status.** **Additional analysis recommended**

---

### Comment 2.10

> The results of the CLOP ablation study in Section 3.12 are severely disconnected from the production model, as the fixed temperature used in the production model (τ=14.0) is completely different from the learnable temperature in the ablation baseline. The hierarchical validation segmentation of the production model is also different from the setup of the ablation experiment. Does the conclusion drawn from the ablation experiment, such as "removing condensation regularization to improve prototype accuracy," hold true in the production model? Completely unknown. If the ablation experiment can only reflect the performance of the CLOP alignment stage and cannot reflect the impact on the final generated quality (KNN, diversity, directionality), then these ablation results have almost zero overall value for evaluating CLOP DiT.

**Response.**  
We agree with the reviewer’s central concern. The current ablation table is informative as a **stage-specific mechanistic analysis of CLOP alignment**, but it should not be interpreted as direct evidence about end-to-end generation quality in the final production configuration. The manuscript already notes this limitation, but we agree that the wording should be made even stricter so that readers do not over-extrapolate from the CLOP ablation baseline to the full pipeline.

Our view is that these ablations still have value, but that value is local: they help explain how different regularization terms affect prototype separation when the principal bottleneck is inter-group discrimination in a highly collapsed raw space. They do **not** by themselves establish that the same intervention would improve downstream DiT performance, diversity, or biological fidelity in the production setting.

In the revision, we will therefore narrow the claim. We will explicitly frame the ablation section as a mechanistic study of the alignment stage only. If we are unable to add a bridge experiment connecting selected CLOP ablations to downstream generation endpoints, we will not use the ablation table to support stronger end-to-end claims.

**Planned manuscript revision.**

- Strengthen the disclaimer that the CLOP ablation baseline is not the production model.
- Remove or soften any wording that implies direct transfer from prototype-accuracy changes to final generation quality.

**Recommended additional work.**  
If feasible, add a small bridge experiment evaluating one or two representative CLOP ablations downstream (e.g., KNN / steering / DivR for selected variants). If this is not feasible, the safer alternative is to keep the section but narrow its scope explicitly.

**Status.** **Additional experiment recommended**

---

### Comment 2.11

> The author reported rare cell enhancement failures, but lacked deeper analysis. Is it because the generated cells are too homogeneous (averaged), or is it because the small shift in latent variable space is amplified in the expression space? Increasing exploration in this area will greatly enhance the academic depth of the paper.

**Response.**  
We agree, and we appreciate the reviewer for identifying the key mechanistic question. Based on the current evidence, the stronger explanation is that the generated rare-cell samples are **too centroid-biased / insufficiently heterogeneous**, rather than simply being catastrophically misplaced. In the current pilot, Cycling-cell F1 decreases from 0.828 to 0.741--0.786 after augmentation, while overall macro-F1 remains broadly stable (0.947--0.964). This pattern suggests that the synthetic samples are not so unrealistic that they globally corrupt the classifier, but they also do not add the within-class boundary variation needed for positive rare-class augmentation.

This interpretation is consistent with the broader evidence in the manuscript: the high-fidelity regime is under-dispersed (DivR = 0.513), cross-dataset gene-wise variance correlation is near zero, and the Discussion already concludes that within-type heterogeneity remains underrepresented. That said, the reviewer’s second possibility is also important: because the decoder is frozen and many-to-one, subtle latent differences may be compressed or distorted when mapped to expression space. We therefore think the most accurate interpretation is that the rare-cell failure reflects **latent under-dispersion first**, with possible additional degradation introduced at the decoding stage.

**Planned manuscript revision.**

- Expand the rare-cell augmentation discussion to distinguish “realistic but too homogeneous” from “grossly misplaced” synthetic samples.
- Explain why the stability of macro-F1 argues against catastrophic off-manifold generation, while the drop in rare-class F1 is more consistent with insufficient within-type heterogeneity.

**Recommended additional work.**  
We recommend a small follow-up analysis comparing rare-type dispersion in latent space and expression space (for example, nearest-neighbor spread, within-type variance, or rare-type centroid shift), which would help separate the roles of latent under-dispersion and decoder compression.

**Status.** **Additional analysis recommended**

---

### Additional Reviewer #2 Concern: Gaussian benchmark vs. diffusion model

> In the discussion section, the author acknowledges that Gaussian benchmarks outperform CLOP DiT in general metrics. If that's the case, why use a complex diffusion model? The author should provide an explanation for this.

**Response.**  
We agree that this point deserves a clearer explanation. The key issue is that the Gaussian baseline and CLOP-DiT are not solving exactly the same scientific problem. The Gaussian baseline is a very strong control when evaluation is restricted to **global distributional similarity within already observed types**, because it samples directly around per-type statistics in the real scGPT latent space. In that sense, it is an efficient type-aware moment-matching baseline.

CLOP-DiT is introduced for a different reason: it provides a **learned prompt-conditioned mapping from structured biological descriptions to cell latents**, rather than requiring access to per-type empirical latent statistics at generation time. It also yields an interpretable fidelity-diversity operating envelope through classifier-free guidance and supports causal prompt interventions (field ablation, swap-label tests) that a Gaussian sampler does not. We therefore do not argue that CLOP-DiT already dominates Gaussian on every distributional metric. Instead, we position it as a proof-of-concept framework for **structured-metadata-conditioned controllable generation**.

In the revision, we will make this distinction more explicit so that the reader understands why a more complex conditional generator is scientifically valuable even when a simple statistical baseline remains competitive on shared global metrics.

**Planned manuscript revision.**

- Add a short Discussion paragraph explaining that Gaussian is a strong type-aware moment-matching baseline, whereas CLOP-DiT addresses prompt-conditioned amortized generation.
- Limit any implication that CLOP-DiT is already uniformly superior to simple baselines.

**Status.** **Text revision only**

---

## Reviewer #3

### Comment 3.1

> The 8 held-out validation datasets share tissue types and cell populations with the training corpus, representing an interpolation test. To demonstrate the robustness of the prompt-based conditioning, the authors should add a strict out-of-distribution experiment. I am very interested if the model is able to generate biologically plausible latents for a completely unseen cell type or an entirely new tissue context (using existing public datasets). I believe that this analysis would largely elevate the impact of this paper.

**Response.**  
We agree. The reviewer is correct that the current 8-study held-out split is an interpolation-style test within the observed distribution rather than a strict OOD benchmark. The manuscript already states this explicitly, and we appreciate the suggestion to strengthen the OOD evaluation.

We do already have preliminary OOD prompt tests in the supplementary material, including six novel cell-type prompts and free-form prompt variants. However, we agree that these should be interpreted cautiously. In their current form, they function more as **preliminary negative evidence** than as a definitive OOD benchmark: only 1 of 6 novel prompts shows marker-hit evidence, and the mean novel-type marker-hit rate is 0.167. That is useful for honesty, but it is not yet the kind of strict OOD demonstration the reviewer is requesting.

In the revision, we will either (i) add a stricter dataset-based OOD experiment if feasible, using a genuinely unseen tissue/cell-type context from public data, or (ii) more clearly label the current OOD prompt tests as exploratory failure analysis rather than robust OOD generalization.

**Planned manuscript revision.**

- Clarify in the main text that the current held-out split is interpolation, not strict OOD.
- Reframe the existing OOD prompt tests as exploratory and preliminary.

**Recommended additional work.**  
A strict OOD dataset-based experiment is strongly recommended and would substantially strengthen the paper.

**Status.** **Additional experiment recommended**

---

### Comment 3.2

> The training dataset is a mix of human (59 datasets) and mouse (21 datasets) samples. Currently, the performance metrics are aggregated. The authors should provide a supplementary experiment or analysis that evaluates generation quality by organism. This would reveal whether the shared CLOP pretraining space successfully facilitates cross-species transfer learning or if the model underperforms on the minority species.

**Response.**  
We agree. The current pooled reporting obscures whether generation quality differs systematically between human and mouse subsets, and the reviewer is right that organism-stratified results would be more informative. The present manuscript includes both species in the training corpus and explicitly lists the dataset counts, but it does not yet report performance by organism.

We do not want to over-interpret the pooled metrics as evidence of cross-species transfer success. In the revision, we will either add organism-stratified supplementary metrics or explicitly temper the manuscript so that it does not imply successful cross-species transfer without direct evidence.

**Planned manuscript revision.**

- Add a statement in the Discussion clarifying that pooled human+mouse evaluation does not establish cross-species transfer.
- If feasible, add organism-stratified evaluation in the supplement.

**Recommended additional work.**  
A species-stratified evaluation is strongly recommended. This should be feasible as an additional analysis on existing metadata and outputs, without changing the model.

**Status.** **Additional analysis recommended**

---

### Comment 3.3

> The pilot experiment on rare-cell augmentation (using "Cycling cells") did not yield improvements in classifier performance. Instead of concluding that augmentation is ineffective, the authors should experiment with different data-mixing strategies. Testing combinations of generated data with traditional techniques (like SMOTE or random oversampling), or fine-tuning the mix ratios, would provide a more comprehensive view of how CLOP-DiT can be leveraged for class imbalance problems.

**Response.**  
We agree. The current rare-cell experiment was intentionally simple and was meant as a proof-of-concept pilot, not as a definitive evaluation of synthetic augmentation as a whole. The negative result should therefore be interpreted more narrowly: **naive augmentation with the current generator and mixing protocol was ineffective**. We agree that it would be too strong to generalize this to all augmentation strategies.

In the revision, we will narrow the wording accordingly. If feasible, we will add one or more hybrid baselines such as random oversampling, SMOTE, or mixed real+synthetic strategies across several ratios. Even if we do not add all of these experiments in the current revision, we will explicitly state that the present result should not be read as a general verdict on synthetic augmentation.

**Planned manuscript revision.**

- Replace any broad wording suggesting “augmentation is ineffective” with the narrower claim that naive augmentation was ineffective in this pilot setup.
- Expand the Discussion to explain why class-boundary enrichment may require more diverse or hybrid augmentation strategies.

**Recommended additional work.**  
A hybrid augmentation experiment (e.g., CLOP-DiT + random oversampling / SMOTE, with ratio sweep) would materially strengthen the paper.

**Status.** **Additional experiment recommended**

---

### Comment 3.4

> In Section 2.2, the authors claim that ZCA whitening was adopted over LayerNorm or mean-centering based on 'preliminary sensitivity checks'. Given the importance of the text-embedding landscape for the overall pipeline, the authors should provide a formal ablation study comparing ZCA against LayerNorm or no-whitening.

**Response.**  
We agree that the present wording is stronger than the current formal evidence in the manuscript. The preprocessing choice was motivated by internal sensitivity checks and by the severe collapse of the raw embedding geometry. In our internal experiment registry, removing cell whitening causes contrastive learning to fail under strongly collapsed raw geometry (raw cell cosine ≈ 0.991). This explains why whitening became part of the production preprocessing stack. However, we agree that this does not substitute for a manuscript-quality, controlled ablation comparing ZCA against mean-centering only, LayerNorm-only, or no-whitening.

We will therefore revise this section in one of two ways. If feasible, we will add a formal supplementary ablation comparing the alternatives directly. If not, we will narrow the wording so that ZCA is presented as an empirically selected implementation choice supported by preliminary diagnostics, rather than as a formally established superiority claim.

**Planned manuscript revision.**

- Soften the current wording around “preliminary sensitivity checks” unless a formal ablation is added.
- Explain more clearly that whitening was introduced to address a severely collapsed encoder geometry.

**Recommended additional work.**  
A formal preprocessing ablation (ZCA vs. mean-centering vs. LayerNorm-only vs. no-whitening) is recommended and would directly address this concern.

**Status.** **Additional experiment recommended**

---

## Suggested Priority Order for Extra Work

If revision time is limited, the following additions would likely give the largest rebuttal value per unit effort:

1. **Strict OOD evaluation** (Reviewer 3.1) — highest impact.
2. **Organism-stratified evaluation** (Reviewer 3.2) — likely feasible from existing outputs.
3. **KNN error taxonomy / confusion analysis** (Reviewer 2.6) — high interpretive value with modest cost.
4. **Baseline gene-wise variance correlation** for Gaussian and unconditional controls (Reviewer 2.9) — directly addresses a central metric concern.
5. **Rare-cell hybrid augmentation baseline** (Reviewer 3.3) — strengthens application discussion.
6. **Formal whitening ablation** (Reviewer 3.4) — important, though potentially more time-consuming if reruns are needed.
7. **Bridge ablation from CLOP to full pipeline** (Reviewer 2.10) — useful but can be replaced by stricter wording if time is constrained.

---

## Final Positioning Sentence for the Rebuttal

If the response letter needs a short unifying sentence, the following framing is likely the safest and most persuasive:

> We agree with the reviewers that the present manuscript is strongest as a proof-of-concept for structured-metadata-conditioned single-cell generation, not yet as a high-fidelity single-cell simulator. In the revision, we will sharpen this scope, strengthen the interpretation of existing results, and, where feasible, add targeted analyses or experiments that directly address OOD robustness, organism stratification, variance fidelity, and augmentation utility.
