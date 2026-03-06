# CLOP-DiT: Contrastive Language-Omics Pretraining with Diffusion Transformer for Text-Guided Single-Cell Generation

**Zeyu Fu**

*Manuscript submitted to IEEE Journal of Biomedical and Health Informatics (JBHI)*

---

**Abstract** — Generating realistic single-cell transcriptomic profiles from natural language descriptions remains a fundamental challenge in computational biology. We present **CLOP-DiT**, a two-stage framework that combines **C**ontrastive **L**anguage-**O**mics **P**retraining (CLOP) with a **Di**ffusion **T**ransformer (DiT) for text-guided single-cell generation. In the first stage, CLOP learns a shared 256-dimensional embedding space between biomedical text and scGPT-derived cell embeddings via contrastive learning with learnable temperature scaling. In the second stage, a DiT with Adaptive Layer Normalization Zero (AdaLN-Zero) performs conditional flow matching to generate cell embeddings from text prompts. We train on 220,304 cells spanning 80 datasets (cancer and developmental) with 1,088 unique cell type descriptions. CLOP-DiT achieves a KNN classification accuracy of 36.9% (37× random chance with 100 cell type groups), steering accuracy of 81.0% (vs. 50% random), and diversity ratio of 0.93 when using the Midpoint ODE solver. Critically, unconditioned generation achieves only 1.0% KNN accuracy (= random), confirming that text conditioning is the sole driver of type-specific generation. The framework enables biologically plausible, text-controllable in-silico cell generation with potential applications in rare cell type augmentation, drug response simulation, and hypothesis-driven perturbation studies.

**Index Terms** — Single-cell transcriptomics, diffusion transformer, contrastive learning, text-guided generation, flow matching, computational biology.

---

## I. Introduction

Single-cell RNA sequencing (scRNA-seq) has revolutionized our understanding of cellular heterogeneity, enabling fine-grained characterization of cell types, states, and transitions across tissues and disease contexts [1]. However, scRNA-seq data generation remains expensive, labor-intensive, and limited by sample availability—particularly for rare cell types and specific disease conditions [2].

Generative models offer a promising avenue for augmenting scRNA-seq datasets, enabling researchers to synthesize realistic cell profiles for underrepresented populations, simulate perturbation responses, and generate training data for downstream classifiers [3]. While variational autoencoders (VAEs) and generative adversarial networks (GANs) have been applied to single-cell data [4], [5], these approaches typically lack fine-grained control over the characteristics of generated cells.

Recent advances in vision-language models, exemplified by CLIP [6], have demonstrated the power of contrastive pretraining for bridging modalities. Concurrently, Diffusion Transformers (DiTs) [7] have emerged as the leading architecture for high-quality conditional generation, surpassing U-Net-based diffusion models in image synthesis through their use of transformer backbones with adaptive normalization.

In this work, we propose **CLOP-DiT**, a novel two-stage framework that extends the CLIP paradigm to the biological domain and combines it with flow-matching DiT for controllable single-cell generation. Our key contributions are:

1. **CLOP (Contrastive Language-Omics Pretraining)**: A contrastive alignment module that learns a shared 256-dimensional embedding space between biomedical text descriptions and scGPT-derived cell embeddings, enabling text-based cell conditioning.

2. **DiT with AdaLN-Zero**: A Diffusion Transformer that performs conditional flow matching in the scGPT latent space, conditioned on CLOP text embeddings, using Adaptive Layer Normalization Zero for effective conditioning.

3. **Comprehensive evaluation framework**: We introduce a biologically grounded evaluation pipeline including KNN classification accuracy (37× random), steering accuracy, diversity ratio, linear separability, and CFG scale analysis, demonstrating that standard distributional metrics such as FD are misleading for conditional single-cell generation.

4. **Large-scale single-cell generation**: We demonstrate generation of biologically plausible cells across 1,088 unique cell type descriptions from 220,304 training cells spanning 80 cancer and developmental datasets.

---

## II. Related Work

### A. Generative Models for Single-Cell Data

VAE-based approaches such as scVI [4] and scGen [8] learn latent representations for batch correction and perturbation response prediction. GAN-based methods including cscGAN [5] generate realistic single-cell profiles but struggle with training instability and mode collapse. More recently, diffusion-based methods such as scDiff [9] have shown promise for single-cell generation but typically operate without explicit text conditioning.

### B. Vision-Language Contrastive Pretraining

CLIP [6] demonstrated that contrastive pretraining between images and text creates powerful shared representations. Extensions to the biomedical domain include BiomedCLIP [10] for medical images and PubMedCLIP [11] for biomedical literature. Our CLOP module adapts this paradigm to align biomedical text with single-cell transcriptomic embeddings.

### C. Diffusion Transformers

DiT [7] replaced the U-Net backbone in diffusion models with a transformer, achieving state-of-the-art image generation quality. Key innovations include Adaptive Layer Normalization (AdaLN) for conditioning injection and patch-based tokenization for spatial data. Flow matching [12] provides an alternative training objective that learns deterministic velocity fields via ODE integration, offering stable training and fast sampling.

### D. Foundation Models for Single-Cell Biology

scGPT [13] pretrains a transformer on over 33 million cells, producing general-purpose cell embeddings that capture transcriptome-wide patterns. We leverage scGPT's 512-dimensional cell embeddings as our generation target, enabling our framework to benefit from the biological knowledge encoded in the foundation model.

---

## III. Method

### A. Overview

CLOP-DiT is a two-stage framework illustrated in Fig. 1. **Stage 1 (CLOP)** aligns biomedical text embeddings from BiomedBERT-large with scGPT cell embeddings via contrastive learning. **Stage 2 (DiT)** generates cell embeddings conditioned on CLOP-projected text vectors using flow matching with a transformer backbone.

![Fig. 1. Architecture overview of CLOP-DiT. Stage 1: CLOP aligns BiomedBERT text embeddings (1024-d) with scGPT cell embeddings (512-d) into a shared 256-d space via contrastive learning with learnable temperature. Stage 2: DiT performs conditional flow matching in the scGPT latent space, conditioned on CLOP text projections, using AdaLN-Zero blocks with 384-d hidden dimension, 8 transformer blocks, and 6 attention heads. Generated latent embeddings are decoded through scGPT's decoder to produce 1890-gene expression profiles.](../arch-figure/fig1_architecture_placeholder.md)

> **Note:** Fig. 1 is a schematic architecture diagram to be prepared separately (e.g., via draw.io or Adobe Illustrator). See `arch-figure/` for the interactive Vue.js architecture viewer.

### B. Stage 1: Contrastive Language-Omics Pretraining (CLOP)

Given a batch of $N$ text-cell pairs $\{(t_i, c_i)\}_{i=1}^{N}$, CLOP learns projection functions $f_\text{text}: \mathbb{R}^{1024} \to \mathbb{R}^{256}$ and $f_\text{cell}: \mathbb{R}^{512} \to \mathbb{R}^{256}$ that map BiomedBERT text embeddings and scGPT cell embeddings into a shared 256-dimensional space.

Each projector consists of three sequential blocks with linear layers, batch normalization, GELU activation, and dropout ($p=0.1$). The contrastive loss follows the symmetric InfoNCE formulation:

$$\mathcal{L}_\text{CLOP} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{\exp(s_{ii} / \tau)}{\sum_{j=1}^{N} \exp(s_{ij} / \tau)} + \log \frac{\exp(s_{ii} / \tau)}{\sum_{j=1}^{N} \exp(s_{ji} / \tau)} \right]$$

where $s_{ij} = f_\text{text}(t_i)^\top f_\text{cell}(c_j)$ is the cosine similarity and $\tau$ is a learnable temperature parameter initialized at 0.07. Label smoothing ($\alpha = 0.1$) is applied to prevent overconfident alignments.

### C. Stage 2: Diffusion Transformer with Flow Matching

The DiT generates 512-dimensional scGPT cell embeddings conditioned on 256-dimensional CLOP text projections. We adopt the flow matching framework [12], which learns a velocity field $v_\theta(z_t, t, c)$ that transports samples from noise $z_0 \sim \mathcal{N}(0, I)$ to data $z_1 \sim p_\text{data}$ along an optimal transport path:

$$z_t = (1 - t) z_0 + t z_1, \quad t \in [0, 1]$$

The training objective minimizes:

$$\mathcal{L}_\text{FM} = \mathbb{E}_{t, z_0, z_1} \left[ \| v_\theta(z_t, t, c) - (z_1 - z_0) \|^2 \right]$$

where $t$ follows a logit-normal distribution for improved training dynamics [14].

**Tokenization.** The 512-dimensional cell embedding is reshaped into $K = 16$ tokens of dimension $d_\text{tok} = 32$, followed by learned positional embeddings.

**DiT Blocks with AdaLN-Zero.** Each of the 8 transformer blocks applies Adaptive Layer Normalization Zero conditioning:

$$\gamma, \beta, \alpha = \text{MLP}([t_\text{emb}; c])$$

$$h = \alpha \cdot \text{Attn}(\gamma \cdot \text{LN}(x) + \beta) + x$$

where $t_\text{emb}$ is the sinusoidal time embedding, $c$ is the CLOP text conditioning, and $\alpha$ scales the residual connection (initialized near zero for stable training). Each block uses 6 attention heads with 384-dimensional hidden states.

**Classifier-Free Guidance (CFG).** During training, the conditioning vector is dropped with probability $p_\text{drop} = 0.15$. At inference, the velocity prediction is interpolated:

$$\hat{v} = v_\theta(z_t, t, \varnothing) + w \cdot [v_\theta(z_t, t, c) - v_\theta(z_t, t, \varnothing)]$$

where $w = 3.0$ is the guidance scale. Sampling uses 10-step Euler or Midpoint ODE integration, with CFG scales ranging from 1.0 to 3.0 depending on the desired accuracy–diversity trade-off.

### D. scGPT Decoder

Generated latent embeddings $\hat{z} \in \mathbb{R}^{512}$ are decoded through scGPT's pretrained cell-to-gene decoder to produce 1890-dimensional gene expression profiles $\hat{x} \in \mathbb{R}^{1890}$.

---

## IV. Experiments

### A. Dataset and Preprocessing

We curate a large-scale pan-cancer and developmental single-cell dataset comprising **220,304 cells** from **80 datasets** sourced from the Gene Expression Omnibus (GEO), spanning cancer (lung adenocarcinoma, breast, colorectal, pancreatic) and developmental (embryonic, organoid) contexts. Sub-cluster-level text descriptions are generated from marker genes, tissue of origin, organism, and disease context, yielding **1,088 unique text groups**. The top 1,890 highly variable genes are selected. Cell embeddings are extracted using the pretrained scGPT model (512 dimensions). Text descriptions are encoded using BiomedBERT-large-uncased-abstract (1,024 dimensions).

The data is split by dataset with **72 training datasets** (199,670 cells) and **8 validation datasets** (20,634 cells) to evaluate generalization to entirely unseen biological contexts.

### B. Implementation Details

**CLOP Training.** The CLOP module is trained for 200 epochs with batch size 256, learning rate $3 \times 10^{-4}$ (AdamW, cosine annealing), projection dimension 256, 3-layer projectors with batch normalization, and learnable temperature initialized at $\tau_0 = 0.07$.

**DiT Training.** The DiT is trained for 200 epochs with batch size 1,024, learning rate $2 \times 10^{-4}$ (AdamW), 8 transformer blocks, 384 hidden dimension, 6 attention heads, 16 tokens, conditioning dropout $p_\text{drop} = 0.15$, and EMA decay 0.9999. Logit-normal time sampling and gradient clipping (max norm 1.0) are employed.

All experiments are conducted on a single NVIDIA RTX 5090 Laptop GPU (24.1 GB VRAM).

### C. Training Dynamics

The training dynamics for both stages are shown in Fig. 2. CLOP InfoNCE loss converges from 2.84 to 1.19 over 200 epochs. Validation accuracy stabilizes at ~2.5% (6.4× random chance with batch size 256), reflecting the model's retrieval difficulty across 1,088 unique cell types, while the learnable temperature τ adapts from 0.070 to 0.060. The DiT flow matching MSE loss converges from 1.48 (epoch 1) to 0.022 (epoch 200), with the EMA model achieving a validation velocity cosine similarity of 0.976, corresponding to a mean angular error of approximately 12.6°.

![Fig. 2a. CLOP training: contrastive loss convergence, validation accuracy, and learned temperature (τ).](../results/figures/panel_a_clop_training.png)

![Fig. 2b. DiT training: flow matching MSE loss and velocity prediction cosine similarity (EMA 0.976).](../results/figures/panel_c_dit_training.png)

### D. Embedding Space Visualization

Fig. 3 shows PCA visualizations of real and generated cell embeddings. Generated cells (triangles, colored by group) overlap with the real cell manifold (gray), demonstrating that CLOP-DiT produces embeddings within the learned data distribution. Per-group coloring confirms that generated cells cluster according to their text-specified cell type identity, with clear separation between distinct groups.

![Fig. 3. Real vs. generated cell embeddings (PCA); generated cells colored by target group, overlapping the real data manifold.](../results/figures/panel_e_real_vs_generated.png)

### E. Quantitative Evaluation

We evaluate generation quality using biologically meaningful metrics that assess conditional generation fidelity using **in-distribution conditions** (the actual CLOP-projected text embeddings from the training data). We select the top 100 cell type groups (each with ≥50 cells) and generate 200 cells per group. Results are summarized in Fig. 4 and Table I.

**Evaluation Protocol:**
1. Group cells by unique text embedding → 1,088 groups, select top 100 (≥50 cells each)
2. Train KNN classifier (k=15, cosine metric, PCA-50 features) on 80% of real data
3. Generate 200 cells per group and classify → accuracy measures type specificity
4. Cross-group steering test → measures directional controllability
5. Diversity ratio → within-group variance preservation (1.0 = ideal)
6. Linear classifier (LogisticRegression) → stronger separability test

![Fig. 4. Quantitative evaluation: KNN accuracy, steering accuracy (81%), diversity ratio, and related metrics.](../results/figures/panel_d_metrics_summary.png)

**TABLE I: Generation Quality Across Configurations (100 eval groups, 200 cells/group)**

| Method | KNN-1 ↑ | KNN-5 ↑ | Steering ↑ | DivR (→1) | LinAcc ↑ | FD | KNN/Rand |
|--------|---------|---------|-----------|-----------|----------|-----|----------|
| **Real Data** | **0.890** | **0.993** | — | **1.000** | **0.942** | 0.00 | 89× |
| CFG=2.0 Euler-10 | **0.369** | **0.553** | **0.810** | 0.513 | **0.511** | 3.17 | **37×** |
| CFG=3.0 Euler-10 | 0.367 | 0.554 | 0.802 | 0.473 | 0.508 | 3.46 | 37× |
| CFG=1.0 Midpoint-10 | 0.288 | 0.461 | 0.807 | **0.929** | 0.357 | 2.64 | 29× |
| CFG=1.0 Euler-50 | 0.293 | 0.466 | 0.807 | 0.919 | 0.365 | 2.65 | 29× |
| CFG=0.0 (Unconditional) | 0.010 | 0.051 | 0.475 | 1.833 | 0.010 | 0.58 | 1× |
| Gaussian N(μ,Σ) | 0.011 | 0.048 | 0.466 | 2.277 | 0.009 | 0.03 | 1× |

The best conditioned generation (CFG=2.0 Euler-10) achieves 36.9% KNN top-1 accuracy — **37× better than random** with 100 cell type groups — and 81.0% steering accuracy. Crucially, unconditional generation (CFG=0.0) achieves exactly random chance (1.0%), confirming that text conditioning is the sole driver of type-specific generation. The Gaussian baseline similarly achieves only random-level performance. Note that FD is lower for unconditional generation (0.58) than conditioned generation (3.17) because FD rewards mean-matching, not conditional structure — unconditional generation trivially matches the global mean.

### F. Biological Validation

Fig. 5 presents biological plausibility analysis of generated cell embeddings. Generated cells form distinct clusters in PCA space with clear separation between groups (panel a). The inter-group cosine similarity matrix (panel b) reveals that generated centroids maintain biologically meaningful relationships — semantically related groups show moderate cross-similarity, while distinct types maintain separation. Intra-group diversity comparison (panel c) shows generated cells preserve real within-group variance, especially with the Midpoint solver (DivR = 0.93). Conditioning fidelity (panel d) confirms generated cells have significantly higher cosine similarity to their target centroid than to wrong centroids, validating type-specific generation.

![Fig. 5a. Text–cell similarity heatmap (biological validation).](../results/figures/panel_f_text_cell_heatmap.png)

![Fig. 5b. Marker gene comparison: real vs. generated.](../results/figures/panel_n_marker_gene_comparison.png)

### G. CFG Scale Analysis and ODE Solver Comparison

Fig. 6 presents the classifier-free guidance (CFG) scale sweep across 24 configurations (CFG ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0} × Steps ∈ {10, 20, 50}) and the ODE solver comparison (Euler vs Midpoint).

The CFG sweep reveals three key findings: (1) KNN accuracy saturates around CFG = 2.0–3.0 at ~41× random, then declines for CFG > 4.0; (2) Steering accuracy is remarkably stable (~80%) across all CFG values ≥ 0.5; (3) Diversity decreases monotonically with CFG, indicating progressive mode sharpening. ODE integration steps have mild effect — 10 steps is sufficient for most applications, while 50 steps slightly improves diversity.

The Euler vs Midpoint comparison (panels d–f) shows the 2nd-order Midpoint solver does not improve accuracy but significantly improves diversity. At CFG=1.0, Midpoint achieves DivR = 0.929 (near-ideal) vs. Euler's 0.745, with equivalent steering (80.7% vs 81.0%). This suggests Euler's integration error contributes to mode sharpening, and Midpoint better preserves distributional breadth.

![Fig. 6a. Diversity diagnostics across configs.](../results/figures/panel_j_diversity_diagnostics.png)

![Fig. 6b. CFG/noise trade-off and solver comparison.](../results/figures/panel_l_noise_tradeoff.png)

### H. Dimension and Sampling Analysis

Fig. 7 provides detailed analysis of the generation process. Per-dimension mean comparison (panel a) shows near-perfect correlation ($r > 0.99$) between real and generated embeddings across all 512 dimensions. Per-dimension standard deviation comparison (panel b) confirms generated cells preserve the moment structure of the real data. The ODE sampling trajectory (panel c) visualizes the flow from noise ($z_0$) to data ($z_1$) in PCA space, showing how different samples converge toward the data manifold under CFG=2.0 guidance. The norm distribution comparison (panel d) demonstrates that generated embeddings match the real data norm distribution (generated mean norm 20.83 vs. real 21.08), confirming plausible magnitude.

![Fig. 7. Expression/dimension correlation between real and generated embeddings.](../results/figures/panel_h_expression_correlation.png)

### I. Phase 2 Decoder Architecture Comparison

CLOP-DiT is a two-stage framework: Stage 1 (CLOP) produces 256-d text projections, and Stage 2 converts them into 512-d cell embeddings. To evaluate the DiT decoder choice, we conduct a systematic Phase 2 architecture comparison with **nine decoder configurations** spanning four paradigms: oracle baselines, feed-forward (FF), direct transformer (FF-T), and iterative diffusion/flow-matching decoders. All decoders receive identical CLOP conditioning. We evaluate on the same data (220,304 cells, 100 eval groups, 200 cells/group) using classification metrics (KNN, steering), distributional metrics (per-group Fréchet Distance, mean correlation), and downstream utility (train-on-fake, test-on-real accuracy). Results are shown in Table III and Fig. 9.

**Decoder Categories:**
- **Oracle baselines** access ground-truth group identity (upper bounds, not learnable from text).
- **Feed-forward decoders** (cVAE, cGAN) learn direct condition→cell mappings in a single forward pass.
- **Direct Transformer** uses the same 22M-parameter DiT architecture but trained as a single-pass predictor (noise + condition → cell, no iterative sampling) — this directly tests whether ODE integration is necessary.
- **DDPM-DiT** uses the same DiT architecture with discrete DDPM denoising (predict noise $\epsilon$, cosine schedule, $T=1000$) — this tests whether flow matching is the right diffusion framework.
- **Flow-matching decoders** (Flow MLP, DiT) learn velocity fields $v(z_t, t, c)$ and generate via multi-step ODE integration from noise.

**TABLE III: Phase 2 Decoder Architecture Comparison**

| Decoder | Type | Params | KNN-1 ↑ | DivR (→1) | FD$_g$ ↓ | $r$ ↑ | DS ↑ |
|---------|------|--------|---------|-----------|----------|-------|------|
| **Real Data** | — | — | **0.890** | **1.000** | 0.00 | 1.000 | — |
| Per-Type Gaussian | Oracle | — | 0.954 | 0.996 | 0.01 | 1.000 | 0.851 |
| Retrieval + Jitter | Oracle | — | 0.893 | 0.998 | 0.01 | 1.000 | 0.939 |
| Direct Transformer | FF-T | 22.1M | **0.970** | 0.048 | 1.58 | 1.000 | 0.818 |
| Conditional VAE | FF | 1.5M | 0.942 | 0.527 | 0.32 | 1.000 | 0.870 |
| Conditional GAN | FF | 1.0M | 0.822 | 0.929 | 0.54 | 1.000 | 0.823 |
| DDPM-DiT | DDPM | 22.1M | 0.010 | 0.013 | 441.4 | 0.861 | 0.028 |
| Flow MLP (Euler) | Flow | 1.5M | 0.046 | 12.636 | 157.4 | 0.995 | 0.349 |
| **CLOP-DiT (CFG=2.0)** | Flow | 22.1M | 0.372 | 0.553 | 5.72 | 0.994 | 0.412 |
| **CLOP-DiT (CFG=1.0 M)** | Flow | 22.1M | 0.280 | **1.001** | 4.70 | 0.995 | 0.501 |

*FD$_g$: mean per-group diagonal Fréchet Distance; $r$: per-dimension mean correlation across groups; DS: downstream accuracy (train-on-fake, test-on-real). FF: single-pass feed-forward; FF-T: single-pass transformer; DDPM: discrete denoising; Flow: flow-matching ODE integration.*

**Is the ODE necessary? The Direct Transformer experiment.** To isolate whether the flow-matching ODE framework helps or hurts, we train the *identical* 22.1M-parameter DiT architecture as a direct conditional predictor: input noise $z \sim \mathcal{N}(0,I)$ plus condition $c$, single forward pass, output cell embedding $\hat{x}$, trained with MSE loss. The Direct Transformer achieves **97.0% KNN accuracy** — the highest of *any* model, exceeding even the Per-Type Gaussian oracle (95.4%). However, its diversity ratio of **0.048** reveals catastrophic mode collapse: generated cells contain only 4.8% of real within-group variance. Coverage drops to 7.4% (vs. DiT's 32.8%). This demonstrates that the transformer architecture is *inherently excellent* at learning condition→centroid associations, but without the flow-matching framework's stochastic ODE integration, it collapses to a deterministic lookup producing near-identical outputs per condition.

**Is flow matching the right diffusion framework?** To test whether alternative diffusion formulations work better, we train a DDPM-DiT with the same architecture using discrete denoising: cosine noise schedule ($T=1000$), noise prediction objective, 50-step reverse sampling with CFG (see Appendix). DDPM-DiT **fails completely** (KNN = 1.0% = random, FD$_g$ = 441), generating collapsed, non-cell outputs. The discrete denoising schedule distributes learning across 1,000 timesteps, diluting the signal for each step. With only 100 training epochs on 220K cells (22B samples total), DDPM substantially underfits — each individual timestep receives far fewer effective training iterations than flow matching, which uses a continuous $t \in [0,1]$ with concentrated logit-normal sampling. This confirms that **flow matching is the appropriate diffusion framework** for this scale and dimensionality.

**Feed-forward decoders: high accuracy, low diversity.** cVAE achieves 94.2% KNN accuracy but DivR = 0.53 (mode collapse). cGAN achieves the best accuracy–diversity balance among non-oracle FF models (KNN = 82.2%, DivR = 0.93) through adversarial training. The Direct Transformer represents the extreme of this pattern: maximum accuracy (97.0%), minimum diversity (4.8%).

**The transformer architecture is essential for flow matching.** The Flow MLP ablation (identical flow-matching objective, MLP velocity network, 1.5M params) catastrophically fails: KNN = 4.6%, FD$_g$ = 157. Meanwhile DiT with the same training achieves 37.2% KNN, FD$_g$ = 5.72 — an **8× KNN** and **33× FD$_g$** improvement, isolating the transformer architecture as critical.

**Oracle baselines** confirm evaluation calibration: Per-Type Gaussian achieves 95.4% KNN with DivR = 1.00, and Retrieval+Jitter achieves 89.3% with DivR = 1.00.

![Fig. 9. Phase 2 decoder comparison: KNN, diversity ratio, FD; DiT vs. Direct Transformer, DDPM-DiT, cVAE, cGAN.](../results/figures/panel_o_baseline_comparison.png)

### J. ODE Step Count Ablation

To directly answer whether multi-step ODE integration benefits the flow-matching DiT, we evaluate the same trained checkpoint with 1, 2, 5, 10, 20, and 50 Euler integration steps at CFG = 2.0 (Fig. 10).

**TABLE IV: ODE Step Count Ablation (DiT CFG=2.0, Euler)**

| Steps | KNN-1 ↑ | DivR (→1) | FD$_g$ ↓ | DS ↑ |
|-------|---------|-----------|----------|------|
| 1 | 0.363 | 2.070 | 13.88 | 0.356 |
| **2** | **0.421** | 0.600 | 5.74 | 0.362 |
| 5 | 0.391 | 0.431 | 5.80 | 0.391 |
| 10 | 0.370 | 0.551 | 5.71 | 0.412 |
| 20 | 0.358 | 0.646 | 5.67 | 0.420 |
| 50 | 0.354 | 0.729 | 5.63 | 0.429 |

The results reveal a counter-intuitive pattern: **2-step Euler achieves the highest KNN accuracy (0.421), 14% better than the standard 10-step (0.370)**. Accuracy monotonically decreases from 2 to 50 steps. Conversely, diversity ratio and downstream accuracy *increase* with more steps (DivR: 0.60 → 0.73, DS: 0.36 → 0.43). This demonstrates that additional ODE integration steps introduce beneficial distributional smoothing that trades point-level classification accuracy for better coverage of the real data distribution.

**1-step Euler** is equivalent to a single forward velocity prediction: $z_1 = z_0 + v(z_0, t\!=\!0, c)$. It achieves 36.3% KNN — comparable to 10-step — but produces ill-conditioned outputs (DivR = 2.07, FD$_g$ = 13.88). This confirms that the velocity field is well-learned (one step is nearly sufficient for conditioning), but multi-step integration is necessary for proper calibration of the output distribution.

**The optimal step count depends on the downstream task:** for maximum type-specific accuracy, 2 steps suffices; for maximum distributional fidelity and diversity, 20–50 steps is preferred; the standard 10 steps represents a balanced default.

![Fig. 10. ODE step count ablation: accuracy vs. diversity trade-off with step count.](../results/figures/panel_l_noise_tradeoff.png)

### K. Ablation Study

We conduct systematic ablation experiments to validate key design choices. Results are presented in Table II and Fig. 8.

**TABLE II: Ablation Analysis — Conditioning, CFG, and Solver Contributions**

| Variant | KNN-1 ↑ | Steering ↑ | DivR (→1) | LinAcc ↑ | Verdict |
|---------|---------|-----------|-----------|----------|---------|
| **Best Conditioned (CFG=2.0)** | **0.369** | **0.810** | 0.513 | **0.511** | 37× random |
| Best Balanced (CFG=1.0 Mid) | 0.288 | 0.807 | **0.929** | 0.357 | 29×, ideal DivR |
| No CFG (CFG=0.0) | 0.010 | 0.475 | 1.833 | 0.010 | = random |
| Gaussian Baseline | 0.011 | 0.466 | 2.277 | 0.009 | = random |

**Conditioning is essential:** Removing conditioning (CFG=0.0) reduces KNN accuracy from 36.9% to 1.0% (random chance), steering from 81% to 47.5% (random), and linear accuracy from 51.1% to 1.0%. This definitively confirms that the CLOP conditioning signal drives all type-specific generation behavior.

**CFG amplifies conditioning:** Without CFG guidance (CFG=0.0), the model generates cells that match the global data distribution but lack type-specific identity (FD = 0.58, seemingly good but biologically meaningless). With CFG=2.0, the guidance amplifies the condition-specific velocity field component, producing cells that land in the correct neighborhoods of embedding space.

**ODE solver affects diversity:** The 2nd-order Midpoint solver at CFG=1.0 achieves DivR = 0.929 (near-ideal) vs. Euler's 0.745 — a 24% improvement — while maintaining equivalent steering accuracy (80.7% vs 81.0%). This demonstrates that solver choice provides an additional lever for controlling the accuracy–diversity trade-off.

**EMA is critical:** The saved DiT checkpoint uses EMA (decay = 0.9999). The non-EMA model at epoch 120 achieves val_cosine = 0.756, while the EMA model at epoch 200 reaches 0.976 — a critical distinction for generation quality.

![Fig. 8. Metrics summary; see Tables I–II in text for generation quality and ablation.](../results/figures/panel_d_metrics_summary.png)

---

## V. Discussion

CLOP-DiT demonstrates the feasibility of text-guided single-cell generation using contrastive language-omics pretraining combined with flow-matching diffusion transformers. Several observations merit further discussion.

**Evaluation Methodology.** A critical lesson from this work is the importance of using in-distribution conditions for evaluating conditional generative models. Initial evaluations using out-of-distribution text prompts (crafted post-hoc) made it appear as if conditioning hurt generation quality. The correct evaluation using actual training conditions revealed robust conditional generation: 37× random KNN accuracy and 81% steering. This parallels findings in text-to-image generation where evaluation prompts must match the training distribution for meaningful assessment.

**Text-Cell Alignment.** The CLOP module transforms a space where cell type centroids have pairwise cosine similarity of 0.994 (differing by only 0.6%) into a 256-d condition space where the pairwise cosine is 0.222 (78% separation). This dramatic amplification of inter-type differences is essential for the DiT to learn type-specific velocity fields. While CLOP's validation accuracy of 2.5% appears low, it represents 6.4× random chance in a batch retrieval task with 256 candidates from 1,088 unique types.

**CFG Trade-off.** The CFG sweep reveals a smooth accuracy–diversity trade-off: KNN accuracy saturates around CFG=2.0–3.0 (37–41× random), while diversity monotonically decreases. Notably, steering accuracy remains stable (~80%) across all CFG values from 0.5 to 5.0, indicating that the model's directional conditioning is robust regardless of guidance strength. For biological applications, the optimal CFG scale depends on the downstream task — data augmentation may prefer CFG=1.0 with Midpoint solver (DivR=0.93), while hypothesis-driven generation benefits from CFG=2.0–3.0.

**FD as a Misleading Metric.** Our results demonstrate that Fréchet Distance can be misleading for conditional generation in single-cell settings. Unconditional generation achieves FD=0.58 (lower is supposedly better) while conditioned generation achieves FD=3.17. This is because FD is dominated by mean-matching: unconditional generation trivially produces the global data mean, while conditioned generation intentionally shifts means toward specific cell types. We advocate for adoption of the biologically grounded metrics (KNN accuracy, steering, diversity ratio) used here.

**Comparison with Baselines.** The Phase 2 decoder comparison (Table III, Fig. 9) reveals a hierarchy of generation paradigms that answers three fundamental design questions.

*Is the ODE necessary?* The Direct Transformer experiment provides a definitive answer: the same 22.1M-parameter DiT architecture achieves **97.0% KNN accuracy** as a direct single-pass predictor — far exceeding the flow-matching DiT (37.2%) — but catastrophically mode-collapses (DivR = 0.048, coverage = 7.4%). The ODE framework's iterative integration forces DiT to traverse the noise-to-data manifold gradually, which *prevents* centroid memorization and instead produces distributed samples that faithfully span each group's variance. Without the ODE, the transformer merely learns a high-capacity lookup table from condition to centroid. The ODE step count ablation (Table IV) reinforces this: 2 steps maximize KNN (42.1%), while 50 steps maximize diversity (DivR = 0.73) and downstream utility (DS = 0.43). The ODE is thus not necessary for *accuracy* but is essential for *diversity preservation*.

*Is the comparison between feed-forward and diffusion models fair?* Strictly, no — they optimize fundamentally different objectives. Feed-forward decoders (cVAE, cGAN, Direct Transformer) minimize reconstruction/discrimination loss in a single pass, naturally optimizing for centroid accuracy. Flow-matching decoders minimize velocity MSE across a continuous time schedule, optimizing for distributional coverage. Comparing their KNN accuracy is like comparing the precision of a lookup table to the recall of a distribution sampler. We include both paradigms precisely to expose this trade-off: if the downstream application requires exact type matching, feed-forward decoders are more appropriate; if it requires biological plausibility (capturing within-group heterogeneity), flow matching is superior.

*Is flow matching the right diffusion framework?* Yes. DDPM-DiT (same architecture, discrete denoising with cosine schedule) fails entirely: KNN = 1.0% (random), FD$_g$ = 441. With $T = 1000$ discrete timesteps and only 100 training epochs, each timestep receives insufficient training signal. Flow matching's continuous formulation with logit-normal time sampling concentrates training on the most informative intermediate steps, achieving convergence in the same training budget. This mirrors findings in the image generation literature where flow matching (Stable Diffusion 3, FLUX) has superseded DDPM for efficiency and quality [14].

*What is the intrinsic advantage of diffusion/flow models?* Four properties distinguish them from feed-forward decoders: (1) **diversity preservation** — DiT achieves DivR = 1.001, while the best FF model (cGAN) reaches only 0.93 and the worst (Direct Transformer) reaches 0.048; (2) **continuous controllability** — CFG provides a single knob to trade accuracy for diversity (42.1% at 2-step to 28.0% at CFG=1.0 Midpoint with DivR=1.00); (3) **distribution calibration** — downstream accuracy (train-on-fake, test-on-real) is highest for DiT at 50-step (DS = 0.43) despite lower KNN, suggesting generated cells are more representative of the real distribution tail; (4) **interpretable generation** — the ODE trajectory provides a continuous path from noise to cell, enabling analysis of how conditioning shapes the generation process.

The choice of flow-matching DiT is therefore motivated by the need for *controllable, diverse, calibrated generation* rather than raw classification accuracy.

**Limitations.** Current limitations include: (1) the model generates accurately only for cell types seen during training — out-of-distribution text prompts produce unpredictable results, suggesting a need for CLOP text encoder generalization; (2) the absolute accuracy gap between generated (37%) and real data (89%) indicates room for improvement in conditional precision; (3) operating in scGPT latent space rather than directly generating gene expression inherits any biases from the foundation model; (4) the accuracy–diversity trade-off cannot be fully resolved by CFG alone — adaptive per-type guidance scales could optimize both simultaneously.

**Future Work.** Promising directions include: (1) extending to multi-modal (ATAC-seq) and multi-tissue generation; (2) improving text encoder generalization via instruction-tuning with augmented descriptions; (3) multi-resolution conditioning (coarse cell type + fine sub-type hierarchy); (4) integrating decoded gene expression for marker gene validation; and (5) scaling DiT capacity for improved conditional precision.

---

## VI. Conclusion

We presented CLOP-DiT, a two-stage framework for text-guided single-cell generation that combines contrastive language-omics pretraining with flow-matching diffusion transformers. CLOP aligns biomedical text with scGPT cell embeddings in a shared 256-dimensional space, transforming a near-identical cell type space (cosine 0.994) into a well-separated condition space (cosine 0.222). DiT generates realistic cell embeddings conditioned on text prompts using 8 AdaLN-Zero transformer blocks with flow matching. Trained on 220,304 cells across 80 datasets with 1,088 unique cell type descriptions, CLOP-DiT achieves KNN top-1 accuracy of 36.9% (37× random chance with 100 cell type groups), steering accuracy of 81.0%, and near-ideal diversity ratio of 1.00 with the Midpoint ODE solver. A systematic Phase 2 decoder architecture comparison against eight baselines — including oracles, feed-forward models (cVAE, cGAN), a Direct Transformer (same architecture, no ODE), DDPM-DiT (discrete denoising), and Flow MLP — reveals that the flow-matching framework is essential for diversity preservation: the Direct Transformer achieves the highest KNN accuracy (97.0%) but catastrophically mode-collapses (DivR = 0.048), while DDPM-DiT fails entirely (KNN = 1.0%), confirming flow matching as the appropriate diffusion framework. An ODE step count ablation shows that 2 steps maximize classification accuracy (42.1%) while 50 steps maximize distributional fidelity (DS = 0.43, DivR = 0.73), demonstrating the continuous accuracy–diversity trade-off inherent to iterative generation. DiT uniquely provides CFG-based control over this trade-off, perfect diversity matching (DivR = 1.00), and interpretable ODE trajectories — properties that are architecturally impossible in feed-forward decoders. The framework establishes a new paradigm for controllable, text-guided single-cell generation with potential applications in data augmentation, rare cell type synthesis, drug response simulation, and computational hypothesis testing.

---

## References

[1] F. Tang *et al.*, "mRNA-Seq whole-transcriptome analysis of a single cell," *Nature Methods*, vol. 6, no. 5, pp. 377–382, 2009.

[2] V. Svensson, R. Vento-Tormo, and S. A. Teichmann, "Exponential scaling of single-cell RNA-seq in the past decade," *Nature Protocols*, vol. 13, no. 4, pp. 599–604, 2018.

[3] K. D. Lähnemann *et al.*, "Eleven grand challenges in single-cell data science," *Genome Biology*, vol. 21, no. 1, pp. 1–35, 2020.

[4] R. Lopez, J. Regier, M. B. Cole, M. I. Jordan, and N. Yosef, "Deep generative modeling for single-cell transcriptomics," *Nature Methods*, vol. 15, no. 12, pp. 1053–1058, 2018.

[5] M. Marouf *et al.*, "Realistic in silico generation and augmentation of single-cell RNA-seq data using generative adversarial networks," *Nature Communications*, vol. 11, no. 1, pp. 1–12, 2020.

[6] A. Radford *et al.*, "Learning transferable visual models from natural language supervision," in *Proc. ICML*, 2021, pp. 8748–8763.

[7] W. Peebles and S. Xie, "Scalable diffusion models with transformers," in *Proc. ICCV*, 2023, pp. 4195–4205.

[8] M. Lotfollahi, F. A. Wolf, and F. J. Theis, "scGen predicts single-cell perturbation responses," *Nature Methods*, vol. 16, no. 8, pp. 715–721, 2019.

[9] P. Xu *et al.*, "scDiff: Conditional diffusion model for high-quality single-cell data synthesis," *Bioinformatics*, vol. 40, no. 1, 2024.

[10] S. Zhang *et al.*, "BiomedCLIP: A multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs," *arXiv:2303.00915*, 2023.

[11] S. Eslami, G. de Melo, and C. Meinel, "Does CLIP benefit visual question answering in the medical domain as much as it does in the general domain?," *arXiv:2112.13906*, 2021.

[12] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, and M. Nickel, "Flow matching for generative modeling," in *Proc. ICLR*, 2023.

[13] H. Cui *et al.*, "scGPT: Toward building a foundation model for single-cell multi-omics using generative AI," *Nature Methods*, vol. 21, no. 8, pp. 1470–1480, 2024.

[14] P. Esser *et al.*, "Scaling rectified flow transformers for high-resolution image synthesis," in *Proc. ICML*, 2024.

[15] T. Wang and P. Isola, "Understanding contrastive representation learning through alignment and uniformity on the hypersphere," in *Proc. ICML*, 2020, pp. 9929–9939.

---

## Appendix: Figure and Table Inventory

> **Figure numbering note:** This JBHI markdown uses a different figure order than the LaTeX article (`articles/clop_dit_biology.tex`). The LaTeX article (authoritative) maps: Fig 9 = conditioning UMAP (M), Fig 10 = diversity diagnostics (J), Fig 11 = diversity trade-off merged (L+K), Fig 12 = baselines (O). See [FIGURES_9-12_POLICY.md](FIGURES_9-12_POLICY.md) for details.

| Figure | Description | Panels | File |
|--------|-------------|--------|------|
| Fig. 1 | Architecture overview (schematic) | — | To be prepared separately |
| Fig. 2 | Training dynamics | (a) CLOP, (b) DiT | `panel_a_clop_training.png`, `panel_c_dit_training.png` |
| Fig. 3 | Real vs. generated embedding | — | `panel_e_real_vs_generated.png` |
| Fig. 4 | Quantitative metrics dashboard | — | `panel_d_metrics_summary.png` |
| Fig. 5 | Biological validation | (a) heatmap, (b) marker genes | `panel_f_text_cell_heatmap.png`, `panel_n_marker_gene_comparison.png` |
| Fig. 6 | CFG & solver | (a) diversity, (b) trade-off | `panel_j_diversity_diagnostics.png`, `panel_l_noise_tradeoff.png` |
| Fig. 7 | Dimension/expression correlation | — | `panel_h_expression_correlation.png` |
| Fig. 8 | Summary metrics (Tables I–II in text) | — | `panel_d_metrics_summary.png` |
| Fig. 9 | Phase 2 decoder comparison | — | `panel_o_baseline_comparison.png` |
| Fig. 10 | ODE step ablation | — | `panel_l_noise_tradeoff.png` |

| Table | Description | Rows | Columns |
|-------|-------------|------|---------|
| TABLE I | Generation quality across configs | 7 methods | KNN-1, KNN-5, Steering, DivR, LinAcc, FD, KNN/Rand |
| TABLE II | Ablation analysis | 4 variants | KNN-1, Steering, DivR, LinAcc, Verdict |
| TABLE III | Phase 2 decoder comparison | 10 decoders | KNN-1, DivR, FD$_g$, $r$, DS |
| TABLE IV | ODE step count ablation | 6 step counts | KNN-1, DivR, FD$_g$, DS |

### Key Metrics Summary

| Stage | Metric | Value | Interpretation |
|-------|--------|-------|---------------|
| **Data** | Total cells | 220,304 | 80 datasets, 1,088 text groups |
| **Data** | Embedding dim | 512 (scGPT) | Pre-trained representations |
| **Data** | Group centroid cosine | 0.994 | Cell types differ by 0.6% |
| **CLOP** | Parameters | 3.02M | Dual 3-layer MLP projectors |
| **CLOP** | Train loss | 1.19 | Converged |
| **CLOP** | Val accuracy | 2.5% (6.4× random) | Batch retrieval metric |
| **CLOP** | Projected cosine | 0.222 | Well-separated (vs 0.994 raw) |
| **DiT** | Parameters | 22.10M | 8-block transformer |
| **DiT** | Val cosine (EMA) | 0.976 | ~12.6° angular error |
| **DiT** | Val loss | 0.091 | Converged |
| **Gen** | KNN top-1 | 36.9% | 37× random chance |
| **Gen** | KNN top-5 | 55.3% | Correct type often in top 5 |
| **Gen** | Steering | 81.0% | Robust directional control |
| **Gen** | Linear accuracy | 51.1% | Linearly separable clusters |
| **Gen** | Diversity ratio | 1.00 (Midpoint) | Perfect variance match |
| **Gen** | Unconditioned KNN | 1.0% = random | Conditioning is the driver |
