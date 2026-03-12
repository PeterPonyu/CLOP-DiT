# Cover Letter

**Date:** March 12, 2026

**To:** Editorial Office, *Biology*, MDPI

**From:**

Song Wang, Ph.D.
State Key Laboratory of Trauma and Chemical Poisoning,
Institute of Combined Injury,
Chongqing Engineering Research Center for Nanomedicine,
College of Preventive Medicine,
Army Medical University, Chongqing 400038, China
Email: swang1981@tmmu.edu.cn

---

Dear Editors,

We are pleased to submit our manuscript entitled **"CLOP-DiT: Structured-Metadata-Conditioned Single-Cell Latent Generation via Contrastive Language–Omics Pretraining and Diffusion Transformers"** for consideration as an Article in *Biology*.

## Motivation and Scope

The rapid growth of single-cell RNA sequencing has created an urgent need for computational methods that can generate realistic synthetic cell profiles from structured biological descriptions. While foundation models for single-cell biology are advancing rapidly, no existing method combines structured metadata conditioning with latent-space diffusion for controllable single-cell generation. Our work addresses this gap and falls squarely within the scope of *Biology*'s coverage of computational biology, bioinformatics, and systems biology.

## Summary of Contributions

CLOP-DiT is a two-stage pipeline that (1) aligns BiomedBERT text embeddings and scGPT cell embeddings in a shared space via contrastive learning (CLOP), then (2) trains a conditional Diffusion Transformer (DiT) with flow matching to generate cell-type-specific latent states from a five-field structured-metadata template (cell type, tissue, organism, marker genes, disease context). Evaluated across 69 cell types from 80 GEO datasets (220,304 cells), CLOP-DiT achieves 36.9% KNN accuracy (25× random chance) and 81.0% steering in a high-fidelity regime, and a diversity ratio of 0.93 in a high-diversity regime. Key findings include:

- **Conditioning field ablation and swap-label permutation tests** provide causal evidence that marker genes are the dominant steering signal.
- **External validation** against PanglaoDB and CellMarker 2.0 confirms biological plausibility of the conditioning markers.
- **Honest reporting of limitations:** a Gaussian baseline outperforms CLOP-DiT on common distributional metrics; generated cells lose within-type heterogeneity (per-gene variance correlation near zero); and a pilot rare-cell augmentation study yielded negative results.

## Novelty and Significance

To our knowledge, CLOP-DiT is the first method to combine contrastive language–omics pretraining with conditional flow-matching diffusion transformers for structured-metadata-conditioned single-cell latent generation. While we present this as a proof-of-concept rather than a mature data-augmentation tool, the pipeline establishes the feasibility of text-guided single-cell generation and identifies concrete next steps (variance-aware training, decoder replacement, out-of-distribution evaluation) for the community.

## Transparency

The manuscript provides full negative results alongside positive findings. All source code, trained checkpoints, configuration files, and figure reproduction scripts are publicly available on GitHub (<https://github.com/PeterPonyu/CLOP-DiT>) and will be archived on Zenodo upon acceptance.

## Confirmations

This manuscript has not been published and is not under consideration elsewhere. All authors have read and approved the submitted version. No conflicts of interest exist. The study used only publicly available, de-identified data from GEO; no ethical approval was required.

We believe this work will be of interest to the readership of *Biology*, particularly researchers working on generative models for single-cell genomics, foundation models in biology, and computational approaches to data augmentation. We look forward to the reviewers' feedback.

Sincerely,

Song Wang, Ph.D.
*Corresponding Author*
