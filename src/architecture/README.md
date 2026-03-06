# src/architecture/

Neural network modules for the CLOP-DiT framework. The pipeline chains these components: text encoder (external BiomedBERT) and cell encoder (scGPT) feed into CLOP alignment, which conditions the DiT generative model, whose outputs are decoded back to gene expression via the scGPT decoder.

## Modules

### `clop/` -- Contrastive Language-Omics Pre-training

The CLOP aligner learns a shared 512-d embedding space between text descriptions and scGPT cell embeddings.

- `aligner.py` -- `CLOPAligner`: paired projection heads, temperature-scaled logits, and inference methods (text-to-cell retrieval, embedding projection).
- `losses.py` -- Loss functions and transforms:
  - `PrototypeSigLIPLoss` -- Primary loss: averages embeddings sharing the same text into prototypes before computing SigLIP pairwise sigmoid loss.
  - `SigLIPLoss` -- Standard sigmoid contrastive loss.
  - `InfoNCELoss` -- Classic softmax contrastive loss (with label smoothing).
  - `TextWhiteningTransform` -- ZCA whitening for text embeddings.
  - `ProjectionHead` -- Multi-layer MLP projector with LayerNorm and dropout.

### `dit.py` -- Diffusion Transformer

`DiT1D`: 1D flow-matching diffusion transformer for generating 512-d cell embeddings.

- AdaLN-Zero conditioning (adaptive LayerNorm with zero-init gating).
- Sinusoidal timestep embedding + condition vector from CLOP projections.
- 8 transformer blocks, 8 attention heads, logit-normal time sampling.
- Velocity prediction: `v(z_t, t, c)` with classifier-free guidance at inference.
- ODE sampling (Euler) for deterministic generation.

### `decoder.py` -- scGPT Expression Decoder

`ScGPTDecoder`: converts generated 512-d cell embeddings back to gene expression vectors (~1890 genes) using pretrained scGPT weights. Injects the cell embedding at the [CLS] position and runs the full transformer encoder + expression decoder head.

### `cell2cell.py` -- Cell-to-Cell Translation

`Cell2CellDiT`: conditional flow matching for intra-modal cell editing (analogous to image img2img). Injects source cell embeddings via AdaLN conditioning alongside timestep, text, and edit-strength signals. Use cases include perturbation prediction and trajectory editing.

### `scgpt_embed.py` -- Standalone scGPT Encoder

`ScGPTCellEncoder`: standalone cell embedding extractor that loads scGPT weights directly (no dependency on the `scgpt` pip package). Reads `best_model.pt` and `vocab.json` from the model directory and produces (N, 512) embeddings from AnnData objects.
