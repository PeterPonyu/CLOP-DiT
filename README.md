# CLOP-DiT

> Text-conditioned single-cell latent generation via contrastive language–omics pretraining and diffusion transformers.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19584069.svg)](https://doi.org/10.5281/zenodo.19584069)

CLOP-DiT is a three-stage pipeline that samples single-cell expression embeddings conditioned on a structured five-field text prompt (cell type, tissue, organism, marker genes, disease context). The first stage is a prototype-aware contrastive aligner that maps frozen BiomedBERT text embeddings and frozen scGPT cell embeddings into a shared 512-dimensional latent space. The second stage is a 1-D Diffusion Transformer trained with conditional flow matching and classifier-free guidance, which samples a latent vector from a Gaussian prior toward the conditioned region of that space. The third stage is the frozen scGPT decoder, used to map the generated latent back to per-gene expression for downstream inspection.

The training corpus is 220,304 cells from 80 publicly available Gene Expression Omnibus (GEO) datasets, deduplicated to 69 evaluation cell types covering human and mouse tumour-microenvironment and developmental contexts.

For the full method, results, and limitations, see the manuscript:

> Zeyu Fu, JianXu Zheng, Jiawei Fu. *CLOP-DiT: Text-Conditioned Single-Cell Latent Generation via Contrastive Language–Omics Pretraining and Diffusion Transformers.* Submitted to *PeerJ Computer Science*, 2026.

## Reported results

The headline metrics from the manuscript, evaluated on 69 deduplicated cell types and reproduced here for convenience:

| Method | KNN-1 | Steering | DivR | LinAcc |
|---|---|---|---|---|
| Real data | 0.890 | – | 1.000 | 0.942 |
| CLOP-DiT, high-fidelity setting (CFG = 2.0) | 0.369 | 0.810 | 0.513 | 0.511 |
| CLOP-DiT, high-diversity setting (CFG = 1.0) | 0.288 | 0.807 | 0.929 | 0.357 |
| Embedding-VAE baseline | 0.112 | 0.547 | 0.744 | 0.189 |
| Gaussian baseline | 0.011 | 0.466 | 2.277 | 0.009 |

KNN-1 is reported over the 69-class problem with random chance ≈ 0.0145; CLOP-DiT at CFG = 2.0 is therefore about 25× above random. DivR ideal = 1.0.

The reported strength of CLOP-DiT is controllable text-conditioned generation. The reported limitations, also discussed in the manuscript, are that within-type variance and gene–gene correlation are only weakly preserved, and that a Gaussian mean-matching baseline outperforms CLOP-DiT on the nine shared distributional metrics. See the manuscript Discussion for the full set of caveats.

## Installation

The code targets Python 3.10. The full dependency list is in `requirements.txt`.

```bash
conda create -n clopdit python=3.10
conda activate clopdit
pip install -e .
```

The training and evaluation experiments were carried out with PyTorch 2.1, CUDA 12.0, scGPT v0.2.1, and Hugging Face Transformers 4.36 on a single NVIDIA RTX 5090 Laptop GPU.

## Generating cells from a text prompt

Once the trained checkpoints are placed under `models/`, a single text prompt can be sampled and decoded with the inference entry point:

```bash
python scripts/inference/05_inference.py \
    --prompt "CD8+ cytotoxic T cells from human lung adenocarcinoma" \
    --num_cells 500 --cfg_scale 2.0 --decode_expression \
    --output generated_cells.h5ad
```

Training the contrastive aligner and the diffusion transformer from cached embeddings:

```bash
python scripts/training/04a_train_clop.py --config configs/clop.yaml
python scripts/training/04b_train_dit.py --config configs/dit.yaml
```

## Data

The training and validation data are derived entirely from public Gene Expression Omnibus (GEO) records. The 80 GEO accession identifiers used in this study are listed in the manuscript appendix (Table S1), and the eight held-out validation studies are listed in Table S2. Each accession is resolvable at `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSExxxxxx`.

The deterministic preprocessing pipeline included in this repository (quality control, highly-variable-gene selection, scGPT encoding, study-level stratified split, and deduplication) can rebuild the analysis cache from those GEO records.

The trained model checkpoints (the CLOP aligner and the DiT generator) and the preprocessed embedding cache are not redistributed in this repository because of single-cell data licensing and file-size constraints. Both are available from the co-corresponding authors on reasonable request.

## Citation

```bibtex
@article{Fu2026CLOPDiT,
  author  = {Fu, Zeyu and Zheng, JianXu and Fu, Jiawei},
  title   = {{CLOP-DiT}: Text-Conditioned Single-Cell Latent Generation
             via Contrastive Language--Omics Pretraining and
             Diffusion Transformers},
  year    = {2026},
  note    = {Preprint; submitted to PeerJ Computer Science},
  doi     = {10.64898/2026.03.26.714457},
  url     = {https://doi.org/10.64898/2026.03.26.714457}
}
```

## License

Released under the MIT License — see `LICENSE`.

## Contact

Zeyu Fu — fuzeyu09@gmail.com
Jiawei Fu — fjw813130855@163.com
