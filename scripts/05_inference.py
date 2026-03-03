#!/usr/bin/env python3
# 05_inference.py — Full CLOP-DiT inference pipeline v0.3
"""
End-to-end inference: Text → Condition → DiT Sampling → scGPT Decoding → Gene Expression

v0.3 improvements:
  - scGPT generate() decoding: cell embeddings → gene expression matrix
  - BiomedBERT-large (1024-d) text encoding
  - Single-cell-type prompts for cleaner generation
  - Output as proper AnnData with gene names

Usage:
    python scripts/05_inference.py \
        --prompt "CD8+ cytotoxic T cells from human lung adenocarcinoma" \
        --num_cells 500 \
        --output generated_cells.h5ad \
        --decode_expression
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import anndata as ad

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.architecture.dit import DiT1D
from src.architecture.clop import CLOPAligner
from src.architecture.decoder import ScGPTDecoder, LinearDecoder
from src.utils.helpers import seed_everything, get_device
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class CLOPDiTInference:
    """Full CLOP-DiT inference pipeline.

    v0.3: Supports scGPT generate() decoding for gene expression output.

    Parameters
    ----------
    dit_checkpoint : str
        Path to DiT model checkpoint.
    clop_checkpoint : str
        Path to CLOP aligner checkpoint.
    decoder_checkpoint : str, optional
        Path to LinearDecoder checkpoint (fallback).
    scgpt_model_dir : str, optional
        Path to scGPT model for generate() decoding.
    text_encoder_name : str
        HuggingFace text encoder model name.
    device : str
        Inference device.
    """

    def __init__(
        self,
        dit_checkpoint: str,
        clop_checkpoint: str,
        decoder_checkpoint: str = None,
        scgpt_model_dir: str = "models/scgpt_pancancer",
        text_encoder_name: str = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.text_encoder_name = text_encoder_name

        # Load CLOP
        logger.info("Loading CLOP aligner...")
        clop_ckpt = torch.load(clop_checkpoint, map_location=self.device)
        clop_config = clop_ckpt.get("config", {})
        self.clop = CLOPAligner(
            text_dim=clop_config.get("text_dim", 1024),
            cell_dim=clop_config.get("cell_dim", 512),
            proj_dim=clop_config.get("proj_dim", 256),
        )
        self.clop.load_state_dict(clop_ckpt["model_state_dict"])
        self.clop.to(self.device).eval()

        # Load DiT
        logger.info("Loading DiT model...")
        dit_ckpt = torch.load(dit_checkpoint, map_location=self.device)
        dit_config = dit_ckpt.get("config", {})
        self.dit = DiT1D(
            latent_dim=dit_config.get("latent_dim", 512),
            hidden_dim=dit_config.get("hidden_dim", 384),
            cond_dim=clop_config.get("proj_dim", 256),
            num_tokens=dit_config.get("num_tokens", 16),
        )
        # Prefer EMA weights if available
        if "ema_state_dict" in dit_ckpt:
            self.dit.load_state_dict(dit_ckpt["ema_state_dict"])
            logger.info("  Using EMA weights")
        else:
            self.dit.load_state_dict(dit_ckpt["model_state_dict"])
        self.dit.to(self.device).eval()

        # Load scGPT decoder (v0.3: primary decoder)
        self.scgpt_decoder = None
        if scgpt_model_dir and Path(scgpt_model_dir).exists():
            logger.info(f"Loading scGPT decoder from {scgpt_model_dir}")
            self.scgpt_decoder = ScGPTDecoder(
                model_dir=scgpt_model_dir,
                device=self.device,
            )

        # Load linear decoder (fallback)
        self.decoder = None
        if decoder_checkpoint and Path(decoder_checkpoint).exists():
            logger.info("Loading linear decoder (fallback)...")
            dec_ckpt = torch.load(decoder_checkpoint, map_location=self.device)
            self.decoder = LinearDecoder(**dec_ckpt.get("config", {}))
            self.decoder.load_state_dict(dec_ckpt["model_state_dict"])
            self.decoder.to(self.device).eval()

        # Load text encoder
        logger.info(f"Loading text encoder: {text_encoder_name}")
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.text_model = AutoModel.from_pretrained(text_encoder_name).to(self.device)
        self.text_model.eval()

        logger.info("Inference pipeline ready.")

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text description to condition vector.

        Parameters
        ----------
        text : str
            Biological description.

        Returns
        -------
        cond : (1, proj_dim) condition vector
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(self.device)

        outputs = self.text_model(**inputs)
        text_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]

        # Project through CLOP
        cond = self.clop.project_text(text_emb)
        return cond

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_cells: int = 100,
        num_steps: int = 4,
        cfg_scale: float = 3.0,
        solver: str = "euler",
        batch_size: int = 256,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate cell embeddings from text prompt.

        Parameters
        ----------
        prompt : str
            Biological description.
        num_cells : int
            Number of cells to generate.
        num_steps : int
            ODE integration steps.
        cfg_scale : float
            Classifier-free guidance scale.
        solver : str
            'euler' or 'midpoint'.
        batch_size : int
            Generation batch size.
        seed : int
            Random seed.

        Returns
        -------
        embeddings : (num_cells, latent_dim) generated cell embeddings
        """
        seed_everything(seed)

        # Encode text
        cond = self.encode_text(prompt)  # (1, proj_dim)
        cond = cond.expand(min(batch_size, num_cells), -1)  # (B, proj_dim)

        all_embeddings = []
        remaining = num_cells

        while remaining > 0:
            B = min(batch_size, remaining)
            cond_batch = cond[:B]

            if solver == "midpoint":
                emb = self.dit.sample_midpoint(cond_batch, num_steps=num_steps, cfg_scale=cfg_scale)
            else:
                emb = self.dit.sample(cond_batch, num_steps=num_steps, cfg_scale=cfg_scale)

            all_embeddings.append(emb.cpu().numpy())
            remaining -= B

        return np.concatenate(all_embeddings, axis=0)

    @torch.no_grad()
    def decode(self, embeddings: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Decode cell embeddings to expression matrix (LinearDecoder fallback).

        Parameters
        ----------
        embeddings : (N, latent_dim) cell embeddings
        batch_size : int

        Returns
        -------
        expression : (N, num_genes) gene expression matrix
        """
        if self.decoder is None:
            raise RuntimeError("No linear decoder loaded. Provide decoder_checkpoint.")

        all_expr = []
        for i in range(0, len(embeddings), batch_size):
            batch = torch.from_numpy(embeddings[i:i+batch_size]).float().to(self.device)
            expr = self.decoder(batch)
            all_expr.append(expr.cpu().numpy())

        return np.concatenate(all_expr, axis=0)

    @torch.no_grad()
    def decode_scgpt(
        self,
        embeddings: np.ndarray,
        reference_adata=None,
        gene_ids: np.ndarray = None,
        gene_names: list = None,
    ) -> dict:
        """Decode cell embeddings to gene expression via scGPT generate().

        v0.3: Uses proper scGPT reconstruction:
            cell_emb → inject at [CLS] → transformer → ExprDecoder → per-gene expression

        Parameters
        ----------
        embeddings : (N, 512) cell embeddings from DiT
        reference_adata : AnnData, optional
            Reference dataset to get gene vocabulary. If provided,
            will encode it first to set up reference gene set.
        gene_ids : (G,) optional gene token IDs
        gene_names : list of str, optional

        Returns
        -------
        result : dict with "expression" (N, G) and "gene_names" (list)
        """
        if self.scgpt_decoder is None:
            raise RuntimeError("No scGPT decoder loaded. Provide scgpt_model_dir.")

        # Set up reference gene set if needed
        if gene_ids is None and self.scgpt_decoder.get_reference_genes() is None:
            if reference_adata is not None:
                logger.info("Setting up reference gene set from provided adata...")
                self.scgpt_decoder.encode(reference_adata)
            else:
                raise ValueError(
                    "No reference gene set available. Provide reference_adata "
                    "or call with gene_ids/gene_names."
                )

        return self.scgpt_decoder.decode(
            cell_embeddings=embeddings,
            gene_ids=gene_ids,
            gene_names=gene_names,
        )

    def generate_adata(
        self,
        prompt: str,
        num_cells: int = 100,
        gene_names: list = None,
        decode_expression: bool = False,
        reference_adata=None,
        **kwargs,
    ) -> ad.AnnData:
        """Generate and return as AnnData object.

        Parameters
        ----------
        prompt : str
        num_cells : int
        gene_names : list of str, optional
        decode_expression : bool
            If True, decode embeddings to gene expression via scGPT generate().
        reference_adata : AnnData, optional
            Reference for scGPT gene vocabulary.
        **kwargs : passed to generate()

        Returns
        -------
        adata : AnnData
        """
        embeddings = self.generate(prompt, num_cells=num_cells, **kwargs)

        if decode_expression and self.scgpt_decoder is not None:
            # v0.3: scGPT generate() decoding
            result = self.decode_scgpt(
                embeddings,
                reference_adata=reference_adata,
                gene_names=gene_names,
            )
            adata = ad.AnnData(
                X=result["expression"],
                var={"gene_name": result["gene_names"]},
            )
            adata.var_names = result["gene_names"]
        elif self.decoder is not None:
            expression = self.decode(embeddings)
            var_names = gene_names or [f"Gene_{i}" for i in range(expression.shape[1])]
            adata = ad.AnnData(
                X=expression,
                var={"gene_name": var_names},
            )
        else:
            adata = ad.AnnData(X=embeddings)

        adata.obs["prompt"] = prompt
        adata.obs["generated"] = True
        adata.obsm["X_clop_dit"] = embeddings

        return adata


def main():
    parser = argparse.ArgumentParser(description="CLOP-DiT v0.3 Inference")
    parser.add_argument("--prompt", type=str, required=True, help="Biological description")
    parser.add_argument("--num_cells", type=int, default=500, help="Number of cells to generate")
    parser.add_argument("--dit_checkpoint", type=str, default="models/checkpoints/dit_best.pth")
    parser.add_argument("--clop_checkpoint", type=str, default="models/checkpoints/clop_best.pth")
    parser.add_argument("--decoder_checkpoint", type=str, default=None)
    parser.add_argument("--scgpt_model_dir", type=str, default="models/scgpt_pancancer",
                        help="scGPT model dir for generate() decoding")
    parser.add_argument("--text_encoder", type=str,
                        default="microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
    parser.add_argument("--output", type=str, default="generated_cells.h5ad")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "midpoint"])
    parser.add_argument("--decode_expression", action="store_true",
                        help="Decode embeddings to gene expression via scGPT generate()")
    parser.add_argument("--reference_h5ad", type=str, default=None,
                        help="Reference h5ad file for scGPT gene vocabulary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    setup_logging()

    pipeline = CLOPDiTInference(
        dit_checkpoint=args.dit_checkpoint,
        clop_checkpoint=args.clop_checkpoint,
        decoder_checkpoint=args.decoder_checkpoint,
        scgpt_model_dir=args.scgpt_model_dir,
        text_encoder_name=args.text_encoder,
        device=args.device,
    )

    logger.info(f"Generating {args.num_cells} cells for: '{args.prompt}'")

    # Load reference data for scGPT gene vocabulary
    reference_adata = None
    if args.decode_expression and args.reference_h5ad:
        import scanpy as sc
        logger.info(f"Loading reference data: {args.reference_h5ad}")
        reference_adata = sc.read_h5ad(args.reference_h5ad)

    adata = pipeline.generate_adata(
        prompt=args.prompt,
        num_cells=args.num_cells,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        solver=args.solver,
        seed=args.seed,
        decode_expression=args.decode_expression,
        reference_adata=reference_adata,
    )

    adata.write_h5ad(args.output)
    logger.info(f"Generated data saved to {args.output}")
    logger.info(f"Shape: {adata.shape}")
    if args.decode_expression:
        logger.info(f"Gene expression decoded via scGPT generate()")
        logger.info(f"Gene names: {adata.var_names[:5].tolist()} ... ({adata.shape[1]} total)")


if __name__ == "__main__":
    main()
