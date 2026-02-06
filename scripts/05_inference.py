#!/usr/bin/env python3
# 05_inference.py — Full CLOP-DiT inference pipeline
"""
End-to-end inference: Text → Condition → DiT Sampling → Decoding → Expression Matrix

Usage:
    python scripts/05_inference.py \
        --prompt "Lung adenocarcinoma treated with cisplatin" \
        --num_cells 500 \
        --output generated_cells.h5ad
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
from src.architecture.decoder import LinearDecoder
from src.utils.helpers import seed_everything, get_device
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class CLOPDiTInference:
    """Full CLOP-DiT inference pipeline.

    Parameters
    ----------
    dit_checkpoint : str
        Path to DiT model checkpoint.
    clop_checkpoint : str
        Path to CLOP aligner checkpoint.
    decoder_checkpoint : str, optional
        Path to decoder checkpoint.
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
        text_encoder_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.text_encoder_name = text_encoder_name

        # Load CLOP
        logger.info("Loading CLOP aligner...")
        clop_ckpt = torch.load(clop_checkpoint, map_location=self.device)
        clop_config = clop_ckpt.get("config", {})
        self.clop = CLOPAligner(
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

        # Load decoder (optional)
        self.decoder = None
        if decoder_checkpoint and Path(decoder_checkpoint).exists():
            logger.info("Loading decoder...")
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
        """Decode cell embeddings to expression matrix.

        Parameters
        ----------
        embeddings : (N, latent_dim) cell embeddings
        batch_size : int

        Returns
        -------
        expression : (N, num_genes) gene expression matrix
        """
        if self.decoder is None:
            raise RuntimeError("No decoder loaded. Provide decoder_checkpoint.")

        all_expr = []
        for i in range(0, len(embeddings), batch_size):
            batch = torch.from_numpy(embeddings[i:i+batch_size]).float().to(self.device)
            expr = self.decoder(batch)
            all_expr.append(expr.cpu().numpy())

        return np.concatenate(all_expr, axis=0)

    def generate_adata(
        self,
        prompt: str,
        num_cells: int = 100,
        gene_names: list = None,
        **kwargs,
    ) -> ad.AnnData:
        """Generate and return as AnnData object.

        Parameters
        ----------
        prompt : str
        num_cells : int
        gene_names : list of str, optional
        **kwargs : passed to generate()

        Returns
        -------
        adata : AnnData
        """
        embeddings = self.generate(prompt, num_cells=num_cells, **kwargs)

        if self.decoder is not None:
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
    parser = argparse.ArgumentParser(description="CLOP-DiT Inference")
    parser.add_argument("--prompt", type=str, required=True, help="Biological description")
    parser.add_argument("--num_cells", type=int, default=500, help="Number of cells to generate")
    parser.add_argument("--dit_checkpoint", type=str, default="models/checkpoints/dit_best.pth")
    parser.add_argument("--clop_checkpoint", type=str, default="models/checkpoints/clop_best.pth")
    parser.add_argument("--decoder_checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="generated_cells.h5ad")
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "midpoint"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    setup_logging()

    pipeline = CLOPDiTInference(
        dit_checkpoint=args.dit_checkpoint,
        clop_checkpoint=args.clop_checkpoint,
        decoder_checkpoint=args.decoder_checkpoint,
        device=args.device,
    )

    logger.info(f"Generating {args.num_cells} cells for: '{args.prompt}'")

    adata = pipeline.generate_adata(
        prompt=args.prompt,
        num_cells=args.num_cells,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        solver=args.solver,
        seed=args.seed,
    )

    adata.write_h5ad(args.output)
    logger.info(f"Generated data saved to {args.output}")
    logger.info(f"Shape: {adata.shape}")


if __name__ == "__main__":
    main()
