#!/usr/bin/env python3
# 06_cell2cell_inference.py — Cell2Cell Latent Editing Inference Pipeline
"""
Cell→Cell inference: Given source cells + target condition, edit cells in latent space.

Modes:
    1. Edit mode: z_src → (add noise × edit_strength) → flow match to target → z_edited
    2. Text2Cell fallback: pure text→cell generation (drops source, uses learned null)

Analogy to Stable Diffusion:
    img2img:  source_image → VAE.encode → add noise(strength) → denoise(prompt) → VAE.decode → edited_image
    cell2cell: source_cell → scGPT.encode → add noise(strength) → flow(condition) → scGPT.decode → edited_cell

Usage:
    # Edit cells toward a target condition
    python scripts/06_cell2cell_inference.py \\
        --source_h5ad GSE123902.h5ad \\
        --target_prompt "CD8+ T cells after anti-PD1 treatment" \\
        --edit_strength 0.5 \\
        --output edited_cells.h5ad

    # Sweep edit strengths
    python scripts/06_cell2cell_inference.py \\
        --source_h5ad GSE123902.h5ad \\
        --target_prompt "Tumor-infiltrating macrophages" \\
        --strength_sweep 0.1,0.3,0.5,0.7,0.9 \\
        --output strength_sweep/

    # Text2Cell fallback (no source)
    python scripts/06_cell2cell_inference.py \\
        --target_prompt "CD4+ regulatory T cells" \\
        --mode text2cell \\
        --num_cells 500 \\
        --output treg_cells.h5ad
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.architecture.cell2cell import Cell2CellDiT
from src.architecture.clop import CLOPAligner
from src.architecture.decoder import ScGPTDecoder
from src.utils.helpers import seed_everything, get_device
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class Cell2CellInference:
    """Cell2Cell editing inference pipeline.

    Loads Cell2Cell model, CLOP aligner, text encoder, and scGPT decoder
    to perform end-to-end cell editing.

    Parameters
    ----------
    cell2cell_checkpoint : str
        Path to Cell2Cell model checkpoint.
    clop_checkpoint : str
        Path to CLOP aligner checkpoint.
    scgpt_model_dir : str
        Path to scGPT model for encoding/decoding.
    text_encoder_name : str
        HuggingFace model for text encoding.
    device : str
        Inference device.
    """

    def __init__(
        self,
        cell2cell_checkpoint: str,
        clop_checkpoint: str,
        scgpt_model_dir: str = "models/scgpt_pancancer",
        text_encoder_name: str = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        device: str = "cuda",
    ):
        self.device = torch.device(device)

        def cfg_value(config: dict, key: str, default):
            value = config.get(key, default)
            return default if value is None else value

        # --- Load CLOP ---
        logger.info("Loading CLOP aligner...")
        clop_ckpt = torch.load(clop_checkpoint, map_location=self.device, weights_only=False)
        clop_config = clop_ckpt.get("config", {})
        self.clop = CLOPAligner(
            text_dim=cfg_value(clop_config, "text_dim", 1024),
            cell_dim=cfg_value(clop_config, "cell_dim", 512),
            proj_dim=cfg_value(clop_config, "proj_dim", 256),
            text_hidden_dim=cfg_value(clop_config, "text_hidden_dim", None),
            cell_hidden_dim=cfg_value(clop_config, "cell_hidden_dim", None),
            text_layers=cfg_value(clop_config, "text_layers", 3),
            cell_layers=cfg_value(clop_config, "cell_layers", 3),
            dropout=cfg_value(clop_config, "dropout", 0.1),
            use_batch_norm=cfg_value(clop_config, "use_batch_norm", True),
            temperature=cfg_value(clop_config, "temperature", 0.07),
            min_temperature=cfg_value(clop_config, "min_temperature", 0.01),
            max_temperature=cfg_value(clop_config, "max_temperature", 0.5),
            label_smoothing=cfg_value(clop_config, "label_smoothing", 0.1),
            use_ema=cfg_value(clop_config, "use_ema", False),
            ema_decay=cfg_value(clop_config, "ema_decay", 0.999),
            use_soft_labels=cfg_value(clop_config, "use_soft_labels", False),
            soft_label_alpha=cfg_value(clop_config, "soft_label_alpha", 2.0),
            soft_label_bias=cfg_value(clop_config, "soft_label_bias", 5.0),
            cell_noise_std=cfg_value(clop_config, "cell_noise_std", 0.0),
            use_whitening=cfg_value(clop_config, "use_whitening", False),
            whitening_eps=cfg_value(clop_config, "whitening_eps", 1e-4),
            loss_type=cfg_value(clop_config, "loss_type", "infonce"),
            auto_duplicate_mask=cfg_value(clop_config, "auto_duplicate_mask", False),
            cohesion_weight=cfg_value(clop_config, "cohesion_weight", 0.1),
            temp_reg_weight=cfg_value(clop_config, "temp_reg_weight", 0.0),
        )
        self.clop.load_state_dict(clop_ckpt["model_state_dict"])
        self.clop.to(self.device).eval()

        # --- Load Cell2Cell ---
        logger.info("Loading Cell2Cell model...")
        c2c_ckpt = torch.load(cell2cell_checkpoint, map_location=self.device, weights_only=False)
        c2c_config = c2c_ckpt.get("config", {})
        self.model = Cell2CellDiT(
            latent_dim=c2c_config.get("latent_dim", 512),
            hidden_dim=c2c_config.get("hidden_dim", 384),
            cond_dim=clop_config.get("proj_dim", 256),
            num_tokens=c2c_config.get("num_tokens", 16),
        )
        # Prefer EMA weights
        if "ema_state_dict" in c2c_ckpt:
            self.model.load_state_dict(c2c_ckpt["ema_state_dict"])
            logger.info("  Using EMA weights")
        else:
            self.model.load_state_dict(c2c_ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        # --- Load scGPT ---
        self.scgpt_decoder = None
        if scgpt_model_dir and Path(scgpt_model_dir).exists():
            logger.info(f"Loading scGPT from {scgpt_model_dir}")
            self.scgpt_decoder = ScGPTDecoder(
                model_dir=scgpt_model_dir,
                device=self.device,
            )

        # --- Load text encoder ---
        logger.info(f"Loading text encoder: {text_encoder_name}")
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.text_model = AutoModel.from_pretrained(text_encoder_name).to(self.device)
        self.text_model.eval()

        logger.info("Cell2Cell inference pipeline ready.")

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text → CLOP-projected condition vector."""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(self.device)
        outputs = self.text_model(**inputs)
        text_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
        cond = self.clop.project_text(text_emb)
        return cond  # (1, proj_dim)

    @torch.no_grad()
    def encode_cells(self, adata: ad.AnnData) -> np.ndarray:
        """Encode source cells to scGPT embeddings.

        Parameters
        ----------
        adata : AnnData with raw counts

        Returns
        -------
        embeddings : (N, 512) scGPT cell embeddings
        """
        if self.scgpt_decoder is None:
            raise RuntimeError("scGPT decoder not loaded")

        # Use scGPT's encode_adata method
        embeddings = self.scgpt_decoder.encode(adata)
        return embeddings

    @torch.no_grad()
    def edit(
        self,
        source_embeddings: np.ndarray,
        target_prompt: str,
        edit_strength: float = 0.5,
        num_steps: int = 4,
        cfg_scale: float = 3.0,
        src_cfg_scale: float = 1.5,
        batch_size: int = 256,
        seed: int = 42,
    ) -> np.ndarray:
        """Edit source cells toward target condition.

        Parameters
        ----------
        source_embeddings : (N, 512) scGPT cell embeddings
        target_prompt : str
            Target biological state description.
        edit_strength : float
            ∈ [0, 1]. 0=identity, 1=full generation (ignores source).
        num_steps : int
            ODE integration steps.
        cfg_scale : float
            Condition guidance scale.
        src_cfg_scale : float
            Source identity preservation scale.
        batch_size : int
        seed : int

        Returns
        -------
        edited_embeddings : (N, 512)
        """
        seed_everything(seed)

        cond = self.encode_text(target_prompt)  # (1, proj_dim)
        N = len(source_embeddings)
        all_edited = []

        for i in range(0, N, batch_size):
            B = min(batch_size, N - i)
            z_src = torch.from_numpy(source_embeddings[i:i+B]).float().to(self.device)
            cond_batch = cond.expand(B, -1)

            z_edited = self.model.edit(
                z_src, cond_batch,
                edit_strength=edit_strength,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                src_cfg_scale=src_cfg_scale,
            )
            all_edited.append(z_edited.cpu().numpy())

        return np.concatenate(all_edited, axis=0)

    @torch.no_grad()
    def text2cell(
        self,
        prompt: str,
        num_cells: int = 100,
        num_steps: int = 4,
        cfg_scale: float = 3.0,
        batch_size: int = 256,
        seed: int = 42,
    ) -> np.ndarray:
        """Text-only generation (no source cells). Fallback mode."""
        seed_everything(seed)

        cond = self.encode_text(prompt)
        all_emb = []
        remaining = num_cells

        while remaining > 0:
            B = min(batch_size, remaining)
            cond_batch = cond.expand(B, -1)
            emb = self.model.sample_text2cell(
                cond_batch, num_steps=num_steps, cfg_scale=cfg_scale
            )
            all_emb.append(emb.cpu().numpy())
            remaining -= B

        return np.concatenate(all_emb, axis=0)

    def edit_adata(
        self,
        source_adata: ad.AnnData,
        target_prompt: str,
        edit_strength: float = 0.5,
        decode_expression: bool = True,
        reference_adata: Optional[ad.AnnData] = None,
        **kwargs,
    ) -> ad.AnnData:
        """End-to-end: source AnnData → edited AnnData.

        Pipeline:
            source_adata → scGPT encode → Cell2Cell edit → scGPT decode → edited_adata
        """
        # Step 1: Encode source cells
        logger.info(f"Encoding {source_adata.n_obs} source cells...")
        source_emb = self.encode_cells(source_adata)

        # Step 2: Edit in latent space
        logger.info(f"Editing with strength={edit_strength}, prompt='{target_prompt}'")
        edited_emb = self.edit(
            source_emb, target_prompt,
            edit_strength=edit_strength, **kwargs
        )

        # Step 3: Decode to expression
        if decode_expression and self.scgpt_decoder is not None:
            logger.info("Decoding edited embeddings via scGPT...")
            ref = reference_adata if reference_adata is not None else source_adata
            # Ensure reference genes are set
            if self.scgpt_decoder.get_reference_genes() is None:
                self.scgpt_decoder.encode(ref)

            result = self.scgpt_decoder.decode(edited_emb)
            adata = ad.AnnData(
                X=result["expression"],
                var={"gene_name": result["gene_names"]},
            )
            adata.var_names = result["gene_names"]
        else:
            adata = ad.AnnData(X=edited_emb)

        # Metadata
        adata.obs["source"] = "cell2cell_edit"
        adata.obs["target_prompt"] = target_prompt
        adata.obs["edit_strength"] = edit_strength
        adata.obsm["X_edited_emb"] = edited_emb
        adata.obsm["X_source_emb"] = source_emb

        # Copy over source metadata if available
        for col in ["cell_type", "tissue", "dataset"]:
            if col in source_adata.obs.columns:
                adata.obs[f"source_{col}"] = source_adata.obs[col].values[:len(adata)]

        return adata

    def edit_strength_sweep(
        self,
        source_adata: ad.AnnData,
        target_prompt: str,
        strengths: List[float] = None,
        decode_expression: bool = True,
        **kwargs,
    ) -> dict:
        """Run editing at multiple strengths, return {strength: adata} dict."""
        if strengths is None:
            strengths = [0.1, 0.3, 0.5, 0.7, 0.9]

        results = {}
        for s in strengths:
            logger.info(f"\n{'='*40} Edit strength = {s:.2f} {'='*40}")
            adata = self.edit_adata(
                source_adata, target_prompt,
                edit_strength=s,
                decode_expression=decode_expression,
                **kwargs,
            )
            results[s] = adata

            # Compute source↔edit distance
            src_emb = adata.obsm["X_source_emb"]
            edit_emb = adata.obsm["X_edited_emb"]
            cos = np.mean([
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                for a, b in zip(src_emb, edit_emb)
            ])
            l2 = np.mean(np.linalg.norm(edit_emb - src_emb, axis=1))
            logger.info(f"  s={s:.2f}: cos(src, edit)={cos:.4f}, L2={l2:.4f}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Cell2Cell Inference")
    parser.add_argument("--mode", type=str, default="edit",
                        choices=["edit", "text2cell", "sweep"],
                        help="Inference mode")
    parser.add_argument("--source_h5ad", type=str, default=None,
                        help="Source cells h5ad (required for edit/sweep modes)")
    parser.add_argument("--target_prompt", type=str, required=True,
                        help="Target biological state description")
    parser.add_argument("--edit_strength", type=float, default=0.5,
                        help="Edit strength ∈ [0, 1]")
    parser.add_argument("--strength_sweep", type=str, default="0.1,0.3,0.5,0.7,0.9",
                        help="Comma-separated strengths for sweep mode")
    parser.add_argument("--num_cells", type=int, default=500,
                        help="Number of cells for text2cell mode")
    parser.add_argument("--cell2cell_checkpoint", type=str,
                        default="models/checkpoints/cell2cell_best.pth")
    parser.add_argument("--clop_checkpoint", type=str,
                        default="models/checkpoints/clop_best.pth")
    parser.add_argument("--scgpt_model_dir", type=str,
                        default="models/scgpt_pancancer")
    parser.add_argument("--text_encoder", type=str,
                        default="microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
    parser.add_argument("--output", type=str, default="edited_cells.h5ad")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--src_cfg_scale", type=float, default=1.5)
    parser.add_argument("--decode_expression", action="store_true",
                        help="Decode to gene expression via scGPT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    setup_logging()

    pipeline = Cell2CellInference(
        cell2cell_checkpoint=args.cell2cell_checkpoint,
        clop_checkpoint=args.clop_checkpoint,
        scgpt_model_dir=args.scgpt_model_dir,
        text_encoder_name=args.text_encoder,
        device=args.device,
    )

    if args.mode == "text2cell":
        # Pure text→cell generation
        logger.info(f"Text2Cell: generating {args.num_cells} cells")
        embeddings = pipeline.text2cell(
            args.target_prompt,
            num_cells=args.num_cells,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
        )
        adata = ad.AnnData(X=embeddings)
        adata.obs["prompt"] = args.target_prompt
        adata.obs["mode"] = "text2cell"
        adata.obsm["X_cell2cell_emb"] = embeddings
        adata.write_h5ad(args.output)
        logger.info(f"Saved {adata.shape} to {args.output}")

    elif args.mode == "sweep":
        # Edit strength sweep
        import scanpy as sc
        source_adata = sc.read_h5ad(args.source_h5ad)
        strengths = [float(s) for s in args.strength_sweep.split(",")]

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = pipeline.edit_strength_sweep(
            source_adata, args.target_prompt,
            strengths=strengths,
            decode_expression=args.decode_expression,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            src_cfg_scale=args.src_cfg_scale,
            seed=args.seed,
        )

        for s, adata in results.items():
            out_path = output_dir / f"edited_s{s:.2f}.h5ad"
            adata.write_h5ad(str(out_path))
            logger.info(f"  s={s:.2f}: saved to {out_path}")

        # Save sweep summary
        summary = {
            "target_prompt": args.target_prompt,
            "source": args.source_h5ad,
            "strengths": strengths,
            "num_cells": source_adata.n_obs,
        }
        with open(output_dir / "sweep_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    else:
        # Standard edit
        import scanpy as sc
        source_adata = sc.read_h5ad(args.source_h5ad)
        logger.info(f"Loaded {source_adata.n_obs} source cells from {args.source_h5ad}")

        adata = pipeline.edit_adata(
            source_adata, args.target_prompt,
            edit_strength=args.edit_strength,
            decode_expression=args.decode_expression,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            src_cfg_scale=args.src_cfg_scale,
            seed=args.seed,
        )

        adata.write_h5ad(args.output)
        logger.info(f"Saved edited cells to {args.output}")
        logger.info(f"Shape: {adata.shape}")
        logger.info(f"Edit strength: {args.edit_strength}")
        logger.info(f"Target: {args.target_prompt}")


if __name__ == "__main__":
    main()
