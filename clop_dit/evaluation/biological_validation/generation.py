# generation.py — Text2Cell generation and real subset building.
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import anndata as ad

from . import constants

logger = logging.getLogger(__name__)


def build_prompts_from_subclusters(
    subcluster_meta: Dict,
    dataset_key: str,
    top_k: int = 4,
    min_conf: float = 0.25,
) -> Dict[str, str]:
    """Auto-build prompts from subcluster metadata."""
    if dataset_key not in subcluster_meta:
        return {}
    ds = subcluster_meta[dataset_key]
    dataset_text = ds.get("dataset_text", "")
    ct_counts = {}
    for cinfo in ds.get("clusters", {}).values():
        if cinfo.get("confidence", 0.0) < min_conf:
            continue
        ct = cinfo.get("cell_type", "Unknown")
        if ct == "Unknown":
            continue
        ct_counts[ct] = ct_counts.get(ct, 0) + int(cinfo.get("n_cells", 0))
    if not ct_counts:
        return {}
    top_types = sorted(ct_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    context = dataset_text.split(".")[0].strip() if dataset_text else "single-cell RNA sequencing"
    prompts = {}
    for ct, _ in top_types:
        prompts[ct] = f"{ct} from {context}."
    return prompts


def generate_cells_text2cell(adata_ref, args, prompts, dit_ckpt=None, scripts_dir: Path | None = None):
    """Generate cells using Text2Cell pipeline with a specific DiT checkpoint."""
    from .io import load_class_from_script

    scripts_dir = scripts_dir or Path(__file__).resolve().parent.parent.parent.parent / "scripts"
    CLOPDiTInference = load_class_from_script(
        scripts_dir / "05_inference.py", "CLOPDiTInference"
    )
    dit_path = dit_ckpt or args.dit_checkpoint
    pipeline = CLOPDiTInference(
        dit_checkpoint=dit_path,
        clop_checkpoint=args.clop_checkpoint,
        scgpt_model_dir=args.scgpt_model_dir,
        text_encoder_name=args.text_encoder,
        device=args.device,
    )
    fake_list = []
    for ct, prompt in prompts.items():
        adata_gen = pipeline.generate_adata(
            prompt=prompt,
            num_cells=args.num_cells_per_type,
            decode_expression=True,
            reference_adata=adata_ref,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
        )
        adata_gen.obs["cell_type"] = ct
        adata_gen.obs["source"] = "Generated"
        fake_list.append(adata_gen)
        logger.info(f"  Generated {adata_gen.n_obs} {ct} cells")
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return ad.concat(fake_list, join="outer", merge="same")


def build_real_subset(adata_ref, indices_map, n_per_type, prompts):
    rng = np.random.RandomState(42)
    real_list = []
    for ct in prompts:
        if ct not in indices_map:
            logger.warning(f"  Cell type '{ct}' absent, skipping")
            continue
        idx = indices_map[ct]
        if len(idx) > n_per_type:
            idx = rng.choice(idx, size=n_per_type, replace=False)
        a = adata_ref[idx].copy()
        a.obs["cell_type"] = ct
        a.obs["source"] = "Real"
        real_list.append(a)
    return ad.concat(real_list, join="outer", merge="same")
