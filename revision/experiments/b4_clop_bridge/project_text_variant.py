"""Project text embeddings through a specific CLOP ablation checkpoint.

Produces a projected_text.npy compatible with DiT training, in the same
format as the production projected_text.npy: (N_cells, 512) where
N_cells matches cell_embeddings_dedup_preprocessed.npy.

Usage:
    python project_text_variant.py --variant no_cohesion \
        --output revision/experiments/b4_clop_bridge/projected_text/projected_text_no_cohesion.npy
"""

from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = REPO_ROOT / "data" / "cached_latents"

VARIANT_CKPTS = {
    "production": REPO_ROOT / "models" / "checkpoints" / "clop_best.pth",
    "abl_baseline": REPO_ROOT / "models" / "checkpoints" / "ablations" / "baseline" / "epochs" / "clop_best.pth",
    "no_cohesion": REPO_ROOT / "models" / "checkpoints" / "ablations" / "no_cohesion" / "epochs" / "clop_best.pth",
    "no_cell_noise": REPO_ROOT / "models" / "checkpoints" / "ablations" / "no_cell_noise" / "epochs" / "clop_best.pth",
    "fixed_temperature": REPO_ROOT / "models" / "checkpoints" / "ablations" / "fixed_temperature" / "epochs" / "clop_best.pth",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANT_CKPTS.keys()))
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt_path = VARIANT_CKPTS[args.variant]
    logger.info(f"Variant: {args.variant}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Device: {device}")

    # Load model
    from src.architecture.clop.aligner import CLOPAligner

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    model = CLOPAligner(
        text_dim=1024, cell_dim=512, proj_dim=512,
        text_layers=3, cell_layers=3,
        loss_type="prototype_siglip",
    )
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    # Load unique preprocessed text embeddings
    for candidate in ["text_embeddings_unique_preprocessed.npy",
                      "text_embeddings_unique.npy",
                      "text_embeddings_dedup.npy"]:
        p = CACHE_DIR / candidate
        if p.exists():
            text_emb_unique = np.load(p)
            logger.info(f"Loaded {candidate}: {text_emb_unique.shape}")
            break
    else:
        raise FileNotFoundError(f"No unique text embedding file in {CACHE_DIR}")

    # Load group IDs to expand unique → per-cell
    gid_path = CACHE_DIR / "text_group_ids_dedup.npy"
    if not gid_path.exists():
        raise FileNotFoundError(f"Missing {gid_path}")
    text_group_ids = np.load(gid_path)
    logger.info(f"Group IDs: {text_group_ids.shape}, unique groups: {np.unique(text_group_ids).size}")

    # Project unique text embeddings
    projected_unique = []
    with torch.no_grad():
        for i in range(0, len(text_emb_unique), 512):
            batch = torch.from_numpy(text_emb_unique[i:i + 512]).float().to(device)
            proj = model.project_text(batch)
            projected_unique.append(proj.cpu().numpy())
    projected_unique = np.concatenate(projected_unique, axis=0)
    logger.info(f"Projected unique: {projected_unique.shape}")

    # Expand to per-cell using group IDs
    projected = projected_unique[text_group_ids]
    logger.info(f"Expanded to per-cell: {projected.shape}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, projected)
    logger.info(f"Saved: {output_path}")

    # Sanity check: compare shape to production
    prod_path = CACHE_DIR / "projected_text.npy"
    if prod_path.exists():
        prod = np.load(prod_path)
        logger.info(f"Production shape: {prod.shape}, this variant: {projected.shape}")
        assert projected.shape == prod.shape, \
            f"Shape mismatch: {projected.shape} vs production {prod.shape}"
        logger.info("Shape matches production. Ready for DiT training.")


if __name__ == "__main__":
    main()
