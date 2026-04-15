"""B4 diagnostic: compare projected text spaces across CLOP ablation variants.

Before committing to multi-hour DiT retrains, this script checks whether
the CLOP ablation variants produce meaningfully different projected text
embeddings. If projections are nearly identical, ablation conclusions
transfer trivially (the DiT sees the same conditioning). If they diverge,
full DiT retraining per variant is needed.

Variants compared:
  - production: models/checkpoints/clop_best.pth (the checkpoint used for
    the existing projected_text.npy)
  - ablation/baseline: models/checkpoints/ablations/baseline/epochs/clop_best.pth
  - ablation/no_cohesion
  - ablation/no_cell_noise
  - ablation/fixed_temperature
"""

from __future__ import annotations

import json
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

# Variant name → checkpoint path
VARIANTS = {
    "production": REPO_ROOT / "models" / "checkpoints" / "clop_best.pth",
    "abl_baseline": REPO_ROOT / "models" / "checkpoints" / "ablations" / "baseline" / "epochs" / "clop_best.pth",
    "no_cohesion": REPO_ROOT / "models" / "checkpoints" / "ablations" / "no_cohesion" / "epochs" / "clop_best.pth",
    "no_cell_noise": REPO_ROOT / "models" / "checkpoints" / "ablations" / "no_cell_noise" / "epochs" / "clop_best.pth",
    "fixed_temperature": REPO_ROOT / "models" / "checkpoints" / "ablations" / "fixed_temperature" / "epochs" / "clop_best.pth",
}


def load_clop_model(checkpoint_path: Path, device: str = "cuda"):
    """Load a CLOPAligner from a checkpoint.

    All ablation variants share the same architecture (text_dim=1024,
    cell_dim=512, proj_dim=512, 3 layers each) — only training
    hyperparameters differed.
    """
    from src.architecture.clop.aligner import CLOPAligner

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    model = CLOPAligner(
        text_dim=1024,
        cell_dim=512,
        proj_dim=512,
        text_layers=3,
        cell_layers=3,
        loss_type="prototype_siglip",
    )
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def project_unique_text(model, text_emb: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Project unique text embeddings through a CLOP text projector."""
    projected = []
    with torch.no_grad():
        for i in range(0, len(text_emb), 512):
            batch = torch.from_numpy(text_emb[i:i + 512]).float().to(device)
            proj = model.project_text(batch)
            projected.append(proj.cpu().numpy())
    return np.concatenate(projected, axis=0)


def pairwise_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity between two matrices of same shape."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (A_norm * B_norm).sum(axis=1)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load unique preprocessed text embeddings (what project_and_save uses)
    text_path = CACHE_DIR / "text_embeddings_unique_preprocessed.npy"
    if not text_path.exists():
        text_path = CACHE_DIR / "text_embeddings_unique.npy"
    if not text_path.exists():
        text_path = CACHE_DIR / "text_embeddings_dedup.npy"
    text_emb = np.load(text_path)
    logger.info(f"Loaded text embeddings: {text_emb.shape} from {text_path.name}")

    # Also load production projected_text for reference
    prod_projected_path = CACHE_DIR / "projected_text.npy"
    prod_projected_ref = None
    if prod_projected_path.exists():
        prod_projected_full = np.load(prod_projected_path)
        # Get unique rows indexed by group_ids
        gid_path = CACHE_DIR / "text_group_ids_dedup.npy"
        if gid_path.exists():
            gids = np.load(gid_path)
            unique_gids = np.unique(gids)
            # Extract one projected vector per unique group
            prod_projected_ref = np.zeros((len(unique_gids), prod_projected_full.shape[1]))
            for i, g in enumerate(unique_gids):
                prod_projected_ref[i] = prod_projected_full[gids == g][0]
            logger.info(f"Production projected_text reference: {prod_projected_ref.shape}")

    # Project text through each variant
    projections = {}
    for name, ckpt_path in VARIANTS.items():
        if not ckpt_path.exists():
            logger.warning(f"  [SKIP] {name}: {ckpt_path} not found")
            continue
        logger.info(f"\nProjecting with {name}: {ckpt_path}")
        model = load_clop_model(ckpt_path, device)
        proj = project_unique_text(model, text_emb, device)
        projections[name] = proj
        logger.info(f"  Shape: {proj.shape}, L2 norm range: [{np.linalg.norm(proj, axis=1).min():.3f}, {np.linalg.norm(proj, axis=1).max():.3f}]")
        del model
        torch.cuda.empty_cache()

    if len(projections) < 2:
        logger.error("Need at least 2 variants to compare")
        return

    # Pairwise comparison
    names = list(projections.keys())
    results = {"n_text_groups": int(text_emb.shape[0]), "variants": names}
    comparisons = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            A, B = projections[a], projections[b]

            cos = pairwise_cosine(A, B)
            l2 = np.linalg.norm(A - B, axis=1)

            comp = {
                "pair": f"{a} vs {b}",
                "cosine_sim": {
                    "mean": round(float(cos.mean()), 6),
                    "median": round(float(np.median(cos)), 6),
                    "min": round(float(cos.min()), 6),
                    "std": round(float(cos.std()), 6),
                },
                "l2_dist": {
                    "mean": round(float(l2.mean()), 4),
                    "median": round(float(np.median(l2)), 4),
                    "max": round(float(l2.max()), 4),
                    "std": round(float(l2.std()), 4),
                },
            }
            comparisons.append(comp)
            logger.info(
                f"\n  {a} vs {b}:\n"
                f"    cosine_sim: mean={cos.mean():.4f}, min={cos.min():.4f}\n"
                f"    L2_dist:    mean={l2.mean():.4f}, max={l2.max():.4f}"
            )

    results["pairwise_comparisons"] = comparisons

    # Compare to production projected_text.npy if available
    # Note: prod_projected_ref has 69 unique groups, projections have 1088 sub-groups
    # Skip direct comparison (shape mismatch); structural comparison below is more meaningful

    # ── Structural comparison: do inter-type similarity patterns match? ──
    # This is the real question for DiT transferability: if type relationships
    # (which types are close/far) are preserved across variants, the DiT would
    # learn equivalent conditioning regardless of absolute coordinates.
    gid_path = CACHE_DIR / "text_group_ids_dedup.npy"
    structural_comparisons = []
    if gid_path.exists():
        gids = np.load(gid_path)
        unique_gids = np.unique(gids)
        n_groups = len(unique_gids)
        logger.info(f"\nStructural comparison: {n_groups} unique cell-type groups")

        # For each variant, compute group-level centroids and pairwise cosine matrix
        group_sim_matrices = {}
        for name, proj in projections.items():
            # Expand projections back to per-cell, then average per group
            # proj is (1088,512) for unique sub-clusters; collapse to (69,512) group centroids
            group_centroids = np.zeros((n_groups, proj.shape[1]))
            for i, g in enumerate(unique_gids):
                mask = gids == g
                # gids indexes into cell array, but proj is indexed by unique text position
                # Actually proj is (1088,512) — one per unique text embedding
                # We need to map text groups → sub-group text embeddings
                # gids maps cells → group, but we have unique text embeddings matching sub-clusters
                # Simplification: use the per-cell projected_text structure
                pass

            # Alternative: use the 1088 projected text vectors directly
            # Compute 1088×1088 cosine similarity matrix for each variant
            # Then compare matrix correlation across variants
            proj_norm = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8)
            sim_mat = proj_norm @ proj_norm.T  # (1088, 1088)
            # Extract upper triangle (excluding diagonal)
            triu_idx = np.triu_indices(sim_mat.shape[0], k=1)
            group_sim_matrices[name] = sim_mat[triu_idx]

        # Compare pairwise similarity structures (Pearson correlation of off-diagonal entries)
        from scipy.stats import pearsonr, spearmanr
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if a in group_sim_matrices and b in group_sim_matrices:
                    r_pearson, _ = pearsonr(group_sim_matrices[a], group_sim_matrices[b])
                    r_spearman, _ = spearmanr(group_sim_matrices[a], group_sim_matrices[b])
                    sc = {
                        "pair": f"{a} vs {b}",
                        "structure_pearson": round(float(r_pearson), 4),
                        "structure_spearman": round(float(r_spearman), 4),
                    }
                    structural_comparisons.append(sc)
                    logger.info(
                        f"  Structure {a} vs {b}: "
                        f"Pearson={r_pearson:.4f}, Spearman={r_spearman:.4f}"
                    )

    results["structural_comparisons"] = structural_comparisons

    # Per-variant summary: centroid spread
    centroid_stats = {}
    for name, proj in projections.items():
        centroid = proj.mean(axis=0)
        dists_from_centroid = np.linalg.norm(proj - centroid, axis=1)
        inter_cos = pairwise_cosine(proj[:-1], proj[1:])
        centroid_stats[name] = {
            "centroid_dist_mean": round(float(dists_from_centroid.mean()), 4),
            "centroid_dist_std": round(float(dists_from_centroid.std()), 4),
            "neighbor_cos_mean": round(float(inter_cos.mean()), 4),
        }
    results["per_variant_structure"] = centroid_stats

    # Verdict — based on STRUCTURAL similarity, not absolute cosine
    # Contrastive projectors have rotational freedom; what matters is
    # whether the inter-type similarity patterns are preserved.
    if structural_comparisons:
        min_struct_r = min(sc["structure_pearson"] for sc in structural_comparisons)
        mean_struct_r = np.mean([sc["structure_pearson"] for sc in structural_comparisons])
        if min_struct_r > 0.95:
            verdict = "STRUCTURALLY_EQUIVALENT"
            verdict_detail = (
                f"All pairwise structural Pearson r > 0.95 (min={min_struct_r:.4f}). "
                "Despite different absolute coordinates (expected from separate contrastive "
                "training runs), the inter-type similarity structure is preserved. "
                "A DiT trained on any variant would see the same type relationships. "
                "Full DiT retraining is unlikely to reveal new information."
            )
        elif min_struct_r > 0.80:
            verdict = "MOSTLY_EQUIVALENT"
            verdict_detail = (
                f"Structural Pearson r: min={min_struct_r:.4f}, mean={mean_struct_r:.4f}. "
                "Inter-type similarity structure is largely preserved across most variants "
                "but some divergence exists. DiT retraining on 1-2 key variants "
                "recommended to confirm."
            )
        else:
            verdict = "STRUCTURALLY_DIFFERENT"
            verdict_detail = (
                f"Structural Pearson r: min={min_struct_r:.4f}. "
                "Ablation variants produce different inter-type similarity patterns. "
                "Full DiT retraining per variant is needed for B4."
            )
    else:
        min_cos = min(c["cosine_sim"]["min"] for c in comparisons)
        verdict = "SUBSTANTIALLY_DIFFERENT"
        verdict_detail = f"No structural comparison available. Absolute min cosine = {min_cos:.4f}."

    results["verdict"] = verdict
    results["verdict_detail"] = verdict_detail
    logger.info(f"\n{'='*60}")
    logger.info(f"VERDICT: {verdict}")
    logger.info(verdict_detail)

    # Save
    (HERE / "projected_text_diagnostic.json").write_text(json.dumps(results, indent=2) + "\n")

    # Human-readable preview
    lines = [
        "B4 — Projected text space diagnostic",
        "=" * 50,
        f"Text groups: {text_emb.shape[0]}",
        f"Variants: {', '.join(names)}",
        "",
        "Pairwise cosine similarity (per-group, then aggregated):",
        f"{'pair':<40s} {'mean':>8s} {'median':>8s} {'min':>8s}",
        "-" * 64,
    ]
    for c in comparisons:
        cs = c["cosine_sim"]
        lines.append(f"{c['pair']:<40s} {cs['mean']:>8.4f} {cs['median']:>8.4f} {cs['min']:>8.4f}")

    lines.append("")
    lines.append("Pairwise L2 distance:")
    lines.append(f"{'pair':<40s} {'mean':>8s} {'median':>8s} {'max':>8s}")
    lines.append("-" * 64)
    for c in comparisons:
        ld = c["l2_dist"]
        lines.append(f"{c['pair']:<40s} {ld['mean']:>8.4f} {ld['median']:>8.4f} {ld['max']:>8.4f}")

    lines.append("")
    if structural_comparisons:
        lines.append("Structural similarity (Pearson r of inter-type similarity matrices):")
        lines.append(f"{'pair':<40s} {'pearson':>8s} {'spearman':>8s}")
        lines.append("-" * 56)
        for sc in structural_comparisons:
            lines.append(f"{sc['pair']:<40s} {sc['structure_pearson']:>8.4f} {sc['structure_spearman']:>8.4f}")

    lines.append("")
    lines.append(f"VERDICT: {verdict}")
    lines.append(verdict_detail)

    preview = "\n".join(lines) + "\n"
    (HERE / "projected_text_diagnostic_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
