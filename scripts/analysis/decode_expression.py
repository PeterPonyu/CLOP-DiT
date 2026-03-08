#!/usr/bin/env python3
"""decode_expression.py — Decode DiT-generated embeddings to gene expression via scGPT.

Critical pipeline step: proves the full CLOP-DiT → gene expression loop works.

Steps:
  1. Load generated cell embeddings from DiT (preprocessed space, L2-normalized)
  2. Inverse-transform to raw scGPT latent space (~norm 21)
  3. Decode via scGPT's generate() → per-gene expression values
  4. Do the same for a reference set of real cells
  5. Compare expression distributions (marker genes, correlations)

Usage:
    python scripts/decode_expression.py
    python scripts/decode_expression.py --n-real 2000 --n-gen 2000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.architecture.decoder import ScGPTDecoder
from src.utils.paths import CACHE_DIR, RESULTS_DIR, SCGPT_DIR, PROCESSED_H5AD_DIR
from src.data_pipeline.embedding_preprocessor import EmbeddingPreprocessor
from src.utils.helpers import seed_everything, get_device

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Known marker genes per cell type for validation
# ──────────────────────────────────────────────────────────────
MARKER_GENES = {
    "T_cell": ["CD3D", "CD3E", "CD3G", "CD2", "IL7R", "LEF1"],
    "CD8_T": ["CD8A", "CD8B", "GZMK", "GZMB", "PRF1", "NKG7"],
    "CD4_T": ["CD4", "IL7R", "CCR7", "SELL", "LEF1", "TCF7"],
    "B_cell": ["CD19", "MS4A1", "CD79A", "CD79B", "PAX5", "BANK1"],
    "NK_cell": ["GNLY", "NKG7", "KLRD1", "KLRB1", "NCR1", "GZMB"],
    "Monocyte": ["CD14", "LYZ", "S100A8", "S100A9", "FCGR3A", "CST3"],
    "Macrophage": ["CD68", "CD163", "MRC1", "MSR1", "MARCO", "C1QA"],
    "Dendritic": ["ITGAX", "FCER1A", "CLEC10A", "CD1C", "LILRA4", "IRF7"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "KDR", "ENG", "FLT1"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "FAP", "THY1"],
    "Epithelial": ["EPCAM", "KRT18", "KRT19", "CDH1", "MUC1", "KRT8"],
    "Hepatocyte": ["ALB", "APOA1", "SERPINA1", "TF", "HP", "FGB"],
    "Beta_cell": ["INS", "IAPP", "MAFA", "NKX6-1", "PDX1", "SLC30A8"],
    "Alpha_cell": ["GCG", "ARX", "IRX2", "TTR", "PCSK2", "GC"],
    "Stem_cell": ["CD34", "KIT", "THY1", "PROM1", "SOX2", "NANOG"],
    "Neutrophil": ["S100A8", "S100A9", "FCGR3B", "CSF3R", "CXCR2", "MMP9"],
    "Plasma_cell": ["SDC1", "MZB1", "JCHAIN", "XBP1", "IGHA1", "IGHG1"],
}

# Flatten to a unique ordered list
ALL_MARKERS = sorted(set(g for genes in MARKER_GENES.values() for g in genes))


def compute_pre_norm_scale(
    preprocessor: EmbeddingPreprocessor,
    raw_embeddings: np.ndarray,
    n_samples: int = 5000,
) -> float:
    """Compute the mean norm of whitened (pre-L2-normalized) embeddings.

    This is needed to invert the L2 normalization step during inverse_transform.
    """
    subset = raw_embeddings[:n_samples].astype(np.float32)
    centered = subset - preprocessor.mean_
    if preprocessor.whiten_matrix_ is not None:
        whitened = centered @ preprocessor.whiten_matrix_.T
    else:
        whitened = centered
    norms = np.linalg.norm(whitened, axis=1)
    scale = float(norms.mean())
    logger.info(f"Pre-norm scale: {scale:.4f} (std={norms.std():.4f})")
    return scale


def setup_gene_vocabulary(
    h5ad_path: str,
    scgpt_decoder: ScGPTDecoder,
) -> tuple:
    """Set up the gene vocabulary by encoding a reference h5ad file.

    Returns (gene_ids, gene_names).
    """
    import anndata as ad

    logger.info(f"Loading reference h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    logger.info(f"  Shape: {adata.shape}")

    # Encode to set up reference genes
    _ = scgpt_decoder.encode(adata)
    ref = scgpt_decoder.get_reference_genes()
    if ref is None:
        raise RuntimeError("Failed to set up reference gene set")

    logger.info(f"  Reference genes: {len(ref['gene_ids'])} "
                f"(from {adata.n_vars} total)")

    # Check marker gene coverage
    ref_set = set(ref["gene_names"])
    markers_found = [g for g in ALL_MARKERS if g in ref_set]
    logger.info(f"  Marker genes in reference: {len(markers_found)}/{len(ALL_MARKERS)}")

    return ref["gene_ids"], ref["gene_names"]


def decode_embeddings(
    embeddings: np.ndarray,
    preprocessor: EmbeddingPreprocessor,
    scgpt_decoder: ScGPTDecoder,
    pre_norm_scale: float,
    gene_ids: np.ndarray,
    gene_names: list,
    batch_size: int = 32,
    label: str = "cells",
) -> np.ndarray:
    """Inverse-transform and decode embeddings to gene expression.

    Parameters
    ----------
    embeddings : (N, 512) preprocessed embeddings (L2-normalized, whitened)
    preprocessor : fitted EmbeddingPreprocessor
    scgpt_decoder : loaded ScGPTDecoder
    pre_norm_scale : float, the scale factor for inverse_transform
    gene_ids, gene_names : reference gene set
    batch_size : int
    label : str, for logging

    Returns
    -------
    expression : (N, G) gene expression matrix
    """
    logger.info(f"Decoding {label}: {embeddings.shape[0]} cells × 512 dims")

    # Step 1: Inverse transform to raw scGPT latent space
    raw_emb = preprocessor.inverse_transform(embeddings, target_norm=pre_norm_scale)
    logger.info(f"  Inverse-transformed: norm_mean={np.linalg.norm(raw_emb, axis=1).mean():.2f}")

    # Step 2: Decode via scGPT
    result = scgpt_decoder.decode(
        cell_embeddings=raw_emb,
        gene_ids=gene_ids,
        gene_names=gene_names,
        batch_size=batch_size,
    )

    expression = result["expression"]
    logger.info(f"  Expression: {expression.shape}, "
                f"mean={expression.mean():.4f}, "
                f"nonzero_frac={np.mean(expression > 0.01):.3f}")

    return expression


def compute_expression_metrics(
    real_expr: np.ndarray,
    gen_expr: np.ndarray,
    gene_names: list,
    real_labels: np.ndarray = None,
    gen_labels: np.ndarray = None,
    type_names: dict = None,
) -> dict:
    """Compare real vs generated gene expression distributions.

    Returns comprehensive metrics dict.
    """
    metrics = {}

    # Overall statistics
    metrics["overall"] = {
        "real_mean": float(real_expr.mean()),
        "gen_mean": float(gen_expr.mean()),
        "real_nonzero_frac": float(np.mean(real_expr > 0.01)),
        "gen_nonzero_frac": float(np.mean(gen_expr > 0.01)),
        "real_std": float(real_expr.std()),
        "gen_std": float(gen_expr.std()),
    }

    # Per-gene correlation (real vs generated mean expression)
    real_gene_means = real_expr.mean(axis=0)
    gen_gene_means = gen_expr.mean(axis=0)
    valid = (real_gene_means > 0.001) | (gen_gene_means > 0.001)
    if valid.sum() > 0:
        from scipy.stats import pearsonr, spearmanr
        r, p = pearsonr(real_gene_means[valid], gen_gene_means[valid])
        rho, _ = spearmanr(real_gene_means[valid], gen_gene_means[valid])
        metrics["gene_correlation"] = {
            "pearson_r": float(r),
            "pearson_p": float(p),
            "spearman_rho": float(rho),
            "n_genes_compared": int(valid.sum()),
        }
        logger.info(f"Gene expression correlation: Pearson r={r:.4f}, Spearman ρ={rho:.4f}")

    # Marker gene analysis
    gene_name_set = {g: i for i, g in enumerate(gene_names)}
    marker_metrics = {}
    for category, markers in MARKER_GENES.items():
        cat_metrics = {}
        for gene in markers:
            if gene in gene_name_set:
                idx = gene_name_set[gene]
                real_vals = real_expr[:, idx]
                gen_vals = gen_expr[:, idx]
                cat_metrics[gene] = {
                    "real_mean": float(real_vals.mean()),
                    "gen_mean": float(gen_vals.mean()),
                    "real_nonzero_frac": float(np.mean(real_vals > 0.01)),
                    "gen_nonzero_frac": float(np.mean(gen_vals > 0.01)),
                    "mean_ratio": float(gen_vals.mean() / (real_vals.mean() + 1e-8)),
                }
        if cat_metrics:
            marker_metrics[category] = cat_metrics
    metrics["marker_genes"] = marker_metrics

    # Per-type expression fidelity (if labels provided)
    if real_labels is not None and gen_labels is not None and type_names is not None:
        type_fidelity = {}
        unique_types = np.unique(np.concatenate([np.unique(real_labels), np.unique(gen_labels)]))
        for t_id in unique_types:
            real_mask = real_labels == t_id
            gen_mask = gen_labels == t_id
            if real_mask.sum() < 3 or gen_mask.sum() < 3:
                continue
            real_t = real_expr[real_mask]
            gen_t = gen_expr[gen_mask]
            # Gene-level correlation for this type
            r_means = real_t.mean(axis=0)
            g_means = gen_t.mean(axis=0)
            v = (r_means > 0.001) | (g_means > 0.001)
            if v.sum() > 10:
                r, _ = pearsonr(r_means[v], g_means[v])
                name = type_names.get(int(t_id), f"Type_{t_id}")
                type_fidelity[name] = {
                    "pearson_r": float(r),
                    "n_real": int(real_mask.sum()),
                    "n_gen": int(gen_mask.sum()),
                }
        metrics["per_type_expression_fidelity"] = type_fidelity

        if type_fidelity:
            rs = [v["pearson_r"] for v in type_fidelity.values()]
            metrics["per_type_summary"] = {
                "mean_pearson_r": float(np.mean(rs)),
                "min_pearson_r": float(np.min(rs)),
                "max_pearson_r": float(np.max(rs)),
                "n_types": len(rs),
            }
            logger.info(f"Per-type expression Pearson r: mean={np.mean(rs):.4f}, "
                        f"min={np.min(rs):.4f}, max={np.max(rs):.4f}")

    return metrics


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Decode DiT embeddings to gene expression")
    parser.add_argument("--generated", default=str(RESULTS_DIR / "generated_embeddings.npy"))
    parser.add_argument("--gen-labels", default=str(RESULTS_DIR / "generated_labels.npy"))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--scgpt-dir", default=str(SCGPT_DIR))
    parser.add_argument("--h5ad-ref", default=None,
                        help="Reference h5ad for gene vocabulary (auto-detected if not set)")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--n-real", type=int, default=2000,
                        help="Number of real cells to decode for comparison")
    parser.add_argument("--n-gen", type=int, default=2000,
                        help="Number of generated cells to decode")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()

    cache = Path(args.cache_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load preprocessor ──
    logger.info("Loading embedding preprocessor...")
    preprocessor = EmbeddingPreprocessor.load(
        str(cache / "cell_preprocessor_preprocessed.npz")
    )

    # ── Load raw embeddings to compute pre-norm scale ──
    logger.info("Computing inverse-transform scale factor...")
    raw_emb = np.load(cache / "cell_embeddings_dedup.npy")
    pre_norm_scale = compute_pre_norm_scale(preprocessor, raw_emb, n_samples=5000)

    # ── Load scGPT decoder ──
    logger.info("Loading scGPT decoder...")
    scgpt = ScGPTDecoder(
        model_dir=args.scgpt_dir,
        device=device,
        batch_size=args.batch_size,
    )

    # ── Set up gene vocabulary from reference h5ad ──
    if args.h5ad_ref:
        ref_h5ad = args.h5ad_ref
    else:
        # Auto-detect: find an h5ad with good gene coverage
        h5ad_dir = PROCESSED_H5AD_DIR
        h5ad_files = sorted(h5ad_dir.glob("*_processed.h5ad"))
        if not h5ad_files:
            raise FileNotFoundError(f"No h5ad files in {h5ad_dir}")
        ref_h5ad = str(h5ad_files[0])
        logger.info(f"Auto-selected reference: {ref_h5ad}")

    gene_ids, gene_names = setup_gene_vocabulary(ref_h5ad, scgpt)

    # ── Prepare real cell subset (stratified by type) ──
    preprocessed = np.load(cache / "cell_embeddings_dedup_preprocessed.npy")
    group_ids = np.load(cache / "text_group_ids_dedup.npy")

    rng = np.random.default_rng(args.seed)
    unique_types = np.unique(group_ids)
    n_per_type = max(5, args.n_real // len(unique_types))
    real_idx = []
    for t in unique_types:
        t_idx = np.where(group_ids == t)[0]
        n = min(n_per_type, len(t_idx))
        real_idx.extend(rng.choice(t_idx, n, replace=False).tolist())
    real_idx = np.array(real_idx[:args.n_real])
    rng.shuffle(real_idx)

    real_pre = preprocessed[real_idx]
    real_labels = group_ids[real_idx]
    logger.info(f"Real cells: {len(real_idx)} (stratified from {len(unique_types)} types)")

    # ── Prepare generated cell subset ──
    gen_emb = np.load(args.generated)
    gen_labels = np.load(args.gen_labels) if Path(args.gen_labels).exists() else None
    n_gen = min(args.n_gen, len(gen_emb))
    gen_idx = rng.choice(len(gen_emb), n_gen, replace=False)
    gen_pre = gen_emb[gen_idx]
    gen_labels_sub = gen_labels[gen_idx] if gen_labels is not None else None
    logger.info(f"Generated cells: {n_gen}")

    # ── Decode real cells ──
    real_expr = decode_embeddings(
        real_pre, preprocessor, scgpt, pre_norm_scale,
        gene_ids, gene_names, args.batch_size, label="real cells",
    )

    # ── Decode generated cells ──
    gen_expr = decode_embeddings(
        gen_pre, preprocessor, scgpt, pre_norm_scale,
        gene_ids, gene_names, args.batch_size, label="generated cells",
    )

    # ── Save expression matrices ──
    np.save(out / "real_expression.npy", real_expr)
    np.save(out / "generated_expression.npy", gen_expr)
    np.save(out / "real_expression_labels.npy", real_labels)
    if gen_labels_sub is not None:
        np.save(out / "generated_expression_labels.npy", gen_labels_sub)
    with open(out / "expression_gene_names.json", "w") as f:
        json.dump(gene_names, f)
    logger.info(f"Saved expression matrices to {out}/")

    # ── Load type names ──
    cap_path = cache / "text_captions_deduplicated.json"
    type_names = {}
    if cap_path.exists():
        with open(cap_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            name = v.split(" are ")[0] if " are " in v else v[:50]
            type_names[int(k)] = name

    # ── Compute metrics ──
    metrics = compute_expression_metrics(
        real_expr, gen_expr, gene_names,
        real_labels=real_labels,
        gen_labels=gen_labels_sub,
        type_names=type_names,
    )

    with open(out / "expression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("Expression Decoding Summary")
    print(f"{'='*60}")
    print(f"Real:      {real_expr.shape} (mean={real_expr.mean():.4f})")
    print(f"Generated: {gen_expr.shape} (mean={gen_expr.mean():.4f})")
    print(f"Genes:     {len(gene_names)}")
    gc = metrics.get("gene_correlation", {})
    print(f"Gene correlation: Pearson r={gc.get('pearson_r', 'N/A')}, "
          f"Spearman ρ={gc.get('spearman_rho', 'N/A')}")
    pts = metrics.get("per_type_summary", {})
    if pts:
        print(f"Per-type expression r: mean={pts['mean_pearson_r']:.4f}, "
              f"min={pts['min_pearson_r']:.4f}, max={pts['max_pearson_r']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
