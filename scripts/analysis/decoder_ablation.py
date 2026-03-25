#!/usr/bin/env python3
"""decoder_ablation.py — Run and record decoder ablation experiments.

Evaluates different decoder approaches on the SAME generated embeddings
(from DiT), recording expression-level metrics for comparison.

Approaches:
  A. scgpt_human (no fine-tuning)         — baseline generate() decoder
  B. scgpt_human + LoRA (rank 8, 2 layers) — light LoRA fine-tuning
  C. scgpt_human + LoRA (rank 16, 4 layers) — heavy LoRA fine-tuning
  D. MLP decoder (3-layer, trained)         — direct embedding→expression MLP

Usage:
    python scripts/analysis/decoder_ablation.py --approach baseline
    python scripts/analysis/decoder_ablation.py --approach lora_light
    python scripts/analysis/decoder_ablation.py --approach lora_heavy
    python scripts/analysis/decoder_ablation.py --approach mlp
    python scripts/analysis/decoder_ablation.py --compare  # print comparison table
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.paths import (
    CACHE_DIR, RESULTS_DIR, SCGPT_DIR, PROCESSED_H5AD_DIR, CHECKPOINT_DIR,
)
from src.utils.helpers import seed_everything, get_device

logger = logging.getLogger(__name__)

ABLATION_DIR = RESULTS_DIR / "ablations" / "decoder"

# Canonical marker genes per cell type (subset for quick evaluation)
MARKER_PANELS = {
    "T_cell": ["CD3D", "CD3E", "CD8A", "CD4", "IL7R"],
    "Macrophage": ["CD68", "CD14", "CSF1R", "MARCO", "MSR1"],
    "NK_cell": ["NKG7", "GNLY", "KLRD1", "NCAM1", "GZMB"],
    "B_cell": ["CD79A", "CD79B", "MS4A1", "CD19", "PAX5"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "FAP", "PDGFRA"],
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "FLT1"],
}


def compute_marker_specificity(expression, labels, gene_names, type_names=None):
    """Compute per-type marker specificity score.

    For each cell type, check if its canonical markers have higher mean
    expression than the global mean. Returns mean specificity ratio.
    """
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    results = {}

    for panel_name, markers in MARKER_PANELS.items():
        matched = [m for m in markers if m in gene_idx]
        if not matched:
            continue

        idxs = [gene_idx[m] for m in matched]
        global_mean = expression[:, idxs].mean(axis=0)

        # Find best matching cell type by highest marker expression
        best_ratio = 0
        for t in np.unique(labels):
            t_expr = expression[labels == t]
            if len(t_expr) < 5:
                continue
            t_mean = t_expr[:, idxs].mean(axis=0)
            ratio = (t_mean / (global_mean + 1e-8)).mean()
            best_ratio = max(best_ratio, ratio)

        results[panel_name] = {
            "n_markers_matched": len(matched),
            "best_specificity_ratio": float(best_ratio),
        }

    mean_spec = np.mean([v["best_specificity_ratio"] for v in results.values()])
    return {"per_panel": results, "mean_specificity": float(mean_spec)}


def compute_expression_metrics(real_expr, gen_expr, real_labels, gen_labels,
                                gene_names):
    """Compute comprehensive expression-level metrics."""
    from scipy.stats import pearsonr, spearmanr

    metrics = {}

    # Overall stats
    metrics["real_mean"] = float(real_expr.mean())
    metrics["gen_mean"] = float(gen_expr.mean())
    metrics["real_std"] = float(real_expr.std())
    metrics["gen_std"] = float(gen_expr.std())

    # Per-gene std (measures whether expression varies across cells)
    metrics["real_per_gene_std"] = float(real_expr.std(axis=0).mean())
    metrics["gen_per_gene_std"] = float(gen_expr.std(axis=0).mean())

    # Per-cell std (measures dynamic range within each cell)
    metrics["real_per_cell_std"] = float(real_expr.std(axis=1).mean())
    metrics["gen_per_cell_std"] = float(gen_expr.std(axis=1).mean())

    # Gene-level correlation (mean expression per gene)
    real_gene_means = real_expr.mean(axis=0)
    gen_gene_means = gen_expr.mean(axis=0)
    r, _ = pearsonr(real_gene_means, gen_gene_means)
    rho, _ = spearmanr(real_gene_means, gen_gene_means)
    metrics["gene_pearson_r"] = float(r)
    metrics["gene_spearman_rho"] = float(rho)

    # Per-type expression variance (key discriminative metric)
    type_means_real = []
    type_means_gen = []
    shared_types = set(np.unique(real_labels)) & set(np.unique(gen_labels))
    for t in sorted(shared_types):
        type_means_real.append(real_expr[real_labels == t].mean(axis=0))
        type_means_gen.append(gen_expr[gen_labels == t].mean(axis=0))

    if type_means_real:
        type_means_real = np.array(type_means_real)
        type_means_gen = np.array(type_means_gen)

        # Between-type variance (should be high if decoder differentiates types)
        metrics["real_between_type_var"] = float(type_means_real.var(axis=0).mean())
        metrics["gen_between_type_var"] = float(type_means_gen.var(axis=0).mean())
        metrics["between_type_var_ratio"] = float(
            metrics["gen_between_type_var"] / (metrics["real_between_type_var"] + 1e-10)
        )

    # Marker specificity
    marker_spec = compute_marker_specificity(gen_expr, gen_labels, gene_names)
    metrics["marker_specificity"] = marker_spec["mean_specificity"]
    metrics["marker_panels"] = marker_spec["per_panel"]

    return metrics


def decode_with_approach(approach, generated_emb, real_emb, gen_labels,
                         real_labels, gene_ids, gene_names, device,
                         batch_size=32):
    """Decode embeddings using the specified approach.

    Returns (gen_expression, real_expression) numpy arrays.
    """
    import torch
    from src.architecture.decoder import ScGPTDecoder

    if approach == "baseline":
        # scgpt_human, no LoRA
        decoder = ScGPTDecoder(model_dir=str(SCGPT_DIR), device=device,
                               batch_size=batch_size)
        gen_result = decoder.decode(generated_emb, gene_ids=gene_ids,
                                    gene_names=gene_names)
        real_result = decoder.decode(real_emb, gene_ids=gene_ids,
                                     gene_names=gene_names)

    elif approach == "lora_light":
        # scgpt_human + LoRA rank 8, last 2 layers (current checkpoint)
        decoder = ScGPTDecoder(model_dir=str(SCGPT_DIR), device=device,
                               batch_size=batch_size)
        lora_path = CHECKPOINT_DIR / "scgpt_lora_best.pth"
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
        decoder.load_lora_weights(lora_path)
        gen_result = decoder.decode(generated_emb, gene_ids=gene_ids,
                                    gene_names=gene_names)
        real_result = decoder.decode(real_emb, gene_ids=gene_ids,
                                     gene_names=gene_names)

    elif approach == "mlp":
        # MLP direct decoder (trained separately)
        mlp_path = CHECKPOINT_DIR / "mlp_decoder_best.pth"
        if not mlp_path.exists():
            raise FileNotFoundError(
                f"MLP decoder not found: {mlp_path}. "
                "Run: python scripts/training/train_mlp_decoder.py first."
            )
        ckpt = torch.load(mlp_path, map_location=device, weights_only=False)
        from src.architecture.decoder import LinearDecoder
        cfg = ckpt.get("config", {})
        num_genes = cfg.get("num_genes", 1790)
        hidden_dims = cfg.get("hidden_dims", [1024, 2048])
        mlp = LinearDecoder(embed_dim=cfg.get("embed_dim", 512),
                            num_genes=num_genes,
                            hidden_dims=hidden_dims,
                            output_activation=cfg.get("output_activation", "none"))
        mlp.load_state_dict(ckpt["model_state_dict"])
        mlp.to(device).eval()

        gen_t = torch.from_numpy(generated_emb).float().to(device)
        real_t = torch.from_numpy(real_emb).float().to(device)
        with torch.no_grad():
            gen_expr = mlp(gen_t).cpu().numpy()
            real_expr = mlp(real_t).cpu().numpy()
        gen_result = {"expression": gen_expr, "gene_names": gene_names}
        real_result = {"expression": real_expr, "gene_names": gene_names}

    else:
        raise ValueError(f"Unknown approach: {approach}")

    return gen_result["expression"], real_result["expression"]


def run_ablation(approach, n_real=2000, n_gen=2000, batch_size=32, seed=42):
    """Run a single decoder ablation and save results."""
    seed_everything(seed)
    device = get_device()
    cache = Path(CACHE_DIR)

    logger.info(f"=== Decoder Ablation: {approach} ===")

    # Load preprocessor
    from src.data_pipeline.embedding_preprocessor import EmbeddingPreprocessor
    prep_path = cache / "cell_preprocessor_preprocessed.npz"
    preprocessor = EmbeddingPreprocessor.load(str(prep_path))

    # Compute pre-norm scale (reuse decode_expression logic)
    raw_emb = np.load(cache / "cell_embeddings_dedup.npy")
    from scripts.analysis.decode_expression import compute_pre_norm_scale
    pre_norm_scale = compute_pre_norm_scale(preprocessor, raw_emb, n_samples=5000)
    del raw_emb

    # Load embeddings
    gen_emb = np.load(RESULTS_DIR / "generated_embeddings.npy")
    gen_labels = np.load(RESULTS_DIR / "generated_labels.npy")
    preprocessed = np.load(cache / "cell_embeddings_dedup_preprocessed.npy")
    real_labels = np.load(cache / "text_group_ids_dedup.npy")

    # Subsample
    rng = np.random.default_rng(seed)
    if n_gen < len(gen_emb):
        idx = rng.choice(len(gen_emb), n_gen, replace=False)
        gen_emb = gen_emb[idx]
        gen_labels = gen_labels[idx]
    if n_real < len(preprocessed):
        idx = rng.choice(len(preprocessed), n_real, replace=False)
        preprocessed = preprocessed[idx]
        real_labels = real_labels[idx]

    # Inverse transform to raw scGPT latent space (for scGPT-based decoders)
    gen_raw = preprocessor.inverse_transform(gen_emb, target_norm=pre_norm_scale)
    real_raw = preprocessor.inverse_transform(preprocessed, target_norm=pre_norm_scale)

    logger.info(f"  Gen raw norm: {np.linalg.norm(gen_raw, axis=1).mean():.2f}")
    logger.info(f"  Real raw norm: {np.linalg.norm(real_raw, axis=1).mean():.2f}")

    # For MLP, use preprocessed embeddings directly (that's what it was trained on)
    if approach == "mlp":
        decode_gen = gen_emb
        decode_real = preprocessed
    else:
        decode_gen = gen_raw
        decode_real = real_raw

    # Set up gene vocabulary
    h5ad_files = sorted(PROCESSED_H5AD_DIR.glob("*_processed.h5ad"))
    if not h5ad_files:
        raise FileNotFoundError(f"No h5ad files in {PROCESSED_H5AD_DIR}")

    if approach in ("baseline", "lora_light"):
        # Need scGPT gene ref from h5ad
        from src.architecture.decoder import ScGPTDecoder
        tmp_decoder = ScGPTDecoder(model_dir=str(SCGPT_DIR), device=device)
        tmp_decoder._load_encoder()
        import anndata as ad
        ref = ad.read_h5ad(h5ad_files[0])
        tmp_decoder._encoder.encode(ref[:2].copy())
        gene_ids = tmp_decoder._encoder._ref_gene_ids
        gene_names = tmp_decoder._encoder._ref_gene_names
        del tmp_decoder, ref
    elif approach == "mlp":
        import torch
        ckpt = torch.load(CHECKPOINT_DIR / "mlp_decoder_best.pth",
                          map_location="cpu", weights_only=False)
        gene_names = ckpt.get("gene_names", [f"gene_{i}" for i in range(
            ckpt["model_state_dict"]["net.4.weight"].shape[0])])
        gene_ids = None
        del ckpt

    # Decode
    t0 = time.time()
    gen_expr, real_expr = decode_with_approach(
        approach, decode_gen, decode_real, gen_labels, real_labels,
        gene_ids, gene_names, device, batch_size,
    )
    decode_time = time.time() - t0
    logger.info(f"  Decode time: {decode_time:.1f}s")

    # Compute metrics
    metrics = compute_expression_metrics(
        real_expr, gen_expr, real_labels, gen_labels, gene_names,
    )
    metrics["approach"] = approach
    metrics["decode_time_s"] = decode_time
    metrics["n_real"] = len(real_expr)
    metrics["n_gen"] = len(gen_expr)
    metrics["n_genes"] = len(gene_names)

    # Save
    out_dir = ABLATION_DIR / approach
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    np.save(out_dir / "gen_expression_sample.npy", gen_expr[:200])
    np.save(out_dir / "real_expression_sample.npy", real_expr[:200])

    logger.info(f"  Saved to {out_dir}")
    return metrics


def print_comparison():
    """Print comparison table of all recorded decoder ablations."""
    approaches = sorted(ABLATION_DIR.iterdir())
    if not approaches:
        print("No ablation results found. Run with --approach first.")
        return

    rows = []
    for d in approaches:
        mpath = d / "metrics.json"
        if not mpath.exists():
            continue
        m = json.load(open(mpath))
        rows.append(m)

    if not rows:
        print("No metrics.json files found.")
        return

    # Header
    cols = [
        ("Approach", "approach", "{}"),
        ("Gene r", "gene_pearson_r", "{:.4f}"),
        ("Gen mean", "gen_mean", "{:.2f}"),
        ("Gen gene σ", "gen_per_gene_std", "{:.4f}"),
        ("BtwType var ratio", "between_type_var_ratio", "{:.4f}"),
        ("Marker spec", "marker_specificity", "{:.4f}"),
        ("Time(s)", "decode_time_s", "{:.1f}"),
    ]

    header = " | ".join(f"{name:>18s}" for name, _, _ in cols)
    print(f"\n{'='*len(header)}")
    print("DECODER ABLATION COMPARISON")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for row in rows:
        line = " | ".join(
            f"{fmt.format(row.get(key, 'N/A')):>18s}" if isinstance(row.get(key), (int, float))
            else f"{'N/A':>18s}" if row.get(key) is None
            else f"{str(row.get(key, 'N/A')):>18s}"
            for _, key, fmt in cols
        )
        print(line)

    print(f"{'='*len(header)}\n")

    # Key insight
    best = max(rows, key=lambda r: r.get("between_type_var_ratio", 0))
    print(f"Best between-type variance ratio: {best['approach']} "
          f"({best.get('between_type_var_ratio', 0):.4f})")
    print(f"  → Higher = decoder better differentiates cell types in gene space")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Decoder ablation experiments")
    parser.add_argument("--approach", choices=["baseline", "lora_light",
                                                "lora_heavy", "mlp"],
                        help="Decoder approach to evaluate")
    parser.add_argument("--compare", action="store_true",
                        help="Print comparison table")
    parser.add_argument("--n-real", type=int, default=2000)
    parser.add_argument("--n-gen", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.compare:
        print_comparison()
        return

    if not args.approach:
        parser.error("Specify --approach or --compare")

    run_ablation(args.approach, args.n_real, args.n_gen,
                 args.batch_size, args.seed)


if __name__ == "__main__":
    main()
