"""A — Raw-HVG vs scGPT-latent within-type variance.

For each (dataset, cell_type) cluster in the cap3k_seed0 pipeline, we
compute a signal-to-noise style "tightness" statistic in two spaces:

  - raw HVG (adata.X, log1p-normalised, 2000-d)
  - scGPT latent (cached_latents, 512-d)

Tightness = mean within-cluster pairwise L2 / mean between-centroid L2
(both computed on the same set of cells, same dataset scope). A larger
value means clusters overlap more (more compressed). Comparing raw vs
latent tightness shows how much heterogeneity the encoder actually
preserves.

If latent tightness >> raw tightness across many (dataset, cell_type)
pairs -> scGPT latent compresses within-type variance relative to the
input signal, pinning down the encoder-stage bottleneck diagnosed in
the cap-increase experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scanpy as sc

HERE = Path(__file__).resolve().parent
# HERE = .../revision/experiments/encoder_bottleneck
# parents[0..2] = experiments, revision, <repo-root>
REPO_ROOT = HERE.parents[2]

H5AD_DIR = REPO_ROOT / "data" / "processed_h5ad_cap3k_seed0"
CACHE = REPO_ROOT / "data" / "cached_latents_cap3k_seed0"
SUB_META = H5AD_DIR / "subcluster_metadata.json"


def within_and_between(X: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Mean intra-cluster L2 (from cluster centroid), mean inter-centroid L2."""
    types = np.unique(labels)
    intra_vals = []
    centroids = []
    for t in types:
        M = X[labels == t]
        if M.shape[0] < 2:
            continue
        mu = M.mean(axis=0)
        centroids.append(mu)
        intra_vals.append(float(np.linalg.norm(M - mu, axis=1).mean()))
    if len(centroids) < 2:
        return (float(np.mean(intra_vals)) if intra_vals else 0.0, 0.0)
    C = np.stack(centroids)
    n_c = C.shape[0]
    inter = []
    for i in range(n_c):
        for j in range(i + 1, n_c):
            inter.append(float(np.linalg.norm(C[i] - C[j])))
    return (float(np.mean(intra_vals)),
            float(np.mean(inter)) if inter else 0.0)


def per_dataset_tightness(
    cells: np.ndarray, labels: np.ndarray
) -> dict:
    """Return {within, between, tightness=within/between}."""
    intra, inter = within_and_between(cells, labels)
    return {
        "mean_within_L2": intra,
        "mean_between_centroid_L2": inter,
        "tightness": intra / inter if inter > 0 else float("nan"),
        "n_types": int(len(np.unique(labels))),
        "n_cells": int(cells.shape[0]),
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)

    sub = json.loads(SUB_META.read_text())
    cells_latent = np.load(CACHE / "cell_embeddings.npy")
    sample_ids = np.load(CACHE / "sample_ids.npy", allow_pickle=True)
    proc = json.loads((CACHE / "processed_datasets.json").read_text())
    if isinstance(proc, list):
        order = [Path(p).stem.replace("_processed", "") for p in proc]
    else:
        order = list(proc.keys())
    ds_to_ord = {ds_id: i for i, ds_id in enumerate(order)}

    per_dataset = []
    for ds_id, ds_block in sub.items():
        if ds_id not in ds_to_ord:
            continue
        ord_k = ds_to_ord[ds_id]
        ds_mask = (sample_ids == ord_k)
        ds_latents = cells_latent[ds_mask]
        if ds_latents.shape[0] < 50:
            continue

        # Load the processed h5ad for raw HVG
        h5_path = H5AD_DIR / f"{ds_id}_processed.h5ad"
        if not h5_path.exists():
            continue
        try:
            adata = sc.read_h5ad(h5_path)
        except Exception:
            continue

        # Build cell -> cell_type mapping from subcluster metadata
        n_cells = int(adata.shape[0])
        ct_of_cell = np.full(n_cells, "Unknown", dtype=object)
        for cid, info in ds_block.get("clusters", {}).items():
            idx = np.asarray(info.get("cell_indices", []), dtype=int)
            valid = idx[idx < n_cells]
            ct_of_cell[valid] = info.get("cell_type", "Unknown")

        # Filter to cells with known cell_type and at least 2 types present
        known_mask = ct_of_cell != "Unknown"
        if known_mask.sum() < 20:
            continue
        types_present = np.unique(ct_of_cell[known_mask])
        if types_present.size < 2:
            continue

        # Raw HVG expression matrix
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X)
        raw_stats = per_dataset_tightness(
            X[known_mask], ct_of_cell[known_mask]
        )

        # Must align latent cell order to adata cell order. Latents were
        # computed in the same order cells were concatenated in the cache
        # build. For a given dataset, ds_latents is that dataset slice.
        # Crucially, the h5ad shape[0] should equal ds_latents.shape[0]
        # (we subsample BEFORE preprocess and cache in the same pass).
        if ds_latents.shape[0] != n_cells:
            # Fall back: skip if out of sync.
            continue
        lat_stats = per_dataset_tightness(
            ds_latents[known_mask], ct_of_cell[known_mask]
        )

        per_dataset.append({
            "dataset": ds_id,
            "n_cells": int(known_mask.sum()),
            "n_types": int(types_present.size),
            "raw": raw_stats,
            "latent": lat_stats,
            "within_L2_ratio_latent_over_raw": (
                lat_stats["mean_within_L2"] / raw_stats["mean_within_L2"]
                if raw_stats["mean_within_L2"] > 0 else float("nan")
            ),
            "between_L2_ratio_latent_over_raw": (
                lat_stats["mean_between_centroid_L2"]
                / raw_stats["mean_between_centroid_L2"]
                if raw_stats["mean_between_centroid_L2"] > 0 else float("nan")
            ),
            "tightness_ratio_latent_over_raw": (
                lat_stats["tightness"] / raw_stats["tightness"]
                if raw_stats["tightness"] > 0 else float("nan")
            ),
        })

    # Aggregate
    def stat(key: str):
        xs = [p[key] for p in per_dataset
              if np.isfinite(p.get(key, float("nan")))]
        if not xs:
            return None
        return {
            "median": round(float(np.median(xs)), 4),
            "mean": round(float(np.mean(xs)), 4),
            "IQR": [round(float(np.percentile(xs, 25)), 4),
                    round(float(np.percentile(xs, 75)), 4)],
            "min": round(float(min(xs)), 4),
            "max": round(float(max(xs)), 4),
            "n": int(len(xs)),
        }

    out = {
        "n_datasets_analyzed": len(per_dataset),
        "notes": {
            "raw_space": "adata.X (log1p-normalised HVG, 2000-d)",
            "latent_space": "scGPT 512-d cached_latents_cap3k_seed0",
            "tightness": "mean_within_L2 / mean_between_centroid_L2",
            "interpretation": "larger tightness = clusters overlap more",
        },
        "tightness_ratio_latent_over_raw": stat(
            "tightness_ratio_latent_over_raw"
        ),
        "within_L2_ratio": stat("within_L2_ratio_latent_over_raw"),
        "between_L2_ratio": stat("between_L2_ratio_latent_over_raw"),
        "per_dataset": sorted(
            per_dataset, key=lambda p: -p["tightness_ratio_latent_over_raw"]
        ),
    }

    (HERE / "raw_vs_latent_variance.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    t = out["tightness_ratio_latent_over_raw"]
    wL = out["within_L2_ratio"]
    bL = out["between_L2_ratio"]
    lines = [
        "A — Raw-HVG vs scGPT-latent within-type variance",
        "================================================",
        f"datasets: {out['n_datasets_analyzed']}",
        "",
        f"Tightness ratio (latent / raw):   "
        f"median={t['median']}  mean={t['mean']}  IQR={t['IQR']}",
        f"Within-type L2 ratio (latent/raw): "
        f"median={wL['median']}  IQR={wL['IQR']}",
        f"Between-type L2 ratio (latent/raw): "
        f"median={bL['median']}  IQR={bL['IQR']}",
        "",
        "Reading:",
        "  tightness_ratio > 1 : latent space makes clusters MORE mixed",
        "                        (i.e. encoder compresses within-type structure)",
        "  tightness_ratio < 1 : latent space tightens within-type",
        "                        distribution (encoder sharpens labels)",
        "",
        "Top 5 datasets where encoder compresses most heavily:",
    ]
    for p in out["per_dataset"][:5]:
        lines.append(
            f"  {p['dataset'][:40]:<40s} n={p['n_cells']:>5d} t_ratio="
            f"{p['tightness_ratio_latent_over_raw']:.3f}"
        )
    lines.append("")
    lines.append("Top 5 where encoder preserves/sharpens:")
    for p in out["per_dataset"][-5:]:
        lines.append(
            f"  {p['dataset'][:40]:<40s} n={p['n_cells']:>5d} t_ratio="
            f"{p['tightness_ratio_latent_over_raw']:.3f}"
        )
    preview = "\n".join(lines) + "\n"
    (HERE / "raw_vs_latent_variance_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
