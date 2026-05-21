"""Within-cluster latent variance: cap3k vs cap10k (seed=0, apples-to-apples).

The first-order per-dataset variance comparison mixes real biological
heterogeneity with cluster proportions. The purer test is: for each
subcluster (as defined by signature-based annotation), compute its
total latent variance at cap3k and at cap10k, pair by dataset +
cell-type label, and compare.

A cap10k / cap3k variance ratio > 1 means the higher cap is exposing
within-type heterogeneity that the cap3k subsample was compressing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

C3_CACHE = REPO_ROOT / "data" / "cached_latents_cap3k_seed0"
C10_CACHE = REPO_ROOT / "data" / "cached_latents_cap10k"

# Subcluster metadata files (one per cap tier)
C3_SUB = REPO_ROOT / "data" / "processed_h5ad_cap3k_seed0" / "subcluster_metadata.json"
C10_SUB = REPO_ROOT / "data" / "processed_h5ad_cap10k" / "subcluster_metadata.json"


def load_cache(cache: Path):
    cells = np.load(cache / "cell_embeddings.npy")
    gids = np.load(cache / "text_group_ids.npy")
    sample_ids = np.load(cache / "sample_ids.npy", allow_pickle=True)
    meta = json.loads((cache / "metadata.json").read_text())
    # processed_datasets lists dataset_ids in order; sample_ids index into it
    proc = json.loads((cache / "processed_datasets.json").read_text())
    return {
        "cells": cells,
        "gids": gids,
        "sample_ids": sample_ids,
        "meta": meta,
        "proc": proc,
    }


def collect_cluster_variances(cache, sub_path: Path) -> dict:
    """For each dataset -> cluster, compute total latent variance across cells.

    We need a cell-index -> cluster mapping. The subcluster_metadata
    stores cluster -> list of cell indices (in-dataset ordering). The
    cache stores cells in concatenated dataset order; the boundaries
    are implicit in sample_ids. We reconstruct the per-dataset slice
    and then pick the cells listed by the subcluster metadata.
    """
    sub = json.loads(sub_path.read_text())
    cells = cache["cells"]
    sample_ids = cache["sample_ids"]
    # sample_ids are integer dataset ordinals (0..N-1)
    # proc is list of dataset_ids or similar; need to map dataset name -> ordinal
    proc = cache["proc"]
    # proc may be list of strings like ".../dataset_processed.h5ad" OR dict.
    if isinstance(proc, dict):
        order = list(proc.keys())
    elif isinstance(proc, list):
        order = [Path(p).stem.replace("_processed", "") for p in proc]
    else:
        raise RuntimeError("cannot parse proc")

    # Map dataset_id -> ordinal
    ds_to_ord = {ds_id: i for i, ds_id in enumerate(order)}

    # Build per-dataset cell slice boundaries from sample_ids order
    # sample_ids corresponds one-to-one with cells (in the same order cells were concatenated)
    # so for a given dataset ordinal k, cells[sample_ids == k] is that dataset's cells
    # and the internal order is the same as in the h5ad reading (subcluster cell_indices
    # refer to that internal order).
    out = {}
    for ds_id, ds_block in sub.items():
        if ds_id not in ds_to_ord:
            continue
        ord_k = ds_to_ord[ds_id]
        ds_mask = (sample_ids == ord_k)
        ds_cells = cells[ds_mask]
        if ds_cells.shape[0] == 0:
            continue
        per_cluster = {}
        for cid, info in ds_block.get("clusters", {}).items():
            cell_idx = info.get("cell_indices", [])
            if len(cell_idx) < 10:
                continue
            # Indices are within-dataset 0..n-1, but QC/HVG may have dropped cells
            # and subcluster_metadata was computed AFTER preprocessing, so these indices
            # should align to ds_cells.
            cell_idx = np.asarray(cell_idx, dtype=int)
            valid = cell_idx[cell_idx < ds_cells.shape[0]]
            if valid.size < 10:
                continue
            X = ds_cells[valid]
            mu = X.mean(axis=0)
            per_cluster[cid] = {
                "cell_type": info.get("cell_type", "Unknown"),
                "confidence": info.get("confidence", 0.0),
                "n": int(X.shape[0]),
                "total_variance": float(X.var(axis=0).sum()),
                "mean_L2_from_centroid": float(
                    np.linalg.norm(X - mu, axis=1).mean()
                ),
            }
        if per_cluster:
            out[ds_id] = per_cluster
    return out


def main() -> None:
    if not C3_SUB.exists() or not C10_SUB.exists():
        print("Missing subcluster metadata. Paths checked:")
        print(f"  {C3_SUB}")
        print(f"  {C10_SUB}")
        return
    c3 = load_cache(C3_CACHE)
    c10 = load_cache(C10_CACHE)
    v3 = collect_cluster_variances(c3, C3_SUB)
    v10 = collect_cluster_variances(c10, C10_SUB)

    # Pair by (dataset, cell_type).
    # Cluster ids are not stable across runs; we pair by cell_type label within a
    # dataset, aggregating multiple clusters of the same type into one pool.
    def aggregate_by_type(per_ds_clusters: dict) -> dict:
        out = defaultdict(lambda: {"n": 0, "total_variance": 0.0, "mean_L2": 0.0})
        for ds, clusters in per_ds_clusters.items():
            type_pool = defaultdict(list)
            for cid, c in clusters.items():
                type_pool[c["cell_type"]].append(c)
            for ct, entries in type_pool.items():
                # Aggregate: weighted mean of variance (weighted by n)
                n_total = sum(e["n"] for e in entries)
                if n_total == 0:
                    continue
                var_w = sum(e["n"] * e["total_variance"] for e in entries) / n_total
                l2_w = sum(e["n"] * e["mean_L2_from_centroid"]
                           for e in entries) / n_total
                out[(ds, ct)] = {
                    "n": n_total,
                    "total_variance": var_w,
                    "mean_L2": l2_w,
                }
        return out

    a3 = aggregate_by_type(v3)
    a10 = aggregate_by_type(v10)

    common = set(a3.keys()) & set(a10.keys())
    pairs = []
    for key in common:
        ds, ct = key
        x = a3[key]; y = a10[key]
        if x["n"] < 20 or y["n"] < 20 or ct == "Unknown":
            continue
        pairs.append({
            "dataset": ds,
            "cell_type": ct,
            "cap3k_n": x["n"],
            "cap10k_n": y["n"],
            "cap3k_var": round(x["total_variance"], 3),
            "cap10k_var": round(y["total_variance"], 3),
            "var_ratio": round(
                y["total_variance"] / max(x["total_variance"], 1e-9), 3
            ),
            "cap3k_L2": round(x["mean_L2"], 4),
            "cap10k_L2": round(y["mean_L2"], 4),
            "L2_ratio": round(
                y["mean_L2"] / max(x["mean_L2"], 1e-9), 3
            ),
        })
    pairs.sort(key=lambda p: -p["var_ratio"])
    ratios = [p["var_ratio"] for p in pairs]
    L2r = [p["L2_ratio"] for p in pairs]

    out = {
        "n_paired_type_in_dataset": len(pairs),
        "n_total_cap3k_cluster_types": sum(
            len({c["cell_type"] for c in d.values()}) for d in v3.values()
        ),
        "n_total_cap10k_cluster_types": sum(
            len({c["cell_type"] for c in d.values()}) for d in v10.values()
        ),
        "var_ratio_cap10k_over_cap3k": {
            "median": round(float(np.median(ratios)), 3) if ratios else None,
            "mean": round(float(np.mean(ratios)), 3) if ratios else None,
            "p25_p75": [round(float(np.percentile(ratios, 25)), 3),
                        round(float(np.percentile(ratios, 75)), 3)]
                        if ratios else None,
            "n_gt_1": int(sum(1 for r in ratios if r > 1)),
            "n_gt_1_5": int(sum(1 for r in ratios if r > 1.5)),
        },
        "L2_ratio_cap10k_over_cap3k": {
            "median": round(float(np.median(L2r)), 3) if L2r else None,
            "mean": round(float(np.mean(L2r)), 3) if L2r else None,
            "p25_p75": [round(float(np.percentile(L2r, 25)), 3),
                        round(float(np.percentile(L2r, 75)), 3)]
                        if L2r else None,
            "n_gt_1": int(sum(1 for r in L2r if r > 1)),
        },
        "top_gains": pairs[:15],
        "top_losses": pairs[-10:],
    }
    (HERE / "within_cluster_variance.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    v = out["var_ratio_cap10k_over_cap3k"]
    L = out["L2_ratio_cap10k_over_cap3k"]
    lines = [
        "Within-cluster latent variance: cap3k_seed0 vs cap10k",
        "=====================================================",
        f"paired (dataset, cell_type) pairs: {out['n_paired_type_in_dataset']}",
        f"cap3k total cluster types: {out['n_total_cap3k_cluster_types']}  "
        f"cap10k: {out['n_total_cap10k_cluster_types']}",
        "",
        f"Variance ratio (cap10k/cap3k):",
        f"  median={v['median']}  mean={v['mean']}  IQR={v['p25_p75']}",
        f"  >1: {v['n_gt_1']}  >1.5: {v['n_gt_1_5']}",
        "",
        f"Mean L2 from centroid ratio (robust):",
        f"  median={L['median']}  mean={L['mean']}  IQR={L['p25_p75']}",
        f"  >1: {L['n_gt_1']}",
        "",
        "Top 10 within-cluster variance gains (cap10k exposes more heterogeneity):",
    ]
    for p in out["top_gains"][:10]:
        lines.append(
            f"  {p['dataset'][:30]:<30s} {p['cell_type'][:30]:<30s}  "
            f"n3={p['cap3k_n']:>4d} n10={p['cap10k_n']:>4d}  "
            f"ratio={p['var_ratio']:.2f}"
        )
    lines.append("")
    lines.append("Top 5 losses (where cap10k compressed within-cluster variance):")
    for p in out["top_losses"][-5:]:
        lines.append(
            f"  {p['dataset'][:30]:<30s} {p['cell_type'][:30]:<30s}  "
            f"n3={p['cap3k_n']:>4d} n10={p['cap10k_n']:>4d}  "
            f"ratio={p['var_ratio']:.2f}"
        )
    (HERE / "within_cluster_variance_preview.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
