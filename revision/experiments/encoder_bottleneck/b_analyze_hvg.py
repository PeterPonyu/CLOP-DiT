"""B — Analyze HVG-count ablation.

For each n_top_genes in {500, 1000, 2000, 4000, 8000} (2000 reused as
the cap3k_seed0 baseline), compute per-dataset scGPT-latent within-
type variance and within-cluster L2 using the SAME subcluster_metadata
cell assignments as the 2000-HVG baseline (QC is HVG-independent so
cell identity is preserved across variants).

Expected outcome: if scGPT saturates regardless of HVG count, all
within-type variance curves should be flat. If HVG is a sub-
bottleneck, variance should scale monotonically with HVG count.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

# 2000-HVG baseline comes from cap3k_seed0
BASELINE_CACHE = REPO_ROOT / "data" / "cached_latents_cap3k_seed0"
BASELINE_H5AD = REPO_ROOT / "data" / "processed_h5ad_cap3k_seed0"
SUB_META = BASELINE_H5AD / "subcluster_metadata.json"

HVG_VARIANTS = {
    500: (REPO_ROOT / "data" / "cached_latents_hvg500",
          REPO_ROOT / "data" / "processed_h5ad_hvg500"),
    1000: (REPO_ROOT / "data" / "cached_latents_hvg1000",
           REPO_ROOT / "data" / "processed_h5ad_hvg1000"),
    2000: (BASELINE_CACHE, BASELINE_H5AD),
    4000: (REPO_ROOT / "data" / "cached_latents_hvg4000",
           REPO_ROOT / "data" / "processed_h5ad_hvg4000"),
    8000: (REPO_ROOT / "data" / "cached_latents_hvg8000",
           REPO_ROOT / "data" / "processed_h5ad_hvg8000"),
}


def load_order(cache: Path) -> dict[str, int]:
    proc = json.loads((cache / "processed_datasets.json").read_text())
    if isinstance(proc, list):
        names = [Path(p).stem.replace("_processed", "") for p in proc]
    else:
        names = list(proc.keys())
    return {n: i for i, n in enumerate(names)}


def within_cluster_L2(cache: Path, sub: dict) -> dict:
    """Compute within-cluster L2 for each (dataset, type) using baseline labels."""
    cells = np.load(cache / "cell_embeddings.npy")
    sample_ids = np.load(cache / "sample_ids.npy", allow_pickle=True)
    order = load_order(cache)

    per_pair = []
    for ds_id, ds_block in sub.items():
        if ds_id not in order:
            continue
        ord_k = order[ds_id]
        ds_mask = (sample_ids == ord_k)
        ds_cells = cells[ds_mask]
        if ds_cells.shape[0] < 20:
            continue
        n_cells = ds_cells.shape[0]
        # Aggregate by cell_type (pool multiple clusters of same type)
        type_pool = {}
        for cid, info in ds_block.get("clusters", {}).items():
            ct = info.get("cell_type", "Unknown")
            if ct == "Unknown":
                continue
            idx = np.asarray(info.get("cell_indices", []), dtype=int)
            valid = idx[idx < n_cells]
            if valid.size < 10:
                continue
            type_pool.setdefault(ct, []).append(valid)
        for ct, idx_lists in type_pool.items():
            all_idx = np.concatenate(idx_lists)
            if all_idx.size < 10:
                continue
            X = ds_cells[all_idx]
            mu = X.mean(axis=0)
            per_pair.append({
                "dataset": ds_id,
                "cell_type": ct,
                "n": int(X.shape[0]),
                "mean_L2_from_centroid": float(
                    np.linalg.norm(X - mu, axis=1).mean()
                ),
                "total_variance": float(X.var(axis=0).sum()),
            })
    return per_pair


def summarize(pairs: list[dict]) -> dict:
    if not pairs:
        return {}
    vs = [p["total_variance"] for p in pairs]
    ls = [p["mean_L2_from_centroid"] for p in pairs]
    return {
        "n_pairs": len(pairs),
        "median_total_variance": round(float(np.median(vs)), 3),
        "mean_total_variance": round(float(np.mean(vs)), 3),
        "median_mean_L2": round(float(np.median(ls)), 4),
        "mean_mean_L2": round(float(np.mean(ls)), 4),
    }


def main() -> None:
    sub = json.loads(SUB_META.read_text())
    rows = []
    for n, (cache, h5ad_dir) in HVG_VARIANTS.items():
        if not (cache / "cell_embeddings.npy").exists():
            print(f"[skip hvg={n}] cache {cache} not yet complete")
            continue
        per = within_cluster_L2(cache, sub)
        s = summarize(per)
        s["n_top_genes"] = int(n)
        s["cache"] = str(cache.relative_to(REPO_ROOT))
        rows.append(s)
        print(f"hvg={n:5d}  n_pairs={s['n_pairs']:4d}  "
              f"median_var={s['median_total_variance']:.3f}  "
              f"median_L2={s['median_mean_L2']:.4f}")

    # Paired comparison to 2000-HVG baseline, if baseline present
    base_rows = [r for r in rows if r["n_top_genes"] == 2000]
    out = {"summary_per_variant": rows}
    if base_rows:
        base_var = base_rows[0]["median_total_variance"]
        base_L2 = base_rows[0]["median_mean_L2"]
        out["ratios_to_2000_baseline"] = [
            {
                "n_top_genes": r["n_top_genes"],
                "var_ratio": round(r["median_total_variance"] / base_var, 4)
                                 if base_var > 0 else None,
                "L2_ratio": round(r["median_mean_L2"] / base_L2, 4)
                                 if base_L2 > 0 else None,
            }
            for r in rows if r["n_top_genes"] != 2000
        ]

    (HERE / "hvg_ablation_summary.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    lines = ["B — HVG-count ablation summary",
             "=============================="]
    for r in rows:
        lines.append(
            f"  n_top_genes={r['n_top_genes']:5d}  "
            f"pairs={r['n_pairs']:4d}  "
            f"median_var={r['median_total_variance']:>8.3f}  "
            f"median_L2={r['median_mean_L2']:.4f}"
        )
    if base_rows:
        lines.append("")
        lines.append("Ratios vs n=2000 baseline:")
        for r in out["ratios_to_2000_baseline"]:
            lines.append(
                f"  hvg={r['n_top_genes']:5d}  "
                f"var_ratio={r['var_ratio']:.3f}  "
                f"L2_ratio={r['L2_ratio']:.3f}"
            )
    (HERE / "hvg_ablation_summary_preview.txt").write_text(
        "\n".join(lines) + "\n"
    )
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
