"""Clean head-to-head variance comparison: cap3k_seed0 vs cap10k_seed0.

Both caches now produced with the same subsample seed so any remaining
difference in per-dataset total variance is attributable to the cap,
not to stochastic draw drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

C3 = REPO_ROOT / "data" / "cached_latents_cap3k_seed0"
C10 = REPO_ROOT / "data" / "cached_latents_cap10k"


def per_group_stats(cells: np.ndarray, gids: np.ndarray) -> dict[int, dict]:
    out = {}
    for gid in np.unique(gids):
        X = cells[gids == gid]
        if X.shape[0] < 10:
            continue
        mu = X.mean(axis=0)
        out[int(gid)] = {
            "n": int(X.shape[0]),
            "total_variance": float(X.var(axis=0).sum()),
            "mean_L2_from_centroid": float(
                np.linalg.norm(X - mu, axis=1).mean()
            ),
        }
    return out


def load(cache: Path):
    cells = np.load(cache / "cell_embeddings.npy")
    gids = np.load(cache / "text_group_ids.npy")
    meta = json.loads((cache / "metadata.json").read_text())
    return cells, gids, meta


def main() -> None:
    c3, g3, m3 = load(C3)
    c10, g10, m10 = load(C10)

    s3 = per_group_stats(c3, g3)
    s10 = per_group_stats(c10, g10)

    # Pair by exact text
    t3_to_gid = {v: int(k) for k, v in m3.items()}
    pairs = []
    for gid10_str, text in m10.items():
        g3_id = t3_to_gid.get(text)
        if g3_id is None:
            continue
        g10_id = int(gid10_str)
        if g3_id not in s3 or g10_id not in s10:
            continue
        a = s3[g3_id]
        b = s10[g10_id]
        pairs.append({
            "text_prefix": text[:60],
            "cap3k_n": a["n"],
            "cap10k_n": b["n"],
            "cap3k_total_var": round(a["total_variance"], 3),
            "cap10k_total_var": round(b["total_variance"], 3),
            "var_ratio": round(
                b["total_variance"] / max(a["total_variance"], 1e-9), 3
            ),
            "cap3k_mean_L2": round(a["mean_L2_from_centroid"], 4),
            "cap10k_mean_L2": round(b["mean_L2_from_centroid"], 4),
            "L2_ratio": round(
                b["mean_L2_from_centroid"]
                / max(a["mean_L2_from_centroid"], 1e-9), 3
            ),
        })

    ratios = [p["var_ratio"] for p in pairs]
    L2ratios = [p["L2_ratio"] for p in pairs]
    out = {
        "seed": 0,
        "n_paired_datasets": len(pairs),
        "cap3k_total_cells": int(c3.shape[0]),
        "cap10k_total_cells": int(c10.shape[0]),
        "var_ratio_cap10k_over_cap3k": {
            "median": round(float(np.median(ratios)), 3),
            "mean": round(float(np.mean(ratios)), 3),
            "p25_p75": [round(float(np.percentile(ratios, 25)), 3),
                        round(float(np.percentile(ratios, 75)), 3)],
            "n_gt_1": int(sum(1 for r in ratios if r > 1)),
            "n_gt_1_5": int(sum(1 for r in ratios if r > 1.5)),
            "n_lt_1": int(sum(1 for r in ratios if r < 1)),
        },
        "L2_ratio_cap10k_over_cap3k": {
            "median": round(float(np.median(L2ratios)), 3),
            "mean": round(float(np.mean(L2ratios)), 3),
            "p25_p75": [round(float(np.percentile(L2ratios, 25)), 3),
                        round(float(np.percentile(L2ratios, 75)), 3)],
            "n_gt_1": int(sum(1 for r in L2ratios if r > 1)),
        },
        "per_dataset": sorted(pairs, key=lambda p: -p["var_ratio"]),
    }
    (HERE / "seed0_variance_comparison.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    v = out["var_ratio_cap10k_over_cap3k"]
    L = out["L2_ratio_cap10k_over_cap3k"]
    lines = [
        "Clean head-to-head: cap3k_seed0 vs cap10k (seed=0)",
        "===================================================",
        f"cap3k cells: {out['cap3k_total_cells']:,}  "
        f"cap10k cells: {out['cap10k_total_cells']:,}  "
        f"pairs: {out['n_paired_datasets']}",
        "",
        f"Variance ratio (cap10k / cap3k):",
        f"  median = {v['median']}  mean = {v['mean']}  "
        f"IQR = {v['p25_p75']}",
        f"  > 1    : {v['n_gt_1']}/{out['n_paired_datasets']}",
        f"  > 1.5  : {v['n_gt_1_5']}/{out['n_paired_datasets']}",
        f"  < 1    : {v['n_lt_1']}/{out['n_paired_datasets']}",
        "",
        f"Mean L2-from-centroid ratio (robust):",
        f"  median = {L['median']}  mean = {L['mean']}  "
        f"IQR = {L['p25_p75']}",
        f"  > 1    : {L['n_gt_1']}/{out['n_paired_datasets']}",
        "",
        "Top 10 variance gains:",
        f"  {'text':<55s} {'n3':>5s} {'n10':>5s} "
        f"{'var3':>8s} {'var10':>8s} {'ratio':>6s}",
    ]
    for p in out["per_dataset"][:10]:
        lines.append(
            f"  {p['text_prefix']:<55s} "
            f"{p['cap3k_n']:>5d} {p['cap10k_n']:>5d} "
            f"{p['cap3k_total_var']:>8.2f} {p['cap10k_total_var']:>8.2f} "
            f"{p['var_ratio']:>6.2f}"
        )
    lines.append("")
    lines.append("Bottom 5 (where cap10k gave less variance):")
    for p in out["per_dataset"][-5:]:
        lines.append(
            f"  {p['text_prefix']:<55s} "
            f"{p['cap3k_n']:>5d} {p['cap10k_n']:>5d} "
            f"{p['cap3k_total_var']:>8.2f} {p['cap10k_total_var']:>8.2f} "
            f"{p['var_ratio']:>6.2f}"
        )
    preview = "\n".join(lines) + "\n"
    (HERE / "seed0_variance_comparison_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
