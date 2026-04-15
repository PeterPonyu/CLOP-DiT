"""Quick diagnostic: per-dataset latent variance, cap3k vs cap10k.

Reads scGPT cell embeddings from both caches (dataset-level text labels)
and computes per-dataset within-dataset variance in the raw scGPT
latent space (pre-CLOP). If cap10k is capturing more heterogeneity,
variance should be higher for datasets where the cap was binding.

No retraining. No assumption about cluster labels. First-order check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

CAP3K = REPO_ROOT / "data" / "cached_latents"
CAP10K = REPO_ROOT / "data" / "cached_latents_cap10k"


def load(cache: Path):
    cells = np.load(cache / "cell_embeddings.npy")
    gids = np.load(cache / "text_group_ids.npy")
    texts = json.loads((cache / "text_strings.json").read_text())
    return cells, gids, texts


def per_group_variance(cells: np.ndarray, gids: np.ndarray) -> dict[int, dict]:
    out = {}
    for gid in np.unique(gids):
        mask = gids == gid
        X = cells[mask]
        if X.shape[0] < 10:
            continue
        # Total variance = sum of per-dim variance
        mu = X.mean(axis=0)
        var_per_dim = X.var(axis=0)
        total_var = float(var_per_dim.sum())
        # Mean intra-set cosine (higher = tighter)
        norms = np.linalg.norm(X - mu, axis=1)
        out[int(gid)] = {
            "n": int(X.shape[0]),
            "total_variance": round(total_var, 3),
            "mean_L2_from_centroid": round(float(norms.mean()), 4),
        }
    return out


def main() -> None:
    c3, g3, t3 = load(CAP3K)
    c10, g10, t10 = load(CAP10K)

    # Map gids in cap10k (dataset-level) to text strings, then to cap3k gids
    # t3 and t10 are {gid_str: text_str}. cap3k has 69 gids (subcluster-level),
    # cap10k has 50 gids (dataset-level). We compare by matching text strings
    # to dataset names. Use text prefixes for a coarse join; fall back to
    # per-gid report when no match.
    v3 = per_group_variance(c3, g3)
    v10 = per_group_variance(c10, g10)

    out = {
        "cap3k": {
            "total_cells": int(c3.shape[0]),
            "n_groups": int(len(v3)),
            "group_type": "subcluster" if len(v3) > 50 else "dataset",
            "median_total_variance": round(
                float(np.median([v["total_variance"] for v in v3.values()])), 3
            ),
            "median_mean_L2_from_centroid": round(
                float(np.median([v["mean_L2_from_centroid"]
                                 for v in v3.values()])), 4
            ),
        },
        "cap10k": {
            "total_cells": int(c10.shape[0]),
            "n_groups": int(len(v10)),
            "group_type": "dataset",
            "median_total_variance": round(
                float(np.median([v["total_variance"] for v in v10.values()])), 3
            ),
            "median_mean_L2_from_centroid": round(
                float(np.median([v["mean_L2_from_centroid"]
                                 for v in v10.values()])), 4
            ),
        },
    }

    # Text-string match for paired comparison
    t3_map = {k: v for k, v in t3.items()}
    t10_map = {k: v for k, v in t10.items()}
    t3_text_to_gid = {v: int(k) for k, v in t3_map.items()}
    paired = []
    for gid_str, text in t10_map.items():
        if text in t3_text_to_gid:
            g3 = t3_text_to_gid[text]
            g_10 = int(gid_str)
            if g3 in v3 and g_10 in v10:
                s3 = v3[g3]; s10 = v10[g_10]
                paired.append({
                    "text_prefix": text[:60],
                    "cap3k_n": s3["n"],
                    "cap10k_n": s10["n"],
                    "cap3k_total_var": s3["total_variance"],
                    "cap10k_total_var": s10["total_variance"],
                    "var_ratio": round(
                        s10["total_variance"] / max(s3["total_variance"], 1e-9),
                        3
                    ),
                })
    if paired:
        ratios = [p["var_ratio"] for p in paired]
        out["paired"] = {
            "n_pairs": len(paired),
            "median_var_ratio_cap10k_over_cap3k": round(
                float(np.median(ratios)), 3
            ),
            "quartiles": [
                round(float(np.percentile(ratios, 25)), 3),
                round(float(np.percentile(ratios, 75)), 3),
            ],
            "n_ratio_gt_1": int(sum(1 for r in ratios if r > 1)),
            "n_ratio_lt_1": int(sum(1 for r in ratios if r < 1)),
            "top_gains": sorted(paired, key=lambda p: -p["var_ratio"])[:10],
            "top_losses": sorted(paired, key=lambda p: p["var_ratio"])[:5],
        }

    (HERE / "quick_latent_variance.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    lines = ["Quick latent variance check (scGPT space, pre-CLOP)",
             "=====================================================",
             f"cap3k:  {out['cap3k']['total_cells']:,} cells, "
             f"{out['cap3k']['n_groups']} groups "
             f"({out['cap3k']['group_type']}), "
             f"median total var = {out['cap3k']['median_total_variance']}",
             f"cap10k: {out['cap10k']['total_cells']:,} cells, "
             f"{out['cap10k']['n_groups']} groups "
             f"({out['cap10k']['group_type']}), "
             f"median total var = {out['cap10k']['median_total_variance']}"]
    if "paired" in out:
        p = out["paired"]
        lines.append("")
        lines.append(
            f"Paired comparison across {p['n_pairs']} matched datasets:"
        )
        lines.append(
            f"  median cap10k/cap3k variance ratio = "
            f"{p['median_var_ratio_cap10k_over_cap3k']}  "
            f"IQR = {p['quartiles']}"
        )
        lines.append(
            f"  ratio > 1 (more heterogeneity at cap10k): "
            f"{p['n_ratio_gt_1']}/{p['n_pairs']}"
        )
        lines.append("")
        lines.append("Top 10 variance gains:")
        for e in p["top_gains"]:
            lines.append(
                f"  {e['text_prefix']:<60s} "
                f"n3={e['cap3k_n']:>5d} n10={e['cap10k_n']:>5d} "
                f"ratio={e['var_ratio']:.2f}"
            )
    preview = "\n".join(lines) + "\n"
    (HERE / "quick_latent_variance_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
