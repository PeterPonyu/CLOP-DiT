"""Confirm the bimodal tails in the first-pass diagnostic are seed-only.

Compares cap3k historical (unseeded) cache against cap3k_seed0 cache.
Both have the same cap (3000), same preprocessing, same scGPT
encoder. The only difference is the sc.pp.subsample random state.

If the bimodal tails of the original cap10k / cap3k comparison were
driven by seed drift, this comparison should show a similar-width
IQR with median near 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

C_HIST = REPO_ROOT / "data" / "cached_latents"
C_SEED = REPO_ROOT / "data" / "cached_latents_cap3k_seed0"


def per_group(cells: np.ndarray, gids: np.ndarray) -> dict[int, dict]:
    out = {}
    for gid in np.unique(gids):
        X = cells[gids == gid]
        if X.shape[0] < 10:
            continue
        out[int(gid)] = {
            "n": int(X.shape[0]),
            "total_variance": float(X.var(axis=0).sum()),
            "mean_L2": float(np.linalg.norm(X - X.mean(0), axis=1).mean()),
        }
    return out


def main() -> None:
    cells_h = np.load(C_HIST / "cell_embeddings.npy")
    gids_h = np.load(C_HIST / "text_group_ids.npy")
    meta_h = json.loads((C_HIST / "metadata.json").read_text())
    cells_s = np.load(C_SEED / "cell_embeddings.npy")
    gids_s = np.load(C_SEED / "text_group_ids.npy")
    meta_s = json.loads((C_SEED / "metadata.json").read_text())

    sh = per_group(cells_h, gids_h)
    ss = per_group(cells_s, gids_s)

    # Pair by exact text
    text_to_gh = {v: int(k) for k, v in meta_h.items()}
    pairs = []
    for gid_s_str, text in meta_s.items():
        gh = text_to_gh.get(text)
        if gh is None:
            continue
        gs = int(gid_s_str)
        if gh not in sh or gs not in ss:
            continue
        a = sh[gh]; b = ss[gs]
        pairs.append({
            "text_prefix": text[:60],
            "n_hist": a["n"], "n_seed": b["n"],
            "var_hist": round(a["total_variance"], 3),
            "var_seed": round(b["total_variance"], 3),
            "ratio_seed_over_hist": round(
                b["total_variance"] / max(a["total_variance"], 1e-9), 3
            ),
            "L2_ratio": round(
                b["mean_L2"] / max(a["mean_L2"], 1e-9), 3
            ),
        })
    pairs.sort(key=lambda p: -p["ratio_seed_over_hist"])

    ratios = [p["ratio_seed_over_hist"] for p in pairs]
    out = {
        "n_pairs": len(pairs),
        "comparison": "cap3k_seed0 / cap3k_historical_unseeded",
        "both_same_cap": 3000,
        "only_difference": "sc.pp.subsample random_state",
        "variance_ratio": {
            "median": round(float(np.median(ratios)), 3),
            "mean": round(float(np.mean(ratios)), 3),
            "IQR": [round(float(np.percentile(ratios, 25)), 3),
                    round(float(np.percentile(ratios, 75)), 3)],
            "n_gt_1_5": int(sum(1 for r in ratios if r > 1.5)),
            "n_lt_0_67": int(sum(1 for r in ratios if r < 2 / 3)),
            "min": round(float(min(ratios)), 3),
            "max": round(float(max(ratios)), 3),
        },
        "top_gains_by_seed_alone": pairs[:5],
        "top_losses_by_seed_alone": pairs[-5:],
    }
    (HERE / "seed_confound_verification.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    v = out["variance_ratio"]
    lines = [
        "Seed-confound verification",
        "===========================",
        "Compares: cap3k_seed0 vs cap3k_historical (unseeded). Same cap,",
        "same preprocessing, same scGPT. Only sc.pp.subsample seed differs.",
        "",
        f"Paired datasets: {out['n_pairs']}",
        f"Variance ratio (seed0 / hist):",
        f"  median={v['median']}  mean={v['mean']}  IQR={v['IQR']}",
        f"  min={v['min']}  max={v['max']}",
        f"  > 1.5 (seed0 wins big) : {v['n_gt_1_5']}",
        f"  < 0.67 (hist won big)  : {v['n_lt_0_67']}",
        "",
        "Interpretation: an IQR much wider than ~[0.95, 1.05] means",
        "seed-drift alone shifts per-dataset latent variance by O(10%)",
        "or more, which fully accounts for the bimodal tails observed",
        "in the first-pass cap10k vs cap3k_historical diagnostic.",
    ]
    (HERE / "seed_confound_verification_preview.txt").write_text(
        "\n".join(lines) + "\n"
    )
    print("\n".join(lines))
    print()
    print("Top 5 gains by seed alone:")
    for p in out["top_gains_by_seed_alone"]:
        print(f"  {p['text_prefix']:<55s}  n_h={p['n_hist']} n_s={p['n_seed']} "
              f"v_h={p['var_hist']:.2f} v_s={p['var_seed']:.2f}  "
              f"ratio={p['ratio_seed_over_hist']:.2f}")
    print("Top 5 losses by seed alone:")
    for p in out["top_losses_by_seed_alone"]:
        print(f"  {p['text_prefix']:<55s}  n_h={p['n_hist']} n_s={p['n_seed']} "
              f"v_h={p['var_hist']:.2f} v_s={p['var_seed']:.2f}  "
              f"ratio={p['ratio_seed_over_hist']:.2f}")


if __name__ == "__main__":
    main()
