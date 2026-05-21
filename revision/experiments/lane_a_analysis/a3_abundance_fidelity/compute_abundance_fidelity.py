"""A3 — Per-type fidelity as a function of training-set abundance.

Operationalises the reviewer's vague "biologically difficult" phrase by
replacing it with "underrepresented in training", quantified as a
quartile split of training cell counts. Reports rank and linear
correlation of each per-type fidelity metric against log training
count, plus a bottom-vs-top-quartile comparison with Mann–Whitney U.

Inputs (all read-only, live paths):
  data/cached_latents/text_group_ids_dedup.npy           int32 (N,)
  data/cached_latents/text_captions_deduplicated.json    {id_str: caption}
  results/per_type_dashboard.json                        list of per-type entries
  results/expression_metrics.json                        per_type_expression_fidelity

Outputs (written next to this script):
  abundance_fidelity.json
  per_type_table.csv
  stats_preview.txt

Join key is cell-type short name (leading phrase of the caption), which
matches 68/69 entries in per_type_expression_fidelity. Missing types
are skipped explicitly in the output.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]

GROUP_IDS_PATH = REPO_ROOT / "data" / "cached_latents" / "text_group_ids_dedup.npy"
CAPTIONS_PATH = REPO_ROOT / "data" / "cached_latents" / "text_captions_deduplicated.json"
DASHBOARD_PATH = REPO_ROOT / "results" / "per_type_dashboard.json"
EXPR_METRICS_PATH = REPO_ROOT / "results" / "expression_metrics.json"


def short_name(full: str) -> str:
    """Derive the canonical short cell-type name from a caption."""
    for marker in [" are ", " is ", " have ", " represent ", " form ", " characterize"]:
        if marker in full:
            return full.split(marker)[0].strip()
    return full.split(".")[0].strip()


def sha256_head(path: Path, limit_bytes: int = 2**27) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit_bytes))
    return h.hexdigest()[:16]


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    r, p = stats.spearmanr(x, y, nan_policy="omit")
    return float(r), float(p)


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def main() -> None:
    group_ids = np.load(GROUP_IDS_PATH)
    captions = json.loads(CAPTIONS_PATH.read_text())
    dashboard = json.loads(DASHBOARD_PATH.read_text())
    efidel = json.loads(EXPR_METRICS_PATH.read_text())["per_type_expression_fidelity"]

    id_to_name = {int(k): short_name(v) for k, v in captions.items()}

    train_count = {}
    uniq, cnt = np.unique(group_ids, return_counts=True)
    for g, c in zip(uniq.tolist(), cnt.tolist()):
        train_count[int(g)] = int(c)

    dash_by_name = {}
    for entry in dashboard:
        name = entry.get("cell_type")
        if name and name not in dash_by_name:
            dash_by_name[name] = entry

    per_type = []
    skipped = []
    for gid, name in id_to_name.items():
        row = {"group_id": gid, "cell_type": name,
               "training_count": train_count.get(gid)}
        if row["training_count"] is None:
            skipped.append({"group_id": gid, "reason": "no training count"})
            continue
        row["log10_training_count"] = float(np.log10(row["training_count"]))

        d = dash_by_name.get(name)
        if d:
            row["centroid_cosine"] = d.get("centroid_cosine")
            row["frechet_distance"] = d.get("frechet_distance")
            row["diversity_ratio"] = d.get("diversity_ratio")
            row["real_intra_cos"] = d.get("real_intra_cos")
            row["gen_intra_cos"] = d.get("gen_intra_cos")
            row["n_real"] = d.get("n_real")
            row["n_gen"] = d.get("n_gen")

        e = efidel.get(name)
        if e:
            row["expr_pearson_r"] = e.get("pearson_r")

        if d is None and e is None:
            skipped.append({"group_id": gid, "cell_type": name,
                            "reason": "no dashboard / no efidel match"})
            continue

        per_type.append(row)

    # --- correlations -------------------------------------------------
    def pull(metric: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        x, y, names = [], [], []
        for r in per_type:
            v = r.get(metric)
            if v is None or not np.isfinite(v):
                continue
            x.append(r["log10_training_count"])
            y.append(float(v))
            names.append(r["cell_type"])
        return np.asarray(x), np.asarray(y), names

    correlations = {}
    for metric in ("centroid_cosine", "expr_pearson_r",
                   "diversity_ratio", "frechet_distance"):
        x, y, _ = pull(metric)
        if x.size < 4:
            correlations[metric] = {"n": int(x.size), "note": "too few"}
            continue
        pr, pp = pearson(x, y)
        sr, sp = spearman(x, y)
        correlations[metric] = {
            "n": int(x.size),
            "pearson_r_vs_log_count": pr,
            "pearson_p": pp,
            "spearman_rho_vs_log_count": sr,
            "spearman_p": sp,
        }

    # --- predictor comparison: abundance vs real heterogeneity ------
    # real_intra_cos = within-type mean pairwise cosine of real embeddings;
    # HIGHER means tighter cluster, LOWER means broader spread. The
    # working hypothesis is that spread is a better predictor of
    # per-type difficulty than training count.
    predictor_comparison = {}
    targets = ("centroid_cosine", "frechet_distance", "diversity_ratio")
    predictors = {
        "log10_training_count":
            np.array([r["log10_training_count"] for r in per_type],
                     dtype=float),
        "real_intra_cos":
            np.array([r.get("real_intra_cos", np.nan) for r in per_type],
                     dtype=float),
    }
    for target in targets:
        y = np.array([r.get(target, np.nan) for r in per_type], dtype=float)
        block = {}
        for pname, px in predictors.items():
            mask = np.isfinite(px) & np.isfinite(y)
            if mask.sum() < 4:
                block[pname] = {"n": int(mask.sum()), "note": "too few"}
                continue
            pr, pp = pearson(px[mask], y[mask])
            sr, sp = spearman(px[mask], y[mask])
            block[pname] = {
                "n": int(mask.sum()),
                "pearson_r": pr, "pearson_p": pp,
                "spearman_rho": sr, "spearman_p": sp,
            }
        predictor_comparison[target] = block

    # --- quartile split ---------------------------------------------
    counts = np.array([r["training_count"] for r in per_type], dtype=float)
    q25, q50, q75 = np.percentile(counts, [25, 50, 75])
    q_bounds = {
        "q25_training_count": float(q25),
        "q50_training_count": float(q50),
        "q75_training_count": float(q75),
    }

    quartile_stats = {}
    for metric in ("centroid_cosine", "expr_pearson_r",
                   "diversity_ratio", "frechet_distance"):
        bot_vals, top_vals = [], []
        for r in per_type:
            v = r.get(metric)
            if v is None or not np.isfinite(v):
                continue
            if r["training_count"] <= q25:
                bot_vals.append(v)
            elif r["training_count"] >= q75:
                top_vals.append(v)
        bot = np.asarray(bot_vals)
        top = np.asarray(top_vals)
        if bot.size < 2 or top.size < 2:
            quartile_stats[metric] = {"note": "insufficient data",
                                      "n_bot": int(bot.size),
                                      "n_top": int(top.size)}
            continue
        mw = stats.mannwhitneyu(bot, top, alternative="two-sided")
        quartile_stats[metric] = {
            "bottom_quartile": {
                "n": int(bot.size),
                "mean": float(bot.mean()),
                "median": float(np.median(bot)),
                "std": float(bot.std()),
            },
            "top_quartile": {
                "n": int(top.size),
                "mean": float(top.mean()),
                "median": float(np.median(top)),
                "std": float(top.std()),
            },
            "mean_delta_top_minus_bottom": float(top.mean() - bot.mean()),
            "mannwhitney_u": float(mw.statistic),
            "mannwhitney_p_two_sided": float(mw.pvalue),
        }

    summary = {
        "inputs": {
            "group_ids": {
                "path": str(GROUP_IDS_PATH.relative_to(REPO_ROOT)),
                "sha256_head": sha256_head(GROUP_IDS_PATH),
            },
            "captions": str(CAPTIONS_PATH.relative_to(REPO_ROOT)),
            "dashboard": str(DASHBOARD_PATH.relative_to(REPO_ROOT)),
            "expression_metrics": str(EXPR_METRICS_PATH.relative_to(REPO_ROOT)),
        },
        "definitions": {
            "underrepresented": "training_count <= Q25 (bottom quartile)",
            "well_represented": "training_count >= Q75 (top quartile)",
            "metrics": [
                "centroid_cosine (real vs generated embedding centroid; higher = better)",
                "expr_pearson_r (per-type pooled Pearson r on expression; higher = better)",
                "diversity_ratio (DivR; ideal = 1.0)",
                "frechet_distance (FD; lower = better)",
            ],
        },
        "n_types_analysed": len(per_type),
        "n_types_skipped": len(skipped),
        "skipped": skipped,
        "quartile_bounds": q_bounds,
        "correlations": correlations,
        "quartile_comparison": quartile_stats,
        "predictor_comparison": predictor_comparison,
        "per_type": per_type,
    }

    (HERE / "abundance_fidelity.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: float(o)
                   if isinstance(o, np.floating) else None) + "\n"
    )

    with (HERE / "per_type_table.csv").open("w", newline="") as f:
        cols = ["group_id", "cell_type", "training_count",
                "log10_training_count", "n_real", "n_gen",
                "centroid_cosine", "frechet_distance",
                "diversity_ratio", "expr_pearson_r"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(per_type, key=lambda x: x["training_count"]):
            w.writerow(r)

    # --- preview ----------------------------------------------------
    lines = [
        "A3 Abundance vs fidelity summary",
        "================================",
        f"types analysed : {len(per_type)} / 69  (skipped: {len(skipped)})",
        f"training-count Q25 / Q50 / Q75 : {int(q25)} / {int(q50)} / {int(q75)}",
        "",
        "Correlation with log10(training count) — higher |r| = stronger abundance effect",
        "--------------------------------------------------------------------------------",
        f"  {'metric':22s}  {'n':>4} {'pearson r':>12} {'p':>8}  {'spearman ρ':>12} {'p':>8}",
    ]
    for m, stats_ in correlations.items():
        if "pearson_r_vs_log_count" in stats_:
            lines.append(
                f"  {m:22s}  {stats_['n']:>4d} "
                f"{stats_['pearson_r_vs_log_count']:>+12.3f} "
                f"{stats_['pearson_p']:>8.1e}  "
                f"{stats_['spearman_rho_vs_log_count']:>+12.3f} "
                f"{stats_['spearman_p']:>8.1e}"
            )
    lines += ["",
              "Bottom quartile vs top quartile (Mann-Whitney two-sided)",
              "---------------------------------------------------------",
              f"  {'metric':22s}  {'bot mean':>12} {'top mean':>12} "
              f"{'Δ(top-bot)':>12} {'p':>10}"]
    for m, s in quartile_stats.items():
        if "mean_delta_top_minus_bottom" in s:
            lines.append(
                f"  {m:22s}  "
                f"{s['bottom_quartile']['mean']:>12.3f} "
                f"{s['top_quartile']['mean']:>12.3f} "
                f"{s['mean_delta_top_minus_bottom']:>+12.3f} "
                f"{s['mannwhitney_p_two_sided']:>10.2e}"
            )
    lines += ["",
              "Predictor comparison — abundance vs real within-type tightness",
              "--------------------------------------------------------------",
              f"  {'target':18s}  {'predictor':22s} "
              f"{'pearson r':>10} {'p':>10}  {'spearman ρ':>10} {'p':>10}"]
    for target, block in predictor_comparison.items():
        for pname, s in block.items():
            if "pearson_r" in s:
                lines.append(
                    f"  {target:18s}  {pname:22s} "
                    f"{s['pearson_r']:>+10.3f} {s['pearson_p']:>10.2e}  "
                    f"{s['spearman_rho']:>+10.3f} {s['spearman_p']:>10.2e}"
                )
        lines.append("")
    preview = "\n".join(lines) + "\n"
    (HERE / "stats_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
