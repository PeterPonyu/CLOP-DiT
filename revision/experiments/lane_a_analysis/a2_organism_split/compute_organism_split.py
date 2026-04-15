"""A2 — Organism-stratified per-type metrics.

Addresses R3.2 by splitting the 69 evaluation cell types into human-
only, mouse-only, and both-organism groups based on the organism
annotations recorded in text_caption_metadata.json, then reporting
core fidelity metrics separately for each group.

Caveat (documented in the output):

  The stratification is type-level, not cell-level. A cell type
  classified as "both" participates in both human and mouse subsets of
  the training corpus; we do not re-run the model separately per
  organism, because the frozen baseline does not emit per-cell
  organism tags. This is a first-order answer; a strict cell-level
  stratification would need to plumb organism labels through the
  generation/evaluation pipeline and is scoped for Lane C.

Inputs (all read-only, live paths):
  data/cached_latents/text_caption_metadata.json  {gid_str: {organism, ...}}
  data/cached_latents/text_captions_deduplicated.json
  results/per_type_dashboard.json
  results/expression_metrics.json -> per_type_expression_fidelity

Outputs (written next to this script):
  organism_split.json
  organism_stats_preview.txt
  organism_per_type_table.csv
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

META_PATH = REPO_ROOT / "data" / "cached_latents" / "text_caption_metadata.json"
CAPTIONS_PATH = REPO_ROOT / "data" / "cached_latents" / "text_captions_deduplicated.json"
DASHBOARD_PATH = REPO_ROOT / "results" / "per_type_dashboard.json"
EXPR_METRICS_PATH = REPO_ROOT / "results" / "expression_metrics.json"


def short_name(full: str) -> str:
    for m in [" are ", " is ", " have ", " represent ", " form ", " characterize"]:
        if m in full:
            return full.split(m)[0].strip()
    return full.split(".")[0].strip()


def sha256_head(path: Path, limit: int = 2**26) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def classify(organisms: list[str]) -> str:
    s = {o.lower() for o in organisms if isinstance(o, str)}
    s.discard("unknown")
    if s == {"human"}:
        return "human_only"
    if s == {"mouse"}:
        return "mouse_only"
    if {"human", "mouse"}.issubset(s):
        return "both"
    if not s:
        return "unknown"
    return "other"


def summarize(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def main() -> None:
    meta = json.loads(META_PATH.read_text())
    captions = json.loads(CAPTIONS_PATH.read_text())
    dashboard = json.loads(DASHBOARD_PATH.read_text())
    efidel = json.loads(EXPR_METRICS_PATH.read_text())["per_type_expression_fidelity"]

    id_to_name = {int(k): short_name(v) for k, v in captions.items()}
    dash_by_name = {}
    for entry in dashboard:
        name = entry.get("cell_type")
        if name and name not in dash_by_name:
            dash_by_name[name] = entry

    per_type = []
    for gid, name in id_to_name.items():
        gid_str = str(gid)
        organisms = meta.get(gid_str, {}).get("organism", [])
        group = classify(organisms if isinstance(organisms, list) else [organisms])

        row = {
            "group_id": gid,
            "cell_type": name,
            "organisms_raw": organisms,
            "organism_group": group,
        }
        d = dash_by_name.get(name)
        if d:
            row["n_real"] = d.get("n_real")
            row["n_gen"] = d.get("n_gen")
            row["centroid_cosine"] = d.get("centroid_cosine")
            row["frechet_distance"] = d.get("frechet_distance")
            row["diversity_ratio"] = d.get("diversity_ratio")
            row["real_intra_cos"] = d.get("real_intra_cos")
            row["gen_intra_cos"] = d.get("gen_intra_cos")
        e = efidel.get(name)
        if e:
            row["expr_pearson_r"] = e.get("pearson_r")
        per_type.append(row)

    # --- per-organism-group summary ---------------------------------
    metrics = ("centroid_cosine", "frechet_distance", "diversity_ratio",
               "expr_pearson_r", "real_intra_cos", "gen_intra_cos")
    groups = ("human_only", "mouse_only", "both", "unknown", "other")

    by_group = {g: {"n_types": 0, "metrics": {}} for g in groups}
    for g in groups:
        rows = [r for r in per_type if r["organism_group"] == g]
        by_group[g]["n_types"] = len(rows)
        for m in metrics:
            vals = np.array([r.get(m, np.nan) for r in rows], dtype=float)
            by_group[g]["metrics"][m] = summarize(vals)

    # --- pairwise tests: human_only vs mouse_only, human_only vs both
    def mw(a: np.ndarray, b: np.ndarray):
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 2 or b.size < 2:
            return {"n_a": int(a.size), "n_b": int(b.size),
                    "note": "insufficient"}
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return {
            "n_a": int(a.size), "n_b": int(b.size),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_delta_a_minus_b": float(a.mean() - b.mean()),
            "mannwhitney_u": float(stat),
            "mannwhitney_p_two_sided": float(p),
        }

    comparisons = {}
    def vals(group: str, m: str) -> np.ndarray:
        return np.array([r.get(m, np.nan) for r in per_type
                         if r["organism_group"] == group], dtype=float)

    for m in metrics:
        comparisons[m] = {
            "human_only_vs_mouse_only": mw(vals("human_only", m),
                                           vals("mouse_only", m)),
            "human_only_vs_both": mw(vals("human_only", m),
                                     vals("both", m)),
            "mouse_only_vs_both": mw(vals("mouse_only", m),
                                     vals("both", m)),
        }

    summary = {
        "inputs": {
            "organism_metadata": {
                "path": str(META_PATH.relative_to(REPO_ROOT)),
                "sha256_head": sha256_head(META_PATH),
            },
            "captions": str(CAPTIONS_PATH.relative_to(REPO_ROOT)),
            "dashboard": str(DASHBOARD_PATH.relative_to(REPO_ROOT)),
            "expression_metrics": str(EXPR_METRICS_PATH.relative_to(REPO_ROOT)),
        },
        "caveats": (
            "Stratification is TYPE-LEVEL, not cell-level. Types tagged "
            "'both' contain training cells from both human and mouse "
            "datasets; their per-type metrics cannot be attributed to "
            "one organism alone. A strict cell-level evaluation is "
            "scoped for Lane C."
        ),
        "group_counts": {g: by_group[g]["n_types"] for g in groups},
        "by_organism_group": by_group,
        "pairwise_mannwhitney": comparisons,
        "per_type": per_type,
    }

    (HERE / "organism_split.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: float(o)
                   if isinstance(o, np.floating) else None) + "\n"
    )

    with (HERE / "organism_per_type_table.csv").open("w", newline="") as f:
        cols = ["group_id", "cell_type", "organism_group", "n_real", "n_gen",
                "centroid_cosine", "frechet_distance", "diversity_ratio",
                "real_intra_cos", "gen_intra_cos", "expr_pearson_r"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(per_type, key=lambda x: (x["organism_group"],
                                                  x["cell_type"])):
            w.writerow(r)

    # --- preview ----------------------------------------------------
    lines = [
        "A2 Organism-stratified per-type metrics",
        "========================================",
        f"Group counts: {summary['group_counts']}",
        "",
        f"{'metric':20s} {'group':12s} {'n':>3s} {'mean':>9s} "
        f"{'median':>9s} {'std':>9s}",
        "-" * 70,
    ]
    for m in metrics:
        for g in ("human_only", "mouse_only", "both"):
            s = by_group[g]["metrics"].get(m, {})
            if s.get("n"):
                lines.append(
                    f"{m:20s} {g:12s} {s['n']:>3d} "
                    f"{s['mean']:>+9.4f} {s['median']:>+9.4f} "
                    f"{s['std']:>9.4f}"
                )
        lines.append("")

    lines += [
        "Pairwise Mann-Whitney (two-sided) — mean Δ (A - B), p",
        "------------------------------------------------------",
    ]
    for m in metrics:
        for comp_name, c in comparisons[m].items():
            if "mannwhitney_p_two_sided" in c:
                lines.append(
                    f"  {m:20s} {comp_name:28s} "
                    f"Δ={c['mean_delta_a_minus_b']:>+8.4f} "
                    f"p={c['mannwhitney_p_two_sided']:>8.2e} "
                    f"(nA={c['n_a']}, nB={c['n_b']})"
                )
        lines.append("")
    preview = "\n".join(lines) + "\n"
    (HERE / "organism_stats_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
