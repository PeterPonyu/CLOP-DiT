"""A5 — Rare-cell augmentation failure mechanism.

Decomposes within-type variance at both the LATENT scale (shared
CLOP-DiT embedding space) and the EXPRESSION scale (per-gene matrix
after decoder) to diagnose whether the augmentation failure reported
for Cycling cells comes from:

  (a) upstream under-dispersion: generator produces near-centroid
      latents, so there is no heterogeneity to propagate;
  (b) downstream compression: generator produces latents with
      reasonable spread, but the scGPT decoder averages them out; or
  (c) both.

Inputs (all read-only, live paths):
  data/cached_latents/cell_embeddings_dedup_preprocessed.npy  (167245, 512)
  data/cached_latents/text_group_ids_dedup.npy                (167245,)
  results/generated_embeddings.npy                             (6900, 512)
  results/generated_labels.npy                                 (6900,)
  results/real_expression.npy                                  (1932, 1790)
  results/real_expression_labels.npy                           (1932,)
  results/generated_expression.npy                             (2000, 1790)
  results/generated_expression_labels.npy                      (2000,)
  data/cached_latents/text_captions_deduplicated.json

Outputs (written next to this script):
  rare_failure_mechanism.json
  rare_failure_preview.txt
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]

REAL_LATENT = REPO_ROOT / "data" / "cached_latents" / "cell_embeddings_dedup_preprocessed.npy"
REAL_LATENT_LABELS = REPO_ROOT / "data" / "cached_latents" / "text_group_ids_dedup.npy"
GEN_LATENT = REPO_ROOT / "results" / "generated_embeddings.npy"
GEN_LATENT_LABELS = REPO_ROOT / "results" / "generated_labels.npy"
REAL_EXPR = REPO_ROOT / "results" / "real_expression.npy"
REAL_EXPR_LABELS = REPO_ROOT / "results" / "real_expression_labels.npy"
GEN_EXPR = REPO_ROOT / "results" / "generated_expression.npy"
GEN_EXPR_LABELS = REPO_ROOT / "results" / "generated_expression_labels.npy"
CAPTIONS = REPO_ROOT / "data" / "cached_latents" / "text_captions_deduplicated.json"

REVIEWER_FLAGGED_CYCLING_ID = 4


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


def intra_cos(x: np.ndarray) -> float:
    """Mean pairwise cosine similarity inside a cluster."""
    if x.shape[0] < 2:
        return float("nan")
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    # sample up to 400 for speed
    n = min(xn.shape[0], 400)
    idx = np.random.default_rng(0).choice(xn.shape[0], n, replace=False)
    s = xn[idx]
    sim = s @ s.T
    iu = np.triu_indices(n, k=1)
    return float(sim[iu].mean())


def within_variance(x: np.ndarray) -> dict:
    """Return total variance (trace of cov), mean per-dim variance, intra_cos."""
    if x.shape[0] < 2:
        return {"n": int(x.shape[0]), "total_var": float("nan"),
                "mean_var": float("nan"), "intra_cos": float("nan")}
    v = np.var(x, axis=0, dtype=np.float64)
    return {
        "n": int(x.shape[0]),
        "total_var": float(v.sum()),
        "mean_var": float(v.mean()),
        "intra_cos": intra_cos(x),
    }


def classify_bottleneck(latent_ratio: float, expr_ratio: float,
                        tol: float = 0.3) -> str:
    """Map (latent_var_ratio, expr_var_ratio) to a mechanism label."""
    low_lat = latent_ratio < 1 - tol
    low_exp = expr_ratio < 1 - tol
    high_lat = latent_ratio > 1 + tol
    if low_lat and low_exp:
        return "upstream+downstream (both compressed)"
    if low_lat and not low_exp:
        return "upstream compression, decoder expands"
    if not low_lat and low_exp:
        return "downstream compression (decoder collapses latent spread)"
    if high_lat and low_exp:
        return "latent overspread, decoder collapses"
    if high_lat and not low_exp:
        return "latent overspread; decoder preserves"
    return "approximately matched"


def main() -> None:
    captions = json.loads(CAPTIONS.read_text())
    id_to_name = {int(k): short_name(v) for k, v in captions.items()}

    real_lat = np.load(REAL_LATENT, mmap_mode="r")
    real_lat_lbl = np.load(REAL_LATENT_LABELS)
    gen_lat = np.load(GEN_LATENT)
    gen_lat_lbl = np.load(GEN_LATENT_LABELS)
    real_expr = np.load(REAL_EXPR)
    real_expr_lbl = np.load(REAL_EXPR_LABELS)
    gen_expr = np.load(GEN_EXPR)
    gen_expr_lbl = np.load(GEN_EXPR_LABELS)

    # Rare types: bottom-quartile training count
    uniq, cnt = np.unique(real_lat_lbl, return_counts=True)
    q25 = int(np.percentile(cnt, 25))
    rare_by_count = sorted(
        [(int(u), int(c)) for u, c in zip(uniq, cnt) if c <= q25],
        key=lambda x: x[1]
    )

    # Type list to analyse: bottom-quartile rare + the reviewer-flagged
    # Cycling type (often not rare by count but is biologically rare in
    # most studies).
    targets = {int(u) for u, _ in rare_by_count}
    targets.add(REVIEWER_FLAGGED_CYCLING_ID)

    # Reference non-rare types (top decile) for contrast
    top_decile = sorted(
        [(int(u), int(c)) for u, c in zip(uniq, cnt)],
        key=lambda x: -x[1]
    )[:5]
    reference = {int(u) for u, _ in top_decile}

    per_type = []
    for gid in sorted(targets | reference):
        r_lat = np.asarray(real_lat[real_lat_lbl == gid])
        g_lat = gen_lat[gen_lat_lbl == gid]
        r_exp = real_expr[real_expr_lbl == gid]
        g_exp = gen_expr[gen_expr_lbl == gid]

        lat_real = within_variance(r_lat)
        lat_gen = within_variance(g_lat)
        exp_real = within_variance(r_exp)
        exp_gen = within_variance(g_exp)

        def ratio(num: dict, den: dict, key: str) -> float:
            a, b = num.get(key), den.get(key)
            if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b == 0:
                return float("nan")
            return a / b

        lat_ratio_total = ratio(lat_gen, lat_real, "total_var")
        exp_ratio_total = ratio(exp_gen, exp_real, "total_var")

        per_type.append({
            "group_id": gid,
            "cell_type": id_to_name.get(gid, f"group_{gid}"),
            "training_count": int(cnt[uniq == gid][0]) if gid in uniq else None,
            "is_rare_by_count": gid in {u for u, _ in rare_by_count},
            "is_reviewer_flagged_cycling": gid == REVIEWER_FLAGGED_CYCLING_ID,
            "is_reference": gid in reference,
            "latent_real": lat_real,
            "latent_gen": lat_gen,
            "expression_real": exp_real,
            "expression_gen": exp_gen,
            "ratios": {
                "latent_total_var_gen_over_real": lat_ratio_total,
                "latent_intra_cos_gen_over_real":
                    ratio(lat_gen, lat_real, "intra_cos"),
                "expression_total_var_gen_over_real": exp_ratio_total,
                "expression_intra_cos_gen_over_real":
                    ratio(exp_gen, exp_real, "intra_cos"),
            },
            "mechanism_verdict": classify_bottleneck(
                lat_ratio_total, exp_ratio_total, tol=0.3
            ),
        })

    # Aggregate mechanism verdicts across rare types
    from collections import Counter
    mech_count_rare = Counter(
        p["mechanism_verdict"] for p in per_type if p["is_rare_by_count"]
    )
    mech_count_ref = Counter(
        p["mechanism_verdict"] for p in per_type if p["is_reference"]
    )

    summary = {
        "inputs": {
            "real_latent": {"path": str(REAL_LATENT.relative_to(REPO_ROOT)),
                            "shape": list(real_lat.shape)},
            "generated_latent": {"path": str(GEN_LATENT.relative_to(REPO_ROOT)),
                                 "shape": list(gen_lat.shape)},
            "real_expression": {"path": str(REAL_EXPR.relative_to(REPO_ROOT)),
                                "shape": list(real_expr.shape)},
            "generated_expression": {"path": str(GEN_EXPR.relative_to(REPO_ROOT)),
                                     "shape": list(gen_expr.shape)},
        },
        "definitions": {
            "rare_by_count": f"training_count <= Q25 ({q25})",
            "reference_types": "top-5 by training count",
            "mechanism_tolerance": 0.3,
            "mechanism_labels": [
                "upstream+downstream (both compressed)",
                "upstream compression, decoder expands",
                "downstream compression (decoder collapses latent spread)",
                "latent overspread, decoder collapses",
                "latent overspread; decoder preserves",
                "approximately matched",
            ],
        },
        "mechanism_counts": {
            "rare_by_count_types":
                {k: v for k, v in mech_count_rare.items()},
            "reference_types":
                {k: v for k, v in mech_count_ref.items()},
        },
        "per_type": per_type,
    }

    (HERE / "rare_failure_mechanism.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: float(o)
                   if isinstance(o, np.floating) else None) + "\n"
    )

    # --- preview ----------------------------------------------------
    lines = [
        "A5 Rare-cell failure mechanism",
        "==============================",
        f"Q25 training count = {q25}  "
        f"({len(rare_by_count)} types are rare-by-count)",
        "",
        f"{'gid':3s} {'n_train':>7s} {'cell_type':38s} "
        f"{'lat_ratio':>9s} {'exp_ratio':>9s} {'latICgen/real':>12s} "
        f"{'mechanism':40s}",
        "-" * 130,
    ]
    def fmt(p):
        r = p["ratios"]
        return (
            f"{p['group_id']:>3d} "
            f"{p.get('training_count', 0):>7d} "
            f"{p['cell_type'][:38]:38s} "
            f"{r['latent_total_var_gen_over_real']:>9.3f} "
            f"{r['expression_total_var_gen_over_real']:>9.3f} "
            f"{r['latent_intra_cos_gen_over_real']:>12.3f} "
            f"{p['mechanism_verdict']:40s}"
        )

    lines.append("-- Reviewer-flagged Cycling (group 4) --")
    for p in per_type:
        if p["is_reviewer_flagged_cycling"]:
            lines.append(fmt(p))
    lines += ["", "-- Rare-by-training-count (bottom quartile) --"]
    for p in per_type:
        if p["is_rare_by_count"] and not p["is_reviewer_flagged_cycling"]:
            lines.append(fmt(p))
    lines += ["", "-- Reference (top-5 by training count) --"]
    for p in per_type:
        if p["is_reference"]:
            lines.append(fmt(p))
    lines += ["", "Mechanism counts over rare-by-count types:"]
    for k, v in mech_count_rare.most_common():
        lines.append(f"  {v:3d} x  {k}")
    lines.append("")
    lines += ["Mechanism counts over reference (non-rare) types:"]
    for k, v in mech_count_ref.most_common():
        lines.append(f"  {v:3d} x  {k}")

    preview = "\n".join(lines) + "\n"
    (HERE / "rare_failure_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
