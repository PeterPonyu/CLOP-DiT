"""B1 decoder-level extension — expression-scale variance baselines.

Completes the R2.9 answer at the gene-expression scale by comparing
CLOP-DiT against two closed-form oracles computed directly on the
decoded-expression matrices. Because we already have paired real and
CLOP-DiT-decoded expression matrices in `results/`, we can fit
Gaussian baselines *at the expression scale* without re-running the
decoder — the resulting oracles are "what would a moment-matching
generator look like if it lived at the expression level".

Three generators compared on the same evaluation slice
(1 932 real cells, 2 000 CLOP-DiT generated cells, 1 790 genes, 69
types):

  - CLOP-DiT  : results/generated_expression.npy
  - Gaussian-per-type (expression-scale oracle): per type t, fit
                       diagonal Gaussian with mean mu_t, per-gene
                       variance var_t on the real expression of that
                       type, then sample per-type counts matching
                       CLOP-DiT. Matches R_mean and per-gene within-
                       type variance by construction.
  - Pooled Gaussian (CFG=0 at expression level): single diagonal
                       Gaussian fit on all real expression; type
                       labels borrowed from CLOP-DiT for per-type
                       metric computation. Ignores type structure.

Metrics reported (following the cross-dataset validation schema
`mean_expr_pearson_r` / `median_variance_ratio` so numbers are
directly comparable to the baseline snapshot):

  - r_mean  : Pearson r of per-gene mean expression, pooled
  - r_var   : Pearson r of per-gene variance, pooled
  - median_variance_ratio : median_g (gen_var_g / real_var_g), pooled
  - within-type mean r_var : mean across types of within-type Pearson
  - within-type pooled median variance ratio
  - fraction of types with positive within-type r_var

Outputs (next to this script):
  expression_baselines.json
  expression_baselines_preview.txt
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]

REAL_EXPR = REPO_ROOT / "results" / "real_expression.npy"
REAL_LABELS = REPO_ROOT / "results" / "real_expression_labels.npy"
GEN_EXPR = REPO_ROOT / "results" / "generated_expression.npy"
GEN_LABELS = REPO_ROOT / "results" / "generated_expression_labels.npy"

SEED = 0


def sha256_head(path: Path, limit: int = 2**26) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def sample_diag_gaussian(mu: np.ndarray, var: np.ndarray, n: int,
                         rng: np.random.Generator) -> np.ndarray:
    std = np.sqrt(np.clip(var, 1e-12, None))
    return mu + rng.standard_normal((n, mu.shape[0])) * std


def pooled_metrics(real: np.ndarray, gen: np.ndarray) -> dict:
    m_real = real.mean(axis=0, dtype=np.float64)
    m_gen = gen.mean(axis=0, dtype=np.float64)
    v_real = real.var(axis=0, dtype=np.float64)
    v_gen = gen.var(axis=0, dtype=np.float64)
    ok = (v_real > 1e-12) & np.isfinite(v_real) & np.isfinite(v_gen)
    r_mean = float(np.corrcoef(m_real, m_gen)[0, 1])
    r_var = float(np.corrcoef(v_real[ok], v_gen[ok])[0, 1])
    ratio = v_gen[ok] / v_real[ok]
    return {
        "n_genes_scored": int(ok.sum()),
        "r_mean_pooled": r_mean,
        "r_var_pooled": r_var,
        "median_variance_ratio_pooled": float(np.median(ratio)),
        "median_std_ratio_pooled": float(np.median(np.sqrt(ratio))),
    }


def within_type_metrics(real: np.ndarray, gen: np.ndarray,
                        real_l: np.ndarray, gen_l: np.ndarray) -> dict:
    shared = sorted(set(real_l.tolist()) & set(gen_l.tolist()))
    rvars = []
    ratios_pool = []
    med_per_type = []
    per_type = {}
    for t in shared:
        rt = real[real_l == t]
        gt = gen[gen_l == t]
        if rt.shape[0] < 3 or gt.shape[0] < 3:
            continue
        vr = rt.var(axis=0, dtype=np.float64)
        vg = gt.var(axis=0, dtype=np.float64)
        ok = (vr > 1e-12) & np.isfinite(vr) & np.isfinite(vg)
        if ok.sum() < 20:
            continue
        corr = np.corrcoef(vr[ok], vg[ok])[0, 1]
        ratio_t = vg[ok] / vr[ok]
        if np.isfinite(corr):
            rvars.append(float(corr))
        ratios_pool.append(ratio_t)
        med_per_type.append(float(np.median(ratio_t)))
        per_type[int(t)] = {
            "n_real": int(rt.shape[0]),
            "n_gen": int(gt.shape[0]),
            "r_var": float(corr) if np.isfinite(corr) else None,
            "median_ratio": float(np.median(ratio_t)),
        }
    pooled = np.concatenate(ratios_pool) if ratios_pool else np.array([])
    return {
        "n_types_scored": len(rvars),
        "mean_r_var_across_types": float(np.mean(rvars)) if rvars else None,
        "median_r_var_across_types": float(np.median(rvars)) if rvars else None,
        "frac_types_positive_r_var":
            float(np.mean([v > 0 for v in rvars])) if rvars else None,
        "pooled_within_median_ratio":
            float(np.median(pooled)) if pooled.size else None,
        "median_of_per_type_median_ratio":
            float(np.median(med_per_type)) if med_per_type else None,
        "per_type": per_type,
    }


def build_generator(real: np.ndarray, real_l: np.ndarray,
                    per_type_counts: dict[int, int],
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample a per-type diagonal Gaussian with sample counts matching a target."""
    rows = []
    lbls = []
    for t, n in per_type_counts.items():
        if n <= 0:
            continue
        rt = real[real_l == t]
        if rt.shape[0] < 2:
            continue
        mu = rt.mean(axis=0)
        var = rt.var(axis=0)
        rows.append(sample_diag_gaussian(mu, var, n, rng))
        lbls.extend([t] * n)
    return (np.concatenate(rows, axis=0).astype(real.dtype),
            np.array(lbls, dtype=real_l.dtype))


def main() -> None:
    rng = np.random.default_rng(SEED)
    real = np.load(REAL_EXPR)
    real_l = np.load(REAL_LABELS)
    gen = np.load(GEN_EXPR)
    gen_l = np.load(GEN_LABELS)

    shared_types = sorted(set(real_l.tolist()) & set(gen_l.tolist()))
    per_type_counts = {int(t): int((gen_l == t).sum()) for t in shared_types}

    # ---- Gaussian-per-type (expression-scale oracle) ---------------
    gauss, gauss_l = build_generator(real, real_l, per_type_counts, rng)

    # ---- Pooled Gaussian ------------------------------------------
    mu_pool = real.mean(axis=0)
    var_pool = real.var(axis=0)
    pooled = sample_diag_gaussian(mu_pool, var_pool, gen.shape[0], rng)
    pooled_l = gen_l.copy()

    generators = {
        "CLOP-DiT": (gen, gen_l,
                     "results/generated_expression.npy (CFG = 2.0)"),
        "Gaussian-per-type (diagonal, expression-scale oracle)":
            (gauss, gauss_l,
             "per-type diagonal Gaussian fit on real_expression.npy"),
        "Pooled Gaussian (CFG = 0 stand-in, expression-scale)":
            (pooled, pooled_l,
             "single diagonal Gaussian fit on all real_expression.npy"),
    }

    summary = {
        "seed": SEED,
        "inputs": {
            "real_expression": {
                "path": str(REAL_EXPR.relative_to(REPO_ROOT)),
                "shape": list(real.shape),
                "sha256_head": sha256_head(REAL_EXPR),
            },
            "generated_expression": {
                "path": str(GEN_EXPR.relative_to(REPO_ROOT)),
                "shape": list(gen.shape),
                "sha256_head": sha256_head(GEN_EXPR),
            },
        },
        "generators": {},
    }

    rows = []
    for name, (arr, lbl, desc) in generators.items():
        p = pooled_metrics(real, arr)
        w = within_type_metrics(real, arr, real_l, lbl)
        summary["generators"][name] = {
            "description": desc,
            "n_samples": int(arr.shape[0]),
            "pooled": p,
            "within_type": w,
        }
        rows.append((name, p, w))

    (HERE / "expression_baselines.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: float(o)
                   if isinstance(o, np.floating) else None) + "\n"
    )

    lines = [
        "B1 — decoder-level (expression-scale) variance baselines",
        "=========================================================",
        "",
        f"{'generator':52s}  {'r_mean':>7s} {'r_var':>7s} "
        f"{'med_var_ratio':>14s}  {'w_mean_r_var':>13s} "
        f"{'w_med_ratio':>12s} {'w_types+':>9s}",
        "-" * 130,
    ]
    for name, p, w in rows:
        lines.append(
            f"{name[:52]:52s}  "
            f"{p['r_mean_pooled']:>+7.3f} "
            f"{p['r_var_pooled']:>+7.3f} "
            f"{p['median_variance_ratio_pooled']:>14.3f}  "
            f"{w['mean_r_var_across_types']:>+13.3f} "
            f"{w['pooled_within_median_ratio']:>12.3f} "
            f"{w['frac_types_positive_r_var']*100:>8.1f}%"
        )
    preview = "\n".join(lines) + "\n"
    (HERE / "expression_baselines_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
