"""B1 — Latent-level Gaussian / pooled-Gaussian baselines against CLOP-DiT.

Addresses R2.9 ("is the near-zero variance recovery CLOP-DiT-specific
or shared by simpler baselines?") at the LATENT scale, which is where
the CLOP-DiT generator emits samples. Comparing at the latent scale
first removes decoder confounds; a decoder-level extension is scoped
as a follow-up.

Three generators are compared on the same evaluation slice:

  - CLOP-DiT  : the existing generated embeddings in results/.
  - Gaussian-per-type: for each cell type t, fit N(mu_t, Sigma_t) on the
                       preprocessed real latents of type t, then sample
                       the same number of cells as the CLOP-DiT
                       generation allocated to that type.
  - Pooled Gaussian: a single N(mu, Sigma) fit on all training latents
                     (organism-agnostic, type-agnostic). Approximates
                     an unconditional generator that has learnt only
                     the first two moments of the whole pool. Stands in
                     for CFG=0 without needing the DiT.

Metrics reported per generator:

  - median variance ratio gen_var / real_var (pooled over all types)
    -> what the pre-revision baseline labelled "r_var ~ 0".
  - gene-wise / dim-wise Pearson r_var (pooled)
  - within-type average variance ratio
  - within-type average Pearson r_var

All computations are on the 512-d CLOP latent space. The preprocessed
real latents in data/cached_latents are the same space the DiT emits.

Inputs (read-only):
  data/cached_latents/cell_embeddings_dedup_preprocessed.npy
  data/cached_latents/text_group_ids_dedup.npy
  results/generated_embeddings.npy
  results/generated_labels.npy

Outputs (next to this script):
  latent_baselines.json
  latent_baselines_preview.txt
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]

REAL_LATENT = REPO_ROOT / "data" / "cached_latents" / "cell_embeddings_dedup_preprocessed.npy"
REAL_LABELS = REPO_ROOT / "data" / "cached_latents" / "text_group_ids_dedup.npy"
GEN_LATENT = REPO_ROOT / "results" / "generated_embeddings.npy"
GEN_LABELS = REPO_ROOT / "results" / "generated_labels.npy"

SEED = 0


def sha256_head(path: Path, limit: int = 2**26) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def sample_gaussian(mu: np.ndarray, cov: np.ndarray, n: int,
                    rng: np.random.Generator, jitter: float = 1e-4) -> np.ndarray:
    """Sample n rows from N(mu, cov). Regularises cov before Cholesky."""
    d = mu.shape[0]
    reg = cov + jitter * np.eye(d)
    try:
        L = np.linalg.cholesky(reg)
    except np.linalg.LinAlgError:
        reg = cov + 1e-2 * np.eye(d)
        L = np.linalg.cholesky(reg)
    z = rng.standard_normal((n, d))
    return mu + z @ L.T


def variance_metrics(gen: np.ndarray, real: np.ndarray,
                     labels_gen: np.ndarray,
                     labels_real: np.ndarray) -> dict:
    """Return pooled and within-type variance-structure metrics."""
    assert gen.shape[1] == real.shape[1]
    pooled_real = np.var(real, axis=0, dtype=np.float64)
    pooled_gen = np.var(gen, axis=0, dtype=np.float64)
    ok_pool = (pooled_real > 1e-12) & np.isfinite(pooled_real)
    rp, gp = pooled_real[ok_pool], pooled_gen[ok_pool]
    ratio_pool = gp / rp
    pooled_r_var = float(np.corrcoef(rp, gp)[0, 1])

    shared_types = sorted(set(labels_real.tolist()) & set(labels_gen.tolist()))
    within_ratios = []
    within_r_vars = []
    within_med_std_ratios = []
    per_type = {}
    for t in shared_types:
        rt = real[labels_real == t]
        gt = gen[labels_gen == t]
        if rt.shape[0] < 3 or gt.shape[0] < 3:
            continue
        vr = np.var(rt, axis=0, dtype=np.float64)
        vg = np.var(gt, axis=0, dtype=np.float64)
        ok = (vr > 1e-12) & np.isfinite(vr) & np.isfinite(vg)
        if ok.sum() < 20:
            continue
        ratio = vg[ok] / vr[ok]
        std_ratio = np.sqrt(vg[ok]) / np.sqrt(vr[ok])
        corr = np.corrcoef(vr[ok], vg[ok])[0, 1]
        if np.isfinite(corr):
            within_r_vars.append(float(corr))
        within_ratios.append(ratio)
        within_med_std_ratios.append(float(np.median(std_ratio)))
        per_type[int(t)] = {
            "n_real": int(rt.shape[0]),
            "n_gen": int(gt.shape[0]),
            "median_var_ratio": float(np.median(ratio)),
            "pearson_r_var": float(corr) if np.isfinite(corr) else None,
            "total_var_ratio": float(vg.sum() / (vr.sum() + 1e-12)),
        }

    pooled_within = np.concatenate(within_ratios) if within_ratios else np.array([])

    return {
        "pooled": {
            "n_dims_scored": int(ok_pool.sum()),
            "pearson_r_var": pooled_r_var,
            "median_var_ratio": float(np.median(ratio_pool)),
            "median_std_ratio": float(np.median(np.sqrt(ratio_pool))),
            "total_var_ratio": float(gp.sum() / (rp.sum() + 1e-12)),
        },
        "within_type": {
            "n_types_scored": len(within_r_vars),
            "mean_pearson_r_var": (
                float(np.mean(within_r_vars)) if within_r_vars else None
            ),
            "median_pearson_r_var": (
                float(np.median(within_r_vars)) if within_r_vars else None
            ),
            "frac_types_positive_r_var": (
                float(np.mean([v > 0 for v in within_r_vars]))
                if within_r_vars else None
            ),
            "pooled_within_median_ratio": (
                float(np.median(pooled_within)) if pooled_within.size else None
            ),
            "median_of_per_type_median_std_ratio": (
                float(np.median(within_med_std_ratios))
                if within_med_std_ratios else None
            ),
        },
        "per_type": per_type,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    real = np.load(REAL_LATENT).astype(np.float32)      # (167245, 512)
    real_lbl = np.load(REAL_LABELS)
    gen = np.load(GEN_LATENT).astype(np.float32)        # (6900, 512)
    gen_lbl = np.load(GEN_LABELS)

    # ---- Gaussian-per-type sample ----------------------------------
    shared_types = sorted(set(real_lbl.tolist()) & set(gen_lbl.tolist()))
    per_type_counts = {int(t): int((gen_lbl == t).sum()) for t in shared_types}

    gauss_rows = []
    gauss_lbl = []
    per_type_fit_stats = {}
    for t in shared_types:
        rt = real[real_lbl == t]
        n_gen = per_type_counts[t]
        if n_gen == 0 or rt.shape[0] < 2:
            continue
        mu = rt.mean(axis=0)
        cov = np.cov(rt, rowvar=False)
        samples = sample_gaussian(mu, cov, n_gen, rng)
        gauss_rows.append(samples)
        gauss_lbl.extend([t] * n_gen)
        per_type_fit_stats[int(t)] = {
            "n_real": int(rt.shape[0]),
            "trace_cov": float(np.trace(cov)),
        }
    gauss = np.concatenate(gauss_rows, axis=0).astype(np.float32)
    gauss_lbl_arr = np.array(gauss_lbl, dtype=gen_lbl.dtype)

    # ---- Pooled Gaussian (CFG=0 stand-in) --------------------------
    mu_pool = real.mean(axis=0)
    cov_pool = np.cov(real, rowvar=False)
    pooled_total = gen.shape[0]
    pooled_samples = sample_gaussian(mu_pool, cov_pool, pooled_total, rng)
    # assign type labels matching CLOP-DiT distribution so per-type
    # metrics are comparable
    pooled_lbl = gen_lbl.copy()

    # ---- metrics ---------------------------------------------------
    metrics_clop = variance_metrics(gen, real, gen_lbl, real_lbl)
    metrics_gauss = variance_metrics(gauss, real, gauss_lbl_arr, real_lbl)
    metrics_pooled = variance_metrics(pooled_samples, real, pooled_lbl, real_lbl)

    summary = {
        "seed": SEED,
        "inputs": {
            "real_latent": {
                "path": str(REAL_LATENT.relative_to(REPO_ROOT)),
                "shape": list(real.shape),
                "sha256_head": sha256_head(REAL_LATENT),
            },
            "generated_latent": {
                "path": str(GEN_LATENT.relative_to(REPO_ROOT)),
                "shape": list(gen.shape),
                "sha256_head": sha256_head(GEN_LATENT),
            },
        },
        "generators": {
            "CLOP-DiT": {
                "n": int(gen.shape[0]),
                "metrics": metrics_clop,
            },
            "Gaussian-per-type": {
                "n": int(gauss.shape[0]),
                "description": "N(mu_t, Sigma_t) fit on real latents per "
                               "cell type, same per-type sample count as CLOP-DiT",
                "metrics": metrics_gauss,
            },
            "Pooled Gaussian (CFG=0 stand-in)": {
                "n": int(pooled_samples.shape[0]),
                "description": "Single N(mu, Sigma) fit on all training "
                               "latents; labels borrowed from CLOP-DiT "
                               "distribution only so per-type metrics are "
                               "computable. No per-type fitting.",
                "metrics": metrics_pooled,
            },
        },
    }

    (HERE / "latent_baselines.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: float(o)
                   if isinstance(o, np.floating) else None) + "\n"
    )

    def row(name: str, m: dict) -> str:
        p = m["pooled"]
        w = m["within_type"]
        return (f"{name:32s}  "
                f"pooled_r_var={p['pearson_r_var']:+.3f}  "
                f"pooled_med_ratio={p['median_var_ratio']:.3f}  "
                f"within_mean_r_var="
                f"{w['mean_pearson_r_var']:+.3f}  "
                f"within_pooled_med_ratio="
                f"{w['pooled_within_median_ratio']:.3f}  "
                f"types+r_var_pos={w['frac_types_positive_r_var']*100:5.1f}%")

    preview = (
        "B1 Latent-level variance-structure comparison\n"
        "==============================================\n\n"
        + row("CLOP-DiT", metrics_clop) + "\n"
        + row("Gaussian-per-type", metrics_gauss) + "\n"
        + row("Pooled Gaussian (CFG=0)", metrics_pooled) + "\n\n"
        "Interpretation reminder\n"
        "-----------------------\n"
        "- Gaussian-per-type matches real mean and covariance per type "
        "by construction; any deviation from perfect metrics is "
        "sampling noise at the per-type count level (many rare types "
        "have <100 real cells; the fit is noisy).\n"
        "- Pooled Gaussian (stand-in for CFG=0) ignores type structure "
        "entirely; it should have approximately zero within-type "
        "variance recovery if the pooled covariance is type-mixed.\n"
        "- CLOP-DiT's position relative to these two floors is the "
        "answer to R2.9.\n"
    )
    (HERE / "latent_baselines_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
