"""Fast B4 evaluation: compute generation_metrics.json for each variant.

Skips the expensive per-type full_evaluation and only computes:
- Overall FD/coverage/MMD (once, on subsampled data)
- Per-type centroid cosine similarity (cheap)
- Per-type FD using simple Gaussian assumption (fast)
"""
import json
import sys
import logging
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import GenerationMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VARIANTS = ["abl_baseline", "no_cohesion", "no_cell_noise"]


def frechet_distance_gaussian(real: np.ndarray, gen: np.ndarray) -> float:
    mu_r, mu_g = real.mean(0), gen.mean(0)
    diff = mu_r - mu_g
    cov_r = np.cov(real, rowvar=False)
    cov_g = np.cov(gen, rowvar=False)
    from scipy.linalg import sqrtm
    cov_prod = sqrtm(cov_r @ cov_g)
    if np.iscomplexobj(cov_prod):
        cov_prod = cov_prod.real
    return float(diff @ diff + np.trace(cov_r + cov_g - 2 * cov_prod))


def evaluate_variant(variant: str) -> dict | None:
    results_dir = HERE / f"results_{variant}"
    cache_dir = HERE / f"cache_{variant}"

    gen_path = results_dir / "generated_embeddings.npy"
    labels_path = results_dir / "generated_labels.npy"
    real_path = cache_dir / "cell_embeddings_dedup_preprocessed.npy"
    real_labels_path = cache_dir / "text_group_ids_dedup.npy"

    if not gen_path.exists():
        logger.warning(f"[SKIP] {variant}: no generated embeddings")
        return None

    gen_cells = np.load(gen_path)
    gen_labels = np.load(labels_path)
    real_cells = np.load(real_path)
    real_labels = np.load(real_labels_path)

    logger.info(f"{variant}: real={real_cells.shape}, gen={gen_cells.shape}")

    # Overall metrics (subsampled)
    rng = np.random.default_rng(42)
    max_n = min(len(real_cells), len(gen_cells), 5000)
    r_idx = rng.choice(len(real_cells), max_n, replace=False)
    g_idx = rng.choice(len(gen_cells), max_n, replace=False) if len(gen_cells) > max_n else np.arange(len(gen_cells))

    overall = GenerationMetrics.full_evaluation(real_cells[r_idx], gen_cells[g_idx])
    logger.info(f"  Overall: FD={overall['frechet_distance']:.4f}, "
                f"Coverage={overall['coverage']:.4f}, MMD={overall.get('mmd_rbf', 0):.6f}")

    # Per-type centroid cosine + fast FD
    type_ids = np.unique(gen_labels)
    type_names_path = cache_dir / "type_names.json"
    type_names = {}
    if type_names_path.exists():
        with open(type_names_path) as f:
            type_names = json.load(f)

    per_type = {}
    for t_id in type_ids:
        real_t = real_cells[real_labels == t_id]
        gen_t = gen_cells[gen_labels == t_id]
        if len(real_t) < 5 or len(gen_t) < 5:
            continue

        name = type_names.get(str(int(t_id)), f"Type_{t_id}")
        real_c = real_t.mean(0)
        gen_c = gen_t.mean(0)
        cos = float(np.dot(real_c, gen_c) / (np.linalg.norm(real_c) * np.linalg.norm(gen_c) + 1e-8))

        try:
            fd = frechet_distance_gaussian(real_t, gen_t)
        except Exception:
            fd = float("nan")

        per_type[name] = {
            "type_id": int(t_id),
            "n_real": int(len(real_t)),
            "n_gen": int(len(gen_t)),
            "centroid_cosine": cos,
            "frechet_distance": fd,
        }

    centroids = [v["centroid_cosine"] for v in per_type.values()]
    fds = [v["frechet_distance"] for v in per_type.values() if not np.isnan(v["frechet_distance"])]
    summary = {}
    if centroids:
        summary = {
            "mean_centroid_cosine": float(np.mean(centroids)),
            "min_centroid_cosine": float(np.min(centroids)),
            "max_centroid_cosine": float(np.max(centroids)),
            "std_centroid_cosine": float(np.std(centroids)),
            "num_types_evaluated": len(centroids),
        }
        if fds:
            summary["mean_fd"] = float(np.mean(fds))
        logger.info(f"  Per-type cosine: mean={np.mean(centroids):.4f}, min={np.min(centroids):.4f}")

    metrics = {"overall": overall, "per_type": per_type, "summary": summary}
    out_path = results_dir / "generation_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Saved → {out_path}")
    return metrics


def main():
    for v in VARIANTS:
        evaluate_variant(v)
    logger.info("All variants evaluated.")


if __name__ == "__main__":
    main()
