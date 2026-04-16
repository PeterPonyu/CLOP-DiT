#!/usr/bin/env python3
"""Lane C — zero-shot strict-OOD evaluation of CLOP-DiT.

Answers R3.1 directly: can the production CLOP+DiT generate biologically
plausible latents for entirely unseen tissues (kidney, cerebellum, testis)
without any retraining?

Pipeline per tissue:
    1. Load OOD h5ad (pre-QC'd, HVG-selected by scripts/data_prep/04b)
    2. Encode real cells with frozen scGPT → (N_real, 512)
    3. For each cell type with >= MIN_CELLS cells:
         a. Build a structured biological prompt (no training-corpus mention)
         b. Generate N_GEN latents via DiT conditioned on the prompt
         c. Compute: centroid cosine, nearest-centroid accuracy (KNN-1 on
            type centroids), Frechet distance, mean pairwise cosine gen↔real
    4. Emit per-cell-type + pooled metrics

Output:
    revision/experiments/lane_c_data/zero_shot_results.json
    revision/experiments/lane_c_data/zero_shot_preview.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.linalg as la
from scipy import sparse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_config import setup_logging  # noqa: E402
from src.architecture.scgpt_embed import ScGPTCellEncoder  # noqa: E402
from src.data_pipeline.embedding_preprocessor import EmbeddingPreprocessor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Prompt templates — match the structured-biological-description format
# used during training, but fill with zero-shot OOD content.
# ═══════════════════════════════════════════════════════════════════════

# Mapping CellxGene cell_type labels → structured prompts.
# Each prompt covers: cell type identity, tissue, species, marker genes,
# disease context.  Phrasing mirrors training-style descriptions.

PROMPTS = {
    # ─── Kidney (census_kidney_processed.h5ad) ───
    "kidney epithelial cell": (
        "Kidney epithelial cells from healthy adult human kidney. "
        "These cells form the nephron tubules and collecting ducts, expressing "
        "marker genes such as SLC34A1, LRP2, CUBN, AQP1, UMOD, and SLC12A3. "
        "Sampled from normal human renal cortex without disease."
    ),
    "endothelial cell": (
        "Endothelial cells from healthy adult human kidney. "
        "Vascular endothelial cells lining glomerular and peritubular capillaries, "
        "expressing PECAM1, VWF, KDR, and CDH5. From normal renal tissue."
    ),
    # ─── Cerebellum (census_cerebellum_processed.h5ad) ───
    "Purkinje cell": (
        "Purkinje cells from healthy adult human cerebellum. "
        "Large GABAergic projection neurons of the cerebellar cortex with "
        "extensively branched dendritic trees, expressing CALB1, PCP2, PVALB, "
        "GRID2, and ITPR1. From normal cerebellar cortex without neurological disease."
    ),
    "granule cell": (
        "Cerebellar granule cells from healthy adult human cerebellum. "
        "Small excitatory glutamatergic neurons in the granular layer, expressing "
        "GABRA6, CBLN1, NEUROD1, and RBFOX3. From normal cerebellar cortex."
    ),
    "interneuron": (
        "Cerebellar interneurons from healthy adult human cerebellum. "
        "Local-circuit inhibitory neurons (basket, stellate, Golgi) in the "
        "cerebellar molecular and granular layers, expressing GAD1, GAD2, "
        "PVALB, and SLC32A1. From normal cerebellar tissue."
    ),
    "glial cell": (
        "Glial cells from healthy adult human cerebellum. "
        "Astrocytes and oligodendrocyte precursors supporting cerebellar neurons, "
        "expressing GFAP, AQP4, SLC1A3, and OLIG1. From normal cerebellar tissue."
    ),
    "ependymal cell": (
        "Ependymal cells from healthy adult human cerebellum. "
        "Ciliated epithelial cells lining the fourth ventricle and choroid plexus "
        "adjacent to cerebellar tissue, expressing FOXJ1, DNAH11, and S100B."
    ),
    "inhibitory interneuron": (
        "Inhibitory interneurons from healthy adult human cerebellum. "
        "GABAergic local-circuit neurons in the cerebellar cortex expressing "
        "GAD1, SLC32A1, PVALB, and NPY. From normal cerebellar molecular layer."
    ),
    "Bergmann glial cell": (
        "Bergmann glial cells from healthy adult human cerebellum. "
        "Specialized radial glia in the Purkinje-cell layer with processes "
        "extending through the molecular layer, expressing GFAP, AQP4, SLC1A3, "
        "and S100B. From normal cerebellar cortex."
    ),
    "CNS interneuron": (
        "Central-nervous-system interneurons from healthy adult human cerebellum. "
        "Local-circuit neurons expressing classical interneuron markers GAD1, "
        "GAD2, SLC32A1, and PVALB. From normal cerebellar tissue."
    ),
    "brainstem motor neuron": (
        "Brainstem motor neurons adjacent to healthy adult human cerebellum. "
        "Cholinergic motor neurons of the brainstem nuclei (vestibular, "
        "oculomotor), expressing CHAT, ISL1, PHOX2B, and MNX1."
    ),
    "cerebellar granule cell precursor": (
        "Cerebellar granule cell precursors from healthy human cerebellum. "
        "Proliferating neural progenitors in the external granular layer "
        "expressing ATOH1, NEUROD1, MKI67, and PAX6. From developing/mature cerebellar tissue."
    ),
    # ─── Testis-fetal (census_testis_fetal_processed.h5ad) ───
    "Leydig cell": (
        "Leydig cells from healthy human fetal testis. "
        "Steroidogenic interstitial cells of the testis producing testosterone, "
        "expressing CYP17A1, INSL3, STAR, HSD3B2, and NR5A1. From normal fetal "
        "gonadal tissue without disease."
    ),
    "type I cell of adrenal cortex": (
        "Type I cells of the adrenal cortex from healthy human fetal adrenal gland. "
        "Steroidogenic cortical cells in the zona fetalis producing DHEA and "
        "cortisol precursors, expressing CYP17A1, CYP11A1, STAR, and NR5A1. "
        "From normal fetal adrenal tissue."
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════

def _frechet_gaussian(real: np.ndarray, gen: np.ndarray) -> float:
    """Frechet distance assuming Gaussian embeddings."""
    mu_r, mu_g = real.mean(0), gen.mean(0)
    diff = mu_r - mu_g
    cov_r = np.cov(real, rowvar=False)
    cov_g = np.cov(gen, rowvar=False)
    cov_prod, _ = la.sqrtm(cov_r @ cov_g, disp=False)
    if np.iscomplexobj(cov_prod):
        cov_prod = cov_prod.real
    return float(diff @ diff + np.trace(cov_r + cov_g - 2.0 * cov_prod))


def _centroid_cosine(real: np.ndarray, gen: np.ndarray) -> float:
    cr, cg = real.mean(0), gen.mean(0)
    return float(np.dot(cr, cg) / (np.linalg.norm(cr) * np.linalg.norm(cg) + 1e-8))


def _nearest_centroid_accuracy(
    gen: np.ndarray,
    real_centroids: dict[str, np.ndarray],
    target_type: str,
) -> float:
    """For each generated cell, does its nearest real-type centroid == target_type?

    Tests whether the model's OOD generations land in a tight cluster whose
    nearest real-cell centroid is the correct type.
    """
    names = list(real_centroids.keys())
    C = np.stack([real_centroids[n] for n in names], axis=0)  # (T, D)
    # Cosine similarity
    gen_n = gen / (np.linalg.norm(gen, axis=1, keepdims=True) + 1e-8)
    C_n = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    sims = gen_n @ C_n.T  # (N, T)
    preds = [names[i] for i in sims.argmax(axis=1)]
    correct = sum(p == target_type for p in preds)
    return float(correct / len(gen))


def _mean_pairwise_cosine(gen: np.ndarray, real: np.ndarray, sample: int = 500) -> float:
    """Mean cosine similarity between each generated cell and a sample of real cells."""
    rng = np.random.default_rng(42)
    n_real = len(real)
    if n_real > sample:
        real = real[rng.choice(n_real, sample, replace=False)]
    gen_n = gen / (np.linalg.norm(gen, axis=1, keepdims=True) + 1e-8)
    real_n = real / (np.linalg.norm(real, axis=1, keepdims=True) + 1e-8)
    return float((gen_n @ real_n.T).mean())


# ═══════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════

MIN_CELLS_PER_TYPE = 30
DEFAULT_NUM_GEN = 200


def load_inference():
    """Load CLOPDiTInference from the numbered script."""
    spec = importlib.util.spec_from_file_location(
        "inference", str(REPO_ROOT / "scripts" / "inference" / "05_inference.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CLOPDiTInference


def encode_real_cells_scgpt(
    h5ad_path: Path,
    scgpt_dir: Path,
    preprocessor: EmbeddingPreprocessor,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode OOD cells with frozen scGPT and apply saved ZCA preprocessor.

    The DiT was trained on ZCA-whitened scGPT embeddings
    (cell_embeddings_dedup_preprocessed.npy), so real cells must go
    through the same whitening pipeline to live in the comparable
    latent space.
    """
    logger.info(f"Loading {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    logger.info(f"  shape: {adata.shape}")

    encoder = ScGPTCellEncoder(str(scgpt_dir))
    raw = encoder.encode(adata)  # (N, 512) raw scGPT
    whitened = preprocessor.transform(raw)
    labels = adata.obs["cell_type"].astype(str).values
    logger.info(
        f"  scGPT embeddings: {raw.shape} → whitened: {whitened.shape}, "
        f"types: {len(np.unique(labels))}"
    )
    return whitened, labels


def evaluate_tissue(
    tissue_id: str,
    h5ad_path: Path,
    inference,
    scgpt_dir: Path,
    preprocessor: EmbeddingPreprocessor,
    num_gen: int,
    num_steps: int,
    cfg_scale: float,
) -> dict:
    """Run zero-shot eval for one tissue."""
    real, labels = encode_real_cells_scgpt(h5ad_path, scgpt_dir, preprocessor)
    unique_types, counts = np.unique(labels, return_counts=True)
    real_centroids = {t: real[labels == t].mean(0) for t in unique_types}

    per_type = {}
    for t, n in zip(unique_types, counts):
        if n < MIN_CELLS_PER_TYPE:
            logger.info(f"  [skip] {t}: only {n} cells")
            continue
        if t not in PROMPTS:
            logger.warning(f"  [missing prompt] {t}: skipping")
            continue

        logger.info(f"  Generating {num_gen} cells for '{t}' (real N={n})")
        gen = inference.generate(
            prompt=PROMPTS[t],
            num_cells=num_gen,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            seed=42,
        )
        # Match the L2-normalisation applied during DiT training/eval (the
        # preprocessed cell embeddings are unit-norm; 05_inference.generate
        # does not normalise, so do it here)
        gen = gen / (np.linalg.norm(gen, axis=1, keepdims=True) + 1e-8)
        real_t = real[labels == t]

        per_type[t] = {
            "n_real": int(n),
            "n_gen": int(num_gen),
            "centroid_cosine": _centroid_cosine(real_t, gen),
            "nearest_centroid_acc": _nearest_centroid_accuracy(gen, real_centroids, t),
            "frechet_distance": _frechet_gaussian(real_t, gen),
            "mean_pairwise_cosine_gen_vs_real": _mean_pairwise_cosine(gen, real_t),
        }
        logger.info(
            f"    centroid_cos={per_type[t]['centroid_cosine']:.3f} "
            f"nearest_acc={per_type[t]['nearest_centroid_acc']:.3f} "
            f"FD={per_type[t]['frechet_distance']:.3f}"
        )

    # Tissue-level pooled summary (means across types)
    if per_type:
        summary = {
            "n_types_evaluated": len(per_type),
            "mean_centroid_cosine": float(np.mean([v["centroid_cosine"] for v in per_type.values()])),
            "mean_nearest_centroid_acc": float(np.mean([v["nearest_centroid_acc"] for v in per_type.values()])),
            "mean_frechet_distance": float(np.mean([v["frechet_distance"] for v in per_type.values()])),
            "mean_gen_vs_real_cosine": float(np.mean([v["mean_pairwise_cosine_gen_vs_real"] for v in per_type.values()])),
        }
    else:
        summary = {"n_types_evaluated": 0}

    return {"tissue_id": tissue_id, "per_type": per_type, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit-checkpoint", default="models/checkpoints/dit_best.pth")
    parser.add_argument("--clop-checkpoint", default="models/checkpoints/clop_best.pth")
    parser.add_argument("--scgpt-dir", default="models/scgpt_human")
    parser.add_argument("--num-gen", type=int, default=DEFAULT_NUM_GEN)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    args = parser.parse_args()

    setup_logging()

    manifest_path = REPO_ROOT / "data" / "processed_h5ad_revision" / "MANIFEST.csv"
    import csv
    tissues = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if row["split"] == "heldout":
                tissues.append({"id": row["dataset_id"], "path": REPO_ROOT / row["source_path"]})
    logger.info(f"Found {len(tissues)} heldout tissues in MANIFEST")

    logger.info(f"Loading CLOPDiTInference (DiT: {args.dit_checkpoint}, CLOP: {args.clop_checkpoint})")
    CLOPDiTInference = load_inference()
    inference = CLOPDiTInference(
        dit_checkpoint=args.dit_checkpoint,
        clop_checkpoint=args.clop_checkpoint,
    )
    scgpt_dir = Path(args.scgpt_dir)

    # Load the ZCA cell preprocessor used during DiT training
    preprocessor_path = REPO_ROOT / "data" / "cached_latents" / "cell_preprocessor_preprocessed.npz"
    logger.info(f"Loading cell preprocessor from {preprocessor_path}")
    preprocessor = EmbeddingPreprocessor.load(str(preprocessor_path))
    logger.info(f"  method={preprocessor.method}, fitted={preprocessor.fitted_}")

    results = {}
    for t in tissues:
        logger.info(f"=== Evaluating {t['id']} ===")
        results[t["id"]] = evaluate_tissue(
            tissue_id=t["id"],
            h5ad_path=t["path"],
            inference=inference,
            scgpt_dir=scgpt_dir,
            preprocessor=preprocessor,
            num_gen=args.num_gen,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
        )

    # Overall five-slice summary
    overall = {
        "mean_centroid_cosine": float(np.mean([
            v["summary"]["mean_centroid_cosine"]
            for v in results.values()
            if v["summary"].get("n_types_evaluated", 0) > 0
        ])),
        "mean_nearest_centroid_acc": float(np.mean([
            v["summary"]["mean_nearest_centroid_acc"]
            for v in results.values()
            if v["summary"].get("n_types_evaluated", 0) > 0
        ])),
        "mean_frechet_distance": float(np.mean([
            v["summary"]["mean_frechet_distance"]
            for v in results.values()
            if v["summary"].get("n_types_evaluated", 0) > 0
        ])),
    }

    output = {
        "experiment": "lane_c_zero_shot_strict_ood",
        "generation_config": {
            "num_gen_per_type": args.num_gen,
            "num_steps": args.num_steps,
            "cfg_scale": args.cfg_scale,
            "dit_checkpoint": args.dit_checkpoint,
            "clop_checkpoint": args.clop_checkpoint,
        },
        "per_tissue": results,
        "overall": overall,
    }

    out_json = HERE / "zero_shot_results.json"
    out_json.write_text(json.dumps(output, indent=2, default=float) + "\n")
    logger.info(f"Wrote {out_json}")

    # Human-readable preview
    lines = [
        "Lane C — zero-shot strict-OOD CLOP-DiT evaluation",
        "=" * 62,
        "",
        f"{'tissue':<24s} {'types':>6s} {'cos':>7s} {'near_acc':>10s} {'FD':>8s}",
        "-" * 62,
    ]
    for t_id, v in results.items():
        s = v["summary"]
        if s.get("n_types_evaluated", 0) == 0:
            lines.append(f"{t_id:<24s} {'0':>6s}  (no eligible types)")
            continue
        lines.append(
            f"{t_id:<24s} {s['n_types_evaluated']:>6d} "
            f"{s['mean_centroid_cosine']:>7.3f} "
            f"{s['mean_nearest_centroid_acc']:>10.3f} "
            f"{s['mean_frechet_distance']:>8.3f}"
        )
    lines.append("")
    lines.append(
        f"Overall: centroid_cos={overall['mean_centroid_cosine']:.3f} "
        f"nearest_acc={overall['mean_nearest_centroid_acc']:.3f} "
        f"FD={overall['mean_frechet_distance']:.3f}"
    )
    preview = "\n".join(lines) + "\n"
    (HERE / "zero_shot_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
