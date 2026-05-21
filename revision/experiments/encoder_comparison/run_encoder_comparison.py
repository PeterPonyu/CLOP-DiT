"""C — Encoder comparison: scGPT-human vs scGPT-pancancer vs PCA.

For a representative subset of datasets, embeds cells with three encoders
and compares within-type / between-type L2 tightness. Uses the same metric
as experiment A (raw-vs-latent variance) to determine whether the 6x
within-type compression is scGPT-architecture-specific or universal.

Encoders:
  - scGPT-human (production, 33M cells, 512-d) — uses cached embeddings
  - scGPT-pancancer (5.7M cancer cells, 512-d) — computed fresh
  - PCA (512-d, trivial non-neural baseline) — computed fresh

Expected outcome:
  - If both scGPT variants compress similarly → architecture limitation
  - If pancancer compresses differently → training data matters
  - PCA gives the non-neural reference point
"""

from __future__ import annotations

import json
import sys
import logging
import tempfile
import shutil
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import h5py

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

H5AD_DIR = REPO_ROOT / "data" / "processed_h5ad_cap3k_seed0"
CACHE_DIR = REPO_ROOT / "data" / "cached_latents_cap3k_seed0"
SUB_META = H5AD_DIR / "subcluster_metadata.json"

SCGPT_HUMAN = REPO_ROOT / "models" / "scgpt_human"
SCGPT_PANCANCER = REPO_ROOT / "models" / "scgpt_pancancer"


def read_h5ad_compat(path: Path):
    """Read h5ad with compatibility workaround for anndata 0.11 vs 0.12.

    anndata 0.12+ writes IOSpec(encoding_type='null') for None values in
    uns, which anndata 0.11.x can't read. Workaround: copy the file and
    strip the problematic entries via h5py before reading.
    """
    try:
        return sc.read_h5ad(path)
    except Exception:
        logger.info(f"  Patching h5ad for anndata compat: {path.name}")
        tmp = Path(tempfile.mktemp(suffix=".h5ad"))
        shutil.copy2(path, tmp)
        with h5py.File(tmp, "r+") as f:
            # Remove entries with 'null' encoding that old anndata can't read
            def _strip_null(group):
                to_del = []
                for key in group:
                    item = group[key]
                    if hasattr(item, "attrs"):
                        enc = item.attrs.get("encoding-type", b"").decode() \
                            if isinstance(item.attrs.get("encoding-type", b""), bytes) \
                            else str(item.attrs.get("encoding-type", ""))
                        if enc == "null":
                            to_del.append(key)
                        elif isinstance(item, h5py.Group):
                            _strip_null(item)
                for key in to_del:
                    del group[key]
            if "uns" in f:
                _strip_null(f["uns"])
        adata = sc.read_h5ad(tmp)
        tmp.unlink(missing_ok=True)
        return adata

# 8 representative datasets spanning the tightness ratio range
SELECTED_DATASETS = [
    "GSE167597_spineMm",              # 1.115 — highest compression
    "GSE262288_breastMetasisHmCancer", # 0.814
    "GSE145929_ProgastinMmDev",       # 0.792
    "GSE226131_HSCMmAged",            # 0.648 — middle
    "GSE222002_TcellsHmCancer",       # 0.646 — middle
    "GSE155109_bcECHmCancer",         # 0.459
    "dentate",                        # 0.458
    "setty",                          # 0.395 — lowest compression
]


def within_and_between(X: np.ndarray, labels: np.ndarray) -> tuple[float, float, int]:
    """Mean intra-cluster L2 (from centroid), mean inter-centroid L2, n_types."""
    types = np.unique(labels)
    intra_vals = []
    centroids = []
    for t in types:
        M = X[labels == t]
        if M.shape[0] < 2:
            continue
        mu = M.mean(axis=0)
        centroids.append(mu)
        intra_vals.append(float(np.linalg.norm(M - mu, axis=1).mean()))
    if len(centroids) < 2:
        return (float(np.mean(intra_vals)) if intra_vals else 0.0, 0.0, len(centroids))
    C = np.stack(centroids)
    n_c = C.shape[0]
    inter = []
    for i in range(n_c):
        for j in range(i + 1, n_c):
            inter.append(float(np.linalg.norm(C[i] - C[j])))
    return (float(np.mean(intra_vals)),
            float(np.mean(inter)) if inter else 0.0,
            n_c)


def tightness_stats(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute within/between/tightness for a given embedding space."""
    intra, inter, n_t = within_and_between(X, labels)
    return {
        "mean_within_L2": round(intra, 4),
        "mean_between_L2": round(inter, 4),
        "tightness": round(intra / inter, 4) if inter > 0 else float("nan"),
        "n_types": n_t,
    }


def load_scgpt_human_embeddings(ds_id: str) -> np.ndarray | None:
    """Load pre-cached scGPT-human embeddings for a dataset."""
    cells = np.load(CACHE_DIR / "cell_embeddings.npy")
    sample_ids = np.load(CACHE_DIR / "sample_ids.npy", allow_pickle=True)
    proc = json.loads((CACHE_DIR / "processed_datasets.json").read_text())
    if isinstance(proc, list):
        order = [Path(p).stem.replace("_processed", "") for p in proc]
    else:
        order = list(proc.keys())
    ds_to_ord = {ds: i for i, ds in enumerate(order)}
    if ds_id not in ds_to_ord:
        return None
    mask = sample_ids == ds_to_ord[ds_id]
    return cells[mask]


def encode_scgpt(adata, model_dir: Path, device: str = "cuda") -> np.ndarray:
    """Encode cells with a scGPT model."""
    from src.architecture.scgpt_embed import ScGPTCellEncoder
    encoder = ScGPTCellEncoder(
        model_dir=str(model_dir),
        device=device,
        max_length=1200,
        batch_size=128,
    )
    emb = encoder.encode(adata)
    return emb.astype(np.float32)


def encode_pca(adata, n_components: int = 512) -> np.ndarray:
    """Encode cells with PCA."""
    import scipy.sparse as sp
    max_comp = min(n_components, adata.shape[0] - 1, adata.shape[1] - 1)
    if sp.issparse(adata.X):
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=max_comp, random_state=42)
        emb = svd.fit_transform(adata.X)
    else:
        sc.tl.pca(adata, n_comps=max_comp)
        emb = adata.obsm["X_pca"]
    if emb.shape[1] < n_components:
        pad = np.zeros((emb.shape[0], n_components - emb.shape[1]))
        emb = np.concatenate([emb, pad], axis=1)
    return emb[:, :n_components].astype(np.float32)


def main() -> None:
    sub = json.loads(SUB_META.read_text())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    results = []

    for ds_id in SELECTED_DATASETS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {ds_id}")
        logger.info(f"{'='*60}")

        h5_path = H5AD_DIR / f"{ds_id}_processed.h5ad"
        if not h5_path.exists():
            logger.warning(f"  [SKIP] {h5_path} not found")
            continue

        # Load h5ad (with anndata compat workaround)
        adata = read_h5ad_compat(h5_path)
        n_cells = adata.shape[0]

        # Build cell-type labels from subcluster metadata
        if ds_id not in sub:
            logger.warning(f"  [SKIP] {ds_id} not in subcluster_metadata")
            continue

        ct_of_cell = np.full(n_cells, "Unknown", dtype=object)
        for cid, info in sub[ds_id].get("clusters", {}).items():
            idx = np.asarray(info.get("cell_indices", []), dtype=int)
            valid = idx[idx < n_cells]
            ct_of_cell[valid] = info.get("cell_type", "Unknown")

        known_mask = ct_of_cell != "Unknown"
        if known_mask.sum() < 20:
            logger.warning(f"  [SKIP] too few labeled cells ({known_mask.sum()})")
            continue
        types_present = np.unique(ct_of_cell[known_mask])
        if types_present.size < 2:
            logger.warning(f"  [SKIP] fewer than 2 cell types")
            continue

        labels = ct_of_cell[known_mask]
        logger.info(f"  {known_mask.sum()} labeled cells, {types_present.size} types")

        row = {"dataset": ds_id, "n_cells": int(known_mask.sum()),
               "n_types": int(types_present.size)}

        # ── 1. Raw HVG ──
        X_raw = adata.X
        if hasattr(X_raw, "toarray"):
            X_raw = X_raw.toarray()
        X_raw = np.asarray(X_raw)
        row["raw"] = tightness_stats(X_raw[known_mask], labels)
        logger.info(f"  raw: tightness={row['raw']['tightness']:.4f}")

        # ── 2. scGPT-human (cached) ──
        emb_human = load_scgpt_human_embeddings(ds_id)
        if emb_human is not None and emb_human.shape[0] == n_cells:
            row["scgpt_human"] = tightness_stats(emb_human[known_mask], labels)
            logger.info(f"  scgpt_human: tightness={row['scgpt_human']['tightness']:.4f}")
        else:
            logger.warning(f"  scgpt_human: cache mismatch or missing")
            row["scgpt_human"] = None

        # ── 3. scGPT-pancancer (computed fresh) ──
        try:
            emb_panc = encode_scgpt(adata, SCGPT_PANCANCER, device)
            row["scgpt_pancancer"] = tightness_stats(emb_panc[known_mask], labels)
            logger.info(f"  scgpt_pancancer: tightness={row['scgpt_pancancer']['tightness']:.4f}")
        except Exception as e:
            logger.error(f"  scgpt_pancancer FAILED: {e}")
            row["scgpt_pancancer"] = None

        # ── 4. PCA (512-d) ──
        try:
            emb_pca = encode_pca(adata, n_components=512)
            row["pca"] = tightness_stats(emb_pca[known_mask], labels)
            logger.info(f"  pca: tightness={row['pca']['tightness']:.4f}")
        except Exception as e:
            logger.error(f"  pca FAILED: {e}")
            row["pca"] = None

        # ── Compression ratios (latent/raw within-L2) ──
        raw_within = row["raw"]["mean_within_L2"]
        if raw_within > 0:
            for enc in ["scgpt_human", "scgpt_pancancer", "pca"]:
                if row.get(enc) and row[enc]["mean_within_L2"] > 0:
                    row[f"{enc}_within_ratio"] = round(
                        row[enc]["mean_within_L2"] / raw_within, 4
                    )
                    row[f"{enc}_tightness_ratio"] = round(
                        row[enc]["tightness"] / row["raw"]["tightness"], 4
                    ) if row["raw"]["tightness"] > 0 else None

        results.append(row)

    # ── Aggregate ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Aggregating {len(results)} datasets")

    def agg(key):
        vals = [r[key] for r in results if key in r and r[key] is not None]
        if not vals:
            return None
        return {
            "median": round(float(np.median(vals)), 4),
            "mean": round(float(np.mean(vals)), 4),
            "IQR": [round(float(np.percentile(vals, 25)), 4),
                    round(float(np.percentile(vals, 75)), 4)],
            "n": len(vals),
        }

    out = {
        "experiment": "encoder_comparison",
        "n_datasets": len(results),
        "selected_datasets": SELECTED_DATASETS,
        "encoders": ["scgpt_human", "scgpt_pancancer", "pca"],
        "aggregate": {
            "scgpt_human_within_ratio": agg("scgpt_human_within_ratio"),
            "scgpt_pancancer_within_ratio": agg("scgpt_pancancer_within_ratio"),
            "pca_within_ratio": agg("pca_within_ratio"),
            "scgpt_human_tightness_ratio": agg("scgpt_human_tightness_ratio"),
            "scgpt_pancancer_tightness_ratio": agg("scgpt_pancancer_tightness_ratio"),
            "pca_tightness_ratio": agg("pca_tightness_ratio"),
        },
        "per_dataset": results,
    }

    (HERE / "encoder_comparison_summary.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    # ── Human-readable preview ──
    lines = [
        "C — Encoder comparison: within-type compression across encoders",
        "=" * 65,
        f"Datasets: {len(results)} (representative subset)",
        "",
        "Per-dataset tightness (within_L2 / between_L2; lower = tighter clusters):",
        "",
        f"{'dataset':<40s} {'raw':>8s} {'human':>8s} {'pancan':>8s} {'pca':>8s}",
        "-" * 72,
    ]
    for r in results:
        lines.append(
            f"{r['dataset'][:40]:<40s} "
            f"{r['raw']['tightness']:>8.4f} "
            f"{r.get('scgpt_human', {}).get('tightness', '—'):>8} "
            f"{r.get('scgpt_pancancer', {}).get('tightness', '—'):>8} "
            f"{r.get('pca', {}).get('tightness', '—'):>8}"
        )

    lines.append("")
    lines.append("Within-L2 compression ratio (encoder / raw; lower = more compression):")
    lines.append("")
    lines.append(f"{'dataset':<40s} {'human':>8s} {'pancan':>8s} {'pca':>8s}")
    lines.append("-" * 64)
    for r in results:
        lines.append(
            f"{r['dataset'][:40]:<40s} "
            f"{r.get('scgpt_human_within_ratio', '—'):>8} "
            f"{r.get('scgpt_pancancer_within_ratio', '—'):>8} "
            f"{r.get('pca_within_ratio', '—'):>8}"
        )

    a = out["aggregate"]
    lines.append("")
    lines.append("Aggregate within-L2 compression ratio (encoder / raw):")
    for enc in ["scgpt_human", "scgpt_pancancer", "pca"]:
        s = a.get(f"{enc}_within_ratio")
        if s:
            lines.append(
                f"  {enc:<20s} median={s['median']:.4f}  "
                f"mean={s['mean']:.4f}  IQR={s['IQR']}"
            )

    lines.append("")
    lines.append("Aggregate tightness ratio (encoder / raw):")
    for enc in ["scgpt_human", "scgpt_pancancer", "pca"]:
        s = a.get(f"{enc}_tightness_ratio")
        if s:
            lines.append(
                f"  {enc:<20s} median={s['median']:.4f}  "
                f"mean={s['mean']:.4f}  IQR={s['IQR']}"
            )

    preview = "\n".join(lines) + "\n"
    (HERE / "encoder_comparison_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
