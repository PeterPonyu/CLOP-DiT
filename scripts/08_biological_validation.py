#!/usr/bin/env python3
# 08_biological_validation.py — Publication-ready biological validation (v2)
"""
Focused biological validation for CLOP-DiT producing 3 core figures:

  Figure 1 — Text2Cell Biological Fidelity (multi-dataset, N×3 panel)
    (a) UMAP: real + generated, colored by cell type + source
    (b) Marker Gene Heatmap: side-by-side real vs generated per cell type
    (c) Gene Mean Correlation scatter (real_recon vs generated, Pearson/R²)

  Figure 2 — Cell2Cell Editing Quality (1×3 panel)
    (a) PCA vector field: source → edited arrows with real target reference
    (b) ΔExpression scatter per gene (real vs predicted shift)
    (c) Top-20 DEG direction consistency barplot

    Figure 3 — External Cell Identity Validation (1×2 panel)
        (a) CellTypist confusion matrix (prompt vs predicted)
        (b) CellTypist label distribution barplot

    + metrics_summary.json with all quantitative metrics & human-readable interpretations

Usage:
  python scripts/08_biological_validation.py \\
      --reference_h5ad data/processed_h5ad/GSE123902_LungAdreHmCancer_processed.h5ad \\
      --dataset_key GSE123902_LungAdreHmCancer \\
      --output_dir figures/biovalidation

  # Multi-dataset
  python scripts/08_biological_validation.py \\
      --dataset_manifest configs/biovalidation_datasets.json \\
      --output_dir figures/biovalidation
"""

import argparse, json, logging, sys, gc, warnings
from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util

import numpy as np
import torch
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging_config import setup_logging
from src.architecture.decoder import ScGPTDecoder
from src.visualization.style import COLORS as VIZ_COLORS, apply_style, save_with_vcd

logger = logging.getLogger(__name__)

apply_style()

COLORS = {
    "Real": VIZ_COLORS["real"],
    "Generated": VIZ_COLORS["generated"],
    "source": VIZ_COLORS["neutral"],
    "edited": VIZ_COLORS["warn"],
    "target": VIZ_COLORS["good"],
}

# ── Marker genes per cell type (fallback defaults) ─────────────────────
# Keep non-overlapping, representative markers to avoid redundancy.
MARKER_GENES = {
    "CD8+ T cells": ["CD8A", "CD8B", "GZMB", "PRF1"],
    "Macrophages":  ["CD68", "CD163", "CSF1R", "MSR1"],
    "Epithelial cells": ["EPCAM", "KRT8", "KRT19", "MUC1"],
    "NK cells":     ["NKG7", "GNLY", "KLRD1", "FCGR3A"],
}

DEFAULT_TEXT2CELL_PROMPTS = {
    "CD8+ T cells": (
        "CD8+ cytotoxic T lymphocytes from human lung adenocarcinoma tumor "
        "microenvironment. Tumor-infiltrating CD8+ T cells expressing cytotoxic "
        "effector molecules."
    ),
    "Macrophages": (
        "Tumor-associated macrophages from human lung adenocarcinoma. "
        "Myeloid macrophage populations in the cancer microenvironment."
    ),
    "Epithelial cells": (
        "Malignant epithelial cells from human lung adenocarcinoma. "
        "Cancer cells of epithelial origin with tumor-associated expression programs."
    ),
    "NK cells": (
        "Natural killer cells infiltrating human lung adenocarcinoma. "
        "Innate lymphoid NK cells with cytotoxic activity in the tumor microenvironment."
    ),
}

# celltypist external validation model (can be overridden)
CELLTYPIST_MODEL = "Human_Lung_Atlas.pkl"


# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def load_class_from_script(script_path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location(class_name, str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def to_dense(x):
    return x.A if hasattr(x, "A") else np.asarray(x)


def log1p_safe(adata: ad.AnnData) -> ad.AnnData:
    """Normalize + log1p only if not yet done."""
    adata = adata.copy()
    if "log1p" not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata


def align_genes(*adatas: ad.AnnData) -> List[ad.AnnData]:
    """Intersect gene sets across multiple AnnData objects."""
    common = adatas[0].var_names
    for a in adatas[1:]:
        common = np.intersect1d(common, a.var_names)
    return [a[:, common].copy() for a in adatas]


def load_prompt_file(path: str) -> Dict[str, str]:
    """Load a JSON file mapping cell_type -> prompt text."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning(f"Prompt file not found: {path}")
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(f"Invalid prompt file format: {path}")
            return {}
        return data
    except Exception as e:
        logger.warning(f"Failed to load prompt file: {e}")
        return {}


def load_subcluster_metadata(path: str) -> Dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning(f"Subcluster metadata not found: {path}")
        return {}
    with open(p) as f:
        return json.load(f)


def build_prompts_from_subclusters(
    subcluster_meta: Dict,
    dataset_key: str,
    top_k: int = 4,
    min_conf: float = 0.25,
) -> Dict[str, str]:
    """Auto-build prompts from subcluster metadata."""
    if dataset_key not in subcluster_meta:
        return {}
    ds = subcluster_meta[dataset_key]
    dataset_text = ds.get("dataset_text", "")
    ct_counts = {}
    for cinfo in ds.get("clusters", {}).values():
        if cinfo.get("confidence", 0.0) < min_conf:
            continue
        ct = cinfo.get("cell_type", "Unknown")
        if ct == "Unknown":
            continue
        ct_counts[ct] = ct_counts.get(ct, 0) + int(cinfo.get("n_cells", 0))
    if not ct_counts:
        return {}
    top_types = sorted(ct_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    context = dataset_text.split(".")[0].strip() if dataset_text else "single-cell RNA sequencing"
    prompts = {}
    for ct, _ in top_types:
        prompts[ct] = f"{ct} from {context}."
    return prompts


def compute_markers_by_cell_type(
    adata: ad.AnnData,
    cell_types: List[str],
    n_genes: int = 6,
) -> Dict[str, List[str]]:
    """Compute representative markers per cell type using rank_genes_groups."""
    adata = adata.copy()
    if "log1p" not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    sc.tl.rank_genes_groups(adata, "cell_type", method="wilcoxon", n_genes=50)
    markers = {}
    for ct in cell_types:
        if ct not in adata.obs["cell_type"].unique():
            continue
        try:
            genes = list(adata.uns["rank_genes_groups"]["names"][ct])
        except Exception:
            genes = []
        # Filter to genes present and keep unique non-overlapping selection
        genes = [g for g in genes if g in adata.var_names]
        markers[ct] = genes[: n_genes * 3]
    return markers


def select_nonredundant_markers(
    marker_pool: Dict[str, List[str]],
    n_genes: int = 6,
) -> Dict[str, List[str]]:
    """Pick non-overlapping markers across cell types."""
    used = set()
    selected = {}
    for ct, genes in marker_pool.items():
        picked = []
        for g in genes:
            if g in used:
                continue
            picked.append(g)
            used.add(g)
            if len(picked) >= n_genes:
                break
        selected[ct] = picked
    return selected


def frechet_distance(real: np.ndarray, gen: np.ndarray) -> float:
    mu_r, mu_g = real.mean(0), gen.mean(0)
    sig_r = np.cov(real, rowvar=False)
    sig_g = np.cov(gen, rowvar=False)
    diff = mu_r - mu_g
    covmean = sqrtm(sig_r @ sig_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig_r + sig_g - 2 * covmean))


def gene_mean_pearson(real: ad.AnnData, gen: ad.AnnData) -> float:
    return float(stats.pearsonr(to_dense(real.X).mean(0), to_dense(gen.X).mean(0))[0])


def marker_specificity_index(adata: ad.AnnData, marker_dict: Dict[str, List[str]] = None) -> float:
    """
    Compute a simple marker specificity index (MSI).
    For each marker gene, compute the fraction of its mean expression that falls
    into the intended cell type; then average across all markers.
    Range: [0,1], higher = more specific expression in intended cell type.
    """
    marker_dict = marker_dict or MARKER_GENES
    markers = []
    for ct, genes in marker_dict.items():
        for g in genes:
            if g in adata.var_names:
                markers.append((ct, g))
    if not markers:
        return float("nan")

    # Precompute per-cell-type means
    ct_means = {}
    for ct in marker_dict.keys():
        ct_cells = adata[adata.obs["cell_type"] == ct]
        if ct_cells.n_obs == 0:
            continue
        ct_means[ct] = to_dense(ct_cells.X).mean(axis=0)

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    scores = []
    for ct, g in markers:
        if ct not in ct_means:
            continue
        gi = gene_to_idx[g]
        denom = 0.0
        for ct2, mean_vec in ct_means.items():
            denom += float(mean_vec[gi])
        if denom <= 0:
            continue
        scores.append(float(ct_means[ct][gi]) / denom)

    return float(np.mean(scores)) if scores else float("nan")


def load_dataset_indices(dataset_key: str, subcluster_meta: Dict = None) -> Dict[str, np.ndarray]:
    sc_meta = subcluster_meta or json.load(open("data/processed_h5ad/subcluster_metadata.json"))
    ds = sc_meta[dataset_key]
    ct_idx: Dict[str, list] = {}
    for _, info in ds["clusters"].items():
        ct = info["cell_type"]
        idx = np.array(info["cell_indices"], dtype=int)
        ct_idx.setdefault(ct, []).append(idx)
    for ct in ct_idx:
        ct_idx[ct] = np.concatenate(ct_idx[ct])
    return ct_idx


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Text2Cell Biological Fidelity
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset_specs(args, subcluster_meta: Dict, prompt_overrides: Dict) -> List[Dict]:
    if args.dataset_manifest:
        with open(args.dataset_manifest) as f:
            specs = json.load(f)
        if not isinstance(specs, list) or not specs:
            raise ValueError("dataset_manifest must be a non-empty JSON list")
    else:
        if not args.reference_h5ad or not args.dataset_key:
            raise ValueError("Provide either --dataset_manifest or both --reference_h5ad and --dataset_key")
        specs = [{
            "dataset_key": args.dataset_key,
            "reference_h5ad": args.reference_h5ad,
            "label": args.dataset_key,
        }]

    prepared = []
    for spec in specs[: args.max_datasets]:
        dataset_key = spec["dataset_key"]
        prompts = spec.get("prompts")
        if not prompts and args.auto_prompts:
            prompts = build_prompts_from_subclusters(
                subcluster_meta, dataset_key,
                top_k=args.prompt_top_k, min_conf=args.prompt_min_conf,
            )
        if not prompts:
            prompts = dict(DEFAULT_TEXT2CELL_PROMPTS)
        # Apply prompt overrides if provided
        for ct, p in prompt_overrides.items():
            if ct in prompts:
                prompts[ct] = p
        spec = dict(spec)
        spec["prompts"] = prompts
        prepared.append(spec)
    return prepared


def generate_cells_text2cell(adata_ref, args, prompts, dit_ckpt=None):
    """Generate cells using Text2Cell pipeline with a specific DiT checkpoint."""
    CLOPDiTInference = load_class_from_script(
        Path(__file__).resolve().parent / "05_inference.py", "CLOPDiTInference"
    )
    dit_path = dit_ckpt or args.dit_checkpoint
    pipeline = CLOPDiTInference(
        dit_checkpoint=dit_path,
        clop_checkpoint=args.clop_checkpoint,
        scgpt_model_dir=args.scgpt_model_dir,
        text_encoder_name=args.text_encoder,
        device=args.device,
    )
    fake_list = []
    for ct, prompt in prompts.items():
        adata_gen = pipeline.generate_adata(
            prompt=prompt, num_cells=args.num_cells_per_type,
            decode_expression=True, reference_adata=adata_ref,
            num_steps=args.num_steps, cfg_scale=args.cfg_scale,
        )
        adata_gen.obs["cell_type"] = ct
        adata_gen.obs["source"] = "Generated"
        fake_list.append(adata_gen)
        logger.info(f"  Generated {adata_gen.n_obs} {ct} cells")
    del pipeline; gc.collect(); torch.cuda.empty_cache()
    return ad.concat(fake_list, join="outer", merge="same")


def build_real_subset(adata_ref, indices_map, n_per_type, prompts):
    rng = np.random.RandomState(42)
    real_list = []
    for ct in prompts:
        if ct not in indices_map:
            logger.warning(f"  Cell type '{ct}' absent, skipping")
            continue
        idx = indices_map[ct]
        if len(idx) > n_per_type:
            idx = rng.choice(idx, size=n_per_type, replace=False)
        a = adata_ref[idx].copy()
        a.obs["cell_type"] = ct
        a.obs["source"] = "Real"
        real_list.append(a)
    return ad.concat(real_list, join="outer", merge="same")


def compute_text2cell_metrics(real_adata, fake_adata, scgpt, marker_dict=None):
    """Compute Text2Cell metrics and preprocessed AnnData for plotting."""
    real_emb = scgpt.encode(real_adata)
    fake_emb = scgpt.encode(fake_adata)
    real_dec = scgpt.decode(real_emb)
    real_recon = ad.AnnData(X=real_dec["expression"])
    real_recon.var_names = real_dec["gene_names"]
    real_recon.obs = real_adata.obs.copy()
    real_recon.obs["source"] = "Real"

    real_n, fake_n = align_genes(real_recon, fake_adata)
    real_n = log1p_safe(real_n)
    fake_n = log1p_safe(fake_n)

    real_x = to_dense(real_n.X)
    fake_x = to_dense(fake_n.X)

    rm, fm = real_x.mean(0), fake_x.mean(0)
    rv, fv = real_x.var(0), fake_x.var(0)
    pr_mean = stats.pearsonr(rm, fm)
    r2_mean = r2_score(rm, fm)
    pr_var = stats.pearsonr(rv, fv)

    pca_dim = min(50, real_x.shape[1])
    pca_r = PCA(n_components=pca_dim).fit_transform(real_x)
    pca_f = PCA(n_components=pca_dim).fit_transform(fake_x)
    fd_gene = frechet_distance(pca_r, pca_f)
    fd_emb = frechet_distance(real_emb, fake_emb)

    msi_real = marker_specificity_index(real_n, marker_dict=marker_dict)
    msi_fake = marker_specificity_index(fake_n, marker_dict=marker_dict)

    metrics = {
        "gene_mean_pearson": float(pr_mean[0]),
        "gene_mean_R2": float(r2_mean),
        "gene_var_pearson": float(pr_var[0]),
        "FD_gene_pca50": float(fd_gene),
        "FD_scgpt_embedding": float(fd_emb),
        "marker_specificity_real": float(msi_real),
        "marker_specificity_generated": float(msi_fake),
    }

    return metrics, real_recon, real_n, fake_n, rm, fm, rv, fv


def figure1_text2cell_multi(dataset_runs, scgpt, output_dir, metrics):
    """Create Figure 1: multi-dataset Text2Cell biological fidelity."""
    logger.info("Creating Figure 1 — Text2Cell Biological Fidelity (multi-dataset)")

    per_dataset = []
    for ds in dataset_runs:
        cell_types = list(ds["prompts"].keys())
        marker_pool = compute_markers_by_cell_type(ds["real_adata"], cell_types, n_genes=6)
        marker_dict = select_nonredundant_markers(marker_pool, n_genes=4)
        if not any(marker_dict.values()):
            marker_dict = MARKER_GENES

        m, real_recon, real_n, fake_n, rm, fm, rv, fv = compute_text2cell_metrics(
            ds["real_adata"], ds["fake_adata"], scgpt, marker_dict=marker_dict
        )
        per_dataset.append({
            "label": ds["label"],
            "metrics": m,
            "marker_dict": marker_dict,
            "real_recon": real_recon,
            "real_n": real_n,
            "fake_n": fake_n,
            "rm": rm,
            "fm": fm,
            "rv": rv,
            "fv": fv,
        })

    metrics["text2cell_multi"] = {
        "datasets": {
            d["label"]: d["metrics"] for d in per_dataset
        },
        "interpretation": {
            "gene_mean_pearson": "Pearson r between per-gene mean expression of real (scGPT-reconstructed) vs generated. >0.9 = high fidelity; 1.0 = perfect.",
            "gene_mean_R2": "R-squared of gene means; fraction of variance explained. >0.8 is good.",
            "gene_var_pearson": "Pearson r of per-gene variance. Captures whether expression variability is preserved.",
            "FD_gene_pca50": "Frechet Distance in PCA-50 gene space. Lower = more similar distributions. <1 is excellent.",
            "FD_scgpt_embedding": "Frechet Distance in scGPT 512-d embedding space. Lower = better.",
            "marker_specificity_real": "Marker specificity index (MSI) on real cells. Higher = markers are enriched in intended cell types.",
            "marker_specificity_generated": "MSI on generated cells. Higher = markers remain cell-type specific after generation.",
        },
    }

    n_rows = len(per_dataset)
    fig_h = max(4.5, 4.2 * n_rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, fig_h))
    if n_rows == 1:
        axes = np.array([axes])

    for i, d in enumerate(per_dataset):
        row_axes = axes[i]
        label = d["label"]
        rm, fm = d["rm"], d["fm"]
        m = d["metrics"]

        # (a) UMAP integration
        _plot_umap_panel(d["real_recon"], dataset_runs[i]["fake_adata"], row_axes[0])
        row_axes[0].set_title(f"(a) UMAP Integration — {label}")

        # (b) Marker gene heatmap
        _plot_marker_heatmap(d["real_n"], d["fake_n"], row_axes[1], marker_dict=d["marker_dict"])
        row_axes[1].set_title(f"(b) Marker Genes — {label}")

        # (c) Gene mean scatter
        ax = row_axes[2]
        ax.scatter(rm, fm, s=4, alpha=0.35, c="#1f77b4", edgecolors="none", rasterized=True)
        lim = [min(rm.min(), fm.min()) - 0.05, max(rm.max(), fm.max()) + 0.05]
        ax.plot(lim, lim, "--", color="#999999", lw=0.8)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Real (scGPT-reconstructed)")
        ax.set_ylabel("Generated")
        ax.set_title(f"(c) Gene Mean Expression — {label}")
        ax.text(
            0.05, 0.92,
            f"Pearson r = {m['gene_mean_pearson']:.4f}\nR² = {m['gene_mean_R2']:.4f}\n"
            f"MSI(real)={m['marker_specificity_real']:.2f}\nMSI(gen)={m['marker_specificity_generated']:.2f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
    save_with_vcd(fig, output_dir / "figure1_text2cell_multi.png", dpi=300, close=False)
    plt.close(fig)
    logger.info(f"  Saved Figure 1 -> {output_dir / 'figure1_text2cell_multi.png'}")


def _plot_umap_panel(real_recon, fake_adata, ax):
    """UMAP integration of real-reconstructed vs generated cells."""
    combined = ad.concat([real_recon, fake_adata], join="outer", merge="same")
    combined.X = np.nan_to_num(to_dense(combined.X), nan=0.0)

    sc.pp.filter_cells(combined, min_counts=1)
    sc.pp.filter_genes(combined, min_counts=1)
    if "log1p" not in combined.uns:
        sc.pp.normalize_total(combined, target_sum=1e4)
        sc.pp.log1p(combined)
    n_genes = combined.shape[1]
    if n_genes > 2000:
        sc.pp.highly_variable_genes(combined, n_top_genes=min(2000, n_genes),
                                    flavor="seurat_v3", subset=True)
    sc.pp.scale(combined, max_value=10)
    sc.tl.pca(combined, n_comps=min(50, combined.shape[1] - 1))
    sc.pp.neighbors(combined, n_neighbors=15)
    sc.tl.umap(combined)

    umap = combined.obsm["X_umap"]
    sources = combined.obs["source"].values
    cell_types = combined.obs["cell_type"].values

    ct_list = sorted(set(cell_types))
    ct_colors = plt.cm.Set2(np.linspace(0, 1, max(len(ct_list), 2)))
    ct_cmap = {ct: ct_colors[i] for i, ct in enumerate(ct_list)}

    for ct in ct_list:
        for src, marker, alpha, sz in [("Real", "o", 0.5, 12), ("Generated", "^", 0.6, 18)]:
            mask = (cell_types == ct) & (sources == src)
            if mask.sum() == 0:
                continue
            ax.scatter(umap[mask, 0], umap[mask, 1], s=sz, alpha=alpha,
                       marker=marker, c=[ct_cmap[ct]], label=f"{ct} ({src})",
                       edgecolors="none", rasterized=True)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("(b) UMAP Integration")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=6, loc="upper right", frameon=False,
              ncol=1, markerscale=0.8, handletextpad=0.3, borderaxespad=0.2)


def _select_marker_genes(real_n, fake_n, marker_dict=None, max_per_type=4):
    marker_dict = marker_dict or MARKER_GENES
    used = set()
    all_markers = []
    marker_ct_labels = []
    for ct, genes in marker_dict.items():
        added = 0
        for g in genes:
            if g in real_n.var_names and g in fake_n.var_names and g not in used:
                all_markers.append(g)
                marker_ct_labels.append(ct)
                used.add(g)
                added += 1
            if added >= max_per_type:
                break
    return all_markers, marker_ct_labels


def _plot_marker_heatmap(real_n, fake_n, ax, marker_dict=None):
    """Side-by-side marker gene heatmap: Real (left) | Generated (right)."""
    all_markers, marker_ct_labels = _select_marker_genes(
        real_n, fake_n, marker_dict=marker_dict, max_per_type=4
    )

    if not all_markers:
        ax.text(0.5, 0.5, "No marker genes found in data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(c) Marker Genes")
        return

    ct_order = list(dict.fromkeys(marker_ct_labels))

    real_mat, gen_mat, row_labels = [], [], []
    for ct in ct_order:
        r_ct = real_n[real_n.obs["cell_type"] == ct]
        g_ct = fake_n[fake_n.obs["cell_type"] == ct]
        if r_ct.n_obs == 0 or g_ct.n_obs == 0:
            continue
        real_expr = to_dense(r_ct[:, all_markers].X).mean(axis=0)
        gen_expr = to_dense(g_ct[:, all_markers].X).mean(axis=0)
        real_mat.append(real_expr)
        gen_mat.append(gen_expr)
        row_labels.append(ct)

    if not row_labels:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("(c) Marker Genes")
        return

    real_mat = np.array(real_mat)
    gen_mat = np.array(gen_mat)

    def zscore_cols(m):
        mu = m.mean(0, keepdims=True)
        sd = m.std(0, keepdims=True)
        sd[sd < 1e-8] = 1
        return (m - mu) / sd

    real_z = zscore_cols(real_mat)
    gen_z = zscore_cols(gen_mat)

    combined = np.hstack([real_z, gen_z])
    cmap = LinearSegmentedColormap.from_list("rg", ["#2166ac", "#f7f7f7", "#b2182b"])
    vmax = min(max(abs(combined.min()), abs(combined.max())), 3.0)

    im = ax.imshow(combined, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation="nearest")

    n_m = len(all_markers)
    xtick_pos = list(range(n_m)) + list(range(n_m, 2 * n_m))
    xtick_labels = all_markers + all_markers
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    ax.axvline(n_m - 0.5, color="black", linewidth=1.5)
    # Place Real/Generated labels above heatmap
    ax.text(n_m * 0.5 - 0.5, -0.8, "Real", ha="center", fontsize=8,
            fontweight="bold", transform=ax.transData)
    ax.text(n_m * 1.5 - 0.5, -0.8, "Generated", ha="center", fontsize=8,
            fontweight="bold", transform=ax.transData)
    ax.set_title("(c) Marker Gene Expression (z-scored)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("z-score", fontsize=7)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Cell2Cell Editing Quality
# ═══════════════════════════════════════════════════════════════════════════

def figure2_cell2cell(adata_ref, indices_map, scgpt, output_dir, args, metrics, prompts):
    """Create Figure 2: 1×3 Cell2Cell editing panel."""
    logger.info("Creating Figure 2 — Cell2Cell Editing Quality")

    Cell2CellInference = load_class_from_script(
        Path(__file__).resolve().parent / "06_cell2cell_inference.py", "Cell2CellInference"
    )

    src_type = "Monocytes" if "Monocytes" in indices_map else list(indices_map.keys())[0]
    tgt_type = "Macrophages" if "Macrophages" in indices_map else list(indices_map.keys())[1]

    rng = np.random.RandomState(42)
    src_idx = indices_map[src_type]
    tgt_idx = indices_map[tgt_type]
    if len(src_idx) > args.num_cells_per_type:
        src_idx = rng.choice(src_idx, size=args.num_cells_per_type, replace=False)
    if len(tgt_idx) > args.num_cells_per_type:
        tgt_idx = rng.choice(tgt_idx, size=args.num_cells_per_type, replace=False)

    real_src = adata_ref[src_idx].copy()
    real_tgt = adata_ref[tgt_idx].copy()

    c2c = Cell2CellInference(
        cell2cell_checkpoint=args.cell2cell_checkpoint,
        clop_checkpoint=args.clop_checkpoint,
        scgpt_model_dir=args.scgpt_model_dir,
        text_encoder_name=args.text_encoder,
        device=args.device,
    )

    target_prompt = prompts.get(
        tgt_type, f"{tgt_type} from human lung adenocarcinoma"
    )
    edited = c2c.edit_adata(
        real_src, target_prompt=target_prompt,
        edit_strength=args.edit_strength, decode_expression=True,
        reference_adata=adata_ref,
        num_steps=args.num_steps, cfg_scale=args.cfg_scale,
    )

    # DEG analysis
    rs, rt, pt = align_genes(log1p_safe(real_src), log1p_safe(real_tgt), log1p_safe(edited))
    delta_real = to_dense(rt.X).mean(0) - to_dense(rs.X).mean(0)
    delta_pred = to_dense(pt.X).mean(0) - to_dense(rs.X).mean(0)
    top_n = 50
    idx_real_top = np.argsort(-np.abs(delta_real))[:top_n]
    idx_pred_top = np.argsort(-np.abs(delta_pred))[:top_n]
    overlap = len(set(idx_real_top) & set(idx_pred_top)) / top_n
    dir_acc = float(np.mean(np.sign(delta_real[idx_real_top]) == np.sign(delta_pred[idx_real_top])))
    r2_delta = r2_score(delta_real, delta_pred)

    # Embeddings for vector field
    src_emb = scgpt.encode(real_src)
    tgt_emb = scgpt.encode(real_tgt)
    edit_emb = edited.obsm.get("X_edited_emb")
    if edit_emb is None:
        edit_emb = scgpt.encode(edited)

    # Embedding proximity metrics
    def _cos(a, b):
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_n = b / (np.linalg.norm(b) + 1e-8)
        return np.dot(a_n, b_n)

    mean_tgt = tgt_emb.mean(0)
    mean_src = src_emb.mean(0)
    cos_to_tgt = _cos(edit_emb, mean_tgt)
    cos_to_src = _cos(edit_emb, mean_src)
    cos_gain = float(np.mean(cos_to_tgt - cos_to_src))

    metrics["cell2cell"] = {
        "source_type": src_type,
        "target_type": tgt_type,
        "DEG_overlap_top50": float(overlap),
        "direction_accuracy": float(dir_acc),
        "delta_R2": float(r2_delta),
        "cosine_to_target_mean": float(np.mean(cos_to_tgt)),
        "cosine_to_source_mean": float(np.mean(cos_to_src)),
        "cosine_gain": cos_gain,
        "interpretation": {
            "DEG_overlap_top50": (
                "Fraction of top-50 differentially expressed genes shared between "
                "real and predicted cell-type transition. Higher = better gene-level "
                "specificity. 1.0 = perfect overlap."
            ),
            "direction_accuracy": (
                "Fraction of top-50 real DEGs where the predicted up/down direction "
                "matches reality. >0.7 is meaningful; 0.5 = random chance."
            ),
            "delta_R2": (
                "R-squared between real vs predicted per-gene expression shift vectors. "
                ">0 means predictions explain real shifts; <0 means worse than a flat "
                "prediction. Higher is better."
            ),
            "cosine_gain": (
                "Mean cosine similarity gain of edited cells toward target vs source "
                "in scGPT embedding space. Positive = moved toward target."
            ),
        },
    }

    del c2c; gc.collect(); torch.cuda.empty_cache()

    # ── Build Figure 2 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Vector field
    ax = axes[0]
    X_all = np.vstack([src_emb, edit_emb, tgt_emb])
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X_all)
    n_s, n_e = len(src_emb), len(edit_emb)
    sp, ep, tp = Xp[:n_s], Xp[n_s:n_s+n_e], Xp[n_s+n_e:]

    ax.scatter(tp[:, 0], tp[:, 1], s=10, alpha=0.4, c=COLORS["target"],
               label=f"Real {tgt_type}", edgecolors="none", rasterized=True)
    ax.scatter(sp[:, 0], sp[:, 1], s=10, alpha=0.4, c=COLORS["source"],
               label=f"Real {src_type}", edgecolors="none", rasterized=True)

    n_arrows = min(150, n_s)
    arrow_idx = rng.choice(n_s, size=n_arrows, replace=False)
    for i in arrow_idx:
        ax.annotate("", xy=(ep[i, 0], ep[i, 1]), xytext=(sp[i, 0], sp[i, 1]),
                     arrowprops=dict(arrowstyle="->", color=COLORS["edited"],
                                     alpha=0.4, lw=0.6))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"(a) Cell Editing Vector Field\n{src_type} → {tgt_type}")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    # (b) ΔExpression scatter
    ax = axes[1]
    ax.scatter(delta_real, delta_pred, s=4, alpha=0.3, c="#ff7f0e",
               edgecolors="none", rasterized=True)
    dlim = [min(delta_real.min(), delta_pred.min()),
            max(delta_real.max(), delta_pred.max())]
    ax.plot(dlim, dlim, "--", color="#999999", lw=0.8)
    ax.set_xlabel("Real Δ expression")
    ax.set_ylabel("Predicted Δ expression")
    ax.set_title("(b) Per-Gene Expression Shift")
    ax.text(0.05, 0.92, f"R² = {r2_delta:.4f}\n(<0 = worse than zero-shift)",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # (c) Embedding proximity boxplot
    ax = axes[2]
    ax.boxplot([cos_to_src, cos_to_tgt], labels=["To Source", "To Target"],
               widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#cccccc", alpha=0.7),
               medianprops=dict(color="#000000"))
    ax.set_ylabel("Cosine similarity (scGPT emb)")
    ax.set_title(f"(c) Embedding Proximity\nGain={cos_gain:.3f}")

    fig.tight_layout()
    save_with_vcd(fig, output_dir / "figure2_cell2cell.png", dpi=300, close=False)
    plt.close(fig)
    logger.info(f"  Saved Figure 2 -> {output_dir / 'figure2_cell2cell.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — External Cell Identity Validation (CellTypist)
# ═══════════════════════════════════════════════════════════════════════════

def figure3_celltypist(fake_adata, output_dir, metrics, model_name=CELLTYPIST_MODEL, real_adata=None):
    """Create Figure 3: external cell identity validation using CellTypist."""
    logger.info("Creating Figure 3 — External Cell Identity Validation (CellTypist)")

    try:
        import celltypist
    except Exception as e:
        logger.warning(f"CellTypist not available: {e}")
        return

    adata = fake_adata.copy()
    adata = log1p_safe(adata)

    result = celltypist.annotate(adata, model=model_name, majority_voting=False)
    pred_labels = result.predicted_labels["predicted_labels"].values
    prompts = adata.obs["cell_type"].values

    real_labels = None
    if real_adata is not None:
        real_tmp = log1p_safe(real_adata.copy())
        real_res = celltypist.annotate(real_tmp, model=model_name, majority_voting=False)
        real_labels = real_res.predicted_labels["predicted_labels"].values

    # build confusion matrix (row-normalized)
    from collections import Counter
    all_counts = Counter(pred_labels)
    if real_labels is not None:
        all_counts.update(Counter(real_labels))
    top_k = 12
    top_labels = [lab for lab, _ in all_counts.most_common(top_k)]
    if len(all_counts) > top_k:
        top_labels.append("Other")

    prompt_types = list(dict.fromkeys(prompts))
    mat = np.zeros((len(prompt_types), len(top_labels)), dtype=float)
    purity = {}
    for i, ct in enumerate(prompt_types):
        ct_labels = pred_labels[prompts == ct]
        counts = Counter(ct_labels)
        total = len(ct_labels)
        if total == 0:
            continue
        top_lab, top_cnt = counts.most_common(1)[0]
        purity[ct] = float(top_cnt / total)
        for j, lab in enumerate(top_labels):
            if lab == "Other":
                mat[i, j] = sum(v for k, v in counts.items() if k not in top_labels)
            else:
                mat[i, j] = counts.get(lab, 0)
        mat[i, :] = mat[i, :] / total

    # heuristic match rate — expanded mapping
    def map_label_to_category(label: str):
        l = label.lower()
        # T cells
        if "cd8" in l or "cytotoxic" in l:
            return "CD8+ T cells"
        if "cd4" in l and ("helper" in l or "naive" in l or "memory" in l or "th1" in l or "th2" in l or "th17" in l):
            return "CD4+ T cells"
        if "cd4" in l:
            return "CD4+ T cells"
        if "treg" in l or "regulatory t" in l or ("foxp3" in l and "t cell" in l):
            return "Regulatory T cells"
        if "gamma" in l and "delta" in l:
            return "Gamma-delta T cells"
        if any(k in l for k in ["t cell", "t lymph"]) and "nk" not in l:
            return "CD8+ T cells"  # default T cells to CD8+
        # NK cells
        if "nk" in l or "natural killer" in l:
            return "NK cells"
        # B cells / Plasma
        if "plasma" in l or "plasmablast" in l:
            return "Plasma cells"
        if any(k in l for k in ["b cell", "b lymph", "b-cell"]):
            return "B cells"
        # Myeloid
        if "macrophage" in l or "mph" in l:
            return "Macrophages"
        if "monocyte" in l:
            return "Monocytes"
        if "dendritic" in l or "dc" == l.strip() or "cdc" in l or "pdc" in l:
            return "Dendritic cells"
        if "neutrophil" in l or "granulocyte" in l:
            return "Neutrophils"
        if "mast" in l:
            return "Mast cells"
        if "megakaryo" in l or "platelet" in l:
            return "Megakaryocytes"
        # Epithelial
        if any(k in l for k in ["epithelial", "at1", "at2", "ciliated", "secretory",
                                 "club", "goblet", "alveolar", "basal", "ionocyte",
                                 "multiciliated", "pneumocyte", "keratinocyte"]):
            return "Epithelial cells"
        # Stromal
        if "fibroblast" in l or "mesenchymal" in l or "caf" in l:
            return "Fibroblasts"
        if "endothelial" in l:
            return "Endothelial cells"
        if "smooth muscle" in l or "myofibroblast" in l:
            return "Smooth muscle cells"
        if "pericyte" in l:
            return "Pericytes"
        # Neural
        if "neuron" in l or "neuronal" in l:
            return "Neurons"
        if "astrocyte" in l:
            return "Astrocytes"
        if "oligodendrocyte" in l:
            return "Oligodendrocytes"
        if "microglia" in l:
            return "Microglia"
        # Hematopoietic
        if any(k in l for k in ["stem cell", "progenitor", "hsc"]):
            return "HSCs/Progenitors"
        if "erythro" in l or "erythroid" in l:
            return "Erythroid progenitors"
        # Hepatic
        if "hepatocyte" in l:
            return "Hepatocytes"
        if "cholangiocyte" in l:
            return "Cholangiocytes"
        # Other
        if "melanocyte" in l or "melanoma" in l:
            return "Melanocytes"
        if "adipocyte" in l:
            return "Adipocytes"
        if "proliferat" in l:
            return "Proliferating cells"
        return None

    match_rates = {}
    for ct in prompt_types:
        ct_labels = pred_labels[prompts == ct]
        if len(ct_labels) == 0:
            match_rates[ct] = float("nan")
            continue
        mapped = [map_label_to_category(lab) for lab in ct_labels]
        match_rates[ct] = float(np.mean([m == ct for m in mapped if m is not None])) if any(m is not None for m in mapped) else 0.0

    # diversity metrics for generated labels
    gen_counts = Counter(pred_labels)
    gen_total = sum(gen_counts.values())
    if gen_total > 0:
        probs = np.array([v / gen_total for v in gen_counts.values()])
        label_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        label_entropy = max(0.0, label_entropy)
    else:
        label_entropy = float("nan")

    per_dataset_match = {}
    if "dataset" in adata.obs:
        for ds in sorted(set(adata.obs["dataset"].values)):
            mask = adata.obs["dataset"].values == ds
            ds_labels = pred_labels[mask]
            ds_prompts = prompts[mask]
            if len(ds_labels) == 0:
                per_dataset_match[ds] = float("nan")
                continue
            ds_rates = []
            for ct in set(ds_prompts):
                ct_labels = ds_labels[ds_prompts == ct]
                if len(ct_labels) == 0:
                    continue
                mapped = [map_label_to_category(lab) for lab in ct_labels]
                if any(m is not None for m in mapped):
                    ds_rates.append(np.mean([m == ct for m in mapped if m is not None]))
            per_dataset_match[ds] = float(np.mean(ds_rates)) if ds_rates else 0.0

    metrics["celltypist"] = {
        "model": model_name,
        "top_label_purity": {k: float(v) for k, v in purity.items()},
        "heuristic_match_rate": {k: float(v) for k, v in match_rates.items()},
        "per_dataset_match_rate": per_dataset_match if per_dataset_match else None,
        "labels_used": top_labels,
        "generated_label_counts": {k: int(v) for k, v in Counter(pred_labels).items()},
        "real_label_counts": {k: int(v) for k, v in Counter(real_labels).items()} if real_labels is not None else None,
        "generated_label_entropy_bits": label_entropy,
        "generated_unique_labels": int(len(gen_counts)),
        "interpretation": {
            "top_label_purity": "For each prompt, fraction of generated cells assigned to the most frequent CellTypist label. Higher = more coherent identity.",
            "heuristic_match_rate": "Keyword-based mapping from CellTypist labels to coarse cell types (heuristic). Higher = better external agreement.",
            "generated_label_counts": "Raw CellTypist label counts for generated cells.",
            "real_label_counts": "Raw CellTypist label counts for real cells in the same prompt types (if provided).",
            "generated_label_entropy_bits": "Label diversity of generated cells in bits. 0 = all assigned to one label; higher = more diverse identities.",
            "generated_unique_labels": "Number of unique CellTypist labels assigned to generated cells. Low values indicate collapse.",
        },
    }

    # ── Build Figure 3 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Confusion matrix
    ax = axes[0]
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(top_labels)))
    ax.set_xticklabels(top_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(prompt_types)))
    ax.set_yticklabels(prompt_types, fontsize=8)
    ax.set_title(f"(a) CellTypist Confusion (row-normalized)\nModel: {model_name}")
    ax.text(0.02, -0.18, f"Generated label entropy: {label_entropy:.2f} bits",
            transform=ax.transAxes, fontsize=7, color="#555555")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Fraction", fontsize=7)

    # (b) Prompt-wise purity + match rate
    ax = axes[1]
    y = np.arange(len(prompt_types))
    purity_vals = [purity.get(ct, float("nan")) for ct in prompt_types]
    match_vals = [match_rates.get(ct, float("nan")) for ct in prompt_types]
    ax.barh(y - 0.2, purity_vals, height=0.35, color="#4c72b0", alpha=0.8, label="Purity")
    ax.barh(y + 0.2, match_vals, height=0.35, color="#55a868", alpha=0.8, label="Match rate")
    ax.set_yticks(y)
    ax.set_yticklabels(prompt_types, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction")
    ax.set_xlim(0, 1.0)
    ax.set_title("(b) Prompt-wise Purity & Match")
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    fig.tight_layout()
    save_with_vcd(fig, output_dir / "figure3_celltypist.png", dpi=300, close=False)
    plt.close(fig)
    logger.info(f"  Saved Figure 3 -> {output_dir / 'figure3_celltypist.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Multi-dataset Summary
# ═══════════════════════════════════════════════════════════════════════════

def figure4_summary(metrics: Dict, output_dir: Path):
    """Create Figure 4: compact multi-dataset summary."""
    if "text2cell_multi" not in metrics:
        logger.warning("Figure 4 skipped: missing text2cell_multi metrics")
        return

    ds_metrics = metrics["text2cell_multi"]["datasets"]
    labels = list(ds_metrics.keys())
    gene_mean = [ds_metrics[l]["gene_mean_pearson"] for l in labels]
    msi_gen = [ds_metrics[l]["marker_specificity_generated"] for l in labels]

    celltypist_rates = None
    if "celltypist" in metrics and metrics["celltypist"].get("per_dataset_match_rate"):
        per_ds = metrics["celltypist"]["per_dataset_match_rate"]
        celltypist_rates = [per_ds.get(l, float("nan")) for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.bar(range(len(labels)), gene_mean, color="#4c72b0", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Pearson r")
    ax.set_title("(a) Gene Mean Correlation")

    ax = axes[1]
    ax.bar(range(len(labels)), msi_gen, color="#55a868", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("MSI")
    ax.set_title("(b) Marker Specificity (Gen)")

    ax = axes[2]
    if celltypist_rates is not None:
        ax.bar(range(len(labels)), celltypist_rates, color="#c44e52", alpha=0.85)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Match rate")
        ax.set_title("(c) CellTypist Match Rate")
    else:
        ax.text(0.5, 0.5, "CellTypist match\nnot available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    save_with_vcd(fig, output_dir / "figure4_summary.png", dpi=300, close=False)
    plt.close(fig)
    logger.info(f"  Saved Figure 4 -> {output_dir / 'figure4_summary.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Biological Validation v2")
    parser.add_argument("--reference_h5ad")
    parser.add_argument("--dataset_key")
    parser.add_argument("--dataset_manifest",
                        help="JSON list of {dataset_key, reference_h5ad, label?, prompts?}")
    parser.add_argument("--max_datasets", type=int, default=3)
    parser.add_argument("--subcluster_meta", default="data/processed_h5ad/subcluster_metadata.json")
    parser.add_argument("--auto_prompts", action="store_true",
                        help="Auto-build prompts from subcluster metadata")
    parser.add_argument("--prompt_file", default="",
                        help="JSON mapping of cell_type -> prompt to override defaults")
    parser.add_argument("--prompt_top_k", type=int, default=4)
    parser.add_argument("--prompt_min_conf", type=float, default=0.25)
    parser.add_argument("--output_dir", default="figures/biovalidation")
    parser.add_argument("--num_cells_per_type", type=int, default=200)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--edit_strength", type=float, default=0.5)
    parser.add_argument("--dit_checkpoint", default="models/checkpoints/dit_best.pth")
    parser.add_argument("--clop_checkpoint", default="models/checkpoints/clop_best.pth")
    parser.add_argument("--cell2cell_checkpoint", default="models/checkpoints/cell2cell_best.pth")
    parser.add_argument("--scgpt_model_dir", default="models/scgpt_pancancer")
    parser.add_argument("--text_encoder", default="microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
    parser.add_argument("--celltypist_model", default=CELLTYPIST_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_figure1", action="store_true")
    parser.add_argument("--skip_figure2", action="store_true")
    parser.add_argument("--skip_figure3", action="store_true")
    parser.add_argument("--skip_figure4", action="store_true")
    args = parser.parse_args()

    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scgpt = ScGPTDecoder(model_dir=args.scgpt_model_dir, device=torch.device(args.device))
    metrics: Dict = {}

    prompt_overrides = load_prompt_file(args.prompt_file)
    subcluster_meta = load_subcluster_metadata(args.subcluster_meta) if args.auto_prompts else {}
    dataset_specs = load_dataset_specs(args, subcluster_meta, prompt_overrides)
    dataset_runs = []

    for spec in dataset_specs:
        dataset_key = spec["dataset_key"]
        ref_path = spec["reference_h5ad"]
        label = spec.get("label", dataset_key)
        prompts = spec.get("prompts", DEFAULT_TEXT2CELL_PROMPTS)

        logger.info(f"Loading reference: {ref_path}")
        adata_ref = sc.read_h5ad(ref_path)
        indices_map = load_dataset_indices(dataset_key, subcluster_meta=subcluster_meta or None)

        fake = None
        real = None
        if not args.skip_figure1 or not args.skip_figure3:
            fake = generate_cells_text2cell(adata_ref, args, prompts)
            fake.obs["dataset"] = label
            real = build_real_subset(adata_ref, indices_map, args.num_cells_per_type, prompts)
            real.obs["dataset"] = label

        dataset_runs.append({
            "label": label,
            "dataset_key": dataset_key,
            "reference_h5ad": ref_path,
            "prompts": prompts,
            "adata_ref": adata_ref,
            "indices_map": indices_map,
            "fake_adata": fake,
            "real_adata": real,
        })

    if not args.skip_figure1:
        figure1_text2cell_multi(dataset_runs, scgpt, output_dir, metrics)

    if not args.skip_figure2:
        preferred = None
        for ds in dataset_runs:
            if "Monocytes" in ds["indices_map"] and "Macrophages" in ds["indices_map"]:
                preferred = ds
                break
        if preferred is None:
            preferred = dataset_runs[0]
        metrics["cell2cell_dataset"] = preferred["label"]
        figure2_cell2cell(
            preferred["adata_ref"], preferred["indices_map"], scgpt,
            output_dir, args, metrics, preferred["prompts"]
        )

    if not args.skip_figure3:
        fake_all = ad.concat([ds["fake_adata"] for ds in dataset_runs if ds["fake_adata"] is not None],
                             join="outer", merge="same")
        real_all = ad.concat([ds["real_adata"] for ds in dataset_runs if ds["real_adata"] is not None],
                             join="outer", merge="same")
        figure3_celltypist(
            fake_all, output_dir, metrics,
            model_name=args.celltypist_model, real_adata=real_all,
        )

    if not args.skip_figure4:
        figure4_summary(metrics, output_dir)

    # ── Save metrics ──
    metrics_path = output_dir / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved -> {metrics_path}")
    logger.info("=== Validation complete ===")


if __name__ == "__main__":
    main()
