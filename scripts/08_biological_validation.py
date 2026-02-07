#!/usr/bin/env python3
# 08_biological_validation.py — Publication-ready biological validation (v2)
"""
Focused biological validation for CLOP-DiT producing 3 core figures:

  Figure 1 — Text2Cell Biological Fidelity (2×2 panel)
    (a) Gene Mean Correlation scatter (real_recon vs generated, Pearson/R²)
    (b) UMAP: real + generated, colored by cell type + source
    (c) Marker Gene Heatmap: side-by-side real vs generated per cell type
    (d) Gene Variance Correlation scatter

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

logger = logging.getLogger(__name__)

# ── Publication style ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "Real": "#2166ac",
    "Generated": "#d6604d",
    "source": "#636363",
    "edited": "#e6550d",
    "target": "#31a354",
}

# ── Marker genes per cell type (curated for lung adeno TME) ─────────────
MARKER_GENES = {
    "CD8+ T cells": ["CD8A", "GZMB", "PRF1", "IFNG", "NKG7", "GZMA"],
    "Macrophages":  ["CD68", "CD163", "CSF1R", "MRC1", "CD14", "MSR1"],
    "Epithelial cells": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "CDH1"],
    "NK cells":     ["NKG7", "KLRD1", "GNLY", "FCGR3A", "NCAM1", "KLRF1"],
}

TEXT2CELL_PROMPTS = {
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


def marker_specificity_index(adata: ad.AnnData) -> float:
    """
    Compute a simple marker specificity index (MSI).
    For each marker gene, compute the fraction of its mean expression that falls
    into the intended cell type; then average across all markers.
    Range: [0,1], higher = more specific expression in intended cell type.
    """
    markers = []
    for ct, genes in MARKER_GENES.items():
        for g in genes:
            if g in adata.var_names:
                markers.append((ct, g))
    if not markers:
        return float("nan")

    # Precompute per-cell-type means
    ct_means = {}
    for ct in MARKER_GENES.keys():
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


def load_dataset_indices(dataset_key: str) -> Dict[str, np.ndarray]:
    sc_meta = json.load(open("data/processed_h5ad/subcluster_metadata.json"))
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

def generate_cells_text2cell(adata_ref, args, dit_ckpt=None):
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
    for ct, prompt in TEXT2CELL_PROMPTS.items():
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


def build_real_subset(adata_ref, indices_map, n_per_type):
    rng = np.random.RandomState(42)
    real_list = []
    for ct in TEXT2CELL_PROMPTS:
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


def figure1_text2cell(real_adata, fake_adata, scgpt, output_dir, metrics):
    """Create Figure 1: 2×2 publication panel."""
    logger.info("Creating Figure 1 — Text2Cell Biological Fidelity")

    # ── Reconstruct real via scGPT for fair comparison ──
    real_emb = scgpt.encode(real_adata)
    fake_emb = scgpt.encode(fake_adata)
    real_dec = scgpt.decode(real_emb)
    real_recon = ad.AnnData(X=real_dec["expression"])
    real_recon.var_names = real_dec["gene_names"]
    real_recon.obs = real_adata.obs.copy()
    real_recon.obs["source"] = "Real"

    # align & normalize
    real_n, fake_n = align_genes(real_recon, fake_adata)
    real_n = log1p_safe(real_n)
    fake_n = log1p_safe(fake_n)

    real_x = to_dense(real_n.X)
    fake_x = to_dense(fake_n.X)

    # ── Compute metrics ──
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

    msi_real = marker_specificity_index(real_n)
    msi_fake = marker_specificity_index(fake_n)

    metrics["text2cell"] = {
        "gene_mean_pearson": float(pr_mean[0]),
        "gene_mean_R2": float(r2_mean),
        "gene_var_pearson": float(pr_var[0]),
        "FD_gene_pca50": float(fd_gene),
        "FD_scgpt_embedding": float(fd_emb),
        "marker_specificity_real": float(msi_real),
        "marker_specificity_generated": float(msi_fake),
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

    # ── Build Figure 1 ──
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35)

    # (a) Gene Mean Scatter
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.scatter(rm, fm, s=4, alpha=0.35, c="#1f77b4", edgecolors="none", rasterized=True)
    lim = [min(rm.min(), fm.min()) - 0.05, max(rm.max(), fm.max()) + 0.05]
    ax_a.plot(lim, lim, "--", color="#999999", lw=0.8)
    ax_a.set_xlim(lim); ax_a.set_ylim(lim)
    ax_a.set_xlabel("Real (scGPT-reconstructed)")
    ax_a.set_ylabel("Generated")
    ax_a.set_title("(a) Gene Mean Expression")
    ax_a.text(0.05, 0.92, f"Pearson r = {pr_mean[0]:.4f}\nR² = {r2_mean:.4f}\nMSI(real)={msi_real:.2f}\nMSI(gen)={msi_fake:.2f}",
              transform=ax_a.transAxes, fontsize=9, va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # (b) UMAP: merged real + generated
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_umap_panel(real_recon, fake_adata, ax_b)
    ax_b.text(0.02, -0.12, "Note: gene-level metrics are influenced by scGPT encode→decode bottleneck",
              transform=ax_b.transAxes, fontsize=7, color="#555555")

    # (c) Marker Gene Heatmap
    ax_c = fig.add_subplot(gs[1, 0])
    _plot_marker_heatmap(real_n, fake_n, ax_c)

    # (d) Gene Variance Scatter
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.scatter(rv, fv, s=4, alpha=0.35, c="#ff7f0e", edgecolors="none", rasterized=True)
    lim_v = [min(rv.min(), fv.min()) - 0.05, max(rv.max(), fv.max()) + 0.05]
    ax_d.plot(lim_v, lim_v, "--", color="#999999", lw=0.8)
    ax_d.set_xlim(lim_v); ax_d.set_ylim(lim_v)
    ax_d.set_xlabel("Real variance")
    ax_d.set_ylabel("Generated variance")
    ax_d.set_title("(d) Gene Variance Correlation")
    ax_d.text(0.05, 0.92, f"Pearson r = {pr_var[0]:.4f}",
              transform=ax_d.transAxes, fontsize=9, va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.savefig(output_dir / "figure1_text2cell.png")
    fig.savefig(output_dir / "figure1_text2cell.pdf")
    plt.close(fig)
    logger.info(f"  Saved Figure 1 -> {output_dir / 'figure1_text2cell.png'}")

    return real_emb, fake_emb


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


def _plot_marker_heatmap(real_n, fake_n, ax):
    """Side-by-side marker gene heatmap: Real (left) | Generated (right)."""
    all_markers = []
    marker_ct_labels = []
    for ct, genes in MARKER_GENES.items():
        for g in genes:
            if g in real_n.var_names and g in fake_n.var_names:
                all_markers.append(g)
                marker_ct_labels.append(ct)

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

def figure2_cell2cell(adata_ref, indices_map, scgpt, output_dir, args, metrics):
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

    target_prompt = TEXT2CELL_PROMPTS.get(
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
    gene_names_aligned = rs.var_names.tolist()

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

    metrics["cell2cell"] = {
        "source_type": src_type,
        "target_type": tgt_type,
        "DEG_overlap_top50": float(overlap),
        "direction_accuracy": float(dir_acc),
        "delta_R2": float(r2_delta),
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

    # (c) Top DEG direction barplot
    ax = axes[2]
    show_n = 20  # top 20 for readability
    top_genes = [gene_names_aligned[i] for i in idx_real_top[:show_n]]
    top_dr = delta_real[idx_real_top[:show_n]]
    top_dp = delta_pred[idx_real_top[:show_n]]

    y_pos = np.arange(show_n)
    ax.barh(y_pos, top_dr, height=0.4, align="center",
            color="#2166ac", alpha=0.7, label="Real shift")
    ax.barh(y_pos + 0.4, top_dp, height=0.4, align="center",
            color="#d6604d", alpha=0.7, label="Predicted shift")
    ax.set_yticks(y_pos + 0.2)
    ax.set_yticklabels(top_genes, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Δ Expression (log1p)")
    ax.set_title(f"(c) Top-20 DEG Shifts\nOverlap={overlap:.0%}, DirAcc={dir_acc:.0%}")
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "figure2_cell2cell.png")
    fig.savefig(output_dir / "figure2_cell2cell.pdf")
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

    # heuristic match rate
    def map_label_to_category(label: str):
        l = label.lower()
        if "cd8" in l or "cytotoxic" in l:
            return "CD8+ T cells"
        if "nk" in l or "natural killer" in l:
            return "NK cells"
        if "macrophage" in l or "mph" in l or "monocyte" in l:
            return "Macrophages"
        if any(k in l for k in ["epithelial", "at1", "at2", "ciliated", "secretory", "club"]):
            return "Epithelial cells"
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

    metrics["celltypist"] = {
        "model": model_name,
        "top_label_purity": {k: float(v) for k, v in purity.items()},
        "heuristic_match_rate": {k: float(v) for k, v in match_rates.items()},
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

    # (b) Label distribution (Generated vs Real)
    ax = axes[1]
    labels_plot = [lab for lab in top_labels]
    gen_counts = Counter(pred_labels)
    gen_vals = []
    for lab in labels_plot:
        if lab == "Other":
            gen_vals.append(sum(v for k, v in gen_counts.items() if k not in top_labels))
        else:
            gen_vals.append(gen_counts.get(lab, 0))

    if real_labels is not None:
        real_counts = Counter(real_labels)
        real_vals = []
        for lab in labels_plot:
            if lab == "Other":
                real_vals.append(sum(v for k, v in real_counts.items() if k not in top_labels))
            else:
                real_vals.append(real_counts.get(lab, 0))
        y = np.arange(len(labels_plot))
        ax.barh(y - 0.2, gen_vals, height=0.35, color="#4c72b0", alpha=0.8, label="Generated")
        ax.barh(y + 0.2, real_vals, height=0.35, color="#55a868", alpha=0.8, label="Real")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    else:
        y = np.arange(len(labels_plot))
        ax.barh(y, gen_vals, height=0.5, color="#4c72b0", alpha=0.8, label="Generated")

    ax.set_yticks(range(len(labels_plot)))
    ax.set_yticklabels(labels_plot, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("# Cells")
    ax.set_title("(b) CellTypist Label Distribution")

    fig.tight_layout()
    fig.savefig(output_dir / "figure3_celltypist.png")
    fig.savefig(output_dir / "figure3_celltypist.pdf")
    plt.close(fig)
    logger.info(f"  Saved Figure 3 -> {output_dir / 'figure3_celltypist.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Biological Validation v2")
    parser.add_argument("--reference_h5ad", required=True)
    parser.add_argument("--dataset_key", required=True)
    parser.add_argument("--output_dir", default="figures/biovalidation")
    parser.add_argument("--num_cells_per_type", type=int, default=200)
    parser.add_argument("--num_steps", type=int, default=4)
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
    args = parser.parse_args()

    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading reference: {args.reference_h5ad}")
    adata_ref = sc.read_h5ad(args.reference_h5ad)
    indices_map = load_dataset_indices(args.dataset_key)

    scgpt = ScGPTDecoder(model_dir=args.scgpt_model_dir, device=torch.device(args.device))
    metrics: Dict = {}

    fake = None
    real = None
    if not args.skip_figure1:
        fake = generate_cells_text2cell(adata_ref, args)
        real = build_real_subset(adata_ref, indices_map, args.num_cells_per_type)
        figure1_text2cell(real, fake, scgpt, output_dir, metrics)

    if not args.skip_figure2:
        figure2_cell2cell(adata_ref, indices_map, scgpt, output_dir, args, metrics)

    if not args.skip_figure3:
        if fake is None:
            fake = generate_cells_text2cell(adata_ref, args)
        if real is None:
            real = build_real_subset(adata_ref, indices_map, args.num_cells_per_type)
        figure3_celltypist(fake, output_dir, metrics, model_name=args.celltypist_model, real_adata=real)

    # ── Save metrics ──
    metrics_path = output_dir / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved -> {metrics_path}")
    logger.info("=== Validation complete ===")


if __name__ == "__main__":
    main()
