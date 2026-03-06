# metrics.py — Metric computations and data helpers for biological validation.
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import anndata as ad
import scanpy as sc
from scipy import stats
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from . import constants

logger = logging.getLogger(__name__)


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


def marker_specificity_index(
    adata: ad.AnnData, marker_dict: Dict[str, List[str]] | None = None
) -> float:
    """Marker specificity index (MSI). Higher = more specific expression in intended cell type."""
    marker_dict = marker_dict or constants.MARKER_GENES
    markers = []
    for ct, genes in marker_dict.items():
        for g in genes:
            if g in adata.var_names:
                markers.append((ct, g))
    if not markers:
        return float("nan")

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
