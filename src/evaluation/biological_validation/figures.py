# figures.py — Publication figures 1–4 for biological validation.
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from . import constants
from .metrics import (
    to_dense,
    log1p_safe,
    align_genes,
    compute_markers_by_cell_type,
    select_nonredundant_markers,
    compute_text2cell_metrics,
)

logger = logging.getLogger(__name__)


def figure1_text2cell_multi(dataset_runs, scgpt, output_dir, metrics):
    """Create Figure 1: multi-dataset Text2Cell biological fidelity."""
    logger.info("Creating Figure 1 — Text2Cell Biological Fidelity (multi-dataset)")

    per_dataset = []
    for ds in dataset_runs:
        cell_types = list(ds["prompts"].keys())
        marker_pool = compute_markers_by_cell_type(ds["real_adata"], cell_types, n_genes=6)
        marker_dict = select_nonredundant_markers(marker_pool, n_genes=4)
        if not any(marker_dict.values()):
            marker_dict = constants.MARKER_GENES

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
        "datasets": {d["label"]: d["metrics"] for d in per_dataset},
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
        _plot_umap_panel(d["real_recon"], dataset_runs[i]["fake_adata"], row_axes[0])
        row_axes[0].set_title(f"(a) UMAP Integration — {label}")
        _plot_marker_heatmap(d["real_n"], d["fake_n"], row_axes[1], marker_dict=d["marker_dict"])
        row_axes[1].set_title(f"(b) Marker Genes — {label}")
        ax = row_axes[2]
        ax.scatter(rm, fm, s=4, alpha=0.35, c="#1f77b4", edgecolors="none", rasterized=True)
        lim = [min(rm.min(), fm.min()) - 0.05, max(rm.max(), fm.max()) + 0.05]
        ax.plot(lim, lim, "--", color="#999999", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
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
    fig.savefig(output_dir / "figure1_text2cell_multi.png")
    fig.savefig(output_dir / "figure1_text2cell_multi.pdf")
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
        sc.pp.highly_variable_genes(combined, n_top_genes=min(2000, n_genes), flavor="seurat_v3", subset=True)
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
    marker_dict = marker_dict or constants.MARKER_GENES
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
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
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
    im = ax.imshow(combined, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    n_m = len(all_markers)
    xtick_pos = list(range(n_m)) + list(range(n_m, 2 * n_m))
    xtick_labels = all_markers + all_markers
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.axvline(n_m - 0.5, color="black", linewidth=1.5)
    ax.text(n_m * 0.5 - 0.5, -0.8, "Real", ha="center", fontsize=8, fontweight="bold", transform=ax.transData)
    ax.text(n_m * 1.5 - 0.5, -0.8, "Generated", ha="center", fontsize=8, fontweight="bold", transform=ax.transData)
    ax.set_title("(c) Marker Gene Expression (z-scored)")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("z-score", fontsize=7)


def figure2_cell2cell(adata_ref, indices_map, scgpt, output_dir, args, metrics, prompts, scripts_dir: Path | None = None):
    """Create Figure 2: 1×3 Cell2Cell editing panel."""
    from .io import load_class_from_script

    logger.info("Creating Figure 2 — Cell2Cell Editing Quality")

    scripts_dir = scripts_dir or Path(__file__).resolve().parent.parent.parent.parent / "scripts"
    Cell2CellInference = load_class_from_script(scripts_dir / "06_cell2cell_inference.py", "Cell2CellInference")

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

    target_prompt = prompts.get(tgt_type, f"{tgt_type} from human lung adenocarcinoma")
    edited = c2c.edit_adata(
        real_src, target_prompt=target_prompt,
        edit_strength=args.edit_strength, decode_expression=True,
        reference_adata=adata_ref,
        num_steps=args.num_steps, cfg_scale=args.cfg_scale,
    )

    rs, rt, pt = align_genes(log1p_safe(real_src), log1p_safe(real_tgt), log1p_safe(edited))
    delta_real = to_dense(rt.X).mean(0) - to_dense(rs.X).mean(0)
    delta_pred = to_dense(pt.X).mean(0) - to_dense(rs.X).mean(0)
    top_n = 50
    idx_real_top = np.argsort(-np.abs(delta_real))[:top_n]
    r2_delta = r2_score(delta_real, delta_pred)
    overlap = len(set(idx_real_top) & set(np.argsort(-np.abs(delta_pred))[:top_n])) / top_n
    dir_acc = float(np.mean(np.sign(delta_real[idx_real_top]) == np.sign(delta_pred[idx_real_top])))

    src_emb = scgpt.encode(real_src)
    tgt_emb = scgpt.encode(real_tgt)
    edit_emb = edited.obsm.get("X_edited_emb")
    if edit_emb is None:
        edit_emb = scgpt.encode(edited)

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
            "DEG_overlap_top50": "Fraction of top-50 DEG shared between real and predicted transition.",
            "direction_accuracy": "Fraction of top-50 real DEGs where predicted up/down matches.",
            "delta_R2": "R-squared between real vs predicted per-gene expression shift.",
            "cosine_gain": "Mean cosine similarity gain of edited cells toward target vs source.",
        },
    }

    del c2c
    gc.collect()
    torch.cuda.empty_cache()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    X_all = np.vstack([src_emb, edit_emb, tgt_emb])
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X_all)
    n_s, n_e = len(src_emb), len(edit_emb)
    sp, ep, tp = Xp[:n_s], Xp[n_s : n_s + n_e], Xp[n_s + n_e :]

    ax = axes[0]
    ax.scatter(tp[:, 0], tp[:, 1], s=10, alpha=0.4, c=constants.COLORS["target"],
               label=f"Real {tgt_type}", edgecolors="none", rasterized=True)
    ax.scatter(sp[:, 0], sp[:, 1], s=10, alpha=0.4, c=constants.COLORS["source"],
               label=f"Real {src_type}", edgecolors="none", rasterized=True)
    n_arrows = min(150, n_s)
    arrow_idx = rng.choice(n_s, size=n_arrows, replace=False)
    for i in arrow_idx:
        ax.annotate("", xy=(ep[i, 0], ep[i, 1]), xytext=(sp[i, 0], sp[i, 1]),
                    arrowprops=dict(arrowstyle="->", color=constants.COLORS["edited"], alpha=0.4, lw=0.6))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"(a) Cell Editing Vector Field\n{src_type} → {tgt_type}")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    ax = axes[1]
    ax.scatter(delta_real, delta_pred, s=4, alpha=0.3, c="#ff7f0e", edgecolors="none", rasterized=True)
    dlim = [min(delta_real.min(), delta_pred.min()), max(delta_real.max(), delta_pred.max())]
    ax.plot(dlim, dlim, "--", color="#999999", lw=0.8)
    ax.set_xlabel("Real Δ expression")
    ax.set_ylabel("Predicted Δ expression")
    ax.set_title("(b) Per-Gene Expression Shift")
    ax.text(0.05, 0.92, f"R² = {r2_delta:.4f}\n(<0 = worse than zero-shift)",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax = axes[2]
    ax.boxplot([cos_to_src, cos_to_tgt], labels=["To Source", "To Target"],
               widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#cccccc", alpha=0.7),
               medianprops=dict(color="#000000"))
    ax.set_ylabel("Cosine similarity (scGPT emb)")
    ax.set_title(f"(c) Embedding Proximity\nGain={cos_gain:.3f}")

    fig.tight_layout()
    fig.savefig(output_dir / "figure2_cell2cell.png")
    fig.savefig(output_dir / "figure2_cell2cell.pdf")
    plt.close(fig)
    logger.info(f"  Saved Figure 2 -> {output_dir / 'figure2_cell2cell.png'}")


def figure3_celltypist(fake_adata, output_dir, metrics, model_name=None, real_adata=None):
    """Create Figure 3: external cell identity validation using CellTypist."""
    logger.info("Creating Figure 3 — External Cell Identity Validation (CellTypist)")
    model_name = model_name or constants.CELLTYPIST_MODEL

    try:
        import celltypist
    except Exception as e:
        logger.warning(f"CellTypist not available: {e}")
        return

    from collections import Counter

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

    def map_label_to_category(label: str):
        l = label.lower()
        if "cd8" in l or "cytotoxic" in l:
            return "CD8+ T cells"
        if "cd4" in l:
            return "CD4+ T cells"
        if "nk" in l or "natural killer" in l:
            return "NK cells"
        if "macrophage" in l or "mph" in l:
            return "Macrophages"
        if "monocyte" in l:
            return "Monocytes"
        if "epithelial" in l or "at1" in l or "at2" in l:
            return "Epithelial cells"
        if "fibroblast" in l or "mesenchymal" in l:
            return "Fibroblasts"
        if "b cell" in l or "b lymph" in l:
            return "B cells"
        return None

    match_rates = {}
    for ct in prompt_types:
        ct_labels = pred_labels[prompts == ct]
        if len(ct_labels) == 0:
            match_rates[ct] = float("nan")
            continue
        mapped = [map_label_to_category(lab) for lab in ct_labels]
        match_rates[ct] = float(np.mean([m == ct for m in mapped if m is not None])) if any(m is not None for m in mapped) else 0.0

    gen_counts = Counter(pred_labels)
    gen_total = sum(gen_counts.values())
    probs = np.array([v / gen_total for v in gen_counts.values()]) if gen_total > 0 else np.array([])
    label_entropy = float(-np.sum(probs * np.log2(probs + 1e-12))) if len(probs) > 0 else float("nan")

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
        "interpretation": {},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(top_labels)))
    ax.set_xticklabels(top_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(prompt_types)))
    ax.set_yticklabels(prompt_types, fontsize=8)
    ax.set_title(f"(a) CellTypist Confusion (row-normalized)\nModel: {model_name}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("Fraction", fontsize=7)

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
    fig.savefig(output_dir / "figure3_celltypist.png")
    fig.savefig(output_dir / "figure3_celltypist.pdf")
    plt.close(fig)
    logger.info(f"  Saved Figure 3 -> {output_dir / 'figure3_celltypist.png'}")


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
        ax.text(0.5, 0.5, "CellTypist match\nnot available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_dir / "figure4_summary.png")
    fig.savefig(output_dir / "figure4_summary.pdf")
    plt.close(fig)
    logger.info(f"  Saved Figure 4 -> {output_dir / 'figure4_summary.png'}")
