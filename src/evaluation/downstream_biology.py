"""
downstream_biology.py — Downstream biological analysis for CLOP-DiT.

Constructs matched AnnData objects for real and generated cells, then runs:
  1. Clustering alignment  (Leiden on real-only vs real+gen, ARI/NMI)
  2. Classifier alignment  (LogReg trained on real, evaluated on generated)
  3. DE concordance         (logFC correlation for key contrasts)
  4. kNN mixing score       (fraction of cross-origin neighbours per type)

All results written to ``results/downstream/`` as JSON summaries ready for
Panel P/Q/R visualisation.

Usage:
    python -m src.evaluation.downstream_biology          # default paths
    python -m src.evaluation.downstream_biology --help   # see options
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# AnnData construction
# ──────────────────────────────────────────────────────────────

def build_matched_adata(
    real_expr_path: str = "results/real_expression.npy",
    gen_expr_path: str = "results/generated_expression.npy",
    real_labels_path: str = "results/real_expression_labels.npy",
    gen_labels_path: str = "results/generated_expression_labels.npy",
    gene_names_path: str = "results/expression_gene_names.json",
    type_names: Optional[Dict[int, str]] = None,
) -> Tuple:
    """Build matched AnnData objects with shared preprocessing.

    Returns ``(adata_real, adata_gen, adata_combined)`` with columns:
        obs["cell_type"]  — string label
        obs["source"]     — "real" or "generated"
        obs["type_id"]    — integer type id
    """
    import anndata as ad
    import scanpy as sc

    for p in [real_expr_path, gen_expr_path, gene_names_path]:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    real = np.load(real_expr_path).astype(np.float32)
    gen = np.load(gen_expr_path).astype(np.float32)
    real_labels = np.load(real_labels_path) if Path(real_labels_path).exists() else np.zeros(real.shape[0], dtype=int)
    gen_labels = np.load(gen_labels_path) if Path(gen_labels_path).exists() else np.zeros(gen.shape[0], dtype=int)

    with open(gene_names_path) as f:
        gene_names = json.load(f)

    type_names = type_names or {}

    def _make_adata(X, labels, source_tag):
        adata = ad.AnnData(X=X)
        adata.var_names = gene_names
        adata.obs["type_id"] = labels.astype(int)
        adata.obs["cell_type"] = [type_names.get(int(t), f"Type_{t}") for t in labels]
        adata.obs["source"] = source_tag
        return adata

    adata_real = _make_adata(real, real_labels, "real")
    adata_gen = _make_adata(gen, gen_labels, "generated")

    import anndata as ad
    adata_combined = ad.concat([adata_real, adata_gen], join="outer")
    adata_combined.obs_names_make_unique()

    # Shared preprocessing: HVG selection on combined, then PCA
    sc.pp.highly_variable_genes(adata_combined, n_top_genes=min(1000, adata_combined.n_vars),
                                flavor="seurat_v3", subset=False)
    n_hvg = adata_combined.var["highly_variable"].sum()
    logger.info(f"Built matched AnnData: real={real.shape[0]}, gen={gen.shape[0]}, "
                f"genes={len(gene_names)}, HVGs={n_hvg}")

    return adata_real, adata_gen, adata_combined


# ──────────────────────────────────────────────────────────────
# 1. Clustering alignment
# ──────────────────────────────────────────────────────────────

def clustering_alignment(
    adata_combined,
    resolution: float = 1.0,
    n_pcs: int = 30,
) -> Dict:
    """Run Leiden clustering on combined data and compute alignment metrics.

    Returns dict with ARI, NMI, per-type mixing scores, and cluster purity.
    """
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    adata = adata_combined.copy()

    # PCA → neighbors → Leiden
    sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1))
    sc.pp.neighbors(adata, n_pcs=min(n_pcs, adata.n_vars - 1))
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden")
    sc.tl.umap(adata)

    # Ground-truth vs Leiden
    gt = adata.obs["cell_type"].values
    leiden = adata.obs["leiden"].values
    ari = adjusted_rand_score(gt, leiden)
    nmi = normalized_mutual_info_score(gt, leiden, average_method="arithmetic")

    # Per-type mixing: for each type, what fraction of kNN neighbors are cross-source?
    source = adata.obs["source"].values
    from scipy.sparse import issparse
    conn = adata.obsp["connectivities"]
    if issparse(conn):
        conn = conn.toarray()

    mixing_scores = {}
    per_type_scores = {}
    for ct in np.unique(gt):
        ct_mask = gt == ct
        ct_real_mask = ct_mask & (source == "real")
        ct_gen_mask = ct_mask & (source == "generated")
        if ct_real_mask.sum() < 2 or ct_gen_mask.sum() < 2:
            continue
        # For generated cells of this type, fraction of neighbors that are real
        gen_idx = np.where(ct_gen_mask)[0]
        cross_fracs = []
        for i in gen_idx:
            neighbors = np.where(conn[i] > 0)[0]
            if len(neighbors) == 0:
                continue
            n_real_neighbors = sum(source[j] == "real" for j in neighbors)
            cross_fracs.append(n_real_neighbors / len(neighbors))
        per_type_scores[ct] = float(np.mean(cross_fracs)) if cross_fracs else 0.0

    # Cluster purity: for each Leiden cluster, max fraction of one cell type
    cluster_purity = []
    for cl in np.unique(leiden):
        cl_mask = leiden == cl
        cl_types = gt[cl_mask]
        if len(cl_types) == 0:
            continue
        _, counts = np.unique(cl_types, return_counts=True)
        cluster_purity.append(float(counts.max() / counts.sum()))

    # Generated cells: fraction assigned to correct cluster
    gen_mask = source == "generated"
    if gen_mask.any():
        gen_clusters = leiden[gen_mask]
        gen_types = gt[gen_mask]
        # For each cluster, determine dominant real type
        cluster_dominant = {}
        real_mask = source == "real"
        for cl in np.unique(leiden):
            cl_real = gt[(leiden == cl) & real_mask]
            if len(cl_real) == 0:
                continue
            vals, counts = np.unique(cl_real, return_counts=True)
            cluster_dominant[cl] = vals[counts.argmax()]
        correct = sum(1 for cl, t in zip(gen_clusters, gen_types)
                      if cluster_dominant.get(cl, "") == t)
        gen_cluster_accuracy = correct / len(gen_clusters) if len(gen_clusters) > 0 else 0
    else:
        gen_cluster_accuracy = 0

    result = {
        "ari_gt_vs_leiden": float(ari),
        "nmi_gt_vs_leiden": float(nmi),
        "mean_cluster_purity": float(np.mean(cluster_purity)) if cluster_purity else 0,
        "gen_cluster_accuracy": float(gen_cluster_accuracy),
        "per_type_mixing": per_type_scores,
        "mean_mixing_score": float(np.mean(list(per_type_scores.values()))) if per_type_scores else 0,
        "n_leiden_clusters": len(np.unique(leiden)),
        "resolution": resolution,
    }

    # Store UMAP coords for plotting
    result["_umap_coords"] = adata.obsm["X_umap"]
    result["_source"] = source
    result["_cell_type"] = gt
    result["_leiden"] = leiden

    logger.info(f"Clustering: ARI={ari:.3f}, NMI={nmi:.3f}, "
                f"gen_cluster_acc={gen_cluster_accuracy:.3f}, "
                f"mean_mixing={result['mean_mixing_score']:.3f}")
    return result


# ──────────────────────────────────────────────────────────────
# 2. Classifier alignment
# ──────────────────────────────────────────────────────────────

def classifier_alignment(
    adata_real,
    adata_gen,
    n_pcs: int = 30,
    test_fraction: float = 0.2,
) -> Dict:
    """Train LogReg on real cells, evaluate on generated cells.

    Returns accuracy, macro-F1, per-type accuracy, confusion matrix,
    and a real-vs-generated discriminator AUC.
    """
    import scanpy as sc
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix as sklearn_cm, roc_auc_score
    )
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    # Combine for shared PCA
    import anndata as ad
    combined = ad.concat([adata_real, adata_gen], join="outer")
    combined.obs_names_make_unique()
    sc.pp.pca(combined, n_comps=min(n_pcs, combined.n_vars - 1))

    n_real = adata_real.n_obs
    real_pca = combined.obsm["X_pca"][:n_real]
    gen_pca = combined.obsm["X_pca"][n_real:]

    le = LabelEncoder()
    real_y = le.fit_transform(adata_real.obs["cell_type"].values)
    gen_y = le.transform(adata_gen.obs["cell_type"].values)
    class_names = le.classes_.tolist()

    # Split real into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        real_pca, real_y, test_size=test_fraction, random_state=42, stratify=real_y
    )

    clf = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1,
                             class_weight="balanced")
    clf.fit(X_train, y_train)

    # Evaluate on real test set
    real_test_pred = clf.predict(X_test)
    real_test_acc = accuracy_score(y_test, real_test_pred)
    real_test_f1 = f1_score(y_test, real_test_pred, average="macro", zero_division=0)

    # Evaluate on generated
    gen_pred = clf.predict(gen_pca)
    gen_acc = accuracy_score(gen_y, gen_pred)
    gen_f1 = f1_score(gen_y, gen_pred, average="macro", zero_division=0)
    cm = sklearn_cm(gen_y, gen_pred)

    # Per-type accuracy on generated
    per_type_acc = {}
    for i, name in enumerate(class_names):
        mask = gen_y == i
        if mask.sum() > 0:
            per_type_acc[name] = float((gen_pred[mask] == i).mean())

    # Real-vs-generated discriminator (should be near 0.5 if distributions match)
    disc_X = np.vstack([real_pca, gen_pca])
    disc_y = np.concatenate([np.zeros(len(real_pca)), np.ones(len(gen_pca))])
    disc_clf = LogisticRegression(max_iter=1000, random_state=42)
    # Use cross-val-like approach: train on 80%, test on 20%
    from sklearn.model_selection import cross_val_predict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disc_proba = cross_val_predict(disc_clf, disc_X, disc_y,
                                       cv=5, method="predict_proba")[:, 1]
    disc_auc = roc_auc_score(disc_y, disc_proba)

    result = {
        "real_test_accuracy": float(real_test_acc),
        "real_test_f1": float(real_test_f1),
        "gen_accuracy": float(gen_acc),
        "gen_f1": float(gen_f1),
        "per_type_accuracy": per_type_acc,
        "discriminator_auc": float(disc_auc),
        "n_classes": len(class_names),
        "class_names": class_names,
        "_confusion_matrix": cm.tolist(),
        "_gen_y": gen_y.tolist(),
        "_gen_pred": gen_pred.tolist(),
        "_disc_proba": disc_proba.tolist(),
        "_disc_y": disc_y.tolist(),
    }

    logger.info(f"Classifier: real_test_acc={real_test_acc:.3f}, "
                f"gen_acc={gen_acc:.3f}, gen_F1={gen_f1:.3f}, "
                f"disc_AUC={disc_auc:.3f}")
    return result


# ──────────────────────────────────────────────────────────────
# 3. Differential expression concordance
# ──────────────────────────────────────────────────────────────

# Default contrasts for DE analysis (names must match caption-derived labels)
DE_CONTRASTS = [
    ("CD8+ cytotoxic T lymphocytes", "CD4+ helper T lymphocytes"),
    ("Tissue-resident macrophages", "Circulating monocytes"),
    ("Generic epithelial cells", "Fibroblasts and mesenchymal stromal cells"),
]


def _resolve_contrast_types(
    adata, type_a: str, type_b: str
) -> Tuple[Optional[str], Optional[str]]:
    """Fuzzy-match contrast type names against available cell types.

    Uses substring matching: returns the first available name that contains
    the target as a prefix (case-insensitive), or vice versa.
    """
    available = adata.obs["cell_type"].unique().tolist()
    resolved = []
    for target in [type_a, type_b]:
        found = None
        target_low = target.lower()
        # Exact match first
        for avail in available:
            if avail.lower() == target_low:
                found = avail
                break
        # Prefix / substring match
        if found is None:
            for avail in available:
                avail_low = avail.lower()
                if target_low.startswith(avail_low[:20]) or avail_low.startswith(target_low[:20]):
                    found = avail
                    break
        # Fallback: any significant overlap
        if found is None:
            target_words = set(target_low.split())
            for avail in available:
                avail_words = set(avail.lower().split())
                overlap = len(target_words & avail_words)
                if overlap >= min(2, len(target_words)):
                    found = avail
                    break
        resolved.append(found)
    return tuple(resolved)


def de_concordance(
    adata_real,
    adata_gen,
    contrasts: Optional[List[Tuple[str, str]]] = None,
    n_top: int = 100,
) -> Dict:
    """Compare DE results between real and generated for each contrast.

    For each contrast (typeA vs typeB):
      - Run Wilcoxon on real, on generated
      - Correlate logFC vectors
      - Compute Jaccard overlap of top-k significant genes
    """
    import scanpy as sc
    from scipy.stats import pearsonr, spearmanr

    contrasts = contrasts or DE_CONTRASTS

    results = {}
    for type_a, type_b in contrasts:
        # Resolve names
        ra, rb = _resolve_contrast_types(adata_real, type_a, type_b)
        ga, gb = _resolve_contrast_types(adata_gen, type_a, type_b)

        if not all([ra, rb, ga, gb]):
            logger.info(f"DE contrast {type_a} vs {type_b}: types not found, skipping")
            continue

        contrast_name = f"{ra[:25]}_vs_{rb[:25]}"

        def _run_de(adata, group_a, group_b):
            """Run DE on a subset of adata."""
            sub = adata[adata.obs["cell_type"].isin([group_a, group_b])].copy()
            if sub.n_obs < 10:
                return None
            sc.tl.rank_genes_groups(sub, "cell_type", groups=[group_a],
                                    reference=group_b, method="wilcoxon",
                                    n_genes=sub.n_vars)
            names = sub.uns["rank_genes_groups"]["names"][group_a]
            scores = sub.uns["rank_genes_groups"]["scores"][group_a]
            logfc = sub.uns["rank_genes_groups"]["logfoldchanges"][group_a]
            pvals = sub.uns["rank_genes_groups"]["pvals_adj"][group_a]
            return {g: {"logfc": float(l), "score": float(s), "pval_adj": float(p)}
                    for g, l, s, p in zip(names, logfc, scores, pvals)}

        real_de = _run_de(adata_real, ra, rb)
        gen_de = _run_de(adata_gen, ga, gb)

        if real_de is None or gen_de is None:
            continue

        # Correlate logFC on shared genes
        shared_genes = sorted(set(real_de.keys()) & set(gen_de.keys()))
        if len(shared_genes) < 10:
            continue

        real_logfc = np.array([real_de[g]["logfc"] for g in shared_genes])
        gen_logfc = np.array([gen_de[g]["logfc"] for g in shared_genes])
        real_padj = np.array([real_de[g]["pval_adj"] for g in shared_genes])
        gen_padj = np.array([gen_de[g]["pval_adj"] for g in shared_genes])
        real_score = np.array([real_de[g]["score"] for g in shared_genes])
        gen_score = np.array([gen_de[g]["score"] for g in shared_genes])

        r_pearson, _ = pearsonr(real_logfc, gen_logfc)
        r_spearman, _ = spearmanr(real_logfc, gen_logfc)

        # Top-k overlap (by absolute logFC)
        real_topk = set(sorted(shared_genes, key=lambda g: abs(real_de[g]["logfc"]),
                               reverse=True)[:n_top])
        gen_topk = set(sorted(shared_genes, key=lambda g: abs(gen_de[g]["logfc"]),
                              reverse=True)[:n_top])
        jaccard = len(real_topk & gen_topk) / len(real_topk | gen_topk) if (real_topk | gen_topk) else 0

        # Sign agreement in top-k shared
        topk_shared = list(real_topk & gen_topk)
        if topk_shared:
            sign_agreement = np.mean([
                np.sign(real_de[g]["logfc"]) == np.sign(gen_de[g]["logfc"])
                for g in topk_shared
            ])
        else:
            sign_agreement = 0

        results[contrast_name] = {
            "logfc_pearson_r": float(r_pearson),
            "logfc_spearman_rho": float(r_spearman),
            "top_k_jaccard": float(jaccard),
            "top_k_sign_agreement": float(sign_agreement),
            "n_shared_genes": len(shared_genes),
            "n_top": n_top,
            "_real_logfc": real_logfc.tolist(),
            "_gen_logfc": gen_logfc.tolist(),
            "_real_padj": real_padj.tolist(),
            "_gen_padj": gen_padj.tolist(),
            "_real_score": real_score.tolist(),
            "_gen_score": gen_score.tolist(),
            "_shared_genes": shared_genes,
        }

        logger.info(f"DE {contrast_name}: logFC_r={r_pearson:.3f}, "
                    f"Jaccard@{n_top}={jaccard:.3f}, sign={sign_agreement:.3f}")

    return results


# ──────────────────────────────────────────────────────────────
# Orchestrator: run all downstream analyses
# ──────────────────────────────────────────────────────────────

def run_all_downstream(
    type_names: Optional[Dict[int, str]] = None,
    output_dir: str = "results/downstream",
    contrasts: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Dict]:
    """Run full downstream biology pipeline and save JSON summaries.

    Returns dict of {analysis_name: results_dict}.
    """
    import scanpy as sc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Running Downstream Biological Analysis")
    logger.info("=" * 60)

    # Build AnnData
    adata_real, adata_gen, adata_combined = build_matched_adata(
        type_names=type_names
    )

    # Shared preprocessing for DE (log-space needed for Wilcoxon)
    adata_real_de = adata_real.copy()
    adata_gen_de = adata_gen.copy()
    # The expression is already from scGPT (binned), normalize for DE
    sc.pp.normalize_total(adata_real_de, target_sum=1e4)
    sc.pp.log1p(adata_real_de)
    sc.pp.normalize_total(adata_gen_de, target_sum=1e4)
    sc.pp.log1p(adata_gen_de)

    all_results = {}

    # 1. Clustering
    logger.info("\n── Clustering alignment ──")
    try:
        clust = clustering_alignment(adata_combined)
        # Save serialisable subset
        save_clust = {k: v for k, v in clust.items() if not k.startswith("_")}
        with open(out / "clustering_alignment.json", "w") as f:
            json.dump(save_clust, f, indent=2)
        # Save arrays for plotting (UMAP coords, labels)
        if "_umap_coords" in clust:
            np.save(out / "clustering_umap_coords.npy", clust["_umap_coords"])
            np.save(out / "clustering_source.npy", clust["_source"])
            np.save(out / "clustering_cell_type.npy", clust["_cell_type"])
            np.save(out / "clustering_leiden.npy", clust["_leiden"])
        all_results["clustering"] = clust
        logger.info(f"Saved → {out / 'clustering_alignment.json'}")
    except Exception as e:
        logger.error(f"Clustering failed: {e}")

    # 2. Classifier
    logger.info("\n── Classifier alignment ──")
    try:
        classif = classifier_alignment(adata_real, adata_gen)
        save_classif = {k: v for k, v in classif.items() if not k.startswith("_")}
        # Include confusion matrix and ROC data in JSON (they're lists, not arrays)
        if "_confusion_matrix" in classif:
            save_classif["confusion_matrix"] = classif["_confusion_matrix"]
        if "_disc_proba" in classif:
            save_classif["disc_proba"] = classif["_disc_proba"]
            save_classif["disc_y"] = classif["_disc_y"]
        with open(out / "classifier_alignment.json", "w") as f:
            json.dump(save_classif, f, indent=2)
        all_results["classifier"] = classif
        logger.info(f"Saved → {out / 'classifier_alignment.json'}")
    except Exception as e:
        logger.error(f"Classifier failed: {e}")

    # 3. DE concordance
    logger.info("\n── DE concordance ──")
    try:
        de_results = de_concordance(adata_real_de, adata_gen_de, contrasts=contrasts)
        save_de = {}
        for k, v in de_results.items():
            save_de[k] = {kk: vv for kk, vv in v.items()}  # keep all keys for JSON
        with open(out / "de_concordance.json", "w") as f:
            json.dump(save_de, f, indent=2)
        all_results["de"] = de_results
        logger.info(f"Saved → {out / 'de_concordance.json'}")
    except Exception as e:
        logger.error(f"DE concordance failed: {e}")

    logger.info(f"\nDownstream analysis complete → {out}/")
    return all_results


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run downstream biological analysis")
    parser.add_argument("--output-dir", default="results/downstream")
    parser.add_argument("--caption-json", default="data/cached_latents_v5.2/text_captions_deduplicated.json")
    args = parser.parse_args()

    # Load type names
    type_names = {}
    cap_path = Path(args.caption_json)
    if cap_path.exists():
        with open(cap_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            name = v.split(" are ")[0] if " are " in v else v[:50]
            type_names[int(k)] = name

    run_all_downstream(type_names=type_names, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
