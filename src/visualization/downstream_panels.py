"""
downstream_panels.py — Panels P, Q, R: downstream biological validation figures.

  Panel P: Clustering & mixing (UMAP overlay, kNN mixing bars, ARI/NMI gauges)
  Panel Q: Classifier alignment (confusion matrix, per-type accuracy, ROC curve)
  Panel R: DE & pathway concordance (logFC scatter, gene overlap heatmap, summary bars)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, TYPE_PALETTE, save_panel, style_axes, quality_color

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# PANEL P: Clustering & Mixing
# ──────────────────────────────────────────────────────────────

def plot_clustering_panel(
    clustering_data: Dict,
    type_names: Dict[int, str],
    output_dir: Path,
    dpi: int = 300,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Panel P: Clustering alignment and real-generated mixing.

    P1: UMAP of real+gen coloured by cell type (generated outlined)
    P2: Per-type kNN mixing score bars
    P3: ARI/NMI/cluster-purity gauges + gen cluster accuracy
    """
    umap_coords = clustering_data.get("_umap_coords")
    source = clustering_data.get("_source")
    cell_type = clustering_data.get("_cell_type")

    if umap_coords is None:
        logger.info("No clustering UMAP data — skipping Panel P")
        return None

    umap_coords = np.asarray(umap_coords)
    source = np.asarray(source)
    cell_type = np.asarray(cell_type)

    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 0.8], wspace=0.3)
    fig.suptitle("Downstream: Clustering & Real–Generated Mixing",
                 fontsize=14, fontweight="bold")

    # ── P1: UMAP coloured by cell type, generated cells outlined ──
    ax = fig.add_subplot(gs[0])
    unique_types = np.unique(cell_type)
    # Build colour map
    ct_colors = {}
    for i, ct in enumerate(sorted(unique_types)):
        ct_colors[ct] = TYPE_PALETTE[i % len(TYPE_PALETTE)]

    # Plot real cells first (plain dots)
    real_mask = source == "real"
    gen_mask = source == "generated"
    for ct in unique_types:
        mask = real_mask & (cell_type == ct)
        if mask.any():
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[ct_colors[ct]], s=4, alpha=0.3, rasterized=True)
    # Plot generated cells (outlined triangles)
    for ct in unique_types:
        mask = gen_mask & (cell_type == ct)
        if mask.any():
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[ct_colors[ct]], s=12, alpha=0.6, marker="^",
                       edgecolors="black", linewidths=0.3, rasterized=True)

    # Legend entries
    ax.scatter([], [], c="gray", s=15, marker="o", label="Real")
    ax.scatter([], [], c="gray", s=15, marker="^", edgecolors="black",
               linewidths=0.3, label="Generated")
    ax.legend(fontsize=9, loc="upper right", markerscale=2)
    style_axes(ax, "umap", title="P1: Expression UMAP (Real + Generated)",
               xlabel="UMAP 1", ylabel="UMAP 2")

    # ── P2: Per-type kNN mixing score ──
    ax2 = fig.add_subplot(gs[1])
    mixing = clustering_data.get("per_type_mixing", {})
    if mixing:
        sorted_types = sorted(mixing.keys(), key=lambda k: mixing[k])
        vals = [mixing[t] for t in sorted_types]
        short_names = [t[:25] for t in sorted_types]
        bar_colors = [quality_color(v, (0.3, 0.15)) for v in vals]

        y_pos = np.arange(len(sorted_types))
        ax2.barh(y_pos, vals, color=bar_colors, height=0.7, edgecolor="white", linewidth=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(short_names, fontsize=6)
        ax2.axvline(x=clustering_data.get("mean_mixing_score", 0),
                     color="#D32F2F", linestyle="--", alpha=0.7, linewidth=1.5,
                     label=f"mean={clustering_data.get('mean_mixing_score', 0):.3f}")
        ax2.set_xlim(0, max(max(vals) * 1.1, 0.5))
        ax2.legend(fontsize=8)
        style_axes(ax2, "bar", title="P2: kNN Mixing Score (gen→real neighbours)",
                   xlabel="Fraction Real Neighbours")
    else:
        ax2.text(0.5, 0.5, "No mixing data", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12)
        ax2.set_title("P2: kNN Mixing Score")

    # ── P3: Alignment gauges ──
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")

    gauge_items = [
        ("ARI", clustering_data.get("ari_gt_vs_leiden", 0), (0.6, 0.3)),
        ("NMI", clustering_data.get("nmi_gt_vs_leiden", 0), (0.6, 0.3)),
        ("Cluster\nPurity", clustering_data.get("mean_cluster_purity", 0), (0.8, 0.5)),
        ("Gen Cluster\nAccuracy", clustering_data.get("gen_cluster_accuracy", 0), (0.5, 0.25)),
        ("Mean\nMixing", clustering_data.get("mean_mixing_score", 0), (0.3, 0.15)),
    ]

    n_items = len(gauge_items)
    for i, (label, val, thresh) in enumerate(gauge_items):
        y = 1.0 - (i + 0.5) / n_items
        color = quality_color(val, thresh)
        # Big number
        ax3.text(0.55, y, f"{val:.3f}", fontsize=18, fontweight="bold",
                 color=color, ha="center", va="center",
                 transform=ax3.transAxes)
        # Label
        ax3.text(0.1, y, label, fontsize=10, ha="left", va="center",
                 transform=ax3.transAxes, color="#333")

    ax3.set_title("P3: Alignment Metrics", fontsize=12, fontweight="bold",
                  pad=10)
    n_clusters = clustering_data.get("n_leiden_clusters", "?")
    ax3.text(0.5, 0.02, f"Leiden clusters: {n_clusters}",
             transform=ax3.transAxes, ha="center", fontsize=9, color="#666")

    if save:
        path = save_panel(fig, output_dir / "panel_p_clustering_mixing.png", dpi)
        logger.info(f"Saved Panel P → {path}")
    return fig


# ──────────────────────────────────────────────────────────────
# PANEL Q: Classifier Alignment
# ──────────────────────────────────────────────────────────────

def plot_classifier_panel(
    classifier_data: Dict,
    output_dir: Path,
    dpi: int = 300,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Panel Q: Classifier alignment between real-trained and generated.

    Q1: Confusion matrix (real-trained classifier on generated cells)
    Q2: Per-type accuracy bars with bootstrap CIs
    Q3: Discriminator ROC curve (should be near diagonal if distributions match)
    """
    cm = classifier_data.get("_confusion_matrix")
    if cm is None:
        logger.info("No classifier data — skipping Panel Q")
        return None

    cm = np.array(cm)
    class_names = classifier_data.get("class_names", [f"C{i}" for i in range(cm.shape[0])])
    per_type_acc = classifier_data.get("per_type_accuracy", {})

    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 0.9], wspace=0.3)
    gen_acc = classifier_data.get("gen_accuracy", 0)
    gen_f1 = classifier_data.get("gen_f1", 0)
    disc_auc = classifier_data.get("discriminator_auc", 0)
    fig.suptitle(
        f"Downstream: Classifier Alignment  —  "
        f"Gen Acc={gen_acc:.3f}  |  Gen F1={gen_f1:.3f}  |  Disc AUC={disc_auc:.3f}",
        fontsize=14, fontweight="bold",
    )

    # ── Q1: Confusion matrix ──
    ax = fig.add_subplot(gs[0])
    # Normalise per row (true label)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    n_classes = len(class_names)
    short_names = [n[:18] for n in class_names]

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(short_names, rotation=90, fontsize=max(4, 9 - n_classes // 10))
    ax.set_yticklabels(short_names, fontsize=max(4, 9 - n_classes // 10))

    # Highlight diagonal
    for i in range(min(n_classes, cm_norm.shape[0])):
        val = cm_norm[i, i]
        color = "white" if val > 0.5 else "black"
        ax.text(i, i, f"{val:.2f}", ha="center", va="center",
                fontsize=max(5, 8 - n_classes // 15), color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.6, label="Recall")
    style_axes(ax, "heatmap", title="Q1: Confusion Matrix (on Generated Cells)",
               xlabel="Predicted", ylabel="True Type")

    # ── Q2: Per-type accuracy bars ──
    ax2 = fig.add_subplot(gs[1])
    if per_type_acc:
        sorted_types = sorted(per_type_acc.keys(), key=lambda k: per_type_acc[k])
        vals = [per_type_acc[t] for t in sorted_types]
        short = [t[:22] for t in sorted_types]
        bar_colors = [quality_color(v, (0.7, 0.4)) for v in vals]

        y_pos = np.arange(len(sorted_types))
        ax2.barh(y_pos, vals, color=bar_colors, height=0.7,
                 edgecolor="white", linewidth=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(short, fontsize=max(5, 8 - len(sorted_types) // 10))
        ax2.axvline(x=gen_acc, color="#D32F2F", linestyle="--", alpha=0.7,
                     linewidth=1.5, label=f"overall={gen_acc:.3f}")
        ax2.set_xlim(0, 1.05)
        ax2.legend(fontsize=8)
        style_axes(ax2, "bar", title="Q2: Per-Type Classification Accuracy",
                   xlabel="Accuracy")
    else:
        ax2.text(0.5, 0.5, "No per-type data", ha="center", va="center",
                 transform=ax2.transAxes)
        ax2.set_title("Q2: Per-Type Accuracy")

    # ── Q3: Discriminator ROC ──
    ax3 = fig.add_subplot(gs[2])
    disc_proba = classifier_data.get("_disc_proba")
    disc_y = classifier_data.get("_disc_y")

    if disc_proba is not None and disc_y is not None:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(disc_y, disc_proba)
        ax3.plot(fpr, tpr, color=COLORS["real"], linewidth=2,
                 label=f"Disc. AUC = {disc_auc:.3f}")
        ax3.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.6,
                 label="Random (AUC = 0.5)")
        ax3.fill_between(fpr, tpr, alpha=0.1, color=COLORS["real"])
        style_axes(ax3, "scatter", title="Q3: Real vs Generated Discriminator",
                   xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax3.set_xlim(-0.02, 1.02)
        ax3.set_ylim(-0.02, 1.02)
        ax3.set_aspect("equal")
        ax3.legend(fontsize=9, loc="lower right")

        # Interpretation text
        if disc_auc < 0.6:
            interp = "Near-random → distributions well-matched"
            color = COLORS["good"]
        elif disc_auc < 0.75:
            interp = "Mild separability → some distributional gap"
            color = COLORS["warn"]
        else:
            interp = "Easily separable → significant gap"
            color = COLORS["bad"]
        ax3.text(0.05, 0.95, interp, transform=ax3.transAxes,
                 fontsize=8, color=color, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8))
    else:
        ax3.text(0.5, 0.5, "No discriminator data", ha="center", va="center",
                 transform=ax3.transAxes)
        ax3.set_title("Q3: Discriminator ROC")

    if save:
        path = save_panel(fig, output_dir / "panel_q_classifier_alignment.png", dpi)
        logger.info(f"Saved Panel Q → {path}")
    return fig


# ──────────────────────────────────────────────────────────────
# PANEL R: DE & Pathway Concordance
# ──────────────────────────────────────────────────────────────

def plot_de_concordance_panel(
    de_data: Dict,
    output_dir: Path,
    dpi: int = 300,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Panel R: Differential expression concordance between real and generated.

    R1: logFC scatter (real vs gen) for the first contrast, density-coloured
    R2: Top-gene overlap heatmap across all contrasts
    R3: Summary bars (Pearson r, Jaccard, sign agreement per contrast)
    """
    if not de_data:
        logger.info("No DE data — skipping Panel R")
        return None

    contrasts = list(de_data.keys())
    n_contrasts = len(contrasts)

    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.8, 1.0], wspace=0.3)
    fig.suptitle("Downstream: DE Concordance — Real vs Generated",
                 fontsize=14, fontweight="bold")

    # ── R1: logFC scatter for first contrast ──
    ax = fig.add_subplot(gs[0])
    first_key = contrasts[0]
    first = de_data[first_key]
    real_logfc = np.array(first.get("_real_logfc", []))
    gen_logfc = np.array(first.get("_gen_logfc", []))
    shared_genes = first.get("_shared_genes", [])

    if len(real_logfc) > 0 and len(gen_logfc) > 0:
        # Density colouring
        from scipy.stats import gaussian_kde
        try:
            xy = np.vstack([real_logfc, gen_logfc])
            kde = gaussian_kde(xy)
            density = kde(xy)
        except Exception:
            density = np.ones_like(real_logfc)

        sc = ax.scatter(real_logfc, gen_logfc, c=density, cmap="viridis",
                        s=8, alpha=0.7, edgecolors="none")
        # Identity line
        lo = min(real_logfc.min(), gen_logfc.min()) * 1.1
        hi = max(real_logfc.max(), gen_logfc.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], color="#E53935", linestyle="--", lw=1.5,
                alpha=0.7, label="y = x")
        ax.fill_between([lo, hi], [lo - 0.5, hi - 0.5], [lo + 0.5, hi + 0.5],
                        alpha=0.05, color="#4CAF50")

        # Label top outliers
        residuals = np.abs(gen_logfc - real_logfc)
        top_idx = np.argsort(residuals)[-min(8, len(residuals)):]
        for i in top_idx:
            if i < len(shared_genes):
                ax.annotate(shared_genes[i], (real_logfc[i], gen_logfc[i]),
                            fontsize=5.5, xytext=(4, 4), textcoords="offset points",
                            arrowprops=dict(arrowstyle="->", lw=0.5, color="#555"),
                            color="#333", fontweight="bold")

        r_val = first.get("logfc_pearson_r", 0)
        ax.legend(fontsize=8, title=f"Pearson r = {r_val:.3f}")
        fig.colorbar(sc, ax=ax, shrink=0.7, label="Density")

    contrast_display = first_key.replace("_", " ")[:40]
    style_axes(ax, "scatter", title=f"R1: logFC Scatter — {contrast_display}",
               xlabel="Real logFC", ylabel="Generated logFC")

    # ── R2: Summary heatmap across contrasts ──
    ax2 = fig.add_subplot(gs[1])
    metric_names = ["Pearson r", "Spearman ρ", "Jaccard", "Sign Agree"]
    metric_keys = ["logfc_pearson_r", "logfc_spearman_rho", "top_k_jaccard", "top_k_sign_agreement"]
    heatmap_data = np.zeros((n_contrasts, len(metric_names)))
    contrast_labels = []
    for i, cname in enumerate(contrasts):
        cd = de_data[cname]
        for j, mk in enumerate(metric_keys):
            heatmap_data[i, j] = cd.get(mk, 0)
        # Short contrast name
        parts = cname.split("_vs_")
        short = f"{parts[0][:12]}…\nvs {parts[1][:12]}…" if len(parts) == 2 else cname[:20]
        contrast_labels.append(short)

    im = ax2.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(len(metric_names)))
    ax2.set_xticklabels(metric_names, fontsize=8, rotation=30, ha="right")
    ax2.set_yticks(range(n_contrasts))
    ax2.set_yticklabels(contrast_labels, fontsize=7)

    # Annotate cells
    for i in range(n_contrasts):
        for j in range(len(metric_names)):
            val = heatmap_data[i, j]
            color = "white" if val > 0.7 or val < 0.3 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=8, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax2, shrink=0.6)
    style_axes(ax2, "heatmap", title="R2: Concordance Across Contrasts")

    # ── R3: Grouped bar chart per contrast ──
    ax3 = fig.add_subplot(gs[2])
    x = np.arange(n_contrasts)
    n_metrics = len(metric_names)
    w = 0.8 / n_metrics
    bar_colors = ["#1976D2", "#4CAF50", "#FF9800", "#9C27B0"]

    for j, (mname, mk) in enumerate(zip(metric_names, metric_keys)):
        vals = [de_data[c].get(mk, 0) for c in contrasts]
        offset = (j - n_metrics / 2 + 0.5) * w
        ax3.bar(x + offset, vals, w, label=mname, color=bar_colors[j],
                alpha=0.85, edgecolor="white")

    short_xlabels = []
    for cname in contrasts:
        parts = cname.split("_vs_")
        short_xlabels.append(f"{parts[0][:15]}…" if len(parts) == 2 else cname[:20])

    ax3.set_xticks(x)
    ax3.set_xticklabels(short_xlabels, fontsize=7, rotation=25, ha="right")
    ax3.set_ylim(0, 1.1)
    ax3.legend(fontsize=7, ncol=2, loc="upper right")
    style_axes(ax3, "bar", title="R3: Per-Contrast Summary", ylabel="Score")

    if save:
        path = save_panel(fig, output_dir / "panel_r_de_concordance.png", dpi)
        logger.info(f"Saved Panel R → {path}")
    return fig


# ──────────────────────────────────────────────────────────────
# Convenience: generate all downstream panels from JSON files
# ──────────────────────────────────────────────────────────────

def generate_downstream_panels(
    downstream_dir: str = "results/downstream",
    output_dir: str = "results/figures",
    type_names: Optional[Dict[int, str]] = None,
    dpi: int = 300,
) -> List[Path]:
    """Load pre-computed downstream JSONs + internal arrays and generate P/Q/R."""
    ds_dir = Path(downstream_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Panel P
    clust_path = ds_dir / "clustering_alignment.json"
    if clust_path.exists():
        with open(clust_path) as f:
            clust_data = json.load(f)
        # Load internal arrays (saved by downstream_biology.py)
        umap_path = ds_dir / "clustering_umap_coords.npy"
        source_path = ds_dir / "clustering_source.npy"
        ct_path = ds_dir / "clustering_cell_type.npy"
        if all(p.exists() for p in [umap_path, source_path, ct_path]):
            clust_data["_umap_coords"] = np.load(umap_path)
            clust_data["_source"] = np.load(source_path, allow_pickle=True)
            clust_data["_cell_type"] = np.load(ct_path, allow_pickle=True)
        fig_p = plot_clustering_panel(clust_data, type_names or {}, out_dir, dpi)
        if fig_p:
            saved.append(out_dir / "panel_p_clustering_mixing.png")
            plt.close(fig_p)

    # Panel Q
    classif_path = ds_dir / "classifier_alignment.json"
    if classif_path.exists():
        with open(classif_path) as f:
            classif_data = json.load(f)
        # The JSON now contains confusion_matrix and disc_proba/disc_y directly
        if "confusion_matrix" in classif_data:
            classif_data["_confusion_matrix"] = classif_data["confusion_matrix"]
        if "disc_proba" in classif_data:
            classif_data["_disc_proba"] = classif_data["disc_proba"]
            classif_data["_disc_y"] = classif_data["disc_y"]
        fig_q = plot_classifier_panel(classif_data, out_dir, dpi)
        if fig_q:
            saved.append(out_dir / "panel_q_classifier_alignment.png")
            plt.close(fig_q)

    # Panel R
    de_path = ds_dir / "de_concordance.json"
    if de_path.exists():
        with open(de_path) as f:
            de_data = json.load(f)
        # Load internal arrays
        for contrast_name in de_data:
            for key in ["_real_logfc", "_gen_logfc", "_shared_genes"]:
                fname = f"de_{contrast_name}_{key.lstrip('_')}.npy"
                arr_path = ds_dir / fname
                if arr_path.exists():
                    arr = np.load(arr_path, allow_pickle=True)
                    de_data[contrast_name][key] = arr.tolist()
        fig_r = plot_de_concordance_panel(de_data, out_dir, dpi)
        if fig_r:
            saved.append(out_dir / "panel_r_de_concordance.png")
            plt.close(fig_r)

    return saved
