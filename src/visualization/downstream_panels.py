"""
downstream_panels.py — Facade for Panels P, Q, R and merged P+Q figure.

Implementation lives in:
  - panels_clustering: plot_clustering_panel (P)
  - panels_classifier: plot_classifier_panel (Q), _plot_classifier_metric_heatmap
  - panels_de_concordance: plot_de_concordance_panel (R)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .panels_classifier import _plot_classifier_metric_heatmap, plot_classifier_panel
from .panels_clustering import plot_clustering_panel
from .panels_de_concordance import plot_de_concordance_panel
from .style import (
    COLORS,
    TYPE_PALETTE,
    add_colorbar_safe,
    apply_style,
    quality_color,
    save_panel,
    set_dense_tick_labels,
    set_figure_suptitle,
    style_axes,
)
from src.utils.paths import RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)


def plot_clustering_and_classifier_merged(
    clustering_data: Dict,
    classifier_data: Dict,
    type_names: Dict[int, str],
    output_dir: Path,
    dpi: int = 300,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Merged figure: clustering + classifier alignment (former P + Q)."""
    apply_style()

    fig = plt.figure(figsize=(15.2, 10.6))
    gs = fig.add_gridspec(2, 3, wspace=0.50, hspace=0.50,
                          height_ratios=[1, 1.05],
                          width_ratios=[1.5, 1.2, 1.0])
    set_figure_suptitle(fig, "Downstream Validation: Clustering & Classifier Alignment", fontsize=11)

    umap_coords = clustering_data.get("_umap_coords")
    source = clustering_data.get("_source")
    cell_type = clustering_data.get("_cell_type")

    ax_p1 = fig.add_subplot(gs[0, 0])
    if umap_coords is not None:
        umap_coords = np.asarray(umap_coords)
        source = np.asarray(source)
        cell_type = np.asarray(cell_type)
        unique_types = np.unique(cell_type)
        ct_colors = {ct: TYPE_PALETTE[i % len(TYPE_PALETTE)]
                     for i, ct in enumerate(sorted(unique_types))}
        real_mask = source == "real"
        gen_mask = source == "generated"
        for ct in unique_types:
            m = real_mask & (cell_type == ct)
            if m.any():
                ax_p1.scatter(umap_coords[m, 0], umap_coords[m, 1],
                              c=[ct_colors[ct]], s=4, alpha=0.3, rasterized=True)
        for ct in unique_types:
            m = gen_mask & (cell_type == ct)
            if m.any():
                ax_p1.scatter(umap_coords[m, 0], umap_coords[m, 1],
                              c=[ct_colors[ct]], s=12, alpha=0.6, marker="^",
                              edgecolors="black", linewidths=0.3, rasterized=True)
        ax_p1.scatter([], [], c=COLORS["real"], s=15, marker="o", label="Real")
        ax_p1.scatter([], [], c=COLORS["generated"], s=15, marker="^", edgecolors="black",
                      linewidths=0.3, label="Generated")
        ax_p1.legend(
            fontsize=8,
            loc="lower left",
            borderaxespad=0.4,
            markerscale=2,
            frameon=False,
        )
        style_axes(ax_p1, "umap", title="UMAP Overlay",
                   xlabel="UMAP 1", ylabel="UMAP 2")
    else:
        ax_p1.text(0.5, 0.5, "No UMAP data", ha="center", va="center",
                   transform=ax_p1.transAxes)

    ax_p2 = fig.add_subplot(gs[0, 1])
    mixing = clustering_data.get("per_type_mixing", {})
    if mixing:
        sorted_types = sorted(mixing.keys(), key=lambda k: mixing[k])
        vals = [mixing[t] for t in sorted_types]
        short_names = [t[:20] for t in sorted_types]
        bar_colors = [quality_color(v, (0.3, 0.15)) for v in vals]
        y_pos = np.arange(len(sorted_types))
        ax_p2.barh(y_pos, vals, color=bar_colors, height=0.7,
                   edgecolor="white", linewidth=0.5)
        ax_p2.set_yticks(y_pos)
        ax_p2.set_yticklabels(short_names, fontsize=7)
        set_dense_tick_labels(ax_p2, axis="y", max_labels=12, fontsize=7, rotation=0)
        ax_p2.axvline(x=clustering_data.get("mean_mixing_score", 0),
                       color=COLORS["bad"], linestyle="--", alpha=0.7, linewidth=1.5,
                       label=f"mean={clustering_data.get('mean_mixing_score', 0):.3f}")
        ax_p2.set_xlim(0, max(max(vals) * 1.1, 0.5))
        ax_p2.legend(fontsize=8, frameon=False, loc="lower right")
        style_axes(ax_p2, "bar", title="kNN Mixing",
                   xlabel="Fraction Real Neighbours")
    else:
        ax_p2.text(0.5, 0.5, "No mixing data", ha="center", va="center",
                   transform=ax_p2.transAxes)

    ax_ps = fig.add_subplot(gs[0, 2])
    summary_items = [
        ("ARI", clustering_data.get("ari_gt_vs_leiden", 0), (0.7, 0.4)),
        ("NMI", clustering_data.get("nmi_gt_vs_leiden", 0), (0.7, 0.4)),
        ("Purity", clustering_data.get("mean_cluster_purity", 0), (0.8, 0.5)),
        ("Mix", clustering_data.get("mean_mixing_score", 0), (0.6, 0.3)),
    ]
    metric_names = [s[0] for s in summary_items]
    metric_vals = [s[1] for s in summary_items]
    thresholds = [s[2] for s in summary_items]
    y_pos = np.arange(len(summary_items))
    # Background reference bars
    ax_ps.barh(y_pos, [1.0] * len(summary_items), height=0.6,
               color="#E0E0E0", edgecolor="none", zorder=1)
    # Foreground quality-colored bars
    bar_colors = [quality_color(v, t) for v, t in zip(metric_vals, thresholds)]
    ax_ps.barh(y_pos, metric_vals, height=0.6, color=bar_colors,
               edgecolor="white", linewidth=0.8, zorder=2)
    for i, val in enumerate(metric_vals):
        ax_ps.text(min(val + 0.03, 0.98), i, f"{val:.3f}",
                   va="center", fontsize=8, zorder=3)
    ax_ps.set_yticks(y_pos)
    ax_ps.set_yticklabels(metric_names, fontsize=9)
    ax_ps.set_xlim(0, 1.15)
    ax_ps.invert_yaxis()
    n_cl = clustering_data.get("n_leiden_clusters", "?")
    ax_ps.set_xlabel(f"Leiden clusters: {n_cl}", fontsize=8)
    style_axes(ax_ps, "bar", title="Cluster Metrics")

    cm = classifier_data.get("_confusion_matrix")
    class_names = classifier_data.get("class_names", [])
    per_type_acc = classifier_data.get("per_type_accuracy", {})
    gen_acc = classifier_data.get("gen_accuracy", 0)
    gen_f1 = classifier_data.get("gen_f1", 0)
    disc_auc = classifier_data.get("discriminator_auc", 0)

    ax_q1 = fig.add_subplot(gs[1, 0])
    if cm is not None:
        cm = np.array(cm)
        n_classes = len(class_names) if class_names else cm.shape[0]
        if not class_names:
            class_names = [f"C{i}" for i in range(n_classes)]
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        im = ax_q1.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        if n_classes > 30:
            tick_step = max(1, n_classes // 7)
            ticks = list(range(0, n_classes, tick_step))
            ax_q1.set_xticks(ticks)
            ax_q1.set_yticks(ticks)
            ax_q1.set_xticklabels([str(i) for i in ticks], fontsize=8)
            ax_q1.set_yticklabels([str(i) for i in ticks], fontsize=8)
        else:
            short = [n[:18] for n in class_names]
            ax_q1.set_xticks(range(n_classes))
            ax_q1.set_yticks(range(n_classes))
            ax_q1.set_xticklabels(short, rotation=90, fontsize=8, ha="center")
            ax_q1.set_yticklabels(short, fontsize=8, ha="right")
        add_colorbar_safe(im, ax=ax_q1, label="Recall",
                         shrink=0.50, pad=0.10, orientation="horizontal", aspect=20)
        style_axes(ax_q1, "heatmap", title="Confusion Matrix",
                   xlabel="Predicted", ylabel="True Type")
    else:
        ax_q1.text(0.5, 0.5, "No confusion matrix", ha="center", va="center",
                   transform=ax_q1.transAxes)

    ax_q2 = fig.add_subplot(gs[1, 1])
    if per_type_acc and cm is not None:
        summary = _plot_classifier_metric_heatmap(
            fig,
            ax_q2,
            np.array(cm),
            class_names or [f"C{i}" for i in range(np.array(cm).shape[0])],
            max_rows=20,
        )
        ax_q2.text(
            0.96,
            0.96,
            f"Acc {gen_acc:.3f}  |  Med F1 {np.median(summary['f1']):.3f}",
            transform=ax_q2.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=COLORS["neutral"],
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#DDDDDD", alpha=0.92),
        )
    else:
        ax_q2.text(0.5, 0.5, "No per-type data", ha="center", va="center",
                   transform=ax_q2.transAxes)

    ax_q3 = fig.add_subplot(gs[1, 2])
    disc_proba = classifier_data.get("_disc_proba")
    disc_y = classifier_data.get("_disc_y")
    if disc_proba is not None and disc_y is not None:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(disc_y, disc_proba)
        ax_q3.plot(fpr, tpr, color=COLORS["real"], linewidth=2,
                   label=f"AUC = {disc_auc:.3f}")
        ax_q3.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.6,
                   label="Random (0.5)")
        ax_q3.fill_between(fpr, tpr, alpha=0.1, color=COLORS["real"])
        ax_q3.set_xlim(-0.02, 1.02)
        ax_q3.set_ylim(-0.02, 1.02)
        ax_q3.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_q3.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_q3.set_aspect("equal")
        ax_q3.legend(fontsize=8, loc="lower right", frameon=False)
        style_axes(ax_q3, "scatter", title="Discriminator ROC",
                   xlabel="FPR", ylabel="TPR")
    else:
        ax_q3.text(0.5, 0.5, "No discriminator data", ha="center", va="center",
                   transform=ax_q3.transAxes)

    if save:
        path = save_panel(
            fig,
            output_dir / "fig_downstream_pq.png",
            dpi,
            layout_rect=(0.02, 0.04, 0.98, 0.96),
        )
        logger.info(f"Saved merged P+Q → {path}")
    return fig


def generate_downstream_panels(
    downstream_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    type_names: Optional[Dict[int, str]] = None,
    dpi: int = 300,
) -> List[Path]:
    """Load pre-computed downstream JSONs + internal arrays and generate P/Q/R."""
    downstream_dir = downstream_dir or str(RESULTS_DIR / "downstream")
    output_dir = output_dir or str(FIG_DIR)
    ds_dir = Path(downstream_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    clust_path = ds_dir / "clustering_alignment.json"
    if clust_path.exists():
        with open(clust_path) as f:
            clust_data = json.load(f)
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

    classif_path = ds_dir / "classifier_alignment.json"
    if classif_path.exists():
        with open(classif_path) as f:
            classif_data = json.load(f)
        if "confusion_matrix" in classif_data:
            classif_data["_confusion_matrix"] = classif_data["confusion_matrix"]
        if "disc_proba" in classif_data:
            classif_data["_disc_proba"] = classif_data["disc_proba"]
            classif_data["_disc_y"] = classif_data["disc_y"]
        fig_q = plot_classifier_panel(classif_data, out_dir, dpi)
        if fig_q:
            saved.append(out_dir / "panel_q_classifier_alignment.png")
            plt.close(fig_q)

    de_path = ds_dir / "de_concordance.json"
    if de_path.exists():
        with open(de_path) as f:
            de_data = json.load(f)
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


__all__ = [
    "plot_clustering_panel",
    "plot_classifier_panel",
    "plot_de_concordance_panel",
    "plot_clustering_and_classifier_merged",
    "generate_downstream_panels",
]
