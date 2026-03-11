"""
fig17_downstream.py — Article Figure 17: Downstream validation panels.

Consolidated module containing:
  - plot_clustering_and_classifier_merged  (merged P+Q for Article Fig 17)
  - plot_clustering_panel                  (standalone Panel P)
  - plot_classifier_panel                  (standalone Panel Q)
  - _compute_classifier_summary            (helper for Panel Q)
  - _plot_classifier_metric_heatmap        (helper for Panel Q)
  - generate_downstream_panels             (entry-point that runs P, Q, R)
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .direct_layout import bind_figure_region
from .explicit_positioning import add_axes_next_to, add_shared_legend_axes
from .style import (
    COLORS,
    FONT_DENSE_YTICK,
    FONT_HEATMAP_CELL,
    FONT_LEGEND,
    TYPE_PALETTE,
    abbreviate_cell_type,
    add_colorbar_safe,
    add_panel_label,
    apply_style,
    quality_color,
    save_panel,
    set_dense_tick_labels,
    set_figure_suptitle,
    style_axes,
)
from ._plot_helpers import plot_umap_overlay, plot_confusion_matrix, plot_roc_curve
from src.utils.paths import RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (from panels_classifier)
# ---------------------------------------------------------------------------

def _compute_classifier_summary(cm: np.ndarray, class_names: List[str]) -> Dict[str, np.ndarray]:
    """Compute per-type classifier summary metrics from a confusion matrix."""
    cm = np.asarray(cm, dtype=float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    true_positive = np.diag(cm)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros_like(true_positive), where=support > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    order = np.argsort(f1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "predicted": predicted,
        "order": order,
        "ordered_names": np.array(class_names, dtype=object)[order],
    }


def _plot_classifier_metric_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    cm: np.ndarray,
    class_names: List[str],
    max_rows: Optional[int] = 18,
    title: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Render a per-type precision/recall/F1 heatmap sorted by hardest classes."""
    summary = _compute_classifier_summary(cm, class_names)
    order = summary["order"]
    ordered_names = summary["ordered_names"]
    metric_matrix = np.column_stack(
        [
            summary["precision"][order],
            summary["recall"][order],
            summary["f1"][order],
        ]
    )
    display_names = ordered_names
    display_matrix = metric_matrix
    rows_truncated = False
    display_mode = "all"
    if max_rows is not None and max_rows > 0 and len(display_names) > max_rows:
        n_worst = max_rows // 2
        n_best = max_rows - n_worst
        idx = list(range(n_worst)) + list(range(len(ordered_names) - n_best, len(ordered_names)))
        display_names = ordered_names[idx]
        display_matrix = metric_matrix[idx]
        rows_truncated = True
        display_mode = "worst+best"

    im = ax.imshow(display_matrix, cmap="inferno", aspect="auto", vmin=0, vmax=1)
    # Add numeric annotations on heatmap cells
    for _ri in range(display_matrix.shape[0]):
        for _ci in range(display_matrix.shape[1]):
            _val = display_matrix[_ri, _ci]
            _color = "white" if _val < 0.5 else "black"
            ax.text(_ci, _ri, f"{_val:.2f}", ha="center", va="center",
                    fontsize=max(FONT_HEATMAP_CELL - 1, 6), color=_color, fontweight="normal")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Prec.", "Rec.", "F1"], fontsize=8)
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(
        [
            textwrap.shorten(str(name).replace("_", " "), width=18, placeholder="…")
            for name in display_names
        ],
        fontsize=7,
    )
    default_title = "Per-Type P/R/F1"
    if rows_truncated:
        if display_mode == "worst+best":
            default_title += f" (worst+best {len(display_names)})"
        else:
            default_title += f" (top {len(display_names)})"
    ax.set_title(title or default_title, fontsize=10)
    ax.set_ylabel("")
    # ax.set_xlabel("Metric")  # Removed to reduce label density

    cax = add_axes_next_to(
        fig,
        ax,
        side="right",
        width=0.008,
        height=ax.get_position().height * 0.48,
        pad=0.010,
        align="bottom",
        y_offset=0.01,
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Score", fontsize=10)
    cbar.ax.tick_params(labelsize=7)
    return summary


# ---------------------------------------------------------------------------
# Panel P: Clustering alignment (from panels_clustering)
# ---------------------------------------------------------------------------

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

    fig = plt.figure(figsize=(13.0, 5.5))
    ax_rect_1, ax_rect_2, ax_rect_3 = bind_figure_region(fig, (0.05, 0.10, 0.98, 0.93)).split_cols(
        [1.10, 1.35, 0.82],
        gap=[0.028, 0.018],
    )
    # suptitle removed per revision; title information moved to LaTeX caption

    ax = ax_rect_1.add_axes(fig)
    add_panel_label(ax, 'a', x=-0.12, y=1.08)
    plot_umap_overlay(ax, umap_coords, source, cell_type,
                      legend_loc="upper left", legend_fontsize=9)

    ax2 = ax_rect_2.add_axes(fig)
    add_panel_label(ax2, 'b', x=-0.12, y=1.08)
    mixing = clustering_data.get("per_type_mixing", {})
    if mixing:
        sorted_types = sorted(mixing.keys(), key=lambda k: mixing[k])
        vals = [mixing[t] for t in sorted_types]
        short_names = [t[:25] for t in sorted_types]
        bar_colors = [quality_color(v, (0.3, 0.15)) for v in vals]

        y_pos = np.arange(len(sorted_types))
        ax2.barh(y_pos, vals, color=bar_colors, height=0.7, edgecolor="white", linewidth=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(short_names, fontsize=8)
        set_dense_tick_labels(ax2, axis="y", max_labels=18, fontsize=8, rotation=0)
        ax2.axvline(x=clustering_data.get("mean_mixing_score", 0),
                     color=COLORS["error_red"], linestyle="--", alpha=0.7, linewidth=1.5,
                     label=f"mean={clustering_data.get('mean_mixing_score', 0):.3f}")
        ax2.set_xlim(0, max(max(vals) * 1.1, 0.5))
        ax2.legend(fontsize=9, loc="upper left", frameon=False)
        style_axes(ax2, "bar", title="kNN Mixing",
                   xlabel="Fraction Real Neighbours")
    else:
        ax2.text(0.5, 0.5, "No mixing data", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=10)
        ax2.set_title("kNN Mixing Score")

    ax3 = ax_rect_3.add_axes(fig)
    add_panel_label(ax3, 'c', x=-0.12, y=1.08)
    ax3.axis("off")

    gauge_items = [
        ("ARI", clustering_data.get("ari_gt_vs_leiden", 0), (0.6, 0.3)),
        ("NMI", clustering_data.get("nmi_gt_vs_leiden", 0), (0.6, 0.3)),
        ("C.Purity", clustering_data.get("mean_cluster_purity", 0), (0.8, 0.5)),
        ("Mean Mix", clustering_data.get("mean_mixing_score", 0), (0.3, 0.15)),
    ]

    n_items = len(gauge_items)
    for i, (label, val, thresh) in enumerate(gauge_items):
        y = 0.90 - i * (0.82 / max(n_items - 1, 1))
        color = quality_color(val, thresh)
        ax3.text(0.55, y, f"{val:.3f}", fontsize=9,
                 color=COLORS["annotation_light"], ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=color,
                           alpha=1.0, linewidth=2.0),
                 transform=ax3.transAxes)
        ax3.text(0.1, y, label, fontsize=9, ha="left", va="center",
                 transform=ax3.transAxes, color="#333")

    ax3.set_title("Alignment Gauges", fontsize=11, pad=8)
    n_clusters = clustering_data.get("n_leiden_clusters", "?")
    ax3.text(0.5, 0.02, f"Leiden clusters: {n_clusters}",
             transform=ax3.transAxes, ha="center", fontsize=8, color="#666")

    if save:
        path = save_panel(fig, output_dir / "fig17_clustering_mixing.png", dpi)
        logger.info(f"Saved Panel P → {path}")
    return fig


# ---------------------------------------------------------------------------
# Panel Q: Classifier alignment (from panels_classifier)
# ---------------------------------------------------------------------------

def plot_classifier_panel(
    classifier_data: Dict,
    output_dir: Path,
    dpi: int = 300,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Panel Q: Classifier alignment between real-trained and generated.

    Q1: Confusion matrix (real-trained classifier on generated cells)
    Q2: Per-type precision/recall/F1 heatmap
    Q3: Discriminator ROC curve
    """
    cm = classifier_data.get("_confusion_matrix")
    if cm is None:
        logger.info("No classifier data — skipping Panel Q")
        return None

    cm = np.array(cm)
    class_names = classifier_data.get("class_names", [f"C{i}" for i in range(cm.shape[0])])
    per_type_acc = classifier_data.get("per_type_accuracy", {})

    fig = plt.figure(figsize=(13.0, 6.0))
    ax_rect_1, ax_rect_2, ax_rect_3 = bind_figure_region(fig, (0.05, 0.10, 0.98, 0.93)).split_cols(
        [1.10, 1.35, 0.82],
        gap=[0.028, 0.018],
    )
    gen_acc = classifier_data.get("gen_accuracy", 0)
    gen_f1 = classifier_data.get("gen_f1", 0)
    disc_auc = classifier_data.get("discriminator_auc", 0)
    # suptitle and stats banner removed per revision; title information moved to LaTeX caption

    ax = ax_rect_1.add_axes(fig)
    add_panel_label(ax, 'a', x=-0.12, y=1.08)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    n_classes = len(class_names)

    if n_classes > 30:
        tick_step = max(1, n_classes // 7)
        tick_positions = list(range(0, n_classes, tick_step))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([str(i) for i in tick_positions], rotation=0, fontsize=8)
        ax.set_yticklabels([str(i) for i in tick_positions], fontsize=8)
    else:
        short_names = [n[:18] for n in class_names]
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(short_names, rotation=90, fontsize=8, ha="center")
        ax.set_yticklabels(short_names, fontsize=8, ha="right")
        set_dense_tick_labels(ax, axis="both", max_labels=20, fontsize=8,
                              rotation=90, ha="center")

    if n_classes <= 30:
        for i in range(min(n_classes, cm_norm.shape[0])):
            val = cm_norm[i, i]
            color = "white" if val > 0.5 else "black"
            ax.text(i + 0.25, i, f"{val:.2f}", ha="left", va="center",
                    fontsize=8, color=color)
    else:
        diag_step = max(1, n_classes // 8)
        safe_end = max(0, n_classes - diag_step)
        for i in range(0, min(safe_end, cm_norm.shape[0]), diag_step):
            val = cm_norm[i, i]
            color = "white" if val > 0.5 else "black"
            ax.text(i, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=FONT_HEATMAP_CELL, color=color)

    add_colorbar_safe(im, ax=ax, shrink=0.5, pad=0.10, label="Recall",
                      orientation="horizontal", aspect=20)
    style_axes(ax, "heatmap", title="Confusion Matrix (on Generated Cells)",
               xlabel="Predicted", ylabel="True Type")

    ax2 = ax_rect_2.add_axes(fig)
    add_panel_label(ax2, 'b', x=-0.12, y=1.08)
    if per_type_acc:
        summary = _plot_classifier_metric_heatmap(fig, ax2, cm, class_names)
        f1 = summary["f1"]
        support = summary["support"]
        ax2.text(
            0.0,
            0.99,
            f"Median F1 = {np.median(f1):.3f}  |  Support median = {np.median(support):.0f}",
            transform=ax2.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#444",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=COLORS["border_light"], alpha=0.92),
        )
    else:
        ax2.text(0.5, 0.5, "No per-type data", ha="center", va="center",
                 transform=ax2.transAxes)
        ax2.set_title("Per-Type Summary")

    ax3 = ax_rect_3.add_axes(fig)
    add_panel_label(ax3, 'c', x=-0.12, y=1.08)
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
        style_axes(ax3, "scatter", title="Real vs Generated Discriminator",
                   xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax3.set_xlim(-0.02, 1.02)
        ax3.set_ylim(-0.02, 1.02)
        ax3.set_aspect("equal")
        ax3.legend(fontsize=FONT_LEGEND, loc="lower left", frameon=False)
        from matplotlib.ticker import MaxNLocator
        ax3.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax3.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

        if disc_auc < 0.6:
            interp = "Near-random: well-matched"
            color = COLORS["good"]
        elif disc_auc < 0.75:
            interp = "Mild separability"
            color = COLORS["warn"]
        else:
            interp = "Easily separable"
            color = COLORS["bad"]
        ax3.text(0.05, 0.95, interp, transform=ax3.transAxes,
                 fontsize=8, color=COLORS["neutral"], va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color,
                           alpha=0.9, linewidth=1.5))
    else:
        ax3.text(0.5, 0.5, "No discriminator data", ha="center", va="center",
                 transform=ax3.transAxes)
        ax3.set_title("Discriminator ROC")

    if save:
        path = save_panel(fig, output_dir / "fig17_classifier_alignment.png", dpi)
        logger.info(f"Saved Panel Q → {path}")
    return fig


# ---------------------------------------------------------------------------
# Merged P+Q figure (from downstream_panels)
# ---------------------------------------------------------------------------

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

    fig = plt.figure(figsize=(16.0, 10.2))
    layout = bind_figure_region(fig, (0.03, 0.10, 0.98, 0.96))
    top_row, bottom_row = layout.split_rows([1, 1.05], hspace=0.42)
    top_rects = top_row.split_cols([1.10, 1.34, 0.82], gap=[0.060, 0.050])
    bottom_rects = bottom_row.split_cols([1.10, 1.34, 0.82], gap=[0.060, 0.050])
    # Title moved to LaTeX caption
    # set_figure_suptitle(fig, "Downstream Validation: Clustering & Classifier Alignment", fontsize=11)

    umap_coords = clustering_data.get("_umap_coords")
    source = clustering_data.get("_source")
    cell_type = clustering_data.get("_cell_type")

    ax_p1 = top_rects[0].add_axes(fig)
    add_panel_label(ax_p1, 'a', x=-0.10, y=1.00)
    if umap_coords is not None:
        umap_coords = np.asarray(umap_coords)
        source = np.asarray(source)
        cell_type = np.asarray(cell_type)
        plot_umap_overlay(ax_p1, umap_coords, source, cell_type,
                          legend_loc="lower left", legend_fontsize=9)
    else:
        ax_p1.text(0.5, 0.5, "No UMAP data", ha="center", va="center",
                   transform=ax_p1.transAxes)

    ax_p2 = top_rects[1].inset(left=0.090, right=0.040).add_axes(fig)
    add_panel_label(ax_p2, 'b', x=-0.12, y=1.05)
    mixing = clustering_data.get("per_type_mixing", {})
    if mixing:
        sorted_types = sorted(mixing.keys(), key=lambda k: mixing[k])
        vals = [mixing[t] for t in sorted_types]
        short_names = [abbreviate_cell_type(t, 20) for t in sorted_types]
        bar_colors = [quality_color(v, (0.3, 0.15)) for v in vals]
        y_pos = np.arange(len(sorted_types))
        ax_p2.barh(y_pos, vals, color=bar_colors, height=0.7,
                   edgecolor="white", linewidth=0.5)
        ax_p2.set_yticks(y_pos)
        ax_p2.set_yticklabels(short_names, fontsize=FONT_DENSE_YTICK)
        set_dense_tick_labels(ax_p2, axis="y", max_labels=12, fontsize=FONT_DENSE_YTICK, rotation=0)
        ax_p2.axvline(x=clustering_data.get("mean_mixing_score", 0),
                       color=COLORS["bad"], linestyle="--", alpha=0.7, linewidth=1.5,
                       label=f"mean={clustering_data.get('mean_mixing_score', 0):.3f}")
        ax_p2.set_xlim(0, max(max(vals) * 1.1, 0.5))
        ax_p2.legend(fontsize=9, frameon=False, loc="lower right")
        style_axes(ax_p2, "bar", title="Sorted kNN Mixing Score",
                   xlabel="Fraction Real Neighbours")
    else:
        ax_p2.text(0.5, 0.5, "No mixing data", ha="center", va="center",
                   transform=ax_p2.transAxes)

    ax_ps = top_rects[2].add_axes(fig)
    add_panel_label(ax_ps, 'c', x=-0.12, y=1.05)
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
               color=COLORS["bg_gauge"], edgecolor="none", zorder=1)
    # Foreground quality-colored bars
    bar_colors = [quality_color(v, t) for v, t in zip(metric_vals, thresholds)]
    ax_ps.barh(y_pos, metric_vals, height=0.6, color=bar_colors,
               edgecolor="white", linewidth=0.8, zorder=2)
    for i, val in enumerate(metric_vals):
        ax_ps.text(min(val + 0.03, 0.98), i, f"{val:.3f}",
                   va="center", fontsize=9, zorder=3)
    ax_ps.set_yticks(y_pos)
    ax_ps.set_yticklabels(metric_names, fontsize=10)
    ax_ps.set_xlim(0, 1.15)
    ax_ps.invert_yaxis()
    n_cl = clustering_data.get("n_leiden_clusters", "?")
    ax_ps.set_xlabel(f"Leiden clusters: {n_cl}", fontsize=9)
    ax_ps.tick_params(axis="x", labelbottom=False, bottom=False)
    style_axes(ax_ps, "bar", title="Clustering Metrics")

    cm = classifier_data.get("_confusion_matrix")
    class_names = classifier_data.get("class_names", [])
    per_type_acc = classifier_data.get("per_type_accuracy", {})
    gen_acc = classifier_data.get("gen_accuracy", 0)
    gen_f1 = classifier_data.get("gen_f1", 0)
    disc_auc = classifier_data.get("discriminator_auc", 0)

    ax_q1 = bottom_rects[0].add_axes(fig)
    add_panel_label(ax_q1, 'd', x=-0.08, y=1.05)
    if cm is not None:
        cm = np.array(cm)
        plot_confusion_matrix(ax_q1, cm, class_names or None,
                              title="Confusion Matrix")
        if cm.shape[0] > 30:
            tick_step = max(4, cm.shape[0] // 3)
            tick_positions = list(range(0, cm.shape[0], tick_step))
            ax_q1.set_xticks(tick_positions)
            ax_q1.set_yticks(tick_positions)
            ax_q1.set_xticklabels([str(i) for i in tick_positions], fontsize=7, rotation=45, ha="right")
            ax_q1.set_yticklabels([str(i) for i in tick_positions], fontsize=7)
    else:
        ax_q1.text(0.5, 0.5, "No confusion matrix", ha="center", va="center",
                   transform=ax_q1.transAxes)

    ax_q2 = bottom_rects[1].inset(left=0.090, right=0.110).add_axes(fig)
    add_panel_label(ax_q2, 'e', x=-0.12, y=1.05)
    note_ax = None
    if per_type_acc and cm is not None:
        summary = _plot_classifier_metric_heatmap(
            fig,
            ax_q2,
            np.array(cm),
            class_names or [f"C{i}" for i in range(np.array(cm).shape[0])],
            max_rows=18,
        )
        note_ax = add_shared_legend_axes(fig, (ax_q2.get_position().x0, ax_q2.get_position().y0 - 0.07, ax_q2.get_position().width, 0.05))
        note_ax.text(
            0.50,
            0.5,
            f"Acc {gen_acc:.3f}  |  Median F1 {np.median(summary['f1']):.3f}",
            transform=note_ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=COLORS["neutral"],
        )
    else:
        ax_q2.text(0.5, 0.5, "No per-type data", ha="center", va="center",
                   transform=ax_q2.transAxes)

    ax_q3 = bottom_rects[2].add_axes(fig)
    add_panel_label(ax_q3, 'f', x=-0.12, y=1.05)
    disc_proba = classifier_data.get("_disc_proba")
    disc_y = classifier_data.get("_disc_y")
    if disc_proba is not None and disc_y is not None:
        plot_roc_curve(ax_q3, disc_y, disc_proba, auc_value=disc_auc)
    else:
        ax_q3.text(0.5, 0.5, "No discriminator data", ha="center", va="center",
                   transform=ax_q3.transAxes)

    if note_ax is not None:
        note_ax.set_position((ax_q2.get_position().x0, ax_q2.get_position().y0 - 0.065, ax_q2.get_position().width, 0.05))

    if save:
        path = save_panel(
            fig,
            output_dir / "fig17_downstream_pq.png",
            dpi,
            layout_rect=(0.02, 0.03, 0.98, 0.95),
        )
        logger.info(f"Saved merged P+Q \u2192 {path}")
    return fig


# ---------------------------------------------------------------------------
# Entry-point: generate all downstream panels (P, Q, R)
# ---------------------------------------------------------------------------

def generate_downstream_panels(
    downstream_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    type_names: Optional[Dict[int, str]] = None,
    dpi: int = 300,
) -> List[Path]:
    """Load pre-computed downstream JSONs + internal arrays and generate P/Q/R."""
    from .panels_de_concordance import plot_de_concordance_panel

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
            saved.append(out_dir / "fig17_clustering_mixing.png")
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
            saved.append(out_dir / "fig17_classifier_alignment.png")
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
    "plot_clustering_and_classifier_merged",
    "plot_clustering_panel",
    "plot_classifier_panel",
    "_compute_classifier_summary",
    "_plot_classifier_metric_heatmap",
    "generate_downstream_panels",
]
