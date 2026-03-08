"""
panels_classifier.py — Panel Q: Classifier alignment (confusion matrix, per-type metrics, ROC).
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .style import (
    add_panel_label,
    COLORS,
    FONT_LEGEND,
    add_colorbar_safe,
    save_panel,
    set_dense_tick_labels,
    set_figure_suptitle,
    style_axes,
)
logger = logging.getLogger(__name__)


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
    max_rows: Optional[int] = 24,
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
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Precision", "Recall", "F1"], fontsize=9)
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(
        [
            textwrap.shorten(str(name).replace("_", " "), width=22, placeholder="…")
            for name in display_names
        ],
        fontsize=8,
    )
    default_title = "Per-Type P/R/F1"
    if rows_truncated:
        if display_mode == "worst+best":
            default_title += f" (worst+best {len(display_names)})"
        else:
            default_title += f" (top {len(display_names)})"
    ax.set_title(title or default_title, fontsize=10)
    ax.set_ylabel("Cell Type (sorted by F1)")
    # ax.set_xlabel("Metric")  # Removed to reduce label density

    cbar = add_colorbar_safe(im, ax=ax, orientation="horizontal", shrink=0.55, pad=0.12, aspect=24)
    cbar.set_label("Score", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    return summary


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
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 0.9],
                          wspace=0.50)
    gen_acc = classifier_data.get("gen_accuracy", 0)
    gen_f1 = classifier_data.get("gen_f1", 0)
    disc_auc = classifier_data.get("discriminator_auc", 0)
    # suptitle and stats banner removed per revision; title information moved to LaTeX caption

    ax = fig.add_subplot(gs[0])
    add_panel_label(ax, 'a')
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
                    fontsize=7, color=color)

    add_colorbar_safe(im, ax=ax, shrink=0.5, pad=0.10, label="Recall",
                      orientation="horizontal", aspect=20)
    style_axes(ax, "heatmap", title="Confusion Matrix (on Generated Cells)",
               xlabel="Predicted", ylabel="True Type")

    ax2 = fig.add_subplot(gs[1])
    add_panel_label(ax2, 'b', x=-0.15, y=1.08)
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
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#DDDDDD", alpha=0.92),
        )
    else:
        ax2.text(0.5, 0.5, "No per-type data", ha="center", va="center",
                 transform=ax2.transAxes)
        ax2.set_title("Per-Type Summary")

    ax3 = fig.add_subplot(gs[2])
    add_panel_label(ax3, 'c', x=-0.15, y=1.02)
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
        path = save_panel(fig, output_dir / "panel_q_classifier_alignment.png", dpi)
        logger.info(f"Saved Panel Q → {path}")
    return fig
