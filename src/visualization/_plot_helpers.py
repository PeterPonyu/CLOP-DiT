"""
Shared plotting helpers to eliminate duplication across panel files.

Extracted from panels_clustering, panels_classifier, and downstream_panels.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .style import (
    COLORS,
    FONT_HEATMAP_CELL,
    TYPE_PALETTE,
    add_colorbar_safe,
    set_dense_tick_labels,
    style_axes,
)


def plot_umap_overlay(
    ax: plt.Axes,
    umap_coords: np.ndarray,
    source: np.ndarray,
    cell_type: np.ndarray,
    *,
    type_palette: np.ndarray = TYPE_PALETTE,
    real_size: float = 4,
    gen_size: float = 12,
    real_alpha: float = 0.3,
    gen_alpha: float = 0.6,
    legend: bool = True,
    legend_loc: str = "upper left",
    legend_fontsize: int = 9,
    title: str = "UMAP Overlay",
) -> dict[str, np.ndarray]:
    """UMAP scatter with real (circles) and generated (triangles) overlaid by cell type.

    Returns
    -------
    ct_colors : dict mapping cell type label to RGBA color
    """
    unique_types = np.unique(cell_type)
    ct_colors = {ct: type_palette[i % len(type_palette)]
                 for i, ct in enumerate(sorted(unique_types))}

    real_mask = source == "real"
    gen_mask = source == "generated"

    for ct in unique_types:
        mask = real_mask & (cell_type == ct)
        if mask.any():
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[ct_colors[ct]], s=real_size, alpha=real_alpha,
                       rasterized=True)
    for ct in unique_types:
        mask = gen_mask & (cell_type == ct)
        if mask.any():
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[ct_colors[ct]], s=gen_size, alpha=gen_alpha,
                       marker="^", edgecolors="black", linewidths=0.3,
                       rasterized=True)

    if legend:
        ax.scatter([], [], c="gray", s=15, marker="o", label="Real")
        ax.scatter([], [], c="gray", s=15, marker="^", edgecolors="black",
                   linewidths=0.3, label="Generated")
        ax.legend(fontsize=legend_fontsize, loc=legend_loc, markerscale=2,
                  frameon=False)

    style_axes(ax, "umap", title=title, xlabel="UMAP 1", ylabel="UMAP 2")
    return ct_colors


def plot_confusion_matrix(
    ax: plt.Axes,
    cm: np.ndarray,
    class_names: list[str] | None = None,
    *,
    cmap: str = "Blues",
    max_labels: int = 30,
    colorbar: bool = True,
    colorbar_orientation: str = "horizontal",
    colorbar_pad: float = 0.10,
    title: str = "Confusion Matrix",
) -> plt.cm.ScalarMappable:
    """Render normalized confusion matrix with adaptive tick labels.

    Returns the imshow mappable for optional external colorbar use.
    """
    cm = np.asarray(cm)
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"C{i}" for i in range(n_classes)]

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    im = ax.imshow(cm_norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    if n_classes > max_labels:
        tick_step = max(1, n_classes // 7)
        ticks = list(range(0, n_classes, tick_step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(i) for i in ticks], fontsize=8)
        ax.set_yticklabels([str(i) for i in ticks], fontsize=8)
        # Sparse diagonal annotations
        diag_step = max(1, n_classes // 8)
        safe_end = max(0, n_classes - diag_step)
        for i in range(0, min(safe_end, cm_norm.shape[0]), diag_step):
            val = cm_norm[i, i]
            color = "white" if val > 0.5 else "black"
            ax.text(i, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=FONT_HEATMAP_CELL, color=color)
    else:
        short = [n[:18] for n in class_names]
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(short, rotation=90, fontsize=8, ha="center")
        ax.set_yticklabels(short, fontsize=8, ha="right")
        set_dense_tick_labels(ax, axis="both", max_labels=20, fontsize=8,
                              rotation=90, ha="center")
        for i in range(min(n_classes, cm_norm.shape[0])):
            val = cm_norm[i, i]
            color = "white" if val > 0.5 else "black"
            ax.text(i + 0.25, i, f"{val:.2f}", ha="left", va="center",
                    fontsize=8, color=color)

    if colorbar:
        add_colorbar_safe(im, ax=ax, label="Recall",
                          shrink=0.50, pad=colorbar_pad,
                          orientation=colorbar_orientation, aspect=20)

    style_axes(ax, "heatmap", title=title,
               xlabel="Predicted", ylabel="True Type")
    return im


def plot_roc_curve(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    auc_value: float | None = None,
    color: str | None = None,
    title: str = "Discriminator ROC",
) -> None:
    """ROC curve with fill_between and random-baseline line."""
    from sklearn.metrics import roc_curve as _roc_curve

    if color is None:
        color = COLORS["real"]

    fpr, tpr, _ = _roc_curve(y_true, y_proba)
    label = f"AUC = {auc_value:.3f}" if auc_value is not None else "AUC"
    ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.6,
            label="Random (0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1, color=color)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    style_axes(ax, "scatter", title=title, xlabel="FPR", ylabel="TPR")
