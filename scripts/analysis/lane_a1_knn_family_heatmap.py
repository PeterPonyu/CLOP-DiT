#!/usr/bin/env python3
"""Lane A-1 KNN family confusion heatmap (manuscript supplementary figure).

Shows the 10x10 family-level confusion matrix for generated embeddings,
highlighting within-family (diagonal) errors as "graceful degradation".
Includes a row-marginal bar showing within-family error fraction per family.

Input : revision/experiments/lane_a_analysis/a1_knn_confusion/knn_confusion.json
        revision/experiments/lane_a_analysis/a1_knn_confusion/family_taxonomy.yaml
Output: results/figures/figS_lane_a1_knn_family_heatmap.{png,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.visualization.style import apply_style, save_with_vcd  # noqa: E402


# Readable short labels for the 10 families
FAMILY_LABELS = {
    "endothelial": "Endothelial",
    "epithelial": "Epithelial",
    "erythroid_and_hspc": "Erythroid/HSPC",
    "lymphoid": "Lymphoid",
    "mast_and_isg": "Mast/ISG",
    "mesenchymal": "Mesenchymal",
    "myeloid": "Myeloid",
    "neural_and_glial": "Neural/Glial",
    "parenchymal_secretory": "Parenchymal Sec.",
    "proliferation_stress_pluripotent": "Prolif./Stress Pluri.",
}


def main() -> None:
    apply_style()

    data_path = REPO / "revision/experiments/lane_a_analysis/a1_knn_confusion/knn_confusion.json"
    data = json.loads(data_path.read_text())

    cm_data = data["confusion_matrix_family"]
    row_labels_raw = cm_data["row_labels"]
    col_labels_raw = cm_data["col_labels"]
    matrix = np.array(cm_data["matrix"], dtype=float)

    # Row-normalize to fractions (fraction of queries per true family)
    row_sums = matrix.sum(axis=1, keepdims=True)
    frac_matrix = matrix / np.where(row_sums == 0, 1, row_sums)

    # Within-family error fraction per family (diagonal / off-diagonal errors only)
    # = diagonal count / total queries (not just errors, to match the 49.2% headline)
    diag_counts = np.diag(matrix)
    error_counts = row_sums.ravel() - diag_counts
    within_family_frac = np.where(
        error_counts > 0,
        diag_counts / row_sums.ravel(),  # accuracy per family
        1.0,
    )
    # What fraction of errors are within-family:
    within_err_of_err = np.where(
        error_counts > 0,
        (diag_counts - np.maximum(row_sums.ravel() - error_counts - diag_counts, 0)) / np.maximum(error_counts, 1),
        0.0,
    )
    # Simpler: within-family accuracy per family (same as within_family_accuracy in JSON)
    per_family_acc = diag_counts / row_sums.ravel()

    short_labels = [FAMILY_LABELS.get(r, r) for r in row_labels_raw]
    n = len(short_labels)

    # Layout: main heatmap + right-side bar
    fig = plt.figure(figsize=(7.2, 6.8), dpi=300)
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[5.5, 1.2],
        wspace=0.08,
        left=0.22, right=0.97,
        top=0.86, bottom=0.34,
    )
    ax_heat = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # --- Heatmap ---
    im = ax_heat.imshow(frac_matrix, cmap="cividis", vmin=0, vmax=1, aspect="auto")

    # Highlight diagonal with a white border
    for i in range(n):
        rect = plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="white", linewidth=2.0,
        )
        ax_heat.add_patch(rect)

    # Annotate cells: show raw count (skip zeros and diagonal which is clear)
    for i in range(n):
        for j in range(n):
            val = int(matrix[i, j])
            if val == 0:
                continue
            bg = frac_matrix[i, j]
            text_color = "white" if bg < 0.5 else "black"
            ax_heat.text(
                j, i, str(val),
                ha="center", va="center",
                fontsize=7.5, color=text_color,
            )

    ax_heat.set_xticks(np.arange(n))
    ax_heat.set_xticklabels(short_labels, rotation=55, ha="right", fontsize=7.0)
    ax_heat.set_yticks(np.arange(n))
    ax_heat.set_yticklabels(short_labels, fontsize=7.0)
    ax_heat.set_xlabel("Predicted family", fontsize=9.5, labelpad=4)
    ax_heat.set_ylabel("True family", fontsize=9.5, labelpad=4)

    # Colorbar below BOTH axes so the heatmap and the right bar shrink
    # by the same fraction and keep their bottom pixel rows aligned.
    cbar = fig.colorbar(im, ax=[ax_heat, ax_bar], orientation="horizontal",
                        fraction=0.040, pad=0.15, shrink=0.62)
    cbar.set_label("Fraction of queries", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    # --- Right bar: within-family accuracy per family ---
    y_pos = np.arange(n)
    bar_colors = ["#2d7bb6" if acc >= 0.95 else "#f4a261" for acc in per_family_acc]
    ax_bar.barh(y_pos, per_family_acc, color=bar_colors, edgecolor="#333333",
                linewidth=0.4, height=0.72, align="center")
    ax_bar.axvline(0.95, color="#666666", linestyle="--", linewidth=0.9)
    ax_bar.set_xlim(0.85, 1.02)
    ax_bar.set_xticks([0.90, 1.00])
    ax_bar.set_xticklabels(["0.90", "1.00"], fontsize=7.5)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([""] * n)
    # Pin the bar axis y-limits to exactly match the heatmap so each bar
    # sits on the same pixel row as its heatmap row.  imshow uses integer
    # centres with a ±0.5 half-cell pad, so we must replicate the same
    # (n-0.5, -0.5) inverted extents here.  (We deliberately avoid sharey,
    # which would propagate the heatmap's ytick *labels* back onto ax_bar
    # and clutter the right panel.)
    ax_bar.set_ylim(n - 0.5, -0.5)
    ax_bar.set_xlabel("Within-family\naccuracy", fontsize=8, labelpad=4)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="x", alpha=0.25)

    # Title with headline number
    overall_graceful = data["within_family_given_wrong_type"]
    family_acc = data["overall"]["family_accuracy"]
    fig.suptitle(
        f"KNN family confusion — {overall_graceful * 100:.1f}% of cell-type errors are within-family\n"
        f"(family-level accuracy {family_acc * 100:.1f}%, n = {data['overall']['n_samples']:,} generated cells)",
        fontsize=9.5,
        y=0.96,
    )

    out_png = REPO / "results/figures/figS_lane_a1_knn_family_heatmap.png"
    save_with_vcd(fig, out_png, dpi=300)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
