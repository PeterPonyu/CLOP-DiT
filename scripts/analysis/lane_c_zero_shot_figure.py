#!/usr/bin/env python3
"""Lane C zero-shot strict-OOD bar chart (manuscript supplementary figure).

Renders per-cell-type nearest-centroid accuracy from the zero-shot
evaluation, grouped by tissue, with random-chance reference lines.

Input : revision/experiments/lane_c_data/zero_shot_results.json
Output: results/figures/figS_lane_c_zero_shot.{png,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.visualization.style import apply_style, COLORS, save_with_vcd  # noqa: E402


def main() -> None:
    apply_style()
    results_path = REPO / "revision/experiments/lane_c_data/zero_shot_results.json"
    data = json.loads(results_path.read_text())

    tissue_ids = ["CENSUS_KIDNEY", "CENSUS_CEREBELLUM", "CENSUS_TESTIS_FETAL"]
    tissue_labels = {
        "CENSUS_KIDNEY": "Kidney\n(Tabula Sapiens)",
        "CENSUS_CEREBELLUM": "Cerebellum\n(brain atlas)",
        "CENSUS_TESTIS_FETAL": "Fetal gonadal\n(adrenal + Leydig)",
    }
    tissue_color = {
        "CENSUS_KIDNEY": COLORS.get("generated", "#3E7BB6"),
        "CENSUS_CEREBELLUM": COLORS.get("real", "#C46B4E"),
        "CENSUS_TESTIS_FETAL": COLORS.get("baseline", "#6A8E7F"),
    }

    rows: list[tuple[str, str, float, float]] = []
    for tid in tissue_ids:
        per_type = data["per_tissue"][tid]["per_type"]
        n_types = len(per_type)
        random_chance = 1.0 / max(n_types, 1)
        for ct, m in per_type.items():
            rows.append((tid, ct, m["nearest_centroid_acc"], random_chance))

    # Abbreviate long cell-type names for legible x-axis labels
    abbrev = {
        "kidney epithelial cell": "kidney epithelial",
        "Purkinje cell": "Purkinje",
        "granule cell": "granule",
        "interneuron": "interneuron",
        "inhibitory interneuron": "inh. interneuron",
        "glial cell": "glial",
        "ependymal cell": "ependymal",
        "Bergmann glial cell": "Bergmann glial",
        "CNS interneuron": "CNS interneuron",
        "brainstem motor neuron": "brainstem motor",
        "cerebellar granule cell precursor": "granule precursor",
        "Leydig cell": "Leydig",
        "type I cell of adrenal cortex": "adrenal cortex I",
    }

    fig, ax = plt.subplots(figsize=(14.0, 5.8))

    xs, heights, colors, labels = [], [], [], []
    tissue_group_bounds: dict[str, tuple[int, int]] = {}
    pos = 0
    for tid in tissue_ids:
        start = pos
        for tid_row, ct, acc, _ in rows:
            if tid_row != tid:
                continue
            xs.append(pos)
            heights.append(acc)
            colors.append(tissue_color[tid])
            labels.append(ct)
            pos += 1
        tissue_group_bounds[tid] = (start, pos - 1)
        pos += 1.8  # wider spacer between tissue groups

    bars = ax.bar(xs, heights, color=colors, width=0.85, edgecolor="#333", linewidth=0.5)

    # Random-chance horizontal segments per tissue group
    for tid, (a, b) in tissue_group_bounds.items():
        n_types = len(data["per_tissue"][tid]["per_type"])
        if n_types == 0:
            continue
        rc = 1.0 / n_types
        ax.hlines(rc, a - 0.5, b + 0.5, colors="#333", linestyles="--",
                  linewidth=1.0, zorder=3,
                  label=f"Random (1/{n_types})" if tid == tissue_ids[0] else None)

    # Tissue group bracket labels at the bottom of the figure (below x-ticks)
    for tid, (a, b) in tissue_group_bounds.items():
        ax.text((a + b) / 2, -0.42, tissue_labels[tid],
                ha="center", va="top", fontsize=11, fontweight="bold",
                transform=ax.get_xaxis_transform(), color=tissue_color[tid])
        # horizontal bracket line under tissue group
        ax.plot([a - 0.4, b + 0.4], [-0.30, -0.30],
                transform=ax.get_xaxis_transform(),
                color=tissue_color[tid], linewidth=2.2, clip_on=False)

    ax.set_xticks(xs)
    ax.set_xticklabels([abbrev.get(l, l) for l in labels],
                       rotation=50, ha="right", fontsize=8.5)
    ax.set_ylabel("Nearest-centroid accuracy (generated → real OOD type)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.axhline(0.0, color="#999", linewidth=0.5, zorder=1)
    ax.set_title(
        "Zero-shot strict-OOD generalization by novel cell type",
        fontsize=13, pad=10,
    )

    # Value labels on top of bars
    for x, h in zip(xs, heights):
        ax.text(x, h + 0.018, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    # Legend above the plot area to avoid overlapping bars
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.12),
              fontsize=9, frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.subplots_adjust(bottom=0.38, top=0.88, left=0.07, right=0.98)

    out_png = REPO / "results/figures/figS_lane_c_zero_shot.png"
    save_with_vcd(fig, out_png)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
