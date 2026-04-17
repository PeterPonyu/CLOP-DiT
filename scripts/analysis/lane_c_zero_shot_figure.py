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
    # Wong 2011 colorblind-safe palette (deut/prot/trit safe)
    tissue_color = {
        "CENSUS_KIDNEY": "#0072B2",          # blue
        "CENSUS_CEREBELLUM": "#D55E00",      # vermillion
        "CENSUS_TESTIS_FETAL": "#009E73",    # bluish-green
    }

    rows_by_tissue: dict[str, list[tuple[str, float, float]]] = {}
    for tid in tissue_ids:
        per_type = data["per_tissue"][tid]["per_type"]
        n_types = len(per_type)
        random_chance = 1.0 / max(n_types, 1)
        rows = []
        for ct, m in per_type.items():
            rows.append((ct, m["nearest_centroid_acc"], random_chance))
        rows_by_tissue[tid] = sorted(rows, key=lambda row: (-row[1], row[0].lower()))

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

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(7.6, 6.8),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.8, 1.0]},
    )
    fig.set_dpi(300)

    for ax, tid in zip(axes, tissue_ids):
        rows = rows_by_tissue[tid]
        labels = [abbrev.get(label, label) for label, _, _ in rows]
        values = [value for _, value, _ in rows]
        random_chance = rows[0][2] if rows else 0.0
        y = np.arange(len(labels))

        ax.barh(
            y,
            values,
            color=tissue_color[tid],
            edgecolor="#333333",
            linewidth=0.4,
            height=0.72,
        )
        ax.axvline(random_chance, color="#333333", linestyle="--", linewidth=1.1)
        random_label_x = random_chance - 0.015 if random_chance >= 0.95 else max(min(random_chance + 0.015, 0.90), 0.03)
        random_label_ha = "right" if random_chance >= 0.95 else "left"
        ax.text(
            random_label_x,
            0.98,
            f"random = {random_chance:.2f}",
            transform=ax.get_xaxis_transform(),
            ha=random_label_ha,
            va="top",
            fontsize=9,
            color="#333333",
        )

        for idx, value in enumerate(values):
            ax.text(
                min(value + 0.018, 1.01),
                idx,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=9.5,
                color="#222222",
            )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.08)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", alpha=0.0)
        ax.set_title(
            tissue_labels[tid],
            fontsize=10.5,
            fontweight="bold",
            color=tissue_color[tid],
            loc="left",
            pad=4,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Nearest-centroid accuracy (generated -> real OOD type)", fontsize=10.5)
    plt.subplots_adjust(left=0.42, right=0.98, top=0.93, bottom=0.16, hspace=0.38)

    out_png = REPO / "results/figures/figS_lane_c_zero_shot.png"
    save_with_vcd(fig, out_png, dpi=300)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
