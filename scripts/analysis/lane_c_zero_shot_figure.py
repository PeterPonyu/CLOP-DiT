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

from src.visualization.style import apply_style, add_panel_label, save_with_vcd  # noqa: E402


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
    # Random baselines follow the full real-tissue vocabularies used by
    # nearest-centroid scoring, not just the prompted/evaluated rows shown
    # in this compact figure. See revision/experiments/lane_c_data/
    # zero_shot_summary.md for the 7/18/2 denominators.
    random_denominators = {
        "CENSUS_KIDNEY": 7,
        "CENSUS_CEREBELLUM": 18,
        "CENSUS_TESTIS_FETAL": 2,
    }

    rows_by_tissue: dict[str, list[tuple[str, float, float]]] = {}
    for tid in tissue_ids:
        per_type = data["per_tissue"][tid]["per_type"]
        random_chance = 1.0 / max(random_denominators.get(tid, len(per_type)), 1)
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

    panel_letters = ["A", "B", "C"]
    for ax_idx, (ax, tid) in enumerate(zip(axes, tissue_ids)):
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
            zorder=2,
        )
        ax.axvline(random_chance, color="#333333", linestyle="--", linewidth=1.1, zorder=1)
        ax.text(
            0.91,
            1.03,
            f"random = {random_chance:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.86),
            clip_on=False,
            zorder=5,
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
                bbox=dict(boxstyle="round,pad=0.06", facecolor="white", edgecolor="none", alpha=0.78),
                zorder=4,
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
        add_panel_label(ax, panel_letters[ax_idx], x=-0.18, y=1.02)

    axes[-1].set_xlabel("Nearest-centroid accuracy (generated -> real OOD type)", fontsize=10.5)
    plt.subplots_adjust(left=0.24, right=0.98, top=0.92, bottom=0.10, hspace=0.55)

    out_png = REPO / "results/figures/figS_lane_c_zero_shot.png"
    save_with_vcd(fig, out_png, dpi=300)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
