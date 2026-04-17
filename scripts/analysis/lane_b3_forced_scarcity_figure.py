#!/usr/bin/env python3
"""Lane B3 forced-scarcity augmentation sweep figure (manuscript supplementary).

Shows rare-cell F1 vs augmentation ratio for five strategies under two
scarcity regimes: natural scarcity (baseline ~0.92) and forced scarcity
(30 training cells, baseline ~0.50).

Panel (a): Megakaryocytes (gid 51)
Panel (b): Ameloblasts   (gid 64)

Inputs:
  revision/experiments/lane_b_retrain/b3_mixing_sweep/mixing_sweep.json
  revision/experiments/lane_b_retrain/b3_mixing_sweep/forced_scarcity_sweep.json
Output:
  results/figures/figS_lane_b3_forced_scarcity.{png,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.visualization.style import apply_style, save_with_vcd  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RATIOS = [1, 2, 5, 10]
RATIO_LABELS = ["1×", "2×", "5×", "10×"]

# Wong 2011 colorblind-safe palette
STRATEGY_META = {
    "oversampling":    {"label": "Random oversampling", "color": "#0072B2", "marker": "o"},
    "smote":           {"label": "SMOTE",               "color": "#D55E00", "marker": "s"},
    "clop":            {"label": "CLOP-DiT",            "color": "#009E73", "marker": "^"},
    "clop+oversamp":   {"label": "CLOP-DiT + oversampling", "color": "#CC79A7", "marker": "D"},
    "clop+smote":      {"label": "CLOP-DiT + SMOTE",    "color": "#F0E442", "marker": "P"},
}

PANEL_LABELS = ["(a)", "(b)"]
CELL_TYPE_TITLES = {
    "51": "Megakaryocytes",
    "64": "Ameloblasts",
}
GID_ORDER = ["51", "64"]

NATURAL_CEILING = 0.92   # approximate natural-scarcity F1 ceiling
FORCED_BASELINE = 0.50   # approximate forced-scarcity (30-cell) baseline F1


def _extract_f1s(sweep: dict, strategy: str) -> list[float]:
    """Return rare_f1 values in RATIOS order."""
    return [sweep[strategy][f"{r}x"]["rare_f1"] for r in RATIOS]


def make_figure(
    natural_data: dict,
    forced_data: dict,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=False,
                             dpi=300)

    for ax_idx, gid in enumerate(GID_ORDER):
        ax = axes[ax_idx]
        nat_sweep = natural_data["per_rare_type"][gid]["sweep"]
        frc_sweep = forced_data["per_rare_type"][gid]["sweep"]

        nat_baseline = natural_data["per_rare_type"][gid]["baseline"]["rare_f1"]
        frc_baseline = forced_data["per_rare_type"][gid]["baseline"]["rare_f1"]

        x = np.arange(len(RATIOS))

        # --- shaded reference bands ---
        ax.axhspan(nat_baseline - 0.01, nat_baseline + 0.01,
                   color="#AACCE8", alpha=0.35, zorder=0)
        ax.axhline(nat_baseline, color="#0072B2", lw=1.0, ls="--",
                   alpha=0.7, zorder=1)

        ax.axhspan(frc_baseline - 0.01, frc_baseline + 0.01,
                   color="#F4C6A0", alpha=0.35, zorder=0)
        ax.axhline(frc_baseline, color="#D55E00", lw=1.0, ls=":",
                   alpha=0.7, zorder=1)

        # --- strategy lines ---
        for strategy, meta in STRATEGY_META.items():
            nat_f1s = _extract_f1s(nat_sweep, strategy)
            frc_f1s = _extract_f1s(frc_sweep, strategy)

            # natural scarcity: solid
            ax.plot(x, nat_f1s, color=meta["color"], marker=meta["marker"],
                    markersize=4, lw=1.4, ls="-", zorder=3)
            # forced scarcity: dashed
            ax.plot(x, frc_f1s, color=meta["color"], marker=meta["marker"],
                    markersize=4, lw=1.4, ls="--", alpha=0.75, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(RATIO_LABELS, fontsize=7)
        ax.set_xlabel("Augmentation ratio", fontsize=8)
        if ax_idx == 0:
            ax.set_ylabel("Rare-cell F1 score", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(CELL_TYPE_TITLES[gid], fontsize=8, pad=4)

        # --- baseline text annotations ---
        ax.text(len(RATIOS) - 0.5, nat_baseline + 0.018,
                "natural-scarcity ceiling", fontsize=7.5,
                color="#0072B2", ha="right", va="bottom")
        ax.text(len(RATIOS) - 0.5, frc_baseline + 0.018,
                "forced-scarcity baseline", fontsize=7.5,
                color="#D55E00", ha="right", va="bottom")

        # panel label
        ax.text(-0.08, 1.06, PANEL_LABELS[ax_idx],
                transform=ax.transAxes, fontweight="bold", fontsize=9,
                va="top", ha="left")

        ax.set_ylim(0.45, 0.97)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --- shared legend using proxy Line2D artists (no phantom axes lines) ---
    from matplotlib.lines import Line2D as _Line2D
    legend_handles = []
    for meta in STRATEGY_META.values():
        handle = _Line2D([], [], color=meta["color"], marker=meta["marker"],
                         markersize=4, lw=1.4, ls="-", label=meta["label"])
        legend_handles.append(handle)
    legend_handles.append(_Line2D([], [], color="gray", lw=1.4, ls="-",
                                  label="natural scarcity (solid)"))
    legend_handles.append(_Line2D([], [], color="gray", lw=1.4, ls="--",
                                  alpha=0.75, label="forced scarcity (dashed)"))

    # Place legend below both panels using axes-coord anchor on left panel
    axes[0].legend(handles=legend_handles, loc="upper left",
                   bbox_to_anchor=(0.0, -0.22),
                   ncol=4, fontsize=6.5, frameon=False,
                   handlelength=1.2, borderpad=0.4, handletextpad=0.4,
                   columnspacing=0.8)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30, left=0.10, right=0.97, top=0.95)
    return fig


def main() -> None:
    apply_style()

    nat_path = REPO / "revision/experiments/lane_b_retrain/b3_mixing_sweep/mixing_sweep.json"
    frc_path = REPO / "revision/experiments/lane_b_retrain/b3_mixing_sweep/forced_scarcity_sweep.json"

    natural_data = json.loads(nat_path.read_text())
    forced_data = json.loads(frc_path.read_text())

    out_path = REPO / "results/figures/figS_lane_b3_forced_scarcity"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = make_figure(natural_data, forced_data)
    fig.canvas.draw()  # flush renderer so VCD bbox queries are accurate
    save_with_vcd(fig, out_path)
    plt.close(fig)
    print(f"Saved: {out_path}.pdf  /  {out_path}.png")


if __name__ == "__main__":
    main()
