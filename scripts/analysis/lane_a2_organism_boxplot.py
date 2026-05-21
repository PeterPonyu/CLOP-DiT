"""
Lane A2 — Organism-stratified boxplots (human_only / mouse_only / dual).

Output:
  results/figures/figS_lane_a2_organism_stratified.pdf  (+ .png)

Six panels (2x3 grid):
  (a) centroid_cosine    (b) frechet_distance   (c) diversity_ratio
  (d) expr_pearson_r     (e) real_intra_cos      (f) n_real (log scale)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import FIG_DIR
from src.visualization.style import apply_style, save_with_vcd

DATA_DIR = REPO_ROOT / "revision/experiments/lane_a_analysis/a2_organism_split"
CSV_PATH = DATA_DIR / "organism_per_type_table.csv"
JSON_PATH = DATA_DIR / "organism_split.json"

OUT_STEM = FIG_DIR / "figS_lane_a2_organism_stratified"

# Colorblind-safe palette
COLORS = {
    "human_only": "#0072B2",
    "mouse_only":  "#D55E00",
    "both":        "#009E73",
}
GROUP_ORDER = ["human_only", "mouse_only", "both"]
GROUP_LABELS = {"human_only": "Human", "mouse_only": "Mouse", "both": "Dual"}

# ---------------------------------------------------------------------------
# Panels definition
# ---------------------------------------------------------------------------
PANELS = [
    dict(col="centroid_cosine",  title="Centroid cosine",       ylabel="Cosine similarity", log=False, panel="a"),
    dict(col="frechet_distance", title="Fréchet distance",      ylabel="FD",                log=False, panel="b"),
    dict(col="diversity_ratio",  title="Diversity ratio (DivR)",ylabel="DivR",              log=False, panel="c"),
    dict(col="expr_pearson_r",   title="Expression Pearson r",  ylabel="Pearson r",         log=False, panel="d"),
    dict(col="real_intra_cos",   title="Real intra-cos tightness", ylabel="Intra-type cosine", log=False, panel="e"),
    dict(col="n_real",           title="Training cell count",   ylabel="# cells",           log=True,  panel="f"),
]


def _pval_str(p: float) -> str:
    if p is None:
        return "n.s."
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def _mannwhitney_p(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(p)


def _load_json_pairwise_pvalues() -> dict[str, float]:
    """Load authoritative human-vs-mouse p-values from the Lane A2 JSON."""
    if not JSON_PATH.exists():
        return {}
    try:
        with open(JSON_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    pvalues: dict[str, float] = {}
    pairwise = data.get("pairwise_mannwhitney", {})
    for metric, comparisons in pairwise.items():
        if not isinstance(comparisons, dict):
            continue
        comparison = comparisons.get("human_only_vs_mouse_only", {})
        if not isinstance(comparison, dict):
            continue
        p = comparison.get("mannwhitney_p_two_sided")
        if p is None:
            continue
        try:
            pvalues[metric] = float(p)
        except (TypeError, ValueError):
            continue
    return pvalues


def make_figure(df: pd.DataFrame, pairwise_pvalues: dict[str, float] | None = None) -> plt.Figure:
    apply_style()
    pairwise_pvalues = pairwise_pvalues or {}

    # Canvas enlarged from (7.2, 4.8) → (8.6, 5.6) so each panel is ~2.4 in²
    # (up from ~1.7 in²); tighter layout + better title/label fontsizes follow
    # from the extra physical room.
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.8), dpi=300)
    axes_flat = axes.flatten()

    for ax, panel in zip(axes_flat, PANELS):
        col = panel["col"]
        log = panel["log"]

        # Collect per-group values
        groups: dict[str, np.ndarray] = {}
        for g in GROUP_ORDER:
            vals = df.loc[df["organism_group"] == g, col].dropna().values.astype(float)
            groups[g] = vals

        # ---- boxplot ----
        positions = [1, 2, 3]
        box_data = [groups[g] for g in GROUP_ORDER]

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            notch=False,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            boxprops=dict(linewidth=1.0),
        )
        for patch, g in zip(bp["boxes"], GROUP_ORDER):
            patch.set_facecolor(COLORS[g])
            patch.set_alpha(0.7)

        # ---- strip plot ----
        rng = np.random.default_rng(42)
        for pos, g in zip(positions, GROUP_ORDER):
            vals = groups[g]
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(
                pos + jitter, vals,
                color=COLORS[g],
                s=12, alpha=0.6, linewidths=0, zorder=3,
            )

        # ---- log scale ----
        if log:
            ax.set_yscale("log")

        # ---- p-value annotation (human vs mouse, top-right) ----
        h_vals = groups["human_only"]
        m_vals = groups["mouse_only"]
        p_calc = _mannwhitney_p(h_vals, m_vals)
        p_use = pairwise_pvalues.get(col, p_calc)

        # Fold the MW p-value into the subplot title so it cannot overlap the
        # title (title airspace) or the data points (plot interior) — these
        # were the two collision regimes from earlier standalone-text attempts
        # at (0.97, 0.97) and (0.97, 0.88).
        _p_suffix = _pval_str(p_use)

        # ---- axes cosmetics ----
        ax.set_xticks(positions)
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], fontsize=9.5)
        ax.set_ylabel(panel["ylabel"], fontsize=10)
        ax.set_title(f"{panel['title']}\nMW {_p_suffix}", fontsize=10.4, pad=6)
        ax.tick_params(axis="y", labelsize=9.5)
        ax.set_xlim(0.4, 3.6)
        ax.grid(axis="y", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

        # ---- expr_pearson_r: suppress offset text and use explicit ticks ----
        if col == "expr_pearson_r":
            ax.yaxis.get_offset_text().set_visible(False)
            ax.set_yticks([0.99990, 0.99995, 1.00000])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.5f}"))

        # ---- log-scale panels: force only decade ticks to avoid stray "10^x"
        # superscript near the title at the top of the axes. ----
        if log:
            ax.yaxis.set_major_locator(FixedLocator([1e2, 1e3, 1e4]))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

        # ---- panel label (bold uppercase, above the title) ----
        # Push the label higher (y=1.22) and slightly further left (x=-0.22)
        # so the bold "A"/"B"/... does not visually collide with either the
        # subplot title (y ~= 1.05) or the p-value annotation pinned at the
        # top-right of the axes.
        from src.visualization.style import add_panel_label
        add_panel_label(ax, panel["panel"], x=-0.10, y=1.05, fontsize=15)

    # ---- figure-level legend ----
    legend_patches = [
        mpatches.Patch(facecolor=COLORS[g], alpha=0.75, label=GROUP_LABELS[g])
        for g in GROUP_ORDER
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    # Suptitle lives INSIDE the rect that tight_layout packs into, so there
    # is no dead whitespace band between the title and Panel A/B/C. Earlier
    # y=1.01 put the title outside the figure top and tight_layout then left
    # a visible gap down to the first row.
    # Suptitle sits just above row-1 panels with a tight gap (y=0.965 inside
    # the rect.top=0.96 band) and uses the default regular weight — per
    # reviewer feedback the previous bold + y=0.99 variant sat too far from
    # the panel row with a heavy emphasis that looked over-styled.
    fig.suptitle(
        "Organism-stratified generation quality",
        fontsize=11, y=0.965,
    )
    fig.subplots_adjust(
        left=0.12, right=0.985,
        bottom=0.14, top=0.86,
        wspace=0.42, hspace=0.88,
    )

    return fig


def main() -> None:
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    pairwise_pvalues = _load_json_pairwise_pvalues()

    # Save
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure(df, pairwise_pvalues)
    save_with_vcd(fig, OUT_STEM)
    plt.close(fig)

    pdf_path = OUT_STEM.with_suffix(".pdf")
    size_kb = pdf_path.stat().st_size / 1024
    print(f"Saved: {pdf_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
