"""Integrated Supplementary Figure S2 composed directly in Python.

This replaces the manuscript-side LaTeX stitching of four supplementary PDFs
with a single Matplotlib-composed appendix figure that controls panel geometry
in one place.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.paths import RESULTS_DIR

from .style import (
    COLORS,
    FONT_ANNOTATION,
    FONT_LABEL,
    FONT_SMALL,
    FONT_TICK,
    FONT_TITLE,
    abbreviate_cell_type,
    add_panel_label,
    apply_style,
    save_with_vcd,
    style_axes,
)
from .direct_layout import bind_figure_region

logger = logging.getLogger(__name__)


_TITLE_SIZE = max(FONT_TITLE - 1, 12)
_LABEL_SIZE = 15
_LABEL_Y = 1.08
_TIER_COLORS = {"pass": "#1B5E20", "warn": "#F9A825", "fail": "#D84315"}
_VARIANT_DISPLAY = {
    "full": "Full prompt",
    "markers_only": "Markers only",
    "celltype_only": "Cell type only",
    "tissue_only": "Tissue only",
    "organism_only": "Organism only",
    "disease_only": "Disease only",
    "no_markers": "No markers",
    "no_celltype": "No cell type",
}


def _placeholder(ax: plt.Axes, title: str) -> None:
    ax.text(
        0.5,
        0.5,
        "Data not available",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=FONT_LABEL,
        color=COLORS["neutral"],
        style="italic",
    )
    ax.set_title(title, fontsize=_TITLE_SIZE, fontweight="normal")
    ax.set_xticks([])
    ax.set_yticks([])
    style_axes(ax)


# ---------------------------------------------------------------------------
# Panels a, b: pseudobulk validation
# ---------------------------------------------------------------------------

def _panel_a(ax: plt.Axes, summary_path: Path, per_type_path: Path) -> None:
    if not summary_path.exists() or not per_type_path.exists():
        _placeholder(ax, "Per-Type Pseudobulk Corr.")
        return

    with open(summary_path) as f:
        summary = json.load(f)
    with open(per_type_path) as f:
        per_type = json.load(f)

    sorted_types = sorted(per_type.items(), key=lambda x: x[1]["pearson_r"], reverse=True)
    names = [abbreviate_cell_type(n, max_len=14) for n, _ in sorted_types]
    pearson_vals = [v["pearson_r"] for _, v in sorted_types]

    median_r = np.median(pearson_vals)
    colors = [
        COLORS["good"] if v >= 0.8 else (COLORS["warn"] if v >= 0.5 else COLORS["bad"])
        for v in pearson_vals
    ]

    n_types = len(names)
    step = max(1, n_types // 12)
    thin_labels = [names[i] if i % step == 0 or i == n_types - 1 else "" for i in range(n_types)]

    ax.barh(range(n_types), pearson_vals, color=colors, height=0.7, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(thin_labels, fontsize=max(FONT_TICK - 1, 9))
    ax.invert_yaxis()
    ax.axvline(median_r, color="#555", linestyle="--", linewidth=1.2, label=f"Median = {median_r:.3f}")
    ax.axvline(0.8, color=COLORS["good"], linestyle=":", linewidth=1.0, alpha=0.5, label="r = 0.8")
    ax.legend(fontsize=FONT_ANNOTATION, frameon=False, loc="lower right")
    style_axes(ax, "bar", xlabel="Pearson r", title="Per-Type Pseudobulk Corr.")



def _panel_b(ax: plt.Axes, summary_path: Path, per_type_path: Path) -> None:
    if not summary_path.exists() or not per_type_path.exists():
        _placeholder(ax, "Pseudobulk Correlation Dist.")
        return

    with open(summary_path) as f:
        summary = json.load(f)
    with open(per_type_path) as f:
        per_type = json.load(f)

    pearson_vals = [v["pearson_r"] for v in per_type.values()]
    if not pearson_vals:
        _placeholder(ax, "Pseudobulk Correlation Dist.")
        return

    median_r = float(np.median(pearson_vals))
    mean_r = float(np.mean(pearson_vals))
    ax.hist(pearson_vals, bins=20, color=COLORS["real"], alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(median_r, color=COLORS["generated"], linestyle="--", linewidth=2, label=f"Median = {median_r:.3f}")
    ax.axvline(mean_r, color=COLORS["accent"], linestyle="-.", linewidth=1.5, label=f"Mean = {mean_r:.3f}")

    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    import matplotlib.ticker as ticker

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_fontsize(max(FONT_TICK - 1, 9))

    text = (
        f"n = {summary['n_types_evaluated']} types\n"
        f"r > 0.9: {summary['pct_pearson_gt_0.9']:.0f}%\n"
        f"r > 0.8: {summary['pct_pearson_gt_0.8']:.0f}%"
    )
    ax.text(
        0.04,
        0.36,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SMALL,
        color=COLORS["neutral"],
    )
    ax.legend(fontsize=FONT_ANNOTATION, frameon=False, loc="upper left")
    style_axes(ax, "default", xlabel="Pearson r", ylabel="Count", title="Pseudobulk Correlation Dist.")


# ---------------------------------------------------------------------------
# Panels c, d, e: failure analysis
# ---------------------------------------------------------------------------

def _panel_c(ax: plt.Axes, analysis_path: Path, tiers_path: Path) -> None:
    if not analysis_path.exists() or not tiers_path.exists():
        _placeholder(ax, "Quality by Family")
        return

    with open(analysis_path) as f:
        analysis = json.load(f)

    patterns = analysis["summary"]["patterns"]
    by_family = patterns.get("by_family", {})
    families = sorted(by_family.keys(), key=lambda f: by_family[f]["mean_score"], reverse=True)
    if not families:
        _placeholder(ax, "Quality by Family")
        return

    y_pos = np.arange(len(families))
    pass_counts = [by_family[f]["tier_counts"].get("pass", 0) for f in families]
    warn_counts = [by_family[f]["tier_counts"].get("warn", 0) for f in families]
    fail_counts = [by_family[f]["tier_counts"].get("fail", 0) for f in families]

    ax.barh(y_pos, pass_counts, color=_TIER_COLORS["pass"], height=0.6, label="Pass")
    ax.barh(y_pos, warn_counts, left=pass_counts, color=_TIER_COLORS["warn"], height=0.6, label="Warn")
    cumulative = [p + w for p, w in zip(pass_counts, warn_counts)]
    ax.barh(y_pos, fail_counts, left=cumulative, color=_TIER_COLORS["fail"], height=0.6, label="Fail")

    short_families = [f.replace("Immune: ", "").replace("Stem/", "S/")[:12] for f in families]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_families, fontsize=max(FONT_TICK - 1, 9))
    ax.invert_yaxis()
    ax.legend(fontsize=FONT_ANNOTATION, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=3)
    style_axes(ax, "bar", xlabel="Number of cell types", title="Quality by Family")



def _panel_d(ax: plt.Axes, analysis_path: Path, tiers_path: Path) -> None:
    if not analysis_path.exists() or not tiers_path.exists():
        _placeholder(ax, "Quality vs Sample Size")
        return

    with open(analysis_path) as f:
        analysis = json.load(f)
    with open(tiers_path) as f:
        tiers = json.load(f)

    per_type = analysis["per_type_metrics"]
    names = list(per_type.keys())
    n_reals = np.array([per_type[n]["n_real"] for n in names])
    scores = np.array([tiers[n]["score"] for n in names])
    tier_labels = [tiers[n]["tier"] for n in names]
    point_colors = [_TIER_COLORS[t] for t in tier_labels]

    ax.scatter(n_reals, scores, c=point_colors, s=30, alpha=0.7, edgecolors="white", linewidth=0.3, zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))

    from scipy import stats as scipy_stats

    log_n = np.log10(n_reals + 1)
    valid = np.isfinite(log_n) & np.isfinite(scores)
    if valid.sum() >= 3:
        slope, intercept, r_val, _, _ = scipy_stats.linregress(log_n[valid], scores[valid])
        x_fit = np.linspace(log_n[valid].min(), log_n[valid].max(), 50)
        ax.plot(10 ** x_fit, slope * x_fit + intercept, color="#555", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.text(0.03, 0.04, f"r={r_val:.3f}", transform=ax.transAxes, fontsize=FONT_ANNOTATION, color=COLORS["neutral"])

    ax.set_ylim(4.5, 8.3)
    ax.set_yticks([5, 6, 7, 8])
    style_axes(ax, "scatter", xlabel="Training cells (log scale)", ylabel="Quality score (0–8)", title="Quality vs Sample Size")



def _panel_e(ax: plt.Axes, analysis_path: Path, tiers_path: Path) -> None:
    if not analysis_path.exists() or not tiers_path.exists():
        _placeholder(ax, "Quality vs Confusability")
        return

    with open(analysis_path) as f:
        analysis = json.load(f)
    with open(tiers_path) as f:
        tiers = json.load(f)

    per_type = analysis["per_type_metrics"]
    names = list(per_type.keys())
    overlaps = np.array([per_type[n]["max_overlap_with_other_type"] for n in names])
    scores = np.array([tiers[n]["score"] for n in names])
    tier_labels = [tiers[n]["tier"] for n in names]
    point_colors = [_TIER_COLORS[t] for t in tier_labels]

    ax.scatter(overlaps, scores, c=point_colors, s=30, alpha=0.7, edgecolors="white", linewidth=0.3, zorder=3)

    from scipy import stats as scipy_stats

    valid = np.isfinite(overlaps) & np.isfinite(scores)
    if valid.sum() >= 5:
        slope, intercept, r_val, _, _ = scipy_stats.linregress(overlaps[valid], scores[valid])
        x_fit = np.linspace(overlaps[valid].min(), overlaps[valid].max(), 50)
        ax.plot(x_fit, slope * x_fit + intercept, color="#555", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.text(0.03, 0.04, f"r={r_val:.3f}", transform=ax.transAxes, fontsize=FONT_ANNOTATION, color=COLORS["neutral"])

    ax.set_ylim(4.5, 8.3)
    ax.set_yticks([5, 6, 7, 8])
    style_axes(ax, "scatter", xlabel="Max cosine overlap", ylabel="Quality score (0–8)", title="Quality vs Confusability")


# ---------------------------------------------------------------------------
# Panels f, g: field ablation
# ---------------------------------------------------------------------------

def _panel_f(ax: plt.Axes, results_path: Path) -> None:
    if not results_path.exists():
        _placeholder(ax, "Steering by Variant")
        return

    with open(results_path) as f:
        results = json.load(f)

    order = ["full", "markers_only", "celltype_only", "no_markers", "no_celltype", "tissue_only", "organism_only", "disease_only"]
    present_variants = [v for v in order if v in results]
    labels = [_VARIANT_DISPLAY.get(v, v) for v in present_variants]
    knn_vals = [results[v]["knn_accuracy"] for v in present_variants]

    y_pos = np.arange(len(labels))
    colors = []
    for v in present_variants:
        if v == "full":
            colors.append(COLORS["real"])
        elif "only" in v:
            colors.append(COLORS["generated"])
        else:
            colors.append(COLORS["accent"])

    bars = ax.barh(y_pos, knn_vals, color=colors, height=0.6, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=max(FONT_TICK - 1, 9))
    ax.invert_yaxis()
    for i, (bar, val) in enumerate(zip(bars, knn_vals)):
        ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=FONT_ANNOTATION, color=COLORS["neutral"])

    style_axes(ax, "bar", xlabel="KNN Accuracy", title="Steering by Variant")



def _panel_g(ax: plt.Axes, results_path: Path) -> None:
    if not results_path.exists():
        _placeholder(ax, "Field Contribution")
        return

    with open(results_path) as f:
        results = json.load(f)

    order = ["full", "markers_only", "celltype_only", "no_markers", "no_celltype", "tissue_only", "organism_only", "disease_only"]
    present_variants = [v for v in order if v in results]
    ret_variants = [v for v in present_variants if v != "full"]
    ret_labels = [_VARIANT_DISPLAY.get(v, v) for v in ret_variants]
    ret_vals = [results[v].get("knn_retention_pct", 0) for v in ret_variants]
    ret_y = np.arange(len(ret_labels))
    ret_colors = [COLORS["good"] if r >= 80 else (COLORS["warn"] if r >= 50 else COLORS["bad"]) for r in ret_vals]

    bars = ax.barh(ret_y, ret_vals, color=ret_colors, height=0.6, edgecolor="white", linewidth=0.3)
    ax.set_yticks(ret_y)
    ax.set_yticklabels(ret_labels, fontsize=max(FONT_TICK - 1, 9))
    ax.invert_yaxis()
    ax.axvline(100, color="#999", linestyle=":", linewidth=1.0, alpha=0.5)
    for i, (bar, val) in enumerate(zip(bars, ret_vals)):
        ax.text(val + 1.0, i, f"{val:.1f}%", va="center", fontsize=FONT_ANNOTATION, color=COLORS["neutral"])

    style_axes(ax, "bar", xlabel="Retention (%)", title="Field Contribution")


# ---------------------------------------------------------------------------
# Panels h, i, j: discriminator analysis
# ---------------------------------------------------------------------------

def _panel_h(ax: plt.Axes, results_path: Path) -> None:
    if not results_path.exists():
        _placeholder(ax, "Separability Driver")
        return

    with open(results_path) as f:
        results = json.load(f)

    mv = results.get("mean_vs_variance", {})
    labels_bar = ["Mean", "Variance"]
    r_mean = abs(mv.get("coef_vs_mean_diff", {}).get("pearson_r", 0))
    r_var = abs(mv.get("coef_vs_var_diff", {}).get("pearson_r", 0))
    vals = [r_mean, r_var]
    bars = ax.bar(labels_bar, vals, color=[COLORS["real"], COLORS["generated"]], width=0.5, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"r={val:.3f}", ha="center", fontsize=FONT_ANNOTATION, color=COLORS["neutral"])

    driver = "Mean shift" if r_mean > r_var else "Variance shift"
    ax.text(0.5, 0.95, f"Primary driver: {driver}", transform=ax.transAxes, ha="center", va="top", fontsize=max(FONT_SMALL + 1, 10), fontweight="bold", color=COLORS["annotation_dark"])
    ax.set_ylim(0, max(vals) * 1.32 if max(vals) > 0 else 1.0)
    style_axes(ax, "bar", ylabel="|Pearson r|", title="Separability Driver")



def _panel_i(ax: plt.Axes, results_path: Path) -> None:
    if not results_path.exists():
        _placeholder(ax, "Dim. Concentration")
        return

    with open(results_path) as f:
        results = json.load(f)

    conc = results.get("coefficient_analysis", {}).get("concentration", {})
    x_labels = ["10", "20", "50"]
    x_vals = [conc.get("top10_fraction", 0), conc.get("top20_fraction", 0), conc.get("top50_fraction", 0)]
    bars = ax.bar(x_labels, [v * 100 for v in x_vals], color=COLORS["accent"], width=0.5, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, x_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.0%}", ha="center", fontsize=FONT_ANNOTATION, color=COLORS["neutral"])

    gini = conc.get("gini", 0)
    ax.text(0.97, 0.97, f"Gini = {gini:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=max(FONT_SMALL + 1, 10), color=COLORS["neutral"])
    ax.set_ylim(0, 100)
    style_axes(ax, "bar", xlabel="Top-K dims", ylabel="% discrim. wt.", title="Dim. Concentration")



def _panel_j(ax: plt.Axes, results_path: Path) -> None:
    if not results_path.exists():
        _placeholder(ax, "Per-Type Separability")
        return

    with open(results_path) as f:
        results = json.load(f)

    per_type = results.get("per_type", {})
    sorted_types = sorted(per_type.items(), key=lambda x: x[1]["auc"], reverse=True)
    if not sorted_types:
        _placeholder(ax, "Per-Type Separability")
        return

    type_names = [abbreviate_cell_type(n, max_len=14) for n, _ in sorted_types]
    auc_vals = [v["auc"] for _, v in sorted_types]
    n_types = len(type_names)
    step = max(1, n_types // 10)
    thin = [type_names[i] if i % step == 0 or i == n_types - 1 else "" for i in range(n_types)]
    pt_colors = [COLORS["bad"] if a > 0.7 else (COLORS["warn"] if a > 0.6 else COLORS["good"]) for a in auc_vals]

    ax.barh(range(n_types), auc_vals, color=pt_colors, height=0.7, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(thin, fontsize=max(FONT_TICK - 1, 9))
    ax.invert_yaxis()
    ax.axvline(0.5, color="#999", linestyle=":", linewidth=1.0, alpha=0.5, label="Chance (0.5)")
    auc_mean = float(np.mean(auc_vals))
    ax.axvline(auc_mean, color="#555", linestyle="--", linewidth=1.2, label=f"Mean = {auc_mean:.3f}")
    ax.legend(fontsize=FONT_SMALL, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.10))
    style_axes(ax, "bar", xlabel="Discriminator AUC", title="Per-Type Separability")



def plot_expression_diagnostics(output_dir: str | Path = "results/figures", dpi: int = 300, save: bool = True):
    """Render the integrated Supplementary Figure S2."""
    apply_style()
    output_dir = Path(output_dir)
    results_dir = Path(RESULTS_DIR)
    val_dir = results_dir / "validation"

    pseudobulk_summary = val_dir / "pseudobulk_validation.json"
    pseudobulk_per_type = val_dir / "pseudobulk_per_type.json"
    failure_analysis = val_dir / "failure_analysis.json"
    failure_tiers = val_dir / "failure_tiers.json"
    field_ablation = val_dir / "field_ablation_results.json"
    discriminator = val_dir / "discriminator_analysis.json"

    fig = plt.figure(figsize=(13.0, 10.0))
    layout = bind_figure_region(fig, (0.055, 0.05, 0.988, 0.965))
    top, bottom = layout.split_rows([1.0, 1.0], gap=0.14)

    # Top row: wider gap before barh panel c (col 2)
    top_weights = [1.55, 1.05, 1.10, 1.08, 1.55]
    top_cols = top.split_cols(top_weights, gap=[0.05, 0.10, 0.06, 0.06])
    # Bottom row: wider gap before g (col 1) barh; narrower j (col 4)
    bot_weights = [1.50, 1.15, 1.08, 1.08, 1.35]
    bottom_cols = bottom.split_cols(bot_weights, gap=[0.10, 0.06, 0.06, 0.10])

    _lbl_x = -0.12  # consistent x-offset for all panel labels

    ax_a = top_cols[0].add_axes(fig)
    _panel_a(ax_a, pseudobulk_summary, pseudobulk_per_type)
    add_panel_label(ax_a, "a", x=_lbl_x, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    ax_b = top_cols[1].add_axes(fig)
    _panel_b(ax_b, pseudobulk_summary, pseudobulk_per_type)
    add_panel_label(ax_b, "b", x=_lbl_x, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    ax_c = top_cols[2].add_axes(fig)
    _panel_c(ax_c, failure_analysis, failure_tiers)
    add_panel_label(ax_c, "c", x=_lbl_x, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    ax_d = top_cols[3].add_axes(fig)
    _panel_d(ax_d, failure_analysis, failure_tiers)
    add_panel_label(ax_d, "d", x=_lbl_x, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    ax_e = top_cols[4].add_axes(fig)
    _panel_e(ax_e, failure_analysis, failure_tiers)
    add_panel_label(ax_e, "e", x=_lbl_x, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    ax_f = bottom_cols[0].add_axes(fig)
    _panel_f(ax_f, field_ablation)
    add_panel_label(ax_f, "f", x=_lbl_x, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    ax_g = bottom_cols[1].add_axes(fig)
    _panel_g(ax_g, field_ablation)
    add_panel_label(ax_g, "g", x=_lbl_x, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    ax_h = bottom_cols[2].add_axes(fig)
    _panel_h(ax_h, discriminator)
    add_panel_label(ax_h, "h", x=_lbl_x, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    ax_i = bottom_cols[3].add_axes(fig)
    _panel_i(ax_i, discriminator)
    add_panel_label(ax_i, "i", x=_lbl_x, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    ax_j = bottom_cols[4].add_axes(fig)
    _panel_j(ax_j, discriminator)
    add_panel_label(ax_j, "j", x=_lbl_x, y=_LABEL_Y, fontsize=_LABEL_SIZE)

    # Harmonize title size and pad across all panels to avoid label overlap
    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i, ax_j]:
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=_TITLE_SIZE, fontweight="normal", pad=4)

    if save:
        stem = output_dir / "figS02_expression_diagnostics"
        save_with_vcd(fig, stem, dpi=dpi)
        logger.info("Saved %s", stem)

    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    logging.basicConfig(level=logging.INFO)
    plot_expression_diagnostics()
