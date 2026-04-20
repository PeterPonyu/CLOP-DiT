"""
fig08_composed.py — Article Figure 8 composite (Step 1 pilot of the
single-producer architecture migration, plan
`.omc/plans/single-producer-architecture-2026-04-20.md`).

Emits ONE canonical PDF (``fig08_composed.pdf``) combining the 9 panels
currently split between ``fig08a_downstream_validation.pdf`` (from
``fig17_downstream.py``) and ``fig08b_de_concordance.pdf`` (from
``fig18_de_concordance.py``) into a single ``plt.figure()`` with a single
3x3 gridspec so panel labels, fonts, and palettes are enforceable at the
module boundary rather than drifting across two source scripts.

Layout (3 rows x 3 cols, labels A-I):

    A  Downstream UMAP overlay             B  Sorted kNN mixing score        C  Clustering metrics
    D  Confusion matrix (gen. cells)       E  Per-type P/R/F1                F  Discriminator ROC
    G  Effect-size concordance (logFC)     H  Concordance across contrasts   I  Per-contrast summary

**Dual-publish:** this module is ADDITIVE — ``fig17_downstream.py`` and
``fig18_de_concordance.py`` continue producing ``fig08a_*.pdf`` and
``fig08b_*.pdf`` unchanged. The composite is a registered-but-not-yet-LaTeX-
referenced asset during the revision window.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from .article_composition import COMPOSITE_VCD_REGISTRY  # noqa: F401  (registry touched below)
from .direct_layout import bind_figure_region
from .explicit_positioning import add_axes_next_to, add_shared_legend_axes
from ._plot_helpers import plot_confusion_matrix, plot_roc_curve, plot_umap_overlay
from .fig17_downstream import _plot_classifier_metric_heatmap
from .style import (
    COLORS,
    FONT_ANNOTATION,
    FONT_DENSE_YTICK,
    FONT_HEATMAP_CELL,
    FONT_TICK,
    FONT_TICK_DENSE,
    FONT_TITLE,
    PANEL_LABEL_FONT_SIZE,
    abbreviate_cell_type,
    add_panel_label,
    apply_style,
    quality_color,
    save_with_vcd,
    set_dense_tick_labels,
    style_axes,
)
from src.utils.paths import FIG_DIR, RESULTS_DIR, load_thresholds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Register composite in the VCD cross-slice coverage registry at import time.
# The canonical registration lives in article_composition.py so the registry
# is populated as soon as `test_composite_vcd_coverage` imports it (without
# needing to import every fig0X_composed.py module first). This module keeps
# the idempotent re-assignment as belt-and-braces documentation for migration
# reviewers.
# ---------------------------------------------------------------------------
COMPOSITE_VCD_REGISTRY["fig08_composed.pdf"] = [
    "fig08a_downstream_validation.pdf",
    "fig08b_de_concordance.pdf",
]


# ---------------------------------------------------------------------------
# Style invariants — one source of truth for the composite figure.
# ---------------------------------------------------------------------------
#
# Horizontal alignment of panel labels:
#   rows A/B/C, D/E/F, G/H/I all share the same y-offset (top-edge alignment)
#   columns (A,D,G), (B,E,H), (C,F,I) share the same x-offset (left-edge alignment)
# Per-column x-offset — column 1 (A/D/G) sits before UMAP / confusion / scatter
# with wide numeric tick strings on G; column 2 (B/E/H) sits before bars /
# PRF heatmap / concordance heatmap; column 3 (C/F/I) sits before metrics /
# ROC / bars with shorter axes left-pad.
_PANEL_LABEL_X_COL1 = -0.12     # A, D, G — matches column-1 left inset of 0.030
_PANEL_LABEL_X_COL2 = -0.14     # B, E, H — matches column-2 left inset of ~0.09
_PANEL_LABEL_X_COL3 = -0.12     # C, F, I — matches column-3 left inset of 0.030
_PANEL_LABEL_Y_ROW1 = 1.04
_PANEL_LABEL_Y_ROW2 = 1.04
_PANEL_LABEL_Y_ROW3 = 1.04

_viz_thresh = load_thresholds().get("visualization", {})
_DOWNSTREAM_BANDS = tuple(_viz_thresh.get("downstream_bands", [0.3, 0.15]))


# ---------------------------------------------------------------------------
# Data loaders (reuse the logic from fig17_downstream.generate_downstream_panels
# and results_visualizer.plot_downstream_panels verbatim so the composite reads
# the same cached artefacts).
# ---------------------------------------------------------------------------

def _load_clustering_data(ds_dir: Path) -> Optional[Dict]:
    clust_path = ds_dir / "clustering_alignment.json"
    if not clust_path.exists():
        return None
    with open(clust_path) as f:
        clust_data = json.load(f)
    for key, fname in [
        ("_umap_coords", "clustering_umap_coords.npy"),
        ("_source", "clustering_source.npy"),
        ("_cell_type", "clustering_cell_type.npy"),
    ]:
        arr_path = ds_dir / fname
        if arr_path.exists():
            clust_data[key] = np.load(arr_path, allow_pickle=True)
    return clust_data


def _load_classifier_data(ds_dir: Path) -> Optional[Dict]:
    classif_path = ds_dir / "classifier_alignment.json"
    if not classif_path.exists():
        return None
    with open(classif_path) as f:
        classif_data = json.load(f)
    if "confusion_matrix" in classif_data:
        classif_data["_confusion_matrix"] = classif_data["confusion_matrix"]
    if "disc_proba" in classif_data:
        classif_data["_disc_proba"] = classif_data["disc_proba"]
        classif_data["_disc_y"] = classif_data["disc_y"]
    return classif_data


def _load_de_data(ds_dir: Path) -> Optional[Dict]:
    de_path = ds_dir / "de_concordance.json"
    if not de_path.exists():
        return None
    with open(de_path) as f:
        de_data = json.load(f)
    for cname in de_data:
        for key in ["_real_logfc", "_gen_logfc", "_shared_genes"]:
            fname = f"de_{cname}_{key.lstrip('_')}.npy"
            arr_path = ds_dir / fname
            if arr_path.exists():
                de_data[cname][key] = np.load(arr_path, allow_pickle=True).tolist()
    return de_data


# ---------------------------------------------------------------------------
# Row renderers — each row takes pre-created axes and draws onto them.
# ---------------------------------------------------------------------------

def _draw_clustering_row(
    fig: plt.Figure,
    ax_a: plt.Axes,
    ax_b: plt.Axes,
    ax_c: plt.Axes,
    clustering_data: Dict,
) -> None:
    """Top row: UMAP overlay (A), sorted kNN mixing score (B), clustering metrics (C)."""
    add_panel_label(ax_a, "a", x=_PANEL_LABEL_X_COL1, y=_PANEL_LABEL_Y_ROW1)
    umap_coords = clustering_data.get("_umap_coords")
    if umap_coords is not None:
        umap_coords = np.asarray(umap_coords)
        source = np.asarray(clustering_data.get("_source"))
        cell_type = np.asarray(clustering_data.get("_cell_type"))
        plot_umap_overlay(
            ax_a, umap_coords, source, cell_type,
            legend_loc="lower left", legend_fontsize=9,
        )
    else:
        ax_a.text(0.5, 0.5, "No UMAP data", ha="center", va="center",
                  transform=ax_a.transAxes)

    add_panel_label(ax_b, "b", x=_PANEL_LABEL_X_COL2, y=_PANEL_LABEL_Y_ROW1)
    mixing = clustering_data.get("per_type_mixing", {})
    if mixing:
        sorted_types = sorted(mixing.keys(), key=lambda k: mixing[k])
        vals = [mixing[t] for t in sorted_types]
        short_names = [abbreviate_cell_type(t, 20) for t in sorted_types]
        bar_colors = [quality_color(v, _DOWNSTREAM_BANDS) for v in vals]
        y_pos = np.arange(len(sorted_types))
        ax_b.barh(y_pos, vals, color=bar_colors, height=0.7,
                  edgecolor="white", linewidth=0.5)
        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(short_names, fontsize=FONT_DENSE_YTICK)
        set_dense_tick_labels(ax_b, axis="y", max_labels=12,
                              fontsize=FONT_DENSE_YTICK, rotation=0)
        ax_b.axvline(x=clustering_data.get("mean_mixing_score", 0),
                     color=COLORS["bad"], linestyle="--", alpha=0.7, linewidth=1.5,
                     label=f"mean={clustering_data.get('mean_mixing_score', 0):.3f}")
        ax_b.set_xlim(0, max(max(vals) * 1.1, 0.5))
        ax_b.legend(fontsize=9, frameon=False, loc="lower right")
        style_axes(ax_b, "bar", title="Sorted kNN Mixing Score",
                   xlabel="Fraction Real Neighbours")
    else:
        ax_b.text(0.5, 0.5, "No mixing data", ha="center", va="center",
                  transform=ax_b.transAxes)

    add_panel_label(ax_c, "c", x=_PANEL_LABEL_X_COL3, y=_PANEL_LABEL_Y_ROW1)
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
    ax_c.barh(y_pos, [1.0] * len(summary_items), height=0.6,
              color=COLORS["bg_gauge"], edgecolor="none", zorder=1)
    bar_colors = [quality_color(v, t) for v, t in zip(metric_vals, thresholds)]
    ax_c.barh(y_pos, metric_vals, height=0.6, color=bar_colors,
              edgecolor="white", linewidth=0.8, zorder=2)
    for i, val in enumerate(metric_vals):
        ax_c.text(min(val + 0.03, 0.98), i, f"{val:.3f}",
                  va="center", fontsize=FONT_ANNOTATION, zorder=3)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(metric_names, fontsize=FONT_TICK)
    ax_c.set_xlim(0, 1.15)
    ax_c.invert_yaxis()
    n_cl = clustering_data.get("n_leiden_clusters", "?")
    ax_c.set_xlabel(f"Leiden clusters: {n_cl}", fontsize=FONT_ANNOTATION)
    ax_c.tick_params(axis="x", labelbottom=False, bottom=False)
    style_axes(ax_c, "bar", title="Clustering Metrics")


def _draw_classifier_row(
    fig: plt.Figure,
    ax_d: plt.Axes,
    ax_e: plt.Axes,
    ax_f: plt.Axes,
    classifier_data: Dict,
) -> None:
    """Middle row: confusion matrix (D), per-type P/R/F1 (E), discriminator ROC (F)."""
    cm = classifier_data.get("_confusion_matrix")
    class_names = classifier_data.get("class_names", [])
    disc_auc = classifier_data.get("discriminator_auc", 0)

    add_panel_label(ax_d, "d", x=_PANEL_LABEL_X_COL1, y=_PANEL_LABEL_Y_ROW2)
    if cm is not None:
        cm = np.array(cm)
        plot_confusion_matrix(
            ax_d, cm, class_names or None,
            title="Confusion Matrix", colorbar_pad=0.18,
        )
        if cm.shape[0] > 30:
            tick_step = max(4, cm.shape[0] // 3)
            tick_positions = list(range(0, cm.shape[0], tick_step))
            ax_d.set_xticks(tick_positions)
            ax_d.set_yticks(tick_positions)
            ax_d.set_xticklabels([str(i) for i in tick_positions],
                                 fontsize=FONT_ANNOTATION, rotation=45, ha="right")
            ax_d.set_yticklabels([str(i) for i in tick_positions],
                                 fontsize=FONT_ANNOTATION)
    else:
        ax_d.text(0.5, 0.5, "No confusion matrix", ha="center", va="center",
                  transform=ax_d.transAxes)

    add_panel_label(ax_e, "e", x=_PANEL_LABEL_X_COL2, y=_PANEL_LABEL_Y_ROW2)
    per_type_acc = classifier_data.get("per_type_accuracy", {})
    gen_acc = classifier_data.get("gen_accuracy", 0)
    note_ax = None
    if per_type_acc and cm is not None:
        summary = _plot_classifier_metric_heatmap(
            fig, ax_e, np.array(cm),
            class_names or [f"C{i}" for i in range(np.array(cm).shape[0])],
            max_rows=14,
        )
        note_pos = ax_e.get_position()
        note_ax = add_shared_legend_axes(
            fig,
            (note_pos.x0, note_pos.y0 - 0.06, note_pos.width, 0.04),
        )
        note_ax.text(
            0.50, 0.5,
            f"Acc {gen_acc:.3f}  |  Median F1 {np.median(summary['f1']):.3f}",
            transform=note_ax.transAxes, ha="center", va="center",
            fontsize=FONT_ANNOTATION, color=COLORS["neutral"],
        )
    else:
        ax_e.text(0.5, 0.5, "No per-type data", ha="center", va="center",
                  transform=ax_e.transAxes)

    add_panel_label(ax_f, "f", x=_PANEL_LABEL_X_COL3, y=_PANEL_LABEL_Y_ROW2)
    disc_proba = classifier_data.get("_disc_proba")
    disc_y = classifier_data.get("_disc_y")
    if disc_proba is not None and disc_y is not None:
        plot_roc_curve(ax_f, disc_y, disc_proba, auc_value=disc_auc)
    else:
        ax_f.text(0.5, 0.5, "No discriminator data", ha="center", va="center",
                  transform=ax_f.transAxes)


def _abbrev_contrast(name: str, max_len: int = 18) -> str:
    parts = name.split("_vs_")
    if len(parts) == 2:
        a = abbreviate_cell_type(parts[0].replace("_", " "), max_len=max_len)
        b = abbreviate_cell_type(parts[1].replace("_", " "), max_len=max_len)
        return f"{a}\nvs {b}"
    return abbreviate_cell_type(name.replace("_", " "), max_len=max_len * 2)


def _draw_de_row(
    fig: plt.Figure,
    ax_g: plt.Axes,
    ax_h: plt.Axes,
    ax_i: plt.Axes,
    de_data: Dict,
) -> None:
    """Bottom row: effect-size concordance scatter (G), concordance heatmap (H),
    per-contrast summary bars (I). Mirrors fig18_de_concordance.py logic."""
    from matplotlib.ticker import MaxNLocator

    contrasts = list(de_data.keys())
    n_contrasts = len(contrasts)

    # Panel G ──────────────────────────────────────────────────────────
    add_panel_label(ax_g, "g", x=_PANEL_LABEL_X_COL1, y=_PANEL_LABEL_Y_ROW3)
    if not contrasts:
        ax_g.text(0.5, 0.5, "No DE data", ha="center", va="center",
                  transform=ax_g.transAxes)
        ax_h.text(0.5, 0.5, "No DE data", ha="center", va="center",
                  transform=ax_h.transAxes)
        ax_i.text(0.5, 0.5, "No DE data", ha="center", va="center",
                  transform=ax_i.transAxes)
        add_panel_label(ax_h, "h", x=_PANEL_LABEL_X_COL2, y=_PANEL_LABEL_Y_ROW3)
        add_panel_label(ax_i, "i", x=_PANEL_LABEL_X_COL3, y=_PANEL_LABEL_Y_ROW3)
        return

    first_key = contrasts[0]
    first = de_data[first_key]
    real_logfc = np.array(first.get("_real_logfc", []))
    gen_logfc = np.array(first.get("_gen_logfc", []))
    real_padj = np.array(first.get("_real_padj", []), dtype=float)
    gen_padj = np.array(first.get("_gen_padj", []), dtype=float)
    shared_genes = first.get("_shared_genes", [])

    if len(real_logfc) > 0 and len(gen_logfc) > 0:
        effect_size = (np.abs(real_logfc) + np.abs(gen_logfc)) / 2.0
        effect_scale = (np.percentile(effect_size, 95)
                        if len(effect_size) > 4 else effect_size.max())
        effect_scale = effect_scale if effect_scale > 0 else 1.0
        marker_sizes = 12 + 140 * np.clip(effect_size / effect_scale, 0, 1.5)

        if len(real_padj) == len(real_logfc) and len(gen_padj) == len(gen_logfc):
            significance = -np.log10(np.clip(np.minimum(real_padj, gen_padj),
                                             1e-50, 1.0))
        else:
            significance = effect_size

        ax_g.hexbin(real_logfc, gen_logfc, gridsize=30, cmap="Blues",
                    alpha=0.25, mincnt=1, linewidths=0, zorder=1)
        sc = ax_g.scatter(
            real_logfc, gen_logfc,
            c=significance, cmap="magma",
            s=marker_sizes, alpha=0.70, edgecolors="none", zorder=3,
        )
        discordant = np.sign(real_logfc) != np.sign(gen_logfc)
        if discordant.any():
            strong_discordant = discordant & (
                effect_size >= np.percentile(effect_size, 85))
            ax_g.scatter(
                real_logfc[strong_discordant], gen_logfc[strong_discordant],
                s=marker_sizes[strong_discordant] * 1.2,
                facecolors="none", edgecolors=COLORS["bad"],
                linewidths=0.8, label="sign disagreement", zorder=4,
            )
        lo = min(real_logfc.min(), gen_logfc.min()) * 1.1
        hi = max(real_logfc.max(), gen_logfc.max()) * 1.1
        ax_g.plot([lo, hi], [lo, hi], color=COLORS["bad"], linestyle="--",
                  lw=1.5, alpha=0.7, label="y = x")
        ax_g.axhline(0, color="#666", linestyle=":", linewidth=1.0, alpha=0.6)
        ax_g.axvline(0, color="#666", linestyle=":", linewidth=1.0, alpha=0.6)

        # Gene-name callouts (top residuals, staggered)
        residuals = np.abs(gen_logfc - real_logfc) * np.maximum(effect_size, 1e-6)
        n_label = min(6, len(residuals))
        top_idx = np.argsort(residuals)[-n_label:]
        placed_points: List[tuple] = []
        for _k, idx in enumerate(top_idx[np.argsort(residuals[top_idx])[::-1]]):
            if idx < len(shared_genes):
                x_pt = real_logfc[idx]
                y_pt = gen_logfc[idx]
                if any(abs(x_pt - px) < 0.35 and abs(y_pt - py) < 0.35
                       for px, py in placed_points):
                    continue
                ox = (28 + 6 * _k) * (-1 if x_pt > np.median(real_logfc) else 1)
                oy = (18 + 5 * (_k % 3)) * (-1 if y_pt > np.median(gen_logfc) else 1)
                ax_g.annotate(
                    shared_genes[idx],
                    (x_pt, y_pt),
                    fontsize=FONT_ANNOTATION,
                    xytext=(ox, oy), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="#666",
                                    connectionstyle="arc3,rad=0.15"),
                    color=COLORS["annotation_dark"], ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="none", ec="none"),
                )
                placed_points.append((x_pt, y_pt))

        pearson_r, pearson_p = scipy_stats.pearsonr(real_logfc, gen_logfc)
        sign_agreement = first.get("top_k_sign_agreement", 0)

        def _fmt_p(p: float) -> str:
            return f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"

        ax_g.text(
            0.97, 0.97,
            f"r = {pearson_r:.3f}  (p {_fmt_p(pearson_p)})\n"
            f"Sign agr. = {sign_agreement:.3f}",
            transform=ax_g.transAxes, ha="right", va="top",
            fontsize=FONT_ANNOTATION, color=COLORS["neutral"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.85, edgecolor="none"),
        )

        cax_g = add_axes_next_to(
            fig, ax_g, side="right", width=0.010,
            height=ax_g.get_position().height * 0.44,
            pad=0.014, align="bottom", y_offset=0.02,
        )
        cbar_g = fig.colorbar(sc, cax=cax_g)
        cbar_g.set_label("")
        cbar_g.ax.tick_params(labelsize=FONT_HEATMAP_CELL)
        ax_g.legend(fontsize=FONT_ANNOTATION, frameon=False,
                    loc="lower left", bbox_to_anchor=(0.01, 0.01))

    parts = first_key.split("_vs_")
    if len(parts) == 2:
        a_short = abbreviate_cell_type(parts[0].replace("_", " "), max_len=20)
        b_short = abbreviate_cell_type(parts[1].replace("_", " "), max_len=20)
        sub_line = f"{a_short}  vs  {b_short}"
    else:
        sub_line = abbreviate_cell_type(first_key.replace("_", " "), max_len=38)
    style_axes(ax_g, "scatter", xlabel="Real logFC", ylabel="Generated logFC")
    ax_g.xaxis.labelpad = 10
    ax_g.set_title(f"Effect-Size Concordance\n{sub_line}",
                   fontsize=FONT_TITLE - 1, pad=4)
    ax_g.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax_g.tick_params(labelsize=FONT_TICK)

    # Panel H ──────────────────────────────────────────────────────────
    add_panel_label(ax_h, "h", x=_PANEL_LABEL_X_COL2, y=_PANEL_LABEL_Y_ROW3)
    metric_names = ["Pears. r", "Spear. \u03c1", "Jacc.@50", "Sign agr."]
    metric_tick_labels = ["Pears.\nr", "Spear.\n\u03c1", "Jacc.\n50", "Sign\nagr."]
    metric_keys = ["logfc_pearson_r", "logfc_spearman_rho",
                   "top_k_jaccard", "top_k_sign_agreement"]
    heatmap_data = np.zeros((n_contrasts, len(metric_names)))
    contrast_labels: List[str] = []
    for i, cname in enumerate(contrasts):
        cd = de_data[cname]
        for j, mk in enumerate(metric_keys):
            heatmap_data[i, j] = cd.get(mk, 0)
        contrast_labels.append(_abbrev_contrast(cname, max_len=12))

    im_h = ax_h.imshow(heatmap_data, cmap="PiYG", aspect="auto", vmin=0, vmax=1)
    ax_h.set_xticks(range(len(metric_names)))
    ax_h.set_xticklabels(metric_tick_labels, fontsize=FONT_TICK_DENSE,
                         rotation=0, ha="center")
    ax_h.set_yticks(range(n_contrasts))
    ax_h.set_yticklabels(contrast_labels, fontsize=FONT_TICK_DENSE)
    cax_h = add_axes_next_to(
        fig, ax_h, side="right", width=0.010,
        height=ax_h.get_position().height * 0.48,
        pad=0.012, align="bottom", y_offset=0.02,
    )
    cbar_h = fig.colorbar(im_h, cax=cax_h)
    cbar_h.ax.tick_params(labelsize=FONT_HEATMAP_CELL)
    style_axes(ax_h, "heatmap", title="Concordance Across Contrasts")
    ax_h.set_title("Concordance Across Contrasts", x=0.60, pad=6)

    # Panel I ──────────────────────────────────────────────────────────
    add_panel_label(ax_i, "i", x=_PANEL_LABEL_X_COL3, y=_PANEL_LABEL_Y_ROW3)
    x = np.arange(n_contrasts)
    n_metrics = len(metric_names)
    w = 0.72 / n_metrics
    bar_colors = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A"]
    for j, (mname, mk) in enumerate(zip(metric_names, metric_keys)):
        vals = [de_data[c].get(mk, 0) for c in contrasts]
        offset = (j - n_metrics / 2 + 0.5) * w
        ax_i.bar(x + offset, vals, w, label=mname,
                 color=bar_colors[j], alpha=0.85, edgecolor="white")
    xs_labels = [_abbrev_contrast(c, max_len=8) for c in contrasts]
    ax_i.set_xticks(x)
    ax_i.set_xticklabels(xs_labels, fontsize=FONT_ANNOTATION,
                         rotation=0, ha="center", multialignment="center")
    ax_i.set_ylim(0, 1.12)
    ax_i.legend(fontsize=FONT_ANNOTATION, ncol=2,
                loc="upper right", frameon=False)
    style_axes(ax_i, "bar", title="Per-Contrast Summary", ylabel="Score")
    ax_i.tick_params(labelsize=FONT_TICK)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_fig08_composed(
    downstream_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    dpi: int = 300,
    save: bool = True,
) -> Optional[Path]:
    """Render the 3x3 composite Fig 8 and save to ``fig08_composed.{pdf,png}``.

    Parameters
    ----------
    downstream_dir : directory with cached downstream JSON/NPY artefacts.
                     Defaults to ``results/downstream``.
    output_dir     : directory to save the composite PDF.
                     Defaults to ``results/figures``.
    dpi            : rasterised preview DPI (PDF is vector).
    save           : if False, return the figure without writing to disk.

    Returns
    -------
    Path to the saved PDF, or ``None`` if the downstream cache is missing.
    """
    apply_style()

    ds_dir = Path(downstream_dir) if downstream_dir else RESULTS_DIR / "downstream"
    out_dir = Path(output_dir) if output_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    clustering_data = _load_clustering_data(ds_dir)
    classifier_data = _load_classifier_data(ds_dir)
    de_data = _load_de_data(ds_dir)

    if clustering_data is None and classifier_data is None and de_data is None:
        logger.warning("No downstream data found under %s — skipping fig08_composed", ds_dir)
        return None

    # 3 rows x 3 cols, same column weights as the existing fig17/fig18 layouts
    # so per-panel aspect ratios match the legacy slices as closely as
    # possible (AE-tolerance calibration, plan §5.6 bullet 5).
    fig = plt.figure(figsize=(16.0, 15.6))
    layout = bind_figure_region(fig, (0.04, 0.04, 0.98, 0.96))
    row1, row2, row3 = layout.split_rows([1.00, 1.00, 1.00], hspace=0.55)

    # Top row weights mirror fig17 merged P+Q top row.
    row1_rects = row1.split_cols([1.10, 1.34, 0.82], gap=[0.060, 0.050])
    # Middle row weights mirror fig17 merged P+Q bottom row.
    row2_rects = row2.split_cols([1.10, 1.34, 0.82], gap=[0.060, 0.050])
    # Bottom row weights mirror fig18 de_concordance.
    row3_rects = row3.split_cols([1.26, 0.92, 0.98], gap=[0.060, 0.055])

    # Uniform column-1 left inset so the shared _PANEL_LABEL_X_COL1 offset
    # keeps labels (A, D, G) horizontally aligned AND clear of axis ticks.
    ax_a = row1_rects[0].inset(left=0.030, right=0.010).add_axes(fig)
    ax_b = row1_rects[1].inset(left=0.090, right=0.040).add_axes(fig)
    ax_c = row1_rects[2].inset(left=0.030, right=0.010).add_axes(fig)

    ax_d = row2_rects[0].inset(left=0.030, right=0.010).add_axes(fig)
    ax_e = row2_rects[1].inset(left=0.060, right=0.070).add_axes(fig)
    ax_f = row2_rects[2].inset(left=0.030, right=0.010).add_axes(fig)

    ax_g = row3_rects[0].inset(left=0.030, right=0.050).add_axes(fig)
    ax_h = row3_rects[1].inset(left=0.090, right=0.020).add_axes(fig)
    ax_i = row3_rects[2].inset(left=0.030, right=0.028).add_axes(fig)

    if clustering_data is not None:
        _draw_clustering_row(fig, ax_a, ax_b, ax_c, clustering_data)
    else:
        for ax, lab, x in [(ax_a, "a", _PANEL_LABEL_X_COL1),
                            (ax_b, "b", _PANEL_LABEL_X_COL2),
                            (ax_c, "c", _PANEL_LABEL_X_COL3)]:
            add_panel_label(ax, lab, x=x, y=_PANEL_LABEL_Y_ROW1)
            ax.text(0.5, 0.5, "No clustering data", ha="center",
                    va="center", transform=ax.transAxes)

    if classifier_data is not None:
        _draw_classifier_row(fig, ax_d, ax_e, ax_f, classifier_data)
    else:
        for ax, lab, x in [(ax_d, "d", _PANEL_LABEL_X_COL1),
                            (ax_e, "e", _PANEL_LABEL_X_COL2),
                            (ax_f, "f", _PANEL_LABEL_X_COL3)]:
            add_panel_label(ax, lab, x=x, y=_PANEL_LABEL_Y_ROW2)
            ax.text(0.5, 0.5, "No classifier data", ha="center",
                    va="center", transform=ax.transAxes)

    if de_data is not None:
        _draw_de_row(fig, ax_g, ax_h, ax_i, de_data)
    else:
        for ax, lab, x in [(ax_g, "g", _PANEL_LABEL_X_COL1),
                            (ax_h, "h", _PANEL_LABEL_X_COL2),
                            (ax_i, "i", _PANEL_LABEL_X_COL3)]:
            add_panel_label(ax, lab, x=x, y=_PANEL_LABEL_Y_ROW3)
            ax.text(0.5, 0.5, "No DE data", ha="center",
                    va="center", transform=ax.transAxes)

    if save:
        path = save_with_vcd(
            fig, out_dir / "fig08_composed.png", dpi,
            layout_rect=(0.02, 0.01, 0.98, 0.96),
        )
        logger.info("Saved fig08_composed \u2192 %s", path)
        plt.close(fig)
        return out_dir / "fig08_composed.pdf"
    return None


# Direct-run entry point so ``python -m src.visualization.fig08_composed`` works.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    plot_fig08_composed()


__all__ = ["plot_fig08_composed"]
