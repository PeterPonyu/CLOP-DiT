"""
fig04_composed.py — Article Figure 4 composite (Step 3 of the single-producer
architecture migration, plan
`.omc/plans/single-producer-architecture-2026-04-20.md`).

Emits ONE canonical PDF (``fig04_composed.pdf``) combining the 8 panels
currently split between ``fig04a_marker_genes.pdf`` (from
``fig08_markers.py``) and ``fig04b_expression_correlation.pdf`` (from
``fig09_expression_corr.py``) into a single ``plt.figure()`` with a single
4-row x 2-col gridspec so panel labels, fonts, and palettes are enforceable
at the module boundary rather than drifting across two source scripts.

Layout (4 rows x 2 cols, labels a-h):

    Row 1 (marker bars / density scatter):
      a  Grouped marker-gene bars (real vs gen)      e  Per-gene density scatter
    Row 2 (dual heatmap / per-type lollipop):
      b  Dual expression heatmap                     f  Per-type expression fidelity
    Row 3 (diff heatmap / marker bars):
      c  Delta expression (Gen - Real) heatmap       g  Marker gene expression (grouped)
    Row 4 (fold-change / residual histogram):
      d  Marker log2 fold change                     h  Per-gene residual distribution

Note: despite the source-file naming (``fig08_markers.py`` /
``fig09_expression_corr.py``), these two producers emit ARTICLE Figure 4's
panels A and B. Source-file numbering diverged from article numbering
pre-revision (plan §3 Step 3); this composite preserves that convention
rather than renaming mid-cycle.

**Dual-publish:** this module is ADDITIVE — ``fig08_markers.py`` and
``fig09_expression_corr.py`` continue producing ``fig04a_*.pdf`` and
``fig04b_*.pdf`` unchanged. The composite is a registered-but-not-yet-LaTeX-
referenced asset during the revision window.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from .article_composition import COMPOSITE_VCD_REGISTRY  # noqa: F401
from .direct_layout import bind_figure_region
from .explicit_positioning import add_axes_next_to
from .fig08_markers import MARKER_PANEL_GENES, MARKER_PANEL_TYPES
from .style import (
    COLORS,
    FONT_DENSE_YTICK,
    PANEL_OFFSET_STD,
    PANEL_OFFSET_WIDE,
    abbreviate_cell_type,
    add_colorbar_safe,
    add_panel_label,
    apply_style,
    save_with_vcd,
)
from src.utils.paths import CACHE_DIR, FIG_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Register composite in the VCD cross-slice coverage registry at import time.
# Canonical registration lives in article_composition.py; the idempotent
# re-assignment here is belt-and-braces documentation for migration reviewers.
# ---------------------------------------------------------------------------
COMPOSITE_VCD_REGISTRY["fig04_composed.pdf"] = [
    "fig04a_marker_genes.pdf",
    "fig04b_expression_correlation.pdf",
]


# ---------------------------------------------------------------------------
# Data loaders — mirror the logic from fig08_markers.plot_marker_gene_comparison
# and fig09_expression_corr.plot_expression_correlation so the composite reads
# the same cached artefacts the per-slice producers read.
# ---------------------------------------------------------------------------

def _load_fig04_inputs(
    real_expr_path: Path,
    gen_expr_path: Path,
    real_labels_path: Path,
    gen_labels_path: Path,
    gene_names_path: Path,
    metrics_path: Path,
    cache_dir: Path,
) -> Optional[Dict]:
    """Load every NPY / JSON artefact needed by both rows.

    Returns a dict with keys: real, gen, real_labels, gen_labels, gene_names,
    metrics, type_names. Returns ``None`` if the minimum required artefacts
    are missing.
    """
    required = [real_expr_path, gen_expr_path, gene_names_path, metrics_path]
    if not all(p.exists() for p in required):
        logger.info("Expression data not found — skipping fig04_composed")
        return None

    real = np.load(real_expr_path)
    gen = np.load(gen_expr_path)
    real_labels = np.load(real_labels_path) if real_labels_path.exists() else None
    gen_labels = np.load(gen_labels_path) if gen_labels_path.exists() else None
    with open(gene_names_path) as f:
        gene_names = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)

    type_names: Dict[int, str] = {}
    cap_path = cache_dir / "text_captions_deduplicated.json"
    if cap_path.exists():
        try:
            with open(cap_path) as _cf:
                raw_caps = json.load(_cf)
            for k, v in raw_caps.items():
                name = v.split(" are ")[0] if " are " in v else v[:50]
                type_names[int(k)] = name
        except Exception:
            type_names = {}

    return {
        "real": real,
        "gen": gen,
        "real_labels": real_labels,
        "gen_labels": gen_labels,
        "gene_names": gene_names,
        "metrics": metrics,
        "type_names": type_names,
    }


# ---------------------------------------------------------------------------
# Row renderers (marker-gene side: panels a, b, c, d). Mirrors
# fig08_markers.plot_marker_gene_comparison, adapted to pre-created axes.
# ---------------------------------------------------------------------------

def _draw_marker_panels(
    fig: plt.Figure,
    ax_a: plt.Axes,
    ax_b: plt.Axes,
    ax_c: plt.Axes,
    ax_d: plt.Axes,
    data: Dict,
) -> None:
    """Left column: grouped marker bars (a), dual heatmap (b), delta heatmap (c),
    log2 fold-change (d). Mirrors fig08_markers.plot_marker_gene_comparison."""
    real = data["real"]
    gen = data["gen"]
    real_labels = data["real_labels"]
    gen_labels = data["gen_labels"]
    gene_names = data["gene_names"]
    type_names = data["type_names"]

    all_marker_genes: List[str] = []
    for _, genes in MARKER_PANEL_GENES.items():
        for gene in genes:
            if gene in gene_names:
                all_marker_genes.append(gene)
    if len(all_marker_genes) < 2:
        for ax, lab in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
            add_panel_label(ax, lab, x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
            ax.text(0.5, 0.5, "Too few markers found",
                    ha="center", va="center", transform=ax.transAxes)
        return

    gene_idx = [gene_names.index(g) for g in all_marker_genes]
    real_marker_means = real[:, gene_idx].mean(axis=0)
    gen_marker_means = gen[:, gene_idx].mean(axis=0)

    selected_type_ids: List[int] = []
    selected_type_names: List[str] = []
    if real_labels is not None:
        for target_name in MARKER_PANEL_TYPES:
            for tid, tname in type_names.items():
                if target_name.lower() in tname.lower() and tid in np.unique(real_labels):
                    selected_type_ids.append(tid)
                    selected_type_names.append(abbreviate_cell_type(tname, 24))
                    break
    if len(selected_type_ids) < 3 and real_labels is not None:
        unique, counts = np.unique(real_labels, return_counts=True)
        top4 = unique[np.argsort(counts)[-4:]]
        selected_type_ids = top4.tolist()
        selected_type_names = [
            abbreviate_cell_type(type_names.get(int(t), f"Type_{t}"), 24)
            for t in selected_type_ids
        ]

    n_markers = len(all_marker_genes)
    n_sel_types = len(selected_type_ids)

    # ── Panel a: Grouped horizontal bar chart ────────────────────────────
    add_panel_label(ax_a, "a", x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
    r_means = np.array([real[:, gi].mean() for gi in gene_idx])
    g_means = np.array([gen[:, gi].mean() for gi in gene_idx])
    r_stds = np.array([real[:, gi].std() for gi in gene_idx])
    g_stds = np.array([gen[:, gi].std() for gi in gene_idx])
    n_real = real.shape[0]
    n_gen = gen.shape[0]
    y_pos = np.arange(n_markers)
    width = 0.35
    ax_a.barh(
        y_pos - width / 2, r_means, width,
        xerr=r_stds / np.sqrt(n_real),
        label="Real", color=COLORS["real"], alpha=0.85,
        edgecolor="white", capsize=3, error_kw=dict(lw=0.8),
    )
    ax_a.barh(
        y_pos + width / 2, g_means, width,
        xerr=g_stds / np.sqrt(n_gen),
        label="Gen", color=COLORS["generated"], alpha=0.85,
        edgecolor="white", capsize=3, error_kw=dict(lw=0.8),
    )
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([f"{g[:12]}" for g in all_marker_genes], fontsize=10)
    ax_a.set_xlabel("Mean Expression", fontsize=11)
    ax_a.set_title("Marker Mean Expression", fontsize=12)
    ax_a.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.35)
    ax_a.set_axisbelow(True)
    ax_a.legend(
        fontsize=9, loc="lower right", frameon=False,
        handlelength=1.3, columnspacing=0.8,
    )

    if n_sel_types < 2 or real_labels is None or gen_labels is None:
        # Fallback — we still need to label b/c/d so panel labels exist.
        for ax, lab in [(ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
            add_panel_label(ax, lab, x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
            ax.text(0.5, 0.5, "Per-type labels not available",
                    ha="center", va="center", transform=ax.transAxes)
        return

    # ── Build per-type heatmap data ──────────────────────────────────────
    real_heat = np.zeros((n_sel_types, n_markers))
    gen_heat = np.zeros((n_sel_types, n_markers))
    for i, tid in enumerate(selected_type_ids):
        r_mask = real_labels == tid
        g_mask = gen_labels == tid
        for j, gi in enumerate(gene_idx):
            if r_mask.any():
                real_heat[i, j] = real[r_mask][:, gi].mean()
            if g_mask.any():
                gen_heat[i, j] = gen[g_mask][:, gi].mean()

    # ── Panel b: Dual heatmap (real | gen) ───────────────────────────────
    add_panel_label(ax_b, "b", x=PANEL_OFFSET_WIDE[0], y=PANEL_OFFSET_WIDE[1])
    combined = np.hstack([real_heat, gen_heat])
    vmin, vmax = combined.min(), combined.max()
    gap_col = np.full((n_sel_types, 1), np.nan)
    display = np.hstack([real_heat, gap_col, gen_heat])

    cmap_b = mcolors.LinearSegmentedColormap.from_list(
        "expr_heat",
        ["#fff3e0", "#ffcc80", "#ff9800", "#e65100", "#bf360c"],
        N=256,
    )
    im_b = ax_b.imshow(display, cmap=cmap_b, aspect="auto", vmin=vmin, vmax=vmax)
    ax_b.set_yticks(range(n_sel_types))
    ax_b.set_yticklabels(
        [abbreviate_cell_type(n, 16) for n in selected_type_names], fontsize=10,
    )
    xtick_pos = list(range(n_markers)) + list(range(n_markers + 1, 2 * n_markers + 1))
    xtick_short = [g[:12] for g in all_marker_genes]
    xtick_labels = xtick_short + xtick_short
    ax_b.set_xticks(xtick_pos)
    ax_b.set_xticklabels(xtick_labels, fontsize=10, rotation=90, ha="center")
    ax_b.axvspan(n_markers - 0.5, n_markers + 0.5, color="#f3f3f3", zorder=0)
    ax_b.axvline(x=n_markers, color="#666", linewidth=1.6, linestyle="-")
    ax_b.axvline(x=n_markers - 0.5, color="black", linewidth=1.5, zorder=5)
    ax_b.set_title("Per-Type Expression Heatmap", fontsize=12)
    ax_b.text(
        n_markers / 2 - 0.5, -0.34, "Real",
        transform=ax_b.get_xaxis_transform(),
        ha="center", va="top", fontsize=10, color=COLORS["real"],
    )
    ax_b.text(
        n_markers + 0.5 + n_markers / 2 - 0.5, -0.34, "Generated",
        transform=ax_b.get_xaxis_transform(),
        ha="center", va="top", fontsize=11, color=COLORS["generated"],
    )
    add_colorbar_safe(im_b, ax=ax_b, label="Expr.", shrink=0.6, pad=0.05)

    # ── Panel c: Delta heatmap (gen - real) ──────────────────────────────
    add_panel_label(ax_c, "c", x=PANEL_OFFSET_WIDE[0], y=PANEL_OFFSET_WIDE[1])
    diff = gen_heat - real_heat
    max_abs = max(abs(diff.min()), abs(diff.max()), 0.01)
    im_c = ax_c.imshow(diff, cmap="RdBu_r", aspect="auto", vmin=-max_abs, vmax=max_abs)
    im_c.set_rasterized(True)
    ax_c.set_yticks(range(n_sel_types))
    ax_c.set_yticklabels(
        [abbreviate_cell_type(n, 16) for n in selected_type_names], fontsize=10,
    )
    ax_c.set_xticks(range(n_markers))
    ax_c.set_xticklabels(all_marker_genes, fontsize=11, rotation=90, ha="center")
    ax_c.set_title("\u0394 Expression (Gen \u2212 Real)", fontsize=11)
    cax_c = add_axes_next_to(
        fig, ax_c, side="right", width=0.010,
        height=ax_c.get_position().height * 0.42,
        pad=0.014, align="bottom", y_offset=0.012,
    )
    cbar_c = fig.colorbar(im_c, cax=cax_c)
    cbar_c.set_label("")
    cbar_c.ax.set_title("\u0394", fontsize=10, pad=2)
    cbar_c.ax.tick_params(labelsize=8)
    for i in range(n_sel_types):
        for j in range(n_markers):
            if abs(diff[i, j]) > max_abs * 0.3:
                txt_color = "white" if abs(diff[i, j]) > max_abs * 0.5 else "black"
                ax_c.text(
                    j, i, f"{diff[i, j]:+.2f}",
                    ha="center", va="center",
                    fontsize=10, color=txt_color, fontweight="normal",
                )

    # ── Panel d: Log2 fold change (sorted diverging bars) ────────────────
    add_panel_label(ax_d, "d", x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
    fc_all = gen_marker_means / (real_marker_means + 1e-8)
    log2fc = np.log2(fc_all + 1e-12)
    sort_fc = np.argsort(log2fc)
    log2fc_sorted = log2fc[sort_fc]
    names_sorted = [all_marker_genes[i] for i in sort_fc]
    bar_colors = [
        COLORS.get("good", "#4CAF50") if v >= 0 else COLORS.get("bad", "#E53935")
        for v in log2fc_sorted
    ]
    ax_d.barh(range(n_markers), log2fc_sorted, color=bar_colors,
              height=0.6, edgecolor="white", alpha=0.85)
    ax_d.axvline(x=0, color="#333", linewidth=1.5, linestyle="-")
    ax_d.set_yticks(range(n_markers))
    ax_d.set_yticklabels(names_sorted, fontsize=10)
    ax_d.set_xlabel("log$_2$ Fold Change (Gen / Real)", fontsize=11)
    ax_d.set_title("Marker Fold Change", fontsize=12)
    for i, lfc in enumerate(log2fc_sorted):
        if abs(lfc) < 0.005:
            continue
        ax_d.text(
            lfc + 0.002 if lfc >= 0 else lfc - 0.002, i,
            f"{lfc:+.2f}",
            va="center", fontsize=10,
            ha="left" if lfc >= 0 else "right",
            fontweight="bold" if abs(lfc) > 0.07 else "normal",
        )


# ---------------------------------------------------------------------------
# Row renderers (expression-correlation side: panels e, f, g, h). Mirrors
# fig09_expression_corr.plot_expression_correlation, adapted to pre-created axes.
# ---------------------------------------------------------------------------

def _draw_correlation_panels(
    fig: plt.Figure,
    ax_e: plt.Axes,
    ax_f: plt.Axes,
    ax_g: plt.Axes,
    ax_h: plt.Axes,
    data: Dict,
) -> None:
    """Right column: per-gene density scatter (e), per-type lollipop (f),
    marker grouped bars (g), residual histogram (h). Mirrors
    fig09_expression_corr.plot_expression_correlation."""
    real = data["real"]
    gen = data["gen"]
    gene_names = data["gene_names"]
    metrics = data["metrics"]

    real_means = real.mean(axis=0)
    gen_means = gen.mean(axis=0)
    residuals = gen_means - real_means

    # ── Panel e: Per-gene density scatter with residual coloring ────────
    add_panel_label(ax_e, "e", x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
    abs_res = np.abs(residuals)
    resid_vmax = float(np.percentile(abs_res, 95)) if len(abs_res) else 1.0
    sc = ax_e.scatter(
        real_means, gen_means, c=abs_res, cmap="magma_r",
        s=10, alpha=0.55, edgecolors="none",
        vmin=0, vmax=resid_vmax,
    )
    lo = min(real_means.min(), gen_means.min()) - 0.2
    hi = max(real_means.max(), gen_means.max()) + 0.2
    ax_e.plot([lo, hi], [lo, hi], color=COLORS["bad"], linestyle="--",
              lw=1.5, alpha=0.7, label="y = x", zorder=1)
    ax_e.fill_between([lo, hi], [lo - 0.1, hi - 0.1], [lo + 0.1, hi + 0.1],
                      alpha=0.06, color=COLORS["good"], zorder=0)
    ax_e.set_xlabel("Real Mean Expression", fontsize=11)
    ax_e.set_ylabel("Generated Mean Expression", fontsize=11)
    ax_e.set_title("Per-Gene Correlation", fontsize=12)
    ax_e.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    ax_e.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    cax_e = add_axes_next_to(
        fig, ax_e, side="right", width=0.012,
        height=ax_e.get_position().height * 0.32,
        pad=0.014, align="bottom", y_offset=0.018,
    )
    cbar_e = fig.colorbar(sc, cax=cax_e)
    if getattr(cbar_e, "solids", None) is not None:
        try:
            cbar_e.solids.set_edgecolor("face")
        except Exception:
            pass
    cbar_e.set_label("|Resid|", fontsize=9)
    cbar_e.set_ticks(np.linspace(0, resid_vmax, 3))
    cbar_fmt = ScalarFormatter(useMathText=True)
    cbar_fmt.set_scientific(True)
    cbar_fmt.set_powerlimits((0, 0))
    cbar_e.formatter = cbar_fmt
    cbar_e.update_ticks()
    cbar_e.ax.tick_params(labelsize=7, length=2, pad=1)
    cbar_e.ax.yaxis.get_offset_text().set_fontsize(7)
    cbar_e.ax.yaxis.get_offset_text().set_visible(True)

    # Gene callouts for top-3 residual outliers.
    outlier_idx = np.argsort(abs_res)[-3:]
    outlier_idx = outlier_idx[np.argsort(abs_res[outlier_idx])[::-1]]
    valid_idx = [i for i in outlier_idx if i < len(gene_names)]
    x_lo, x_hi = ax_e.get_xlim()
    y_lo, y_hi = ax_e.get_ylim()
    x_span = x_hi - x_lo + 1e-12
    y_span = y_hi - y_lo + 1e-12
    for i in valid_idx:
        xv, yv = real_means[i], gen_means[i]
        x_frac = (xv - x_lo) / x_span
        y_frac = (yv - y_lo) / y_span
        dx_pt = -24 if x_frac > 0.62 else 18
        dy_pt = -16 if y_frac > 0.62 else 14
        ax_e.annotate(
            gene_names[i],
            xy=(xv, yv), xycoords="data",
            xytext=(dx_pt, dy_pt), textcoords="offset points",
            fontsize=14,
            ha="right" if dx_pt < 0 else "left", va="center",
            color="#333",
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="#BBB", lw=0.4, alpha=0.95),
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#888",
                            alpha=0.65, shrinkA=1, shrinkB=1),
            zorder=6, annotation_clip=True,
        )

    # ── Panel f: Per-type deviation lollipop chart (log scale) ───────────
    add_panel_label(ax_f, "f", x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
    per_type = metrics.get("per_type_expression_fidelity", {})
    if per_type:
        type_names_sorted = sorted(per_type.keys(),
                                   key=lambda k: per_type[k]["pearson_r"])
        type_rs = [per_type[n]["pearson_r"] for n in type_names_sorted]
        n_show_each = min(5, len(type_rs) // 2)
        if len(type_rs) > 2 * n_show_each + 2:
            show_idx = (list(range(n_show_each))
                        + list(range(len(type_rs) - n_show_each, len(type_rs))))
            type_names_sorted = [type_names_sorted[i] for i in show_idx]
            type_rs = [type_rs[i] for i in show_idx]
        short_names = [abbreviate_cell_type(n, 24) for n in type_names_sorted]
        type_devs = [max(1 - r, 1e-12) for r in type_rs]
        y_pos = np.arange(len(type_devs))
        dev_color = COLORS["real"]
        ax_f.hlines(y_pos, min(type_devs) * 0.5, type_devs,
                    color="#DDD", linewidth=0.8, zorder=1)
        ax_f.scatter(type_devs, y_pos, c=dev_color, s=30, zorder=3,
                     edgecolors="white", linewidths=0.5)
        ax_f.set_yticks(y_pos)
        ax_f.set_yticklabels(short_names, fontsize=FONT_DENSE_YTICK, ha="right")
        ax_f.set_xscale("log")
        ax_f.set_xlabel("Deviation (1 \u2212 r)", fontsize=11)
        ax_f.text(
            0.95, 0.05, "Expression in scGPT binned space",
            transform=ax_f.transAxes, ha="right", va="bottom",
            fontsize=9, style="italic", color=COLORS["neutral"],
        )
    else:
        ax_f.text(0.5, 0.5, "No per-type data",
                  ha="center", va="center", transform=ax_f.transAxes)
    ax_f.set_title("Per-Type Expression Fidelity", fontsize=12)

    # ── Panel g: Marker gene grouped bars with error bars + fc callouts ──
    add_panel_label(ax_g, "g", x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
    marker_dict = metrics.get("marker_genes", {})
    selected_cats: List[str] = []
    selected_genes: List[List[str]] = []
    for cat, genes_list in marker_dict.items():
        found = [g for g in genes_list if g in gene_names]
        if found:
            selected_cats.append(cat)
            selected_genes.append(found[:2])
        if len(selected_cats) >= 6:
            break
    if selected_cats:
        all_marker_genes_g: List[str] = []
        for _cat, gg in zip(selected_cats, selected_genes):
            for g in gg:
                all_marker_genes_g.append(g)
        gidx = [gene_names.index(g) for g in all_marker_genes_g]
        r_means = np.array([real[:, i].mean() for i in gidx])
        g_means = np.array([gen[:, i].mean() for i in gidx])
        r_stds = np.array([real[:, i].std() for i in gidx])
        g_stds = np.array([gen[:, i].std() for i in gidx])
        n_real = real.shape[0]
        n_gen = gen.shape[0]
        x = np.arange(len(all_marker_genes_g))
        width = 0.35
        ax_g.bar(x - width / 2, r_means, width, yerr=r_stds / np.sqrt(n_real),
                 label="Real", color=COLORS["real"], alpha=0.85,
                 edgecolor="white", capsize=3, error_kw=dict(lw=0.8))
        ax_g.bar(x + width / 2, g_means, width, yerr=g_stds / np.sqrt(n_gen),
                 label="Gen", color=COLORS["generated"], alpha=0.85,
                 edgecolor="white", capsize=3, error_kw=dict(lw=0.8))
        for i, (rm, gm) in enumerate(zip(r_means, g_means)):
            fc = gm / (rm + 1e-8)
            if abs(fc - 1.0) > 0.10:
                color = COLORS["good"] if 0.95 <= fc <= 1.05 else COLORS["bad"]
                ax_g.text(
                    i, max(rm, gm) + max(r_stds[i], g_stds[i]) * 0.5 + 0.01,
                    f"{fc:.2f}\u00d7", ha="center", fontsize=10, color=color,
                )
        ax_g.set_xticks(x)
        ax_g.set_xticklabels(
            [g for g in all_marker_genes_g], fontsize=10, rotation=90, ha="center",
        )
        ax_g.set_ylabel("Expression (mean \u00b1 SEM)", fontsize=11)
    else:
        ax_g.text(0.5, 0.5, "No marker genes found",
                  ha="center", va="center", transform=ax_g.transAxes)
    ax_g.set_title("Marker Gene Expression", fontsize=12)
    ax_g.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

    # ── Panel h: Per-gene residual distribution ─────────────────────────
    add_panel_label(ax_h, "h", x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
    ax_h.hist(residuals, bins=60, color=COLORS["real"], alpha=0.7,
              edgecolor="white", density=True)
    ax_h.tick_params(axis="x", labelsize=10, rotation=30)
    ax_h.axvline(x=0, color=COLORS["bad"], linestyle="--",
                 linewidth=1.5, label="Zero")
    ax_h.axvline(
        x=residuals.mean(), color=COLORS["warn"], linestyle="-",
        linewidth=1.5, label=f"Mean={residuals.mean():.3f}",
    )
    ax_h.set_xlabel("Residual (Gen \u2212 Real)", fontsize=11)
    ax_h.set_ylabel("Density", fontsize=11)
    ax_h.set_title("Per-Gene Residual Distribution", fontsize=12)
    ax_h.legend(fontsize=10, frameon=False)
    ax_h.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax_h.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    pct_within_01 = (np.abs(residuals) < 0.1).mean() * 100
    pct_within_001 = (np.abs(residuals) < 0.01).mean() * 100
    ax_h.text(
        0.95, 0.95,
        f"|\u0394|<0.01: {pct_within_001:.0f}%\n|\u0394|<0.10: {pct_within_01:.0f}%",
        transform=ax_h.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="none", alpha=0.9),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_fig04_composed(
    real_expr_path: Optional[str] = None,
    gen_expr_path: Optional[str] = None,
    real_labels_path: Optional[str] = None,
    gen_labels_path: Optional[str] = None,
    gene_names_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    dpi: int = 300,
    save: bool = True,
) -> Optional[Path]:
    """Render the 4-row x 2-col composite Fig 4 and save to
    ``fig04_composed.{pdf,png}``.

    Parameters
    ----------
    real_expr_path, gen_expr_path        : per-cell expression matrix NPYs.
    real_labels_path, gen_labels_path    : per-cell type-label NPYs.
    gene_names_path                      : JSON list of gene names.
    metrics_path                         : expression_metrics.json.
    cache_dir                            : directory with text_captions_deduplicated.json.
    output_dir                           : directory to save the composite PDF.
    dpi                                  : rasterised preview DPI (PDF is vector).
    save                                 : if False, return None without writing.

    Returns
    -------
    Path to the saved PDF, or ``None`` if the expression cache is missing.
    """
    apply_style()

    real_p = Path(real_expr_path) if real_expr_path else RESULTS_DIR / "real_expression.npy"
    gen_p = Path(gen_expr_path) if gen_expr_path else RESULTS_DIR / "generated_expression.npy"
    real_lab_p = Path(real_labels_path) if real_labels_path else RESULTS_DIR / "real_expression_labels.npy"
    gen_lab_p = Path(gen_labels_path) if gen_labels_path else RESULTS_DIR / "generated_expression_labels.npy"
    gene_p = Path(gene_names_path) if gene_names_path else RESULTS_DIR / "expression_gene_names.json"
    metrics_p = Path(metrics_path) if metrics_path else RESULTS_DIR / "expression_metrics.json"
    cache = Path(cache_dir) if cache_dir else CACHE_DIR
    out_dir = Path(output_dir) if output_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_fig04_inputs(
        real_expr_path=real_p,
        gen_expr_path=gen_p,
        real_labels_path=real_lab_p,
        gen_labels_path=gen_lab_p,
        gene_names_path=gene_p,
        metrics_path=metrics_p,
        cache_dir=cache,
    )
    if data is None:
        logger.warning("No Fig 4 data sources found — skipping fig04_composed")
        return None

    # ── Layout: one figure, 4 rows x 2 cols. Each panel gets ~5in x 3in of
    # space, which matches the per-slice producers' 12.0x6.4 / 12.0x6.8 frames
    # halved horizontally. Total figsize is chosen so that when the PDF is
    # rendered at `\\includegraphics[width=\\textwidth]` (7in) the effective
    # scale is 0.7, keeping 10pt+ text above the VCD 7pt rendered-minimum.
    fig = plt.figure(figsize=(10.0, 15.83))
    layout = bind_figure_region(fig, (0.06, 0.04, 0.97, 0.97))
    row1, row2, row3, row4 = layout.split_rows(
        [1.00, 1.00, 1.00, 1.00], hspace=0.58,
    )

    # Left column (marker-gene panels) uses wider left inset to carry
    # multi-character y-tick strings ("EPCAM", "COL1A2"); right column
    # (correlation panels) has shorter left axes. Column widths are equal.
    r1 = row1.split_cols([1.0, 1.0], gap=0.050)
    r2 = row2.split_cols([1.0, 1.0], gap=0.050)
    r3 = row3.split_cols([1.0, 1.0], gap=0.050)
    r4 = row4.split_cols([1.0, 1.0], gap=0.050)

    ax_a = r1[0].inset(left=0.060, right=0.020).add_axes(fig)
    ax_e = r1[1].inset(left=0.060, right=0.030).add_axes(fig)

    ax_b = r2[0].inset(left=0.060, right=0.020).add_axes(fig)
    ax_f = r2[1].inset(left=0.100, right=0.020).add_axes(fig)

    ax_c = r3[0].inset(left=0.060, right=0.040).add_axes(fig)
    ax_g = r3[1].inset(left=0.060, right=0.020).add_axes(fig)

    ax_d = r4[0].inset(left=0.060, right=0.020).add_axes(fig)
    ax_h = r4[1].inset(left=0.060, right=0.020).add_axes(fig)

    _draw_marker_panels(fig, ax_a, ax_b, ax_c, ax_d, data)
    _draw_correlation_panels(fig, ax_e, ax_f, ax_g, ax_h, data)

    if save:
        path = save_with_vcd(
            fig, out_dir / "fig04_composed.png", dpi,
            layout_rect=(0.02, 0.01, 0.98, 0.98),
        )
        logger.info("Saved fig04_composed \u2192 %s", path)
        plt.close(fig)
        return out_dir / "fig04_composed.pdf"
    return None


# Direct-run entry point so ``python -m src.visualization.fig04_composed`` works.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    plot_fig04_composed()


__all__ = ["plot_fig04_composed"]
