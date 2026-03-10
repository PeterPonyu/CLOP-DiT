"""
fig08_markers.py -- Article Figure 8: Marker Gene Comparison.

Per-type real vs generated expression with grouped bars and annotated heatmaps.

  N1: Grouped horizontal bar chart -- mean expression per marker (real vs gen)
  N2: Dual heatmap with cell-value annotations and row-normalized coloring
  N3: Difference heatmap with statistical significance indicators
  N4: Log2 fold-change diverging horizontal bar chart for all markers

Standalone function extracted from panels_expression for composability and
VCD-integrated saving via an optional ``save_panel_fn`` callback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, abbreviate_cell_type, add_colorbar_safe, add_panel_label, save_with_vcd
from .panel_geometry import apply_layout_rect

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Module-level constants (biologically meaningful marker panels)
# ──────────────────────────────────────────────────────────────

# Biologically meaningful markers covering major lineages
MARKER_PANEL_GENES: Dict[str, List[str]] = {
    "CD8+ T":       ["CD8A", "GZMB"],
    "Myeloid":      ["CD68", "CD163"],
    "Epithelial":   ["EPCAM", "KRT8"],
    "Stromal":      ["COL1A1", "COL1A2"],
}

# Representative types to show per-type breakdown
MARKER_PANEL_TYPES: List[str] = [
    "CD8+ cytotoxic T lymphocytes",
    "Tumor-associated macrophages",
    "Epithelial tumor cells",
    "Fibroblasts and mesenchymal stromal cell",
]


# ──────────────────────────────────────────────────────────────
# FIGURE 8: Marker Gene Comparison (per-type real vs generated)
# ──────────────────────────────────────────────────────────────

def plot_marker_gene_comparison(
    real_expr_path: str = "results/real_expression.npy",
    gen_expr_path: str = "results/generated_expression.npy",
    real_labels_path: str = "results/real_expression_labels.npy",
    gen_labels_path: str = "results/generated_expression_labels.npy",
    gene_names_path: str = "results/expression_gene_names.json",
    metrics_path: str = "results/expression_metrics.json",
    type_names: Optional[Dict[int, str]] = None,
    output_dir: Path = Path("results/figures"),
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Rich marker-gene comparison with grouped bars and annotated heatmaps.

    N1: Grouped horizontal bar chart -- mean expression per marker (real vs gen)
    N2: Dual heatmap with cell-value annotations and row-normalized coloring
    N3: Difference heatmap with statistical significance indicators
    N4: Log2 fold-change diverging horizontal bar chart for all markers

    Parameters
    ----------
    real_expr_path : path to real expression matrix (.npy, cells x genes)
    gen_expr_path : path to generated expression matrix (.npy, cells x genes)
    real_labels_path : path to real cell-type label array (.npy)
    gen_labels_path : path to generated cell-type label array (.npy)
    gene_names_path : path to gene names JSON list
    metrics_path : path to expression metrics JSON
    type_names : dict mapping integer type IDs to human-readable names;
                 if ``None``, falls back to ``"Type_<id>"`` labels
    output_dir : directory for saved figures
    dpi : figure resolution
    save : whether to save figure to disk
    save_panel_fn : optional callable(fig, basename, output_dir, dpi) for
                    VCD-integrated saving; falls back to ``save_with_vcd``
    """
    if type_names is None:
        type_names = {}

    paths = [real_expr_path, gen_expr_path, gene_names_path, metrics_path]
    if not all(Path(p).exists() for p in paths):
        logger.info("Expression data not found -- skipping Panel N")
        return None

    real = np.load(real_expr_path)
    gen = np.load(gen_expr_path)
    real_labels = np.load(real_labels_path) if Path(real_labels_path).exists() else None
    gen_labels = np.load(gen_labels_path) if Path(gen_labels_path).exists() else None
    with open(gene_names_path) as f:
        gene_names = json.load(f)

    all_marker_genes: List[str] = []
    marker_cats: List[str] = []
    for cat, genes in MARKER_PANEL_GENES.items():
        for g in genes:
            if g in gene_names:
                all_marker_genes.append(g)
                marker_cats.append(cat)
    if len(all_marker_genes) < 2:
        logger.warning("Too few markers found -- skipping Panel N")
        return None

    gene_idx = [gene_names.index(g) for g in all_marker_genes]
    real_marker_means = real[:, gene_idx].mean(axis=0)
    gen_marker_means = gen[:, gene_idx].mean(axis=0)

    selected_type_ids: List[int] = []
    selected_type_names: List[str] = []
    if real_labels is not None:
        for target_name in MARKER_PANEL_TYPES:
            for tid, tname in type_names.items():
                if target_name.lower() in tname.lower():
                    if tid in np.unique(real_labels):
                        selected_type_ids.append(tid)
                        selected_type_names.append(abbreviate_cell_type(tname, 24))
                        break
    if len(selected_type_ids) < 3 and real_labels is not None:
        unique, counts = np.unique(real_labels, return_counts=True)
        top4 = unique[np.argsort(counts)[-4:]]
        selected_type_ids = top4.tolist()
        selected_type_names = [abbreviate_cell_type(type_names.get(int(t), f"Type_{t}"), 24)
                               for t in selected_type_ids]

    n_markers = len(all_marker_genes)
    n_sel_types = len(selected_type_ids)

    fig = plt.figure(figsize=(10.4, 7.3))
    gs = fig.add_gridspec(2, 2, wspace=0.52, hspace=0.44)
    apply_layout_rect(fig, (0.03, 0.08, 0.98, 0.93))
    # Note: Figure-level title removed per revision requirements

    # -- N1: Grouped horizontal bar chart -- mean expression per marker (real vs gen) --
    ax1 = fig.add_subplot(gs[0, 0])
    r_means = np.array([real[:, gi].mean() for gi in gene_idx])
    g_means = np.array([gen[:, gi].mean() for gi in gene_idx])
    r_stds = np.array([real[:, gi].std() for gi in gene_idx])
    g_stds = np.array([gen[:, gi].std() for gi in gene_idx])
    n_real = real.shape[0]
    n_gen = gen.shape[0]

    y_pos = np.arange(n_markers)
    width = 0.35
    ax1.barh(y_pos - width / 2, r_means, width, xerr=r_stds / np.sqrt(n_real),
             label="Real", color=COLORS["real"], alpha=0.85, edgecolor="white",
             capsize=3, error_kw=dict(lw=0.8))
    ax1.barh(y_pos + width / 2, g_means, width, xerr=g_stds / np.sqrt(n_gen),
             label="Gen", color=COLORS["generated"], alpha=0.85, edgecolor="white",
             capsize=3, error_kw=dict(lw=0.8))

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{g[:15]}" for g in all_marker_genes], fontsize=10)
    ax1.set_xlabel("Mean Expression", fontsize=11)
    ax1.set_title("Marker Expression by Lineage", fontsize=12)
    ax1.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.35)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=10, loc="lower left", frameon=False,
               bbox_to_anchor=(0.0, 1.02, 1.0, 0.12),
               mode="expand", ncol=2, borderaxespad=0.0)
    add_panel_label(ax1, 'a', x=-0.10, y=1.05)

    # -- N2 & N3: Heatmaps (if per-type labels) --
    if n_sel_types >= 2 and real_labels is not None and gen_labels is not None:
        real_heat = np.zeros((n_sel_types, n_markers))
        gen_heat = np.zeros((n_sel_types, n_markers))
        real_heat_std = np.zeros((n_sel_types, n_markers))
        gen_heat_std = np.zeros((n_sel_types, n_markers))
        for i, tid in enumerate(selected_type_ids):
            r_mask = real_labels == tid
            g_mask = gen_labels == tid
            for j, gi in enumerate(gene_idx):
                if r_mask.any():
                    real_heat[i, j] = real[r_mask][:, gi].mean()
                    real_heat_std[i, j] = real[r_mask][:, gi].std()
                if g_mask.any():
                    gen_heat[i, j] = gen[g_mask][:, gi].mean()
                    gen_heat_std[i, j] = gen[g_mask][:, gi].std()

        # N2: Side-by-side annotated heatmaps
        ax2 = fig.add_subplot(gs[0, 1])
        combined = np.hstack([real_heat, gen_heat])
        vmin, vmax = combined.min(), combined.max()
        gap_col = np.full((n_sel_types, 1), np.nan)
        display = np.hstack([real_heat, gap_col, gen_heat])

        cmap_n2 = mcolors.LinearSegmentedColormap.from_list(
            "expr_heat", ["#fff3e0", "#ffcc80", "#ff9800", "#e65100", "#bf360c"], N=256)
        im = ax2.imshow(display, cmap=cmap_n2, aspect="auto", vmin=vmin, vmax=vmax)
        ax2.set_yticks(range(n_sel_types))
        ax2.set_yticklabels([abbreviate_cell_type(n, 20) for n in selected_type_names], fontsize=10)
        xtick_pos = list(range(n_markers)) + list(range(n_markers + 1, 2 * n_markers + 1))
        xtick_labels = all_marker_genes + all_marker_genes
        ax2.set_xticks(xtick_pos)
        ax2.set_xticklabels(xtick_labels, fontsize=10, rotation=90, ha="center")
        # Use an explicit center band so the Real|Generated split survives print and downscaling.
        ax2.axvspan(n_markers - 0.5, n_markers + 0.5, color="#f3f3f3", zorder=0)
        ax2.axvline(x=n_markers, color="#666", linewidth=1.6, linestyle="-")
        ax2.axvline(x=n_markers - 0.5, color='black', linewidth=1.5, zorder=5)
        ax2.set_title("Per-Type \u00d7 Marker (Real | Gen)", fontsize=11, pad=8)
        ax2.text(n_markers / 2 - 0.5, -1.2, "Real", ha="center",
             fontsize=11, color=COLORS["real"])
        ax2.text(n_markers + 0.5 + n_markers / 2 - 0.5, -1.2, "Generated",
             ha="center", fontsize=11, color=COLORS["generated"])

        add_colorbar_safe(im, ax=ax2, label="Expr.", shrink=0.6, pad=0.05)
        add_panel_label(ax2, 'b', x=-0.10, y=1.05)

        # N3: Difference heatmap with significance
        ax3 = fig.add_subplot(gs[1, 0])
        diff = gen_heat - real_heat
        pct_diff = diff / (np.abs(real_heat) + 1e-8) * 100
        max_abs = max(abs(diff.min()), abs(diff.max()), 0.01)

        im3 = ax3.imshow(diff, cmap="RdBu_r", aspect="auto",
                 vmin=-max_abs, vmax=max_abs)
        im3.set_rasterized(True)
        ax3.set_yticks(range(n_sel_types))
        ax3.set_yticklabels([abbreviate_cell_type(n, 20) for n in selected_type_names], fontsize=10)
        ax3.set_xticks(range(n_markers))
        ax3.set_xticklabels(all_marker_genes, fontsize=11, rotation=90, ha="center")
        ax3.set_title("\u0394 Expression (Gen \u2212 Real)", fontsize=11)
        add_colorbar_safe(im3, ax=ax3, label="\u0394", shrink=0.72, pad=0.06, aspect=14)
        add_panel_label(ax3, 'c', x=-0.10, y=1.05)
        # Only annotate cells with large differences
        for i in range(n_sel_types):
            for j in range(n_markers):
                if abs(diff[i, j]) > max_abs * 0.3:
                    txt_color = "white" if abs(diff[i, j]) > max_abs * 0.5 else "black"
                    ax3.text(j, i, f"{diff[i, j]:+.2f}",
                             ha="center", va="center", fontsize=10, color=txt_color,
                             fontweight="normal")

        # N4: Log2 fold-change diverging horizontal bar chart
        ax4 = fig.add_subplot(gs[1, 1])
        fc_all = gen_marker_means / (real_marker_means + 1e-8)
        log2fc = np.log2(fc_all + 1e-12)
        sort_fc = np.argsort(log2fc)
        log2fc_sorted = log2fc[sort_fc]
        names_sorted = [all_marker_genes[i] for i in sort_fc]

        bar_colors = [COLORS.get("good", "#4CAF50") if v >= 0
                      else COLORS.get("bad", "#E53935") for v in log2fc_sorted]

        ax4.barh(range(n_markers), log2fc_sorted,
                 color=bar_colors, height=0.6, edgecolor="white",
                 alpha=0.85)
        ax4.axvline(x=0, color="#333", linewidth=1.5, linestyle="-")
        ax4.set_yticks(range(n_markers))
        ax4.set_yticklabels(names_sorted, fontsize=10)
        ax4.set_xlabel("log$_2$ Fold Change (Gen / Real)", fontsize=11)
        ax4.set_title("Marker Fold Change", fontsize=12)
        add_panel_label(ax4, 'd', x=-0.10, y=1.05)
        for i, lfc in enumerate(log2fc_sorted):
            if abs(lfc) < 0.005:
                continue
            ax4.text(lfc + 0.002 if lfc >= 0 else lfc - 0.002, i,
                     f"{lfc:+.2f}", va="center", fontsize=10,
                     ha="left" if lfc >= 0 else "right",
                     fontweight="bold" if abs(lfc) > 0.07 else "normal")
    else:
        ax_fallback = fig.add_subplot(gs[0, 1])
        add_panel_label(ax_fallback, 'b')
        ax_fallback.text(0.5, 0.5, "Per-type labels not available",
                         ha="center", va="center", transform=ax_fallback.transAxes)

    if save:
        if save_panel_fn:
            save_panel_fn(fig, "fig08_marker_genes", Path(output_dir), dpi)
        else:
            save_with_vcd(fig, Path(output_dir) / "fig08_marker_genes.png", dpi)
    return fig
