#!/usr/bin/env python3
"""Gene-gene correlation analysis within cell types.

Computes within-cell-type gene-gene Pearson correlation matrices for real
and generated expression data.  Reports:
  1. Per-type upper-tri correlation between R_real and R_gen (Mantel-style).
  2. Aggregate statistics across all types.
  3. A 4-panel diagnostic figure saved to results/figures/.

Usage:
    python scripts/analysis/gene_gene_correlation.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.visualization.direct_layout import bind_figure_region
from src.visualization.explicit_positioning import add_axes_next_to
from src.visualization.style import apply_style, COLORS, FONT_TITLE, FONT_LABEL, add_panel_label, save_with_vcd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MIN_CELLS = 10          # need at least this many cells per type
TOP_GENES = 200         # use top variable genes for tractability
SEED = 42

np.random.seed(SEED)


def _load_data():
    gen_expr = np.load(RESULTS / "generated_expression.npy")
    real_expr = np.load(RESULTS / "real_expression.npy")
    gen_labels = np.load(RESULTS / "generated_expression_labels.npy")
    real_labels = np.load(RESULTS / "real_expression_labels.npy")
    with open(RESULTS / "expression_gene_names.json") as f:
        gene_names = json.load(f)
    return gen_expr, real_expr, gen_labels, real_labels, gene_names


def _select_hvg(real_expr, n=TOP_GENES):
    """Select top-n most variable genes by variance across all real cells."""
    var = np.var(real_expr, axis=0)
    idx = np.argsort(var)[-n:]
    return np.sort(idx)


def _upper_tri(mat):
    """Extract strict upper triangle."""
    return mat[np.triu_indices_from(mat, k=1)]


def _corr_matrix(X):
    """Pearson correlation matrix; nan-safe diagonal = 1."""
    n = X.shape[1]
    if X.shape[0] < 3:
        return np.full((n, n), np.nan)
    # center
    X_c = X - X.mean(axis=0, keepdims=True)
    norms = np.sqrt((X_c ** 2).sum(axis=0, keepdims=True))
    norms[norms == 0] = 1.0
    X_n = X_c / norms
    R = X_n.T @ X_n / max(X.shape[0] - 1, 1)  # sample corr
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -1.0, 1.0)


def analyse():
    apply_style()
    gen_expr, real_expr, gen_labels, real_labels, gene_names = _load_data()

    # Select highly variable genes
    hvg_idx = _select_hvg(real_expr, TOP_GENES)
    gene_subset = [gene_names[i] for i in hvg_idx]
    gen_sub = gen_expr[:, hvg_idx]
    real_sub = real_expr[:, hvg_idx]

    types = sorted(set(real_labels) & set(gen_labels))
    per_type_results = {}

    for t in types:
        r_mask = real_labels == t
        g_mask = gen_labels == t
        if r_mask.sum() < MIN_CELLS or g_mask.sum() < MIN_CELLS:
            continue
        R_real = _corr_matrix(real_sub[r_mask])
        R_gen = _corr_matrix(gen_sub[g_mask])
        ut_real = _upper_tri(R_real)
        ut_gen = _upper_tri(R_gen)
        valid = np.isfinite(ut_real) & np.isfinite(ut_gen)
        if valid.sum() < 10:
            continue
        mantel_r = np.corrcoef(ut_real[valid], ut_gen[valid])[0, 1]
        rmse = np.sqrt(np.mean((ut_real[valid] - ut_gen[valid]) ** 2))
        per_type_results[int(t)] = {
            "mantel_r": float(mantel_r),
            "rmse": float(rmse),
            "n_real": int(r_mask.sum()),
            "n_gen": int(g_mask.sum()),
        }

    mantel_vals = [v["mantel_r"] for v in per_type_results.values()]
    rmse_vals = [v["rmse"] for v in per_type_results.values()]

    summary = {
        "n_types_analysed": len(per_type_results),
        "n_genes": TOP_GENES,
        "mean_mantel_r": float(np.mean(mantel_vals)),
        "median_mantel_r": float(np.median(mantel_vals)),
        "std_mantel_r": float(np.std(mantel_vals)),
        "min_mantel_r": float(np.min(mantel_vals)),
        "max_mantel_r": float(np.max(mantel_vals)),
        "mean_rmse": float(np.mean(rmse_vals)),
        "per_type": per_type_results,
    }

    out_json = RESULTS / "gene_gene_correlation.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {out_json}")
    print(f"  Types analysed: {len(per_type_results)}")
    print(f"  Mean Mantel r: {np.mean(mantel_vals):.4f} ± {np.std(mantel_vals):.4f}")
    print(f"  Median Mantel r: {np.median(mantel_vals):.4f}")
    print(f"  Mean RMSE: {np.mean(rmse_vals):.4f}")

    # --- Produce figure ---
    _make_figure(per_type_results, gen_sub, real_sub, gen_labels, real_labels,
                 gene_subset, types, label_offset=4)
    return summary


def _make_figure(per_type_results, gen_sub, real_sub, gen_labels, real_labels,
                 gene_subset, types, label_offset: int = 0):
    from src.visualization.style import (
        abbreviate_cell_type, FONT_TITLE, FONT_LABEL, FONT_TICK,
        FONT_ANNOTATION, FONT_SMALL, FONT_HEATMAP_CELL, style_axes,
    )
    from scipy import stats as scipy_stats

    # Load captions for cell-type names
    captions_path = ROOT / "data" / "cached_latents" / "text_captions_deduplicated.json"
    _type_names = {}
    if captions_path.exists():
        with open(captions_path) as f:
            _cap = json.load(f)
        for k, v in _cap.items():
            short = v.split(",")[0][:50] if isinstance(v, str) else str(v)[:50]
            _type_names[int(k)] = short

    mantel_vals = [v["mantel_r"] for v in per_type_results.values()]

    fig = plt.figure(figsize=(14.0, 8.0), dpi=300)
    layout = bind_figure_region(fig, (0.08, 0.10, 0.93, 0.94))
    top_row, bottom_row = layout.split_rows([0.92, 1.08], hspace=0.40)
    top_left, top_right = top_row.split_cols(2, wspace=0.34)
    bottom_left, bottom_right = bottom_row.split_cols([0.92, 1.08], wspace=0.34)

    # ── Panel (a): Distribution with null baseline ──
    ax = top_left.add_axes(fig)
    add_panel_label(ax, chr(ord('a') + label_offset), x=-0.12, y=1.06)

    # Compute null baseline: permuted gene labels within each type
    rng = np.random.default_rng(42)
    null_mantels = []
    n_perm = 50
    for _ in range(n_perm):
        for t in per_type_results:
            r_mask = real_labels == t
            g_mask = gen_labels == t
            rs = real_sub[r_mask][:, :50]
            gs_sub = gen_sub[g_mask][:, :50]
            if rs.shape[0] < 10 or gs_sub.shape[0] < 10:
                continue
            # Permute gene order in generated to break real structure
            perm_idx = rng.permutation(gs_sub.shape[1])
            gs_perm = gs_sub[:, perm_idx]
            R_real = _corr_matrix(rs)
            R_perm = _corr_matrix(gs_perm)
            ut_r = _upper_tri(R_real)
            ut_p = _upper_tri(R_perm)
            valid = np.isfinite(ut_r) & np.isfinite(ut_p)
            if valid.sum() > 10:
                null_mantels.append(np.corrcoef(ut_r[valid], ut_p[valid])[0, 1])

    # Plot histograms
    bins = np.linspace(min(min(mantel_vals), min(null_mantels) if null_mantels else 0) - 0.05,
                       max(max(mantel_vals), max(null_mantels) if null_mantels else 0.5) + 0.05, 30)
    if null_mantels:
        ax.hist(null_mantels, bins=bins, color="#BDBDBD", alpha=0.5, edgecolor="white",
                density=True, label=f"Permuted null (n={len(null_mantels)})")
    ax.hist(mantel_vals, bins=bins, color=COLORS["real"], edgecolor="white", alpha=0.8,
            density=True, label=f"Observed (n={len(mantel_vals)})")

    ax.axvline(np.median(mantel_vals), color=COLORS["generated"], ls="--", lw=1.5,
               label=f"Median = {np.median(mantel_vals):.3f}")
    ax.axvline(np.mean(mantel_vals), color=COLORS["annotation_dark"], ls="-", lw=1.5,
               label=f"Mean = {np.mean(mantel_vals):.3f}")

    # Bootstrap 95% CI for mean
    boot_means = [np.mean(rng.choice(mantel_vals, len(mantel_vals), replace=True))
                  for _ in range(1000)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    null_mean = np.mean(null_mantels) if null_mantels else 0
    ax.text(0.03, 0.97,
            f"Mean 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]\n"
            f"Null mean: {null_mean:.3f}"
            if all(m > null_mean for m in mantel_vals)
            else f"Mean 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]\n"
                 f"Null mean: {null_mean:.3f}",
            transform=ax.transAxes, ha="left", va="top",
              fontsize=FONT_SMALL, color=COLORS["neutral"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"))

    ax.legend(fontsize=FONT_ANNOTATION - 1, frameon=True,
              framealpha=0.85, edgecolor="none",
              loc="upper right", bbox_to_anchor=(0.99, 0.98))
    style_axes(ax, "default",
               xlabel="Upper-triangle Pearson r (real vs. gen corr. matrix)",
               ylabel="Density",
               title="Gene-Gene Correlation Preservation")

    # Shared color scale: tighten to the observed central range so structure is visible
    best_type = max(per_type_results, key=lambda k: per_type_results[k]["mantel_r"])
    worst_type = min(per_type_results, key=lambda k: per_type_results[k]["mantel_r"])

    best_mask_real = real_labels == best_type
    best_mask_gen = gen_labels == best_type
    worst_mask_real = real_labels == worst_type
    worst_mask_gen = gen_labels == worst_type
    best_diff_preview = _corr_matrix(gen_sub[best_mask_gen][:, :50]) - _corr_matrix(real_sub[best_mask_real][:, :50])
    worst_diff_preview = _corr_matrix(gen_sub[worst_mask_gen][:, :50]) - _corr_matrix(real_sub[worst_mask_real][:, :50])
    combined_abs = np.concatenate(
        [
            np.abs(best_diff_preview[np.isfinite(best_diff_preview)]),
            np.abs(worst_diff_preview[np.isfinite(worst_diff_preview)]),
        ]
    )
    heatmap_abs_scale = float(np.nanpercentile(combined_abs, 97)) if combined_abs.size else 0.5
    heatmap_abs_scale = max(heatmap_abs_scale, 0.08)

    # ── Panel (b): Best-preserved cell type ──
    best_name = abbreviate_cell_type(_type_names.get(best_type, f"Type {best_type}"), max_len=35)
    best_r = per_type_results[best_type]["mantel_r"]
    best_rmse = per_type_results[best_type]["rmse"]

    r_mask = real_labels == best_type
    g_mask = gen_labels == best_type
    R_real = _corr_matrix(real_sub[r_mask][:, :50])
    R_gen = _corr_matrix(gen_sub[g_mask][:, :50])

    ax2 = top_right.add_axes(fig)
    add_panel_label(ax2, chr(ord('a') + label_offset + 1), x=-0.12, y=1.06)
    diff = R_gen - R_real
    im = ax2.imshow(
        diff,
        cmap="RdBu_r",
        vmin=-heatmap_abs_scale,
        vmax=heatmap_abs_scale,
        aspect="auto",
        interpolation="nearest",
        alpha=1.0,
    )
    ax2.set_ylim(49.5, -0.5)
    ax2.set_yticks([0, 10, 20, 30, 40, 49])
    ax2.set_title(f"Best: {best_name}", fontsize=FONT_TITLE)
    ax2.set_xlabel("Gene index (top 50 HVG)", fontsize=FONT_LABEL)
    ax2.set_ylabel("Gene index", fontsize=FONT_LABEL)

    # Summary annotation
    mad = np.nanmean(np.abs(diff))
    ax2.text(0.97, 0.03,
             f"r = {best_r:.3f}\nMAD = {mad:.3f}\nRMSE = {best_rmse:.3f}",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=FONT_ANNOTATION, color="black")
    cax2 = add_axes_next_to(
        fig,
        ax2,
        side="right",
        width=0.008,
        height=ax2.get_position().height * 0.52,
        pad=0.018,
        align="bottom",
        y_offset=0.01,
    )
    cb2 = fig.colorbar(im, cax=cax2)
    cb2.set_label("")
    cb2.ax.set_title("Δr", fontsize=FONT_SMALL, pad=2)
    cb2.ax.tick_params(labelsize=FONT_HEATMAP_CELL - 1)

    # ── Panel (c): Worst-preserved cell type ──
    worst_name = abbreviate_cell_type(_type_names.get(worst_type, f"Type {worst_type}"), max_len=35)
    worst_r = per_type_results[worst_type]["mantel_r"]
    worst_rmse = per_type_results[worst_type]["rmse"]

    r_mask = real_labels == worst_type
    g_mask = gen_labels == worst_type
    R_real_w = _corr_matrix(real_sub[r_mask][:, :50])
    R_gen_w = _corr_matrix(gen_sub[g_mask][:, :50])

    ax3 = bottom_left.add_axes(fig)
    add_panel_label(ax3, chr(ord('a') + label_offset + 2), x=-0.12, y=1.06)
    diff_w = R_gen_w - R_real_w
    im2 = ax3.imshow(
        diff_w,
        cmap="RdBu_r",
        vmin=-heatmap_abs_scale,
        vmax=heatmap_abs_scale,
        aspect="auto",
        interpolation="nearest",
        alpha=1.0,
    )
    ax3.set_ylim(49.5, -0.5)
    ax3.set_yticks([0, 10, 20, 30, 40, 49])
    ax3.set_title(f"Worst: {worst_name}", fontsize=FONT_TITLE)
    ax3.set_xlabel("Gene index (top 50 HVG)", fontsize=FONT_LABEL)
    ax3.set_ylabel("Gene index", fontsize=FONT_LABEL)

    mad_w = np.nanmean(np.abs(diff_w))
    ax3.text(0.97, 0.03,
             f"r = {worst_r:.3f}\nMAD = {mad_w:.3f}\nRMSE = {worst_rmse:.3f}",
             transform=ax3.transAxes, ha="right", va="bottom",
             fontsize=FONT_ANNOTATION, color="black")
    cax3 = add_axes_next_to(
        fig,
        ax3,
        side="right",
        width=0.008,
        height=ax3.get_position().height * 0.52,
        pad=0.018,
        align="bottom",
        y_offset=0.01,
    )
    cb3 = fig.colorbar(im2, cax=cax3)
    cb3.set_label("")
    cb3.ax.set_title("Δr", fontsize=FONT_SMALL, pad=2)
    cb3.ax.tick_params(labelsize=FONT_HEATMAP_CELL - 1)

    # ── Panel (d): Replace non-informative cell-count panel ──
    # Use Mantel r vs per-type mean expression variance (biological heterogeneity)
    ax4 = bottom_right.add_axes(fig)
    add_panel_label(ax4, chr(ord('a') + label_offset + 3), x=-0.12, y=1.06)

    # Compute mean expression variance per type as a proxy for heterogeneity
    type_het = []
    type_mantel = []
    type_labels_d = []
    for t in per_type_results:
        r_mask = real_labels == t
        if r_mask.sum() < 10:
            continue
        expr_var = np.mean(np.var(real_sub[r_mask], axis=0))
        type_het.append(expr_var)
        type_mantel.append(per_type_results[t]["mantel_r"])
        type_labels_d.append(t)

    type_het = np.array(type_het)
    type_mantel = np.array(type_mantel)

    ax4.scatter(type_het, type_mantel, s=35, alpha=0.6, c=COLORS["real"],
                edgecolors="white", linewidth=0.4, zorder=3)

    if len(type_het) >= 3 and np.std(type_het) > 0:
        spearman_r, spearman_p = scipy_stats.spearmanr(type_het, type_mantel)
        pearson_r, pearson_p = scipy_stats.pearsonr(type_het, type_mantel)

        # Regression line
        slope, intercept, _, _, _ = scipy_stats.linregress(type_het, type_mantel)
        x_fit = np.linspace(type_het.min(), type_het.max(), 100)
        ax4.plot(x_fit, slope * x_fit + intercept, color=COLORS["generated"],
                 linewidth=1.5, alpha=0.7, label="OLS fit")

        stat_text = (f"Spearman \u03c1 = {spearman_r:.3f}\n"
                 f"Pearson r = {pearson_r:.3f}")

        # Label top 3 outliers by residual with fixed callout slots so the
        # small annotation texts stay readable at manuscript scale.
        residuals = np.abs(type_mantel - (slope * type_het + intercept))
        top3 = np.argsort(residuals)[-3:]
        slots = [(0.83, 0.94, "left"), (0.83, 0.14, "left"), (0.72, 0.08, "left")]
        for idx, (slot_x, slot_y, ha) in zip(top3[np.argsort(residuals[top3])[::-1]], slots):
            t_id = type_labels_d[idx]
            lbl = abbreviate_cell_type(_type_names.get(t_id, f"Type {t_id}"), max_len=18)
            ax4.text(
                slot_x,
                slot_y,
                lbl,
                transform=ax4.transAxes,
                fontsize=8.5,
                ha=ha,
                va="center",
                color=COLORS["annotation_dark"],
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.86),
                zorder=6,
                clip_on=False,
            )
            connector = ConnectionPatch(
                xyA=(type_het[idx], type_mantel[idx]),
                coordsA=ax4.transData,
                xyB=(slot_x - 0.015, slot_y),
                coordsB=ax4.transAxes,
                axesA=ax4,
                axesB=ax4,
                arrowstyle="-",
                lw=0.55,
                color="#888",
                alpha=0.7,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0.12",
            )
            connector.set_clip_on(False)
            connector.set_zorder(2)
            ax4.add_artist(connector)
    else:
        stat_text = "Insufficient variance for correlation"

    ax4.text(0.03, 0.78, stat_text,
             transform=ax4.transAxes, ha="left", va="top",
             fontsize=FONT_SMALL, color=COLORS["neutral"])

    ax4.legend(fontsize=FONT_ANNOTATION, frameon=False)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune="both"))
    style_axes(ax4, "scatter",
               xlabel="Mean Gene Expression Variance",
               ylabel="Upper-Tri Pearson r",
               title="Preservation vs. Expression Heterogeneity")

    out_path = FIG_DIR / "fig09b_gene_gene_correlation.png"
    save_with_vcd(fig, out_path, dpi=300, layout_rect=(0.02, 0.04, 0.98, 0.93))
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    analyse()
