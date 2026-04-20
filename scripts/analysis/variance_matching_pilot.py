#!/usr/bin/env python3
"""Pilot variance-matching experiment using Sliced Wasserstein Distance (SWD).

This script evaluates the feasibility of a latent-space variance-matching
regularizer by computing per-cell-type SWD between real and generated latent
distributions. It also compares per-dimension variance statistics.

This is a diagnostic/analysis script, not a training modification.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.visualization.direct_layout import bind_figure_region
from src.visualization.style import apply_style, COLORS, add_panel_label, save_with_vcd
from matplotlib.patches import ConnectionPatch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sliced_wasserstein_distance(x: np.ndarray, y: np.ndarray, n_proj: int = 100,
                                 seed: int = 42) -> float:
    """Compute Sliced Wasserstein Distance between two point clouds.

    Parameters
    ----------
    x, y : (N, D) and (M, D) arrays
    n_proj : number of random projections
    seed : random seed

    Returns
    -------
    swd : float, average 1D Wasserstein distance across projections
    """
    rng = np.random.RandomState(seed)
    d = x.shape[1]
    # Random projection directions (unit vectors)
    theta = rng.randn(n_proj, d)
    theta = theta / np.linalg.norm(theta, axis=1, keepdims=True)

    # Project both point clouds
    proj_x = x @ theta.T  # (N, n_proj)
    proj_y = y @ theta.T  # (M, n_proj)

    # Sort each projection
    proj_x = np.sort(proj_x, axis=0)
    proj_y = np.sort(proj_y, axis=0)

    # Interpolate to same size if needed
    n, m = proj_x.shape[0], proj_y.shape[0]
    if n != m:
        # Interpolate the smaller to match the larger
        target_n = max(n, m)
        if n < target_n:
            idx = np.linspace(0, n - 1, target_n)
            proj_x = np.array([np.interp(idx, np.arange(n), proj_x[:, j])
                               for j in range(n_proj)]).T
        if m < target_n:
            idx = np.linspace(0, m - 1, target_n)
            proj_y = np.array([np.interp(idx, np.arange(m), proj_y[:, j])
                               for j in range(n_proj)]).T

    # Average absolute difference across quantiles and projections
    swd = np.mean(np.abs(proj_x - proj_y))
    return float(swd)


def main():
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "data" / "cached_latents"
    results_dir = project_root / "results"
    output_dir = results_dir / "variance_matching_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load real embeddings and group IDs
    print("[var_pilot] Loading real embeddings and group IDs...")
    real_emb = np.load(cache_dir / "cell_embeddings_dedup_preprocessed.npy")
    group_ids = np.load(cache_dir / "text_group_ids_dedup.npy")

    # Load generated embeddings
    gen_emb = np.load(results_dir / "generated_embeddings.npy")
    gen_labels = np.load(results_dir / "generated_labels.npy")

    # Load captions for labeling
    with open(cache_dir / "text_captions_deduplicated.json") as f:
        captions = json.load(f)

    unique_types = sorted(np.unique(group_ids))
    print(f"[var_pilot] Real: {real_emb.shape}, Gen: {gen_emb.shape}")
    print(f"[var_pilot] {len(unique_types)} cell types")

    results = []

    for gid in unique_types:
        real_mask = group_ids == gid
        gen_mask = gen_labels == gid

        real_subset = real_emb[real_mask]
        gen_subset = gen_emb[gen_mask]

        if len(gen_subset) < 5:
            continue

        # Per-dimension variance
        real_var = np.var(real_subset, axis=0)
        gen_var = np.var(gen_subset, axis=0)
        var_ratio = np.mean(gen_var) / (np.mean(real_var) + 1e-10)
        var_corr = np.corrcoef(real_var, gen_var)[0, 1]

        # SWD
        swd = sliced_wasserstein_distance(real_subset, gen_subset, n_proj=200)

        # Per-dimension mean absolute error
        real_mean = np.mean(real_subset, axis=0)
        gen_mean = np.mean(gen_subset, axis=0)
        mean_cos = float(np.dot(real_mean, gen_mean) /
                         (np.linalg.norm(real_mean) * np.linalg.norm(gen_mean) + 1e-10))

        caption = captions[str(gid)] if str(gid) in captions else f"Type {gid}"
        # Truncate caption
        short_name = caption.split(",")[0][:40] if isinstance(caption, str) else str(caption)[:40]

        results.append({
            "group_id": int(gid),
            "name": short_name,
            "n_real": int(real_mask.sum()),
            "n_gen": int(gen_mask.sum()),
            "swd": swd,
            "var_ratio": float(var_ratio),
            "var_corr": float(var_corr),
            "mean_cos": mean_cos,
            "mean_real_var": float(np.mean(real_var)),
            "mean_gen_var": float(np.mean(gen_var)),
        })

    # Summary statistics
    swd_values = [r["swd"] for r in results]
    var_ratios = [r["var_ratio"] for r in results]
    var_corrs = [r["var_corr"] for r in results]

    summary = {
        "n_types_evaluated": len(results),
        "swd_mean": float(np.mean(swd_values)),
        "swd_median": float(np.median(swd_values)),
        "swd_std": float(np.std(swd_values)),
        "swd_min": float(np.min(swd_values)),
        "swd_max": float(np.max(swd_values)),
        "var_ratio_mean": float(np.mean(var_ratios)),
        "var_ratio_median": float(np.median(var_ratios)),
        "var_corr_mean": float(np.mean(var_corrs)),
        "var_corr_median": float(np.median(var_corrs)),
        "per_type": results,
    }

    with open(output_dir / "swd_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[var_pilot] === Summary ===")
    print(f"  Types evaluated: {len(results)}")
    print(f"  SWD:        mean={np.mean(swd_values):.4f}, median={np.median(swd_values):.4f}")
    print(f"  Var ratio:  mean={np.mean(var_ratios):.4f}, median={np.median(var_ratios):.4f}")
    print(f"  Var corr:   mean={np.mean(var_corrs):.4f}, median={np.median(var_corrs):.4f}")

    # ── Generate figure ──
    apply_style()
    from src.visualization.style import (
        abbreviate_cell_type, FONT_TITLE, FONT_LABEL, FONT_TICK,
        FONT_ANNOTATION, FONT_SMALL, style_axes,
    )
    from scipy import stats as scipy_stats

    fig = plt.figure(figsize=(14.0, 8.0), dpi=300)
    layout = bind_figure_region(fig, (0.11, 0.10, 0.99, 0.96))
    # Uses the repository's direct rectangle layout engine, not GridSpec or
    # matplotlib's automatic/constrained layout.
    # - split_rows(..., hspace=...) controls the vertical gap between rows.
    # - split_cols(..., wspace=...) controls the horizontal gap between columns.
    top_row, bottom_row = layout.split_rows([1.20, 0.92], hspace=0.36)
    top_left, top_right = top_row.split_cols([1.2, 1.0], wspace=0.20)
    bottom_left, bottom_right = bottom_row.split_cols([1.16, 1.04], wspace=0.26)

    # ── Panel (a): SWD per Cell Type (sorted bar chart) ──
    sorted_results = sorted(results, key=lambda r: r["swd"], reverse=True)
    names = [abbreviate_cell_type(r["name"], max_len=18) for r in sorted_results]
    swds = [r["swd"] for r in sorted_results]
    n_reals_sorted = [r["n_real"] for r in sorted_results]
    swd_mean = np.mean(swds)
    swd_std = np.std(swds)

    ax = top_left.inset(left=0.02, right=0.01).add_axes(fig)
    add_panel_label(ax, 'a', x=-0.10, y=0.97)

    # Color: orange for outliers (>mean+1σ), blue otherwise; add legend
    colors = [COLORS["generated"] if s > swd_mean + swd_std else COLORS["real"] for s in swds]
    bars = ax.barh(range(len(names)), swds, color=colors, height=0.7, edgecolor="white", linewidth=0.3)

    # Full cell-type labels, adaptively thinned
    n_types = len(names)
    step = max(1, n_types // 22)
    thin_labels = []
    for i in range(n_types):
        if i % step == 0 or i == n_types - 1:
            thin_labels.append(names[i])
        else:
            thin_labels.append("")
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(thin_labels, fontsize=9)
    ax.invert_yaxis()

    # Mean + 1σ threshold lines
    ax.axvline(swd_mean, color="#555", linestyle="--", alpha=0.7, linewidth=1.2,
               label=f"Mean = {swd_mean:.4f}")
    ax.axvline(swd_mean + swd_std, color=COLORS["accent"], linestyle=":", alpha=0.6,
               linewidth=1.0, label=f"+1\u03c3 = {swd_mean + swd_std:.4f}")

    # Sample-size callouts: embed the n=... label inline at the bar tip for the
    # top-2 outlier bars only. Inline text (right-aligned inside the bar) avoids
    # leader lines that previously crossed into the bar area.
    outlier_indices = [i for i, s in enumerate(swds) if s > swd_mean + swd_std][:2]
    for idx in outlier_indices:
        row = sorted_results[idx]
        ax.text(
            row["swd"] * 0.985,
            idx,
            f"n={row['n_real']:,}",
            transform=ax.transData,
            fontsize=8.5,
            ha="right",
            va="center",
            color="white",
            zorder=6,
            clip_on=True,
        )

    ax.legend(fontsize=FONT_ANNOTATION, frameon=False,
              loc="lower right")
    style_axes(ax, "bar", xlabel="SWD", title="Latent SWD per Cell Type")
    ax.set_xlabel("SWD", fontsize=FONT_LABEL)
    ax.xaxis.labelpad = 2
    ax.xaxis.set_label_coords(0.5, -0.06)

    # ── Panel (b): Variance Ratio — strip + box plot ──
    ax2 = top_right.inset(left=0.02, right=0.02).add_axes(fig)
    add_panel_label(ax2, 'b', x=-0.10, y=0.97)

    vr_arr = np.array(var_ratios)
    vr_median = np.median(vr_arr)
    vr_q25, vr_q75 = np.percentile(vr_arr, [25, 75])
    pct_in_band = 100 * np.mean((vr_arr >= 0.9) & (vr_arr <= 1.1))

    # Horizontal box + jitter
    bp = ax2.boxplot(vr_arr, vert=False, widths=0.5, positions=[0],
                     patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor=COLORS["real"], alpha=0.25, linewidth=1.0),
                     medianprops=dict(color=COLORS["generated"], linewidth=2),
                     whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0))
    jitter_y = np.random.default_rng(42).uniform(-0.18, 0.18, len(vr_arr))
    ax2.scatter(vr_arr, jitter_y, s=22, alpha=0.6, c=COLORS["real"],
                edgecolors="white", linewidth=0.3, zorder=3)

    # Reference line + tolerance band
    ax2.axvline(1.0, color=COLORS["generated"], linestyle="--", linewidth=1.5, label="Ideal (1.0)")
    ax2.axvspan(0.9, 1.1, alpha=0.08, color=COLORS["good"], label="0.9\u20131.1 band")

    ax2.text(0.97, 0.97,
             f"Median = {vr_median:.3f}\n"
             f"IQR = [{vr_q25:.3f}, {vr_q75:.3f}]",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=FONT_ANNOTATION, color=COLORS["neutral"])

    ax2.set_yticks([])
    ax2.legend(fontsize=FONT_ANNOTATION, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    style_axes(ax2, "default",
               xlabel="Variance ratio (gen/real)",
               title="Per-Type Latent Variance Ratio")
    ax2.set_xlabel("Variance ratio (gen/real)", fontsize=FONT_LABEL)
    ax2.xaxis.labelpad = 2
    ax2.xaxis.set_label_coords(0.5, -0.06)

    # ── Panel (c): Per-Dimension Variance Correlation — ECDF + box ──
    ax3 = bottom_left.inset(right=0.02).add_axes(fig)
    add_panel_label(ax3, 'c', x=-0.10, y=1.02)

    vc_arr = np.array(var_corrs)
    vc_sorted = np.sort(vc_arr)
    ecdf_y = np.arange(1, len(vc_sorted) + 1) / len(vc_sorted)

    ax3.plot(vc_sorted, ecdf_y, color=COLORS["real"], linewidth=2, label="Observed ECDF")
    ax3.axvline(0, color="#999", linestyle=":", linewidth=1.0, alpha=0.6)

    vc_mean = np.mean(vc_arr)
    vc_median = np.median(vc_arr)
    pct_positive = 100 * np.mean(vc_arr > 0)

    ax3.axvline(vc_median, color=COLORS["generated"], linestyle="--", linewidth=1.5,
                label=f"Median = {vc_median:.3f}")

    ax3.text(0.97, 0.03,
             f"Mean = {vc_mean:.3f}\n"
             f"Median = {vc_median:.3f}\n"
             f"{pct_positive:.0f}% positive",
             transform=ax3.transAxes, ha="right", va="bottom",
             fontsize=FONT_SMALL, color=COLORS["neutral"])

    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=FONT_ANNOTATION, frameon=False, loc="upper left")
    style_axes(ax3, "default",
               xlabel="Per-dim variance corr. (real vs gen)",
               ylabel="Cumulative Proportion",
               title="Dimension-Wise Variance Correlation")
    ax3.xaxis.labelpad = 1
    ax3.set_title("Dimension-Wise Variance Correlation", fontsize=FONT_TITLE - 2, pad=0, y=0.985)

    # ── Panel (d): SWD vs. Training Cell Count ──
    ax4 = bottom_right.inset(left=0.05, right=0.02).add_axes(fig)
    add_panel_label(ax4, 'd', x=-0.10, y=1.02)

    n_reals = np.array([r["n_real"] for r in results])
    swd_arr = np.array(swd_values)

    ax4.scatter(n_reals, swd_arr, c=COLORS["real"], alpha=0.55, s=35,
                edgecolors="white", linewidth=0.4, zorder=3)
    ax4.set_xscale("log")
    x_min = max(1.0, float(n_reals.min()) * 0.85)
    x_max = float(n_reals.max()) * 1.12
    ax4.set_xlim(x_min, x_max)
    min_exp = int(np.floor(np.log10(x_min)))
    max_exp = int(np.ceil(np.log10(x_max)))
    major_ticks = [10**e for e in range(min_exp, max_exp + 1) if x_min <= 10**e <= x_max]
    if major_ticks:
        ax4.set_xticks(major_ticks)

    # Regression line + stats
    log_n = np.log10(n_reals + 1)
    valid = np.isfinite(log_n) & np.isfinite(swd_arr)

    if valid.sum() >= 3:
        slope, intercept, r_val, p_val, _ = scipy_stats.linregress(log_n[valid], swd_arr[valid])
        spearman_r, spearman_p = scipy_stats.spearmanr(log_n[valid], swd_arr[valid])

        # Regression smoother
        x_fit = np.linspace(log_n[valid].min(), log_n[valid].max(), 100)
        y_fit = slope * x_fit + intercept
        ax4.plot(10 ** x_fit, y_fit, color=COLORS["generated"], linewidth=1.5,
                 linestyle="-", alpha=0.7, label="OLS fit")

        # Confidence band (approx)
        n_pts = valid.sum()
        se = np.sqrt(np.sum((swd_arr[valid] - (slope * log_n[valid] + intercept)) ** 2) / (n_pts - 2))
        y_ci = 1.96 * se
        ax4.fill_between(10 ** x_fit, y_fit - y_ci, y_fit + y_ci,
                         alpha=0.08, color=COLORS["generated"])

        stat_text = (f"Pearson r = {r_val:.3f}\n"
                 f"Spearman \u03c1 = {spearman_r:.3f}")
    else:
        stat_text = "Insufficient data for correlation"

    ax4.text(0.03, 0.04, stat_text,
             transform=ax4.transAxes, ha="left", va="bottom",
             fontsize=FONT_ANNOTATION, color=COLORS["neutral"])

    # Label top 3 residual outliers. Labels are pinned to interior axes-fraction
    # slots (never outside the axes rectangle), with a connector line back to
    # each data point. This is the same pattern used in panel (a) and avoids the
    # "label drifted outside the axes" failure mode that offset_points-in-data
    # space exhibits near the plot edges.
    top3_idx = np.argsort(swd_arr)[-3:][::-1]
    # Interior anchor slots (axes-fraction) — bias toward the upper-left so the
    # connector reaches the rightmost outliers without crossing other points.
    slots = [(0.08, 0.94), (0.08, 0.82), (0.08, 0.70)]
    for (slot_x, slot_y), idx in zip(slots, top3_idx):
        lbl = abbreviate_cell_type(results[idx]["name"], max_len=18)
        ax4.text(
            slot_x,
            slot_y,
            lbl,
            transform=ax4.transAxes,
            fontsize=9,
            ha="left",
            va="center",
            color=COLORS["annotation_dark"],
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.88),
            zorder=6,
            clip_on=False,
        )
        conn = ConnectionPatch(
            xyA=(n_reals[idx], swd_arr[idx]),
            coordsA=ax4.transData,
            xyB=(slot_x + 0.02, slot_y),
            coordsB=ax4.transAxes,
            axesA=ax4,
            axesB=ax4,
            arrowstyle="-",
            lw=0.5,
            color="#888",
            alpha=0.7,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0.08",
        )
        conn.set_clip_on(False)
        conn.set_zorder(2)
        ax4.add_artist(conn)

    ax4.legend(fontsize=FONT_ANNOTATION, frameon=False, loc="upper right")
    style_axes(ax4, "scatter",
               xlabel="Training cells (log scale)",
               ylabel="Sliced Wasserstein Distance",
               title="SWD vs. Training Cell Count")
    ax4.xaxis.labelpad = 1
    ax4.set_title("SWD vs. Training Cell Count", fontsize=FONT_TITLE - 2, pad=0, y=0.985)

    fig_path = output_dir / "fig09a_variance_matching.png"
    save_with_vcd(fig, fig_path, dpi=300, layout_rect=(0.08, 0.06, 0.99, 0.96))
    print(f"\n[var_pilot] Figure saved to {fig_path}")

    # Also save to results/figures/ with the article-delivery basename
    fig_dir = project_root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for suffix in (".pdf",):
        src = output_dir / f"fig09a_variance_matching{suffix}"
        dst = fig_dir / f"fig09a_variance_matching{suffix}"
        if src.exists():
            shutil.copy(src, dst)
    src_live = output_dir / "_live_vcd" / "fig09a_variance_matching.json"
    if src_live.exists():
        dst_live_dir = fig_dir / "_live_vcd"
        dst_live_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_live, dst_live_dir / "fig09a_variance_matching.json")
    print(f"[var_pilot] Copied to {fig_dir / 'fig09a_variance_matching.pdf'}")
    plt.close()


if __name__ == "__main__":
    main()
