"""
fig05_metrics.py — Fig 05 (metrics summary) and diversity violin for CLOP-DiT.

Fig 05: Training convergence, quality radar, diversity gauges, expression fidelity.
Diversity violin: Intra-type cosine distributions for the most shifted cell types.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..utils.constants import RANDOM_SEED
from .direct_layout import bind_figure_region
from .style import COLORS, FONT_LEGEND_DENSE, FONT_LABEL, FONT_SMALL, FONT_TITLE, FONT_TICK_DENSE, FONT_ANNOTATION, abbreviate_cell_type, add_panel_label, apply_style, save_with_vcd, set_figure_suptitle
from ._utils import sample_pairwise_cosines
from src.utils.paths import CACHE_DIR, RESULTS_DIR, FIG_DIR, CHECKPOINT_DIR

logger = logging.getLogger(__name__)


def plot_diversity_distributions_violin(
    cache_dir: Optional[str] = None,
    div_metrics_path: Optional[str] = None,
    generated_path: Optional[str] = None,
    generated_labels_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    dpi: int = 300,
    save: bool = True,
    ax: Optional[plt.Axes] = None,
    top_n: int = 6,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Violin comparison of intra-type cosine distributions for the most shifted cell types."""
    cache_dir = cache_dir or str(CACHE_DIR)
    div_metrics_path = div_metrics_path or str(RESULTS_DIR / "diversity_diagnostics.json")
    output_dir = output_dir or str(FIG_DIR)
    div_path = Path(div_metrics_path)
    cache = Path(cache_dir)
    real_path = cache / "cell_embeddings_dedup_preprocessed.npy"
    real_labels_path = cache / "text_group_ids_dedup.npy"

    if generated_path is None:
        for candidate in [str(RESULTS_DIR / "generated_embeddings.npy"), str(CHECKPOINT_DIR / "generated_cells.npy")]:
            if Path(candidate).exists():
                generated_path = candidate
                break
    if generated_labels_path is None and (RESULTS_DIR / "generated_labels.npy").exists():
        generated_labels_path = str(RESULTS_DIR / "generated_labels.npy")

    needed_paths = [div_path, real_path, real_labels_path]
    if generated_path is not None:
        needed_paths.append(Path(generated_path))
    if generated_labels_path is not None:
        needed_paths.append(Path(generated_labels_path))
    if any(not p.exists() for p in needed_paths):
        logger.warning("Missing diversity violin inputs — skipping violin enhancement")
        return None

    with open(div_path) as f:
        div_data = json.load(f)

    per_type = div_data.get("test1_intratype_diversity", {}).get("per_type", {})
    if not per_type:
        logger.warning("No per-type diversity diagnostics found — skipping violin enhancement")
        return None

    ranked = []
    for name, entry in per_type.items():
        rr = entry.get("real_real_cos", {}).get("mean", 0.0)
        gg = entry.get("gen_gen_cos", {}).get("mean", 0.0)
        rg = entry.get("real_gen_cos", {}).get("mean", 0.0)
        gap = abs(gg - rr) + 0.5 * abs(rg - rr)
        ranked.append((gap, name, entry))
    ranked.sort(reverse=True)
    selected = ranked[:top_n]

    real_embeddings = np.load(real_path)
    real_labels = np.load(real_labels_path)
    gen_embeddings = np.load(Path(generated_path))
    gen_labels = np.load(Path(generated_labels_path))

    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=(13.0, 4.6))
        ax = bind_figure_region(fig, (0.06, 0.16, 0.98, 0.92)).add_axes(fig)
    else:
        fig = ax.figure

    violin_data = []
    violin_positions = []
    violin_colors = []
    xtick_positions = []
    xtick_labels = []
    mean_points_x = []
    mean_points_y = []
    mean_points_c = []

    colors = {
        "Real–Real": COLORS["real"],
        "Gen–Gen": COLORS["generated"],
        "Real–Gen": COLORS["neutral"],
    }

    for idx, (_, name, entry) in enumerate(selected):
        type_id = entry.get("type_id")
        if type_id is None:
            continue
        real_subset = real_embeddings[real_labels == type_id]
        gen_subset = gen_embeddings[gen_labels == type_id]
        if len(real_subset) < 2 or len(gen_subset) < 2:
            continue

        rr = sample_pairwise_cosines(real_subset, None, seed=RANDOM_SEED + idx)
        gg = sample_pairwise_cosines(gen_subset, None, seed=RANDOM_SEED + 100 + idx)
        rg = sample_pairwise_cosines(real_subset, gen_subset, seed=RANDOM_SEED + 200 + idx)
        if min(len(rr), len(gg), len(rg)) == 0:
            continue

        base = idx * 3.2
        positions = [base - 0.55, base, base + 0.55]
        datasets = [rr, gg, rg]
        labels = ["Real–Real", "Gen–Gen", "Real–Gen"]
        for pos, dataset, label in zip(positions, datasets, labels):
            violin_data.append(dataset)
            violin_positions.append(pos)
            violin_colors.append(colors[label])
            mean_points_x.append(pos)
            mean_points_y.append(float(np.mean(dataset)))
            mean_points_c.append(colors[label])

        xtick_positions.append(base)
        short_name = abbreviate_cell_type(name, 24)
        xtick_labels.append(short_name)

    if not violin_data:
        logger.warning("No diversity violin samples could be computed")
        if created_fig:
            plt.close(fig)
        return None

    parts = ax.violinplot(
        violin_data,
        positions=violin_positions,
        widths=0.5,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], violin_colors):
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(0.65)
    parts["cmedians"].set_color(COLORS["median_dark"])
    parts["cmedians"].set_linewidth(1.1)

    ax.scatter(
        mean_points_x,
        mean_points_y,
        s=20,
        c=mean_points_c,
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
        clip_on=True,
    )
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Pairwise Cosine Similarity")
    ax.set_title("Diversity Distribution Tails (most shifted cell types)")
    ax.grid(True, axis="y", alpha=0.22)
    all_min = min(float(np.min(v)) for v in violin_data)
    all_max = max(float(np.max(v)) for v in violin_data)
    y_lo = min(-0.15, all_min - 0.02)
    y_hi = max(0.28, all_max + 0.02)
    ax.set_ylim(y_lo, y_hi)
    ax.text(
        0.01,
        0.98,
        "Selection criterion: largest real/gen intra-type cosine gap",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=COLORS["neutral"],
    )
    for label, color in colors.items():
        ax.plot([], [], color=color, linewidth=6, alpha=0.8, label=label)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        fontsize=FONT_LEGEND_DENSE,
        ncol=3,
        frameon=False,
        borderaxespad=0.2,
        handlelength=1.6,
        columnspacing=1.0,
    )

    if created_fig and save:
        path = Path(output_dir) / "fig05_diversity_distributions_violin.png"
        if save_panel_fn is not None:
            save_panel_fn(fig, path, dpi)
        else:
            save_with_vcd(fig, path, dpi)
    return fig


def plot_metrics_summary(
    clop_hist: Optional[Dict] = None,
    dit_hist: Optional[Dict] = None,
    gen_metrics_path: Optional[str] = None,
    expr_metrics_path: Optional[str] = None,
    div_metrics_path: Optional[str] = None,
    benchmark_report_path: Optional[str] = None,
    bootstrap_cis_path: Optional[str] = None,
    baseline_metrics_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Visual metrics dashboard -- replaces table with bar charts + radar.

    D1: Training convergence (horizontal bars for final key metrics)
        + bootstrap 95% CI whiskers where available
    D2: Generation quality radar (FD, coverage, diversity, centroid cos, expr r)
        + Gaussian baseline overlay for comparison
        + numeric annotations on each vertex
    D3: Diversity gauges (diversity ratio, collapsed types, cond gain)
        + bootstrap 95% CI whiskers + baseline reference markers
    D4: Configuration + expression summary (compact annotated bars)
        + r=0.9999 reference line + headline summary text box
    """
    from matplotlib.patches import FancyBboxPatch  # noqa: F401 (kept for parity)

    gen_metrics_path = gen_metrics_path or str(RESULTS_DIR / "generation_metrics.json")
    expr_metrics_path = expr_metrics_path or str(RESULTS_DIR / "expression_metrics.json")
    div_metrics_path = div_metrics_path or str(RESULTS_DIR / "diversity_diagnostics.json")
    benchmark_report_path = benchmark_report_path or str(RESULTS_DIR / "benchmark_report.json")
    bootstrap_cis_path = bootstrap_cis_path or str(RESULTS_DIR / "bootstrap_cis.json")
    baseline_metrics_path = baseline_metrics_path or str(RESULTS_DIR / "baseline_metrics.json")
    output_dir = output_dir or str(FIG_DIR)

    # ── Collect all data ──
    train_metrics: Dict[str, float] = {}
    if clop_hist:
        h = clop_hist
        train_metrics["CLOP Val Loss"] = h["val_loss"][-1]
        train_metrics["Proto Accuracy"] = h["val_proto_acc"][-1]
        train_metrics["Top-5 Accuracy"] = h["val_proto_top5"][-1]
        train_metrics["Text-Cell Align"] = h.get("val_text_cell_align", h.get("val_proto_acc", [0.0]))[-1]
    if dit_hist:
        h = dit_hist
        train_metrics["DiT Val Loss"] = h["val_loss"][-1]
        if "val_cosine_sim" in h:
            train_metrics["DiT Val Cosine"] = h["val_cosine_sim"][-1]

    gen_metrics: Dict[str, float] = {}
    gen_path = Path(gen_metrics_path)
    if gen_path.exists():
        with open(gen_path) as f:
            gen_data = json.load(f)
        overall = gen_data.get("overall", {})
        summary = gen_data.get("summary", {})
        gen_metrics = {
            "FD": overall.get("frechet_distance", 0),
            "MMD": overall.get("mmd_rbf", 0),
            "Coverage": overall.get("coverage", 0),
            "Density": overall.get("density", 0),
            "Centroid Cos": summary.get("mean_centroid_cosine", 0),
        }

    div_metrics: Dict[str, float] = {}
    div_path = Path(div_metrics_path)
    if div_path.exists():
        with open(div_path) as f:
            div_data = json.load(f)
        t1 = div_data.get("test1_intratype_diversity", {}).get("summary", {})
        t2 = div_data.get("test2_memorization", {})
        t5 = div_data.get("test5_condition_sensitivity", {}).get("summary", {})
        div_metrics = {
            "Diversity Ratio": t1.get("mean_diversity_ratio", 0),
            "Collapsed": t1.get("n_collapsed", 0),
            "Total Types": t1.get("n_collapsed", 0) + t1.get("n_healthy", 0),
            "NN Distance": t2.get("nn_cosine_distance", {}).get("mean", 0),
            "Near-copies": t2.get("n_very_close", 0),
            "Cond Gain": t5.get("mean_diversity_gain", 0),
        }

    expr_metrics: Dict[str, float] = {}
    expr_path = Path(expr_metrics_path)
    if expr_path.exists():
        with open(expr_path) as f:
            expr_data = json.load(f)
        gene_corr = expr_data.get("gene_correlation", {})
        per_type_sum = expr_data.get("per_type_summary", {})
        expr_metrics = {
            "Gene Pearson r": gene_corr.get("pearson_r", 0),
            "Gene Spearman": gene_corr.get("spearman_rho", 0),
            "Per-Type r (mean)": per_type_sum.get("mean_pearson_r", 0),
            "Per-Type r (min)": per_type_sum.get("min_pearson_r", 0),
            "Genes": gene_corr.get("n_genes_compared", 0),
        }

    cfg_meta: Dict = {}
    gen_meta_path = RESULTS_DIR / "generation_metadata.json"
    if gen_meta_path.exists():
        with open(gen_meta_path) as f:
            cfg_meta = json.load(f)

    core_metrics: Dict[str, float] = {}
    bench_path = Path(benchmark_report_path)
    if bench_path.exists():
        try:
            with open(bench_path) as f:
                bench = json.load(f)
            # tolerate schema drift
            candidates = [
                bench.get("core_metrics", {}),
                bench.get("operating_point", {}),
                bench.get("results", {}).get("core_metrics", {}),
            ]
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                if "knn_top1" in c:
                    core_metrics["KNN-1"] = float(c["knn_top1"])
                if "steering_accuracy" in c:
                    core_metrics["Steering"] = float(c["steering_accuracy"])
                if "diversity_ratio" in c:
                    core_metrics["Diversity Ratio"] = float(c["diversity_ratio"])
                if "linear_acc" in c:
                    core_metrics["Linear Acc"] = float(c["linear_acc"])
        except Exception:
            core_metrics = {}

    # ── Load bootstrap CIs ──
    bootstrap_cis: Dict = {}
    bci_path = Path(bootstrap_cis_path)
    if bci_path.exists():
        try:
            with open(bci_path) as f:
                bci_data = json.load(f)
            bootstrap_cis = bci_data.get("metrics", {})
        except Exception:
            bootstrap_cis = {}

    # ── Load baseline metrics for comparison ──
    baseline_data: Dict = {}
    bl_path = Path(baseline_metrics_path)
    if bl_path.exists():
        try:
            with open(bl_path) as f:
                baseline_data = json.load(f)
        except Exception:
            baseline_data = {}

    # Also extract baselines from benchmark report (richer data)
    bench_baselines: Dict = {}
    if bench_path.exists():
        try:
            with open(bench_path) as f:
                bench_full = json.load(f)
            bench_methods = bench_full.get("methods", {})
            for method_name, method_data in bench_methods.items():
                if method_name != "CLOP-DiT":
                    bench_baselines[method_name] = method_data
        except Exception:
            bench_baselines = {}

    if not train_metrics and not gen_metrics:
        return None

    fig = plt.figure(figsize=(15.2, 8.4))
    layout = bind_figure_region(fig, (0.07, 0.08, 0.94, 0.95))
    top_row, bottom_row = layout.split_rows(2, hspace=0.34)
    top_left, top_right = top_row.split_cols([1.12, 1.0], wspace=0.50)
    bottom_left, bottom_right = bottom_row.split_cols([1.12, 1.0], wspace=0.44)
    # Note: Figure-level title removed per revision requirements

    # Map from display metric names to bootstrap_cis keys
    _bci_key_map = {
        "KNN-1": "knn_top1",
        "Linear Acc": "linear_acc",
        "Steering": "steering",
        "Diversity Ratio": "diversity_ratio",
    }

    # ── D1: Training convergence bars ──
    ax1 = top_left.add_axes(fig)
    if train_metrics:
        if core_metrics:
            train_metrics = {**core_metrics, **train_metrics}
        names = list(train_metrics.keys())
        metric_abbrev = {
            "CLOP Val Loss": "CLOP Loss",
            "Proto Accuracy": "Proto Acc",
            "Top-5 Accuracy": "Top-5 Acc",
            "Text-Cell Align": "Text-Cell",
            "DiT Val Loss": "DiT Loss",
            "DiT Val Cosine": "DiT Cosine",
        }
        vals = list(train_metrics.values())
        display_vals = []
        for n, v in zip(names, vals):
            if "Loss" in n:
                display_vals.append(v)
            else:
                display_vals.append(v * 100 if v <= 1.0 else v)
        colors_d1 = []
        for n, v in zip(names, vals):
            if "Loss" in n:
                colors_d1.append(COLORS["bad"] if v > 1.0 else COLORS["warn"] if v > 0.1 else COLORS["good"])
            else:
                colors_d1.append(COLORS["good"] if v > 0.8 else COLORS["warn"] if v > 0.5 else COLORS["bad"])

        y_pos = np.arange(len(names))
        bars = ax1.barh(y_pos, display_vals, color=colors_d1, height=0.6,
                        edgecolor="white", linewidth=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([metric_abbrev.get(name, name) for name in names], fontsize=10)

        # Add value labels and bootstrap CI whiskers
        for i, (bar, dv, n) in enumerate(zip(bars, display_vals, names)):
            unit = "" if "Loss" in n else "%"
            bci_key = _bci_key_map.get(n)
            ci_text = ""
            if bci_key and bci_key in bootstrap_cis:
                ci = bootstrap_cis[bci_key]
                ci_lo = ci.get("ci_95_lower", 0)
                ci_hi = ci.get("ci_95_upper", 0)
                # Convert to display scale (percentage for non-loss metrics)
                if "Loss" not in n:
                    ci_lo_d = ci_lo * 100 if ci_lo <= 1.0 else ci_lo
                    ci_hi_d = ci_hi * 100 if ci_hi <= 1.0 else ci_hi
                else:
                    ci_lo_d = ci_lo
                    ci_hi_d = ci_hi
                # Draw error bar whisker
                bar_center = bar.get_y() + bar.get_height() / 2
                ax1.plot([ci_lo_d, ci_hi_d], [bar_center, bar_center],
                         color=COLORS["annotation_dark"], linewidth=1.2, zorder=5)
                ax1.plot([ci_lo_d, ci_lo_d], [bar_center - 0.12, bar_center + 0.12],
                         color=COLORS["annotation_dark"], linewidth=1.0, zorder=5)
                ax1.plot([ci_hi_d, ci_hi_d], [bar_center - 0.12, bar_center + 0.12],
                         color=COLORS["annotation_dark"], linewidth=1.0, zorder=5)
                ci_text = f" [{ci_lo * 100:.1f}, {ci_hi * 100:.1f}]" if "Loss" not in n else ""

            fmt = f"{dv:.4f}" if "Loss" in n else f"{dv:.1f}{unit}"
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     fmt + ci_text, va="center", fontsize=FONT_SMALL)
        ax1.set_xlabel("Value (accuracy shown as %)")
        ax1.set_title("Training Convergence", fontsize=12)
        ax1.invert_yaxis()
        from matplotlib.ticker import MaxNLocator as _MNL
        ax1.xaxis.set_major_locator(_MNL(nbins=5, prune="both"))
        ax1.grid(axis='both', alpha=0.15, linestyle='--')

    else:
        ax1.text(0.5, 0.5, "No training history available",
                 ha="center", va="center", transform=ax1.transAxes,
                 fontsize=10, color=COLORS["neutral"])
        ax1.set_title("Training Convergence", fontsize=12)
    add_panel_label(ax1, 'a', x=-0.10, y=1.05)

    # ── D2: Generation quality bar chart (replaces radar for clarity) ──
    ax2 = top_right.add_axes(fig)
    if gen_metrics or expr_metrics:
        bar_labels = []
        bar_vals = []
        if gen_metrics:
            fd_score = max(0, 1.0 - gen_metrics.get("FD", 1.0))
            bar_labels.append("FD\n(inv)")
            bar_vals.append(fd_score)
            bar_labels.append("Coverage")
            bar_vals.append(gen_metrics.get("Coverage", 0))
            bar_labels.append("Centroid\nCos")
            bar_vals.append(gen_metrics.get("Centroid Cos", 0))
        if div_metrics:
            bar_labels.append("Diversity\nRatio")
            bar_vals.append(div_metrics.get("Diversity Ratio", 0))
        if expr_metrics:
            bar_labels.append("Gene\nCorr")
            bar_vals.append(expr_metrics.get("Gene Pearson r", 0))

        if bar_vals:
            gauss_bl = baseline_data.get("Gaussian N(\u03bc,\u03c3\u00b2I)", {})
            if not gauss_bl:
                gauss_bl = bench_baselines.get("Gaussian N(\u03bc,\u03c3\u00b2I)", {})

            baseline_vals = []
            for lbl in bar_labels:
                if "FD" in lbl:
                    baseline_vals.append(max(0, 1.0 - gauss_bl.get("frechet_distance", 1.0)) if gauss_bl else 0.0)
                elif "Coverage" in lbl:
                    baseline_vals.append(gauss_bl.get("coverage", 0.0) if gauss_bl else 0.0)
                elif "Centroid" in lbl:
                    baseline_vals.append(gauss_bl.get("mean_centroid_cosine", 0.0) if gauss_bl else 0.0)
                elif "Diversity" in lbl:
                    baseline_vals.append(min(gauss_bl.get("diversity_ratio", 0.0), 1.0) if gauss_bl else 0.0)
                elif "Gene" in lbl:
                    baseline_vals.append(0.0)
                else:
                    baseline_vals.append(0.0)

            x_pos = np.arange(len(bar_vals))
            width = 0.34 if gauss_bl else 0.58
            bars_clop = ax2.bar(x_pos - (width / 2 if gauss_bl else 0.0), bar_vals,
                                color=COLORS["real"], alpha=0.88,
                                edgecolor="white", linewidth=0.8, width=width,
                                label="CLOP-DiT")
            bars_gauss = []
            if gauss_bl:
                bars_gauss = ax2.bar(x_pos + width / 2, baseline_vals,
                                     color=COLORS["baseline_gauss"], alpha=0.72,
                                     edgecolor="white", linewidth=0.8, width=width,
                                     label="Gaussian baseline")
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(bar_labels, fontsize=FONT_TICK_DENSE)
            ax2.set_ylim(0, 1.15)
            ax2.set_ylabel("Score", fontsize=FONT_LABEL)
            # Add value annotations on bars.
            for bar, val in zip(bars_clop, bar_vals):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f"{val:.3f}", ha="center", va="bottom",
                         fontsize=FONT_ANNOTATION, fontweight="normal",
                         color=COLORS["real"], zorder=8)
            for bar, val in zip(bars_gauss, baseline_vals):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f"{val:.3f}", ha="center", va="bottom",
                         fontsize=FONT_SMALL, color=COLORS["baseline_gauss"], zorder=8)
            if gauss_bl:
                ax2.legend(fontsize=FONT_LEGEND_DENSE, frameon=False, loc="upper left", ncol=2)
        ax2.set_title("Quality Profile", fontsize=FONT_TITLE)
        add_panel_label(ax2, 'b', x=-0.10, y=1.05)
    else:
        ax2.text(0.5, 0.5, "No generation data", ha="center",
                 va="center", transform=ax2.transAxes)
        ax2.set_title("Generation Quality Profile")
        add_panel_label(ax2, 'b', x=-0.10, y=1.05)

    # ── D3: Diversity gauges ──
    ax3 = bottom_left.add_axes(fig)
    if div_metrics:
        gauge_items = [
            ("Diversity\nRatio", div_metrics.get("Diversity Ratio", 0), 1.0,
             COLORS["good"] if div_metrics.get("Diversity Ratio", 0) >= 0.8 else
             COLORS["warn"] if div_metrics.get("Diversity Ratio", 0) >= 0.5 else COLORS["bad"],
             "diversity_ratio"),
            ("Cond\nGain", div_metrics.get("Cond Gain", 0), 3.0,
             COLORS["good"] if div_metrics.get("Cond Gain", 0) >= 1.5 else
             COLORS["warn"] if div_metrics.get("Cond Gain", 0) >= 1.0 else COLORS["bad"],
             None),
            ("NN\nDistance", div_metrics.get("NN Distance", 0), 1.0,
             COLORS["good"] if div_metrics.get("NN Distance", 0) >= 0.3 else
             COLORS["warn"] if div_metrics.get("NN Distance", 0) >= 0.1 else COLORS["bad"],
             None),
        ]

        # Get Gaussian baseline values for reference markers
        gauss_bl_d3 = baseline_data.get("Gaussian N(\u03bc,\u03c3\u00b2I)", {})
        if not gauss_bl_d3:
            gauss_bl_d3 = bench_baselines.get("Gaussian N(\u03bc,\u03c3\u00b2I)", {})
        gauss_div_ratio = gauss_bl_d3.get("diversity_ratio", None)

        gauge_text_x = max(item[2] for item in gauge_items) + 0.22
        for i, (label, val, max_val, color, bci_key) in enumerate(gauge_items):
            ax3.barh(i, max_val, height=0.5, color=COLORS["bg_gauge"],
                     edgecolor="none", zorder=1)
            ax3.barh(i, min(val, max_val), height=0.5, color=color,
                     edgecolor="white", linewidth=0.8, zorder=2)

            # Build annotation text with CI if available
            ci_text = ""
            display_val = val
            if bci_key and bci_key in bootstrap_cis:
                ci = bootstrap_cis[bci_key]
                ci_lo = ci.get("ci_95_lower", 0)
                ci_hi = ci.get("ci_95_upper", 0)
                # Use bootstrap point estimate for consistency with CI bounds
                display_val = ci.get("mean", ci.get("point_estimate", val))
                ci_text = f"  [{ci_lo:.3f}, {ci_hi:.3f}]"

            ann_text = f"{display_val:.3f}" + ci_text
            ax3.text(gauge_text_x, i, ann_text,
                     va="center", fontsize=10, zorder=3)


        ax3.set_yticks(range(len(gauge_items)))
        ax3.set_yticklabels([g[0] for g in gauge_items], fontsize=10)
        ax3.invert_yaxis()
        ax3.set_xlim(0, gauge_text_x + 0.70)

        collapsed = div_metrics.get("Collapsed", 0)
        total = div_metrics.get("Total Types", 69)
        copies = div_metrics.get("Near-copies", 0)
        ax3.text(0.98, 0.02,
                 f"Collapsed: {collapsed}/{total} | Near-copies: {copies}",
                 transform=ax3.transAxes, ha="right", va="bottom",
                 fontsize=10, color=COLORS["neutral"])
        ax3.set_title("Diversity Health", fontsize=12)
        ax3.set_xlabel("Score")
        from matplotlib.ticker import MaxNLocator as _MNL3
        ax3.xaxis.set_major_locator(_MNL3(nbins=5, prune="both"))
        ax3.grid(axis='x', alpha=0.15, linestyle='--')
    else:
        ax3.text(0.5, 0.5, "No diversity data", ha="center", va="center",
                 transform=ax3.transAxes)
        ax3.set_title("Diversity Health")
    add_panel_label(ax3, 'c', x=-0.10, y=1.05)

    # ── D4: Expression fidelity + config ──
    ax4 = bottom_right.add_axes(fig)
    if expr_metrics:
        expr_items = [
            ("Gene Pearson r", expr_metrics.get("Gene Pearson r", 0)),
            ("Gene Spearman", expr_metrics.get("Gene Spearman", 0)),
            ("Per-Type r (mean)", expr_metrics.get("Per-Type r (mean)", 0)),
            ("Per-Type r (min)", expr_metrics.get("Per-Type r (min)", 0)),
        ]
        y_pos = np.arange(len(expr_items))
        # Plot deviation (1 - r) on log scale instead of raw values
        deviations = [max(1 - v, 1e-12) for _, v in expr_items]
        single_color = COLORS["real"]
        bars = ax4.barh(y_pos, deviations, color=single_color, height=0.5,
                        edgecolor="white", linewidth=0.8)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([n for n, _ in expr_items], fontsize=10)
        for i, (bar, dev) in enumerate(zip(bars, deviations)):
            ax4.text(bar.get_width() * 1.3, bar.get_y() + bar.get_height() / 2,
                     f"{dev:.1e}", va="center", fontsize=9)
        ax4.invert_yaxis()
        ax4.set_xscale('log')
        ax4.set_xlabel("Deviation (1 \u2212 r)", fontsize=FONT_LABEL)
        ax4.set_title("Expression Fidelity", fontsize=12)
        ax4.grid(axis='both', alpha=0.15, linestyle='--')
        ax4.text(0.98, 0.02, "Lower = better (log scale)",
                 transform=ax4.transAxes, ha="right", va="bottom",
                 fontsize=FONT_SMALL, color=COLORS["neutral"], style="italic")

    else:
        ax4.text(0.5, 0.5, "No expression data", ha="center", va="center",
                 transform=ax4.transAxes)
        ax4.set_title("Expression Fidelity")
    add_panel_label(ax4, 'd', x=-0.10, y=1.05)

    if save:
        path = Path(output_dir) / "fig03a_metrics_summary.png"
        if save_panel_fn is not None:
            save_panel_fn(fig, path, dpi)
        else:
            save_with_vcd(fig, path, dpi)
    return fig
