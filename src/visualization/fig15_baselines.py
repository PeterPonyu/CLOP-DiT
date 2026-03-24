"""
fig15_baselines.py — Fig 15: Baseline comparison (publication quality).

  O1: Grouped bar chart — FD\u2193, Centroid Cosine\u2191, Diversity Ratio\u2191, Coverage\u2191
  O2: Ranked dot plot showing normalised scores per method (replaces radar)
  O3: Absolute delta bar chart vs CLOP-DiT (replaces % improvement strip)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .direct_layout import bind_figure_region
from .explicit_positioning import add_shared_legend_axes
from .style import COLORS, FONT_SMALL, FONT_ANNOTATION, METHOD_COLORS, save_panel, style_axes, add_panel_label
from src.utils.paths import CACHE_DIR, RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)



def plot_baseline_comparison(
    gen_metrics_path: Optional[str] = None,
    div_metrics_path: Optional[str] = None,
    baseline_metrics_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    output_dir: Optional[Path] = None,
    dpi: int = 300,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Fig 15: CLOP-DiT vs baselines (publication quality, 3-panel).

    O1: Grouped bar chart with full metric names and direction arrows
    O2: Ranked dot plot (normalised [0,1], higher = better)
    O3: Absolute delta bar chart (CLOP-DiT minus baseline)
    """
    gen_metrics_path = gen_metrics_path or str(RESULTS_DIR / "generation_metrics.json")
    div_metrics_path = div_metrics_path or str(RESULTS_DIR / "diversity_diagnostics.json")
    baseline_metrics_path = baseline_metrics_path or str(RESULTS_DIR / "baseline_metrics.json")
    cache_dir = cache_dir or str(CACHE_DIR)
    output_dir = output_dir or FIG_DIR
    gen_path = Path(gen_metrics_path)
    if not gen_path.exists():
        logger.info("No generation metrics — skipping Fig 15")
        return None

    with open(gen_path) as f:
        gen_data = json.load(f)
    clop_metrics = gen_data.get("overall", {})
    clop_summary = gen_data.get("summary", {})

    # Load baselines
    baselines: Dict[str, Dict] = {}
    bl_path = Path(baseline_metrics_path)
    if bl_path.exists():
        with open(bl_path) as f:
            baselines = json.load(f)
    else:
        logger.info("Computing baselines on-the-fly...")
        baselines = _compute_baselines(cache_dir)

    if not baselines:
        logger.info("No baseline data — skipping Fig 15")
        return None

    # Load diversity for CLOP-DiT
    div_path = Path(div_metrics_path)
    div_ratio_clop = 0.0
    if div_path.exists():
        with open(div_path) as f:
            div_data = json.load(f)
        div_ratio_clop = div_data.get(
            "test1_intratype_diversity", {}
        ).get("summary", {}).get("mean_diversity_ratio", 0)

    # Build comparison data
    methods = {"CLOP-DiT": {
        "FD": clop_metrics.get("frechet_distance", 0),
        "Centroid Cosine": clop_summary.get("mean_centroid_cosine", 0),
        "Diversity Ratio": div_ratio_clop,
        "Coverage": clop_metrics.get("coverage", 0),
    }}
    for bl_name, bl_data in baselines.items():
        methods[bl_name] = {
            "FD": bl_data.get("frechet_distance", 0),
            "Centroid Cosine": bl_data.get("mean_centroid_cosine", 0),
            "Diversity Ratio": bl_data.get("diversity_ratio", 0),
            "Coverage": bl_data.get("coverage", 0),
        }

    method_names = list(methods.keys())
    n_methods = len(method_names)
    # Full metric names with direction arrows for axis labels
    metric_labels = ["FD \u2193", "Cosine \u2191", "Diversity \u2191", "Coverage \u2191"]
    metric_keys = ["FD", "Centroid Cosine", "Diversity Ratio", "Coverage"]
    # Direction: lower-is-better for FD, higher-is-better for others
    directions = ["lower", "higher", "higher", "higher"]

    fig = plt.figure(figsize=(15.2, 6.2))
    ax_rect_1, ax_rect_2, ax_rect_3 = bind_figure_region(fig, (0.04, 0.13, 0.98, 0.91)).split_cols(
        [1.20, 0.86, 1.20],
        gap=[0.060, 0.060],
    )

    # ── O1: Grouped bar chart ──
    ax = ax_rect_1.add_axes(fig)
    add_panel_label(ax, 'a', x=-0.12, y=1.02)
    x = np.arange(len(metric_labels))
    w = 0.8 / n_methods
    for i, mname in enumerate(method_names):
        vals = [methods[mname][k] for k in metric_keys]
        offset = (i - n_methods / 2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=mname,
               color=METHOD_COLORS.get(mname, COLORS["neutral"]),
               alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10, rotation=0, ha="center")
    handles_a, labels_a = ax.get_legend_handles_labels()
    style_axes(ax, "bar", title="Key Metrics Comparison", ylabel="Value")

    # ── O2: Ranked dot plot (normalised scores) ──
    ax2 = ax_rect_2.inset(left=0.020, right=0.010).add_axes(fig)
    add_panel_label(ax2, 'b', x=-0.12, y=1.02)

    # Normalise each metric to [0,1] with direction awareness
    all_vals = {k: [methods[m][k] for m in method_names] for k in metric_keys}
    normalized = {}
    for k, direction in zip(metric_keys, directions):
        mn, mx = min(all_vals[k]), max(all_vals[k])
        rng = mx - mn if mx > mn else 1
        if direction == "lower":
            normalized[k] = [(mx - v) / rng for v in all_vals[k]]
        else:
            normalized[k] = [(v - mn) / rng for v in all_vals[k]]

    # Compute aggregate normalised score per method
    agg_scores = {}
    for i, mname in enumerate(method_names):
        agg_scores[mname] = np.mean([normalized[k][i] for k in metric_keys])
    sorted_methods = sorted(method_names, key=lambda m: agg_scores[m], reverse=True)

    y_pos = np.arange(len(sorted_methods))
    for j, mk in enumerate(metric_keys):
        vals = [normalized[mk][method_names.index(m)] for m in sorted_methods]
        ax2.scatter(vals, y_pos, s=80, marker="oDsv"[j],
                    color=f"C{j}", alpha=0.85, zorder=3,
                    label=metric_labels[j])
    # Connect dots with lines for each method
    for i, mname in enumerate(sorted_methods):
        vals = [normalized[mk][method_names.index(mname)] for mk in metric_keys]
        ax2.plot(vals, [i] * len(vals), color=METHOD_COLORS.get(mname, "#999"),
                 linewidth=1.2, alpha=0.4, zorder=1)
        # Annotate aggregate score
        score_x = min(1.08, max(vals) + 0.04)
        ax2.text(score_x, i, f"{agg_scores[mname]:.2f}", va="center",
                 ha="left", fontsize=FONT_SMALL, color=COLORS["neutral"])

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_methods, fontsize=9)
    ax2.set_xlim(-0.05, 1.15)
    ax2.invert_yaxis()
    handles_b, labels_b = ax2.get_legend_handles_labels()
    style_axes(ax2, "default", title="Normalised Scores (1 = best)",
               xlabel="Normalised Value")

    # ── O3: Absolute delta bar chart (CLOP-DiT minus baseline) ──
    ax3 = ax_rect_3.inset(left=0.10, right=0.02).add_axes(fig)
    add_panel_label(ax3, 'c', x=-0.18, y=1.02)

    clop_vals = methods["CLOP-DiT"]
    bl_names = [bl for bl in baselines]
    y_labels = []
    y_vals = []

    for bl_name in bl_names:
        bl_vals = methods.get(bl_name, {})
        for mk, direction in zip(metric_keys, directions):
            cv = clop_vals.get(mk, 0)
            bv = bl_vals.get(mk, 0)
            if direction == "lower":
                # For FD, improvement = baseline - CLOP (positive = CLOP is better)
                delta = bv - cv
            else:
                # For higher-is-better, improvement = CLOP - baseline
                delta = cv - bv
            short_bl = bl_name[:14]
            y_labels.append(f"{mk[:3]} | {short_bl}")
            y_vals.append(delta)

    y_pos = np.arange(len(y_labels))
    bar_colors_final = [COLORS["good"] if v > 0 else COLORS["bad"] for v in y_vals]
    ax3.barh(y_pos, y_vals, color=bar_colors_final, height=0.6,
             edgecolor="white", linewidth=0.5, alpha=0.85)

    # Annotate values at bar tips
    for i, val in enumerate(y_vals):
        x_text = val + 0.03 if val >= 0 else min(-0.02, val + 0.08)
        ha = "left"
        ax3.text(x_text, i, f"{val:+.3f}", va="center", ha=ha,
                 fontsize=FONT_ANNOTATION, color="#333")

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(y_labels, fontsize=8, ha="right")
    ax3.axvline(x=0, color=COLORS["neutral"], linewidth=1.2)
    ax3.invert_yaxis()

    # Add group separators between baselines
    n_metrics = len(metric_keys)
    for g in range(1, len(bl_names)):
        sep_y = g * n_metrics - 0.5
        ax3.axhline(y=sep_y, color="#DDD", linewidth=1, linestyle="--")

    legend_ax_a = add_shared_legend_axes(fig, (ax.get_position().x0, 0.03, ax.get_position().width, 0.06))
    legend_ax_a.legend(handles_a, labels_a, fontsize=9, loc="center",
                       ncol=min(n_methods, 3), frameon=False)

    legend_ax_b = add_shared_legend_axes(fig, (ax2.get_position().x0, 0.005, ax2.get_position().width, 0.06))
    legend_ax_b.legend(handles_b, labels_b, fontsize=8, loc="center",
                       frameon=False, ncol=4, handletextpad=0.3, columnspacing=0.6)

    style_axes(ax3, "bar", title="CLOP-DiT Advantage (\u0394 metric)",
               xlabel="Absolute Improvement")
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

    if save:
        path = save_panel(fig, output_dir / "fig15_baseline_comparison.png", dpi)
        logger.info(f"Saved Fig 15 \u2192 {path}")
    return fig


def _compute_baselines(cache_dir: str = "data/cached_latents") -> Dict[str, Dict]:
    """Compute Gaussian and Shuffled baselines on-the-fly."""
    from src.evaluation.metrics import GenerationMetrics

    cache = Path(cache_dir)
    cell_path = cache / "cell_embeddings_dedup_preprocessed.npy"
    gid_path = cache / "text_group_ids_dedup.npy"
    gen_path = RESULTS_DIR / "generated_embeddings.npy"
    gen_lab_path = RESULTS_DIR / "generated_labels.npy"

    if not all(p.exists() for p in [cell_path, gid_path, gen_path, gen_lab_path]):
        return {}

    real_cells = np.load(cell_path)
    group_ids = np.load(gid_path)
    gen_cells = np.load(gen_path)
    gen_labels = np.load(gen_lab_path)

    unique_types = np.sort(np.unique(group_ids))
    rng = np.random.default_rng(42)

    # Gaussian baseline
    gauss_cells = []
    gauss_labels = []
    n_per = len(gen_cells) // len(unique_types) if len(unique_types) > 0 else 100
    for tid in unique_types:
        r = real_cells[group_ids == tid]
        centroid = r.mean(axis=0)
        std_val = r.std()
        samples = rng.normal(0, std_val, size=(n_per, real_cells.shape[1]))
        samples += centroid
        norms = np.linalg.norm(samples, axis=1, keepdims=True) + 1e-8
        samples = samples / norms
        gauss_cells.append(samples)
        gauss_labels.extend([tid] * n_per)
    gauss_cells = np.concatenate(gauss_cells, axis=0)
    gauss_labels = np.array(gauss_labels)

    n_sub = min(5000, len(gauss_cells), len(real_cells))
    r_idx = rng.choice(len(real_cells), n_sub, replace=False)
    g_idx = rng.choice(len(gauss_cells), n_sub, replace=False)
    gauss_overall = GenerationMetrics.full_evaluation(real_cells[r_idx], gauss_cells[g_idx])
    gauss_cosines = []
    gauss_div_ratios = []
    for tid in unique_types:
        r = real_cells[group_ids == tid]
        g = gauss_cells[gauss_labels == tid]
        if len(r) < 5 or len(g) < 5:
            continue
        rc = r.mean(0); rc /= np.linalg.norm(rc) + 1e-8
        gc = g.mean(0); gc /= np.linalg.norm(gc) + 1e-8
        gauss_cosines.append(float(np.dot(rc, gc)))
        r_sub = r[rng.choice(len(r), min(100, len(r)), replace=False)]
        g_sub = g[rng.choice(len(g), min(100, len(g)), replace=False)]
        r_n = r_sub / (np.linalg.norm(r_sub, axis=1, keepdims=True) + 1e-8)
        g_n = g_sub / (np.linalg.norm(g_sub, axis=1, keepdims=True) + 1e-8)
        rr_sim = (r_n @ r_n.T)[np.triu_indices(len(r_n), k=1)].mean()
        gg_sim = (g_n @ g_n.T)[np.triu_indices(len(g_n), k=1)].mean()
        real_div = 1.0 - rr_sim
        gen_div = 1.0 - gg_sim
        if real_div > 1e-6:
            gauss_div_ratios.append(gen_div / real_div)

    # Shuffled baseline
    shuffled_labels = gen_labels.copy()
    rng.shuffle(shuffled_labels)
    shuf_cosines = []
    for tid in unique_types:
        r = real_cells[group_ids == tid]
        g = gen_cells[shuffled_labels == tid]
        if len(r) < 5 or len(g) < 5:
            continue
        rc = r.mean(0); rc /= np.linalg.norm(rc) + 1e-8
        gc = g.mean(0); gc /= np.linalg.norm(gc) + 1e-8
        shuf_cosines.append(float(np.dot(rc, gc)))

    shuf_overall = GenerationMetrics.full_evaluation(
        real_cells[r_idx],
        gen_cells[rng.choice(len(gen_cells), n_sub, replace=False)]
    )

    # Random N(0,I) baseline — uninformative prior
    rand_cells = rng.standard_normal(size=gen_cells.shape)
    rand_cells = rand_cells / (np.linalg.norm(rand_cells, axis=1, keepdims=True) + 1e-8)
    rand_labels = rng.choice(unique_types, size=len(gen_cells))
    rand_overall = GenerationMetrics.full_evaluation(
        real_cells[r_idx],
        rand_cells[rng.choice(len(rand_cells), n_sub, replace=False)]
    )
    rand_cosines = []
    rand_div_ratios = []
    for tid in unique_types:
        r = real_cells[group_ids == tid]
        g = rand_cells[rand_labels == tid]
        if len(r) < 5 or len(g) < 5:
            continue
        rc = r.mean(0); rc /= np.linalg.norm(rc) + 1e-8
        gc = g.mean(0); gc /= np.linalg.norm(gc) + 1e-8
        rand_cosines.append(float(np.dot(rc, gc)))
        g_sub = g[rng.choice(len(g), min(100, len(g)), replace=False)]
        g_n = g_sub / (np.linalg.norm(g_sub, axis=1, keepdims=True) + 1e-8)
        gg_sim = (g_n @ g_n.T)[np.triu_indices(len(g_n), k=1)].mean()
        gen_div = 1.0 - gg_sim
        r_sub = r[rng.choice(len(r), min(100, len(r)), replace=False)]
        r_n = r_sub / (np.linalg.norm(r_sub, axis=1, keepdims=True) + 1e-8)
        rr_sim = (r_n @ r_n.T)[np.triu_indices(len(r_n), k=1)].mean()
        real_div = 1.0 - rr_sim
        if real_div > 1e-6:
            rand_div_ratios.append(gen_div / real_div)

    # Mean-only baseline — centroid collapse (zero diversity)
    mean_cells = []
    mean_labels = []
    for tid in unique_types:
        r = real_cells[group_ids == tid]
        centroid = r.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        mean_cells.append(np.tile(centroid, (n_per, 1)))
        mean_labels.extend([tid] * n_per)
    mean_cells = np.concatenate(mean_cells, axis=0)
    mean_labels = np.array(mean_labels)
    mean_overall = GenerationMetrics.full_evaluation(
        real_cells[r_idx],
        mean_cells[rng.choice(len(mean_cells), n_sub, replace=False)]
    )
    mean_cosines = []
    for tid in unique_types:
        r = real_cells[group_ids == tid]
        g = mean_cells[mean_labels == tid]
        if len(r) < 5 or len(g) < 5:
            continue
        rc = r.mean(0); rc /= np.linalg.norm(rc) + 1e-8
        gc = g.mean(0); gc /= np.linalg.norm(gc) + 1e-8
        mean_cosines.append(float(np.dot(rc, gc)))

    baselines = {
        "Gaussian N(\u03bc,\u03c3\u00b2I)": {
            "frechet_distance": gauss_overall.get("frechet_distance", 0),
            "mean_centroid_cosine": float(np.mean(gauss_cosines)) if gauss_cosines else 0,
            "diversity_ratio": float(np.mean(gauss_div_ratios)) if gauss_div_ratios else 0,
            "coverage": gauss_overall.get("coverage", 0),
        },
        "Shuffled Labels": {
            "frechet_distance": shuf_overall.get("frechet_distance", 0),
            "mean_centroid_cosine": float(np.mean(shuf_cosines)) if shuf_cosines else 0,
            "diversity_ratio": 1.0,
            "coverage": shuf_overall.get("coverage", 0),
        },
        "Random N(0,I)": {
            "frechet_distance": rand_overall.get("frechet_distance", 0),
            "mean_centroid_cosine": float(np.mean(rand_cosines)) if rand_cosines else 0,
            "diversity_ratio": float(np.mean(rand_div_ratios)) if rand_div_ratios else 0,
            "coverage": rand_overall.get("coverage", 0),
        },
        "Mean-only (collapse)": {
            "frechet_distance": mean_overall.get("frechet_distance", 0),
            "mean_centroid_cosine": float(np.mean(mean_cosines)) if mean_cosines else 0,
            "diversity_ratio": 0.0,
            "coverage": mean_overall.get("coverage", 0),
        },
    }

    bl_path = RESULTS_DIR / "baseline_metrics.json"
    with open(bl_path, "w") as f:
        json.dump(baselines, f, indent=2)
    logger.info(f"Saved baseline metrics → {bl_path}")
    return baselines
