"""
baseline_panels.py — Panel O: Baseline comparison with graphical O3.

  O1: Grouped bar chart — FD, centroid cosine, diversity ratio, coverage
  O2: Polar radar chart with normalized metrics
  O3: Relative improvement strip (graphical, no tables)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, save_panel, style_axes, GRIDSPEC_TIGHT, set_dense_tick_labels

logger = logging.getLogger(__name__)

METHOD_COLORS = ["#1976D2", "#FF7043", "#4CAF50", "#9C27B0", "#FFC107"]


def plot_baseline_comparison(
    gen_metrics_path: str = "results/generation_metrics.json",
    div_metrics_path: str = "results/diversity_diagnostics.json",
    baseline_metrics_path: str = "results/baseline_metrics.json",
    cache_dir: str = "data/cached_latents_v5.2",
    output_dir: Path = Path("results/figures"),
    dpi: int = 300,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Panel O: CLOP-DiT vs baselines with graphical O3 (no table).

    O1: Grouped bar chart
    O2: Polar radar
    O3: Relative improvement strip (horizontal bars showing % improvement)
    """
    gen_path = Path(gen_metrics_path)
    if not gen_path.exists():
        logger.info("No generation metrics — skipping Panel O")
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
        logger.info("No baseline data — skipping Panel O")
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
    metric_labels = ["FD ↓", "Centroid Cosine ↑", "Diversity Ratio ↑", "Coverage ↑"]
    metric_keys = ["FD", "Centroid Cosine", "Diversity Ratio", "Coverage"]

    fig = plt.figure(figsize=(22, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], **GRIDSPEC_TIGHT)
    fig.suptitle("Baseline Comparison — CLOP-DiT vs Simple Baselines",
                 fontsize=14, fontweight="bold")

    # ── O1: Grouped bar chart ──
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(metric_labels))
    w = 0.8 / n_methods
    for i, mname in enumerate(method_names):
        vals = [methods[mname][k] for k in metric_keys]
        offset = (i - n_methods / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=mname,
                      color=METHOD_COLORS[i % len(METHOD_COLORS)],
                      alpha=0.85, edgecolor="white")
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=7, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.legend(fontsize=9)
    style_axes(ax, "bar", title="O1: Key Metrics Comparison", ylabel="Value")

    # ── O2: Radar chart ──
    ax_placeholder = fig.add_subplot(gs[1])
    # Normalize metrics for radar [0,1]; FD inverted
    all_vals = {k: [methods[m][k] for m in method_names] for k in metric_keys}
    normalized = {}
    for k in metric_keys:
        mn, mx = min(all_vals[k]), max(all_vals[k])
        rng = mx - mn if mx > mn else 1
        if k == "FD":
            normalized[k] = [(mx - v) / rng for v in all_vals[k]]
        else:
            normalized[k] = [(v - mn) / rng for v in all_vals[k]]

    angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
    angles += angles[:1]
    ax_placeholder.remove()
    ax_radar = fig.add_subplot(gs[1], polar=True)
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=8)

    for i, mname in enumerate(method_names):
        vals = [normalized[k][i] for k in metric_keys]
        vals += vals[:1]
        ax_radar.plot(angles, vals, "o-", linewidth=2, label=mname,
                      color=METHOD_COLORS[i % len(METHOD_COLORS)], markersize=6)
        ax_radar.fill(angles, vals, alpha=0.1,
                      color=METHOD_COLORS[i % len(METHOD_COLORS)])
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax_radar.set_title("O2: Normalized Radar", pad=20)

    # ── O3: Relative improvement strip (graphical — replaces table) ──
    ax3 = fig.add_subplot(gs[2])
    # For each baseline, compute % improvement of CLOP-DiT vs that baseline
    clop_vals = methods["CLOP-DiT"]
    improvement_data = {}
    for bl_name in baselines:
        bl_vals = methods.get(bl_name, {})
        improvements = {}
        for mk in metric_keys:
            cv = clop_vals.get(mk, 0)
            bv = bl_vals.get(mk, 0)
            if mk == "FD":  # lower is better: improvement = (bl - clop) / bl
                if bv > 1e-8:
                    improvements[mk] = (bv - cv) / bv
                else:
                    improvements[mk] = 0
            else:  # higher is better: improvement = (clop - bl) / bl
                if bv > 1e-8:
                    improvements[mk] = (cv - bv) / max(bv, 1e-8)
                else:
                    improvements[mk] = 0
        improvement_data[bl_name] = improvements

    # Plot as grouped horizontal bars
    y_labels = []
    y_vals = []
    y_colors = []
    bl_color_map = {name: METHOD_COLORS[i + 1] for i, name in enumerate(baselines)}

    for bl_name, imps in improvement_data.items():
        for mk in metric_keys:
            y_labels.append(f"{mk}\nvs {bl_name[:15]}")
            y_vals.append(imps.get(mk, 0) * 100)  # as percentage
            y_colors.append(bl_color_map.get(bl_name, "#999"))

    y_pos = np.arange(len(y_labels))
    bar_colors_final = [COLORS["good"] if v > 0 else COLORS["bad"] for v in y_vals]
    ax3.barh(y_pos, y_vals, color=bar_colors_final, height=0.6,
             edgecolor="white", linewidth=0.5, alpha=0.85)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(y_labels, fontsize=7, ha="left")
    set_dense_tick_labels(ax3, axis="y", max_labels=16, fontsize=7, rotation=0)
    ax3.axvline(x=0, color="#333", linewidth=1.2)
    ax3.invert_yaxis()

    # Value labels
    for i, v in enumerate(y_vals):
        sign = "+" if v > 0 else ""
        ax3.text(v + (2 if v >= 0 else -2), i, f"{sign}{v:.1f}%",
                 va="center", fontsize=8, fontweight="bold",
                 color=COLORS["good"] if v > 0 else COLORS["bad"],
                 ha="left" if v >= 0 else "right")

    style_axes(ax3, "bar", title="O3: CLOP-DiT Relative Improvement",
               xlabel="Improvement (%)")

    if save:
        path = save_panel(fig, output_dir / "panel_o_baseline_comparison.png", dpi)
        logger.info(f"Saved Panel O → {path}")
    return fig


def _compute_baselines(cache_dir: str = "data/cached_latents_v5.2") -> Dict[str, Dict]:
    """Compute Gaussian and Shuffled baselines on-the-fly."""
    from src.evaluation.metrics import GenerationMetrics

    cache = Path(cache_dir)
    cell_path = cache / "cell_embeddings_dedup_preprocessed.npy"
    gid_path = cache / "text_group_ids_dedup.npy"
    gen_path = Path("results/generated_embeddings.npy")
    gen_lab_path = Path("results/generated_labels.npy")

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
        "Gaussian N(μ,σ²I)": {
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

    bl_path = Path("results/baseline_metrics.json")
    with open(bl_path, "w") as f:
        json.dump(baselines, f, indent=2)
    logger.info(f"Saved baseline metrics → {bl_path}")
    return baselines
