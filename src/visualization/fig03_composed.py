"""
fig03_composed.py — Article Figure 3 composite (Step 2 of the single-producer
architecture migration, plan
`.omc/plans/single-producer-architecture-2026-04-20.md`).

Emits ONE canonical PDF (``fig03_composed.pdf``) combining the 10 panels
currently split across ``fig03a_metrics_summary.pdf`` (from
``fig05_metrics.py``), ``fig03b_per_type_fidelity.pdf`` (from
``fig06_fidelity.py``), and ``fig03c_text_cell_alignment.pdf`` (from
``fig07_alignment.py``) into a single ``plt.figure()`` with a single 3-row
gridspec so panel labels, fonts, and palettes are enforceable at the module
boundary rather than drifting across three source scripts.

Layout (3 rows, 4/3/3 panels, labels a-j):

    Row 1 (from fig05_metrics):
      a  Training convergence      b  Quality profile       c  Diversity gauges    d  Expression fidelity
    Row 2 (from fig06_fidelity):
      e  Real/Gen centroid cosine  f  Frechet outlier       g  Fidelity vs abundance
    Row 3 (from fig07_alignment):
      h  Text-cell heatmap         i  Per-type alignment    j  Diag vs off-diag distribution

**Dual-publish:** this module is ADDITIVE — ``fig05_metrics.py``,
``fig06_fidelity.py``, and ``fig07_alignment.py`` continue producing
``fig03a_*.pdf``, ``fig03b_*.pdf``, and ``fig03c_*.pdf`` unchanged. The
composite is a registered-but-not-yet-LaTeX-referenced asset during the
revision window.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from ..utils.constants import RANDOM_SEED  # noqa: F401 (reserved for future use)
from .article_composition import COMPOSITE_VCD_REGISTRY  # noqa: F401
from .direct_layout import bind_figure_region
from .explicit_positioning import add_axes_next_to
from .style import (
    COLORS,
    FONT_ANNOTATION,
    FONT_HEATMAP_CELL,
    FONT_LABEL,
    FONT_LEGEND_DENSE,
    FONT_SMALL,
    FONT_TICK_DENSE,
    FONT_TITLE,
    PANEL_OFFSET_FARLEFT_WIDE,
    PANEL_OFFSET_LEFT,
    PANEL_OFFSET_STD,
    PANEL_OFFSET_WIDE,
    abbreviate_cell_type,
    add_colorbar_safe,
    add_panel_label,
    apply_style,
    quality_color,
    save_with_vcd,
    set_adaptive_ytick_labels,
    style_axes,
)
from src.utils.paths import CACHE_DIR, CHECKPOINT_DIR, FIG_DIR, RESULTS_DIR, load_thresholds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Register composite in the VCD cross-slice coverage registry at import time.
# Canonical registration lives in article_composition.py; the idempotent
# re-assignment here is belt-and-braces documentation for migration reviewers.
# ---------------------------------------------------------------------------
COMPOSITE_VCD_REGISTRY["fig03_composed.pdf"] = [
    "fig03a_metrics_summary.pdf",
    "fig03b_per_type_fidelity.pdf",
    "fig03c_text_cell_alignment.pdf",
]

_viz_thresh = load_thresholds().get("visualization", {})
_COSINE_QUALITY_BANDS = tuple(_viz_thresh.get("cosine_quality_bands", [0.9, 0.7]))


# ---------------------------------------------------------------------------
# Data loaders — mirror the logic from fig05/fig06/fig07 so the composite reads
# the same cached artefacts the per-slice producers read.
# ---------------------------------------------------------------------------

def _load_fig03_inputs(
    clop_hist: Optional[Dict],
    dit_hist: Optional[Dict],
    gen_metrics_path: Path,
    expr_metrics_path: Path,
    div_metrics_path: Path,
    benchmark_report_path: Path,
    bootstrap_cis_path: Path,
    baseline_metrics_path: Path,
    cache_dir: Path,
) -> Dict:
    """Load every JSON/NPY artefact needed by the three rows.

    Returns a dict with keys: train_metrics, gen_metrics, div_metrics,
    expr_metrics, core_metrics, bootstrap_cis, baseline_data, bench_baselines,
    per_type_gen, per_type_div, collapsed_type_ids, text_heatmap.
    """
    out: Dict = {
        "train_metrics": {},
        "gen_metrics": {},
        "div_metrics": {},
        "expr_metrics": {},
        "core_metrics": {},
        "bootstrap_cis": {},
        "baseline_data": {},
        "bench_baselines": {},
        "per_type_gen": None,
        "per_type_div": None,
        "collapsed_type_ids": set(),
        "diversity_by_type_id": {},
        "text_heatmap": None,
    }

    # ── training history ─────────────────────────────────────────────────
    train_metrics: Dict[str, float] = {}
    if clop_hist:
        h = clop_hist
        train_metrics["CLOP Val Loss"] = h["val_loss"][-1]
        train_metrics["Proto Accuracy"] = h["val_proto_acc"][-1]
        train_metrics["Top-5 Accuracy"] = h["val_proto_top5"][-1]
        train_metrics["Text-Cell Align"] = h.get(
            "val_text_cell_align", h.get("val_proto_acc", [0.0])
        )[-1]
    if dit_hist:
        h = dit_hist
        train_metrics["DiT Val Loss"] = h["val_loss"][-1]
        if "val_cosine_sim" in h:
            train_metrics["DiT Val Cosine"] = h["val_cosine_sim"][-1]
    out["train_metrics"] = train_metrics

    # ── generation metrics (overall + per-type) ─────────────────────────
    if gen_metrics_path.exists():
        with open(gen_metrics_path) as f:
            gen_data = json.load(f)
        overall = gen_data.get("overall", {})
        summary = gen_data.get("summary", {})
        out["gen_metrics"] = {
            "FD": overall.get("frechet_distance", 0),
            "MMD": overall.get("mmd_rbf", 0),
            "Coverage": overall.get("coverage", 0),
            "Density": overall.get("density", 0),
            "Centroid Cos": summary.get("mean_centroid_cosine", 0),
        }
        out["per_type_gen"] = gen_data.get("per_type", {})
        out["gen_summary"] = summary

    # ── diversity diagnostics ────────────────────────────────────────────
    if div_metrics_path.exists():
        with open(div_metrics_path) as f:
            div_data = json.load(f)
        t1 = div_data.get("test1_intratype_diversity", {}).get("summary", {})
        t2 = div_data.get("test2_memorization", {})
        t5 = div_data.get("test5_condition_sensitivity", {}).get("summary", {})
        out["div_metrics"] = {
            "Diversity Ratio": t1.get("mean_diversity_ratio", 0),
            "Collapsed": t1.get("n_collapsed", 0),
            "Total Types": t1.get("n_collapsed", 0) + t1.get("n_healthy", 0),
            "NN Distance": t2.get("nn_cosine_distance", {}).get("mean", 0),
            "Near-copies": t2.get("n_very_close", 0),
            "Cond Gain": t5.get("mean_diversity_gain", 0),
        }
        per_type_div = div_data.get("test1_intratype_diversity", {}).get("per_type", {})
        out["per_type_div"] = per_type_div
        for entry in per_type_div.values():
            t_id = entry.get("type_id")
            ratio = entry.get("diversity_ratio")
            if t_id is None or ratio is None:
                continue
            out["diversity_by_type_id"][int(t_id)] = float(ratio)
            if ratio < 0.5:
                out["collapsed_type_ids"].add(int(t_id))

    # ── expression metrics ──────────────────────────────────────────────
    if expr_metrics_path.exists():
        with open(expr_metrics_path) as f:
            expr_data = json.load(f)
        gene_corr = expr_data.get("gene_correlation", {})
        per_type_sum = expr_data.get("per_type_summary", {})
        out["expr_metrics"] = {
            "Gene Pearson r": gene_corr.get("pearson_r", 0),
            "Gene Spearman": gene_corr.get("spearman_rho", 0),
            "Per-Type r (mean)": per_type_sum.get("mean_pearson_r", 0),
            "Per-Type r (min)": per_type_sum.get("min_pearson_r", 0),
            "Genes": gene_corr.get("n_genes_compared", 0),
        }

    # ── benchmark core + baselines ──────────────────────────────────────
    if benchmark_report_path.exists():
        try:
            with open(benchmark_report_path) as f:
                bench = json.load(f)
            candidates = [
                bench.get("core_metrics", {}),
                bench.get("operating_point", {}),
                bench.get("results", {}).get("core_metrics", {}),
            ]
            core_metrics: Dict[str, float] = {}
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
            out["core_metrics"] = core_metrics
            bench_methods = bench.get("methods", {})
            for mname, mdata in bench_methods.items():
                if mname != "CLOP-DiT":
                    out["bench_baselines"][mname] = mdata
        except Exception:
            pass

    # ── bootstrap CIs ───────────────────────────────────────────────────
    if bootstrap_cis_path.exists():
        try:
            with open(bootstrap_cis_path) as f:
                out["bootstrap_cis"] = json.load(f).get("metrics", {})
        except Exception:
            pass

    # ── baseline metrics ────────────────────────────────────────────────
    if baseline_metrics_path.exists():
        try:
            with open(baseline_metrics_path) as f:
                out["baseline_data"] = json.load(f)
        except Exception:
            pass

    # ── text-cell alignment inputs ──────────────────────────────────────
    proj_text_path = cache_dir / "projected_text.npy"
    proj_cell_path = cache_dir / "projected_cells.npy"
    gid_path = cache_dir / "text_group_ids_dedup.npy"
    if proj_text_path.exists() and proj_cell_path.exists() and gid_path.exists():
        proj_text = np.load(proj_text_path)
        proj_cells = np.load(proj_cell_path)
        group_ids = np.load(gid_path)
        gid_text_path = cache_dir / "text_group_ids.npy"
        if gid_text_path.exists() and proj_text.shape[0] != group_ids.shape[0]:
            text_group_ids = np.load(gid_text_path)
        else:
            text_group_ids = group_ids
        type_names: Dict[int, str] = {}
        cap_path = cache_dir / "text_captions_deduplicated.json"
        if cap_path.exists():
            with open(cap_path) as _cf:
                raw_caps = json.load(_cf)
            for k, v in raw_caps.items():
                name = v.split(" are ")[0] if " are " in v else v[:50]
                type_names[int(k)] = name
        out["text_heatmap"] = {
            "proj_text": proj_text,
            "proj_cells": proj_cells,
            "group_ids": group_ids,
            "text_group_ids": text_group_ids,
            "type_names": type_names,
        }

    return out


# ---------------------------------------------------------------------------
# Row renderers — each takes pre-created axes and draws onto them. Mirrors
# the logic from fig05_metrics.plot_metrics_summary,
# fig06_fidelity.plot_per_type_generation, and
# fig07_alignment.plot_text_cell_heatmap.
# ---------------------------------------------------------------------------

def _draw_metrics_row(
    fig: plt.Figure,
    ax_a: plt.Axes,
    ax_b: plt.Axes,
    ax_c: plt.Axes,
    ax_d: plt.Axes,
    data: Dict,
) -> None:
    """Top row: training convergence (a), quality profile (b),
    diversity gauges (c), expression fidelity (d)."""
    train_metrics = dict(data.get("train_metrics", {}))
    core_metrics = data.get("core_metrics", {})
    gen_metrics = data.get("gen_metrics", {})
    div_metrics = data.get("div_metrics", {})
    expr_metrics = data.get("expr_metrics", {})
    bootstrap_cis = data.get("bootstrap_cis", {})
    baseline_data = data.get("baseline_data", {})
    bench_baselines = data.get("bench_baselines", {})

    _bci_key_map = {
        "KNN-1": "knn_top1",
        "Linear Acc": "linear_acc",
        "Steering": "steering",
        "Diversity Ratio": "diversity_ratio",
    }

    # ── Panel a: Training convergence ─────────────────────────────────
    add_panel_label(ax_a, "a", x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
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
        colors_a = []
        for n, v in zip(names, vals):
            if "Loss" in n:
                colors_a.append(
                    COLORS["bad"] if v > 1.0 else COLORS["warn"] if v > 0.1 else COLORS["good"]
                )
            else:
                colors_a.append(
                    COLORS["good"] if v > 0.8 else COLORS["warn"] if v > 0.5 else COLORS["bad"]
                )
        y_pos = np.arange(len(names))
        bars = ax_a.barh(y_pos, display_vals, color=colors_a, height=0.6,
                         edgecolor="white", linewidth=0.8)
        ax_a.set_yticks(y_pos)
        ax_a.set_yticklabels([metric_abbrev.get(n, n) for n in names], fontsize=10)
        for bar, dv, n in zip(bars, display_vals, names):
            unit = "" if "Loss" in n else "%"
            bci_key = _bci_key_map.get(n)
            ci_text = ""
            if bci_key and bci_key in bootstrap_cis:
                ci = bootstrap_cis[bci_key]
                ci_lo = ci.get("ci_95_lower", 0)
                ci_hi = ci.get("ci_95_upper", 0)
                if "Loss" not in n:
                    ci_lo_d = ci_lo * 100 if ci_lo <= 1.0 else ci_lo
                    ci_hi_d = ci_hi * 100 if ci_hi <= 1.0 else ci_hi
                else:
                    ci_lo_d = ci_lo
                    ci_hi_d = ci_hi
                bar_center = bar.get_y() + bar.get_height() / 2
                ax_a.plot([ci_lo_d, ci_hi_d], [bar_center, bar_center],
                          color=COLORS["annotation_dark"], linewidth=1.2, zorder=5)
                ax_a.plot([ci_lo_d, ci_lo_d], [bar_center - 0.12, bar_center + 0.12],
                          color=COLORS["annotation_dark"], linewidth=1.0, zorder=5)
                ax_a.plot([ci_hi_d, ci_hi_d], [bar_center - 0.12, bar_center + 0.12],
                          color=COLORS["annotation_dark"], linewidth=1.0, zorder=5)
                ci_text = (
                    f" [{ci_lo * 100:.1f}, {ci_hi * 100:.1f}]" if "Loss" not in n else ""
                )
            fmt = f"{dv:.4f}" if "Loss" in n else f"{dv:.1f}{unit}"
            ax_a.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                      fmt + ci_text, va="center", fontsize=FONT_SMALL)
        ax_a.set_xlabel("Value (accuracy shown as %)")
        ax_a.set_title("Training Convergence", fontsize=12)
        ax_a.invert_yaxis()
        ax_a.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax_a.grid(axis="both", alpha=0.15, linestyle="--")
    else:
        ax_a.text(0.5, 0.5, "No training history available",
                  ha="center", va="center", transform=ax_a.transAxes,
                  fontsize=10, color=COLORS["neutral"])
        ax_a.set_title("Training Convergence", fontsize=12)

    # ── Panel b: Quality profile ──────────────────────────────────────
    # Wide-left offset keeps 'B' clear of the 1.2 ytick on Quality Profile.
    add_panel_label(ax_b, "b",
                    x=PANEL_OFFSET_FARLEFT_WIDE[0],
                    y=PANEL_OFFSET_FARLEFT_WIDE[1])
    if gen_metrics or expr_metrics:
        bar_labels = []
        bar_vals = []
        if gen_metrics:
            fd_score = max(0, 1.0 - gen_metrics.get("FD", 1.0))
            bar_labels.append("FD*"); bar_vals.append(fd_score)
            bar_labels.append("Cov."); bar_vals.append(gen_metrics.get("Coverage", 0))
            bar_labels.append("Cent."); bar_vals.append(gen_metrics.get("Centroid Cos", 0))
        if div_metrics:
            bar_labels.append("Div.")
            bar_vals.append(div_metrics.get("Diversity Ratio", 0))
        if expr_metrics:
            bar_labels.append("Gene")
            bar_vals.append(expr_metrics.get("Gene Pearson r", 0))
        if bar_vals:
            gauss_bl = baseline_data.get("Gaussian N(\u03bc,\u03c3\u00b2I)", {})
            if not gauss_bl:
                gauss_bl = bench_baselines.get("Gaussian N(\u03bc,\u03c3\u00b2I)", {})
            baseline_vals = []
            for lbl in bar_labels:
                if "FD" in lbl:
                    baseline_vals.append(
                        max(0, 1.0 - gauss_bl.get("frechet_distance", 1.0)) if gauss_bl else 0.0
                    )
                elif "Coverage" in lbl:
                    baseline_vals.append(gauss_bl.get("coverage", 0.0) if gauss_bl else 0.0)
                elif "Centroid" in lbl:
                    baseline_vals.append(
                        gauss_bl.get("mean_centroid_cosine", 0.0) if gauss_bl else 0.0
                    )
                elif "Diversity" in lbl:
                    baseline_vals.append(
                        min(gauss_bl.get("diversity_ratio", 0.0), 1.0) if gauss_bl else 0.0
                    )
                else:
                    baseline_vals.append(0.0)
            x_pos = np.arange(len(bar_vals))
            width = 0.34 if gauss_bl else 0.58
            bars_clop = ax_b.bar(
                x_pos - (width / 2 if gauss_bl else 0.0), bar_vals,
                color=COLORS["real"], alpha=0.88,
                edgecolor="white", linewidth=0.8, width=width, label="CLOP-DiT",
            )
            bars_gauss = []
            if gauss_bl:
                bars_gauss = ax_b.bar(
                    x_pos + width / 2, baseline_vals,
                    color=COLORS["baseline_gauss"], alpha=0.72,
                    edgecolor="white", linewidth=0.8, width=width, label="Gaussian baseline",
                )
            ax_b.set_xticks(x_pos)
            ax_b.set_xticklabels(bar_labels, fontsize=FONT_TICK_DENSE)
            ax_b.set_ylim(0, 1.15)
            ax_b.set_ylabel("Score", fontsize=FONT_LABEL)
            for bar, val in zip(bars_clop, bar_vals):
                ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                          f"{val:.2f}", ha="center", va="bottom",
                          fontsize=FONT_ANNOTATION, color=COLORS["real"], zorder=8)
            for bar, val in zip(bars_gauss, baseline_vals):
                ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                          f"{val:.2f}", ha="center", va="bottom",
                          fontsize=FONT_SMALL, color=COLORS["baseline_gauss"], zorder=8)
            if gauss_bl:
                ax_b.legend(fontsize=FONT_LEGEND_DENSE, frameon=False,
                            loc="upper left", ncol=2)
        ax_b.set_title("Quality Profile", fontsize=FONT_TITLE)
    else:
        ax_b.text(0.5, 0.5, "No generation data", ha="center",
                  va="center", transform=ax_b.transAxes)
        ax_b.set_title("Generation Quality Profile")

    # ── Panel c: Diversity gauges ─────────────────────────────────────
    add_panel_label(ax_c, "c", x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
    if div_metrics:
        gauge_items = [
            ("Diversity\nRatio", div_metrics.get("Diversity Ratio", 0), 1.0,
             COLORS["good"] if div_metrics.get("Diversity Ratio", 0) >= 0.8 else
             COLORS["warn"] if div_metrics.get("Diversity Ratio", 0) >= 0.5 else
             COLORS["bad"], "diversity_ratio"),
            ("Cond\nGain", div_metrics.get("Cond Gain", 0), 3.0,
             COLORS["good"] if div_metrics.get("Cond Gain", 0) >= 1.5 else
             COLORS["warn"] if div_metrics.get("Cond Gain", 0) >= 1.0 else
             COLORS["bad"], None),
            ("NN\nDistance", div_metrics.get("NN Distance", 0), 1.0,
             COLORS["good"] if div_metrics.get("NN Distance", 0) >= 0.3 else
             COLORS["warn"] if div_metrics.get("NN Distance", 0) >= 0.1 else
             COLORS["bad"], None),
        ]
        gauge_text_x = max(item[2] for item in gauge_items) + 0.22
        for i, (label, val, max_val, color, bci_key) in enumerate(gauge_items):
            ax_c.barh(i, max_val, height=0.5, color=COLORS["bg_gauge"],
                      edgecolor="none", zorder=1)
            ax_c.barh(i, min(val, max_val), height=0.5, color=color,
                      edgecolor="white", linewidth=0.8, zorder=2)
            ci_text = ""
            display_val = val
            if bci_key and bci_key in bootstrap_cis:
                ci = bootstrap_cis[bci_key]
                ci_lo = ci.get("ci_95_lower", 0)
                ci_hi = ci.get("ci_95_upper", 0)
                display_val = ci.get("mean", ci.get("point_estimate", val))
                ci_text = f"  [{ci_lo:.3f}, {ci_hi:.3f}]"
            ann_text = f"{display_val:.3f}" + ci_text
            ax_c.text(gauge_text_x, i, ann_text, va="center", fontsize=10, zorder=3)
        ax_c.set_yticks(range(len(gauge_items)))
        ax_c.set_yticklabels([g[0] for g in gauge_items], fontsize=10)
        ax_c.invert_yaxis()
        ax_c.set_xlim(0, gauge_text_x + 0.70)
        collapsed = div_metrics.get("Collapsed", 0)
        total = div_metrics.get("Total Types", 69)
        copies = div_metrics.get("Near-copies", 0)
        ax_c.text(0.98, 0.02,
                  f"Collapsed: {collapsed}/{total} | Near-copies: {copies}",
                  transform=ax_c.transAxes, ha="right", va="bottom",
                  fontsize=10, color=COLORS["neutral"])
        ax_c.set_title("Diversity Health", fontsize=12)
        ax_c.set_xlabel("Score")
        ax_c.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax_c.grid(axis="x", alpha=0.15, linestyle="--")
    else:
        ax_c.text(0.5, 0.5, "No diversity data", ha="center",
                  va="center", transform=ax_c.transAxes)
        ax_c.set_title("Diversity Health")

    # ── Panel d: Expression fidelity ──────────────────────────────────
    add_panel_label(ax_d, "d", x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
    if expr_metrics:
        expr_items = [
            ("Gene r", expr_metrics.get("Gene Pearson r", 0)),
            ("Gene \u03c1", expr_metrics.get("Gene Spearman", 0)),
            ("Type r (\u03bc)", expr_metrics.get("Per-Type r (mean)", 0)),
            ("Type r (min)", expr_metrics.get("Per-Type r (min)", 0)),
        ]
        y_pos = np.arange(len(expr_items))
        deviations = [max(1 - v, 1e-12) for _, v in expr_items]
        bars = ax_d.barh(y_pos, deviations, color=COLORS["real"], height=0.5,
                         edgecolor="white", linewidth=0.8)
        ax_d.set_yticks(y_pos)
        ax_d.set_yticklabels([n for n, _ in expr_items], fontsize=10)
        for bar, dev in zip(bars, deviations):
            ax_d.text(bar.get_width() * 1.3, bar.get_y() + bar.get_height() / 2,
                      f"{dev:.1e}", va="center", fontsize=9)
        ax_d.invert_yaxis()
        ax_d.set_xscale("log")
        from matplotlib.ticker import LogLocator
        ax_d.xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
        # Clamp xlim so the right-most tick does not bleed past the panel edge.
        _dmin = min(deviations) if deviations else 1e-6
        _dmax = max(deviations) if deviations else 1.0
        ax_d.set_xlim(_dmin * 0.5, _dmax * 2.5)
        ax_d.set_xlabel("Deviation (1 \u2212 r)", fontsize=FONT_LABEL)
        ax_d.set_title("Expression Fidelity", fontsize=12)
        ax_d.grid(axis="both", alpha=0.15, linestyle="--")
        ax_d.text(0.98, 0.02, "Lower = better (log scale)",
                  transform=ax_d.transAxes, ha="right", va="bottom",
                  fontsize=FONT_SMALL, color=COLORS["neutral"], style="italic")
    else:
        ax_d.text(0.5, 0.5, "No expression data", ha="center",
                  va="center", transform=ax_d.transAxes)
        ax_d.set_title("Expression Fidelity")


def _draw_fidelity_row(
    fig: plt.Figure,
    ax_e: plt.Axes,
    ax_f: plt.Axes,
    ax_g: plt.Axes,
    data: Dict,
) -> None:
    """Middle row: centroid cosine per type (e), Frechet outlier profile (f),
    fidelity vs abundance (g). Mirrors fig06_fidelity.plot_per_type_generation."""
    per_type = data.get("per_type_gen") or {}
    if not per_type:
        add_panel_label(ax_e, "e", x=PANEL_OFFSET_WIDE[0], y=PANEL_OFFSET_WIDE[1])
        add_panel_label(ax_f, "f", x=PANEL_OFFSET_WIDE[0], y=PANEL_OFFSET_WIDE[1])
        add_panel_label(ax_g, "g", x=0, y=PANEL_OFFSET_STD[1])
        for ax in (ax_e, ax_f, ax_g):
            ax.text(0.5, 0.5, "No per-type data",
                    ha="center", va="center", transform=ax.transAxes)
        return

    names = list(per_type.keys())
    cosines = [per_type[n]["centroid_cosine"] for n in names]
    fds = [per_type[n].get("frechet_distance", float("nan")) for n in names]
    n_real = [per_type[n]["n_real"] for n in names]
    type_ids = [per_type[n].get("type_id") for n in names]
    short_names = [abbreviate_cell_type(n, max_len=26) for n in names]
    diversity_by_type_id = data.get("diversity_by_type_id", {})
    collapsed_type_ids = data.get("collapsed_type_ids", set())
    div_ratios = [
        diversity_by_type_id.get(int(t), np.nan) if t is not None else np.nan
        for t in type_ids
    ]
    fd_array = np.asarray(fds, dtype=float)
    cos_array = np.asarray(cosines, dtype=float)
    n_real_array = np.asarray(n_real, dtype=float)
    div_array = np.asarray(div_ratios, dtype=float)
    fd_valid = np.isfinite(fd_array)
    fd_mean = float(np.nanmean(fd_array)) if fd_valid.any() else float("nan")
    summary = data.get("gen_summary", {})

    # ── Panel e: Real<->Gen centroid cosine (sorted) ─────────────────
    add_panel_label(ax_e, "e", x=PANEL_OFFSET_WIDE[0], y=PANEL_OFFSET_WIDE[1])
    sorted_idx = np.argsort(cosines)
    sorted_cos = [cosines[i] for i in sorted_idx]
    sorted_names_cos = [short_names[i] for i in sorted_idx]
    sorted_type_ids = [type_ids[i] for i in sorted_idx]
    colors_e = []
    for v, t_id in zip(sorted_cos, sorted_type_ids):
        if t_id is not None and int(t_id) in collapsed_type_ids:
            colors_e.append(COLORS["bad"])
        else:
            colors_e.append(quality_color(v, _COSINE_QUALITY_BANDS))
    ax_e.barh(range(len(sorted_cos)), sorted_cos, color=colors_e, height=0.8)
    set_adaptive_ytick_labels(ax_e, sorted_names_cos, max_visible=18,
                              fontsize=FONT_HEATMAP_CELL)
    ax_e.set_xlabel("Centroid Cosine Similarity")
    ax_e.set_title("Real\u2194Gen Centroid Cosine", fontsize=12)
    ax_e.axvline(
        x=summary.get("mean_centroid_cosine", 0), color=COLORS["bad"],
        linestyle="--", alpha=0.5, label="mean (see caption)",
    )
    ax_e.set_xlim(0, 1.05)
    ax_e.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))
    ax_e.legend(fontsize=10, frameon=False, loc="lower right")
    style_axes(ax_e, kind="bar")

    # ── Panel f: Frechet outlier profile ─────────────────────────────
    add_panel_label(ax_f, "f", x=PANEL_OFFSET_WIDE[0], y=PANEL_OFFSET_WIDE[1])
    if fd_valid.any():
        fd_idx = np.where(fd_valid)[0][np.argsort(fd_array[fd_valid])]
        fd_vals = fd_array[fd_idx]
        fd_names = [short_names[i] for i in fd_idx]
        fd_min = float(np.nanmin(fd_vals))
        fd_ptp = float(np.nanmax(fd_vals) - fd_min) or 1.0
        fd_norm = np.clip((fd_vals - fd_min) / fd_ptp, 0, 1)
        colors_f = plt.cm.PiYG_r(fd_norm)
        y_pos = np.arange(len(fd_vals))
        ax_f.hlines(y_pos, 0, fd_vals, color=colors_f, linewidth=2.8, alpha=0.85)
        ax_f.scatter(fd_vals, y_pos, s=28 + 70 * fd_norm, color=colors_f,
                     edgecolors="white", linewidths=0.4, zorder=3)
        if np.isfinite(fd_mean):
            ax_f.axvline(fd_mean, color=COLORS["bad"], linestyle="--",
                         alpha=0.7, linewidth=1.5, label="mean (see caption)")
        set_adaptive_ytick_labels(ax_f, fd_names, max_visible=18,
                                  fontsize=FONT_HEATMAP_CELL)
        ax_f.set_xlabel("Fr\u00e9chet Distance (lower = better)")
        ax_f.set_title("Fr\u00e9chet Outlier Profile")
        ax_f.set_xlim(0, float(np.nanmax(fd_vals)) * 1.12)
        ax_f.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
        style_axes(ax_f, kind="bar")
    else:
        ax_f.text(0.5, 0.5, "No valid FD values", ha="center",
                  va="center", transform=ax_f.transAxes)

    # ── Panel g: Fidelity vs abundance (scatter) ─────────────────────
    # Column-3 panel: bare 0 for x-offset per fig06 G3 convention.
    add_panel_label(ax_g, "g", x=0, y=PANEL_OFFSET_STD[1])
    fd_for_size = np.where(
        fd_valid, fd_array,
        np.nanmedian(fd_array[fd_valid]) if fd_valid.any() else 1.0,
    )
    fd_min_s = float(np.nanmin(fd_for_size)) if np.isfinite(fd_for_size).any() else 0.0
    fd_ptp_s = float(np.nanmax(fd_for_size) - fd_min_s) if np.isfinite(fd_for_size).any() else 1.0
    fd_ptp_s = fd_ptp_s or 1.0
    bubble_sizes = 50 + 220 * np.clip((fd_for_size - fd_min_s) / fd_ptp_s, 0, 1)
    x_vals = np.log10(np.maximum(n_real_array, 1))
    if np.isfinite(div_array).any():
        color_values = np.where(
            np.isfinite(div_array), div_array,
            np.nanmedian(div_array[np.isfinite(div_array)]),
        )
        sc = ax_g.scatter(
            x_vals, cos_array, c=color_values, cmap="viridis",
            vmin=0.5, vmax=1.05, s=bubble_sizes, alpha=0.78,
            edgecolors="white", linewidth=0.6, clip_on=False,
        )
        cax = add_axes_next_to(
            fig, ax_g, side="right", width=0.008,
            height=ax_g.get_position().height * 0.34,
            pad=0.012, align="bottom", y_offset=0.012,
        )
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("")
        cbar.ax.set_title("Div.\nratio", fontsize=10, pad=4)
        cbar.ax.tick_params(labelsize=10)
    else:
        ax_g.scatter(
            x_vals, cos_array, c=COLORS["real"], s=bubble_sizes,
            alpha=0.78, edgecolors="white", linewidth=0.6, clip_on=False,
        )
    if len(x_vals) > 1:
        slope, intercept = np.polyfit(x_vals, cos_array, deg=1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax_g.plot(x_line, slope * x_line + intercept, color=COLORS["trend_dark"],
                  linestyle="--", linewidth=1.3, label="Trend")
    # Minimal callouts — 3 worst types only.
    candidate_idx = np.argsort(cos_array)[:8]
    label_offsets = [(-30, -12), (28, -10), (-28, 14), (28, 12), (16, -20), (-26, 20)]
    annotated = 0
    placed: list[tuple[float, float]] = []
    for rank, i in enumerate(candidate_idx):
        if annotated >= 3:
            break
        if cos_array[i] >= 0.94:
            continue
        if any(abs(x_vals[i] - px) < 0.25 and abs(cos_array[i] - py) < 0.04
               for px, py in placed):
            continue
        x_off, y_off = label_offsets[rank % len(label_offsets)]
        if x_vals[i] > np.median(x_vals):
            x_off = min(x_off, -10)
        else:
            x_off = max(x_off, 10)
        ax_g.annotate(
            abbreviate_cell_type(short_names[i], max_len=10),
            (x_vals[i], cos_array[i]),
            fontsize=FONT_SMALL + 2,
            xytext=(x_off, y_off), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor="none", alpha=0.75),
        )
        placed.append((x_vals[i], cos_array[i]))
        annotated += 1
    ax_g.set_xlabel("log10(Number of Real Cells)", fontsize=13)
    ax_g.set_ylabel("Centroid Cosine Similarity", fontsize=13)
    ax_g.set_title("Fidelity vs Abundance", fontsize=14)
    ax_g.tick_params(axis="both", labelsize=12)
    ax_g.axhline(y=0.9, color=COLORS["good"], linestyle=":", alpha=0.4)
    ax_g.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_g.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_g.set_xlim(x_vals.min() - 0.10, x_vals.max() + 0.10)
    style_axes(ax_g, kind="scatter")


def _draw_alignment_row(
    fig: plt.Figure,
    ax_h: plt.Axes,
    ax_i: plt.Axes,
    ax_j: plt.Axes,
    data: Dict,
) -> None:
    """Bottom row: text-cell heatmap (h), per-type alignment bars (i),
    diagonal vs off-diagonal distribution (j). Mirrors
    fig07_alignment.plot_text_cell_heatmap."""
    heat = data.get("text_heatmap")
    if not heat:
        add_panel_label(ax_h, "h", x=PANEL_OFFSET_WIDE[0], y=PANEL_OFFSET_WIDE[1])
        add_panel_label(ax_i, "i", x=PANEL_OFFSET_LEFT[0], y=PANEL_OFFSET_LEFT[1])
        add_panel_label(ax_j, "j", x=PANEL_OFFSET_WIDE[0], y=PANEL_OFFSET_WIDE[1])
        for ax in (ax_h, ax_i, ax_j):
            ax.text(0.5, 0.5, "No alignment data",
                    ha="center", va="center", transform=ax.transAxes)
        return

    proj_text = heat["proj_text"]
    proj_cells = heat["proj_cells"]
    group_ids = heat["group_ids"]
    text_group_ids = heat["text_group_ids"]
    type_names = heat["type_names"]

    unique_types = np.sort(np.unique(group_ids))
    n_types = len(unique_types)
    text_centroids = np.zeros((n_types, proj_text.shape[1]), dtype=np.float32)
    cell_centroids = np.zeros((n_types, proj_cells.shape[1]), dtype=np.float32)
    for i, t in enumerate(unique_types):
        mask = group_ids == t
        tc = proj_text[text_group_ids == t].mean(axis=0)
        text_centroids[i] = tc / (np.linalg.norm(tc) + 1e-8)
        cc = proj_cells[mask].mean(axis=0)
        cell_centroids[i] = cc / (np.linalg.norm(cc) + 1e-8)
    sim_matrix = text_centroids @ cell_centroids.T
    x_labels = [abbreviate_cell_type(type_names.get(int(t), f"T{t}"), max_len=13)
                for t in unique_types]
    y_labels = [abbreviate_cell_type(type_names.get(int(t), f"T{t}"), max_len=22)
                for t in unique_types]
    diag = np.diag(sim_matrix)
    mean_diag = diag.mean()
    std_diag = diag.std()
    median_diag = float(np.median(diag))
    off_diag = sim_matrix[~np.eye(n_types, dtype=bool)]
    mean_off = off_diag.mean()
    std_off = off_diag.std()
    median_off = float(np.median(off_diag))

    pooled_std = np.sqrt((std_diag ** 2 + std_off ** 2) / 2.0)
    cohens_d = (mean_diag - mean_off) / pooled_std if pooled_std > 0 else float("inf")
    try:
        from scipy.stats import mannwhitneyu
        _u, p_value = mannwhitneyu(diag, off_diag, alternative="greater")
    except ImportError:
        p_value = None
    n_excellent = int(np.sum(diag >= 0.9))
    n_good = int(np.sum((diag >= 0.7) & (diag < 0.9)))
    n_poor = int(np.sum(diag < 0.7))

    sort_order = np.argsort(-diag)
    sim_sorted = sim_matrix[sort_order][:, sort_order]
    x_labels_sorted = [x_labels[i] for i in sort_order]
    y_labels_sorted = [y_labels[i] for i in sort_order]

    # ── Panel h: clustered heatmap ────────────────────────────────────
    add_panel_label(ax_h, "h", x=PANEL_OFFSET_WIDE[0], y=PANEL_OFFSET_WIDE[1])
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_heat",
        [
            "#1a237e", "#283593", "#42a5f5", "#e3f2fd", "#fff9c4",
            "#ffcc80", "#ff7043", "#d32f2f", "#b71c1c",
        ],
        N=256,
    )
    im = ax_h.imshow(sim_sorted, cmap=cmap, vmin=-0.1, vmax=1.0,
                     aspect="auto", interpolation="nearest")
    im.set_rasterized(True)
    step_x = max(10, int(np.ceil(n_types / 6)))
    step_y = max(5, int(np.ceil(n_types / 12)))
    _xtl = [x_labels_sorted[i] if i % step_x == 0 else "" for i in range(n_types)]
    _ytl = [y_labels_sorted[i] if i % step_y == 0 else "" for i in range(n_types)]
    ax_h.set_xticks(range(n_types))
    ax_h.set_yticks(range(n_types))
    ax_h.set_xticklabels(_xtl, rotation=55, fontsize=8, ha="right")
    ax_h.set_yticklabels(_ytl, fontsize=10.5, ha="right")
    ax_h.set_ylabel("Cell Type (text prototypes)", fontsize=11)
    ax_h.set_title("Cosine Similarity (sorted by diagonal)", fontsize=12)
    ax_h.plot([0, n_types - 1], [0, n_types - 1],
              color="white", linewidth=0.8, linestyle=":", alpha=0.6, zorder=3)
    cax = add_axes_next_to(
        fig, ax_h, side="right", width=0.006,
        height=ax_h.get_position().height * 0.42,
        pad=0.012, align="bottom", y_offset=0.012,
    )
    cbar = add_colorbar_safe(im, ax=ax_h, cax=cax, shrink=1.0, pad=0.0, aspect=14)
    cbar.set_label("")
    cbar.ax.set_title("Cos.\nsim.", fontsize=10, pad=4)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    n_inset = min(6, n_types)
    ax_inset = ax_h.inset_axes([0.68, 0.68, 0.24, 0.24])
    im_inset = ax_inset.imshow(
        sim_sorted[:n_inset, :n_inset], cmap=cmap, vmin=-0.1, vmax=1.0,
        aspect="auto", interpolation="nearest",
    )
    im_inset.set_rasterized(True)
    ax_inset.set_xticks([]); ax_inset.set_yticks([])
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("white"); spine.set_linewidth(1.5)
    ax_inset._clop_styled = True
    style_axes(ax_h, kind="heatmap")

    # ── Panel i: per-type alignment bars ─────────────────────────────
    add_panel_label(ax_i, "i", x=PANEL_OFFSET_LEFT[0], y=PANEL_OFFSET_LEFT[1])
    sorted_idx_asc = np.argsort(diag)
    d_asc = diag[sorted_idx_asc]
    labels_asc = [y_labels[i] for i in sorted_idx_asc]
    ax_i.axvspan(0.9, 1.08, color=COLORS["good"], alpha=0.06, zorder=0)
    ax_i.axvspan(0.7, 0.9, color=COLORS["warn"], alpha=0.06, zorder=0)
    ax_i.axvspan(0.0, 0.7, color=COLORS["bad"], alpha=0.06, zorder=0)
    color_map = [quality_color(v, _COSINE_QUALITY_BANDS) for v in d_asc]
    ax_i.barh(range(n_types), d_asc, color=color_map, height=0.8,
              edgecolor="white", linewidth=0.3)
    set_adaptive_ytick_labels(ax_i, labels_asc, max_visible=18, fontsize=10)
    ax_i.set_xlabel("Cosine Similarity", fontsize=11)
    _prev_y = -999
    for idx_bar in list(range(min(4, n_types))):
        val_bar = d_asc[idx_bar]
        if abs(idx_bar - _prev_y) < 1.5 and idx_bar != 0 and idx_bar < n_types - 3:
            continue
        _prev_y = idx_bar
        ax_i.text(val_bar + 0.01, idx_bar, f"{val_bar:.3f}",
                  va="center", ha="left", fontsize=10,
                  color=COLORS["annotation_medium"])
    ax_i.axvline(x=mean_diag, color=COLORS["bad"], linestyle="--",
                 alpha=0.7, linewidth=1.5)
    ax_i.axvline(x=0.9, color=COLORS["good"], linestyle=":", alpha=0.5, linewidth=1.0)
    ax_i.axvline(x=0.7, color=COLORS["warn"], linestyle=":", alpha=0.5, linewidth=1.0)
    ax_i.set_title("Per-Type Alignment", fontsize=FONT_TITLE, pad=8, x=0.60)
    ax_i.set_xlim(0, 1.08)
    ax_i.axvline(x=median_diag, color=COLORS["heatmap_purple"], linestyle="-.",
                 alpha=0.6, linewidth=1.0)
    ax_i.text(
        0.98, 0.98,
        f"\u03bc={mean_diag:.3f} | med={median_diag:.3f}\n"
        f"{n_excellent} excellent | {n_good} good | {n_poor} poor",
        transform=ax_i.transAxes, ha="right", va="top",
        fontsize=FONT_SMALL - 1, color=COLORS["annotation_dark"],
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                  edgecolor="none", alpha=0.85),
        zorder=10,
    )
    style_axes(ax_i, kind="bar")

    # ── Panel j: diagonal vs off-diagonal distribution ───────────────
    add_panel_label(ax_j, "j", x=PANEL_OFFSET_WIDE[0], y=PANEL_OFFSET_WIDE[1])
    ax_j.hist(
        diag, bins=12, alpha=0.7, color=COLORS["real"], edgecolor="white",
        label=f"Diagonal (n={len(diag)})", density=True, orientation="horizontal",
    )
    ax_j.hist(
        off_diag, bins=25, alpha=0.45, color=COLORS["generated"], edgecolor="white",
        label=f"Off-diagonal (n={len(off_diag)})", density=True, orientation="horizontal",
    )
    try:
        from scipy.stats import gaussian_kde
        kde_diag = gaussian_kde(diag, bw_method=0.3)
        y_kde = np.linspace(diag.min() - 0.05, diag.max() + 0.05, 200)
        ax_j.plot(kde_diag(y_kde), y_kde, color=COLORS["real"],
                  linewidth=1.5, linestyle="-", alpha=0.8)
        kde_off = gaussian_kde(off_diag, bw_method=0.3)
        y_kde_off = np.linspace(
            off_diag.min() - 0.05, min(off_diag.max() + 0.05, 1.0), 300,
        )
        ax_j.plot(kde_off(y_kde_off), y_kde_off, color=COLORS["generated"],
                  linewidth=1.5, linestyle="-", alpha=0.7)
    except ImportError:
        pass
    ax_j.axhline(y=mean_diag, color=COLORS["real"], linestyle="--", linewidth=1.5)
    ax_j.axhline(y=mean_off, color=COLORS["generated"], linestyle=":", linewidth=1.5)
    ax_j.axhline(y=median_diag, color=COLORS["real"], linestyle="-.",
                 linewidth=1.0, alpha=0.6)
    ax_j.axhline(y=median_off, color=COLORS["generated"], linestyle="-.",
                 linewidth=1.0, alpha=0.6)
    ax_j.set_ylabel("Cosine Similarity", fontsize=11)
    ax_j.set_xlabel("Density", fontsize=11)
    ax_j.set_title("Diag vs Off-Diag", fontsize=FONT_TITLE)
    _ylim_lo = min(off_diag.min() - 0.02, diag.min() - 0.02)
    ax_j.set_ylim(_ylim_lo, 1.05)
    ax_j.yaxis.set_major_locator(MaxNLocator(nbins=8, prune="both"))
    ax_j.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax_j.tick_params(axis="both", labelsize=10)
    # Inline diagonal/off-diagonal legend anchored center-left inside the
    # axes. Placed in the sparse density region (mid-range cosine where
    # neither distribution has significant mass) to avoid masking bars.
    ax_j.legend(fontsize=FONT_SMALL, frameon=False, loc="center left",
                bbox_to_anchor=(0.02, 0.55), ncol=1,
                handlelength=1.0, handletextpad=0.3, borderaxespad=0.2,
                labelspacing=0.25)
    ax_j.grid(True, axis="both", alpha=0.2, linewidth=0.4)
    stat_anno = "Mann\u2013Whitney U test\n" + f"$d$={cohens_d:.1f}"
    if p_value is not None:
        if p_value < 1e-10:
            stat_anno += ",  $p$<1e-10***"
        elif p_value < 0.001:
            stat_anno += f",  $p$={p_value:.1e}***"
        elif p_value < 0.01:
            stat_anno += f",  $p$={p_value:.3f}**"
        elif p_value < 0.05:
            stat_anno += f",  $p$={p_value:.3f}*"
        else:
            stat_anno += f",  $p$={p_value:.3f} n.s."
    ax_j.text(
        0.97, 0.03, stat_anno, transform=ax_j.transAxes,
        fontsize=FONT_SMALL, va="bottom", ha="right", color=COLORS["annotation_dark"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.92,
                  edgecolor=COLORS["border_light"], linewidth=0.5),
        zorder=10,
    )
    style_axes(ax_j, kind="default")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_fig03_composed(
    clop_hist: Optional[Dict] = None,
    dit_hist: Optional[Dict] = None,
    cache_dir: Optional[str] = None,
    gen_metrics_path: Optional[str] = None,
    expr_metrics_path: Optional[str] = None,
    div_metrics_path: Optional[str] = None,
    benchmark_report_path: Optional[str] = None,
    bootstrap_cis_path: Optional[str] = None,
    baseline_metrics_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    dpi: int = 300,
    save: bool = True,
) -> Optional[Path]:
    """Render the 3-row composite Fig 3 and save to ``fig03_composed.{pdf,png}``.

    Parameters
    ----------
    clop_hist, dit_hist : optional training-history dicts (keys ``val_loss``,
        ``val_proto_acc``, ``val_proto_top5``, ``val_text_cell_align``,
        ``val_cosine_sim``); loaded from checkpoint JSONs by callers.
    cache_dir           : directory with projected_text/cells/group_ids NPYs
                          (defaults to ``data/cache``).
    *_path              : JSON artefact paths (default to ``results/*.json``).
    output_dir          : directory to save the composite PDF
                          (defaults to ``results/figures``).
    dpi                 : rasterised preview DPI (PDF is vector).
    save                : if False, return ``None`` without writing.

    Returns
    -------
    Path to the saved PDF, or ``None`` if all data sources are missing.
    """
    apply_style()

    cache = Path(cache_dir) if cache_dir else CACHE_DIR
    gen_path = Path(gen_metrics_path) if gen_metrics_path else RESULTS_DIR / "generation_metrics.json"
    expr_path = Path(expr_metrics_path) if expr_metrics_path else RESULTS_DIR / "expression_metrics.json"
    div_path = Path(div_metrics_path) if div_metrics_path else RESULTS_DIR / "diversity_diagnostics.json"
    bench_path = Path(benchmark_report_path) if benchmark_report_path else RESULTS_DIR / "benchmark_report.json"
    bci_path = Path(bootstrap_cis_path) if bootstrap_cis_path else RESULTS_DIR / "bootstrap_cis.json"
    bl_path = Path(baseline_metrics_path) if baseline_metrics_path else RESULTS_DIR / "baseline_metrics.json"
    out_dir = Path(output_dir) if output_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-load training histories if not supplied.
    if clop_hist is None:
        clop_hist_path = CHECKPOINT_DIR / "CLOP" / "versions" / "latest" / "clop_history.json"
        if clop_hist_path.exists():
            try:
                with open(clop_hist_path) as f:
                    clop_hist = json.load(f)
            except Exception:
                clop_hist = None
    if dit_hist is None:
        dit_hist_path = CHECKPOINT_DIR / "DiT" / "versions" / "latest" / "dit_history.json"
        if dit_hist_path.exists():
            try:
                with open(dit_hist_path) as f:
                    dit_hist = json.load(f)
            except Exception:
                dit_hist = None

    data = _load_fig03_inputs(
        clop_hist=clop_hist,
        dit_hist=dit_hist,
        gen_metrics_path=gen_path,
        expr_metrics_path=expr_path,
        div_metrics_path=div_path,
        benchmark_report_path=bench_path,
        bootstrap_cis_path=bci_path,
        baseline_metrics_path=bl_path,
        cache_dir=cache,
    )

    if (not data["train_metrics"] and not data["gen_metrics"]
            and not data["per_type_gen"] and data["text_heatmap"] is None):
        logger.warning("No Fig 3 data sources found — skipping fig03_composed")
        return None

    # ── Layout: one figure, 3 rows (metrics / fidelity / alignment).
    # Row heights: row 1 = 1.0 (4 panels), row 2 = 0.95 (3 panels),
    # row 3 = 1.05 (3 panels incl. wide heatmap). Figsize mirrors the legacy
    # slices' combined vertical extent (fig05 8.4 + fig06 5.8 + fig07 7.4
    # ≈ 21.6 in; scaled slightly tighter with shared spacing).
    fig = plt.figure(figsize=(16.0, 19.2))
    layout = bind_figure_region(fig, (0.06, 0.04, 0.97, 0.97))
    row1, row2, row3 = layout.split_rows([1.00, 0.95, 1.05], hspace=0.60)

    # Row 1: 4 equal-weight columns (a/b/c/d).
    r1 = row1.split_cols([1.0, 1.0, 1.0, 1.0], gap=[0.045, 0.050, 0.060])
    ax_a = r1[0].inset(left=0.040, right=0.012).add_axes(fig)
    ax_b = r1[1].inset(left=0.040, right=0.012).add_axes(fig)
    ax_c = r1[2].inset(left=0.040, right=0.012).add_axes(fig)
    ax_d = r1[3].inset(left=0.040, right=0.055).add_axes(fig)

    # Row 2: 3 columns (e/f/g) with wider centre for Frechet + narrower
    # right for fidelity-vs-abundance scatter, mirroring fig06 weights.
    r2 = row2.split_cols([1.00, 1.18, 0.76], gap=[0.048, 0.044])
    ax_e = r2[0].inset(left=0.060, right=0.010).add_axes(fig)
    ax_f = r2[1].inset(left=0.110, right=0.018).add_axes(fig)
    ax_g = r2[2].inset(left=0.032, right=0.032).add_axes(fig)

    # Row 3: 3 columns (h/i/j) with wide heatmap, narrow bars, narrower hist.
    r3 = row3.split_cols([1.56, 0.92, 0.68], gap=[0.048, 0.036])
    ax_h = r3[0].inset(left=0.060, right=0.010).add_axes(fig)
    ax_i = r3[1].inset(left=0.100, right=0.014).add_axes(fig)
    ax_j = r3[2].inset(left=0.050, right=0.020).add_axes(fig)

    _draw_metrics_row(fig, ax_a, ax_b, ax_c, ax_d, data)
    _draw_fidelity_row(fig, ax_e, ax_f, ax_g, data)
    _draw_alignment_row(fig, ax_h, ax_i, ax_j, data)

    if save:
        path = save_with_vcd(
            fig, out_dir / "fig03_composed.png", dpi,
            layout_rect=(0.02, 0.01, 0.98, 0.98),
        )
        logger.info("Saved fig03_composed \u2192 %s", path)
        plt.close(fig)
        return out_dir / "fig03_composed.pdf"
    return None


# Direct-run entry point so ``python -m src.visualization.fig03_composed`` works.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    plot_fig03_composed()


__all__ = ["plot_fig03_composed"]
