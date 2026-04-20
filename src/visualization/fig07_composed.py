"""
fig07_composed.py — Article Figure 7 composite (Step 4 of the single-producer
architecture migration, plan
`.omc/plans/single-producer-architecture-2026-04-20.md`).

Emits ONE canonical PDF (``fig07_composed.pdf``) combining the 5 panels
currently split between ``fig07a_expression_diversity.pdf`` (from
``fig14_expr_diversity.py``) and ``fig07b_baseline_comparison.pdf`` (from
``fig15_baselines.py``) into a single ``plt.figure()`` with a single 2-row
gridspec so panel labels, fonts, and palettes are enforceable at the
module boundary rather than drifting across two source scripts.

Note: ``fig07c_benchmark.pdf`` (from ``fig16_benchmark.py``) is intentionally
EXCLUDED from this composite per plan §4 exemption. The Option I-a
labels-x3 structural design of the benchmark panel is not compatible with
the shared composite font/scale budget, so the benchmark panel continues
to publish as a separate standalone figure.

Layout (2 rows, labels a-e):

    Row 1 (from fig14_expr_diversity, 2 panels):
      a  Expression variability summary (cell/gene std, real vs gen)
      b  Per-type gene std ratio (min / mean / max)
    Row 2 (from fig15_baselines, 3 panels):
      c  Grouped bar chart — key metrics comparison
      d  Ranked dot plot — normalised scores
      e  Absolute delta bar chart — CLOP-DiT advantage

**Dual-publish:** this module is ADDITIVE — ``fig14_expr_diversity.py`` and
``fig15_baselines.py`` continue producing ``fig07a_*.pdf`` and
``fig07b_*.pdf`` unchanged. The composite is a registered-but-not-yet-LaTeX-
referenced asset during the revision window.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .article_composition import COMPOSITE_VCD_REGISTRY  # noqa: F401
from .direct_layout import bind_figure_region
from .style import (
    COLORS,
    FONT_ANNOTATION,
    FONT_SMALL,
    METHOD_COLORS,
    PANEL_OFFSET_FARLEFT_WIDE,
    PANEL_OFFSET_STD,
    PANEL_OFFSET_WIDE,
    add_panel_label,
    apply_style,
    save_with_vcd,
    style_axes,
)
from src.utils.paths import CACHE_DIR, FIG_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Register composite in the VCD cross-slice coverage registry at import time.
# Canonical registration lives in article_composition.py; the idempotent
# re-assignment here is belt-and-braces documentation for migration reviewers.
# Note: fig07c_benchmark.pdf is NOT listed here — it is exempt per plan §4
# (Option I-a labels-x3 structural design incompatible with the shared
# composite scale budget).
# ---------------------------------------------------------------------------
COMPOSITE_VCD_REGISTRY["fig07_composed.pdf"] = [
    "fig07a_expression_diversity.pdf",
    "fig07b_baseline_comparison.pdf",
]


# ---------------------------------------------------------------------------
# Data loaders — mirror the logic from fig14_expr_diversity /
# fig15_baselines so the composite reads the same cached artefacts that the
# per-slice producers read.
# ---------------------------------------------------------------------------

def _load_fig07_inputs(
    gen_metrics_path: Path,
    div_metrics_path: Path,
    baseline_metrics_path: Path,
    cache_dir: Path,
) -> Dict:
    """Load every JSON artefact needed by the two rows.

    Returns a dict with keys: t6_data (diversity test 6), methods (CLOP-DiT +
    baselines), method_names, metric_keys, metric_labels, directions.
    Returns an empty dict if no inputs are present.
    """
    out: Dict = {}

    # ── Diversity (Row 1, fig14) ─────────────────────────────────────────
    if div_metrics_path.exists():
        try:
            with open(div_metrics_path) as f:
                div_data = json.load(f)
            out["t6_data"] = div_data.get("test6_expression_diversity", {})
            out["div_ratio_clop"] = div_data.get(
                "test1_intratype_diversity", {}
            ).get("summary", {}).get("mean_diversity_ratio", 0)
        except Exception:
            out["t6_data"] = {}
            out["div_ratio_clop"] = 0.0
    else:
        out["t6_data"] = {}
        out["div_ratio_clop"] = 0.0

    # ── Generation + baselines (Row 2, fig15) ───────────────────────────
    if gen_metrics_path.exists():
        try:
            with open(gen_metrics_path) as f:
                gen_data = json.load(f)
            clop_metrics = gen_data.get("overall", {})
            clop_summary = gen_data.get("summary", {})
        except Exception:
            clop_metrics = {}
            clop_summary = {}
    else:
        clop_metrics = {}
        clop_summary = {}

    baselines: Dict[str, Dict] = {}
    if baseline_metrics_path.exists():
        try:
            with open(baseline_metrics_path) as f:
                baselines = json.load(f)
        except Exception:
            baselines = {}

    methods: Dict[str, Dict[str, float]] = {}
    if clop_metrics or clop_summary:
        methods["CLOP-DiT"] = {
            "FD": clop_metrics.get("frechet_distance", 0),
            "Centroid Cosine": clop_summary.get("mean_centroid_cosine", 0),
            "Diversity Ratio": out["div_ratio_clop"],
            "Coverage": clop_metrics.get("coverage", 0),
        }
    for bl_name, bl_data in baselines.items():
        methods[bl_name] = {
            "FD": bl_data.get("frechet_distance", 0),
            "Centroid Cosine": bl_data.get("mean_centroid_cosine", 0),
            "Diversity Ratio": bl_data.get("diversity_ratio", 0),
            "Coverage": bl_data.get("coverage", 0),
        }
    out["methods"] = methods
    out["method_names"] = list(methods.keys())
    out["metric_keys"] = ["FD", "Centroid Cosine", "Diversity Ratio", "Coverage"]
    out["metric_labels"] = ["FD \u2193", "Cosine \u2191",
                             "Diversity \u2191", "Coverage \u2191"]
    out["directions"] = ["lower", "higher", "higher", "higher"]
    return out


# ---------------------------------------------------------------------------
# Row renderers — mirror plot_expression_diversity_panel (fig14) and
# plot_baseline_comparison (fig15), adapted to pre-created axes.
# ---------------------------------------------------------------------------

def _draw_diversity_row(
    fig: plt.Figure,
    ax_a: plt.Axes,
    ax_b: plt.Axes,
    data: Dict,
) -> None:
    """Top row: expression variability summary (a), per-type gene std ratio (b).
    Mirrors fig14_expr_diversity.plot_expression_diversity_panel."""
    t6_data = data.get("t6_data", {}) or {}
    overall = t6_data.get("overall") if t6_data else None

    add_panel_label(ax_a, "a", x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
    if not overall:
        ax_a.text(0.5, 0.5, "No diversity data", ha="center", va="center",
                  transform=ax_a.transAxes, fontsize=10, color=COLORS["neutral"])
        ax_a.set_title("Expression Variability Summary", fontsize=12)
    else:
        labels = ["Cell Std\n(across genes)", "Gene Std\n(across cells)"]
        real_vals = [overall["real_mean_cell_std"], overall["real_mean_gene_std"]]
        gen_vals = [overall["gen_mean_cell_std"], overall["gen_mean_gene_std"]]
        x = np.arange(2)
        w = 0.35
        ax_a.bar(x - w / 2, real_vals, w, label="Real",
                 color=COLORS["real"], alpha=0.8)
        ax_a.bar(x + w / 2, gen_vals, w, label="Generated",
                 color=COLORS["generated"], alpha=0.8)
        # Ratio annotations above each bar pair.
        annotation_tops = []
        for _bi in range(len(real_vals)):
            if real_vals[_bi] > 0:
                ratio = gen_vals[_bi] / real_vals[_bi]
                max_h = max(real_vals[_bi], gen_vals[_bi])
                if ratio >= 1.03:
                    y_offset = 1.12
                elif ratio <= 0.97:
                    y_offset = 1.07
                else:
                    y_offset = 1.02
                ann_y = max_h * y_offset
                annotation_tops.append(ann_y)
                ax_a.text(_bi, ann_y + 0.02, f"ratio={ratio:.2f}",
                          ha="center", fontsize=9, fontweight="normal",
                          color=COLORS["annotation_dark"])
        if annotation_tops:
            ax_a.set_ylim(0, max(max(real_vals + gen_vals) * 1.08,
                                  max(annotation_tops) * 1.10))
        ax_a.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(labels, fontsize=10)
        ax_a.set_ylabel("Standard Deviation", fontsize=11)
        ax_a.set_title("Expression Variability Summary", fontsize=12, x=0.58)
        ax_a.legend(frameon=False, fontsize=9)

    add_panel_label(ax_b, "b", x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
    pt_ratio = t6_data.get("per_type_gene_std_ratio", {}) if t6_data else {}
    if pt_ratio:
        vals = [pt_ratio["min"], pt_ratio["mean"], pt_ratio["max"]]
        lbls = ["Min", "Mean", "Max"]
        colors = [COLORS["bad"] if v < 0.5 else COLORS["good"] for v in vals]
        ax_b.bar(lbls, vals, color=colors, alpha=0.8, edgecolor="white")
        ax_b.axhline(y=1.0, color="black", ls="--", lw=1, alpha=0.5,
                     label="ratio=1 (equal diversity)")
        ax_b.set_ylabel("Gene Std Ratio (gen / real)", fontsize=11)
        ax_b.set_title("Per-Type Gene Std Ratio", fontsize=12)
        ax_b.legend(fontsize=9, frameon=False)
    else:
        ax_b.text(0.5, 0.5, "No per-type data", ha="center", va="center",
                  transform=ax_b.transAxes, fontsize=10, color=COLORS["neutral"])
        ax_b.set_title("Per-Type Gene Std Ratio", fontsize=12)


def _draw_baseline_row(
    fig: plt.Figure,
    ax_c: plt.Axes,
    ax_d: plt.Axes,
    ax_e: plt.Axes,
    data: Dict,
) -> None:
    """Bottom row: grouped bar comparison (c), ranked dot plot (d),
    absolute delta bars (e). Mirrors fig15_baselines.plot_baseline_comparison."""
    methods = data.get("methods", {}) or {}
    method_names: List[str] = data.get("method_names", []) or []
    metric_keys: List[str] = data.get("metric_keys", [])
    metric_labels: List[str] = data.get("metric_labels", [])
    directions: List[str] = data.get("directions", [])
    n_methods = len(method_names)

    # ── Panel c: Grouped bar chart ───────────────────────────────────────
    add_panel_label(ax_c, "c", x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
    if not methods or n_methods == 0:
        ax_c.text(0.5, 0.5, "No method data", ha="center", va="center",
                  transform=ax_c.transAxes, fontsize=10, color=COLORS["neutral"])
        ax_c.set_title("Key Metrics Comparison", fontsize=12)
        add_panel_label(ax_d, "d", x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
        ax_d.text(0.5, 0.5, "No method data", ha="center", va="center",
                  transform=ax_d.transAxes, fontsize=10, color=COLORS["neutral"])
        ax_d.set_title("Normalised Scores", fontsize=12)
        # Column-3 panel — bare 0 x-offset per the fig06 G3 convention.
        add_panel_label(ax_e, "e", x=0, y=PANEL_OFFSET_STD[1])
        ax_e.text(0.5, 0.5, "No baseline data", ha="center", va="center",
                  transform=ax_e.transAxes, fontsize=10, color=COLORS["neutral"])
        ax_e.set_title("CLOP-DiT Advantage", fontsize=12)
        return

    x = np.arange(len(metric_labels))
    w = 0.8 / max(n_methods, 1)
    handles_c: List = []
    labels_c: List[str] = []
    for i, mname in enumerate(method_names):
        vals = [methods[mname][k] for k in metric_keys]
        offset = (i - n_methods / 2 + 0.5) * w
        bar = ax_c.bar(x + offset, vals, w, label=mname,
                       color=METHOD_COLORS.get(mname, COLORS["neutral"]),
                       alpha=0.85, edgecolor="white")
        handles_c.append(bar)
        labels_c.append(mname)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(metric_labels, fontsize=10, rotation=0, ha="center")
    # Inline legend keeps the composite self-contained (no add_shared_legend_axes
    # which depends on a reserved band at the bottom of the figure).
    ax_c.legend(fontsize=FONT_SMALL, frameon=False, loc="upper left",
                ncol=min(n_methods, 2), handlelength=1.2, handletextpad=0.4,
                columnspacing=0.8)
    style_axes(ax_c, "bar", title="Key Metrics Comparison", ylabel="Value")

    # ── Panel d: Ranked dot plot (normalised scores) ─────────────────────
    add_panel_label(ax_d, "d", x=PANEL_OFFSET_STD[0], y=PANEL_OFFSET_STD[1])
    all_vals = {k: [methods[m][k] for m in method_names] for k in metric_keys}
    normalized: Dict[str, List[float]] = {}
    for k, direction in zip(metric_keys, directions):
        mn, mx = min(all_vals[k]), max(all_vals[k])
        rng = mx - mn if mx > mn else 1
        if direction == "lower":
            normalized[k] = [(mx - v) / rng for v in all_vals[k]]
        else:
            normalized[k] = [(v - mn) / rng for v in all_vals[k]]
    agg_scores: Dict[str, float] = {}
    for i, mname in enumerate(method_names):
        agg_scores[mname] = float(np.mean([normalized[k][i] for k in metric_keys]))
    sorted_methods = sorted(method_names, key=lambda m: agg_scores[m], reverse=True)
    y_pos = np.arange(len(sorted_methods))
    for j, mk in enumerate(metric_keys):
        vals = [normalized[mk][method_names.index(m)] for m in sorted_methods]
        ax_d.scatter(vals, y_pos, s=96, marker="oDsv"[j],
                     color=f"C{j}", alpha=0.85, zorder=3,
                     label=metric_labels[j])
    for i, mname in enumerate(sorted_methods):
        vals = [normalized[mk][method_names.index(mname)] for mk in metric_keys]
        ax_d.plot(vals, [i] * len(vals),
                  color=METHOD_COLORS.get(mname, "#999"),
                  linewidth=1.2, alpha=0.4, zorder=1)
        score_x = min(1.08, max(vals) + 0.04)
        ax_d.text(score_x, i, f"{agg_scores[mname]:.2f}", va="center",
                  ha="left", fontsize=FONT_SMALL + 1, color=COLORS["neutral"])
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(sorted_methods, fontsize=10)
    ax_d.set_xlim(-0.05, 1.15)
    ax_d.invert_yaxis()
    ax_d.legend(fontsize=FONT_SMALL, frameon=False, loc="lower right",
                ncol=2, handlelength=1.0, handletextpad=0.3,
                columnspacing=0.6)
    style_axes(ax_d, "default", title="Normalised Scores (1 = best)",
               xlabel="Normalised Value")

    # ── Panel e: Absolute delta bar chart (CLOP-DiT - baseline) ─────────
    # Column-3 panel with wide y-tick strings ("Pea | Gaussian N..."); use
    # the wider FARLEFT offset so the label clears the long tick labels.
    add_panel_label(ax_e, "e",
                    x=PANEL_OFFSET_FARLEFT_WIDE[0],
                    y=PANEL_OFFSET_FARLEFT_WIDE[1])
    clop_vals = methods.get("CLOP-DiT", {})
    bl_names = [m for m in method_names if m != "CLOP-DiT"]
    y_labels: List[str] = []
    y_vals: List[float] = []
    for bl_name in bl_names:
        bl_vals = methods.get(bl_name, {})
        for mk, direction in zip(metric_keys, directions):
            cv = clop_vals.get(mk, 0)
            bv = bl_vals.get(mk, 0)
            if direction == "lower":
                delta = bv - cv
            else:
                delta = cv - bv
            short_bl = bl_name[:14]
            y_labels.append(f"{mk[:3]} | {short_bl}")
            y_vals.append(delta)
    if not y_labels:
        ax_e.text(0.5, 0.5, "No baseline data", ha="center", va="center",
                  transform=ax_e.transAxes, fontsize=10, color=COLORS["neutral"])
        ax_e.set_title("CLOP-DiT Advantage", fontsize=12)
        return
    y_pos = np.arange(len(y_labels))
    bar_colors_final = [COLORS["good"] if v > 0 else COLORS["bad"] for v in y_vals]
    ax_e.barh(y_pos, y_vals, color=bar_colors_final, height=0.6,
              edgecolor="white", linewidth=0.5, alpha=0.85)
    for i, val in enumerate(y_vals):
        x_text = val + 0.03 if val >= 0 else min(-0.02, val + 0.08)
        ax_e.text(x_text, i, f"{val:+.3f}", va="center", ha="left",
                  fontsize=FONT_ANNOTATION, color="#333")
    ax_e.set_yticks(y_pos)
    ax_e.set_yticklabels(y_labels, fontsize=8, ha="right")
    ax_e.axvline(x=0, color=COLORS["neutral"], linewidth=1.2)
    ax_e.invert_yaxis()
    n_metrics = len(metric_keys)
    for g in range(1, len(bl_names)):
        sep_y = g * n_metrics - 0.5
        ax_e.axhline(y=sep_y, color="#DDD", linewidth=1, linestyle="--")
    style_axes(ax_e, "bar", title="CLOP-DiT Advantage (\u0394 metric)",
               xlabel="Absolute Improvement")
    ax_e.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_fig07_composed(
    gen_metrics_path: Optional[str] = None,
    div_metrics_path: Optional[str] = None,
    baseline_metrics_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    dpi: int = 300,
    save: bool = True,
) -> Optional[Path]:
    """Render the 2-row composite Fig 7 (panels a/b from fig14 + c/d/e from
    fig15) and save to ``fig07_composed.{pdf,png}``.

    Parameters
    ----------
    gen_metrics_path       : path to generation_metrics.json.
    div_metrics_path       : path to diversity_diagnostics.json.
    baseline_metrics_path  : path to baseline_metrics.json.
    cache_dir              : directory with text_captions_deduplicated.json etc.
    output_dir             : directory to save the composite PDF.
    dpi                    : rasterised preview DPI (PDF is vector).
    save                   : if False, return None without writing.

    Returns
    -------
    Path to the saved PDF, or ``None`` if no data sources are available.
    """
    apply_style()

    gen_path = Path(gen_metrics_path) if gen_metrics_path else RESULTS_DIR / "generation_metrics.json"
    div_path = Path(div_metrics_path) if div_metrics_path else RESULTS_DIR / "diversity_diagnostics.json"
    bl_path = Path(baseline_metrics_path) if baseline_metrics_path else RESULTS_DIR / "baseline_metrics.json"
    cache = Path(cache_dir) if cache_dir else CACHE_DIR
    out_dir = Path(output_dir) if output_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_fig07_inputs(
        gen_metrics_path=gen_path,
        div_metrics_path=div_path,
        baseline_metrics_path=bl_path,
        cache_dir=cache,
    )

    if not data.get("t6_data") and not data.get("methods"):
        logger.warning("No Fig 7 data sources found — skipping fig07_composed")
        return None

    # ── Layout: one figure, 2 rows.
    # Row 1 is the diversity pair (fig14 source, 2 panels); Row 2 is the
    # baselines trio (fig15 source, 3 panels). Figsize width 10.0in is the
    # Step-3 tuning target: when LaTeX scales the PDF to \textwidth (7in),
    # the effective composed scale is 0.70 so the minimum_font_size gate
    # (size * 0.70 >= 7pt) is satisfied for 10pt+ body text.
    fig = plt.figure(figsize=(10.0, 9.0))
    layout = bind_figure_region(fig, (0.06, 0.04, 0.97, 0.96))
    row1, row2 = layout.split_rows([0.88, 1.08], hspace=0.62)

    # Row 1: 2 equal-weight columns (a, b). Fig14's source is 15.2x3.6 with a
    # left inset of 0.06 and right inset of 0.04; we mirror those proportions
    # per column here.
    r1 = row1.split_cols([1.0, 1.0], gap=0.08)
    ax_a = r1[0].inset(left=0.070, right=0.020).add_axes(fig)
    ax_b = r1[1].inset(left=0.070, right=0.020).add_axes(fig)

    # Row 2: 3 columns (c, d, e) with weights mirroring fig15's
    # [1.20, 0.86, 1.20] split for O1 / O2 / O3.
    r2 = row2.split_cols([1.20, 0.86, 1.20], gap=[0.055, 0.055])
    ax_c = r2[0].inset(left=0.060, right=0.020).add_axes(fig)
    ax_d = r2[1].inset(left=0.040, right=0.020).add_axes(fig)
    ax_e = r2[2].inset(left=0.160, right=0.030).add_axes(fig)

    _draw_diversity_row(fig, ax_a, ax_b, data)
    _draw_baseline_row(fig, ax_c, ax_d, ax_e, data)

    if save:
        path = save_with_vcd(
            fig, out_dir / "fig07_composed.png", dpi,
            layout_rect=(0.02, 0.01, 0.98, 0.98),
        )
        logger.info("Saved fig07_composed \u2192 %s", path)
        plt.close(fig)
        return out_dir / "fig07_composed.pdf"
    return None


# Direct-run entry point so ``python -m src.visualization.fig07_composed`` works.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    plot_fig07_composed()


__all__ = ["plot_fig07_composed"]
