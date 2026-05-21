"""
figS3_expression_decoder.py -- Supplementary Figure S3: Expression Fidelity & Decoder Analysis.

Composite figure merging three analyses into a 2x2 layout:
  Top row:    (a) Variance scatter   (b) Variance ratio histogram
  Bottom row: (c) Augmentation F1    (d) Decoder ablation comparison

Data sources:
  - Panels a, b: results/real_expression.npy, results/generated_expression.npy,
                  results/expression_gene_names.json  (from Fig 24)
  - Panel c:     results/downstream/embedding_augmentation.json  (from Fig 29)
  - Panel d:     results/ablations/decoder/{baseline,lora_light,mlp}/metrics.json  (from Fig 31)

Missing data files are handled gracefully: the panel shows a placeholder message.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..direct_layout import bind_figure_region
from ..style import (
    COLORS,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TICK,
    FONT_TITLE,
    add_panel_label,
    apply_style,
    save_with_vcd,
    style_axes,
)
from src.utils.paths import RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)

# Decoder ablation display config (mirrors fig31)
APPROACH_LABELS = {
    "baseline": "scGPT",
    "lora_light": "+LoRA",
    "mlp": "MLP",
}

APPROACH_COLORS = {
    "baseline": COLORS.get("real", "#2196F3"),
    "lora_light": COLORS.get("generated", "#4CAF50"),
    "mlp": COLORS.get("baseline_3", "#FF9800"),
}


def _placeholder(ax: plt.Axes, label: str, panel_id: str, *, label_fontsize: int = FONT_TITLE) -> None:
    """Show a 'data not available' message on the given axes."""
    ax.text(
        0.5, 0.5, f"{label}\nData not available",
        ha="center", va="center", fontsize=FONT_LABEL,
        transform=ax.transAxes, color="#999999",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    style_axes(ax)
    add_panel_label(ax, panel_id, fontsize=label_fontsize)


# ─────────────────────────────────────────────────────────────
# Panel a: Variance scatter (log-log)
# ─────────────────────────────────────────────────────────────
def _panel_a(ax: plt.Axes, real_var: np.ndarray, gen_var: np.ndarray) -> None:
    """Scatter of real vs generated per-gene variance on log-log axes."""
    eps = 1e-10
    valid = (real_var > eps) & (gen_var > eps)
    rv = real_var[valid]
    gv = gen_var[valid]

    # Pearson r on log-transformed values
    r = np.corrcoef(np.log10(rv), np.log10(gv))[0, 1] if valid.sum() > 2 else 0.0

    # Color by deviation from identity (|log2 ratio|)
    log2_ratio = np.log2(gv / rv)
    deviation = np.abs(log2_ratio)
    sc = ax.scatter(
        rv, gv, c=deviation, cmap="RdYlBu_r", s=4, alpha=0.5,
        edgecolors="none", vmin=0, vmax=3,
    )

    # Identity line
    lo = min(rv.min(), gv.min()) * 0.5
    hi = max(rv.max(), gv.max()) * 2.0
    ax.plot([lo, hi], [lo, hi], "--", color=COLORS.get("neutral", "#999"),
            linewidth=1, alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Real variance", fontsize=FONT_LABEL)
    ax.set_ylabel("Gen. variance", fontsize=FONT_LABEL)
    ax.set_title("Variance Scatter", fontsize=FONT_TITLE, fontweight="normal")
    ax.tick_params(labelsize=FONT_TICK)

    # Move annotation to bottom-right where scatter density is low and the
    # identity line passes through the upper-left region.
    ax.text(
        0.97, 0.06,
        f"r = {r:.3f}\nn = {valid.sum()} genes",
        transform=ax.transAxes, fontsize=FONT_LEGEND,
        va="bottom", ha="right", color=COLORS.get("annotation_dark", "#333"),
    )

    style_axes(ax)


# ─────────────────────────────────────────────────────────────
# Panel b: Variance ratio histogram
# ─────────────────────────────────────────────────────────────
def _panel_b(ax: plt.Axes, real_var: np.ndarray, gen_var: np.ndarray) -> None:
    """Histogram of log2(var_gen / var_real)."""
    eps = 1e-10
    valid = (real_var > eps) & (gen_var > eps)
    log2_ratio = np.log2(gen_var[valid] / real_var[valid])

    ax.hist(log2_ratio, bins=60, color=COLORS.get("generated", "#4CAF50"),
            edgecolor="none", alpha=0.75)
    ax.axvline(0, color=COLORS.get("neutral", "#999"), ls="--", lw=1.2, label="Perfect match")

    # Fraction of genes with >2-fold deficit (ratio < -1 in log2).
    # Place annotation in lower-right where histogram density is low.
    frac_deficit = np.mean(log2_ratio < -1)
    ax.text(
        0.97, 0.38,
        f">2-fold deficit:\n{frac_deficit:.1%} of {valid.sum()} genes",
        transform=ax.transAxes, fontsize=FONT_LEGEND,
        va="top", ha="right", color=COLORS.get("annotation_dark", "#333"),
    )

    ax.set_xlabel(r"$\log_2$(var$_{\rm gen}$ / var$_{\rm real}$)", fontsize=FONT_LABEL)
    ax.set_ylabel("Gene count", fontsize=FONT_LABEL)
    ax.set_title("Variance Ratio", fontsize=FONT_TITLE, fontweight="normal")
    ax.tick_params(labelsize=FONT_TICK)
    # Place "Perfect match" legend inside the panel at upper-left to avoid
    # it crowding the x-axis below; frameon=False keeps it unobtrusive.
    ax.legend(fontsize=FONT_LEGEND, frameon=False, loc="upper left",
              bbox_to_anchor=(0.02, 0.98))

    style_axes(ax)


# ─────────────────────────────────────────────────────────────
# Panel c: Augmentation delta-F1 bar chart
# ─────────────────────────────────────────────────────────────
def _panel_c(ax: plt.Axes, data: dict) -> None:
    """Bar chart of delta-F1 at 1x, 5x, 10x augmentation ratios."""
    ratios, delta_f1 = [], []
    for key in ["ratio_1x", "ratio_5x", "ratio_10x"]:
        if key not in data:
            continue
        label = key.replace("ratio_", "")
        ratios.append(label)
        delta_f1.append(data[key].get("delta_f1", 0))

    if not ratios:
        _placeholder(ax, "Augmentation F1", "k")
        return

    x = np.arange(len(ratios))
    bars = ax.bar(x, delta_f1, 0.55, color=COLORS.get("generated", "#4CAF50"),
                  edgecolor="none", alpha=0.85)

    # Horizontal dashed line at 0
    ax.axhline(0, color=COLORS.get("neutral", "#999"), ls="--", lw=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(ratios, fontsize=FONT_TICK)
    ax.set_xlabel("Augmentation ratio", fontsize=FONT_LABEL)
    ax.set_ylabel(r"$\Delta$ F1", fontsize=FONT_LABEL)
    ax.set_title("Embedding Augmentation", fontsize=FONT_TITLE, fontweight="normal", loc="right")

    style_axes(ax)


# ─────────────────────────────────────────────────────────────
# Panel d: Decoder ablation grouped bars
# ─────────────────────────────────────────────────────────────
def _panel_d(ax: plt.Axes, metrics: dict, approaches: list[str]) -> None:
    """Grouped bars showing per-gene std for each decoder approach."""
    from matplotlib.patches import Patch

    labels = [APPROACH_LABELS.get(a, a) for a in approaches]
    colors = [APPROACH_COLORS.get(a, "#999") for a in approaches]
    x = np.arange(len(approaches))
    w = 0.35

    real_vals = [metrics[a].get("real_per_gene_std", 0) for a in approaches]
    gen_vals = [metrics[a].get("gen_per_gene_std", 0) for a in approaches]

    ax.bar(x - w / 2, real_vals, w, color="#90CAF9", edgecolor="none")
    ax.bar(x + w / 2, gen_vals, w, color=colors, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK, rotation=25, ha="right")
    ax.set_ylabel("Per-gene std", fontsize=FONT_LABEL)
    ax.set_title("Decoder Comparison", fontsize=FONT_TITLE, fontweight="normal", loc="right")

    # Build legend with real + per-approach generated entries
    legend_handles = [Patch(facecolor="#90CAF9", label="Real")]
    for a in approaches:
        legend_handles.append(
            Patch(facecolor=APPROACH_COLORS.get(a, "#999"),
                  label=APPROACH_LABELS.get(a, a))
        )
    # Move legend further below (bbox y = -0.30) to clear the rightmost bars
    # and the x-tick rotation labels, and switch to ncol=4 (all on one row)
    # to minimise vertical extent.
    ax.legend(handles=legend_handles, fontsize=FONT_LEGEND, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4)

    style_axes(ax)


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────
def plot_expression_decoder(
    output_dir: str | Path = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Render Supplementary Figure S3: Expression Fidelity & Decoder Analysis.

    Parameters
    ----------
    output_dir : figure output directory
    dpi : export resolution
    save : whether to save figure
    save_panel_fn : optional callback for custom saving

    Returns
    -------
    fig : Figure or None on error
    """
    apply_style()
    output_dir = Path(output_dir)

    # ── Load data for panels a & b (variance) ──
    real_var, gen_var = None, None
    try:
        real_expr = np.load(RESULTS_DIR / "real_expression.npy")
        gen_expr = np.load(RESULTS_DIR / "generated_expression.npy")
        real_var = np.var(real_expr, axis=0)
        gen_var = np.var(gen_expr, axis=0)
    except FileNotFoundError:
        logger.warning("Expression data not found — panels a, b will be skipped")

    # ── Load data for panel c (augmentation) ──
    aug_data = None
    aug_path = RESULTS_DIR / "downstream" / "embedding_augmentation.json"
    if aug_path.exists():
        try:
            with open(aug_path) as f:
                aug_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load augmentation data: %s", e)
    else:
        logger.warning("Augmentation data not found: %s", aug_path)

    # ── Load data for panel d (decoder ablation) ──
    ablation_dir = RESULTS_DIR / "ablations" / "decoder"
    dec_approaches: list[str] = []
    dec_metrics: dict = {}
    for name in ["baseline", "lora_light", "mlp"]:
        mpath = ablation_dir / name / "metrics.json"
        if not mpath.exists():
            continue
        try:
            with open(mpath) as f:
                dec_metrics[name] = json.load(f)
            dec_approaches.append(name)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load decoder metrics for %s: %s", name, e)

    # ── Build figure ──
    # figsize=(8,5.5) chosen so that at 0.48\textwidth composition the text
    # scale (~0.40x) matches figS01a at 0.96\textwidth with figsize=16.
    _S3_LABEL_SIZE = 18  # keep non-article supplement labels inside the refreshed 18–22pt band

    fig = plt.figure(figsize=(6.0, 5.5))
    # Wider left margin for log-scale y-axis labels
    layout = bind_figure_region(fig, (0.15, 0.12, 0.96, 0.94))
    row_top, row_bot = layout.split_rows([1, 1], gap=0.34)

    col_a, col_b = row_top.split_cols([1, 1], wspace=0.44)
    col_c, col_d = row_bot.split_cols([1, 1], wspace=0.44)

    _label_pos = {
        "i": (0.02, 1.02),   # move inside to avoid the log-scale ytick column
        "j": (-0.08, 1.10),
        "k": (-0.08, 0.98),
        "l": (-0.08, 0.98),
    }

    # ── Panel a: Variance scatter ──
    ax_a = col_a.add_axes(fig)
    if real_var is not None and gen_var is not None:
        _panel_a(ax_a, real_var, gen_var)
    else:
        _placeholder(ax_a, "Variance Scatter", "i", label_fontsize=_S3_LABEL_SIZE)
    add_panel_label(ax_a, "i", x=_label_pos["i"][0], y=_label_pos["i"][1], fontsize=_S3_LABEL_SIZE)

    # ── Panel b: Variance ratio histogram ──
    ax_b = col_b.add_axes(fig)
    if real_var is not None and gen_var is not None:
        _panel_b(ax_b, real_var, gen_var)
    else:
        _placeholder(ax_b, "Variance Ratio", "j", label_fontsize=_S3_LABEL_SIZE)
    add_panel_label(ax_b, "j", x=_label_pos["j"][0], y=_label_pos["j"][1], fontsize=_S3_LABEL_SIZE)

    # ── Panel c: Augmentation F1 ──
    ax_c = col_c.add_axes(fig)
    if aug_data is not None:
        _panel_c(ax_c, aug_data)
    else:
        _placeholder(ax_c, "Augmentation F1", "k", label_fontsize=_S3_LABEL_SIZE)
    add_panel_label(ax_c, "k", x=_label_pos["k"][0], y=_label_pos["k"][1], fontsize=_S3_LABEL_SIZE)

    # ── Panel d: Decoder ablation ──
    ax_d = col_d.add_axes(fig)
    if len(dec_approaches) >= 2:
        _panel_d(ax_d, dec_metrics, dec_approaches)
    else:
        _placeholder(ax_d, "Decoder Comparison", "l", label_fontsize=_S3_LABEL_SIZE)
    add_panel_label(ax_d, "l", x=_label_pos["l"][0], y=_label_pos["l"][1], fontsize=_S3_LABEL_SIZE)

    # ── Save ──
    if save:
        stem = output_dir / "figS01c_expression_decoder"
        if save_panel_fn is not None:
            save_panel_fn(fig, stem)
        else:
            save_with_vcd(fig, stem, dpi=dpi)
        logger.info("Saved %s", stem)

    return fig


# Allow direct execution for debugging
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    logging.basicConfig(level=logging.INFO)
    fig = plot_expression_decoder()
