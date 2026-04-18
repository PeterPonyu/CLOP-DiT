"""Fig 03: Training panels — CLOP and DiT training dynamics.

Publication-ready with minimum 10pt source fonts for composed_scale=0.70.
Extracted from ``ResultsVisualizer`` so that panels can be generated
independently (e.g. from a notebook or CI script) without instantiating
the full visualiser class.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import math

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.ticker import FixedLocator, MaxNLocator

from .direct_layout import bind_figure_region
from .style import COLORS, FONT_LEGEND_DENSE, FONT_LABEL, FONT_TITLE, SUPTITLE_Y_CLOSE, apply_style, save_with_vcd, set_figure_suptitle, set_scientific_tickformat, add_panel_label

logger = logging.getLogger(__name__)


LOSS_LEGEND_KW = {
    "loc": "upper center",
    "bbox_to_anchor": (0.50, -0.18),
    "fontsize": FONT_LEGEND_DENSE,
    "frameon": False,
    "ncol": 2,
}


def _add_training_phase_bands(ax: plt.Axes, max_epoch: int) -> None:
    """Annotate three coarse training phases for readability."""
    if max_epoch < 12:
        return
    p1_end = max(4, int(max_epoch * 0.06))
    p2_end = max(p1_end + 2, int(max_epoch * 0.9))
    ax.axvspan(1, p1_end, color=COLORS["accent"], alpha=0.08, lw=0)
    ax.axvspan(p1_end, p2_end, color=COLORS["neutral"], alpha=0.05, lw=0)
    ax.axvspan(p2_end, max_epoch, color=COLORS["baseline_gauss"], alpha=0.07, lw=0)
    ax.text(
        0.97, 0.78,
        "Phase I: rapid\nPhase II: refine\nPhase III: converge",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        color=COLORS.get("neutral", "#555555"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"),
    )


# ──────────────────────────────────────────────────────────────
# PANEL A: CLOP Training Dynamics
# ──────────────────────────────────────────────────────────────

def plot_clop_training(
    hist: Dict,
    output_dir: Path,
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """4-panel CLOP training dynamics.

    A1: Train/Val contrastive loss
    A2: Temperature curve (fixed @ 14.0)
    A3: Prototype accuracy (train/val + top-5, top-10)
    A4: Embedding quality (alignment, uniformity, inter-sep, t<->c alignment)

    Parameters
    ----------
    hist : dict
        CLOP training history containing at minimum ``train_loss``,
        ``val_loss``, ``temperature``, ``val_proto_acc``, and
        ``train_proto_acc``.  Optional keys: ``val_proto_top5``,
        ``val_proto_top10``, ``val_text_cell_align``, ``val_inter_sep``,
        ``val_mean_cosine_sim``.
    output_dir : Path
        Directory for saved figures.
    dpi : int
        Resolution for raster output.
    save : bool
        Whether to persist the figure to disk.
    save_panel_fn : callable, optional
        ``fn(fig, basename, output_dir, dpi)`` -- drop-in replacement for
        the default save helper (e.g. a VCD-integrated saver from the
        ``ResultsVisualizer``).  Falls back to ``_default_save``.

    Returns
    -------
    fig or None
        The matplotlib Figure, or *None* when *hist* is empty/missing.
    """
    if not hist:
        logger.warning("No CLOP history -- skipping Panel A")
        return None

    h = hist
    epochs = np.arange(1, len(h["train_loss"]) + 1)

    fig = plt.figure(figsize=(7.3, 5.7))
    layout = bind_figure_region(fig, (0.06, 0.10, 0.98, 0.94))
    top_row, bottom_row = layout.split_rows(2, hspace=0.34)
    top_left, top_right = top_row.split_cols(2, wspace=0.34)
    bottom_left, bottom_right = bottom_row.split_cols(2, wspace=0.34)
    # Note: Figure-level title removed per revision requirements; panel labels added below

    # ── A1: Loss curves ──
    ax_a1 = top_left.add_axes(fig)
    ax_a1.plot(epochs, h["train_loss"], label="Train", color=COLORS["real"])
    ax_a1.plot(epochs, h["val_loss"], label="Val", color=COLORS["generated"], linestyle="--")
    ax_a1.set_xlabel("Epoch", fontsize=11)
    ax_a1.set_ylabel("Contrastive Loss", fontsize=11)
    ax_a1.set_title("Loss Convergence", fontsize=12)
    ax_a1.legend(loc="upper right", fontsize=FONT_LEGEND_DENSE, frameon=False)
    ax_a1.set_xlim(0, max(epochs) * 1.05)
    ax_a1.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='upper'))
    ax_a1.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    add_panel_label(ax_a1, 'a', x=-0.10, y=1.12)

    # ── A2: Inter-type separation (replaces trivially flat temperature panel) ──
    ax_a2 = top_right.add_axes(fig)
    if "val_inter_sep" in h:
        ax_a2.plot(epochs, h["val_inter_sep"], color=COLORS["accent"], linewidth=2, label="Inter-sep")
        ax_a2.set_ylabel("Cosine Separation", fontsize=FONT_LABEL)
        ax_a2.set_title("Inter-Type Separation", fontsize=FONT_TITLE)
        ax_a2.set_ylim(0, 1.05)
        ax_a2.legend(loc="lower right", fontsize=FONT_LEGEND_DENSE, frameon=False)
    else:
        # Fallback: show fixed temperature as config note
        ax_a2.text(0.5, 0.5, "Fixed \u03c4 = 14.0\n(not learned)", transform=ax_a2.transAxes,
                   ha="center", va="center", fontsize=FONT_LABEL, color=COLORS["neutral"])
        ax_a2.set_title("Temperature Config", fontsize=FONT_TITLE)
    ax_a2.set_xlabel("Epoch", fontsize=FONT_LABEL)
    ax_a2.set_xlim(0, max(epochs) * 1.05)
    ax_a2.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
    add_panel_label(ax_a2, 'b', x=0.00, y=1.12)

    # ── A3: Prototype accuracy ──
    ax_a3 = bottom_left.add_axes(fig)
    ax_a3.plot(
        epochs, np.array(h["val_proto_acc"]) * 100,
        label="Val Acc", color=COLORS["baseline_shuffle"], linewidth=2,
    )
    ax_a3.plot(
        epochs, np.array(h["train_proto_acc"]) * 100,
        label="Train Acc", color=COLORS["baseline_shuffle"], linestyle=":", alpha=0.6,
    )
    if "val_proto_top5" in h:
        ax_a3.plot(
            epochs, np.array(h["val_proto_top5"]) * 100,
            label="Top-5", color=COLORS["neutral"], linestyle="--",
        )
    if "val_proto_top10" in h:
        ax_a3.plot(
            epochs, np.array(h["val_proto_top10"]) * 100,
            label="Top-10", color=COLORS["baseline_gauss"], linestyle="--",
        )
    ax_a3.set_xlabel("Epoch", fontsize=11)
    ax_a3.set_ylabel("Accuracy (%)", fontsize=11)
    ax_a3.set_title("Classification Accuracy", fontsize=12)
    ax_a3.set_ylim(0, 105)
    ax_a3.legend(loc="lower right", fontsize=FONT_LEGEND_DENSE, frameon=False, ncol=2)
    ax_a3.set_xlim(0, max(epochs) * 1.05)
    ax_a3.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
    ax_a3.yaxis.set_major_locator(MaxNLocator(nbins=4))
    add_panel_label(ax_a3, 'c', x=0.00, y=1.12)

    # ── A4: Embedding quality metrics ──
    ax_a4 = bottom_right.add_axes(fig)
    quality_metrics = [
        ("val_text_cell_align", "Text\u2194Cell", COLORS["accent"]),
        ("val_inter_sep", "Inter-sep", COLORS["warn"]),
        ("val_mean_cosine_sim", "Mean Cos", COLORS["real"]),
    ]
    for key, label, color in quality_metrics:
        if key in h:
            ax_a4.plot(epochs, h[key], label=label, color=color)
    ax_a4.set_xlabel("Epoch", fontsize=11)
    ax_a4.set_ylabel("Score", fontsize=11)
    ax_a4.set_title("Embedding Quality", fontsize=12)
    ax_a4.set_ylim(0, 1.05)
    ax_a4.legend(loc="center right", fontsize=FONT_LEGEND_DENSE, frameon=False)
    ax_a4.set_xlim(0, max(epochs) * 1.05)
    ax_a4.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
    ax_a4.yaxis.set_major_locator(MaxNLocator(nbins=4))
    add_panel_label(ax_a4, 'd', x=0.00, y=1.12)

    # ── Save ──
    if save:
        if save_panel_fn:
            save_panel_fn(fig, "fig03_clop_training", output_dir, dpi)
        else:
            save_with_vcd(fig, Path(output_dir) / "fig03_clop_training.png", dpi)
    return fig


# ──────────────────────────────────────────────────────────────
# PANEL C: DiT Training Dynamics
# ──────────────────────────────────────────────────────────────

def plot_dit_training(
    hist: Dict,
    output_dir: Path,
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """2x2 DiT flow-matching training dynamics.

    Matches Panel A layout for side-by-side placement.

    C1 (top-left):     Train/Val MSE loss (log scale)
    C2 (top-right):    Cosine similarity (fidelity)
    C3 (bottom-left):  Learning-rate schedule
    C4 (bottom-right): Convergence rate (train vs val normalized loss gap)

    Parameters
    ----------
    hist : dict
        DiT training history containing ``train_loss``, ``val_loss``,
        ``val_cosine_sim``, and ``lr``.
    output_dir : Path
        Directory for saved figures.
    dpi : int
        Resolution for raster output.
    save : bool
        Whether to persist the figure to disk.
    save_panel_fn : callable, optional
        ``fn(fig, basename, output_dir, dpi)`` -- drop-in replacement for
        the default save helper.

    Returns
    -------
    fig or None
        The matplotlib Figure, or *None* when *hist* is empty/missing.
    """
    if not hist:
        logger.warning("No DiT history -- skipping Panel C")
        return None

    h = hist
    epochs = np.arange(1, len(h["train_loss"]) + 1)

    fig = plt.figure(figsize=(9.5, 6.3))
    layout = bind_figure_region(fig, (0.06, 0.10, 0.98, 0.94))
    top_row, bottom_row = layout.split_rows(2, hspace=0.34)
    top_left, top_right = top_row.split_cols(2, wspace=0.34)
    bottom_left, bottom_right = bottom_row.split_cols(2, wspace=0.34)
    # Note: Figure-level title removed per revision requirements; panel labels added below

    # ── C1: Loss ──
    ax_c1 = top_left.add_axes(fig)
    ax_c1.plot(epochs, h["train_loss"], label="Train MSE", color=COLORS["real"])
    ax_c1.plot(epochs, h["val_loss"], label="Val MSE", color=COLORS["generated"], linestyle="--")
    ax_c1.set_xlabel("Epoch", fontsize=11)
    ax_c1.set_ylabel("Flow-Matching Loss", fontsize=11)
    ax_c1.set_title("Loss Convergence", fontsize=12)
    ax_c1.set_yscale("log")
    all_loss = list(h["train_loss"]) + list(h["val_loss"])
    _arr = [v for v in all_loss if v and v > 0]
    _ymin = max(min(_arr) * 0.6, 1e-5)
    _ymax = max(_arr) * 2.5
    ax_c1.set_ylim(_ymin, _ymax)
    ax_c1.legend(fontsize=FONT_LEGEND_DENSE, loc="upper right", frameon=False)
    ax_c1.set_xlim(0, max(epochs) * 1.02)
    from matplotlib.ticker import MaxNLocator, FixedLocator
    import math
    ax_c1.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="both"))
    # Explicit ticks: only powers of 10 strictly within the ylim (no overflow)
    _lo_exp = math.ceil(math.log10(_ymin * 1.01))
    _hi_exp = math.floor(math.log10(_ymax * 0.99))
    _decade_ticks = [10**e for e in range(_lo_exp, _hi_exp + 1)]
    if _decade_ticks:
        ax_c1.yaxis.set_major_locator(FixedLocator(_decade_ticks))
    add_panel_label(ax_c1, 'e', x=-0.10, y=1.08)

    # ── C2: Cosine similarity ──
    ax_c2 = top_right.add_axes(fig)
    ax_c2.plot(epochs, h["val_cosine_sim"], color=COLORS["baseline_gauss"], linewidth=2)
    ax_c2.set_xlabel("Epoch", fontsize=11)
    ax_c2.set_ylabel("Cosine Similarity", fontsize=11)
    ax_c2.set_title("Fidelity (Cosine)", fontsize=12)
    ax_c2.set_ylim(0.6, 1.0)
    ax_c2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
    ax_c2.set_xlim(0, max(epochs) * 1.05)
    ax_c2.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="both"))
    ax_c2.locator_params(axis='y', nbins=4)
    add_panel_label(ax_c2, 'f', x=-0.10, y=1.08)

    # ── C3: Learning rate ──
    ax_c3 = bottom_left.add_axes(fig)
    ax_c3.plot(epochs, h["lr"], color=COLORS["baseline_shuffle"], linewidth=1.5)
    ax_c3.set_xlabel("Epoch", fontsize=11)
    ax_c3.set_ylabel("Learning Rate", fontsize=11)
    ax_c3.set_title("LR Schedule", fontsize=12)
    set_scientific_tickformat(ax_c3, axis="y", scilimits=(-4, -4))
    ax_c3.set_xlim(0, max(epochs) * 1.05)
    ax_c3.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="both"))
    add_panel_label(ax_c3, 'g', x=0.00, y=1.08)

    # ── C4: Convergence rate (train vs val) ──
    ax_c4 = bottom_right.add_axes(fig)

    dit_tl = np.array(h["train_loss"])
    dit_vl = np.array(h["val_loss"])
    tl_gap = dit_tl[0] - dit_tl.min()
    vl_gap = dit_vl[0] - dit_vl.min()
    train_norm = (dit_tl - dit_tl.min()) / tl_gap if tl_gap > 1e-12 else np.zeros_like(dit_tl)
    val_norm = (dit_vl - dit_vl.min()) / vl_gap if vl_gap > 1e-12 else np.zeros_like(dit_vl)

    progress = np.linspace(0, 100, len(dit_tl))
    ax_c4.plot(progress, train_norm, label="Train", color=COLORS["real"], linewidth=1.5)
    ax_c4.plot(progress, val_norm, label="Val", color=COLORS["generated"],
               linewidth=1.5, linestyle="--")
    ax_c4.axhline(y=0.1, color="gray", linestyle=":", alpha=0.5, label="90% converged")
    ax_c4.set_xlabel("Training Progress (%)", fontsize=11)
    ax_c4.set_ylabel("Remaining Loss Gap", fontsize=11)
    ax_c4.set_title("Convergence Rate", fontsize=12)
    ax_c4.set_xlim(0, 100)
    ax_c4.set_ylim(-0.05, 1.05)
    ax_c4.legend(fontsize=FONT_LEGEND_DENSE, loc="upper right", frameon=False)
    ax_c4.locator_params(axis='x', nbins=4)
    ax_c4.locator_params(axis='y', nbins=4)
    add_panel_label(ax_c4, 'h', x=-0.10, y=1.08)

    # ── Save ──
    if save:
        if save_panel_fn:
            save_panel_fn(fig, "fig03_dit_training", output_dir, dpi)
        else:
            save_with_vcd(fig, Path(output_dir) / "fig03_dit_training.png", dpi)
    return fig


# ──────────────────────────────────────────────────────────────
# MERGED: Training Dynamics (A + C combined)
# ──────────────────────────────────────────────────────────────

def plot_training_dynamics_combined(
    clop_hist: Dict,
    dit_hist: Dict,
    output_dir: Path,
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Combined training dynamics figure.

    Top row (4 panels): CLOP — A1 Loss, A2 Temperature, A3 Accuracy, A4 Embedding Quality
    Bottom row (4 panels): DiT — C1 Loss, C2 Cosine Similarity, C3 LR Schedule, C4 Convergence Comparison

    Parameters
    ----------
    clop_hist, dit_hist : dict
        Training histories for CLOP and DiT respectively.
    output_dir : Path
        Directory for saved figures.
    dpi : int
        Resolution for raster output.
    save : bool
        Whether to persist the figure to disk.
    save_panel_fn : callable, optional
        Drop-in save helper.

    Returns
    -------
    fig or None
    """
    if not clop_hist and not dit_hist:
        logger.warning("No training histories — skipping combined training panel")
        return None

    apply_style()
    fig = plt.figure(figsize=(14.8, 9.4))
    layout = bind_figure_region(fig, (0.06, 0.10, 0.955, 0.87))
    top_row, bottom_row = layout.split_rows([1, 1], hspace=0.48)
    top_cols = top_row.split_cols([1.0, 1.0, 1.0, 1.2], wspace=0.42)
    bottom_cols = bottom_row.split_cols([1.0, 1.0, 1.0, 1.2], wspace=0.42)
    # Note: Figure-level title removed per revision requirements; panel labels added below

    # ════════════════════════════════════════════════════════════
    # Top row: CLOP (4 panels spanning columns 0-3)
    # ════════════════════════════════════════════════════════════
    if clop_hist:
        h = clop_hist
        epochs = np.arange(1, len(h["train_loss"]) + 1)

        # A1: Loss
        ax_a1 = top_cols[0].add_axes(fig)
        ax_a1.plot(epochs, h["train_loss"], label="Train", color=COLORS["real"])
        ax_a1.plot(epochs, h["val_loss"], label="Val", color=COLORS["generated"], linestyle="--")
        ax_a1.set_xlabel("Epoch", fontsize=11)
        ax_a1.set_ylabel("Contrastive Loss", fontsize=11)
        ax_a1.set_title("CLOP Loss", fontsize=12)
        ax_a1.legend(**LOSS_LEGEND_KW)
        ax_a1.set_xlim(0, max(epochs) * 1.05)
        ax_a1.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='upper'))
        ax_a1.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
        ax_a1.tick_params(axis='y', which='both', pad=2)
        _add_training_phase_bands(ax_a1, int(max(epochs)))
        add_panel_label(ax_a1, 'a', x=-0.12, y=1.14)

        # A2: Inter-type separation (replaces trivially flat temperature panel)
        ax_a2 = top_cols[1].add_axes(fig)
        if "val_inter_sep" in h:
            ax_a2.plot(epochs, h["val_inter_sep"], color=COLORS["accent"], linewidth=2, label="Inter-sep")
            ax_a2.set_ylabel("Cosine Separation", fontsize=FONT_LABEL)
            ax_a2.set_title("Inter-Type Separation", fontsize=FONT_TITLE)
            ax_a2.set_ylim(0, 1.05)
            ax_a2.legend(loc="lower right", fontsize=FONT_LEGEND_DENSE, frameon=False)
        else:
            # Fallback: show fixed temperature as config note
            ax_a2.text(0.5, 0.5, "Fixed \u03c4 = 14.0\n(not learned)", transform=ax_a2.transAxes,
                       ha="center", va="center", fontsize=FONT_LABEL, color=COLORS["neutral"])
            ax_a2.set_title("Temperature Config", fontsize=FONT_TITLE)
        ax_a2.set_xlabel("Epoch", fontsize=FONT_LABEL)
        ax_a2.set_xlim(0, max(epochs) * 1.05)
        ax_a2.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='upper'))
        add_panel_label(ax_a2, 'b', x=0.00, y=1.14)

        # A3: Accuracy
        ax_a3 = top_cols[2].add_axes(fig)
        ax_a3.plot(epochs, np.array(h["val_proto_acc"]) * 100,
                label="Val Acc", color=COLORS["baseline_shuffle"], linewidth=2)
        ax_a3.plot(epochs, np.array(h["train_proto_acc"]) * 100,
                label="Train Acc", color=COLORS["baseline_shuffle"], linestyle=":", alpha=0.6)
        if "val_proto_top5" in h:
            ax_a3.plot(epochs, np.array(h["val_proto_top5"]) * 100,
                    label="Top-5", color=COLORS["neutral"], linestyle="--")
        if "val_proto_top10" in h:
            ax_a3.plot(epochs, np.array(h["val_proto_top10"]) * 100,
                    label="Top-10", color=COLORS["baseline_gauss"], linestyle="--")
        ax_a3.set_xlabel("Epoch", fontsize=11)
        ax_a3.set_ylabel("Accuracy (%)", fontsize=11)
        ax_a3.set_title("Classification Accuracy", fontsize=12)
        ax_a3.set_ylim(0, 105)
        ax_a3.legend(loc="lower right", fontsize=FONT_LEGEND_DENSE, frameon=False, ncol=2)
        ax_a3.set_xlim(0, max(epochs) * 1.05)
        ax_a3.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='upper'))
        ax_a3.yaxis.set_major_locator(MaxNLocator(nbins=4))
        add_panel_label(ax_a3, 'c', x=0.00, y=1.14)

        # A4: Embedding quality
        ax_a4 = top_cols[3].add_axes(fig)
        quality_metrics = [
            ("val_text_cell_align", "Text\u2194Cell", COLORS["accent"]),
            ("val_inter_sep", "Inter-sep", COLORS["warn"]),
            ("val_mean_cosine_sim", "Mean Cos", COLORS["real"]),
        ]
        for key, label, color in quality_metrics:
            if key in h:
                ax_a4.plot(epochs, h[key], label=label, color=color)
        ax_a4.set_xlabel("Epoch", fontsize=11)
        ax_a4.set_ylabel("Score", fontsize=11)
        ax_a4.set_title("Embedding Quality", fontsize=12)
        ax_a4.set_ylim(0, 1.05)
        ax_a4.legend(loc="center right", fontsize=FONT_LEGEND_DENSE, frameon=False)
        ax_a4.set_xlim(0, max(epochs) * 1.05)
        ax_a4.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='upper'))
        ax_a4.yaxis.set_major_locator(MaxNLocator(nbins=4))
        add_panel_label(ax_a4, 'd', x=0.00, y=1.14)

    # ════════════════════════════════════════════════════════════
    # Bottom row: DiT (3 plots + 1 summary, columns 0-3)
    # ════════════════════════════════════════════════════════════
    if dit_hist:
        h = dit_hist
        epochs = np.arange(1, len(h["train_loss"]) + 1)

        # C1: Loss
        ax_c1 = bottom_cols[0].add_axes(fig)
        ax_c1.plot(epochs, h["train_loss"], label="Train MSE", color=COLORS["real"])
        ax_c1.plot(epochs, h["val_loss"], label="Val MSE", color=COLORS["generated"], linestyle="--")
        ax_c1.set_xlabel("Epoch", fontsize=11)
        ax_c1.set_ylabel("Flow-Matching Loss", fontsize=11)
        ax_c1.set_title("DiT Loss", fontsize=12)
        ax_c1.set_yscale("log")
        all_loss = list(h["train_loss"]) + list(h["val_loss"])
        _arr = [v for v in all_loss if v and v > 0]
        _ymin = max(min(_arr) * 0.6, 1e-5)
        _ymax = max(_arr) * 2.5
        ax_c1.set_ylim(_ymin, _ymax)
        ax_c1.legend(**LOSS_LEGEND_KW)
        ax_c1.set_xlim(0, max(epochs) * 1.02)
        ax_c1.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, prune="both"))
        _lo_exp = math.ceil(math.log10(_ymin * 1.01))
        _hi_exp = math.floor(math.log10(_ymax * 0.99))
        _decade_ticks = [10**e for e in range(_lo_exp, _hi_exp + 1)]
        if _decade_ticks:
            ax_c1.yaxis.set_major_locator(FixedLocator(_decade_ticks))
        _add_training_phase_bands(ax_c1, int(max(epochs)))
        add_panel_label(ax_c1, 'e', x=-0.12, y=1.08)

        # C2: Cosine similarity
        ax_c2 = bottom_cols[1].add_axes(fig)
        ax_c2.plot(epochs, h["val_cosine_sim"], color=COLORS["baseline_gauss"], linewidth=2)
        ax_c2.set_xlabel("Epoch", fontsize=11)
        ax_c2.set_ylabel("Cosine Similarity", fontsize=11)
        ax_c2.set_title("Fidelity (Cosine)", fontsize=12)
        ax_c2.set_ylim(0.6, 1.0)
        ax_c2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
        ax_c2.set_xlim(0, max(epochs) * 1.05)
        ax_c2.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, prune="both"))
        ax_c2.locator_params(axis='y', nbins=4)
        add_panel_label(ax_c2, 'f', x=-0.12, y=1.08)

        # C3: Learning rate
        ax_c3 = bottom_cols[2].add_axes(fig)
        ax_c3.plot(epochs, h["lr"], color=COLORS["baseline_shuffle"], linewidth=1.5)
        ax_c3.set_xlabel("Epoch", fontsize=11)
        ax_c3.set_ylabel("Learning Rate", fontsize=11)
        ax_c3.set_title("LR Schedule", fontsize=12)
        set_scientific_tickformat(ax_c3, axis="y", scilimits=(-4, -4))
        ax_c3.set_xlim(0, max(epochs) * 1.05)
        ax_c3.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, prune="both"))
        add_panel_label(ax_c3, 'g', x=0.00, y=1.08)

        # C4: Normalized convergence comparison (CLOP + DiT)
        ax_c4 = bottom_cols[3].add_axes(fig)

        # Normalize val loss: 1 = initial gap, 0 = fully converged
        dit_vl = np.array(h["val_loss"])
        dit_gap = dit_vl[0] - dit_vl.min()
        if dit_gap > 1e-12:
            dit_norm = (dit_vl - dit_vl.min()) / dit_gap
        else:
            dit_norm = np.zeros_like(dit_vl)
        dit_progress = np.linspace(0, 100, len(dit_vl))
        ax_c4.plot(dit_progress, dit_norm, label="DiT val loss",
                   color=COLORS["generated"], linewidth=1.8)

        if clop_hist:
            clop_vl = np.array(clop_hist["val_loss"])
            clop_gap = clop_vl[0] - clop_vl.min()
            if clop_gap > 1e-12:
                clop_norm = (clop_vl - clop_vl.min()) / clop_gap
            else:
                clop_norm = np.zeros_like(clop_vl)
            clop_progress = np.linspace(0, 100, len(clop_vl))
            ax_c4.plot(clop_progress, clop_norm, label="CLOP val loss",
                       color=COLORS["real"], linewidth=1.8)

        ax_c4.axhline(y=0.1, color="gray", linestyle=":", alpha=0.5,
                       label="90% converged")
        ax_c4.set_xlabel("Training Progress (%)", fontsize=11)
        ax_c4.set_ylabel("Remaining Loss Gap", fontsize=11)
        ax_c4.set_title("Convergence Comparison", fontsize=12)
        ax_c4.set_xlim(0, 100)
        ax_c4.set_ylim(0, 1.05)
        ax_c4.legend(fontsize=FONT_LEGEND_DENSE, loc="upper right", frameon=False)
        ax_c4.locator_params(axis='x', nbins=4)
        ax_c4.locator_params(axis='y', nbins=4)
        add_panel_label(ax_c4, 'h', x=-0.12, y=1.08)

    if save:
        if save_panel_fn:
            save_panel_fn(fig, "fig02a_training_dynamics", output_dir, dpi)
        else:
            save_with_vcd(fig, Path(output_dir) / "fig02a_training_dynamics.png", dpi)
    return fig
