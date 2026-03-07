"""Training panels (A, C) -- CLOP and DiT training dynamics.

Publication-ready with minimum 10pt source fonts for composed_scale=0.70.
Extracted from ``ResultsVisualizer`` so that panels can be generated
independently (e.g. from a notebook or CI script) without instantiating
the full visualiser class.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, FONT_LEGEND_DENSE, SUPTITLE_Y_CLOSE, apply_style, save_with_vcd, set_figure_suptitle, set_scientific_tickformat, add_panel_label

logger = logging.getLogger(__name__)


def _add_training_phase_bands(ax: plt.Axes, max_epoch: int) -> None:
    """Annotate three coarse training phases for readability."""
    if max_epoch < 12:
        return
    p1_end = max(4, int(max_epoch * 0.06))
    p2_end = max(p1_end + 2, int(max_epoch * 0.9))
    ax.axvspan(1, p1_end, color=COLORS["accent"], alpha=0.08, lw=0)
    ax.axvspan(p1_end, p2_end, color=COLORS["neutral"], alpha=0.05, lw=0)
    ax.axvspan(p2_end, max_epoch, color=COLORS["baseline_gauss"], alpha=0.07, lw=0)
    ax.text(0.03, 0.97, "Phase I: rapid\nPhase II: refine\nPhase III: converge",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75))


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

    fig = plt.figure(figsize=(7.4, 6.0))
    gs = fig.add_gridspec(2, 2, wspace=0.45, hspace=0.50)
    # Note: Figure-level title removed per revision requirements; panel labels added below

    # ── A1: Loss curves ──
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a1.plot(epochs, h["train_loss"], label="Train", color=COLORS["real"])
    ax_a1.plot(epochs, h["val_loss"], label="Val", color=COLORS["generated"], linestyle="--")
    ax_a1.set_xlabel("Epoch", fontsize=10)
    ax_a1.set_ylabel("Contrastive Loss", fontsize=10)
    ax_a1.set_title("Loss Convergence", fontsize=11)
    ax_a1.legend(loc="upper right", fontsize=FONT_LEGEND_DENSE, frameon=False)
    ax_a1.set_xlim(0, max(epochs) * 1.08)
    ax_a1.locator_params(axis='x', nbins=3)
    ax_a1.locator_params(axis='y', nbins=4)
    add_panel_label(ax_a1, 'a', x=0.02, y=0.98)

    # ── A2: Temperature stability ──
    ax_a2 = fig.add_subplot(gs[0, 1])
    ax_a2.plot(epochs, h["temperature"], color=COLORS["baseline_gauss"], linewidth=2)
    ax_a2.set_xlabel("Epoch", fontsize=10)
    ax_a2.set_ylabel("Temperature (\u03c4)", fontsize=10)
    ax_a2.set_title("Temperature Stability", fontsize=11)
    ax_a2.axhline(y=14.0, color="gray", linestyle=":", alpha=0.5, label="\u03c4=14.0")
    ax_a2.set_ylim(13.5, 14.5)
    ax_a2.legend(loc="lower right", fontsize=FONT_LEGEND_DENSE, frameon=False)
    ax_a2.set_xlim(0, max(epochs) * 1.08)
    ax_a2.locator_params(axis='x', nbins=4)
    ax_a2.locator_params(axis='y', nbins=4)
    add_panel_label(ax_a2, 'b', x=0.02, y=0.98)

    # ── A3: Prototype accuracy ──
    ax_a3 = fig.add_subplot(gs[1, 0])
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
    ax_a3.set_xlabel("Epoch", fontsize=10)
    ax_a3.set_ylabel("Accuracy (%)", fontsize=10)
    ax_a3.set_title("Classification Accuracy", fontsize=11)
    ax_a3.set_ylim(0, 105)
    ax_a3.legend(loc="lower right", fontsize=FONT_LEGEND_DENSE, frameon=False, ncol=2)
    ax_a3.set_xlim(0, max(epochs) * 1.08)
    ax_a3.locator_params(axis='x', nbins=4)
    ax_a3.locator_params(axis='y', nbins=4)
    add_panel_label(ax_a3, 'c', x=0.02, y=0.98)

    # ── A4: Embedding quality metrics ──
    ax_a4 = fig.add_subplot(gs[1, 1])
    quality_metrics = [
        ("val_text_cell_align", "Text\u2194Cell", COLORS["accent"]),
        ("val_inter_sep", "Inter-sep", COLORS["warn"]),
        ("val_mean_cosine_sim", "Mean Cos", COLORS["real"]),
    ]
    for key, label, color in quality_metrics:
        if key in h:
            ax_a4.plot(epochs, h[key], label=label, color=color)
    ax_a4.set_xlabel("Epoch", fontsize=10)
    ax_a4.set_ylabel("Score", fontsize=10)
    ax_a4.set_title("Embedding Quality", fontsize=11)
    ax_a4.set_ylim(0, 1.05)
    ax_a4.legend(loc="center right", fontsize=FONT_LEGEND_DENSE, frameon=False)
    ax_a4.set_xlim(0, max(epochs) * 1.08)
    ax_a4.locator_params(axis='x', nbins=4)
    ax_a4.locator_params(axis='y', nbins=4)
    add_panel_label(ax_a4, 'd', x=0.02, y=0.98)

    # ── Save ──
    if save:
        if save_panel_fn:
            save_panel_fn(fig, "panel_a_clop_training", output_dir, dpi)
        else:
            save_with_vcd(fig, Path(output_dir) / "panel_a_clop_training.png", dpi)
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
    C4 (bottom-right): Key metrics summary table

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

    fig = plt.figure(figsize=(9.8, 6.9))
    gs_c = fig.add_gridspec(2, 2, wspace=0.45, hspace=0.50)
    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.11, right=0.95)
    # Note: Figure-level title removed per revision requirements; panel labels added below

    # ── C1: Loss ──
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c1.plot(epochs, h["train_loss"], label="Train MSE", color=COLORS["real"])
    ax_c1.plot(epochs, h["val_loss"], label="Val MSE", color=COLORS["generated"], linestyle="--")
    ax_c1.set_xlabel("Epoch", fontsize=10)
    ax_c1.set_ylabel("Flow-Matching Loss", fontsize=10)
    ax_c1.set_title("Loss Convergence", fontsize=11)
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
    add_panel_label(ax_c1, 'e', x=0.02, y=0.98)

    # ── C2: Cosine similarity ──
    ax_c2 = fig.add_subplot(gs_c[0, 1])
    ax_c2.plot(epochs, h["val_cosine_sim"], color=COLORS["baseline_gauss"], linewidth=2)
    ax_c2.set_xlabel("Epoch", fontsize=10)
    ax_c2.set_ylabel("Cosine Similarity", fontsize=10)
    ax_c2.set_title("Fidelity (Cosine)", fontsize=11)
    ax_c2.set_ylim(0.6, 1.0)
    ax_c2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
    ax_c2.set_xlim(0, max(epochs) * 1.05)
    ax_c2.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="both"))
    ax_c2.locator_params(axis='y', nbins=4)
    add_panel_label(ax_c2, 'f', x=0.02, y=0.98)

    # ── C3: Learning rate ──
    ax_c3 = fig.add_subplot(gs_c[1, 0])
    ax_c3.plot(epochs, h["lr"], color=COLORS["baseline_shuffle"], linewidth=1.5)
    ax_c3.set_xlabel("Epoch", fontsize=10)
    ax_c3.set_ylabel("Learning Rate", fontsize=10)
    ax_c3.set_title("LR Schedule", fontsize=11)
    set_scientific_tickformat(ax_c3, axis="y", scilimits=(-4, -4))
    ax_c3.set_xlim(0, max(epochs) * 1.05)
    ax_c3.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="both"))
    add_panel_label(ax_c3, 'g', x=0.02, y=0.98)

    # ── C4: Key metrics summary (text) ──
    ax_c4 = fig.add_subplot(gs_c[1, 1])
    ax_c4.axis("off")
    val_cos_f = h["val_cosine_sim"][-1]
    lr_f = h["lr"][-1]
    summary = (
        f"Val cosine: {val_cos_f:.4f}\n"
        f"Final LR: {lr_f:.2e}\n"
        "EMA decay: 0.9999\n"
        "10-step Euler / Midpoint"
    )
    ax_c4.text(
        0.5, 0.5, summary,
        ha="center", va="center", fontsize=10,
        transform=ax_c4.transAxes, family="sans-serif",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=COLORS["bg_light"], edgecolor=COLORS["neutral"]),
    )
    add_panel_label(ax_c4, 'h', x=0.02, y=0.98)

    # ── Save ──
    if save:
        if save_panel_fn:
            save_panel_fn(fig, "panel_c_dit_training", output_dir, dpi)
        else:
            save_with_vcd(fig, Path(output_dir) / "panel_c_dit_training.png", dpi)
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
    Bottom row (3 panels): DiT — C1 Loss, C2 Cosine Similarity, C3 LR Schedule

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
    fig = plt.figure(figsize=(14.4, 8.6))
    gs = fig.add_gridspec(2, 4, wspace=0.55, hspace=0.52,
                          width_ratios=[1.0, 1.0, 1.0, 1.2], height_ratios=[1, 1])
    # Note: Figure-level title removed per revision requirements; panel labels added below
    fig._clop_layout_rect = (0.02, 0.03, 0.98, 0.92)

    # ════════════════════════════════════════════════════════════
    # Top row: CLOP (4 panels spanning columns 0-3)
    # ════════════════════════════════════════════════════════════
    if clop_hist:
        h = clop_hist
        epochs = np.arange(1, len(h["train_loss"]) + 1)

        # A1: Loss
        ax_a1 = fig.add_subplot(gs[0, 0])
        ax_a1.plot(epochs, h["train_loss"], label="Train", color=COLORS["real"])
        ax_a1.plot(epochs, h["val_loss"], label="Val", color=COLORS["generated"], linestyle="--")
        ax_a1.set_xlabel("Epoch", fontsize=10)
        ax_a1.set_ylabel("Contrastive Loss", fontsize=10)
        ax_a1.set_title("CLOP Loss", fontsize=11)
        ax_a1.legend(loc="upper right", fontsize=FONT_LEGEND_DENSE, frameon=False)
        ax_a1.set_xlim(0, max(epochs) * 1.08)
        ax_a1.locator_params(axis='x', nbins=3)
        ax_a1.locator_params(axis='y', nbins=4)
        _add_training_phase_bands(ax_a1, int(max(epochs)))
        add_panel_label(ax_a1, 'a', x=0.02, y=0.98)

        # A2: Temperature
        ax_a2 = fig.add_subplot(gs[0, 1])
        ax_a2.plot(epochs, h["temperature"], color=COLORS["baseline_gauss"], linewidth=2)
        ax_a2.set_xlabel("Epoch", fontsize=10)
        ax_a2.set_ylabel("Temperature (\u03c4)", fontsize=10)
        ax_a2.set_title("Temperature Stability", fontsize=11)
        ax_a2.axhline(y=14.0, color="gray", linestyle=":", alpha=0.5, label="\u03c4=14.0")
        ax_a2.set_ylim(13.5, 14.5)
        ax_a2.legend(loc="lower right", fontsize=FONT_LEGEND_DENSE, frameon=False)
        ax_a2.set_xlim(0, max(epochs) * 1.08)
        ax_a2.locator_params(axis='x', nbins=4)
        ax_a2.locator_params(axis='y', nbins=4)
        add_panel_label(ax_a2, 'b', x=0.02, y=0.98)

        # A3: Accuracy
        ax_a3 = fig.add_subplot(gs[0, 2])
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
        ax_a3.set_xlabel("Epoch", fontsize=10)
        ax_a3.set_ylabel("Accuracy (%)", fontsize=10)
        ax_a3.set_title("Classification Accuracy", fontsize=11)
        ax_a3.set_ylim(0, 105)
        ax_a3.legend(loc="lower right", fontsize=FONT_LEGEND_DENSE, frameon=False, ncol=2)
        ax_a3.set_xlim(0, max(epochs) * 1.08)
        ax_a3.locator_params(axis='x', nbins=4)
        ax_a3.locator_params(axis='y', nbins=4)
        add_panel_label(ax_a3, 'c', x=0.02, y=0.98)

        # A4: Embedding quality
        ax_a4 = fig.add_subplot(gs[0, 3])
        quality_metrics = [
            ("val_text_cell_align", "Text\u2194Cell", COLORS["accent"]),
            ("val_inter_sep", "Inter-sep", COLORS["warn"]),
            ("val_mean_cosine_sim", "Mean Cos", COLORS["real"]),
        ]
        for key, label, color in quality_metrics:
            if key in h:
                ax_a4.plot(epochs, h[key], label=label, color=color)
        ax_a4.set_xlabel("Epoch", fontsize=10)
        ax_a4.set_ylabel("Score", fontsize=10)
        ax_a4.set_title("Embedding Quality", fontsize=11)
        ax_a4.set_ylim(0, 1.05)
        ax_a4.legend(loc="center right", fontsize=FONT_LEGEND_DENSE, frameon=False)
        ax_a4.set_xlim(0, max(epochs) * 1.08)
        ax_a4.locator_params(axis='x', nbins=4)
        ax_a4.locator_params(axis='y', nbins=4)
        add_panel_label(ax_a4, 'd', x=0.02, y=0.98)

    # ════════════════════════════════════════════════════════════
    # Bottom row: DiT (3 panels spanning columns 0-2, col 3 empty)
    # ════════════════════════════════════════════════════════════
    if dit_hist:
        h = dit_hist
        epochs = np.arange(1, len(h["train_loss"]) + 1)

        from matplotlib.ticker import MaxNLocator, FixedLocator
        import math

        # C1: Loss
        ax_c1 = fig.add_subplot(gs[1, 0])
        ax_c1.plot(epochs, h["train_loss"], label="Train MSE", color=COLORS["real"])
        ax_c1.plot(epochs, h["val_loss"], label="Val MSE", color=COLORS["generated"], linestyle="--")
        ax_c1.set_xlabel("Epoch", fontsize=10)
        ax_c1.set_ylabel("Flow-Matching Loss", fontsize=10)
        ax_c1.set_title("DiT Loss", fontsize=11)
        ax_c1.set_yscale("log")
        all_loss = list(h["train_loss"]) + list(h["val_loss"])
        _arr = [v for v in all_loss if v and v > 0]
        _ymin = max(min(_arr) * 0.6, 1e-5)
        _ymax = max(_arr) * 2.5
        ax_c1.set_ylim(_ymin, _ymax)
        ax_c1.legend(fontsize=FONT_LEGEND_DENSE, loc="upper right", frameon=False)
        ax_c1.set_xlim(0, max(epochs) * 1.02)
        ax_c1.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="both"))
        _lo_exp = math.ceil(math.log10(_ymin * 1.01))
        _hi_exp = math.floor(math.log10(_ymax * 0.99))
        _decade_ticks = [10**e for e in range(_lo_exp, _hi_exp + 1)]
        if _decade_ticks:
            ax_c1.yaxis.set_major_locator(FixedLocator(_decade_ticks))
        _add_training_phase_bands(ax_c1, int(max(epochs)))
        add_panel_label(ax_c1, 'e', x=0.02, y=0.98)

        # C2: Cosine similarity
        ax_c2 = fig.add_subplot(gs[1, 1])
        ax_c2.plot(epochs, h["val_cosine_sim"], color=COLORS["baseline_gauss"], linewidth=2)
        ax_c2.set_xlabel("Epoch", fontsize=10)
        ax_c2.set_ylabel("Cosine Similarity", fontsize=10)
        ax_c2.set_title("Fidelity (Cosine)", fontsize=11)
        ax_c2.set_ylim(0.6, 1.0)
        ax_c2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
        ax_c2.set_xlim(0, max(epochs) * 1.05)
        ax_c2.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="both"))
        ax_c2.locator_params(axis='y', nbins=4)
        add_panel_label(ax_c2, 'f', x=0.02, y=0.98)

        # C3: Learning rate
        ax_c3 = fig.add_subplot(gs[1, 2])
        ax_c3.plot(epochs, h["lr"], color=COLORS["baseline_shuffle"], linewidth=1.5)
        ax_c3.set_xlabel("Epoch", fontsize=10)
        ax_c3.set_ylabel("Learning Rate", fontsize=10)
        ax_c3.set_title("LR Schedule", fontsize=11)
        set_scientific_tickformat(ax_c3, axis="y", scilimits=(-4, -4))
        ax_c3.set_xlim(0, max(epochs) * 1.05)
        ax_c3.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="both"))
        add_panel_label(ax_c3, 'g', x=0.02, y=0.98)

    if save:
        if save_panel_fn:
            save_panel_fn(fig, "fig_training_dynamics", output_dir, dpi)
        else:
            save_with_vcd(fig, Path(output_dir) / "fig_training_dynamics.png", dpi)
    return fig
