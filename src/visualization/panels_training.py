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

from .style import COLORS, apply_style, save_with_vcd, set_scientific_tickformat

logger = logging.getLogger(__name__)


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
    fig.suptitle("CLOP Contrastive Pre-training (v9.3)", fontsize=11, y=0.99)

    # ── A1: Loss curves ──
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, h["train_loss"], label="Train", color=COLORS["real"])
    ax.plot(epochs, h["val_loss"], label="Val", color=COLORS["generated"], linestyle="--")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Contrastive Loss", fontsize=10)
    ax.set_title("Loss Convergence", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_xlim(0, max(epochs) * 1.08)
    ax.locator_params(axis='x', nbins=3)
    ax.locator_params(axis='y', nbins=4)

    # ── A2: Temperature stability ──
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, h["temperature"], color=COLORS["baseline_gauss"], linewidth=2)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Temperature (\u03c4)", fontsize=10)
    ax.set_title("Temperature Stability", fontsize=11)
    ax.axhline(y=14.0, color="gray", linestyle=":", alpha=0.5, label="\u03c4=14.0")
    ax.set_ylim(13.5, 14.5)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_xlim(0, max(epochs) * 1.08)
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=4)

    # ── A3: Prototype accuracy ──
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(
        epochs, np.array(h["val_proto_acc"]) * 100,
        label="Val Acc", color=COLORS["baseline_shuffle"], linewidth=2,
    )
    ax.plot(
        epochs, np.array(h["train_proto_acc"]) * 100,
        label="Train Acc", color=COLORS["baseline_shuffle"], linestyle=":", alpha=0.6,
    )
    if "val_proto_top5" in h:
        ax.plot(
            epochs, np.array(h["val_proto_top5"]) * 100,
            label="Top-5", color=COLORS["neutral"], linestyle="--",
        )
    if "val_proto_top10" in h:
        ax.plot(
            epochs, np.array(h["val_proto_top10"]) * 100,
            label="Top-10", color=COLORS["baseline_gauss"], linestyle="--",
        )
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("Classification Accuracy", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)
    ax.set_xlim(0, max(epochs) * 1.08)
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=4)

    # ── A4: Embedding quality metrics ──
    ax = fig.add_subplot(gs[1, 1])
    quality_metrics = [
        ("val_text_cell_align", "Text\u2194Cell", COLORS["accent"]),
        ("val_inter_sep", "Inter-sep", COLORS["warn"]),
        ("val_mean_cosine_sim", "Mean Cos", COLORS["real"]),
    ]
    for key, label, color in quality_metrics:
        if key in h:
            ax.plot(epochs, h[key], label=label, color=color)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Embedding Quality", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center right", fontsize=8, frameon=False)
    ax.set_xlim(0, max(epochs) * 1.08)
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=4)

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
    fig.suptitle("DiT Flow-Matching Training", fontsize=11, y=0.98)

    # ── C1: Loss ──
    ax = fig.add_subplot(gs_c[0, 0])
    ax.plot(epochs, h["train_loss"], label="Train MSE", color=COLORS["real"])
    ax.plot(epochs, h["val_loss"], label="Val MSE", color=COLORS["generated"], linestyle="--")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Flow-Matching Loss", fontsize=10)
    ax.set_title("Loss Convergence", fontsize=11)
    ax.set_yscale("log")
    all_loss = list(h["train_loss"]) + list(h["val_loss"])
    _arr = [v for v in all_loss if v and v > 0]
    _ymin = max(min(_arr) * 0.6, 1e-5)
    _ymax = max(_arr) * 2.5
    ax.set_ylim(_ymin, _ymax)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.set_xlim(0, max(epochs) * 1.02)
    from matplotlib.ticker import MaxNLocator, FixedLocator
    import math
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="both"))
    # Explicit ticks: only powers of 10 strictly within the ylim (no overflow)
    _lo_exp = math.ceil(math.log10(_ymin * 1.01))
    _hi_exp = math.floor(math.log10(_ymax * 0.99))
    _decade_ticks = [10**e for e in range(_lo_exp, _hi_exp + 1)]
    if _decade_ticks:
        ax.yaxis.set_major_locator(FixedLocator(_decade_ticks))

    # ── C2: Cosine similarity ──
    ax = fig.add_subplot(gs_c[0, 1])
    ax.plot(epochs, h["val_cosine_sim"], color=COLORS["baseline_gauss"], linewidth=2)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Cosine Similarity", fontsize=10)
    ax.set_title("Fidelity (Cosine)", fontsize=11)
    ax.set_ylim(0.6, 1.0)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlim(0, max(epochs) * 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="both"))
    ax.locator_params(axis='y', nbins=4)

    # ── C3: Learning rate ──
    ax = fig.add_subplot(gs_c[1, 0])
    ax.plot(epochs, h["lr"], color=COLORS["baseline_shuffle"], linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Learning Rate", fontsize=10)
    ax.set_title("LR Schedule", fontsize=11)
    set_scientific_tickformat(ax, axis="y", scilimits=(-4, -4))
    ax.set_xlim(0, max(epochs) * 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="both"))

    # ── C4: Key metrics summary (text) ──
    ax = fig.add_subplot(gs_c[1, 1])
    ax.axis("off")
    val_cos_f = h["val_cosine_sim"][-1]
    lr_f = h["lr"][-1]
    summary = (
        f"Val cosine: {val_cos_f:.4f}\n"
        f"Final LR: {lr_f:.2e}\n"
        "EMA decay: 0.9999\n"
        "10-step Euler / Midpoint"
    )
    ax.text(
        0.5, 0.5, summary,
        ha="center", va="center", fontsize=10,
        transform=ax.transAxes, family="sans-serif",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=COLORS["bg_light"], edgecolor=COLORS["neutral"]),
    )

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
    fig = plt.figure(figsize=(14.4, 8.2))
    gs = fig.add_gridspec(2, 4, wspace=0.55, hspace=0.52,
                          width_ratios=[1.0, 1.0, 1.0, 1.2], height_ratios=[1, 1])
    fig.suptitle("Training Dynamics", fontsize=12, y=0.98)

    # ════════════════════════════════════════════════════════════
    # Top row: CLOP (4 panels spanning columns 0-3)
    # ════════════════════════════════════════════════════════════
    if clop_hist:
        h = clop_hist
        epochs = np.arange(1, len(h["train_loss"]) + 1)

        # A1: Loss
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(epochs, h["train_loss"], label="Train", color=COLORS["real"])
        ax.plot(epochs, h["val_loss"], label="Val", color=COLORS["generated"], linestyle="--")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Contrastive Loss", fontsize=10)
        ax.set_title("CLOP Loss", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.set_xlim(0, max(epochs) * 1.08)
        ax.locator_params(axis='x', nbins=3)
        ax.locator_params(axis='y', nbins=4)

        # A2: Temperature
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(epochs, h["temperature"], color=COLORS["baseline_gauss"], linewidth=2)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Temperature (\u03c4)", fontsize=10)
        ax.set_title("Temperature Stability", fontsize=11)
        ax.axhline(y=14.0, color="gray", linestyle=":", alpha=0.5, label="\u03c4=14.0")
        ax.set_ylim(13.5, 14.5)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.set_xlim(0, max(epochs) * 1.08)
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='y', nbins=4)

        # A3: Accuracy
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(epochs, np.array(h["val_proto_acc"]) * 100,
                label="Val Acc", color=COLORS["baseline_shuffle"], linewidth=2)
        ax.plot(epochs, np.array(h["train_proto_acc"]) * 100,
                label="Train Acc", color=COLORS["baseline_shuffle"], linestyle=":", alpha=0.6)
        if "val_proto_top5" in h:
            ax.plot(epochs, np.array(h["val_proto_top5"]) * 100,
                    label="Top-5", color=COLORS["neutral"], linestyle="--")
        if "val_proto_top10" in h:
            ax.plot(epochs, np.array(h["val_proto_top10"]) * 100,
                    label="Top-10", color=COLORS["baseline_gauss"], linestyle="--")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_title("Classification Accuracy", fontsize=11)
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)
        ax.set_xlim(0, max(epochs) * 1.08)
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='y', nbins=4)

        # A4: Embedding quality
        ax = fig.add_subplot(gs[0, 3])
        quality_metrics = [
            ("val_text_cell_align", "Text\u2194Cell", COLORS["accent"]),
            ("val_inter_sep", "Inter-sep", COLORS["warn"]),
            ("val_mean_cosine_sim", "Mean Cos", COLORS["real"]),
        ]
        for key, label, color in quality_metrics:
            if key in h:
                ax.plot(epochs, h[key], label=label, color=color)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title("Embedding Quality", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="center right", fontsize=8, frameon=False)
        ax.set_xlim(0, max(epochs) * 1.08)
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='y', nbins=4)

    # ════════════════════════════════════════════════════════════
    # Bottom row: DiT (3 panels spanning columns 0-2, col 3 empty)
    # ════════════════════════════════════════════════════════════
    if dit_hist:
        h = dit_hist
        epochs = np.arange(1, len(h["train_loss"]) + 1)

        from matplotlib.ticker import MaxNLocator, FixedLocator
        import math

        # C1: Loss
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(epochs, h["train_loss"], label="Train MSE", color=COLORS["real"])
        ax.plot(epochs, h["val_loss"], label="Val MSE", color=COLORS["generated"], linestyle="--")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Flow-Matching Loss", fontsize=10)
        ax.set_title("DiT Loss", fontsize=11)
        ax.set_yscale("log")
        all_loss = list(h["train_loss"]) + list(h["val_loss"])
        _arr = [v for v in all_loss if v and v > 0]
        _ymin = max(min(_arr) * 0.6, 1e-5)
        _ymax = max(_arr) * 2.5
        ax.set_ylim(_ymin, _ymax)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_xlim(0, max(epochs) * 1.02)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="both"))
        _lo_exp = math.ceil(math.log10(_ymin * 1.01))
        _hi_exp = math.floor(math.log10(_ymax * 0.99))
        _decade_ticks = [10**e for e in range(_lo_exp, _hi_exp + 1)]
        if _decade_ticks:
            ax.yaxis.set_major_locator(FixedLocator(_decade_ticks))

        # C2: Cosine similarity
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(epochs, h["val_cosine_sim"], color=COLORS["baseline_gauss"], linewidth=2)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Cosine Similarity", fontsize=10)
        ax.set_title("Fidelity (Cosine)", fontsize=11)
        ax.set_ylim(0.6, 1.0)
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
        ax.set_xlim(0, max(epochs) * 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="both"))
        ax.locator_params(axis='y', nbins=4)

        # C3: Learning rate
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(epochs, h["lr"], color=COLORS["baseline_shuffle"], linewidth=1.5)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Learning Rate", fontsize=10)
        ax.set_title("LR Schedule", fontsize=11)
        set_scientific_tickformat(ax, axis="y", scilimits=(-4, -4))
        ax.set_xlim(0, max(epochs) * 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, prune="both"))

    if save:
        if save_panel_fn:
            save_panel_fn(fig, "fig_training_dynamics", output_dir, dpi)
        else:
            save_with_vcd(fig, Path(output_dir) / "fig_training_dynamics.png", dpi)
    return fig
