"""Training panels (A, C) — CLOP and DiT training dynamics."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, save_panel, style_axes

logger = logging.getLogger(__name__)


def plot_clop_training(
    hist: Dict,
    output_dir,
    dpi: int = 300,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Panel A: 4-subplot CLOP training dynamics."""
    if not hist:
        logger.warning("No CLOP history — skipping Panel A")
        return None

    h = hist
    epochs = np.arange(1, len(h["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("CLOP Contrastive Pre-training (v9.3)", fontsize=14, fontweight="bold")

    # A1: Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, h["train_loss"], label="Train", color="#2196F3")
    ax.plot(epochs, h["val_loss"], label="Val", color="#FF5722", linestyle="--")
    style_axes(ax, "scatter", xlabel="Epoch", ylabel="Contrastive Loss",
               title="A1: Loss Convergence")
    ax.legend()
    ax.annotate(f'{h["train_loss"][-1]:.4f}', xy=(epochs[-1], h["train_loss"][-1]),
                fontsize=8, color="#2196F3", ha="right")
    ax.annotate(f'{h["val_loss"][-1]:.4f}', xy=(epochs[-1], h["val_loss"][-1]),
                fontsize=8, color="#FF5722", ha="right")

    # A2: Temperature stability
    ax = axes[0, 1]
    ax.plot(epochs, h["temperature"], color="#4CAF50", linewidth=2)
    style_axes(ax, "scatter", xlabel="Epoch", ylabel="Temperature (τ)",
               title="A2: Temperature Stability")
    ax.axhline(y=14.0, color="gray", linestyle=":", alpha=0.5, label="τ=14.0 (fixed)")
    ax.set_ylim(13.5, 14.5)
    ax.legend()

    # A3: Prototype accuracy
    ax = axes[1, 0]
    ax.plot(epochs, np.array(h["val_proto_acc"]) * 100, label="Val Proto Acc",
            color="#9C27B0", linewidth=2)
    ax.plot(epochs, np.array(h["train_proto_acc"]) * 100, label="Train Proto Acc",
            color="#9C27B0", linestyle=":", alpha=0.6)
    if "val_proto_top5" in h:
        ax.plot(epochs, np.array(h["val_proto_top5"]) * 100, label="Val Top-5",
                color="#00BCD4", linestyle="--")
    if "val_proto_top10" in h:
        ax.plot(epochs, np.array(h["val_proto_top10"]) * 100, label="Val Top-10",
                color="#8BC34A", linestyle="--")
    style_axes(ax, "scatter", xlabel="Epoch", ylabel="Accuracy (%)",
               title="A3: Cell-Type Classification Accuracy")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8)
    final_acc = h["val_proto_acc"][-1] * 100
    ax.annotate(f'{final_acc:.1f}%', xy=(epochs[-1], final_acc),
                fontsize=9, fontweight="bold", color="#9C27B0", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#9C27B0", alpha=0.8))

    # A4: Embedding quality
    ax = axes[1, 1]
    quality_metrics = [
        ("val_text_cell_align", "Text↔Cell Alignment", "#E91E63"),
        ("val_inter_sep", "Inter-type Separation", "#FF9800"),
        ("val_mean_cosine_sim", "Mean Cosine Sim", "#3F51B5"),
    ]
    for key, label, color in quality_metrics:
        if key in h:
            ax.plot(epochs, h[key], label=label, color=color)
    style_axes(ax, "scatter", xlabel="Epoch", ylabel="Score",
               title="A4: Embedding Quality Metrics")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)

    if save:
        save_panel(fig, output_dir / "panel_a_clop_training.png", dpi)
        logger.info(f"Saved Panel A → {output_dir / 'panel_a_clop_training.png'}")
    return fig


def plot_dit_training(
    hist: Dict,
    output_dir,
    dpi: int = 300,
    save: bool = True,
) -> Optional[plt.Figure]:
    """Panel C: 3-subplot DiT flow-matching training."""
    if not hist:
        logger.warning("No DiT history — skipping Panel C")
        return None

    h = hist
    epochs = np.arange(1, len(h["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("DiT Flow-Matching Training", fontsize=14, fontweight="bold")

    # C1: Loss
    ax = axes[0]
    ax.plot(epochs, h["train_loss"], label="Train MSE", color="#2196F3")
    ax.plot(epochs, h["val_loss"], label="Val MSE", color="#FF5722", linestyle="--")
    style_axes(ax, "scatter", xlabel="Epoch", ylabel="Flow-Matching Loss",
               title="C1: Loss Convergence")
    ax.set_yscale("log")
    ax.legend()
    ax.annotate(f'{h["train_loss"][-1]:.4f}', xy=(epochs[-1], h["train_loss"][-1]),
                fontsize=8, color="#2196F3", ha="right")
    ax.annotate(f'{h["val_loss"][-1]:.4f}', xy=(epochs[-1], h["val_loss"][-1]),
                fontsize=8, color="#FF5722", ha="right")

    # C2: Cosine similarity
    ax = axes[1]
    ax.plot(epochs, h["val_cosine_sim"], color="#4CAF50", linewidth=2)
    style_axes(ax, "scatter", xlabel="Epoch", ylabel="Cosine Similarity",
               title="C2: Generation Fidelity (Cosine)")
    ax.set_ylim(0.6, 1.0)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
    final_cos = h["val_cosine_sim"][-1]
    ax.annotate(f'{final_cos:.4f}', xy=(epochs[-1], final_cos),
                fontsize=9, fontweight="bold", color="#4CAF50", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#4CAF50", alpha=0.8))

    # C3: Learning rate
    ax = axes[2]
    ax.plot(epochs, h["lr"], color="#9C27B0", linewidth=1.5)
    style_axes(ax, "scatter", xlabel="Epoch", ylabel="Learning Rate",
               title="C3: LR Schedule (Cosine Decay)")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-4, -4))

    if save:
        save_panel(fig, output_dir / "panel_c_dit_training.png", dpi)
        logger.info(f"Saved Panel C → {output_dir / 'panel_c_dit_training.png'}")
    return fig
