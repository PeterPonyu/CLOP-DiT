"""
fig31_decoder_ablation.py — Fig 31: Decoder ablation comparison.

Compares three decoder approaches (scGPT baseline, LoRA fine-tuned, MLP)
on the same generated embeddings, showing that expression-level limitations
are due to the embedding representation rather than the decoder architecture.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from .style import (
    COLORS,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TICK,
    FONT_TITLE,
    FONT_ANNOTATION,
    add_panel_label,
    apply_style,
    save_panel,
    style_axes,
)
from src.utils.paths import RESULTS_DIR, FIG_DIR

logger = logging.getLogger(__name__)

APPROACH_LABELS = {
    "baseline": "scGPT\n(frozen)",
    "lora_light": "scGPT\n+LoRA",
    "mlp": "MLP\n(trained)",
}

APPROACH_COLORS = {
    "baseline": COLORS.get("real", "#2196F3"),
    "lora_light": COLORS.get("generated", "#4CAF50"),
    "mlp": COLORS.get("baseline_3", "#FF9800"),
}


def plot_decoder_ablation(
    ablation_dir: str | Path = "results/ablations/decoder",
    output_dir: str | Path = "results/figures",
    dpi: int = 300,
    save: bool = True,
    save_panel_fn: Optional[Callable] = None,
) -> Optional[plt.Figure]:
    """Fig 31: Decoder ablation comparison.

    Panel (a): Per-gene std comparison (real vs generated per approach)
    Panel (b): Between-type variance ratio
    Panel (c): Marker specificity per panel
    Panel (d): Decode speed comparison
    """
    apply_style()
    ablation_dir = Path(ablation_dir)
    output_dir = Path(output_dir)

    approaches = []
    metrics = {}
    for name in ["baseline", "lora_light", "mlp"]:
        mpath = ablation_dir / name / "metrics.json"
        if not mpath.exists():
            logger.warning("Missing ablation: %s", mpath)
            continue
        with open(mpath) as f:
            metrics[name] = json.load(f)
        approaches.append(name)

    if len(approaches) < 2:
        logger.warning("Need at least 2 ablation results, found %d", len(approaches))
        return None

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    labels = [APPROACH_LABELS.get(a, a) for a in approaches]
    colors = [APPROACH_COLORS.get(a, "#999") for a in approaches]

    # ── Panel (a): Per-gene std ──
    ax = axes[0, 0]
    x = np.arange(len(approaches))
    w = 0.35
    real_vals = [metrics[a]["real_per_gene_std"] for a in approaches]
    gen_vals = [metrics[a]["gen_per_gene_std"] for a in approaches]
    ax.bar(x - w / 2, real_vals, w, color="#90CAF9", edgecolor="none", label="Real")
    ax.bar(x + w / 2, gen_vals, w, color=colors, edgecolor="none", label="Generated")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylabel("Per-gene std", fontsize=FONT_LABEL)
    ax.set_title("Expression variance\nacross genes", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND, frameon=False)
    style_axes(ax)
    add_panel_label(ax, "a")

    # ── Panel (b): Between-type variance ratio ──
    ax = axes[0, 1]
    ratios = [metrics[a]["between_type_var_ratio"] for a in approaches]
    bars = ax.bar(x, ratios, 0.6, color=colors, edgecolor="none")
    ax.axhline(1.0, color="#999", ls="--", lw=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylabel("Variance ratio\n(gen / real)", fontsize=FONT_LABEL)
    ax.set_title("Between-type\nvariance ratio", fontsize=FONT_TITLE)
    for b, v in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                f"{v:.2f}", ha="center", fontsize=FONT_ANNOTATION)
    style_axes(ax)
    add_panel_label(ax, "b")

    # ── Panel (c): Marker specificity per panel ──
    ax = axes[1, 0]
    panels_to_show = ["T_cell", "Macrophage", "NK_cell", "B_cell",
                      "Fibroblast", "Epithelial"]
    panel_labels = [p.replace("_", "\n") for p in panels_to_show]
    n_panels = len(panels_to_show)
    n_apps = len(approaches)
    bar_w = 0.8 / n_apps
    for i, a in enumerate(approaches):
        mp = metrics[a].get("marker_panels", {})
        vals = [mp.get(p, {}).get("best_specificity_ratio", 1.0) for p in panels_to_show]
        offset = (i - n_apps / 2 + 0.5) * bar_w
        ax.bar(np.arange(n_panels) + offset, vals, bar_w,
               color=APPROACH_COLORS.get(a, "#999"), edgecolor="none",
               label=APPROACH_LABELS.get(a, a).replace("\n", " "))
    ax.axhline(1.0, color="#999", ls="--", lw=0.8, zorder=0)
    ax.set_xticks(np.arange(n_panels))
    ax.set_xticklabels(panel_labels, fontsize=FONT_TICK - 1)
    ax.set_ylabel("Specificity ratio", fontsize=FONT_LABEL)
    ax.set_title("Marker gene specificity\nper cell type", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND - 1, frameon=False, ncol=1, loc="upper right")
    style_axes(ax)
    add_panel_label(ax, "c")

    # ── Panel (d): Decode speed ──
    ax = axes[1, 1]
    times = [metrics[a]["decode_time_s"] for a in approaches]
    bars = ax.barh(x, times, 0.6, color=colors, edgecolor="none")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=FONT_TICK)
    ax.set_xlabel("Decode time (seconds)", fontsize=FONT_LABEL)
    ax.set_title("Inference speed\n(2000 cells)", fontsize=FONT_TITLE)
    for b, v in zip(bars, times):
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2,
                f"{v:.1f}s", va="center", fontsize=FONT_ANNOTATION)
    style_axes(ax)
    add_panel_label(ax, "d")

    fig.tight_layout(pad=1.5)

    if save:
        out = output_dir / "fig31_decoder_ablation.pdf"
        out_png = output_dir / "fig31_decoder_ablation.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        logger.info("Saved Fig 31 → %s", out)

    if save_panel_fn:
        save_panel_fn(fig, "fig31_decoder_ablation")

    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    logging.basicConfig(level=logging.INFO)
    plot_decoder_ablation()
    plt.close("all")
