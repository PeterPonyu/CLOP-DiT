#!/usr/bin/env python3
"""Generate evaluation pipeline schematic figure for the paper.

Produces a flowchart-style diagram showing the exact evaluation pipeline:
data splits, PCA fitting, KNN training, and metric computation stages.

Usage:
    python scripts/analysis/evaluation_pipeline_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.visualization.style import apply_style

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _rounded_box(ax, xy, w, h, text, fc="#E8F4FD", ec="#2C3E50", fontsize=11,
                 lw=1.2, text_color="#2C3E50", bold=False):
    """Draw a rounded rectangle with centered text."""
    box = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.08", facecolor=fc,
        edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight=weight, zorder=3,
            wrap=True)
    return box


def _arrow(ax, start, end, color="#555", lw=1.2, style="->"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=1)


def make_figure():
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(11, 7.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 7.5)
    ax.axis("off")

    # Title
    ax.text(5.0, 7.1, "Evaluation Pipeline Schematic", fontsize=15,
            ha="center", va="center", weight="bold", color="#2C3E50")

    # ── Row 1: Data Sources ──
    y1 = 6.0
    _rounded_box(ax, (0.0, y1), 2.4, 0.7,
                 "80 GEO Datasets\n(220,304 cells, 1,088 groups)",
                 fc="#D5E8D4", bold=True)
    _arrow(ax, (2.4, y1 + 0.35), (3.0, y1 + 0.35))

    _rounded_box(ax, (3.0, y1), 2.0, 0.7,
                 "Deduplication\n1,088 → 69 types",
                 fc="#FFF2CC")
    _arrow(ax, (5.0, y1 + 0.35), (5.6, y1 + 0.35))

    _rounded_box(ax, (5.6, y1), 2.2, 0.7,
                 "Cell-type Stratified Split\n72 train / 8 held-out",
                 fc="#FFF2CC")
    _arrow(ax, (7.8, y1 + 0.35), (8.3, y1 + 0.35))

    _rounded_box(ax, (8.3, y1), 2.0, 0.7,
                 "scGPT Encoding\n512-d embeddings",
                 fc="#DAE8FC")

    # ── Row 2: Real data splits ──
    y2 = 4.7
    _arrow(ax, (9.3, y1), (9.3, y2 + 0.7))

    _rounded_box(ax, (0.3, y2), 2.6, 0.7,
                 "Real Embeddings (80% train)\nPCA-50 fit on this split only",
                 fc="#D5E8D4", bold=False)

    _rounded_box(ax, (3.5, y2), 2.4, 0.7,
                 "Real Embeddings (20% test)\nPCA transform applied",
                 fc="#D5E8D4")

    _rounded_box(ax, (6.5, y2), 3.5, 0.7,
                 "DiT Generation\n69 types × 100 cells = 6,900\nCFG sweep: 0.5–5.0",
                 fc="#E1D5E7", bold=True)

    _arrow(ax, (9.3, y2 + 0.7), (8.25, y2 + 0.7))

    # Arrows from real train to PCA/KNN
    _arrow(ax, (1.6, y2), (1.6, y2 - 0.4))
    _arrow(ax, (4.7, y2), (4.7, y2 - 0.4))
    _arrow(ax, (8.25, y2), (8.25, y2 - 0.4))

    # ── Row 3: PCA + KNN ──
    y3 = 3.2
    _rounded_box(ax, (0.0, y3), 3.2, 0.7,
                 "KNN Classifier (k=15, cosine)\nTrained on 80% real PCA-50",
                 fc="#DAE8FC", bold=True)

    _rounded_box(ax, (3.8, y3), 2.8, 0.7,
                 "Same PCA-50 transform\napplied to generated + test",
                 fc="#FFF2CC")

    _rounded_box(ax, (7.2, y3), 3.0, 0.7,
                 "Per-type statistics\n(centroids, covariances)\nfrom real data",
                 fc="#DAE8FC")

    _arrow(ax, (3.2, y3 + 0.35), (3.8, y3 + 0.35))
    _arrow(ax, (6.6, y3 + 0.35), (7.2, y3 + 0.35))

    # ── Row 4: Metrics ──
    y4 = 1.7
    _arrow(ax, (1.6, y3), (1.6, y4 + 0.7))
    _arrow(ax, (5.2, y3), (5.2, y4 + 0.7))
    _arrow(ax, (8.7, y3), (8.7, y4 + 0.7))

    _rounded_box(ax, (0.0, y4), 2.0, 0.7,
                 "KNN Accuracy\n(top-1, top-5)",
                 fc="#F8CECC", bold=True)

    _rounded_box(ax, (2.3, y4), 1.8, 0.7,
                 "Steering\nAccuracy",
                 fc="#F8CECC", bold=True)

    _rounded_box(ax, (4.4, y4), 1.8, 0.7,
                 "Diversity\nRatio",
                 fc="#F8CECC", bold=True)

    _rounded_box(ax, (6.5, y4), 2.0, 0.7,
                 "FD, Coverage\nDensity, Cosine",
                 fc="#F8CECC", bold=True)

    _rounded_box(ax, (8.8, y4), 1.5, 0.7,
                 "Linear\nAccuracy",
                 fc="#F8CECC", bold=True)

    # ── Row 5: Composites ──
    y5 = 0.4
    _arrow(ax, (5.0, y4), (5.0, y5 + 0.7))

    _rounded_box(ax, (1.5, y5), 3.2, 0.7,
                 "Common-Metrics Composite (9)\nPRIMARY BENCHMARK",
                 fc="#E74C3C", ec="#C0392B", text_color="white", bold=True)

    _rounded_box(ax, (5.5, y5), 3.2, 0.7,
                 "Full Composite (17)\n+ 8 downstream biology metrics\n(structurally biased)",
                 fc="#F5B7B1", ec="#C0392B")

    _arrow(ax, (4.7, y5 + 0.35), (5.5, y5 + 0.35))

    # ── Row 6: Bootstrap ──
    ax.text(9.5, y5 + 0.35, "Bootstrap\n95% CI\n(B=1000)",
            ha="center", va="center", fontsize=11, style="italic",
            color="#7F8C8D",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F9F9F9", ec="#BDC3C7", lw=0.8))

    _arrow(ax, (8.7, y5 + 0.35), (9.0, y5 + 0.35), style="->", color="#BDC3C7")

    # Legend
    legend_items = [
        ("#D5E8D4", "Real Data"),
        ("#E1D5E7", "Generated Data"),
        ("#DAE8FC", "Fitted Models"),
        ("#FFF2CC", "Transforms"),
        ("#F8CECC", "Metrics"),
    ]
    for i, (color, label) in enumerate(legend_items):
        x = 0.0 + i * 2.1
        p = mpatches.FancyBboxPatch((x, -0.35), 0.3, 0.2,
                                     boxstyle="round,pad=0.03",
                                     facecolor=color, edgecolor="#666", lw=0.5)
        ax.add_patch(p)
        ax.text(x + 0.4, -0.25, label, fontsize=11, va="center", color="#333")

    out_png = FIG_DIR / "fig_evaluation_pipeline.png"
    from src.visualization.style import save_with_vcd
    save_with_vcd(fig, out_png, dpi=300)
    plt.close(fig)
    print(f"Saved evaluation pipeline figure to {out_png}")


if __name__ == "__main__":
    make_figure()
