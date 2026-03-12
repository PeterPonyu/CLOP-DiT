#!/usr/bin/env python3
"""
generate_architecture_figure.py -- Publication-quality CLOP-DiT pipeline architecture figure.

Generates a three-stage pipeline diagram showing:
  Stage 1 (CLOP):  Text/Cell encoders -> projectors -> shared contrastive space
  Stage 2 (DiT):   Noise -> DiT1D with AdaLN-Zero conditioning -> ODE solver -> latent
  Stage 3 (Decode): Generated latent -> scGPT decoder -> gene expression profile

Output: results/figures/fig_architecture.png and .pdf at 300 DPI.

Usage:
    python scripts/generate_architecture_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.visualization.direct_layout import bind_figure_region
from src.visualization.style import (
    COLORS,
    FONT_ARCH_LABEL,
    FONT_ARCH_SUBLABEL,
    apply_style,
    add_panel_label,
    save_panel,
)

apply_style()
matplotlib.rcParams.update({
    "axes.grid": False,
    "figure.constrained_layout.use": False,
})

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_TEXT_DARK = COLORS["real"]
C_TEXT_MID = "#1E88E5"
C_TEXT_LIGHT = "#90CAF9"
C_TEXT_BOX = "#E3F2FD"

C_CELL_DARK = COLORS["baseline_gauss"]
C_CELL_MID = "#43A047"
C_CELL_LIGHT = "#A5D6A7"
C_CELL_BOX = "#E8F5E9"

C_SHARED = COLORS["baseline_shuffle"]
C_SHARED_BOX = "#F3E5F5"

C_GEN_DARK = COLORS["generated"]
C_GEN_MID = "#E65100"
C_GEN_LIGHT = "#FFCCBC"
C_GEN_BOX = "#FFF3E0"

C_DECODE_DARK = "#004D40"
C_DECODE_MID = "#00796B"
C_DECODE_BOX = "#E0F2F1"

C_GREY = COLORS["neutral"]
C_MID_GREY = "#607D8B"
C_WHITE = "#FFFFFF"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_box(ax, xy, w, h, label, sublabel=None, facecolor=C_WHITE,
             edgecolor=C_GREY, fontsize=10, sublabel_size=9.0,
             textcolor="black", bold=False, linewidth=1.0, zorder=3,
             boxstyle="round,pad=0.08"):
    """Draw a rounded box with centred label text."""
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=boxstyle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        mutation_scale=0.5,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    y_off = h * 0.12 if sublabel else 0
    ax.text(
        x + w / 2, y + h / 2 + y_off,
        label,
        ha="center", va="center",
        fontsize=fontsize, fontweight=weight, color=textcolor,
        zorder=zorder + 1,
    )
    if sublabel:
        ax.text(
            x + w / 2, y + h / 2 - h * 0.20,
            sublabel,
            ha="center", va="center",
            fontsize=sublabel_size, color=C_GREY,
            zorder=zorder + 1,
        )
    return box


def draw_arrow(ax, start, end, color=C_GREY, linewidth=1.2,
               style="->", connectionstyle="arc3,rad=0", zorder=2,
               shrinkA=2, shrinkB=2):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color=color,
        linewidth=linewidth,
        connectionstyle=connectionstyle,
        zorder=zorder,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        mutation_scale=10,
    )
    ax.add_patch(arrow)
    return arrow


def draw_stage_bg(ax, xy, w, h, label, color, alpha=0.10, label_color=None):
    """Draw a translucent background rectangle for a pipeline stage."""
    x, y = xy
    bg = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04",
        facecolor=color,
        edgecolor=color,
        linewidth=1.0,
        alpha=alpha,
        zorder=0,
    )
    ax.add_patch(bg)
    lc = label_color or color
    ax.text(
        x + w / 2, y + h + 0.02,
        label,
        ha="center", va="bottom",
        fontsize=10, fontweight="normal",
        color=lc,
        zorder=1,
    )


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def create_architecture_figure(output_dir=None):
    if output_dir is None:
        from src.utils.paths import FIG_DIR
        output_dir = Path(FIG_DIR)
    output_dir = Path(output_dir)
    fig = plt.figure(figsize=(10.0, 3.6))
    ax = bind_figure_region(fig, (0.01, 0.02, 0.99, 0.97)).add_axes(fig)
    ax.set_xlim(-0.20, 7.65)
    ax.set_ylim(-0.18, 3.18)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_facecolor(C_WHITE)

    BW = 0.78
    BH = 0.38
    SBW = 0.55
    SBH = 0.32
    gap = 0.12

    # Stage backgrounds
    draw_stage_bg(ax, (-0.05, -0.05), 3.65, 3.00,
                  "Stage 1: CLOP Alignment",
                  C_TEXT_DARK, alpha=0.15, label_color="black")
    draw_stage_bg(ax, (3.60, -0.05), 2.50, 3.00,
                  "Stage 2: DiT Generation",
                  C_GEN_DARK, alpha=0.15, label_color="black")
    draw_stage_bg(ax, (6.10, -0.05), 1.35, 3.00,
                  "Stage 3: Decoding",
                  C_DECODE_DARK, alpha=0.15, label_color="black")

    # Panel labels (data coordinates — track stage backgrounds regardless of bind_figure_region)
    ax.text(-0.05, 3.10, "(a)", ha="left", va="bottom", fontsize=14, fontweight="bold", color="black",
            clip_on=False, zorder=10)
    ax.text(3.60, 3.10, "(b)", ha="left", va="bottom", fontsize=14, fontweight="bold", color="black",
            clip_on=False, zorder=10)
    ax.text(6.10, 3.10, "(c)", ha="left", va="bottom", fontsize=14, fontweight="bold", color="black",
            clip_on=False, zorder=10)

    ax.text(1.68, 2.80, "train: align text and cell latents",
            ha="center", va="center", fontsize=FONT_ARCH_SUBLABEL, color="black", zorder=2)
    ax.text(4.80, 2.82, "ODE latent sampling",
            ha="center", va="center", fontsize=FONT_ARCH_SUBLABEL, color="black", zorder=2)
    ax.text(6.90, 2.70, "decode to genes",
            ha="center", va="center", fontsize=FONT_ARCH_SUBLABEL, color="black", zorder=2)

    # Stage 1: CLOP
    ty = 2.25
    tx0 = 0.02
    draw_box(ax, (tx0, ty), SBW, SBH, "Text\nDescription",
             facecolor=C_TEXT_BOX, edgecolor=C_TEXT_DARK, fontsize=FONT_ARCH_LABEL,
             textcolor="black")

    tx1 = tx0 + SBW + gap
    draw_box(ax, (tx1, ty - 0.05), BW, BH + 0.10, "BiomedBERT",
             sublabel="frozen | 340M",
             facecolor=C_TEXT_LIGHT, edgecolor=C_TEXT_DARK, fontsize=FONT_ARCH_SUBLABEL,
             textcolor="black", linewidth=1.3)
    draw_arrow(ax, (tx0 + SBW, ty + SBH / 2),
               (tx1, ty + SBH / 2 - 0.02), color=C_TEXT_MID, linewidth=1.2)

    tx2 = tx1 + BW + gap
    zca_w = 0.45
    draw_box(ax, (tx2, ty + 0.02), zca_w, SBH - 0.04, "ZCA",
             facecolor=C_TEXT_BOX, edgecolor=C_TEXT_DARK, fontsize=FONT_ARCH_LABEL,
             textcolor="black")
    draw_arrow(ax, (tx1 + BW, ty + SBH / 2 - 0.02),
               (tx2, ty + SBH / 2), color=C_TEXT_MID, linewidth=1.2)

    tx3 = tx2 + zca_w + gap
    draw_box(ax, (tx3, ty - 0.05), BW, BH + 0.10, "Text Proj.",
             sublabel="MLP 1024→512",
             facecolor=C_TEXT_LIGHT, edgecolor=C_TEXT_DARK, fontsize=FONT_ARCH_SUBLABEL,
             textcolor="black")
    draw_arrow(ax, (tx2 + zca_w, ty + SBH / 2),
               (tx3, ty + SBH / 2 - 0.02), color=C_TEXT_MID, linewidth=1.2)

    cy = 0.60
    cx0 = 0.02
    draw_box(ax, (cx0, cy), SBW, SBH, "Cell\nProfile",
             facecolor=C_CELL_BOX, edgecolor=C_CELL_DARK, fontsize=FONT_ARCH_LABEL,
             textcolor="black")

    cx1 = cx0 + SBW + gap
    draw_box(ax, (cx1, cy - 0.05), BW, BH + 0.10, "scGPT Enc.",
             sublabel="frozen | 51M",
             facecolor=C_CELL_LIGHT, edgecolor=C_CELL_DARK, fontsize=FONT_ARCH_SUBLABEL,
             textcolor="black", linewidth=1.3)
    draw_arrow(ax, (cx0 + SBW, cy + SBH / 2),
               (cx1, cy + SBH / 2 - 0.02), color=C_CELL_MID, linewidth=1.2)

    cx2 = tx3
    draw_box(ax, (cx2, cy - 0.05), BW, BH + 0.10, "Cell Proj.",
             sublabel="MLP 512→512",
             facecolor=C_CELL_LIGHT, edgecolor=C_CELL_DARK, fontsize=FONT_ARCH_SUBLABEL,
             textcolor="black")
    draw_arrow(ax, (cx1 + BW, cy + SBH / 2 - 0.02),
               (cx2, cy + SBH / 2 - 0.02), color=C_CELL_MID, linewidth=1.2)

    shared_w = 0.62
    shared_h = 0.72
    shared_x = tx3 + BW + gap + 0.02
    shared_y = 1.22
    shared_box = FancyBboxPatch(
        (shared_x, shared_y), shared_w, shared_h,
        boxstyle="round,pad=0.04",
        facecolor=C_SHARED_BOX,
        edgecolor=C_SHARED,
        linewidth=1.0,
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(shared_box)
    ax.text(shared_x + shared_w / 2, shared_y + shared_h / 2,
            "Shared\nSpace\n(512-d)",
            ha="center", va="center", fontsize=FONT_ARCH_LABEL,
            color="black", zorder=3)

    draw_arrow(ax, (tx3 + BW, ty + SBH / 2 - 0.02),
               (shared_x, shared_y + shared_h * 0.75),
               color=C_TEXT_MID, linewidth=1.2,
               connectionstyle="arc3,rad=-0.1")
    draw_arrow(ax, (cx2 + BW, cy + SBH / 2 - 0.02),
               (shared_x, shared_y + shared_h * 0.25),
               color=C_CELL_MID, linewidth=1.2,
               connectionstyle="arc3,rad=0.1")

    loss_y = 1.52
    loss_x = 0.80
    loss_w = 1.10
    draw_box(ax, (loss_x, loss_y), loss_w, SBH, "PrototypeSigLIP",
             facecolor="#FFF9C4", edgecolor="#F9A825", fontsize=FONT_ARCH_LABEL,
             textcolor="black", linewidth=0.8)
    ax.text(loss_x + loss_w / 2, loss_y + SBH + 0.08, "training only",
            ha="center", va="bottom", fontsize=FONT_ARCH_SUBLABEL,
            color="black", zorder=6)
    draw_arrow(ax, (loss_x + loss_w, loss_y + SBH / 2),
               (shared_x, shared_y + shared_h / 2),
               color="#F9A825", linewidth=0.8, style="<->")

    ax.text(cx1 + BW / 2, cy + BH + 0.16,
            "scGPT latent =\nDiT training target",
            ha="center", va="bottom", fontsize=FONT_ARCH_SUBLABEL, color=C_MID_GREY,
            fontweight="normal", zorder=5)

    # Stage 2: DiT
    dit_x0 = 3.80
    dit_y_mid = 1.40
    z0_w = 0.50
    draw_box(ax, (dit_x0, dit_y_mid), z0_w, SBH,
             r"$z_0$", sublabel="512-d",
             facecolor=C_GEN_BOX, edgecolor=C_GEN_DARK, fontsize=FONT_ARCH_SUBLABEL,
             textcolor="black")
    ax.text(dit_x0 + z0_w / 2, dit_y_mid + SBH + 0.03,
            r"$\sim\mathcal{N}(0,I)$",
            ha="center", va="bottom", fontsize=FONT_ARCH_SUBLABEL, color="black", zorder=5)

    dit_bx = dit_x0 + z0_w + 0.12
    dit_bw = 1.00
    dit_bh = 1.10
    dit_by = dit_y_mid - 0.40
    draw_box(ax, (dit_bx, dit_by), dit_bw, dit_bh, "",
             facecolor=C_GEN_LIGHT, edgecolor=C_GEN_DARK,
             linewidth=1.4, zorder=2)

    cx_dit = dit_bx + dit_bw / 2
    ax.text(cx_dit, dit_by + dit_bh - 0.14, "DiT1D",
            ha="center", va="center", fontsize=10,
            color="black", zorder=5)
    ax.text(cx_dit, dit_by + dit_bh - 0.32, "8 AdaLN-Zero",
            ha="center", va="center", fontsize=FONT_ARCH_SUBLABEL,
            color="black", zorder=5)
    ax.text(cx_dit, dit_by + dit_bh - 0.46, "22.1M params",
            ha="center", va="center", fontsize=FONT_ARCH_SUBLABEL,
            color=C_GREY, zorder=5)
    ax.text(cx_dit, dit_by + dit_bh - 0.66,
            r"$v(z_t, t, c)$",
            ha="center", va="center", fontsize=FONT_ARCH_SUBLABEL,
            color="black", zorder=5)

    draw_arrow(ax, (dit_x0 + z0_w, dit_y_mid + SBH / 2),
               (dit_bx, dit_y_mid + SBH / 2),
               color=C_GEN_MID, linewidth=1.3)

    ode_x = dit_bx + dit_bw + 0.12
    ode_w = 0.60
    draw_box(ax, (ode_x, dit_y_mid - 0.02), ode_w, SBH + 0.06, "ODE",
             sublabel="Euler/Mid",
             facecolor=C_GEN_BOX, edgecolor=C_GEN_DARK, fontsize=FONT_ARCH_LABEL,
             textcolor="black")
    draw_arrow(ax, (dit_bx + dit_bw, dit_y_mid + SBH / 2),
               (ode_x, dit_y_mid + SBH / 2),
               color=C_GEN_MID, linewidth=1.3)

    cond_bx = dit_bx + 0.02
    cond_by = dit_by - 0.52
    cond_bw = 0.88
    cond_bh = 0.36
    draw_box(ax, (cond_bx, cond_by), cond_bw, cond_bh,
             "Cond. Embed",
             sublabel="512→512",
             facecolor=C_SHARED_BOX, edgecolor=C_SHARED, fontsize=FONT_ARCH_SUBLABEL,
             textcolor="black")

    draw_arrow(ax, (shared_x + shared_w, shared_y + shared_h * 0.3),
               (cond_bx, cond_by + cond_bh / 2),
               color=C_SHARED, linewidth=1.0,
               connectionstyle="arc3,rad=-0.08")
    cond_label_x = cond_bx + cond_bw / 2
    cond_label_y = cond_by - 0.22
    cond_label = FancyBboxPatch(
        (cond_label_x - 0.32, cond_label_y - 0.07), 0.64, 0.16,
        boxstyle="round,pad=0.03",
        facecolor=C_WHITE,
        edgecolor=C_SHARED,
        linewidth=0.8,
        zorder=5,
    )
    ax.add_patch(cond_label)
    ax.text(cond_label_x, cond_label_y, "condition c",
            ha="center", va="center", fontsize=FONT_ARCH_SUBLABEL,
            color="black", zorder=6)
    draw_arrow(ax, (cond_bx + cond_bw / 2, cond_by + cond_bh),
               (dit_bx + dit_bw * 0.35, dit_by),
               color=C_SHARED, linewidth=1.0)

    adaln_box_x = cond_bx + cond_bw / 2 + 0.72
    adaln_box_y = cond_by + cond_bh + 0.10
    adaln_box = FancyBboxPatch(
        (adaln_box_x, adaln_box_y), 0.42, 0.17,
        boxstyle="round,pad=0.03",
        facecolor=C_WHITE,
        edgecolor=C_SHARED,
        linewidth=0.8,
        zorder=5,
    )
    ax.add_patch(adaln_box)
    ax.text(adaln_box_x + 0.21, adaln_box_y + 0.085,
            "AdaLN", ha="center", va="center", fontsize=FONT_ARCH_SUBLABEL,
            color="black", zorder=6)

    time_bx = cond_bx + cond_bw + 0.10
    time_by = cond_by
    draw_box(ax, (time_bx, time_by), 0.50, cond_bh,
             r"$t$", sublabel="sin emb",
             facecolor=C_GEN_BOX, edgecolor=C_GEN_DARK, fontsize=FONT_ARCH_LABEL,
             textcolor="black")
    draw_arrow(ax, (time_bx + 0.25, time_by + cond_bh),
               (dit_bx + dit_bw * 0.65, dit_by),
               color=C_GEN_MID, linewidth=0.8,
               connectionstyle="arc3,rad=0.10")

    cfg_y = 2.34
    ax.text(cx_dit, cfg_y + 0.24, "CFG guidance",
            ha="center", va="center", fontsize=FONT_ARCH_LABEL,
            color="black", zorder=5)
    ax.text(cx_dit, cfg_y,
            r"$\mathbf{v} = v_{\rm unc} + s(v_{\rm cond} - v_{\rm unc})$",
            ha="center", va="center", fontsize=FONT_ARCH_LABEL,
            color="black", zorder=5)

    # Stage 3: decoder
    dec_x0 = 6.18
    dec_y_mid = 1.40
    z1_w = 0.40
    draw_box(ax, (dec_x0, dec_y_mid), z1_w, SBH,
             r"$z_1$",
             facecolor=C_GEN_BOX, edgecolor=C_GEN_DARK, fontsize=FONT_ARCH_SUBLABEL,
             textcolor="black")
    ax.text(dec_x0 + z1_w / 2, dec_y_mid - 0.10, "512-d",
            ha="center", va="top", fontsize=FONT_ARCH_SUBLABEL, color=C_GREY, zorder=5)

    draw_arrow(ax, (ode_x + ode_w, dit_y_mid + SBH / 2),
               (dec_x0, dec_y_mid + SBH / 2),
               color=C_GEN_MID, linewidth=1.3)
    sampled_label_x = (ode_x + ode_w + dec_x0) / 2 - 0.10
    sampled_label_y = dec_y_mid + SBH / 2 + 0.38
    sampled_label = FancyBboxPatch(
        (sampled_label_x - 0.41, sampled_label_y - 0.07), 0.82, 0.16,
        boxstyle="round,pad=0.03",
        facecolor=C_WHITE,
        edgecolor=C_GEN_DARK,
        linewidth=0.8,
        zorder=8,
    )
    ax.add_patch(sampled_label)
    ax.text(sampled_label_x, sampled_label_y, "sampled latent",
            ha="center", va="center", fontsize=FONT_ARCH_SUBLABEL, color="black", zorder=9)

    dec_bx = dec_x0 + z1_w + 0.10
    dec_w = 0.70
    dec_h = BH + 0.12
    decoder_y = dec_y_mid - 0.15
    decoder_box = FancyBboxPatch(
        (dec_bx, decoder_y), dec_w, dec_h,
        boxstyle="round,pad=0.08",
        facecolor=C_DECODE_BOX,
        edgecolor=C_DECODE_DARK,
        linewidth=1.3,
        zorder=3,
        mutation_scale=0.5,
    )
    ax.add_patch(decoder_box)
    ax.text(
        dec_bx + dec_w / 2,
        decoder_y + dec_h * 0.66,
        "scGPT\nDecoder",
        ha="center",
        va="center",
        fontsize=FONT_ARCH_LABEL,
        color="black",
        zorder=4,
    )
    ax.text(
        dec_bx + dec_w / 2,
        decoder_y + dec_h * 0.20,
        "frozen",
        ha="center",
        va="center",
        fontsize=FONT_ARCH_SUBLABEL,
        color=C_GREY,
        zorder=4,
    )
    draw_arrow(ax, (dec_x0 + z1_w, dec_y_mid + SBH / 2),
               (dec_bx, dec_y_mid + SBH / 2 - 0.04),
               color=C_DECODE_MID, linewidth=1.3)

    out_x = dec_bx - 0.10
    out_y = dec_y_mid + 0.68
    out_w = 0.90
    draw_box(ax, (out_x, out_y), out_w, SBH + 0.04, "Gene Expr.\nProfile",
             facecolor="#E8EAF6", edgecolor="#283593", fontsize=FONT_ARCH_LABEL,
             textcolor="black")
    draw_arrow(ax, (dec_bx + dec_w / 2, decoder_y + dec_h),
               (out_x + out_w / 2, out_y),
               color=C_DECODE_MID, linewidth=1.3)

    # Legend
    legend_y = -0.08
    legend_items = [
        (C_TEXT_LIGHT, C_TEXT_DARK, "Text Path"),
        (C_CELL_LIGHT, C_CELL_DARK, "Cell Path"),
        (C_GEN_LIGHT, C_GEN_DARK, "Generation"),
        (C_SHARED_BOX, C_SHARED, "Shared Space"),
        (C_DECODE_BOX, C_DECODE_DARK, "Decoding"),
    ]
    lx = 1.2
    for fc, ec, label in legend_items:
        box = FancyBboxPatch(
            (lx, legend_y), 0.18, 0.14,
            boxstyle="round,pad=0.02",
            facecolor=fc, edgecolor=ec,
            linewidth=0.6, zorder=5,
        )
        ax.add_patch(box)
        ax.text(lx + 0.22, legend_y + 0.06, label,
                fontsize=FONT_ARCH_SUBLABEL, va="center", color=ec, zorder=5)
        lx += 1.15

    output_path = output_dir / "fig_architecture.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_panel(fig, output_path, dpi=300, close=False)
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    import argparse
    from src.utils.paths import FIG_DIR

    parser = argparse.ArgumentParser(description="Generate CLOP-DiT architecture figure (Fig 1).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"Output directory for PNG/PDF (default: {FIG_DIR})")
    args = parser.parse_args()
    create_architecture_figure(output_dir=args.output_dir)
