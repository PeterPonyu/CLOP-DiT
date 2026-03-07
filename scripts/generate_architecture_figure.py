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

# Ensure project root is on sys.path for style imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.visualization.style import (
    COLORS,
    FONT_ARCH_LABEL,
    FONT_ARCH_SUBLABEL,
    apply_style,
    save_panel,
)

# ---------------------------------------------------------------------------
# Apply global publication style
# ---------------------------------------------------------------------------
apply_style()
matplotlib.rcParams.update({
    "axes.grid": False,
    "figure.constrained_layout.use": False,
})

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
# Stage 1: CLOP (blue text / green cell)
C_TEXT_DARK = COLORS["real"]
C_TEXT_MID = "#1E88E5"
C_TEXT_LIGHT = "#90CAF9"
C_TEXT_BOX = "#E3F2FD"
C_TEXT_ACCENT = "#1565C0"

C_CELL_DARK = COLORS["baseline_gauss"]
C_CELL_MID = "#43A047"
C_CELL_LIGHT = "#A5D6A7"
C_CELL_BOX = "#E8F5E9"
C_CELL_ACCENT = "#2E7D32"

C_SHARED = COLORS["baseline_shuffle"]
C_SHARED_LIGHT = "#CE93D8"
C_SHARED_BOX = "#F3E5F5"

# Stage 2: DiT (orange generation)
C_GEN_DARK = COLORS["generated"]
C_GEN_MID = "#E65100"
C_GEN_LIGHT = "#FFCCBC"
C_GEN_BOX = "#FFF3E0"
C_GEN_ACCENT = "#BF360C"

# Stage 3: Decode (teal)
C_DECODE_DARK = "#004D40"
C_DECODE_MID = "#00796B"
C_DECODE_BOX = "#E0F2F1"
C_DECODE_ACCENT = "#00695C"

# Neutral
C_GREY = "#455A64"
C_MID_GREY = "#607D8B"
C_LIGHT_GREY = "#B0BEC5"
C_WHITE = "#FFFFFF"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_box(ax, xy, w, h, label, sublabel=None, facecolor=C_WHITE,
             edgecolor=C_GREY, fontsize=8, sublabel_size=7.0,
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
    txt = ax.text(
        x + w / 2, y + h + 0.02,
        label,
        ha="center", va="bottom",
        fontsize=8, fontweight="normal",
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
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    # Use non-equal aspect so we can fill the canvas properly; xlim extends past 7.0 so right side (decoder, gene expr) is not truncated
    ax.set_xlim(-0.1, 7.45)
    ax.set_ylim(-0.3, 3.4)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_facecolor(C_WHITE)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.04)

    # Box dimensions (compact for 7-inch width)
    BW = 0.78   # standard box width
    BH = 0.38   # standard box height
    SBW = 0.55  # small box width
    SBH = 0.32  # small box height
    gap = 0.08  # horizontal gap between boxes

    # ===================================================================
    #  Stage backgrounds
    # ===================================================================
    draw_stage_bg(ax, (-0.05, -0.15), 3.25, 3.30,
                  "Stage 1: CLOP Alignment",
                  C_TEXT_DARK, alpha=0.08, label_color=C_TEXT_DARK)
    draw_stage_bg(ax, (3.30, -0.15), 2.30, 3.30,
                  "Stage 2: DiT Generation",
                  C_GEN_DARK, alpha=0.08, label_color=C_GEN_DARK)
    draw_stage_bg(ax, (5.70, -0.15), 1.25, 3.30,
                  "Stage 3: Decoding",
                  C_DECODE_DARK, alpha=0.08, label_color=C_DECODE_DARK)

    # ===================================================================
    #  CLOP Stage -- Text path (top) & Cell path (bottom)
    # ===================================================================

    # --- Text path (top row y~2.4) ---
    ty = 2.40
    tx0 = 0.02

    # Text Description input
    draw_box(ax, (tx0, ty), SBW, SBH, "Text\nDescription",
            facecolor=C_TEXT_BOX, edgecolor=C_TEXT_DARK, fontsize=FONT_ARCH_LABEL,
            textcolor=C_TEXT_DARK, bold=True)

    # BiomedBERT-large
    tx1 = tx0 + SBW + gap
    draw_box(ax, (tx1, ty - 0.05), BW, BH + 0.10, "BiomedBERT",
             sublabel="frozen | 340M | 1024-d",
             facecolor=C_TEXT_LIGHT, edgecolor=C_TEXT_DARK, fontsize=7,
             textcolor=C_TEXT_DARK, bold=True, linewidth=1.3)
    draw_arrow(ax, (tx0 + SBW, ty + SBH / 2),
               (tx1, ty + SBH / 2 - 0.02), color=C_TEXT_MID, linewidth=1.2)

    # ZCA Whitening
    tx2 = tx1 + BW + gap
    zca_w = 0.45
    draw_box(ax, (tx2, ty + 0.02), zca_w, SBH - 0.04, "ZCA",
            facecolor=C_TEXT_BOX, edgecolor=C_TEXT_DARK, fontsize=FONT_ARCH_LABEL,
            textcolor=C_TEXT_DARK, bold=True)
    draw_arrow(ax, (tx1 + BW, ty + SBH / 2 - 0.02),
               (tx2, ty + SBH / 2), color=C_TEXT_MID, linewidth=1.2)

    # Text Projector
    tx3 = tx2 + zca_w + gap
    draw_box(ax, (tx3, ty - 0.05), BW, BH + 0.10, "Text Proj.",
             sublabel="MLP 1024\u2192256",
            facecolor=C_TEXT_LIGHT, edgecolor=C_TEXT_DARK, fontsize=7.0,
             textcolor=C_TEXT_DARK, bold=True)
    draw_arrow(ax, (tx2 + zca_w, ty + SBH / 2),
               (tx3, ty + SBH / 2 - 0.02), color=C_TEXT_MID, linewidth=1.2)

    # --- Cell path (bottom row y~0.5) ---
    cy = 0.50
    cx0 = 0.02

    # Cell Profile input
    draw_box(ax, (cx0, cy), SBW, SBH, "Cell\nProfile",
            facecolor=C_CELL_BOX, edgecolor=C_CELL_DARK, fontsize=FONT_ARCH_LABEL,
            textcolor=C_CELL_DARK, bold=True)

    # scGPT Encoder
    cx1 = cx0 + SBW + gap
    draw_box(ax, (cx1, cy - 0.05), BW, BH + 0.10, "scGPT Enc.",
             sublabel="frozen | 51M | 512-d",
             facecolor=C_CELL_LIGHT, edgecolor=C_CELL_DARK, fontsize=7,
             textcolor=C_CELL_DARK, bold=True, linewidth=1.3)
    draw_arrow(ax, (cx0 + SBW, cy + SBH / 2),
               (cx1, cy + SBH / 2 - 0.02), color=C_CELL_MID, linewidth=1.2)

    # Cell Projector (aligned with Text Projector x-position)
    cx2 = tx3
    draw_box(ax, (cx2, cy - 0.05), BW, BH + 0.10, "Cell Proj.",
             sublabel="MLP 512\u2192256",
            facecolor=C_CELL_LIGHT, edgecolor=C_CELL_DARK, fontsize=7.0,
             textcolor=C_CELL_DARK, bold=True)
    draw_arrow(ax, (cx1 + BW, cy + SBH / 2 - 0.02),
               (cx2, cy + SBH / 2 - 0.02), color=C_CELL_MID, linewidth=1.2)

    # --- Shared Space (between text and cell) ---
    shared_w = 0.62
    shared_h = 0.85
    shared_x = tx3 + BW + gap + 0.02
    shared_y = 1.18

    box = FancyBboxPatch(
        (shared_x, shared_y), shared_w, shared_h,
        boxstyle="round,pad=0.04",
        facecolor=C_SHARED_BOX,
        edgecolor=C_SHARED,
        linewidth=1.0,
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(shared_x + shared_w / 2, shared_y + shared_h / 2,
            "Shared\nSpace\n(256-d)",
            ha="center", va="center", fontsize=FONT_ARCH_LABEL,
            color=C_SHARED, zorder=3)

    # Arrows: Text Proj -> Shared (from top)
    draw_arrow(ax, (tx3 + BW, ty + SBH / 2 - 0.02),
               (shared_x, shared_y + shared_h * 0.75),
               color=C_TEXT_MID, linewidth=1.2,
               connectionstyle="arc3,rad=-0.1")
    # Cell Proj -> Shared (from bottom)
    draw_arrow(ax, (cx2 + BW, cy + SBH / 2 - 0.02),
               (shared_x, shared_y + shared_h * 0.25),
               color=C_CELL_MID, linewidth=1.2,
               connectionstyle="arc3,rad=0.1")

    # --- PrototypeSigLIP Loss label ---
    loss_y = 1.58
    loss_x = 0.80
    loss_w = 1.10
    draw_box(ax, (loss_x, loss_y), loss_w, SBH, "PrototypeSigLIP",
            facecolor="#FFF9C4", edgecolor="#F9A825", fontsize=FONT_ARCH_LABEL,
            textcolor="#E65100", bold=True, linewidth=0.8)
    draw_arrow(ax, (loss_x + loss_w, loss_y + SBH / 2),
               (shared_x, shared_y + shared_h / 2),
               color="#F9A825", linewidth=0.8, style="<->")

    # scGPT latent annotation
    ax.text(cx1 + BW / 2, cy + BH + 0.22,
            "scGPT latent =\nDiT training target",
            ha="center", va="bottom", fontsize=FONT_ARCH_SUBLABEL, color=C_MID_GREY,
            fontweight="medium", zorder=5)

    # ===================================================================
    #  DiT Stage (centre)
    # ===================================================================
    dit_x0 = 3.45
    dit_y_mid = 1.45

    # z0 noise input
    z0_w = 0.50
    draw_box(ax, (dit_x0, dit_y_mid), z0_w, SBH,
             r"$z_0$", sublabel="512-d",
             facecolor=C_GEN_BOX, edgecolor=C_GEN_DARK, fontsize=7,
             textcolor=C_GEN_DARK, bold=True)
    ax.text(dit_x0 + z0_w / 2, dit_y_mid + SBH + 0.03,
            r"$\sim\mathcal{N}(0,I)$",
            ha="center", va="bottom", fontsize=FONT_ARCH_SUBLABEL, color=C_GEN_DARK, zorder=5)

    # DiT1D main block
    dit_bx = dit_x0 + z0_w + 0.12
    dit_bw = 1.00
    dit_bh = 1.10
    dit_by = dit_y_mid - 0.40
    draw_box(ax, (dit_bx, dit_by), dit_bw, dit_bh, "",
             facecolor=C_GEN_LIGHT, edgecolor=C_GEN_DARK,
             bold=True, linewidth=1.4, zorder=2)

    # Labels inside DiT block
    cx_dit = dit_bx + dit_bw / 2
    ax.text(cx_dit, dit_by + dit_bh - 0.14, "DiT1D",
            ha="center", va="center", fontsize=9,
            color=C_GEN_DARK, zorder=5)
    ax.text(cx_dit, dit_by + dit_bh - 0.32, "8 AdaLN-Zero",
            ha="center", va="center", fontsize=FONT_ARCH_SUBLABEL,
            color=C_GEN_DARK, zorder=5)
    ax.text(cx_dit, dit_by + dit_bh - 0.46, "22.1M params",
            ha="center", va="center", fontsize=FONT_ARCH_SUBLABEL,
            color=C_GREY, zorder=5)
    ax.text(cx_dit, dit_by + dit_bh - 0.66,
            r"$v(z_t, t, c)$",
            ha="center", va="center", fontsize=7,
            color=C_GEN_DARK, zorder=5)

    # Arrow: z0 -> DiT
    draw_arrow(ax, (dit_x0 + z0_w, dit_y_mid + SBH / 2),
               (dit_bx, dit_y_mid + SBH / 2),
               color=C_GEN_MID, linewidth=1.3)

    # ODE Solver
    ode_x = dit_bx + dit_bw + 0.12
    ode_w = 0.60
    draw_box(ax, (ode_x, dit_y_mid - 0.02), ode_w, SBH + 0.06, "ODE",
            sublabel="Euler/Mid",
            facecolor=C_GEN_BOX, edgecolor=C_GEN_DARK, fontsize=FONT_ARCH_LABEL,
            textcolor=C_GEN_DARK, bold=True)
    draw_arrow(ax, (dit_bx + dit_bw, dit_y_mid + SBH / 2),
               (ode_x, dit_y_mid + SBH / 2),
               color=C_GEN_MID, linewidth=1.3)

    # Condition Embedder
    cond_bx = dit_bx + 0.02
    cond_by = dit_by - 0.52
    cond_bw = 0.88
    cond_bh = 0.36
    draw_box(ax, (cond_bx, cond_by), cond_bw, cond_bh,
            "Cond. Embed",
            sublabel="256\u2192384",
            facecolor=C_SHARED_BOX, edgecolor=C_SHARED, fontsize=FONT_ARCH_SUBLABEL,
            textcolor=C_SHARED, bold=True)

    # Shared Space -> Condition Embedder
    draw_arrow(ax, (shared_x + shared_w, shared_y + shared_h * 0.3),
               (cond_bx, cond_by + cond_bh / 2),
               color=C_SHARED, linewidth=1.0,
               connectionstyle="arc3,rad=-0.08")
    # Condition Embedder -> DiT
    draw_arrow(ax, (cond_bx + cond_bw / 2, cond_by + cond_bh),
               (dit_bx + dit_bw * 0.35, dit_by),
               color=C_SHARED, linewidth=1.0)
    ax.text(dit_bx + dit_bw * 0.35 + 0.35, dit_by - 0.12,
            "AdaLN", ha="center", va="center", fontsize=FONT_ARCH_SUBLABEL,
            color=C_SHARED, zorder=5)

    # Timestep embedding
    time_bx = cond_bx + cond_bw + 0.10
    time_by = cond_by
    draw_box(ax, (time_bx, time_by), 0.50, cond_bh,
            r"$t$", sublabel="sin emb",
            facecolor=C_GEN_BOX, edgecolor=C_GEN_DARK, fontsize=FONT_ARCH_LABEL,
            textcolor=C_GEN_DARK)
    draw_arrow(ax, (time_bx + 0.25, time_by + cond_bh),
               (dit_bx + dit_bw * 0.65, dit_by),
               color=C_GEN_MID, linewidth=0.8,
               connectionstyle="arc3,rad=0.10")

    # CFG formula (top)
    cfg_y = 2.75
    ax.text(cx_dit, cfg_y,
            r"$\mathbf{v} = v_{\rm unc} + s(v_{\rm cond} - v_{\rm unc})$",
            ha="center", va="center", fontsize=FONT_ARCH_LABEL,
            color=C_GEN_DARK, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=C_GEN_BOX,
                      edgecolor=C_GEN_DARK, linewidth=0.6, alpha=0.9))
    ax.text(cx_dit, cfg_y + 0.25, "CFG guidance",
            ha="center", va="center", fontsize=FONT_ARCH_LABEL,
            color=C_GEN_DARK, zorder=5)

    # ===================================================================
    #  Decoder Stage (right)
    # ===================================================================
    dec_x0 = 5.82
    dec_y_mid = 1.45

    # z1 output
    z1_w = 0.40
    draw_box(ax, (dec_x0, dec_y_mid), z1_w, SBH,
             r"$z_1$",
             facecolor=C_GEN_BOX, edgecolor=C_GEN_DARK, fontsize=7,
             textcolor=C_GEN_DARK, bold=True)
    ax.text(dec_x0 + z1_w / 2, dec_y_mid - 0.10, "512-d",
            ha="center", va="top", fontsize=FONT_ARCH_SUBLABEL, color=C_GREY, zorder=5)

    # Arrow: ODE -> z1
    draw_arrow(ax, (ode_x + ode_w, dit_y_mid + SBH / 2),
               (dec_x0, dec_y_mid + SBH / 2),
               color=C_GEN_MID, linewidth=1.3)

    # scGPT Decoder
    dec_bx = dec_x0 + z1_w + 0.10
    dec_w = 0.70
    dec_h = BH + 0.12
    draw_box(ax, (dec_bx, dec_y_mid - 0.08), dec_w, dec_h,
            "scGPT\nDecoder",
            sublabel="frozen",
            facecolor=C_DECODE_BOX, edgecolor=C_DECODE_DARK, fontsize=FONT_ARCH_LABEL,
            textcolor=C_DECODE_DARK, bold=True, linewidth=1.3)
    draw_arrow(ax, (dec_x0 + z1_w, dec_y_mid + SBH / 2),
               (dec_bx, dec_y_mid + SBH / 2),
               color=C_DECODE_MID, linewidth=1.3)

    # Output: Gene Expression
    out_x = dec_bx - 0.10
    out_y = dec_y_mid + 0.62
    out_w = 0.90
    draw_box(ax, (out_x, out_y), out_w, SBH + 0.04, "Gene Expr.\nProfile",
            facecolor="#E8EAF6", edgecolor="#283593", fontsize=FONT_ARCH_LABEL,
            textcolor="#283593", bold=True)
    draw_arrow(ax, (dec_bx + dec_w / 2, dec_y_mid - 0.08 + dec_h),
               (out_x + out_w / 2, out_y),
               color=C_DECODE_MID, linewidth=1.3)

    # ===================================================================
    #  Legend at bottom
    # ===================================================================
    legend_y = -0.22
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

    # ===================================================================
    #  Save
    # ===================================================================
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
