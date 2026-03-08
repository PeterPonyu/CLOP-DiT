#!/usr/bin/env python3
"""
Generate PNG files from PDF figures on demand.

Usage:
    python generate_pngs.py --all              # Generate all PNGs
    python generate_pngs.py --panel A         # Generate panel A PNG
    python generate_pngs.py --panel B --dpi 300 # Generate with custom DPI
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Panel list (A-S)
PANELS = [
    "panel_a_clop_training",
    "panel_b_clop_embedding_umap",
    "panel_c_umap_quality",
    "panel_d_markers_per_type",
    "panel_e_clop_metrics",
    "panel_f_type_composition",
    "panel_g_cell_generation",
    "panel_h_marker_heatmap",
    "panel_i_umap_cell_types",
    "panel_j_diversity_metrics",
    "panel_k_conditioning",
    "panel_l_clustering",
    "panel_m_classifier_performance",
    "panel_n_de_concordance",
    "panel_o_diversity_umap",
    "panel_p_marker_genes",
    "panel_q_downstream_tasks",
    "panel_r_baseline_comparison",
    "panel_s_cell_cell_transfer",
]


def pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 150) -> bool:
    """Convert PDF to PNG using ImageMagick or pdftoppm."""
    try:
        # Try pdftoppm first (faster, better quality)
        result = subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(png_path.parent / png_path.stem)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # pdftoppm adds -1 suffix, rename
            generated = png_path.parent / f"{png_path.stem}-1.png"
            if generated.exists():
                generated.rename(png_path)
            return True
    except FileNotFoundError:
        pass

    # Fallback to ImageMagick
    try:
        result = subprocess.run(
            ["convert", "-density", str(dpi), str(pdf_path), "-quality", "90", str(png_path)],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        print("Error: Neither pdftoppm nor ImageMagick convert found.")
        print("Install with: sudo apt-get install poppler-utils (pdftoppm)")
        print("Or: sudo apt-get install imagemagick (convert)")
        return False


def generate_panel(panel_letter: str, dpi: int = 150) -> bool:
    """Generate PNG for a specific panel."""
    panel_name = f"panel_{panel_letter.lower()}"

    # Find matching panel
    matching = [p for p in PANELS if p.startswith(panel_name)]
    if not matching:
        print(f"Error: Panel '{panel_letter}' not found.")
        print(f"Available panels: {', '.join([p[6].upper() for p in PANELS])}")
        return False

    panel = matching[0]
    pdf_path = Path(__file__).parent / f"{panel}.pdf"
    png_path = Path(__file__).parent / "generated_pngs" / f"{panel}.png"

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        return False

    png_path.parent.mkdir(exist_ok=True)

    print(f"Generating {panel}.png (DPI: {dpi})...")
    if pdf_to_png(pdf_path, png_path, dpi):
        size_kb = png_path.stat().st_size / 1024
        print(f"  Created: {png_path} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"  Failed to generate {panel}.png")
        return False


def generate_all(dpi: int = 150) -> int:
    """Generate PNGs for all panels."""
    success = 0
    failed = 0

    for panel in PANELS:
        pdf_path = Path(__file__).parent / f"{panel}.pdf"
        if not pdf_path.exists():
            print(f"Skipping {panel} (PDF not found)")
            failed += 1
            continue

        if generate_panel(panel[6], dpi):  # Extract panel letter
            success += 1
        else:
            failed += 1

    print(f"\nGenerated: {success} PNGs")
    print(f"Failed: {failed}")
    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate PNG files from PDF figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_pngs.py --all              # Generate all PNGs
  python generate_pngs.py --panel A          # Generate panel A
  python generate_pngs.py --panel A --dpi 300  # High DPI
  python generate_pngs.py --panel S          # Generate panel S (last one)
        """,
    )

    parser.add_argument("--all", action="store_true", help="Generate all panels")
    parser.add_argument("--panel", type=str, help="Generate specific panel (A-S)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for PNG (default: 150)")

    args = parser.parse_args()

    if not args.all and not args.panel:
        parser.print_help()
        sys.exit(1)

    if args.all:
        sys.exit(generate_all(args.dpi))
    elif args.panel:
        success = generate_panel(args.panel, args.dpi)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
