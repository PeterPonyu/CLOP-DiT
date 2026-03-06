"""Article figure delivery — single source of truth for the 15 MDPI article figures.

Verify PDFs exist in a source directory and create symlinks (or copies) in the
article figures directory so LaTeX can include them. The manifest below is the
canonical list; scripts and docs should reference this module.

Usage:
    python -m src.visualization.article_delivery              # verify + symlink
    python -m src.visualization.article_delivery --check-only  # verify only
    python -m src.visualization.article_delivery --copy      # copy instead of symlink
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Single source of truth: 15 article figure basenames (no .pdf).
# Order: 6 merged figures, then 9 standalone panels (matches LaTeX and FIGURE_ORGANIZATION.md).
ARTICLE_FIGURE_BASENAMES: List[str] = [
    "fig_architecture",
    "fig_training_dynamics",
    "fig_embedding_space",
    "fig_fidelity_alignment",
    "fig_diversity_tradeoff",
    "fig_downstream_pq",
    "panel_d_metrics_summary",
    "panel_n_marker_gene_comparison",
    "panel_h_expression_correlation",
    "panel_i_expression_analysis",
    "panel_m_conditioning_umap",
    "panel_j_diversity_diagnostics",
    "panel_o_baseline_comparison",
    "panel_s_benchmark",
    "panel_r_de_concordance",
]

# Basename -> producer file path (relative to repo root). Used for presentation-policy tests.
# Must match FIGURE_ORGANIZATION.md "Canonical producers" table.
ARTICLE_FIGURE_PRODUCERS: List[Tuple[str, str]] = [
    ("fig_architecture", "scripts/generate_architecture_figure.py"),
    ("fig_training_dynamics", "src/visualization/panels_training.py"),
    ("fig_embedding_space", "src/visualization/panels_merged.py"),
    ("fig_fidelity_alignment", "src/visualization/panels_merged.py"),
    ("fig_diversity_tradeoff", "src/visualization/results_visualizer.py"),
    ("fig_downstream_pq", "src/visualization/downstream_panels.py"),
    ("panel_d_metrics_summary", "src/visualization/panels_metrics.py"),
    ("panel_n_marker_gene_comparison", "src/visualization/panels_expression.py"),
    ("panel_h_expression_correlation", "src/visualization/panels_expression.py"),
    ("panel_i_expression_analysis", "src/visualization/panels_expression.py"),
    ("panel_m_conditioning_umap", "scripts/conditioning_analysis.py"),
    ("panel_j_diversity_diagnostics", "scripts/diversity_diagnostics.py"),
    ("panel_o_baseline_comparison", "src/visualization/baseline_panels.py"),
    ("panel_s_benchmark", "src/visualization/benchmark_panels.py"),
    ("panel_r_de_concordance", "src/visualization/panels_de_concordance.py"),
]


def deliver_figures(
    source_dir: Path,
    target_dir: Path,
    *,
    symlink: bool = True,
    check_only: bool = False,
) -> bool:
    """Verify all article figure PDFs exist in source_dir and optionally link/copy to target_dir.

    Parameters
    ----------
    source_dir : directory containing the generated PDFs (e.g. results/figures)
    target_dir : directory for the article (e.g. articles/figures)
    symlink : if True, create symlinks; if False, copy files (ignored when check_only=True)
    check_only : if True, only verify presence in source_dir; do not modify target_dir

    Returns
    -------
    True if all 15 PDFs are present (and, when not check_only, successfully linked/copied).
    """
    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()

    missing: List[str] = []
    for base in ARTICLE_FIGURE_BASENAMES:
        path = source_dir / f"{base}.pdf"
        if not path.is_file():
            missing.append(f"{base}.pdf")
    if missing:
        print(f"Missing {len(missing)} of {len(ARTICLE_FIGURE_BASENAMES)} article figure PDFs in {source_dir}:", file=sys.stderr)
        for m in missing:
            print(f"  MISSING: {m}", file=sys.stderr)
        return False

    if check_only:
        return True

    target_dir.mkdir(parents=True, exist_ok=True)
    for base in ARTICLE_FIGURE_BASENAMES:
        src = source_dir / f"{base}.pdf"
        dst = target_dir / f"{base}.pdf"
        if symlink:
            # Use resolve() for src so symlink is absolute and portable
            lnk = dst
            if lnk.exists():
                lnk.unlink()
            lnk.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
    return True


def _default_dirs():
    """Default source/target dirs without importing src.utils (avoids pulling in torch)."""
    import os
    root = Path(__file__).resolve().parent.parent.parent
    source = Path(os.environ.get("CLOPDIT_FIG_DIR", str(root / "results" / "figures")))
    target = Path(os.environ.get("CLOPDIT_ARTICLE_FIGURES_DIR", str(root / "articles" / "figures")))
    return source, target


def main() -> int:
    default_source, default_target = _default_dirs()
    parser = argparse.ArgumentParser(
        description="Verify article figure PDFs and create symlinks (or copies) in the article figures directory."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify all 15 PDFs exist in source dir; do not create symlinks/copies",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks (default: symlink)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=f"Source directory with generated PDFs (default: {default_source})",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help=f"Target directory for article figures (default: {default_target})",
    )
    args = parser.parse_args()

    source = args.source_dir if args.source_dir is not None else default_source
    target = args.target_dir if args.target_dir is not None else default_target

    if args.check_only:
        print(f"Checking {len(ARTICLE_FIGURE_BASENAMES)} article figures in {source}...")
        ok = deliver_figures(source, target, symlink=True, check_only=True)
        if ok:
            print("  All 15 PDFs present.")
        else:
            print("Run 'bash scripts/regenerate_report.sh' to generate them.", file=sys.stderr)
        return 0 if ok else 1

    print(f"Checking {len(ARTICLE_FIGURE_BASENAMES)} article figures in {source}...")
    ok = deliver_figures(source, target, symlink=not args.copy, check_only=False)
    if not ok:
        print("Run 'bash scripts/regenerate_report.sh' to generate them.", file=sys.stderr)
        return 1
    print("  All 15 PDFs present.")
    mode = "copied" if args.copy else "symlinked"
    print(f"  15 figures {mode} in {target}.")
    print("")
    print("All 15 article figures verified and symlinked." if not args.copy else "All 15 article figures verified and copied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
