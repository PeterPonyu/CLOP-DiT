"""Article figure delivery — single source of truth for the article-facing figure assets.

Verify PDF figure assets exist in a source directory and copy them
to the article figures directory so LaTeX can include the PDF
assets directly. The manifest below is the canonical list;
scripts and docs should reference this module.

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
from typing import Dict, List, Tuple

# Single source of truth: 22 article-facing component basenames (no suffix).
# Order matches the revised manuscript sequence, including the strict-OOD figure
# now promoted into the main revised article body.
#
# The manuscript now uses article-index-aware component names:
#   Fig 1  -> fig01a_*, fig01b_*
#   Fig 2  -> fig02a_*, fig02b_*
#   ...
#   Fig 9  -> fig09a_*, fig09b_*
#   Fig S1 -> figS01_*
#   Fig S2 -> figS02_*
#
# Both the generated source assets in results/figures/ and the delivered
# article-facing assets use the same basenames so there is no second naming
# layer to drift out of sync.

_SOURCE_BASENAMES: List[str] = [
    # Main body display Figure 1
    "fig01a_architecture",
    "fig01b_evaluation_pipeline",
    # Main body display Figure 2
    "fig02a_training_dynamics",
    "fig02b_embedding_space",
    # Main body display Figure 3
    "fig03a_metrics_summary",
    "fig03b_per_type_fidelity",
    "fig03c_text_cell_alignment",
    # Main body display Figure 4
    "fig04a_marker_genes",
    "fig04b_expression_correlation",
    # Main body display Figure 5
    "fig05a_expression_analysis",
    "fig05b_conditioning_landscape",
    # Main body display Figure 6
    "fig06_diversity_diagnostics",
    # Main body display Figure 7
    "fig07a_expression_diversity",
    "fig07b_baseline_comparison",
    "fig07c_benchmark",
    # Main body display Figure 8
    "fig08a_downstream_validation",
    "fig08b_de_concordance",
    # Main body display Figure 9
    "fig09a_variance_matching",
    "fig09b_gene_gene_correlation",
    # Main body strict-OOD figure
    "figS_lane_c_zero_shot",
    # Supplementary display Figure S1 (single Python-composed appendix figure)
    "figS01_supplementary_validation",
    # Supplementary display Figure S2 (single Python-composed appendix figure)
    "figS02_expression_diagnostics",
]

ARTICLE_FIGURE_BASENAMES: List[str] = list(_SOURCE_BASENAMES)

# Map from article-facing basename → generated source basename.
# This is currently an identity map because source assets now follow article numbering.
_SOURCE_MAP = {basename: basename for basename in ARTICLE_FIGURE_BASENAMES}

# Basename -> producer file path (relative to repo root). Order matches Fig 1–20.
ARTICLE_FIGURE_PRODUCERS: List[Tuple[str, str]] = [
    ("fig01a_architecture", "scripts/analysis/generate_architecture_figure.py"),
    ("fig01b_evaluation_pipeline", "scripts/analysis/evaluation_pipeline_figure.py"),
    ("fig02a_training_dynamics", "src/visualization/fig03_training.py"),
    ("fig02b_embedding_space", "src/visualization/fig04_embedding.py"),
    ("fig03a_metrics_summary", "src/visualization/fig05_metrics.py"),
    ("fig03b_per_type_fidelity", "src/visualization/fig06_fidelity.py"),
    ("fig03c_text_cell_alignment", "src/visualization/fig07_alignment.py"),
    ("fig04a_marker_genes", "src/visualization/fig08_markers.py"),
    ("fig04b_expression_correlation", "src/visualization/fig09_expression_corr.py"),
    ("fig05a_expression_analysis", "src/visualization/fig10_expression_analysis.py"),
    ("fig05b_conditioning_landscape", "src/visualization/fig11_conditioning.py"),
    ("fig06_diversity_diagnostics", "src/visualization/fig12_diversity.py"),
    ("fig07a_expression_diversity", "src/visualization/fig14_expr_diversity.py"),
    ("fig07b_baseline_comparison", "src/visualization/fig15_baselines.py"),
    ("fig07c_benchmark", "src/visualization/fig16_benchmark.py"),
    ("fig08a_downstream_validation", "src/visualization/fig17_downstream.py"),
    ("fig08b_de_concordance", "src/visualization/fig18_de_concordance.py"),
    ("fig09a_variance_matching", "scripts/analysis/variance_matching_pilot.py"),
    ("fig09b_gene_gene_correlation", "scripts/analysis/gene_gene_correlation.py"),
    ("figS_lane_c_zero_shot", "scripts/analysis/lane_c_zero_shot_figure.py"),
    ("figS01_supplementary_validation", "src/visualization/figS01_supplementary_validation.py"),
    ("figS02_expression_diagnostics", "src/visualization/figS02_expression_diagnostics.py"),
]

_N_FIGURES = len(ARTICLE_FIGURE_BASENAMES)
_ARTICLE_INCLUDE_SUFFIX = ".pdf"
_SOURCE_REQUIRED_SUFFIXES = (".pdf",)
_TARGET_DELIVERY_SUFFIXES = (".pdf",)
_PREVIEW_SUFFIXES = {".jpg"}
_SOURCE_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png"}
_TARGET_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png"}
_TARGET_KEEP_NAMES = {"README.md"}


def expected_source_artifacts(*, keep_preview: bool = False) -> set[str]:
    """Return the canonical generated figure artifacts in ``results/figures``."""
    suffixes = {".pdf"}
    if keep_preview:
        suffixes.update(_PREVIEW_SUFFIXES)
    return {f"{source_base}{suffix}" for source_base in _SOURCE_BASENAMES for suffix in suffixes}


def expected_target_artifacts() -> set[str]:
    """Return the canonical article-facing artifacts in ``articles/figures``."""
    return {
        f"{article_base}{suffix}"
        for article_base in ARTICLE_FIGURE_BASENAMES
        for suffix in _TARGET_DELIVERY_SUFFIXES
    } | _TARGET_KEEP_NAMES


def cleanup_legacy_artifacts(
    source_dir: Path,
    target_dir: Path,
    *,
    keep_preview: bool = True,
) -> Dict[str, list[str]]:
    """Remove stale figure artifacts outside the canonical manifest.

    Returns a summary mapping ``source_removed`` / ``target_removed`` to the
    deleted filenames.
    """
    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()

    source_keep = expected_source_artifacts(keep_preview=keep_preview)
    target_keep = expected_target_artifacts()
    removed = {"source_removed": [], "target_removed": []}

    if source_dir.exists():
        for item in sorted(source_dir.iterdir()):
            if not item.is_file():
                continue
            if item.suffix.lower() not in _SOURCE_SUFFIXES:
                continue
            if not (item.stem.startswith("fig") or item.name == "clop_dit_full_report.pdf"):
                continue
            if item.name in source_keep:
                continue
            item.unlink()
            removed["source_removed"].append(item.name)

    if target_dir.exists():
        for item in sorted(target_dir.iterdir()):
            if item.name in target_keep:
                continue
            if item.suffix.lower() not in _TARGET_SUFFIXES:
                continue
            if not (item.stem.startswith("fig") or item.is_symlink()):
                continue
            item.unlink()
            removed["target_removed"].append(item.name)

    return removed


def deliver_figures(
    source_dir: Path,
    target_dir: Path,
    *,
    symlink: bool = False,
    check_only: bool = False,
    cleanup: bool = False,
    keep_preview: bool = True,
) -> bool:
    """Verify all article figure PDFs exist in source_dir and optionally copy to target_dir.

    Parameters
    ----------
    source_dir : directory containing the generated figures (e.g. results/figures)
    target_dir : directory for the article (e.g. articles/figures)
    symlink : if True, create symlinks; if False, copy files (ignored when check_only=True)
    check_only : if True, only verify presence in source_dir; do not modify target_dir

    Returns
    -------
    True if all required PDFs are present (and, when not check_only,
    successfully linked/copied).
    """
    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()

    missing: List[str] = []
    for article_base in ARTICLE_FIGURE_BASENAMES:
        source_base = _SOURCE_MAP[article_base]
        for suffix in _SOURCE_REQUIRED_SUFFIXES:
            path = source_dir / f"{source_base}{suffix}"
            if not path.is_file():
                missing.append(f"{source_base}{suffix} (→ {article_base}{suffix})")
    if missing:
        print(
            f"Missing {len(missing)} required article figure assets across {_N_FIGURES} figures in {source_dir}:",
            file=sys.stderr,
        )
        for m in missing:
            print(f"  MISSING: {m}", file=sys.stderr)
        return False

    if check_only:
        return True

    target_dir.mkdir(parents=True, exist_ok=True)
    if cleanup:
        removed = cleanup_legacy_artifacts(source_dir, target_dir, keep_preview=keep_preview)
        for category, names in removed.items():
            if names:
                print(f"  Cleaned {category}: {', '.join(names)}")
    for article_base in ARTICLE_FIGURE_BASENAMES:
        source_base = _SOURCE_MAP[article_base]
        for suffix in _TARGET_DELIVERY_SUFFIXES:
            src = source_dir / f"{source_base}{suffix}"
            dst = target_dir / f"{article_base}{suffix}"
            if symlink:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
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
        description="Verify article PDF figures and copy them to the article figures directory."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=f"Only verify all {_N_FIGURES} PDF figures exist in source dir; do not copy",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=True,
        help="Copy files instead of creating symlinks (default: copy)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not remove stale figure artifacts outside the canonical manifest",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=f"Source directory with generated JPEG/PDF figure pairs (default: {default_source})",
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
        print(f"Checking {_N_FIGURES} article figure PDFs in {source}...")
        ok = deliver_figures(source, target, symlink=False, check_only=True)
        if ok:
            print(f"  All {_N_FIGURES} PDF figures present.")
        else:
            print("Run 'bash scripts/regenerate_report.sh' to generate them.", file=sys.stderr)
        return 0 if ok else 1

    print(f"Checking {_N_FIGURES} article figure PDFs in {source}...")
    ok = deliver_figures(
        source,
        target,
        symlink=not args.copy,
        check_only=False,
        cleanup=not args.no_cleanup,
    )
    if not ok:
        print("Run 'bash scripts/regenerate_report.sh' to generate them.", file=sys.stderr)
        return 1
    print(f"  All {_N_FIGURES} PDF figures present.")
    mode = "copied" if args.copy else "symlinked"
    print(f"  {_N_FIGURES} PDF figures {mode} to {target}.")
    print("")
    print(
        f"All {_N_FIGURES} article PDF figures verified and {mode}; "
        f"LaTeX should include the {_ARTICLE_INCLUDE_SUFFIX} assets."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
