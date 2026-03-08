#!/usr/bin/env python3
"""
run_regeneration.py — Regenerate all 17 article figures from cached JSON results,
run VCD on every output, fix violations, and verify the full submission package.

Usage:
    python scripts/pipeline/run_regeneration.py [--no-vcd] [--skip-arch]

Output:
    results/figures/*.pdf         — regenerated figures
    articles/figures/*.pdf        — symlinks for LaTeX
    results/vcd_report.json       — per-figure VCD findings
    results/vcd_report_summary.md — human-readable VCD summary
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless

# ── paths ──────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "vcd"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

import matplotlib.pyplot as plt
from src.utils.paths import FIG_DIR, RESULTS_DIR, ARTICLE_FIGURES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("regenerate")

CLOP_HIST = REPO / "models" / "checkpoints" / "CLOP" / "versions" / "v9.3" / "clop_history.json"
DIT_HIST  = REPO / "models" / "checkpoints" / "DiT" / "versions" / "v2.0" / "dit_history.json"


def run_results_visualizer():
    """Generate all figures A–S via the ResultsVisualizer."""
    log.info("── Initialising ResultsVisualizer ──")
    from visualization.results_visualizer import ResultsVisualizer

    clop_hist_path = str(CLOP_HIST) if CLOP_HIST.exists() else None
    dit_hist_path  = str(DIT_HIST) if DIT_HIST and DIT_HIST.exists() else None

    viz = ResultsVisualizer(
        clop_history_path=clop_hist_path,
        dit_history_path=dit_hist_path,
        cache_dir=str(REPO / "data" / "cached_latents_v5.2"),
        output_dir=str(FIG_DIR),
        dpi=300,
        auto_refine=True,
        max_refine_passes=3,
    )

    log.info("── Generating all panels ──")
    saved = viz.generate_full_report(include_umap=True)
    plt.close("all")
    log.info("ResultsVisualizer saved %d panels", len(saved))
    return saved


def run_architecture_figure():
    """Generate Fig 1: architecture diagram (standalone script, no heavy deps)."""
    import subprocess, sys
    arch_script = REPO / "scripts" / "analysis" / "generate_architecture_figure.py"
    if not arch_script.exists():
        log.warning("Architecture script not found: %s", arch_script)
        return None
    log.info("── Generating fig_architecture.pdf ──")
    result = subprocess.run(
        [sys.executable, str(arch_script)],
        capture_output=True, text=True, cwd=str(REPO)
    )
    if result.returncode != 0:
        log.error("Architecture figure failed:\n%s", result.stderr[-2000:])
        return None
    arch_pdf = FIG_DIR / "fig_architecture.pdf"
    log.info("Architecture figure: %s (%s)", arch_pdf, "exists" if arch_pdf.exists() else "MISSING")
    return arch_pdf if arch_pdf.exists() else None


def run_vcd_on_figures(pdf_list: list[Path]) -> dict:
    """Run the Visual Conflict Detector on each PDF (rendered to a figure)."""
    try:
        from visual_conflict_detector import detect_conflicts_in_file, summarize_issues
    except ImportError as e:
        log.warning("VCD unavailable: %s — skipping VCD pass", e)
        return {}

    vcd_results: dict = {}
    total_warn = total_info = 0
    failures = []

    for pdf in sorted(pdf_list):
        if not pdf.exists():
            log.warning("VCD skip (missing): %s", pdf.name)
            continue
        try:
            issues = detect_conflicts_in_file(str(pdf))
            warnings  = [i for i in issues if getattr(i, "level", "").upper() in ("WARNING","WARN","ERROR","CRITICAL")]
            info_only = [i for i in issues if getattr(i, "level", "").upper() in ("INFO","HINT","LOW")]
            total_warn += len(warnings)
            total_info += len(info_only)
            vcd_results[pdf.name] = {
                "warnings": [str(i) for i in warnings],
                "info": [str(i) for i in info_only],
                "total": len(issues)
            }
            level = "PASS" if not warnings else ("WARN" if len(warnings) < 3 else "FAIL")
            log.info("[VCD %-35s]  %s  warn=%d info=%d",
                     pdf.name[:35], level, len(warnings), len(info_only))
            if warnings:
                for w in warnings[:5]:
                    log.info("    %s", str(w)[:120])
                if len(warnings) > 5:
                    log.info("    ... +%d more", len(warnings) - 5)
                failures.append((pdf.name, warnings))
        except Exception as exc:
            log.error("VCD error on %s: %s", pdf.name, exc)
            vcd_results[pdf.name] = {"error": str(exc)}

    log.info("── VCD summary: total_warnings=%d  total_info=%d  files_with_warnings=%d ──",
             total_warn, total_info, len(failures))
    vcd_results["__summary__"] = {
        "total_warnings": total_warn,
        "total_info": total_info,
        "files_with_warnings": len(failures),
    }
    return vcd_results


def save_vcd_report(vcd: dict):
    """Write JSON + markdown VCD reports to results/."""
    json_path = RESULTS_DIR / "vcd_report.json"
    with open(json_path, "w") as f:
        json.dump(vcd, f, indent=2)
    log.info("VCD JSON saved: %s", json_path)

    md_lines = ["# VCD Report — " + time.strftime("%Y-%m-%d %H:%M"), ""]
    summary = vcd.get("__summary__", {})
    md_lines += [
        f"**Total warnings:** {summary.get('total_warnings', '?')}",
        f"**Files with warnings:** {summary.get('files_with_warnings', '?')}",
        "",
        "## Per-Figure",
        "",
    ]
    for name, data in vcd.items():
        if name.startswith("__"):
            continue
        w = data.get("warnings", [])
        md_lines.append(f"### {name}")
        if not w:
            md_lines.append("- PASS (0 warnings)")
        else:
            for item in w:
                md_lines.append(f"- WARNING: {item[:200]}")
        md_lines.append("")

    md_path = RESULTS_DIR / "vcd_report_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    log.info("VCD markdown saved: %s", md_path)


def run_article_delivery():
    """Create symlinks in articles/figures/ and verify submission package."""
    log.info("── Running article_delivery.py ──")
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(REPO / "src" / "visualization" / "article_delivery.py")],
        capture_output=True, text=True, cwd=str(REPO)
    )
    for line in result.stdout.splitlines():
        log.info("  [delivery] %s", line)
    if result.stderr:
        for line in result.stderr.splitlines()[:20]:
            log.warning("  [delivery stderr] %s", line)
    if result.returncode != 0:
        log.error("article_delivery.py exited with code %d", result.returncode)
    return result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate all article figures + VCD")
    parser.add_argument("--no-vcd",    action="store_true", help="Skip VCD pass")
    parser.add_argument("--skip-arch", action="store_true", help="Skip architecture figure")
    parser.add_argument("--no-delivery", action="store_true", help="Skip article_delivery")
    args = parser.parse_args()

    t0 = time.time()
    log.info("=" * 70)
    log.info("CLOP-DiT Figure Regeneration Pipeline — %s", time.strftime("%Y-%m-%d"))
    log.info("=" * 70)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate all panels from ResultsVisualizer
    try:
        saved = run_results_visualizer()
    except Exception as e:
        log.error("ResultsVisualizer failed: %s", e, exc_info=True)
        saved = []

    # 2. Architecture figure (standalone)
    if not args.skip_arch:
        arch = run_architecture_figure()
        if arch:
            saved.append(arch)

    all_pdfs = sorted(set(FIG_DIR.glob("panel_*.pdf")) | set(FIG_DIR.glob("fig_*.pdf")))
    log.info("Total PDFs in results/figures: %d", len(all_pdfs))
    for p in all_pdfs:
        log.info("  ✓ %s", p.name)

    # 3. VCD pass
    if not args.no_vcd:
        vcd = run_vcd_on_figures(all_pdfs)
        save_vcd_report(vcd)
    else:
        log.info("VCD skipped (--no-vcd)")

    # 4. Article delivery (symlinks)
    if not args.no_delivery:
        ret = run_article_delivery()
        if ret == 0:
            log.info("Article delivery: OK")
        else:
            log.warning("Article delivery: FAILED (exit %d)", ret)

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info("Done in %.1fs", elapsed)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
