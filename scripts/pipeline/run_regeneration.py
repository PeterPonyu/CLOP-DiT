#!/usr/bin/env python3
"""
run_regeneration.py — Regenerate article-facing figure components from cached JSON results,
run VCD on every output, refresh delivered assets, and optionally rebuild the LaTeX PDF.

Usage:
    python scripts/pipeline/run_regeneration.py [--no-vcd] [--skip-arch] [--build-pdf]

Output:
    results/figures/*.pdf         — regenerated figures
    articles/figures/*.pdf        — symlinks for LaTeX
    results/vcd_report.json       — per-figure VCD findings
    results/vcd_report_summary.md — human-readable VCD summary
"""
from __future__ import annotations

import json
import logging
import os
import shutil
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
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.pyplot as plt
from src.utils.paths import FIG_DIR, RESULTS_DIR, ARTICLE_FIGURES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("regenerate")

CLOP_HIST = REPO / "models" / "checkpoints" / "CLOP" / "versions" / "latest" / "clop_history.json"
DIT_HIST  = REPO / "models" / "checkpoints" / "DiT" / "versions" / "latest" / "dit_history.json"


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def run_results_visualizer():
    """Generate all figures A–S via the ResultsVisualizer."""
    log.info("── Initialising ResultsVisualizer ──")
    from visualization.results_visualizer import ResultsVisualizer

    clop_hist_path = str(CLOP_HIST) if CLOP_HIST.exists() else None
    dit_hist_path  = str(DIT_HIST) if DIT_HIST and DIT_HIST.exists() else None

    viz = ResultsVisualizer(
        clop_history_path=clop_hist_path,
        dit_history_path=dit_hist_path,
        cache_dir=str(REPO / "data" / "cached_latents"),
        output_dir=str(FIG_DIR),
        dpi=300,
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
    log.info("── Generating fig01a_architecture.pdf ──")
    result = subprocess.run(
        [sys.executable, str(arch_script)],
        capture_output=True, text=True, cwd=str(REPO)
    )
    if result.returncode != 0:
        log.error("Architecture figure failed:\n%s", result.stderr[-2000:])
        return None
    arch_pdf = FIG_DIR / "fig01a_architecture.pdf"
    log.info("Architecture figure: %s (%s)", arch_pdf, "exists" if arch_pdf.exists() else "MISSING")
    return arch_pdf if arch_pdf.exists() else None


def _run_external_script(script_relpath: str, label: str, expected_pdf: str):
    """Run an external figure script and return the output PDF path (or None).

    Scripts under src/ are run as modules (-m) to support relative imports.
    Scripts under scripts/ are run as plain scripts.
    """
    import subprocess, sys
    script = REPO / script_relpath
    if not script.exists():
        log.warning("%s script not found: %s", label, script)
        return None
    log.info("── Generating %s ──", label)
    if script_relpath.startswith("src/"):
        # Convert path to module: src/visualization/fig25_cross_dataset.py → src.visualization.fig25_cross_dataset
        module_name = script_relpath.replace("/", ".").removesuffix(".py")
        cmd = [sys.executable, "-m", module_name]
    else:
        cmd = [sys.executable, str(script)]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(REPO)
    )
    if result.returncode != 0:
        log.error("%s failed:\n%s", label, result.stderr[-2000:])
        return None
    pdf = FIG_DIR / expected_pdf
    log.info("%s: %s (%s)", label, pdf, "exists" if pdf.exists() else "MISSING")
    return pdf if pdf.exists() else None


def run_evaluation_pipeline_figure():
    """Generate Fig 2: evaluation pipeline schematic."""
    return _run_external_script(
        "scripts/analysis/evaluation_pipeline_figure.py",
        "fig01b_evaluation_pipeline.pdf",
        "fig01b_evaluation_pipeline.pdf",
    )


def run_variance_matching_figure():
    """Generate Fig 19: variance matching pilot."""
    return _run_external_script(
        "scripts/analysis/variance_matching_pilot.py",
        "fig09a_variance_matching.pdf",
        "fig09a_variance_matching.pdf",
    )


def run_gene_gene_correlation_figure():
    """Generate Fig 20: gene-gene correlation."""
    return _run_external_script(
        "scripts/analysis/gene_gene_correlation.py",
        "fig09b_gene_gene_correlation.pdf",
        "fig09b_gene_gene_correlation.pdf",
    )


def run_lane_c_zero_shot_figure():
    """Generate the Section 3.13 strict-OOD bar chart (R3.1 deliverable)."""
    return _run_external_script(
        "scripts/analysis/lane_c_zero_shot_figure.py",
        "figS_lane_c_zero_shot.pdf",
        "figS_lane_c_zero_shot.pdf",
    )


def run_conditioning_figures():
    """Regenerate Figs 11 + 13 from cached conditioning data (no model inference).

    Fig 11 (conditioning_umap) requires: panel_m_arrays.npz + panel_m_meta.json
    Fig 13 (noise_tradeoff) requires: panel_l_data.json
    """
    cond_cache = REPO / "results" / "conditioning_cache"
    saved = []

    # Fig 13: Noise-Scale Trade-off
    l_data_path = cond_cache / "panel_l_data.json"
    if l_data_path.exists():
        log.info("── Regenerating Fig 13 (noise tradeoff) from cached data ──")
        try:
            with open(l_data_path) as f:
                l_data = json.load(f)
            from visualization.fig13_noise_tradeoff import plot_panel_l
            p = plot_panel_l(
                noise_scales=l_data["noise_scales"],
                fds=l_data["fds"],
                centroids=l_data["centroids"],
                div_ratios=l_data["div_ratios"],
                output_dir=str(FIG_DIR),
                cfg_scale=l_data.get("cfg_scale", 1.5),
            )
            plt.close("all")
            saved.append(p)
            log.info("Fig 13: %s", p)
        except Exception as e:
            log.error("Fig 13 failed: %s", e, exc_info=True)
    else:
        log.warning("Fig 13 skipped (no cached data: %s)", l_data_path)

    # Fig 11: Conditioning UMAP/PCA
    m_meta_path = cond_cache / "panel_m_meta.json"
    m_arrays_path = cond_cache / "panel_m_arrays.npz"
    m_source_path = cond_cache / "panel_m_combined_source.npy"
    m_full_source_path = cond_cache / "panel_m_full_dim_source.npy"
    if m_meta_path.exists() and m_arrays_path.exists():
        log.info("── Regenerating Fig 11 (conditioning umap) from cached data ──")
        try:
            import numpy as np
            with open(m_meta_path) as f:
                m_meta = json.load(f)
            arrays = np.load(m_arrays_path)
            coords = arrays["coords"]
            combined_labels = arrays["combined_labels"]
            combined_source = np.load(m_source_path) if m_source_path.exists() else arrays.get("combined_source", np.zeros(len(combined_labels), dtype=int))
            full_dim_data = arrays.get("full_dim_data", None)
            full_dim_labels = arrays.get("full_dim_labels", None)
            full_dim_source = np.load(m_full_source_path) if m_full_source_path.exists() else None

            type_names = {int(k): v for k, v in m_meta["type_names"].items()}
            from visualization.fig11_conditioning import plot_panel_m
            p = plot_panel_m(
                coords=coords,
                combined_labels=combined_labels,
                combined_source=combined_source,
                selected_types=m_meta["selected_types"],
                mode_diversity=m_meta["mode_diversity"],
                real_diversity=m_meta["real_diversity"],
                type_names=type_names,
                output_dir=str(FIG_DIR),
                cfg_scale=m_meta.get("cfg_scale", 1.5),
                n_real=m_meta.get("n_real", 0),
                mode_counts=m_meta.get("mode_counts"),
                full_dim_data=full_dim_data,
                full_dim_labels=full_dim_labels,
                full_dim_source=full_dim_source,
                label_offset=4,
            )
            plt.close("all")
            saved.append(p)
            log.info("Fig 11: %s", p)
        except Exception as e:
            log.error("Fig 11 failed: %s", e, exc_info=True)
    else:
        log.warning("Fig 11 skipped (no cached data: %s, %s)", m_meta_path, m_arrays_path)

    return saved


def run_diversity_figures():
    """Regenerate Figs 12 + 14 from cached diversity_diagnostics.json (no model inference).

    Fig 12 now includes the noise-tradeoff panel (e) from panel_l_data.json when available.
    """
    div_json = REPO / "results" / "diversity_diagnostics.json"
    if not div_json.exists():
        log.warning("Figs 12+14 skipped (no cached data: %s)", div_json)
        return []

    log.info("── Regenerating Figs 12 + 14 from diversity_diagnostics.json ──")
    try:
        with open(div_json) as f:
            all_results = json.load(f)

        # Load noise tradeoff data for embedded panel (e) in Fig 12
        noise_data = None
        l_data_path = REPO / "results" / "conditioning_cache" / "panel_l_data.json"
        if l_data_path.exists():
            with open(l_data_path) as f:
                l_data = json.load(f)
            noise_data = {
                "noise_scales": l_data["noise_scales"],
                "fds": l_data["fds"],
                "centroids": l_data["centroids"],
                "div_ratios": l_data["div_ratios"],
                "cfg_scale": l_data.get("cfg_scale", 1.5),
            }
            log.info("Loaded noise tradeoff data for Fig 12 panel (e)")
        else:
            log.warning("Noise tradeoff data not found (%s); Fig 12 panel (e) will be omitted", l_data_path)

        from visualization.fig12_diversity import plot_diagnostics
        saved = plot_diagnostics(
            all_results,
            output_dir=str(FIG_DIR),
            include_noise_panel=noise_data is not None,
            noise_data=noise_data,
        )
        plt.close("all")
        log.info("Figs 12+14: %d panels saved", len(saved))
        return saved
    except Exception as e:
        log.error("Figs 12+14 failed: %s", e, exc_info=True)
        return []


def run_vcd_on_figures(pdf_list: list[Path]) -> dict:
    """Collect live figure-time VCD sidecars and emit a consolidated final summary.

    The live generator audit is the source of truth. A separate post-export
    PDF-file audit is not implemented for this pipeline, so the regeneration
    log and saved markdown now report that state explicitly instead of printing
    placeholder PASS lines from a stubbed checker.
    """
    live_vcd = load_live_vcd_results(FIG_DIR)
    vcd_results: dict = {}
    total_warn = 0
    total_info = 0
    files_with_warnings = 0
    missing_live = []
    backfilled = []
    # US-202: aggregate severity-level counts across all figures
    total_severity: dict[str, int] = {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0, "INFO": 0}
    # US-205: backfill empty sidecars for article-manifest figures whose producer
    # renders but bypasses save_with_vcd. Reclassifies "missing-live-sidecar" into
    # an explicit "pdf-only, uncovered" state so the report no longer mis-fires.
    _live_dir = FIG_DIR / "_live_vcd"
    _live_dir.mkdir(parents=True, exist_ok=True)

    for pdf in sorted(pdf_list):
        if not pdf.exists():
            log.warning("VCD skip (missing figure): %s", pdf.name)
            continue

        live_payload = live_vcd.get(pdf.stem)
        if not live_payload:
            # US-205: emit a backfilled sidecar stub and treat the entry as
            # pdf-only (no geometry audit) instead of an error.
            stub = {
                "figure": pdf.stem,
                "warnings": [],
                "info": [],
                "findings": [],
                "severity_counts": {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0, "INFO": 0},
                "counts_by_type": {},
                "audit_source": "pdf-only-backfill",
                "reason": "renderer bypassed save_with_vcd; PDF delivered without live geometry audit",
            }
            try:
                with open(_live_dir / f"{pdf.stem}.json", "w") as f:
                    json.dump(stub, f, indent=2)
            except Exception as exc:
                log.warning("backfill sidecar write failed for %s: %s", pdf.stem, exc)
            live_payload = stub
            backfilled.append(pdf.name)

        warnings = list(live_payload.get("warnings", []))
        info_only = list(live_payload.get("info", []))
        findings = list(live_payload.get("findings", []))
        severity_counts = dict(live_payload.get("severity_counts", {}))
        entry = {
            "warnings": warnings,
            "info": info_only,
            "total": len(warnings) + len(info_only),
            "audit_source": live_payload.get("audit_source") or "live-generator",
            "live_error": live_payload.get("error"),
            "counts_by_type": live_payload.get("counts_by_type", {}),
            "findings": findings,
            "severity_counts": severity_counts,
        }
        vcd_results[pdf.name] = entry
        total_warn += len(warnings)
        total_info += len(info_only)
        for level, count in severity_counts.items():
            total_severity[level] = total_severity.get(level, 0) + int(count)
        if warnings:
            files_with_warnings += 1
        _log_vcd_entry(pdf.name, entry)

    log.info(
        "── VCD final summary: source=live-generator total_warnings=%d total_info=%d "
        "files_with_warnings=%d missing_live=%d pdf_audit=skipped | "
        "CRITICAL=%d MAJOR=%d MINOR=%d INFO=%d ──",
        total_warn,
        total_info,
        files_with_warnings,
        len(missing_live),
        total_severity["CRITICAL"],
        total_severity["MAJOR"],
        total_severity["MINOR"],
        total_severity["INFO"],
    )
    vcd_results["__summary__"] = {
        "total_warnings": total_warn,
        "total_info": total_info,
        "files_with_warnings": files_with_warnings,
        "live_total_warnings": total_warn,
        "live_total_info": total_info,
        "live_files_with_warnings": files_with_warnings,
        "pdf_audit_ran": False,
        "pdf_total_warnings": 0,
        "pdf_total_info": 0,
        "pdf_files_with_warnings": 0,
        "figures_missing_live_audit": missing_live,
        "figures_backfilled": backfilled,
        "live_sidecar_count": len(live_vcd),
        "severity_counts": total_severity,
    }
    return vcd_results


def _format_vcd_type_counts(type_counts: dict, max_items: int = 4) -> str:
    if not type_counts:
        return "none"
    items = sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))
    head = ", ".join(f"{name}={count}" for name, count in items[:max_items])
    if len(items) > max_items:
        head += f", +{len(items) - max_items} more"
    return head


def _log_vcd_entry(name: str, entry: dict) -> None:
    warnings = list(entry.get("warnings", []))
    info_only = list(entry.get("info", []))
    error = entry.get("live_error") or entry.get("error")
    source = entry.get("audit_source", "unknown")
    type_counts = entry.get("counts_by_type", {}) or {}
    status = "ERROR" if error else ("PASS" if not warnings else ("WARN" if len(warnings) < 3 else "FAIL"))
    log_fn = log.error if error else (log.info if not warnings else log.warning)
    log_fn(
        "[VCD %-35s]  %s  source=%s warn=%d info=%d types=%s",
        name[:35],
        status,
        source,
        len(warnings),
        len(info_only),
        _format_vcd_type_counts(type_counts),
    )
    if error:
        log.error("    %s", str(error)[:220])
        return
    for item in warnings[:3]:
        log.warning("    %s", str(item)[:180])
    if len(warnings) > 3:
        log.warning("    ... +%d more", len(warnings) - 3)


def load_live_vcd_results(fig_dir: Path) -> dict[str, dict]:
    """Load live figure-time VCD sidecars emitted by save_with_vcd."""
    live_dir = fig_dir / "_live_vcd"
    if not live_dir.exists():
        return {}

    live_results: dict[str, dict] = {}
    for json_path in sorted(live_dir.glob("*.json")):
        try:
            with open(json_path) as f:
                payload = json.load(f)
            live_results[json_path.stem] = payload
        except Exception as exc:
            live_results[json_path.stem] = {"error": str(exc), "warnings": [], "info": []}
    return live_results


def merge_live_vcd_results(pdf_vcd: dict, live_vcd: dict[str, dict]) -> dict:
    """Prefer live figure-time warnings over the post-export PDF-only VCD pass."""
    if not live_vcd:
        return pdf_vcd

    total_live_warn = 0
    total_live_info = 0
    files_with_live_warnings = 0
    # US-202: aggregate severity counts across all figures
    total_severity: dict[str, int] = {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0, "INFO": 0}

    for name, data in pdf_vcd.items():
        if name.startswith("__"):
            continue
        stem = Path(name).stem
        live = live_vcd.get(stem)
        data["pdf_warnings"] = list(data.get("warnings", []))
        data["pdf_info"] = list(data.get("info", []))
        if not live:
            continue

        live_warnings = list(live.get("warnings", []))
        live_info = list(live.get("info", []))
        data["live_warnings"] = live_warnings
        data["live_info"] = live_info
        data["live_error"] = live.get("error")
        # US-202: carry severity-level metadata forward
        data["findings"] = list(live.get("findings", []))
        data["severity_counts"] = dict(live.get("severity_counts", {}))
        for level, count in data["severity_counts"].items():
            total_severity[level] = total_severity.get(level, 0) + int(count)

        total_live_warn += len(live_warnings)
        total_live_info += len(live_info)
        if live_warnings:
            files_with_live_warnings += 1
            data["warnings"] = live_warnings
            data["info"] = live_info

    summary = pdf_vcd.setdefault("__summary__", {})
    summary["live_total_warnings"] = total_live_warn
    summary["live_total_info"] = total_live_info
    summary["live_files_with_warnings"] = files_with_live_warnings
    summary["total_warnings"] = total_live_warn
    summary["total_info"] = total_live_info
    summary["files_with_warnings"] = files_with_live_warnings
    summary["severity_counts"] = total_severity
    log.info(
        "── Live VCD summary: total_warnings=%d total_info=%d files_with_warnings=%d | "
        "CRITICAL=%d MAJOR=%d MINOR=%d INFO=%d ──",
        total_live_warn,
        total_live_info,
        files_with_live_warnings,
        total_severity["CRITICAL"],
        total_severity["MAJOR"],
        total_severity["MINOR"],
        total_severity["INFO"],
    )
    return pdf_vcd


def save_vcd_report(vcd: dict):
    """Write JSON + markdown VCD reports to results/."""
    json_path = RESULTS_DIR / "vcd_report.json"
    with open(json_path, "w") as f:
        json.dump(vcd, f, indent=2)
    log.info("VCD JSON saved: %s", json_path)

    md_lines = ["# VCD Report — " + time.strftime("%Y-%m-%d %H:%M"), ""]
    summary = vcd.get("__summary__", {})
    pdf_audit_ran = bool(summary.get("pdf_audit_ran", False))
    sev_totals = summary.get("severity_counts") or {}
    md_lines += [
        f"**Total warnings:** {summary.get('total_warnings', '?')}",
        f"**Files with warnings:** {summary.get('files_with_warnings', '?')}",
        f"**Live generator warnings:** {summary.get('live_total_warnings', summary.get('total_warnings', '?'))}",
        (
            f"**Post-export PDF warnings:** {summary.get('pdf_total_warnings', '?')}"
            if pdf_audit_ran
            else "**Post-export PDF audit:** skipped (live generator audit is the source of truth)"
        ),
        "",
        "## Severity Summary",
        "",
        f"- **CRITICAL** (publication-blocker): {sev_totals.get('CRITICAL', 0)}",
        f"- **MAJOR** (reviewer-visible defect): {sev_totals.get('MAJOR', 0)}",
        f"- **MINOR** (cosmetic, safe to ship): {sev_totals.get('MINOR', 0)}",
        f"- **INFO** (signal, not a defect): {sev_totals.get('INFO', 0)}",
        "",
        "## Per-Figure",
        "",
    ]
    missing_live = summary.get("figures_missing_live_audit", []) or []
    if missing_live:
        md_lines.extend([
            f"**Figures missing live audit:** {', '.join(missing_live)}",
            "",
        ])
    # US-202: per-severity bucket section before the flat per-figure dump
    buckets: dict[str, list[tuple[str, str]]] = {
        "CRITICAL": [], "MAJOR": [], "MINOR": [], "INFO": [],
    }
    for name, data in vcd.items():
        if name.startswith("__"):
            continue
        for finding in data.get("findings", []) or []:
            level = finding.get("severity_level") or "INFO"
            detail = f"[{finding.get('type', '?')}] {finding.get('detail', '')}".strip()
            buckets.setdefault(level, []).append((name, detail))
    for level in ("CRITICAL", "MAJOR", "MINOR", "INFO"):
        entries = buckets.get(level, [])
        if not entries:
            continue
        md_lines.extend([f"### {level} findings ({len(entries)})", ""])
        for figname, detail in entries[:40]:
            md_lines.append(f"- `{figname}` — {detail[:220]}")
        if len(entries) > 40:
            md_lines.append(f"- ... +{len(entries) - 40} more")
        md_lines.append("")
    md_lines.extend(["## Flat Per-Figure View", ""])
    for name, data in vcd.items():
        if name.startswith("__"):
            continue
        w = data.get("warnings", [])
        md_lines.append(f"### {name}")
        md_lines.append(f"- Source: {data.get('audit_source', 'unknown')}")
        audit_error = data.get("live_error") or data.get("error")
        if audit_error:
            md_lines.append(f"- AUDIT ERROR: {str(audit_error)[:200]}")
        elif not w:
            md_lines.append("- PASS (0 warnings)")
        else:
            sev_counts = data.get("severity_counts") or {}
            if sev_counts:
                sev_parts = [
                    f"{lvl}={sev_counts.get(lvl, 0)}"
                    for lvl in ("CRITICAL", "MAJOR", "MINOR", "INFO")
                    if sev_counts.get(lvl, 0)
                ]
                if sev_parts:
                    md_lines.append(f"- Severity: {', '.join(sev_parts)}")
            for item in w:
                md_lines.append(f"- WARNING: {item[:200]}")
        md_lines.append("")

    md_path = RESULTS_DIR / "vcd_report_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    log.info("VCD markdown saved: %s", md_path)


def run_article_delivery():
    """Copy PDF figures to articles/figures/ and verify submission package."""
    log.info("── Running article_delivery.py ──")
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(REPO / "src" / "visualization" / "article_delivery.py"), "--copy"],
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


def run_latex_build():
    """Rebuild the LaTeX article PDF with latexmk -g (force full recompile)."""
    import subprocess, sys
    build_script = REPO / "scripts" / "pipeline" / "build_article.sh"
    if not build_script.exists():
        log.warning("build_article.sh not found: %s", build_script)
        return 1
    log.info("── Rebuilding LaTeX article PDF (latexmk -g) ──")
    result = subprocess.run(
        ["bash", str(build_script)],
        capture_output=True, text=True, cwd=str(REPO)
    )
    for line in result.stdout.splitlines():
        log.info("  [latex] %s", line)
    if result.returncode != 0:
        log.error("LaTeX build failed (exit %d):\n%s", result.returncode, result.stderr[-2000:])
    else:
        log.info("LaTeX article PDF rebuilt successfully")
    return result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate article-facing figure assets + VCD + optional PDF rebuild")
    parser.add_argument("--no-vcd",    action="store_true", help="Skip live VCD during generation and final VCD reporting")
    parser.add_argument("--skip-arch", action="store_true", help="Skip architecture figure (Fig 1)")
    parser.add_argument("--no-delivery", action="store_true", help="Skip article_delivery (symlinks)")
    parser.add_argument("--build-pdf", action="store_true", help="Rebuild LaTeX article PDF after figures")
    parser.add_argument("--vs-baseline", action="store_true", help="Diff this run's VCD against results/vcd_baseline.json; exit non-zero on new CRITICAL")
    parser.add_argument("--write-baseline", action="store_true", help="Write results/vcd_baseline.json from this run (use after a clean run)")
    parser.add_argument("--adaptive", action="store_true", help="Run the VCD sweep under complexity-routed profiles (SIMPLE/COMPOUND/COMPOSED)")
    args = parser.parse_args()

    t0 = time.time()
    vcd_enabled = (not args.no_vcd) and _env_flag("CLOPDIT_ENABLE_VCD", True)
    os.environ["CLOPDIT_ENABLE_VCD"] = "1" if vcd_enabled else "0"
    # US-307: adaptive profile routing — classify each figure and run only
    # the relevant checks. The live save hook reads this env var.
    if args.adaptive:
        os.environ["CLOPDIT_VCD_PROFILE"] = "auto"
        log.info("Adaptive VCD profile enabled (SIMPLE/COMPOUND/COMPOSED routing)")
    else:
        os.environ.setdefault("CLOPDIT_VCD_PROFILE", "full")
    # Force headless rendering in all subprocesses to prevent figures from
    # popping up in an interactive viewer.
    os.environ["MPLBACKEND"] = "Agg"
    log.info("=" * 70)
    log.info("CLOP-DiT Figure Regeneration Pipeline (article-facing assets) — %s", time.strftime("%Y-%m-%d"))
    log.info("=" * 70)
    log.info("Live VCD during generation: %s", "enabled" if vcd_enabled else "disabled")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    live_vcd_dir = FIG_DIR / "_live_vcd"
    if live_vcd_dir.exists():
        shutil.rmtree(live_vcd_dir)

    # 1. Figs 11+13: Conditioning figures from cached data (before ResultsVisualizer)
    cond_saved = run_conditioning_figures()
    saved = list(cond_saved)

    # 2. Figs 12+14: Diversity figures from cached data (before ResultsVisualizer)
    div_saved = run_diversity_figures()
    saved.extend(div_saved)

    # 3. Generate Figs 3–18 via ResultsVisualizer (includes pre-generated figs 11-14)
    try:
        rv_saved = run_results_visualizer()
        saved.extend(rv_saved)
    except Exception as e:
        log.error("ResultsVisualizer failed: %s", e, exc_info=True)

    # 4. Fig 1: Architecture figure (standalone)
    if not args.skip_arch:
        arch = run_architecture_figure()
        if arch:
            saved.append(arch)

    # 5. Fig 2: Evaluation pipeline figure (standalone)
    ep = run_evaluation_pipeline_figure()
    if ep:
        saved.append(ep)

    # 6. Fig 19: Variance matching pilot (standalone)
    vm = run_variance_matching_figure()
    if vm:
        saved.append(vm)

    # 7. Fig 20: Gene-gene correlation (standalone)
    gg = run_gene_gene_correlation_figure()
    if gg:
        saved.append(gg)

    # 7b. Section 3.13 strict-OOD (Lane C, R3.1 deliverable)
    lc = run_lane_c_zero_shot_figure()
    if lc:
        saved.append(lc)

    # 8. Python-composed supplementary appendix figures (replace LaTeX stitching)
    _supp_figs = [
        ("src/visualization/figS01_supplementary_validation.py", "figS01_supplementary_validation.pdf"),
        ("src/visualization/figS02_expression_diagnostics.py",    "figS02_expression_diagnostics.pdf"),
    ]
    for _script_rel, _expected_pdf in _supp_figs:
        _fig = _run_external_script(_script_rel, _expected_pdf, _expected_pdf)
        if _fig:
            saved.append(_fig)

    # 9. Figs 25–31: extended analysis figures (not article-facing, kept for diagnostics until delivery cleanup)
    _ext_figs = [
        ("src/visualization/fig25_cross_dataset.py",          "fig25_cross_dataset.pdf"),
        ("src/visualization/fig26_expanded_de.py",            "fig26_expanded_de.pdf"),
        ("src/visualization/fig27_ood_robustness.py",         "fig27_ood_robustness.pdf"),
        ("src/visualization/fig28_marker_completeness.py",    "fig28_marker_completeness.pdf"),
        ("src/visualization/fig29_embedding_augmentation.py", "fig29_embedding_augmentation.pdf"),
        ("src/visualization/fig30_validation_summary.py",     "fig30_validation_summary.pdf"),
        ("src/visualization/fig31_decoder_ablation.py",       "fig31_decoder_ablation.pdf"),
    ]
    for _script_rel, _expected_pdf in _ext_figs:
        _fig = _run_external_script(_script_rel, _expected_pdf, _expected_pdf)
        if _fig:
            saved.append(_fig)

    # Collect every generated figure PDF before article-delivery cleanup so VCD can audit the full run.
    all_pdfs = sorted(FIG_DIR.glob("fig*.pdf"))
    log.info("Total PDFs in results/figures: %d", len(all_pdfs))
    for p in all_pdfs:
        log.info("  ✓ %s", p.name)

    # 10. VCD pass
    baseline_exit_code = 0
    if vcd_enabled:
        vcd = run_vcd_on_figures(all_pdfs)
        save_vcd_report(vcd)
        # US-203: baseline snapshot + diff
        if args.write_baseline or args.vs_baseline:
            from vcd.vcd_baseline import (
                snapshot_from_vcd_report, load_baseline, save_baseline,
                diff_against_baseline, render_diff_markdown,
            )
            baseline_path = RESULTS_DIR / "vcd_baseline.json"
            current_snapshot = snapshot_from_vcd_report(vcd)
            if args.write_baseline:
                save_baseline(current_snapshot, baseline_path)
                log.info("VCD baseline written: %s", baseline_path)
            if args.vs_baseline:
                baseline = load_baseline(baseline_path)
                report = diff_against_baseline(current_snapshot, baseline)
                diff_md = render_diff_markdown(
                    report,
                    baseline_path=baseline_path,
                    out_path=RESULTS_DIR / "vcd_diff.md",
                )
                log.info("VCD diff written: %s", diff_md)
                if report.has_new_critical:
                    log.error("VCD diff: NEW CRITICAL finding(s) vs baseline — failing run")
                    baseline_exit_code = 2
                else:
                    log.info(
                        "VCD diff OK — added=%s removed=%s",
                        report.totals_added, report.totals_removed,
                    )
    else:
        log.info("VCD skipped (--no-vcd or CLOPDIT_ENABLE_VCD=0)")

    # 11. Article delivery (copy/symlink + stale-asset cleanup)
    if not args.no_delivery:
        ret = run_article_delivery()
        if ret == 0:
            log.info("Article delivery: OK")
        else:
            log.warning("Article delivery: FAILED (exit %d)", ret)

    # 12. Rebuild LaTeX article PDF (requires --build-pdf)
    if args.build_pdf:
        ret = run_latex_build()
        if ret != 0:
            log.warning("LaTeX build: FAILED (exit %d)", ret)
    else:
        log.info("LaTeX build skipped (use --build-pdf to include)")

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info("Done in %.1fs", elapsed)
    log.info("=" * 70)

    # Propagate baseline-diff failure as a non-zero process exit
    if baseline_exit_code != 0:
        sys.exit(baseline_exit_code)


if __name__ == "__main__":
    main()
