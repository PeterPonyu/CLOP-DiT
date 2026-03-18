#!/usr/bin/env python3
"""Per-type metric dashboard.

Aggregates all available per-type metrics (embedding quality, diversity,
gene-gene correlation) into a single sortable table and renders an HTML
dashboard and a compact PDF version.

Outputs:
    results/per_type_dashboard.html   — Interactive HTML table
    results/per_type_dashboard.json   — Aggregated JSON
    results/figures/per_type_heatmap.png/.pdf  — Heatmap summary figure

Usage:
    python scripts/analysis/per_type_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.visualization.style import apply_style, FONT_TITLE, FONT_LABEL, FONT_SMALL, save_with_vcd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── data loading ─────────────────────────────────────────────────────


def _load_generation_metrics() -> dict:
    """Per-type from generation_metrics.json."""
    path = RESULTS / "generation_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("per_type", {})


def _load_diversity_diagnostics() -> dict:
    """Per-type from diversity_diagnostics.json."""
    path = RESULTS / "diversity_diagnostics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("test1_intratype_diversity", {}).get("per_type", {})


def _load_gene_gene_correlation() -> dict:
    """Per-type from gene_gene_correlation.json."""
    path = RESULTS / "gene_gene_correlation.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("per_type", {})


# ── aggregation ──────────────────────────────────────────────────────


def aggregate_per_type() -> list[dict]:
    """Merge all per-type metrics into a single table (list of dicts)."""
    gen_metrics = _load_generation_metrics()
    div_metrics = _load_diversity_diagnostics()
    corr_metrics = _load_gene_gene_correlation()

    # Collect all type keys
    all_types = set()
    for d in [gen_metrics, div_metrics]:
        all_types.update(d.keys())

    # Map type_id → name from generation_metrics (names as keys there)
    # corr_metrics uses numeric string IDs — we'll map via type_id
    type_id_to_name = {}
    for name, info in gen_metrics.items():
        tid = str(info.get("type_id", ""))
        type_id_to_name[tid] = name

    rows = []
    for type_name in sorted(all_types):
        row = {"cell_type": type_name}

        # Generation metrics
        gm = gen_metrics.get(type_name, {})
        row["n_real"] = gm.get("n_real", 0)
        row["n_gen"] = gm.get("n_gen", 0)
        row["centroid_cosine"] = gm.get("centroid_cosine")
        row["frechet_distance"] = gm.get("frechet_distance")

        # Diversity
        dm = div_metrics.get(type_name, {})
        row["diversity_ratio"] = dm.get("diversity_ratio")
        rr = dm.get("real_real_cos", {})
        gg = dm.get("gen_gen_cos", {})
        row["real_intra_cos"] = rr.get("mean")
        row["gen_intra_cos"] = gg.get("mean")

        # Gene-gene correlation
        tid = str(gm.get("type_id", ""))
        cm = corr_metrics.get(tid, {})
        row["mantel_r"] = cm.get("mantel_r")
        row["corr_rmse"] = cm.get("rmse")

        rows.append(row)

    return rows


# ── output formatters ────────────────────────────────────────────────


METRIC_COLS = [
    ("cell_type", "Cell Type", "s"),
    ("n_real", "N (real)", "d"),
    ("n_gen", "N (gen)", "d"),
    ("centroid_cosine", "Centroid cos", ".4f"),
    ("frechet_distance", "FD", ".4f"),
    ("diversity_ratio", "Div. ratio", ".4f"),
    ("real_intra_cos", "Real intra cos", ".4f"),
    ("gen_intra_cos", "Gen intra cos", ".4f"),
    ("mantel_r", "Mantel r", ".4f"),
    ("corr_rmse", "Corr RMSE", ".6f"),
]


def render_html(rows: list[dict], path: Path):
    """Write a sortable HTML dashboard."""
    lines = [
        "<!DOCTYPE html>",
        '<html><head><meta charset="utf-8">',
        "<title>CLOP-DiT Per-Type Metric Dashboard</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { color: #333; }",
        "table { border-collapse: collapse; width: 100%; font-size: 13px; }",
        "th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; }",
        "th { background: #f5f5f5; cursor: pointer; user-select: none; }",
        "th:first-child, td:first-child { text-align: left; }",
        "tr:nth-child(even) { background: #fafafa; }",
        "tr:hover { background: #e8f0fe; }",
        ".good { color: #2e7d32; } .bad { color: #c62828; }",
        "</style>",
        "</head><body>",
        "<h1>CLOP-DiT Per-Type Metric Dashboard</h1>",
        f"<p>{len(rows)} cell types | Generated by per_type_dashboard.py</p>",
        "<table id='t'>",
        "<thead><tr>",
    ]

    for _, header, _ in METRIC_COLS:
        lines.append(f"  <th onclick='sortTable(this)'>{escape(header)}</th>")
    lines.append("</tr></thead><tbody>")

    for row in rows:
        lines.append("<tr>")
        for key, _, fmt in METRIC_COLS:
            val = row.get(key)
            if val is None:
                lines.append("  <td>—</td>")
            elif fmt == "s":
                lines.append(f"  <td>{escape(str(val))}</td>")
            elif fmt == "d":
                lines.append(f"  <td>{int(val)}</td>")
            else:
                lines.append(f"  <td>{val:{fmt}}</td>")
        lines.append("</tr>")

    lines.append("</tbody></table>")

    # Minimal JS for column sorting
    lines.append("<script>")
    lines.append("""
function sortTable(th) {
  var table = document.getElementById('t');
  var rows = Array.from(table.tBodies[0].rows);
  var idx = Array.from(th.parentNode.children).indexOf(th);
  var asc = th.dataset.asc !== '1'; th.dataset.asc = asc ? '1' : '0';
  rows.sort(function(a, b) {
    var va = a.cells[idx].textContent.trim();
    var vb = b.cells[idx].textContent.trim();
    var na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  });
  rows.forEach(function(r) { table.tBodies[0].appendChild(r); });
}
""")
    lines.append("</script></body></html>")

    path.write_text("\n".join(lines))
    print(f"HTML dashboard: {path}")


def make_heatmap(rows: list[dict]):
    """Heatmap of key metrics across cell types."""
    apply_style()

    metric_keys = ["centroid_cosine", "frechet_distance", "diversity_ratio", "mantel_r"]
    metric_labels = ["Centroid cos", "FD", "Div. ratio", "Mantel r"]

    # Filter to types with at least one available metric
    valid = [r for r in rows if any(r.get(k) is not None for k in metric_keys)]
    if not valid:
        print("No data for heatmap.")
        return

    # Sort by centroid cosine (descending)
    valid.sort(key=lambda r: r.get("centroid_cosine") or 0, reverse=True)

    names = [r["cell_type"][:30] for r in valid]
    matrix = np.full((len(valid), len(metric_keys)), np.nan)
    for i, r in enumerate(valid):
        for j, k in enumerate(metric_keys):
            v = r.get(k)
            if v is not None:
                matrix[i, j] = v

    # Z-score normalise each column for colour mapping
    col_mean = np.nanmean(matrix, axis=0)
    col_std = np.nanstd(matrix, axis=0)
    col_std[col_std == 0] = 1.0
    z = (matrix - col_mean) / col_std

    # Limit display to top/bottom 20 types
    n_show = min(40, len(valid))
    if len(valid) > n_show:
        idx = list(range(20)) + list(range(len(valid) - 20, len(valid)))
        z = z[idx]
        names = [names[i] for i in idx]

    fig_height = max(4, 0.25 * len(names))
    fig, ax = plt.subplots(figsize=(6, fig_height))
    im = ax.imshow(z, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=FONT_SMALL, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=max(FONT_SMALL - 2, 5))
    ax.set_title("Per-type metric summary (z-scored)", fontsize=FONT_TITLE)
    plt.colorbar(im, ax=ax, shrink=0.5, label="z-score")
    fig.tight_layout()

    out = str(FIG_DIR / "per_type_heatmap")
    save_with_vcd(fig, out)
    plt.close(fig)
    print(f"Heatmap saved: {out}.png / .pdf")


def main():
    print("Aggregating per-type metrics...")
    rows = aggregate_per_type()
    print(f"  {len(rows)} cell types found")

    # Save JSON
    out_json = RESULTS / "per_type_dashboard.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"JSON: {out_json}")

    # Save HTML
    render_html(rows, RESULTS / "per_type_dashboard.html")

    # Print compact table (top 15 by centroid cosine)
    rows_sorted = sorted(rows, key=lambda r: r.get("centroid_cosine") or 0, reverse=True)
    print(f"\n{'Cell Type':<35} {'Cos':>6} {'FD':>8} {'Div':>6} {'Mantel':>7}")
    print("-" * 65)
    for r in rows_sorted[:15]:
        ct = r["cell_type"][:34]
        cos = f"{r['centroid_cosine']:.4f}" if r.get("centroid_cosine") is not None else "—"
        fd = f"{r['frechet_distance']:.4f}" if r.get("frechet_distance") is not None else "—"
        div = f"{r['diversity_ratio']:.4f}" if r.get("diversity_ratio") is not None else "—"
        mnt = f"{r['mantel_r']:.4f}" if r.get("mantel_r") is not None else "—"
        print(f"{ct:<35} {cos:>6} {fd:>8} {div:>6} {mnt:>7}")
    if len(rows_sorted) > 15:
        print(f"  ... ({len(rows_sorted) - 15} more types)")

    # Heatmap figure
    make_heatmap(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
