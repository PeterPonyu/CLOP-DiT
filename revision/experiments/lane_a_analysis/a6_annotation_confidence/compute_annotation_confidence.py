"""A6 — Per-type annotation confidence audit.

Aggregates Jaccard-style `confidence` values already stored in
`data/processed_h5ad/subcluster_metadata.json` so we can report, for
each of the canonical cell types used in evaluation, how confident the
marker-signature matcher was on the clusters that received that label.

Key statistics per cell type:
  - n_clusters            : how many Leiden clusters were annotated
  - n_cells               : total cells under this label
  - median / IQR / min    : Jaccard confidence distribution
  - frac_weak             : share with confidence < 0.30
  - n_datasets            : how many GSE datasets contributed
  - label_source_counts   : how many "signature" vs "marker_only"

Output: `annotation_confidence.json` and a short human-readable summary.

Read-only. No modification to upstream pipeline.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
META = REPO_ROOT / "data" / "processed_h5ad" / "subcluster_metadata.json"


def iqr(xs: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return (xs[0] if xs else 0.0, xs[0] if xs else 0.0)
    s = sorted(xs)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4]
    return q1, q3


def main() -> None:
    d = json.loads(META.read_text())
    per_type: dict[str, dict] = defaultdict(lambda: {
        "confidences": [],
        "n_cells": 0,
        "datasets": set(),
        "label_sources": defaultdict(int),
    })
    per_dataset: dict[str, dict] = {}
    all_conf: list[float] = []

    for ds_id, ds in d.items():
        ds_confs: list[float] = []
        unknown_clusters = 0
        for cid, c in ds.get("clusters", {}).items():
            ct = c.get("cell_type", "Unknown")
            conf = float(c.get("confidence", 0.0))
            n = int(c.get("n_cells", 0))
            label_source = c.get("label_source", "unknown")
            t = per_type[ct]
            t["confidences"].append(conf)
            t["n_cells"] += n
            t["datasets"].add(ds_id)
            t["label_sources"][label_source] += 1
            ds_confs.append(conf)
            all_conf.append(conf)
            if ct == "Unknown":
                unknown_clusters += 1
        if ds_confs:
            per_dataset[ds_id] = {
                "n_clusters": len(ds_confs),
                "n_unknown_clusters": unknown_clusters,
                "median_confidence": round(statistics.median(ds_confs), 3),
                "min_confidence": round(min(ds_confs), 3),
                "max_confidence": round(max(ds_confs), 3),
            }

    out_per_type = {}
    for ct, t in per_type.items():
        confs = t["confidences"]
        q1, q3 = iqr(confs)
        out_per_type[ct] = {
            "n_clusters": len(confs),
            "n_cells": t["n_cells"],
            "n_datasets": len(t["datasets"]),
            "median_confidence": round(statistics.median(confs), 3),
            "iqr": [round(q1, 3), round(q3, 3)],
            "min_confidence": round(min(confs), 3),
            "max_confidence": round(max(confs), 3),
            "frac_weak_lt_0.30": round(
                sum(1 for c in confs if c < 0.30) / len(confs), 3
            ),
            "frac_low_lt_0.15": round(
                sum(1 for c in confs if c < 0.15) / len(confs), 3
            ),
            "label_sources": dict(t["label_sources"]),
        }

    # Overall distribution
    n = len(all_conf)
    q1_all, q3_all = iqr(all_conf)
    overall = {
        "n_clusters_total": n,
        "n_datasets": len(per_dataset),
        "median_confidence": round(statistics.median(all_conf), 3),
        "iqr": [round(q1_all, 3), round(q3_all, 3)],
        "min_confidence": round(min(all_conf), 3),
        "max_confidence": round(max(all_conf), 3),
        "frac_weak_lt_0.30": round(
            sum(1 for c in all_conf if c < 0.30) / n, 3
        ),
        "frac_low_lt_0.15": round(
            sum(1 for c in all_conf if c < 0.15) / n, 3
        ),
        "n_unique_cell_types": len(out_per_type),
    }

    summary = {
        "source": str(META.relative_to(REPO_ROOT)),
        "overall": overall,
        "per_cell_type": dict(sorted(
            out_per_type.items(),
            key=lambda kv: kv[1]["median_confidence"],
        )),
        "per_dataset": per_dataset,
    }

    (HERE / "annotation_confidence.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    # Short human-readable preview
    lines = ["A6 — annotation confidence audit",
             "================================",
             f"datasets={overall['n_datasets']}  "
             f"clusters={overall['n_clusters_total']}  "
             f"types={overall['n_unique_cell_types']}",
             f"overall median={overall['median_confidence']}  "
             f"IQR={overall['iqr']}  "
             f"frac<0.30={overall['frac_weak_lt_0.30']}  "
             f"frac<0.15={overall['frac_low_lt_0.15']}",
             "",
             "Bottom 15 types by median Jaccard confidence:",
             f"  {'cell_type':<42s} {'n_clu':>5s} {'n_cells':>7s} "
             f"{'median':>6s} {'IQR':>14s} {'weak<.30':>9s}"]
    bottom = list(summary["per_cell_type"].items())[:15]
    for ct, s in bottom:
        iqr_str = f"[{s['iqr'][0]:.2f},{s['iqr'][1]:.2f}]"
        lines.append(
            f"  {ct[:42]:<42s} {s['n_clusters']:>5d} {s['n_cells']:>7d} "
            f"{s['median_confidence']:>6.3f} {iqr_str:>14s} "
            f"{s['frac_weak_lt_0.30']:>9.2f}"
        )
    lines.append("")
    lines.append("Top 5 types by median Jaccard confidence:")
    top = list(summary["per_cell_type"].items())[-5:][::-1]
    for ct, s in top:
        iqr_str = f"[{s['iqr'][0]:.2f},{s['iqr'][1]:.2f}]"
        lines.append(
            f"  {ct[:42]:<42s} {s['n_clusters']:>5d} {s['n_cells']:>7d} "
            f"{s['median_confidence']:>6.3f} {iqr_str:>14s} "
            f"{s['frac_weak_lt_0.30']:>9.2f}"
        )
    preview = "\n".join(lines) + "\n"
    (HERE / "annotation_confidence_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
