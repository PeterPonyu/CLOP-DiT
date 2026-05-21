#!/usr/bin/env python3
"""Build a lightweight strict-OOD candidate inventory for Lane C.

This script does not curate new data. It only audits the current baseline
metadata text for obvious target-tissue overlap and combines that with the
candidate-source notes already documented in Lane C planning docs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = REPO_ROOT / "data" / "processed_h5ad" / "metadata_structured.json"
OUTPUT_CSV = REPO_ROOT / "revision" / "experiments" / "lane_c_data" / "strict_ood_candidate_inventory.csv"
OUTPUT_MD = REPO_ROOT / "revision" / "experiments" / "lane_c_data" / "strict_ood_candidate_inventory.md"


TARGETS = [
    {
        "target_tissue": "kidney",
        "aliases": ["kidney", "renal", "nephron", "proximal tubule", "collecting duct"],
        "candidate_source": "MULTI_TISSUE_PILOT + feasibility note",
        "candidate_label": "GSE131685 / GSE140989 / HCA kidney v2",
    },
    {
        "target_tissue": "testis",
        "aliases": ["testis", "testicular"],
        "candidate_source": "feasibility note",
        "candidate_label": "GTEx testis snRNA-seq",
    },
    {
        "target_tissue": "intestine",
        "aliases": ["intestine", "intestinal", "gut", "ileum", "colon", "duodenum"],
        "candidate_source": "feasibility note",
        "candidate_label": "HCA gut v2",
    },
    {
        "target_tissue": "cerebellum",
        "aliases": ["cerebellum", "cerebellar"],
        "candidate_source": "feasibility note",
        "candidate_label": "Allen cerebellum snRNA-seq",
    },
    {
        "target_tissue": "distal airway",
        "aliases": ["distal airway", "airway", "airway progenitor", "bronchiolar"],
        "candidate_source": "feasibility note",
        "candidate_label": "HCA lung upper airway",
    },
    {
        "target_tissue": "merkel-like",
        "aliases": ["merkel", "merkel cell carcinoma", "mcc", "neuroendocrine skin"],
        "candidate_source": "feasibility note",
        "candidate_label": "NCBI GEO Merkel-cell carcinoma scRNA-seq",
    },
]


def load_metadata() -> dict[str, dict]:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def overlap_examples(text_map: dict[str, dict], aliases: list[str]) -> list[str]:
    hits: list[str] = []
    lowered_aliases = [alias.lower() for alias in aliases]
    for dataset_id, payload in text_map.items():
        text = str(payload.get("text", "")).lower()
        if any(alias in text for alias in lowered_aliases):
            hits.append(dataset_id)
    return hits


def recommendation(hit_count: int) -> str:
    if hit_count == 0:
        return "preferred_zero_overlap_candidate"
    return "manual_overlap_audit_required"


def render_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Lane C — Strict-OOD Candidate Inventory",
        "",
        "This inventory is a **baseline-overlap audit**, not a curated execution manifest.",
        "It helps decide which target tissues are still plausible first-wave strict-OOD candidates.",
        "",
        "| Target tissue | Candidate source | Candidate label | Baseline overlap count | Recommendation |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_tissue']} | {row['candidate_source']} | {row['candidate_label']} | "
            f"{row['baseline_overlap_count']} | {row['recommendation']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `preferred_zero_overlap_candidate` means the current `metadata_structured.json` text audit found no obvious baseline mention of that tissue string family.",
            "- `manual_overlap_audit_required` means at least one current baseline dataset text already matches the target alias family, so the tissue is risky as a first strict-OOD choice unless the scope is narrowed more carefully.",
            "",
            "## Baseline overlap examples",
            "",
        ]
    )
    for row in rows:
        examples = row["baseline_overlap_examples"] or []
        joined = ", ".join(examples) if examples else "none"
        lines.append(f"- **{row['target_tissue']}**: {joined}")
    return "\n".join(lines) + "\n"


def main() -> int:
    metadata = load_metadata()
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        examples = overlap_examples(metadata, target["aliases"])
        rows.append(
            {
                "target_tissue": target["target_tissue"],
                "candidate_source": target["candidate_source"],
                "candidate_label": target["candidate_label"],
                "baseline_overlap_count": len(examples),
                "baseline_overlap_examples": examples,
                "recommendation": recommendation(len(examples)),
            }
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_tissue",
                "candidate_source",
                "candidate_label",
                "baseline_overlap_count",
                "baseline_overlap_examples",
                "recommendation",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "baseline_overlap_examples": ";".join(row["baseline_overlap_examples"]),
                }
            )

    OUTPUT_MD.write_text(render_markdown(rows), encoding="utf-8")
    print(f"Wrote {OUTPUT_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUTPUT_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
