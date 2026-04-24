#!/usr/bin/env python3
"""Check strict-OOD tissue separation from a CSV dataset manifest.

The check is intentionally simple and repo-local:

- held-out tissues must not appear in any training split row
- each held-out tissue must appear in at least one evaluation-side row

Example:
    python scripts/check_strict_ood.py \
        --manifest data/manifest.csv \
        --heldout kidney --heldout "distal airway"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TRAIN_SPLITS = {"train", "training"}
DEFAULT_EVAL_SPLITS = {"val", "valid", "validation", "test", "eval", "evaluation", "heldout"}


@dataclass(frozen=True)
class ManifestRow:
    dataset_id: str
    tissue: str
    split: str
    raw: dict[str, str]


def parse_tissues(value: str) -> set[str]:
    normalized = value.strip().lower()
    if not normalized:
        return set()
    separators = ["|", ";", ","]
    parts = [normalized]
    for sep in separators:
        next_parts: list[str] = []
        for part in parts:
            next_parts.extend(piece.strip() for piece in part.split(sep))
        parts = next_parts
    return {part for part in parts if part}


def load_manifest(path: Path, tissue_column: str, split_column: str, dataset_id_column: str) -> list[ManifestRow]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {path}")
        required = {tissue_column, split_column, dataset_id_column}
        missing = [field for field in required if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"manifest missing required columns {missing}: {path}")
        rows = []
        for row in reader:
            rows.append(
                ManifestRow(
                    dataset_id=(row.get(dataset_id_column) or "").strip(),
                    tissue=(row.get(tissue_column) or "").strip(),
                    split=(row.get(split_column) or "").strip().lower(),
                    raw=row,
                )
            )
    return rows


def build_report(
    rows: list[ManifestRow],
    heldout_tissues: set[str],
    train_splits: set[str],
    eval_splits: set[str],
) -> dict[str, object]:
    leakage: list[dict[str, str]] = []
    seen_eval: dict[str, int] = {tissue: 0 for tissue in heldout_tissues}

    for row in rows:
        tissues = parse_tissues(row.tissue)
        matched = heldout_tissues & tissues
        if not matched:
            continue
        if row.split in train_splits:
            leakage.append(
                {
                    "dataset_id": row.dataset_id,
                    "split": row.split,
                    "tissue": row.tissue,
                }
            )
        if row.split in eval_splits:
            for tissue in matched:
                seen_eval[tissue] += 1

    missing_eval = sorted([tissue for tissue, count in seen_eval.items() if count == 0])
    ok = not leakage and not missing_eval
    return {
        "ok": ok,
        "heldout_tissues": sorted(heldout_tissues),
        "row_count": len(rows),
        "leakage": leakage,
        "missing_eval_tissues": missing_eval,
        "seen_eval_counts": seen_eval,
    }


def parse_csv_set(value: str | None, fallback: set[str]) -> set[str]:
    if value is None or value.strip() == "":
        return set(fallback)
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check strict-OOD train/eval leakage from a dataset manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--heldout", action="append", default=[], help="Repeatable held-out tissue name.")
    parser.add_argument("--heldout-csv", default="", help="Comma-separated held-out tissue names.")
    parser.add_argument("--tissue-column", default="tissue")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--dataset-id-column", default="dataset_id")
    parser.add_argument("--train-splits", default=",".join(sorted(DEFAULT_TRAIN_SPLITS)))
    parser.add_argument("--eval-splits", default=",".join(sorted(DEFAULT_EVAL_SPLITS)))
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    heldout_tissues = {item.strip().lower() for item in args.heldout if item.strip()}
    heldout_tissues.update({item.strip().lower() for item in args.heldout_csv.split(",") if item.strip()})
    if not heldout_tissues:
        parser.error("at least one --heldout or --heldout-csv entry is required")

    train_splits = parse_csv_set(args.train_splits, DEFAULT_TRAIN_SPLITS)
    eval_splits = parse_csv_set(args.eval_splits, DEFAULT_EVAL_SPLITS)
    try:
        rows = load_manifest(args.manifest, args.tissue_column, args.split_column, args.dataset_id_column)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            print(f"strict-OOD check failed: {exc}", file=sys.stderr)
        return 2

    report = build_report(rows, heldout_tissues, train_splits, eval_splits)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(f"strict-OOD {status}")
        print(f"held-out tissues: {', '.join(report['heldout_tissues'])}")
        print(f"manifest rows: {report['row_count']}")
        print(f"train leakage rows: {len(report['leakage'])}")
        print(f"missing eval tissues: {', '.join(report['missing_eval_tissues']) or 'none'}")
        if report["leakage"]:
            print("leakage details:")
            for item in report["leakage"]:
                print(f"  - {item['dataset_id']} | split={item['split']} | tissue={item['tissue']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
