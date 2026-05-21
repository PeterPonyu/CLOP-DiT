#!/usr/bin/env python3
"""Validate the narrowed Lane-C conditional-go staffing gate.

This checker is intentionally lightweight and repo-local. It verifies:

- scope stays narrowed to Priority-2 strict-OOD only
- total engineer-day estimate stays <= 18
- required owner fields are populated
- referenced readiness paths exist

Example:
    python scripts/check_lane_c_staffing_gate.py \
        --plan revision/experiments/lane_c_data/conditional_go_staffing_plan.json \
        --json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REQUIRED_OWNER_KEYS = [
    "b2_strict_ood_leakage",
    "b3_label_bridge",
    "clop_retrain",
    "dit_retrain",
    "five_slice_reporting",
]

REQUIRED_READINESS_KEYS = [
    "ingest_manifest",
    "strict_ood_checker",
    "label_bridge",
    "baseline_metrics",
]

READY_LABEL_BRIDGE_STATUSES = {"reviewed", "approved", "novel"}
DEFAULT_EVAL_SPLITS = {"val", "valid", "validation", "test", "eval", "evaluation", "heldout"}


def load_plan(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"plan not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def build_report(plan: dict, repo_root: Path, plan_root: Path) -> dict[str, object]:
    failures: list[str] = []

    scope = str(plan.get("scope", "")).strip().lower()
    if scope != "priority-2 strict-ood only":
        failures.append("scope must remain 'priority-2 strict-ood only'")

    estimate = plan.get("engineer_day_estimate", {})
    total = estimate.get("total", None)
    if not isinstance(total, (int, float)):
        failures.append("engineer_day_estimate.total must be numeric")
    elif float(total) > 18:
        failures.append("engineer_day_estimate.total exceeds 18 engineer-days")

    owners = plan.get("owners", {})
    missing_owners = [
        key for key in REQUIRED_OWNER_KEYS
        if str(owners.get(key, "")).strip() == ""
    ]
    if missing_owners:
        failures.append(f"missing owners: {', '.join(missing_owners)}")

    readiness = plan.get("readiness", {})
    missing_readiness_keys = [key for key in REQUIRED_READINESS_KEYS if key not in readiness]
    if missing_readiness_keys:
        failures.append(f"missing readiness keys: {', '.join(missing_readiness_keys)}")

    missing_paths = []
    readiness_paths: dict[str, Path] = {}
    for key in REQUIRED_READINESS_KEYS:
        if key not in readiness:
            continue
        candidate_raw = Path(str(readiness[key]))
        candidate = candidate_raw if candidate_raw.is_absolute() else plan_root / candidate_raw
        if not candidate.exists():
            fallback_candidate = repo_root / candidate_raw
            if fallback_candidate.exists():
                candidate = fallback_candidate
        readiness_paths[key] = candidate
        if not candidate.exists():
            missing_paths.append({"key": key, "path": str(candidate)})
    if missing_paths:
        failures.append("one or more readiness paths do not exist")

    manifest_rows = None
    label_bridge_rows = None
    reviewed_label_bridge_rows = None
    eval_manifest_rows = None
    if "ingest_manifest" in readiness_paths and readiness_paths["ingest_manifest"].exists():
        manifest_path = readiness_paths["ingest_manifest"]
        manifest_rows = count_csv_rows(manifest_path)
        if manifest_rows == 0:
            failures.append("ingest manifest has no dataset rows yet")
        else:
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                eval_manifest_rows = sum(
                    1 for row in reader
                    if str(row.get("split", "")).strip().lower() in DEFAULT_EVAL_SPLITS
                )
            if eval_manifest_rows == 0:
                failures.append("ingest manifest has no eval/heldout rows yet")
    if "label_bridge" in readiness_paths and readiness_paths["label_bridge"].exists():
        label_bridge_path = readiness_paths["label_bridge"]
        label_bridge_rows = count_csv_rows(label_bridge_path)
        if label_bridge_rows == 0:
            failures.append("label bridge has no review rows yet")
        else:
            with label_bridge_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                reviewed_label_bridge_rows = sum(
                    1 for row in reader
                    if str(row.get("status", "")).strip().lower() in READY_LABEL_BRIDGE_STATUSES
                )
            if reviewed_label_bridge_rows == 0:
                failures.append("label bridge has no reviewed/approved rows yet")

    status = str(plan.get("status", "")).strip().lower()
    if status not in {"draft_not_approved", "approved"}:
        failures.append("status must be 'draft_not_approved' or 'approved'")
    if status == "approved" and missing_owners:
        failures.append("status cannot be 'approved' while owners are missing")
    if status != "approved":
        failures.append("status is not approved yet")

    ok = not failures
    return {
        "ok": ok,
        "scope": scope,
        "engineer_day_total": total,
        "missing_owners": missing_owners,
        "missing_paths": missing_paths,
        "manifest_rows": manifest_rows,
        "eval_manifest_rows": eval_manifest_rows,
        "label_bridge_rows": label_bridge_rows,
        "reviewed_label_bridge_rows": reviewed_label_bridge_rows,
        "status": status,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the narrowed Lane-C conditional-go staffing gate.")
    parser.add_argument(
        "--plan",
        default="revision/experiments/lane_c_data/conditional_go_staffing_plan.json",
        type=Path,
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    try:
        plan_path = args.plan if args.plan.is_absolute() else repo_root / args.plan
        plan = load_plan(plan_path)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            print(f"lane-c staffing gate failed: {exc}", file=sys.stderr)
        return 2

    report = build_report(plan, repo_root, plan_path.parent)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(f"lane-c staffing gate {status}")
        print(f"scope: {report['scope']}")
        print(f"engineer_day_total: {report['engineer_day_total']}")
        print(f"missing_owners: {', '.join(report['missing_owners']) or 'none'}")
        if report["manifest_rows"] is not None:
            print(f"manifest_rows: {report['manifest_rows']}")
        if report["eval_manifest_rows"] is not None:
            print(f"eval_manifest_rows: {report['eval_manifest_rows']}")
        if report["label_bridge_rows"] is not None:
            print(f"label_bridge_rows: {report['label_bridge_rows']}")
        if report["reviewed_label_bridge_rows"] is not None:
            print(f"reviewed_label_bridge_rows: {report['reviewed_label_bridge_rows']}")
        if report["missing_paths"]:
            print("missing_paths:")
            for item in report["missing_paths"]:
                print(f"  - {item['key']}: {item['path']}")
        if report["failures"]:
            print("failures:")
            for failure in report["failures"]:
                print(f"  - {failure}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
