from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build_strict_ood_candidate_inventory.py"
OUTPUT_CSV = REPO_ROOT / "revision" / "experiments" / "lane_c_data" / "strict_ood_candidate_inventory.csv"
OUTPUT_MD = REPO_ROOT / "revision" / "experiments" / "lane_c_data" / "strict_ood_candidate_inventory.md"


def test_candidate_inventory_script_runs() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert OUTPUT_CSV.exists()
    assert OUTPUT_MD.exists()


def test_candidate_inventory_contains_all_lane_c_targets() -> None:
    with OUTPUT_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    targets = {row["target_tissue"] for row in rows}
    assert targets == {
        "kidney",
        "testis",
        "intestine",
        "cerebellum",
        "distal airway",
        "merkel-like",
    }


def test_candidate_inventory_flags_obvious_overlap_risks() -> None:
    with OUTPUT_CSV.open(newline="", encoding="utf-8") as handle:
        rows = {row["target_tissue"]: row for row in csv.DictReader(handle)}

    assert rows["distal airway"]["recommendation"] == "manual_overlap_audit_required"
    assert rows["merkel-like"]["recommendation"] == "manual_overlap_audit_required"
    assert rows["kidney"]["recommendation"] == "preferred_zero_overlap_candidate"
