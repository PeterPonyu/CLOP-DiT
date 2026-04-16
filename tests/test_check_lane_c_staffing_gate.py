from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_lane_c_staffing_gate.py"
PLAN = REPO_ROOT / "revision" / "experiments" / "lane_c_data" / "conditional_go_staffing_plan.json"


def write_plan(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_staffing_gate_templates_exist() -> None:
    assert SCRIPT.exists()
    assert PLAN.exists()


def test_staffing_gate_fails_when_owners_are_missing() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--plan", str(PLAN), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert payload["missing_owners"] == []
    assert "status is not approved yet" in payload["failures"]
    assert payload["manifest_rows"] >= 1
    assert payload["eval_manifest_rows"] >= 1
    assert payload["label_bridge_rows"] >= 1
    assert payload["reviewed_label_bridge_rows"] >= 1


def test_staffing_gate_passes_for_fully_populated_plan(tmp_path: Path) -> None:
    ingest_manifest = tmp_path / "MANIFEST.csv"
    ingest_manifest.write_text(
        "dataset_id,source_path,organism,tissue,split,priority,notes\n"
        "eval-kidney,data/processed_h5ad_revision/eval_kidney.h5ad,human,kidney,heldout,priority-2,ready\n",
        encoding="utf-8",
    )
    label_bridge = tmp_path / "label_bridge.csv"
    label_bridge.write_text(
        "dataset_id,source_label,target_label,status,reviewer_a,reviewer_b,notes\n"
        "eval-kidney,proximal_tubule,proximal_tubule,approved,a,b,ready\n",
        encoding="utf-8",
    )
    checker = tmp_path / "check_strict_ood.py"
    checker.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    baseline = tmp_path / "metrics_frozen"
    baseline.mkdir()

    plan = tmp_path / "plan.json"
    write_plan(
        plan,
        {
            "schema_version": 1,
            "status": "approved",
            "scope": "priority-2 strict-ood only",
            "engineer_day_estimate": {"total": 18},
            "owners": {
                "b2_strict_ood_leakage": "owner-a",
                "b3_label_bridge": "owner-b",
                "clop_retrain": "owner-c",
                "dit_retrain": "owner-d",
                "five_slice_reporting": "owner-e",
            },
            "readiness": {
                "ingest_manifest": str(ingest_manifest.relative_to(tmp_path)),
                "strict_ood_checker": str(checker.relative_to(tmp_path)),
                "label_bridge": str(label_bridge.relative_to(tmp_path)),
                "baseline_metrics": str(baseline.relative_to(tmp_path)),
            },
        },
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--plan", str(plan), "--json"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["missing_owners"] == []
    assert payload["manifest_rows"] == 1
    assert payload["eval_manifest_rows"] == 1
    assert payload["label_bridge_rows"] == 1
    assert payload["reviewed_label_bridge_rows"] == 1


def test_staffing_gate_fails_when_budget_exceeds_cap(tmp_path: Path) -> None:
    ingest_manifest = tmp_path / "MANIFEST.csv"
    ingest_manifest.write_text(
        "dataset_id,source_path,organism,tissue,split,priority,notes\n"
        "eval-kidney,data/processed_h5ad_revision/eval_kidney.h5ad,human,kidney,heldout,priority-2,ready\n",
        encoding="utf-8",
    )
    label_bridge = tmp_path / "label_bridge.csv"
    label_bridge.write_text(
        "dataset_id,source_label,target_label,status,reviewer_a,reviewer_b,notes\n"
        "eval-kidney,proximal_tubule,proximal_tubule,approved,a,b,ready\n",
        encoding="utf-8",
    )
    checker = tmp_path / "check_strict_ood.py"
    checker.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    baseline = tmp_path / "metrics_frozen"
    baseline.mkdir()

    plan = tmp_path / "plan.json"
    write_plan(
        plan,
        {
            "schema_version": 1,
            "status": "approved",
            "scope": "priority-2 strict-ood only",
            "engineer_day_estimate": {"total": 19},
            "owners": {
                "b2_strict_ood_leakage": "owner-a",
                "b3_label_bridge": "owner-b",
                "clop_retrain": "owner-c",
                "dit_retrain": "owner-d",
                "five_slice_reporting": "owner-e",
            },
            "readiness": {
                "ingest_manifest": str(ingest_manifest.relative_to(tmp_path)),
                "strict_ood_checker": str(checker.relative_to(tmp_path)),
                "label_bridge": str(label_bridge.relative_to(tmp_path)),
                "baseline_metrics": str(baseline.relative_to(tmp_path)),
            },
        },
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--plan", str(plan), "--json"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert any("exceeds 18 engineer-days" in failure for failure in payload["failures"])


def test_staffing_gate_fails_when_not_approved_even_if_other_fields_are_populated(tmp_path: Path) -> None:
    ingest_manifest = tmp_path / "MANIFEST.csv"
    ingest_manifest.write_text(
        "dataset_id,source_path,organism,tissue,split,priority,notes\n"
        "eval-kidney,data/processed_h5ad_revision/eval_kidney.h5ad,human,kidney,heldout,priority-2,ready\n",
        encoding="utf-8",
    )
    label_bridge = tmp_path / "label_bridge.csv"
    label_bridge.write_text(
        "dataset_id,source_label,target_label,status,reviewer_a,reviewer_b,notes\n"
        "eval-kidney,proximal_tubule,proximal_tubule,approved,a,b,ready\n",
        encoding="utf-8",
    )
    checker = tmp_path / "check_strict_ood.py"
    checker.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    baseline = tmp_path / "metrics_frozen"
    baseline.mkdir()

    plan = tmp_path / "plan.json"
    write_plan(
        plan,
        {
            "schema_version": 1,
            "status": "draft_not_approved",
            "scope": "priority-2 strict-ood only",
            "engineer_day_estimate": {"total": 18},
            "owners": {
                "b2_strict_ood_leakage": "owner-a",
                "b3_label_bridge": "owner-b",
                "clop_retrain": "owner-c",
                "dit_retrain": "owner-d",
                "five_slice_reporting": "owner-e",
            },
            "readiness": {
                "ingest_manifest": str(ingest_manifest.relative_to(tmp_path)),
                "strict_ood_checker": str(checker.relative_to(tmp_path)),
                "label_bridge": str(label_bridge.relative_to(tmp_path)),
                "baseline_metrics": str(baseline.relative_to(tmp_path)),
            },
        },
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--plan", str(plan), "--json"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert "status is not approved yet" in payload["failures"]


def test_staffing_gate_fails_when_label_bridge_rows_are_only_proposed(tmp_path: Path) -> None:
    ingest_manifest = tmp_path / "MANIFEST.csv"
    ingest_manifest.write_text(
        "dataset_id,source_path,organism,tissue,split,priority,notes\n"
        "eval-kidney,data/processed_h5ad_revision/eval_kidney.h5ad,human,kidney,heldout,priority-2,ready\n",
        encoding="utf-8",
    )
    label_bridge = tmp_path / "label_bridge.csv"
    label_bridge.write_text(
        "dataset_id,source_label,target_label,status,reviewer_a,reviewer_b,notes\n"
        "eval-kidney,proximal tubule cell,Kidney proximal tubule cells,proposed,,,draft\n",
        encoding="utf-8",
    )
    checker = tmp_path / "check_strict_ood.py"
    checker.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    baseline = tmp_path / "metrics_frozen"
    baseline.mkdir()

    plan = tmp_path / "plan.json"
    write_plan(
        plan,
        {
            "schema_version": 1,
            "status": "approved",
            "scope": "priority-2 strict-ood only",
            "engineer_day_estimate": {"total": 18},
            "owners": {
                "b2_strict_ood_leakage": "owner-a",
                "b3_label_bridge": "owner-b",
                "clop_retrain": "owner-c",
                "dit_retrain": "owner-d",
                "five_slice_reporting": "owner-e",
            },
            "readiness": {
                "ingest_manifest": str(ingest_manifest.relative_to(tmp_path)),
                "strict_ood_checker": str(checker.relative_to(tmp_path)),
                "label_bridge": str(label_bridge.relative_to(tmp_path)),
                "baseline_metrics": str(baseline.relative_to(tmp_path)),
            },
        },
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--plan", str(plan), "--json"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["reviewed_label_bridge_rows"] == 0
    assert "label bridge has no reviewed/approved rows yet" in payload["failures"]
