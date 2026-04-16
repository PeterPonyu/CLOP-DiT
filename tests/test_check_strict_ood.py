from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_strict_ood.py"


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset_id", "source_path", "organism", "tissue", "split", "priority", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_templates_exist() -> None:
    assert (REPO_ROOT / "data" / "processed_h5ad_revision" / "README.md").exists()
    assert (REPO_ROOT / "data" / "processed_h5ad_revision" / "MANIFEST.csv").exists()
    assert (REPO_ROOT / "data" / "processed_h5ad_revision" / "label_bridge.csv").exists()
    assert SCRIPT.exists()


def test_strict_ood_check_passes_when_heldout_tissues_are_eval_only(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.csv"
    write_manifest(
        manifest,
        [
            {
                "dataset_id": "train-lung",
                "source_path": "data/processed_h5ad/train_lung.h5ad",
                "organism": "human",
                "tissue": "lung",
                "split": "train",
                "priority": "baseline",
                "notes": "",
            },
            {
                "dataset_id": "eval-kidney",
                "source_path": "data/processed_h5ad_revision/eval_kidney.h5ad",
                "organism": "human",
                "tissue": "kidney",
                "split": "heldout",
                "priority": "priority-2",
                "notes": "",
            },
        ],
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", "kidney", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["missing_eval_tissues"] == []


def test_strict_ood_check_fails_on_train_leakage(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.csv"
    write_manifest(
        manifest,
        [
            {
                "dataset_id": "train-kidney",
                "source_path": "data/processed_h5ad/train_kidney.h5ad",
                "organism": "human",
                "tissue": "kidney",
                "split": "train",
                "priority": "priority-2",
                "notes": "",
            },
            {
                "dataset_id": "eval-kidney",
                "source_path": "data/processed_h5ad_revision/eval_kidney.h5ad",
                "organism": "human",
                "tissue": "kidney",
                "split": "heldout",
                "priority": "priority-2",
                "notes": "",
            },
        ],
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", "kidney", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert payload["leakage"]


def test_strict_ood_check_fails_when_no_eval_rows_exist(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.csv"
    write_manifest(
        manifest,
        [
            {
                "dataset_id": "train-lung",
                "source_path": "data/processed_h5ad/train_lung.h5ad",
                "organism": "human",
                "tissue": "lung",
                "split": "train",
                "priority": "baseline",
                "notes": "",
            }
        ],
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--heldout", "kidney", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["missing_eval_tissues"] == ["kidney"]
