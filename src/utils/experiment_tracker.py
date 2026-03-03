# experiment_tracker.py — Unified experiment versioning and result management
"""
Manages experiment versions, checkpoints, metrics, and figures in a structured
directory hierarchy. Every training run produces a self-contained version folder
with full traceability.

Directory layout:
    results/
    ├── v1.0_baseline/
    │   ├── config.yaml          # Frozen config at run time
    │   ├── checkpoints/         # clop_best.pth, dit_best.pth, ...
    │   ├── metrics/             # evaluation_metrics.json, clop_history.json, ...
    │   ├── figures/             # All visualizations
    │   └── run_info.json        # Timestamps, git hash, system info
    ├── v1.1_logit_normal/
    │   └── ...
    └── versions.json            # Registry of all versions with summaries
"""

import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any


class ExperimentTracker:
    """Manage experiment versions with full traceability.

    Parameters
    ----------
    base_dir : str or Path
        Root results directory.
    """

    VERSIONS_FILE = "versions.json"

    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._versions = self._load_versions()

    # ========================================================================
    #  Version registry
    # ========================================================================

    def _load_versions(self) -> Dict:
        path = self.base_dir / self.VERSIONS_FILE
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {"versions": []}

    def _save_versions(self):
        with open(self.base_dir / self.VERSIONS_FILE, "w") as f:
            json.dump(self._versions, f, indent=2, default=str)

    # ========================================================================
    #  Create / manage versions
    # ========================================================================

    def create_version(
        self,
        version_id: str,
        description: str,
        config: Dict,
        changes: Optional[str] = None,
    ) -> Path:
        """Create a new experiment version directory.

        Parameters
        ----------
        version_id : str
            e.g. "v1.0_baseline", "v1.1_logit_normal"
        description : str
            Human-readable description of this version.
        config : dict
            Full training configuration (will be saved frozen).
        changes : str, optional
            Summary of changes from previous version.

        Returns
        -------
        version_dir : Path
        """
        version_dir = self.base_dir / version_id

        if version_dir.exists():
            raise ValueError(
                f"Version '{version_id}' already exists at {version_dir}. "
                f"Use a different version_id or remove the existing directory."
            )

        # Create directory structure
        for sub in ["checkpoints", "metrics", "figures"]:
            (version_dir / sub).mkdir(parents=True)

        # Save frozen config
        import yaml
        with open(version_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Save run info
        run_info = {
            "version_id": version_id,
            "description": description,
            "changes": changes,
            "created_at": datetime.now().isoformat(),
            "system": {
                "hostname": platform.node(),
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "git_hash": self._get_git_hash(),
            "status": "created",
        }
        with open(version_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)

        # Register version
        self._versions["versions"].append({
            "version_id": version_id,
            "description": description,
            "changes": changes,
            "created_at": run_info["created_at"],
            "status": "created",
        })
        self._save_versions()

        return version_dir

    def update_status(self, version_id: str, status: str, summary: Optional[Dict] = None):
        """Update version status (created → training → completed / failed)."""
        # Update run_info.json
        info_path = self.base_dir / version_id / "run_info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            info["status"] = status
            info[f"{status}_at"] = datetime.now().isoformat()
            if summary:
                info["summary"] = summary
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)

        # Update registry
        for v in self._versions["versions"]:
            if v["version_id"] == version_id:
                v["status"] = status
                if summary:
                    v["summary"] = summary
                break
        self._save_versions()

    def save_metrics(self, version_id: str, filename: str, metrics: Dict):
        """Save metrics JSON to the version's metrics directory."""
        path = self.base_dir / version_id / "metrics" / filename
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    def get_version_dir(self, version_id: str) -> Path:
        """Get the directory for a given version."""
        return self.base_dir / version_id

    def get_checkpoint_dir(self, version_id: str) -> Path:
        return self.base_dir / version_id / "checkpoints"

    def get_figures_dir(self, version_id: str) -> Path:
        return self.base_dir / version_id / "figures"

    def get_metrics_dir(self, version_id: str) -> Path:
        return self.base_dir / version_id / "metrics"

    def list_versions(self) -> list:
        """List all registered versions."""
        return self._versions["versions"]

    # ========================================================================
    #  Comparison across versions
    # ========================================================================

    def compare_versions(self, metric_file: str = "evaluation_metrics.json") -> Dict:
        """Compare a specific metric file across all versions.

        Returns
        -------
        comparison : dict mapping version_id → metrics
        """
        comparison = {}
        for v in self._versions["versions"]:
            vid = v["version_id"]
            path = self.base_dir / vid / "metrics" / metric_file
            if path.exists():
                with open(path) as f:
                    comparison[vid] = json.load(f)
        return comparison

    # ========================================================================
    #  Utilities
    # ========================================================================

    @staticmethod
    def _get_git_hash() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return "unknown"
