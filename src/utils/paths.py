"""Centralized path constants for the CLOP-DiT project.

All default directories derive from PROJECT_ROOT.  Scripts and modules
should import from here instead of hardcoding path strings.

Sources (first wins):
  1. Environment variables: CLOPDIT_CACHE_DIR, CLOPDIT_RESULTS_DIR, etc.
  2. configs/pipeline.yaml if present
  3. Built-in defaults below
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
LOG_DIR = PROJECT_ROOT / "logs"


def _load_pipeline_config() -> dict[str, Any]:
    """Load configs/pipeline.yaml if present. Returns empty dict if missing."""
    path = CONFIG_DIR / "pipeline.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """Resolve path; if relative, interpret relative to base."""
    p = Path(value)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _get_path(env_key: str, config_key: str, default: Path) -> Path:
    """Prefer env, then pipeline config, then default."""
    env_val = os.environ.get(env_key)
    if env_val:
        return _resolve_path(env_val)
    cfg = _load_pipeline_config()
    if config_key in cfg and cfg[config_key]:
        return _resolve_path(cfg[config_key])
    return default


# Path constants (env > pipeline.yaml > defaults)
CACHE_DIR = _get_path(
    "CLOPDIT_CACHE_DIR",
    "cache_dir",
    PROJECT_ROOT / "data" / "cached_latents_v5.2",
)
RESULTS_DIR = _get_path(
    "CLOPDIT_RESULTS_DIR",
    "results_dir",
    PROJECT_ROOT / "results",
)
FIG_DIR = _get_path(
    "CLOPDIT_FIG_DIR",
    "fig_dir",
    RESULTS_DIR / "figures",
)
ARTICLE_FIGURES_DIR = _get_path(
    "CLOPDIT_ARTICLE_FIGURES_DIR",
    "article_figures_dir",
    PROJECT_ROOT / "articles" / "figures",
)
CHECKPOINT_DIR = _get_path(
    "CLOPDIT_CKPT_DIR",
    "checkpoint_dir",
    PROJECT_ROOT / "models" / "checkpoints",
)

# Optional paths from pipeline config (no env names in plan; use same pattern)
_pipeline = _load_pipeline_config()
PROCESSED_H5AD_DIR = _resolve_path(
    _pipeline.get("processed_h5ad_dir", "data/processed_h5ad")
) if _pipeline else (PROJECT_ROOT / "data" / "processed_h5ad")
SCGPT_DIR = _resolve_path(
    _pipeline.get("scgpt_dir", "models/scgpt_pancancer")
) if _pipeline else (PROJECT_ROOT / "models" / "scgpt_pancancer")
ARTICLE_DIR = _resolve_path(
    _pipeline.get("article_dir", "articles")
) if _pipeline else (PROJECT_ROOT / "articles")
ARTICLE_TEX = _pipeline.get("article_tex", "clop_dit_biology.tex") if _pipeline else "clop_dit_biology.tex"
