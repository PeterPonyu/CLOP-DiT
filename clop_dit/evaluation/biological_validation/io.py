# io.py — Loading prompts, metadata, dataset specs for biological validation.
from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Dict, List

from . import constants

logger = logging.getLogger(__name__)


def load_class_from_script(script_path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location(class_name, str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def load_prompt_file(path: str) -> Dict[str, str]:
    """Load a JSON file mapping cell_type -> prompt text."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning(f"Prompt file not found: {path}")
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(f"Invalid prompt file format: {path}")
            return {}
        return data
    except Exception as e:
        logger.warning(f"Failed to load prompt file: {e}")
        return {}


def load_subcluster_metadata(path: str) -> Dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning(f"Subcluster metadata not found: {path}")
        return {}
    with open(p) as f:
        return json.load(f)


def load_dataset_indices(
    dataset_key: str,
    subcluster_meta: Dict | None = None,
    subcluster_meta_path: str | None = None,
) -> Dict[str, "np.ndarray"]:
    import numpy as np

    if subcluster_meta is None and subcluster_meta_path:
        p = Path(subcluster_meta_path)
        if p.exists():
            with open(p) as f:
                subcluster_meta = json.load(f)
    if subcluster_meta is None:
        return {}
    if dataset_key not in subcluster_meta:
        return {}
    ds = subcluster_meta[dataset_key]
    ct_idx: Dict[str, list] = {}
    for _, info in ds.get("clusters", {}).items():
        ct = info["cell_type"]
        idx = np.array(info["cell_indices"], dtype=int)
        ct_idx.setdefault(ct, []).append(idx)
    for ct in ct_idx:
        ct_idx[ct] = np.concatenate(ct_idx[ct])
    return ct_idx


def load_dataset_specs(args, subcluster_meta: Dict, prompt_overrides: Dict) -> List[Dict]:
    from .generation import build_prompts_from_subclusters

    if args.dataset_manifest:
        with open(args.dataset_manifest) as f:
            specs = json.load(f)
        if not isinstance(specs, list) or not specs:
            raise ValueError("dataset_manifest must be a non-empty JSON list")
    else:
        if not args.reference_h5ad or not args.dataset_key:
            raise ValueError(
                "Provide either --dataset_manifest or both --reference_h5ad and --dataset_key"
            )
        specs = [
            {
                "dataset_key": args.dataset_key,
                "reference_h5ad": args.reference_h5ad,
                "label": args.dataset_key,
            }
        ]

    prepared = []
    for spec in specs[: getattr(args, "max_datasets", 10)]:
        dataset_key = spec["dataset_key"]
        prompts = spec.get("prompts")
        if not prompts and getattr(args, "auto_prompts", False):
            prompts = build_prompts_from_subclusters(
                subcluster_meta,
                dataset_key,
                top_k=getattr(args, "prompt_top_k", 4),
                min_conf=getattr(args, "prompt_min_conf", 0.25),
            )
        if not prompts:
            prompts = dict(constants.DEFAULT_TEXT2CELL_PROMPTS)
        for ct, p in prompt_overrides.items():
            if ct in prompts:
                prompts[ct] = p
        spec = dict(spec)
        spec["prompts"] = prompts
        prepared.append(spec)
    return prepared
