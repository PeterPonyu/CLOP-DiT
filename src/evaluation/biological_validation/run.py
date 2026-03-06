# run.py — Main entry for biological validation (figures 1–4 and metrics).
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict

import torch
import scanpy as sc
import anndata as ad

from src.utils.logging_config import setup_logging
from src.utils.helpers import seed_everything
from src.utils.paths import PROJECT_ROOT, FIG_DIR
from src.architecture.decoder import ScGPTDecoder

from .io import load_prompt_file, load_subcluster_metadata, load_dataset_specs, load_dataset_indices
from .generation import generate_cells_text2cell, build_real_subset
from .figures import figure1_text2cell_multi, figure2_cell2cell, figure3_celltypist, figure4_summary

logger = logging.getLogger(__name__)

SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _write_dataset_tables(metrics: Dict, output_dir: Path) -> None:
    """Write per-dataset tables for reviewer-facing inspection."""
    text2cell = metrics.get("text2cell_multi", {}).get("datasets", {})
    if text2cell:
        metric_keys = [
            "gene_mean_pearson",
            "gene_mean_R2",
            "gene_var_pearson",
            "FD_gene_pca50",
            "FD_scgpt_embedding",
            "marker_specificity_real",
            "marker_specificity_generated",
        ]
        rows = []
        for label, row_metrics in text2cell.items():
            row = {"dataset": label}
            for key in metric_keys:
                row[key] = row_metrics.get(key)
            rows.append(row)

        with open(output_dir / "text2cell_per_dataset.json", "w") as f:
            json.dump(rows, f, indent=2)

        with open(output_dir / "text2cell_per_dataset.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", *metric_keys])
            writer.writeheader()
            writer.writerows(rows)

    per_ds_match = metrics.get("celltypist", {}).get("per_dataset_match_rate")
    if per_ds_match:
        match_rows = [
            {"dataset": label, "celltypist_match_rate": value}
            for label, value in per_ds_match.items()
        ]
        with open(output_dir / "celltypist_per_dataset.json", "w") as f:
            json.dump(match_rows, f, indent=2)
        with open(output_dir / "celltypist_per_dataset.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "celltypist_match_rate"])
            writer.writeheader()
            writer.writerows(match_rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Biological Validation v2")
    parser.add_argument("--reference_h5ad")
    parser.add_argument("--dataset_key")
    parser.add_argument("--dataset_manifest", help="JSON list of {dataset_key, reference_h5ad, label?, prompts?}")
    parser.add_argument("--max_datasets", type=int, default=3)
    parser.add_argument("--subcluster_meta", default="data/processed_h5ad/subcluster_metadata.json")
    parser.add_argument("--auto_prompts", action="store_true", help="Auto-build prompts from subcluster metadata")
    parser.add_argument("--prompt_file", default="", help="JSON mapping of cell_type -> prompt to override defaults")
    parser.add_argument("--prompt_top_k", type=int, default=4)
    parser.add_argument("--prompt_min_conf", type=float, default=0.25)
    parser.add_argument("--output_dir", default=str(FIG_DIR / "biological_validation"))
    parser.add_argument("--num_cells_per_type", type=int, default=200)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--edit_strength", type=float, default=0.5)
    parser.add_argument("--dit_checkpoint", default="models/checkpoints/dit_best.pth")
    parser.add_argument("--clop_checkpoint", default="models/checkpoints/clop_best.pth")
    parser.add_argument("--cell2cell_checkpoint", default="models/checkpoints/cell2cell_best.pth")
    parser.add_argument("--scgpt_model_dir", default="models/scgpt_pancancer")
    parser.add_argument("--text_encoder", default="microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
    parser.add_argument("--celltypist_model", default="Human_Lung_Atlas.pkl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_figure1", action="store_true")
    parser.add_argument("--skip_figure2", action="store_true")
    parser.add_argument("--skip_figure3", action="store_true")
    parser.add_argument("--skip_figure4", action="store_true")
    args = parser.parse_args()

    setup_logging()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scgpt = ScGPTDecoder(model_dir=args.scgpt_model_dir, device=torch.device(args.device))
    metrics: Dict = {}

    prompt_overrides = load_prompt_file(args.prompt_file)
    subcluster_meta = load_subcluster_metadata(args.subcluster_meta) if args.auto_prompts else {}
    dataset_specs = load_dataset_specs(args, subcluster_meta, prompt_overrides)
    dataset_runs = []

    for spec in dataset_specs:
        dataset_key = spec["dataset_key"]
        ref_path = spec["reference_h5ad"]
        label = spec.get("label", dataset_key)
        prompts = spec.get("prompts", {})

        logger.info(f"Loading reference: {ref_path}")
        adata_ref = sc.read_h5ad(ref_path)
        indices_map = load_dataset_indices(
            dataset_key,
            subcluster_meta=subcluster_meta if subcluster_meta else None,
            subcluster_meta_path=args.subcluster_meta,
        )

        fake = None
        real = None
        if not args.skip_figure1 or not args.skip_figure3:
            fake = generate_cells_text2cell(adata_ref, args, prompts, scripts_dir=SCRIPTS_DIR)
            fake.obs["dataset"] = label
            real = build_real_subset(adata_ref, indices_map, args.num_cells_per_type, prompts)
            real.obs["dataset"] = label

        dataset_runs.append({
            "label": label,
            "dataset_key": dataset_key,
            "reference_h5ad": ref_path,
            "prompts": prompts,
            "adata_ref": adata_ref,
            "indices_map": indices_map,
            "fake_adata": fake,
            "real_adata": real,
        })

    metrics["run_config"] = {
        "seed": args.seed,
        "num_cells_per_type": args.num_cells_per_type,
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "edit_strength": args.edit_strength,
        "prompt_top_k": args.prompt_top_k,
        "prompt_min_conf": args.prompt_min_conf,
        "max_datasets": args.max_datasets,
    }
    metrics["dataset_manifest"] = [
        {
            "label": ds["label"],
            "dataset_key": ds["dataset_key"],
            "reference_h5ad": ds["reference_h5ad"],
            "n_prompt_types": len(ds["prompts"]),
        }
        for ds in dataset_runs
    ]

    if not args.skip_figure1:
        figure1_text2cell_multi(dataset_runs, scgpt, output_dir, metrics)

    if not args.skip_figure2:
        preferred = None
        for ds in dataset_runs:
            if "Monocytes" in ds["indices_map"] and "Macrophages" in ds["indices_map"]:
                preferred = ds
                break
        if preferred is None:
            preferred = dataset_runs[0]
        metrics["cell2cell_dataset"] = preferred["label"]
        figure2_cell2cell(
            preferred["adata_ref"],
            preferred["indices_map"],
            scgpt,
            output_dir,
            args,
            metrics,
            preferred["prompts"],
            scripts_dir=SCRIPTS_DIR,
        )

    if not args.skip_figure3:
        fake_all = ad.concat(
            [ds["fake_adata"] for ds in dataset_runs if ds["fake_adata"] is not None],
            join="outer",
            merge="same",
        )
        real_all = ad.concat(
            [ds["real_adata"] for ds in dataset_runs if ds["real_adata"] is not None],
            join="outer",
            merge="same",
        )
        figure3_celltypist(
            fake_all,
            output_dir,
            metrics,
            model_name=args.celltypist_model,
            real_adata=real_all,
        )

    if not args.skip_figure4:
        figure4_summary(metrics, output_dir)

    metrics_path = output_dir / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    _write_dataset_tables(metrics, output_dir)
    logger.info(f"Metrics saved -> {metrics_path}")
    logger.info("=== Validation complete ===")
