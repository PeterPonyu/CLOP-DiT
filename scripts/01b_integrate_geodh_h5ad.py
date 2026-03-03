#!/usr/bin/env python3
"""
Integrate GEO-DataHub-converted h5ad datasets into CLOP-DiT processed_h5ad.

This script is designed to keep preprocessing equivalent to
`00_prepare_all_data.py` and `01_integrate_h5_datasets.py`:
  - QC filters: min_genes, min_cells, mito < 20%
  - counts layer preservation
  - normalize_total + log1p
  - HVG selection (seurat_v3 fallback to seurat)

Default behavior uses one canonical `*_merged.h5ad` per GSE to avoid
duplicate leakage from derivative files in GEO-DataHub output.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def infer_gse_id(name: str) -> Optional[str]:
    match = re.search(r"(GSE\d+)", name)
    return match.group(1) if match else None


def load_dataset_meta(meta_root: Path, gse_id: str) -> Dict:
    meta_path = meta_root / gse_id / "dataset_meta.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to read {meta_path}: {exc}")
        return {}


def build_text_description(gse_id: str, meta: Dict) -> str:
    geo = meta.get("geo_metadata", {}) if isinstance(meta, dict) else {}
    title = str(geo.get("title", "")).strip()
    summary = str(geo.get("summary", "")).strip()
    design = str(geo.get("overall_design", "")).strip()
    organism = str(geo.get("organism", "")).strip()

    title = title.replace("\n", " ").replace("\r", " ").strip()
    summary = summary.replace("\n", " ").replace("\r", " ").strip()
    design = design.replace("\n", " ").replace("\r", " ").strip()

    parts = [
        "Single-cell RNA sequencing dataset",
        f"from GEO series {gse_id}",
    ]
    if organism:
        parts.append(f"({organism})")
    intro = " ".join(parts) + "."

    details = []
    if title:
        details.append(f"Title: {title}.")
    if summary:
        details.append(f"Summary: {summary}")
    elif design:
        details.append(f"Design: {design}")

    text = f"{intro} {' '.join(details)}".strip()
    if len(text) < 30:
        return f"Single-cell RNA sequencing data from GEO series {gse_id}."
    return text


def select_source_files(source_dir: Path, strategy: str) -> List[Path]:
    all_h5ad = sorted(source_dir.glob("*.h5ad"))
    if strategy == "all":
        return all_h5ad

    grouped: Dict[str, List[Path]] = {}
    for path in all_h5ad:
        gse_id = infer_gse_id(path.name)
        if not gse_id:
            continue
        grouped.setdefault(gse_id, []).append(path)

    selected: List[Path] = []
    for gse_id in sorted(grouped.keys()):
        names = grouped[gse_id]
        merged = [p for p in names if p.name.endswith("_merged.h5ad")]
        if merged:
            selected.append(merged[0])
        else:
            selected.append(names[0])
    return selected


def read_h5ad_with_subsample(path: Path, max_cells: int, random_seed: int) -> Tuple[Optional[sc.AnnData], str]:
    try:
        adata_backed = sc.read_h5ad(path, backed="r")
    except Exception as exc:
        return None, f"read_failed: {exc}"

    n_obs = adata_backed.n_obs
    n_vars = adata_backed.n_vars

    try:
        if max_cells and n_obs > max_cells:
            rng = np.random.default_rng(random_seed)
            chosen = np.sort(rng.choice(n_obs, size=max_cells, replace=False))
            adata = adata_backed[chosen, :].to_memory()
            note = f"subsampled_{max_cells}_from_{n_obs}"
        else:
            adata = adata_backed.to_memory()
            note = f"full_{n_obs}"
    finally:
        adata_backed.file.close()

    logger.info(f"Loaded {path.name}: {n_obs}x{n_vars} -> {adata.shape[0]}x{adata.shape[1]}")
    return adata, note


def extract_count_like_matrix(adata: sc.AnnData) -> Tuple[Optional[sc.AnnData], str]:
    x = adata.X
    test_block = x[: min(100, adata.n_obs), : min(200, adata.n_vars)]
    block = test_block.toarray() if sp.issparse(test_block) else np.asarray(test_block)

    int_like_x = np.allclose(block, np.round(block), atol=1e-3)
    non_negative_x = np.nanmin(block) >= -1e-8

    if int_like_x and non_negative_x:
        return adata, "x_count_like"

    if adata.raw is not None:
        raw_x = adata.raw.X
        raw_block = raw_x[: min(100, adata.n_obs), : min(200, adata.raw.n_vars)]
        raw_block = raw_block.toarray() if sp.issparse(raw_block) else np.asarray(raw_block)
        int_like_raw = np.allclose(raw_block, np.round(raw_block), atol=1e-3)
        non_negative_raw = np.nanmin(raw_block) >= -1e-8
        if int_like_raw and non_negative_raw:
            raw_adata = sc.AnnData(X=adata.raw.X.copy(), obs=adata.obs.copy(), var=adata.raw.var.copy())
            return raw_adata, "raw_count_like"

    for layer_name in ["counts", "raw_counts", "count"]:
        if layer_name in adata.layers:
            lx = adata.layers[layer_name]
            l_block = lx[: min(100, adata.n_obs), : min(200, adata.n_vars)]
            l_block = l_block.toarray() if sp.issparse(l_block) else np.asarray(l_block)
            int_like_l = np.allclose(l_block, np.round(l_block), atol=1e-3)
            non_negative_l = np.nanmin(l_block) >= -1e-8
            if int_like_l and non_negative_l:
                adata.X = lx.copy()
                return adata, f"layer_{layer_name}_count_like"

    return None, "no_count_like_matrix"


def preprocess_adata(
    adata: sc.AnnData,
    dataset_id: str,
    min_genes: int,
    min_cells: int,
    min_final_cells: int,
    n_top_genes: int,
    target_sum: float,
) -> Tuple[Optional[sc.AnnData], str]:
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    if sp.issparse(adata.X):
        if adata.X.dtype in (np.int64, np.int32):
            adata.X = adata.X.astype(np.float32)
    else:
        if adata.X.dtype in (np.int64, np.int32):
            adata.X = adata.X.astype(np.float32)

    gene_sample = list(adata.var_names[:50])
    if gene_sample and sum(g.startswith(("ENSG", "ENSMUSG", "ENSGAL")) for g in gene_sample) > len(gene_sample) * 0.6:
        return None, "ensembl_gene_ids_incompatible"

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if adata.n_obs < min_final_cells:
        return None, f"too_few_cells_after_filter_cells:{adata.n_obs}"

    adata.var["mt"] = (
        adata.var_names.str.startswith("MT-")
        | adata.var_names.str.startswith("mt-")
    )
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()

    if adata.n_obs < min_final_cells:
        return None, f"too_few_cells_after_mito_filter:{adata.n_obs}"

    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    if adata.n_vars > n_top_genes:
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top_genes,
                flavor="seurat_v3",
                layer="counts",
                subset=False,
            )
        except Exception:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top_genes,
                flavor="seurat",
                subset=False,
            )
        adata = adata[:, adata.var["highly_variable"]].copy()

    if adata.n_obs < min_final_cells or adata.n_vars < 300:
        return None, f"too_small_after_preprocess:{adata.n_obs}x{adata.n_vars}"

    logger.info(f"[{dataset_id}] final {adata.shape[0]}x{adata.shape[1]}")
    return adata, "ok"


def main():
    parser = argparse.ArgumentParser(description="Integrate GEO-DataHub h5ad into CLOP-DiT")
    parser.add_argument("--source_dir", type=str, default="/home/zeyufu/Desktop/GEO-DataHub/h5ad_output")
    parser.add_argument("--meta_root", type=str, default="/home/zeyufu/Desktop/GEO-DataHub/downloads")
    parser.add_argument("--output_dir", type=str, default="data/processed_h5ad")
    parser.add_argument("--main_metadata", type=str, default="data/processed_h5ad/metadata_structured.json")
    parser.add_argument("--selection", type=str, default="canonical", choices=["canonical", "all"])
    parser.add_argument("--dataset_suffix", type=str, default="geodh")
    parser.add_argument("--max_cells", type=int, default=3000)
    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument("--min_genes", type=int, default=200)
    parser.add_argument("--min_cells", type=int, default=3)
    parser.add_argument("--min_final_cells", type=int, default=200)
    parser.add_argument("--target_sum", type=float, default=1e4)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    setup_logging()

    source_dir = Path(args.source_dir)
    meta_root = Path(args.meta_root)
    output_dir = Path(args.output_dir)
    main_metadata = Path(args.main_metadata)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = select_source_files(source_dir, args.selection)

    print("=" * 72)
    print("Integrate GEO-DataHub h5ad into CLOP-DiT")
    print("=" * 72)
    print(f"Source dir:        {source_dir}")
    print(f"Selection mode:    {args.selection}")
    print(f"Candidate files:   {len(files)}")
    print(f"Output dir:        {output_dir}")
    print(f"Dataset suffix:    {args.dataset_suffix}")
    print(f"Max cells/file:    {args.max_cells}")
    print("=" * 72)

    metadata_new: Dict[str, Dict] = {}
    report: List[Dict] = []
    processed = 0
    skipped = 0
    total_cells = 0

    for idx, file_path in enumerate(files, start=1):
        gse_id = infer_gse_id(file_path.name)
        if not gse_id:
            skipped += 1
            report.append({"file": file_path.name, "status": "skipped", "reason": "no_gse_id"})
            print(f"[{idx}/{len(files)}] {file_path.name}: skipped (no GSE id)")
            continue

        dataset_id = f"{gse_id}_{args.dataset_suffix}"
        out_path = output_dir / f"{dataset_id}_processed.h5ad"

        if out_path.exists() and not args.overwrite:
            try:
                adata_existing = sc.read_h5ad(out_path, backed="r")
                n_cells, n_genes = adata_existing.shape
                adata_existing.file.close()
            except Exception:
                n_cells, n_genes = 0, 0

            meta = load_dataset_meta(meta_root, gse_id)
            text = build_text_description(gse_id, meta)
            geo = meta.get("geo_metadata", {}) if isinstance(meta, dict) else {}

            metadata_new[dataset_id] = {
                "text": text,
                "n_cells": int(n_cells),
                "n_genes": int(n_genes),
                "source_file": str(file_path),
                "source_dir": "GEO-DataHub/h5ad_output",
                "gse_id": gse_id,
                "organism": geo.get("organism", ""),
                "geo_title": geo.get("title", ""),
            }
            report.append({
                "file": file_path.name,
                "gse_id": gse_id,
                "dataset_id": dataset_id,
                "status": "kept_existing",
                "n_cells": int(n_cells),
                "n_genes": int(n_genes),
            })
            processed += 1
            total_cells += int(n_cells)
            print(f"[{idx}/{len(files)}] {file_path.name}: kept existing")
            continue

        print(f"[{idx}/{len(files)}] {file_path.name}")
        adata, load_note = read_h5ad_with_subsample(file_path, args.max_cells, args.random_seed)
        if adata is None:
            skipped += 1
            report.append({
                "file": file_path.name,
                "gse_id": gse_id,
                "dataset_id": dataset_id,
                "status": "skipped",
                "reason": load_note,
            })
            print(f"  -> skipped: {load_note}")
            continue

        adata, count_note = extract_count_like_matrix(adata)
        if adata is None:
            skipped += 1
            report.append({
                "file": file_path.name,
                "gse_id": gse_id,
                "dataset_id": dataset_id,
                "status": "skipped",
                "reason": count_note,
                "load_note": load_note,
            })
            print(f"  -> skipped: {count_note}")
            continue

        adata, prep_note = preprocess_adata(
            adata,
            dataset_id=dataset_id,
            min_genes=args.min_genes,
            min_cells=args.min_cells,
            min_final_cells=args.min_final_cells,
            n_top_genes=args.n_top_genes,
            target_sum=args.target_sum,
        )

        if adata is None:
            skipped += 1
            report.append({
                "file": file_path.name,
                "gse_id": gse_id,
                "dataset_id": dataset_id,
                "status": "skipped",
                "reason": prep_note,
                "load_note": load_note,
                "count_note": count_note,
            })
            print(f"  -> skipped: {prep_note}")
            continue

        adata.obs["gse_id"] = gse_id
        adata.obs["source_name"] = file_path.name
        adata.obs["integration_source"] = "GEO-DataHub"

        adata.write_h5ad(out_path)

        meta = load_dataset_meta(meta_root, gse_id)
        geo = meta.get("geo_metadata", {}) if isinstance(meta, dict) else {}
        text = build_text_description(gse_id, meta)

        n_cells, n_genes = adata.shape
        metadata_new[dataset_id] = {
            "text": text,
            "n_cells": int(n_cells),
            "n_genes": int(n_genes),
            "source_file": str(file_path),
            "source_dir": "GEO-DataHub/h5ad_output",
            "gse_id": gse_id,
            "organism": geo.get("organism", ""),
            "geo_title": geo.get("title", ""),
        }

        report.append({
            "file": file_path.name,
            "gse_id": gse_id,
            "dataset_id": dataset_id,
            "status": "processed",
            "n_cells": int(n_cells),
            "n_genes": int(n_genes),
            "load_note": load_note,
            "count_note": count_note,
            "prep_note": prep_note,
            "output": str(out_path),
        })

        processed += 1
        total_cells += int(n_cells)
        print(f"  -> processed: {n_cells} cells x {n_genes} genes")

    geodh_meta_path = output_dir / "metadata_geodh_integrated.json"
    with open(geodh_meta_path, "w") as f:
        json.dump(metadata_new, f, indent=2, ensure_ascii=False)

    merged_main_count = 0
    if main_metadata.exists():
        with open(main_metadata) as f:
            merged = json.load(f)
    else:
        merged = {}

    merged.update(metadata_new)
    merged_main_count = len(merged)

    with open(main_metadata, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    report_path = output_dir / "geodh_integration_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "source_dir": str(source_dir),
                "selection": args.selection,
                "candidate_files": len(files),
                "processed": processed,
                "skipped": skipped,
                "total_cells_added_or_kept": total_cells,
                "metadata_entries_added_or_updated": len(metadata_new),
                "main_metadata_entries": merged_main_count,
                "details": report,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 72)
    print("GEO integration complete")
    print("=" * 72)
    print(f"Processed:                 {processed}")
    print(f"Skipped:                   {skipped}")
    print(f"Cells (added/kept):        {total_cells:,}")
    print(f"New metadata file:         {geodh_meta_path}")
    print(f"Main metadata updated:     {main_metadata}")
    print(f"Integration report:        {report_path}")
    print("Next: run scripts/03_cache_latents.py to include integrated datasets in training cache")


if __name__ == "__main__":
    main()
