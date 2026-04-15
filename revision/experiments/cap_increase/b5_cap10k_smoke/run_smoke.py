"""Cap10k smoke test — 5 datasets.

Runs the standard preprocess_adata flow with max_cells=10000 on five
representative cap-binding merged h5ad files from the archived
GEO-DataHub source, writing output to an isolated directory to avoid
any possibility of overwriting the baseline cap3k artifacts.

Purpose: verify the pipeline still runs end-to-end with a higher cap,
measure wall-clock per dataset, and check the post-HVG shape is
sensible (~10k x 2000).

Inputs (read-only):
  /home/zeyufu/Desktop/.archive/GEO-DataHub/h5ad_output/*.h5ad

Outputs:
  data/processed_h5ad_cap10k_smoke/<dataset>_cap10k.h5ad
  revision/experiments/cap_increase/b5_cap10k_smoke/smoke_report.json
"""

from __future__ import annotations

import gc
import json
import sys
import time
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

import scanpy as sc  # noqa: E402

from scripts.data_prep import __name__ as _  # ensure package path
# we re-implement preprocess_adata here to keep the smoke test
# independent of any future changes to 00_prepare_all_data.py
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402


SOURCE = Path("/home/zeyufu/Desktop/.archive/GEO-DataHub/h5ad_output")
OUT = REPO_ROOT / "data" / "processed_h5ad_cap10k_smoke"
MAX_CELLS = 10000

DATASETS = [
    # (gse_id, file, raw_cell_count_note)
    ("GSE308145", "GSE308145_merged.h5ad",   643442),
    ("GSE272187", "GSE272187_merged.h5ad",    27090),
    ("GSE248214", "GSE248214_merged.h5ad",    46953),
    ("GSE227644", "GSE227644_merged.h5ad",    52020),
    ("GSE279781", "GSE279781_merged.h5ad",   150628),
]


def preprocess(adata, dataset_id, n_top_genes=2000, target_sum=1e4):
    if sp.issparse(adata.X):
        if adata.X.dtype in (np.int64, np.int32):
            adata.X = adata.X.astype(np.float32)
    else:
        if adata.X.dtype in (np.int64, np.int32):
            adata.X = adata.X.astype(np.float32)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = (adata.var_names.str.startswith("MT-") |
                       adata.var_names.str.startswith("mt-"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    if adata.shape[0] < 50:
        return None
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    if adata.shape[1] > n_top_genes:
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_top_genes, flavor="seurat_v3",
                layer="counts", subset=False,
            )
        except Exception:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_top_genes, flavor="seurat",
                subset=False,
            )
        adata = adata[:, adata.var["highly_variable"]].copy()
    return adata


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for gse, filename, raw_note in DATASETS:
        src = SOURCE / filename
        dst = OUT / f"{gse}_cap10k_processed.h5ad"
        record = {
            "gse_id": gse,
            "source_file": str(src),
            "source_bytes": src.stat().st_size if src.exists() else None,
            "raw_cell_count_from_prior_note": raw_note,
            "max_cells_target": MAX_CELLS,
        }
        if not src.exists():
            record["status"] = "missing_source"
            reports.append(record); continue
        try:
            t0 = time.time()
            tracemalloc.start()
            adata = sc.read_h5ad(src)
            record["raw_shape_loaded"] = [int(adata.shape[0]),
                                           int(adata.shape[1])]
            t_load = time.time() - t0

            if adata.shape[0] > MAX_CELLS:
                sc.pp.subsample(adata, n_obs=MAX_CELLS)
            record["post_subsample_shape"] = [int(adata.shape[0]),
                                                int(adata.shape[1])]

            t1 = time.time()
            adata = preprocess(adata, gse)
            t_preproc = time.time() - t1

            if adata is None:
                record["status"] = "dropped_after_qc"
                reports.append(record); continue

            record["post_hvg_shape"] = [int(adata.shape[0]),
                                          int(adata.shape[1])]

            # Save isolated
            t2 = time.time()
            adata.write(dst)
            t_save = time.time() - t2

            cur, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            record["load_seconds"] = round(t_load, 2)
            record["preprocess_seconds"] = round(t_preproc, 2)
            record["save_seconds"] = round(t_save, 2)
            record["total_seconds"] = round(time.time() - t0, 2)
            record["peak_traced_memory_mb"] = round(peak / 1024 ** 2, 1)
            record["output_bytes"] = dst.stat().st_size
            record["status"] = "ok"
            print(f"[{gse}] OK  shape={record['post_hvg_shape']}  "
                  f"t={record['total_seconds']}s  "
                  f"peak={record['peak_traced_memory_mb']}MB")
        except Exception as e:
            record["status"] = "error"
            record["error"] = repr(e)
            print(f"[{gse}] FAIL  {e!r}")
        finally:
            try:
                del adata
            except Exception:
                pass
            gc.collect()
        reports.append(record)

    summary = {
        "max_cells": MAX_CELLS,
        "source_dir": str(SOURCE),
        "output_dir": str(OUT),
        "n_datasets": len(reports),
        "n_ok": sum(1 for r in reports if r.get("status") == "ok"),
        "results": reports,
    }
    (HERE / "smoke_report.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    # preview
    lines = ["cap10k smoke test",
             "=================",
             f"datasets={summary['n_datasets']}  "
             f"ok={summary['n_ok']}",
             f"{'gse':<10s} {'status':<12s} {'post_hvg':>14s} "
             f"{'t_s':>8s} {'peak_MB':>9s}"]
    for r in reports:
        shape = r.get("post_hvg_shape", ["-", "-"])
        lines.append(
            f"{r['gse_id']:<10s} {r.get('status','-'):<12s} "
            f"{str(shape):>14s} "
            f"{r.get('total_seconds','-')!s:>8} "
            f"{r.get('peak_traced_memory_mb','-')!s:>9}"
        )
    preview = "\n".join(lines) + "\n"
    (HERE / "smoke_report_preview.txt").write_text(preview)
    print("\n" + preview)


if __name__ == "__main__":
    main()
