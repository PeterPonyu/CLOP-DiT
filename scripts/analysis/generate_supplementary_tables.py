#!/usr/bin/env python3
"""Generate supplementary tables S1, S2, S3 for the CLOP-DiT manuscript.

Outputs:
    articles/supplementary_tables.tex  — LaTeX tables (booktabs) for inclusion
    results/supplementary_data.json    — Raw data in machine-readable form

Table S1: All 80 GEO dataset identifiers with cell counts and descriptions
Table S2: 8 held-out validation datasets (dataset-level split, seed=42)
Table S3: Full CFG-scale / solver / step sweep results

Usage:
    python scripts/analysis/generate_supplementary_tables.py
"""

import json
import re
import textwrap
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METADATA_STRUCTURED = PROJECT_ROOT / "data/processed_h5ad/metadata_structured.json"
METADATA_CACHE = PROJECT_ROOT / "data/cached_latents/metadata.json"
SAMPLE_IDS_PATH = PROJECT_ROOT / "data/cached_latents/sample_ids.npy"
EVAL_RESULTS = PROJECT_ROOT / "results/archive/v5_final/final_evaluation.json"
OUTPUT_TEX = PROJECT_ROOT / "articles/supplementary_tables.tex"
OUTPUT_JSON = PROJECT_ROOT / "results/supplementary_data.json"

# Dataset processing order (alphabetical h5ad filenames from cache build)
DATASET_ORDER = [
    "GSE103867_geodh", "GSE110686_geodh", "GSE115571_LPSMmDev",
    "GSE117988_MCCPBMCCancer", "GSE117988_MCCTumorCancer", "GSE120505_bloodAged",
    "GSE123813_bccHmCancer", "GSE123813_sccHmCancer", "GSE123902_LungAdreHmCancer",
    "GSE124310_MMHmCancer", "GSE130148_LungHmDev", "GSE132509_acutelymluekPBMCHmCancer",
    "GSE138709_LiverCancer", "GSE142653pitHmDev", "GSE143423_lbm_CancerBrainHm",
    "GSE143423_tnbc_CancerBrainHm", "GSE145929_ProgastinMmDev", "GSE145929_UrineMmDev",
    "GSE148218_bmALLHmCancer", "GSE149655_CAHmCancer", "GSE155109_bcECHmCancer",
    "GSE155109_bcStromaHmCancer", "GSE163558_stomachHmCancer", "GSE165784_RetinaHmDev",
    "GSE165844_LSKMmBatch", "GSE167597_spineMm", "GSE168181_BreastHmCancer",
    "GSE175975_geodh", "GSE183904_GastricHmCancer", "GSE189070_astrocytesSCIMmDev",
    "GSE189357_lungAdreHmCancer", "GSE192857_hESCHmTimes", "GSE212502_scRNA25100",
    "GSE213740_ADHm", "GSE220913_scRNA25100", "GSE222002_TcellsHmCancer",
    "GSE222369_NKsLymphomaHmCancer", "GSE225600_breast_CancerHm",
    "GSE225857_liverColonMetasisHmCancer", "GSE226131_HSCMmAged",
    "GSE226762_geodh", "GSE227644_geodh", "GSE227719_scRNA25100",
    "GSE228499_breastHmCancer", "GSE235787_bcellsALLHmCancer",
    "GSE236565_scRNA25100", "GSE248214_geodh", "GSE253355_bmNicheHm",
    "GSE262288_breastMetasisHmCancer", "GSE272187_geodh", "GSE275119_TeethMmDev",
    "GSE277740_scRNA25100", "GSE279781_geodh", "GSE280847_scRNA25100",
    "GSE283205_hepatoblastomaCancer", "GSE283397_scRNA25100", "GSE288211_scRNA25100",
    "GSE289611_scRNA25100", "GSE291166_scRNA25100", "GSE299623_geodh",
    "GSE300862_scRNA25100", "GSE302433_scRNA25100", "GSE303309_scRNA25100",
    "GSE306567_scRNA25100", "GSE306676_scRNA25100", "GSE307261_scRNA25100",
    "GSE307774_scRNA25100", "GSE308428_scRNA25100", "GSE309368_scRNA25100",
    "GSE315628_geodh", "GSE318353_geodh", "GSE98638_TcellLiverHmCancer",
    "bm_GSE120446", "dentate", "endo", "hESC_GSE144024", "hemato",
    "ifnHSPC_GSE226824", "lung", "setty",
]


def extract_geo_accession(dataset_key: str) -> str:
    """Extract GEO accession (GSExxxxxx) from dataset key."""
    match = re.search(r"(GSE\d+)", dataset_key)
    if match:
        return match.group(1)
    # Datasets without explicit GSE in the key
    geo_map = {
        "dentate": "GSE104323",
        "endo": "GSE75748",
        "hemato": "GSE72857",
        "lung": "GSE141259",
        "setty": "GSE139369",
    }
    return geo_map.get(dataset_key, dataset_key)


def make_short_description(text: str, max_len: int = 80) -> str:
    """Truncate description for table display."""
    # Take first sentence or truncate
    first_sent = text.split(". ")[0]
    if len(first_sent) > max_len:
        return first_sent[:max_len - 3] + "..."
    return first_sent


def escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def compute_dataset_split(n_datasets: int = 80, val_split: float = 0.1,
                           seed: int = 42):
    """Replicate the dataset-level split from src/data_pipeline/dataset.py.

    Returns (train_ids, val_ids) as sorted lists of integer dataset indices.
    """
    unique_ids = np.arange(n_datasets)
    rng = np.random.default_rng(seed)
    shuffled_ids = unique_ids.copy()
    rng.shuffle(shuffled_ids)
    n_val = max(1, int(n_datasets * val_split))
    val_ids = sorted(shuffled_ids[:n_val].tolist())
    train_ids = sorted(shuffled_ids[n_val:].tolist())
    return train_ids, val_ids


def load_metadata():
    """Load dataset metadata from metadata_structured.json."""
    with open(METADATA_STRUCTURED) as f:
        return json.load(f)


def load_eval_results():
    """Load evaluation results from final_evaluation.json."""
    with open(EVAL_RESULTS) as f:
        return json.load(f)


def build_dataset_table(metadata: dict, train_ids: list, val_ids: list):
    """Build the full dataset table (Table S1).

    Returns list of dicts with: idx, geo_accession, description, n_cells, split.
    """
    val_set = set(val_ids)
    rows = []
    for idx, dataset_key in enumerate(DATASET_ORDER):
        info = metadata.get(dataset_key, {})
        n_cells = info.get("n_cells", "?")
        text = info.get("text", "")
        geo = extract_geo_accession(dataset_key)
        split = "Val" if idx in val_set else "Train"
        short_desc = make_short_description(text, max_len=90)
        rows.append({
            "idx": idx + 1,
            "geo_accession": geo,
            "dataset_key": dataset_key,
            "description": short_desc,
            "full_description": text,
            "n_cells": n_cells,
            "split": split,
        })
    return rows


def build_val_table(metadata: dict, val_ids: list):
    """Build the validation dataset table (Table S2)."""
    rows = []
    total_cells = 0
    for rank, idx in enumerate(val_ids, 1):
        dataset_key = DATASET_ORDER[idx]
        info = metadata.get(dataset_key, {})
        n_cells = info.get("n_cells", 0)
        text = info.get("text", "")
        geo = extract_geo_accession(dataset_key)
        short_desc = make_short_description(text, max_len=100)
        total_cells += n_cells
        rows.append({
            "rank": rank,
            "geo_accession": geo,
            "dataset_key": dataset_key,
            "description": short_desc,
            "full_description": text,
            "n_cells": n_cells,
            "sample_idx": idx,
        })
    return rows, total_cells


def build_cfg_table(eval_data: dict):
    """Build the CFG sweep table (Table S3)."""
    rows = []

    # Main results (named configs)
    for config_name, metrics in eval_data.get("results", {}).items():
        if config_name == "Gaussian":
            solver = "---"
            steps = "---"
            cfg = "---"
        else:
            parts = config_name.split()
            cfg = parts[0].replace("CFG=", "") if "CFG=" in parts[0] else "---"
            if len(parts) > 1:
                solver_steps = parts[1].split("-")
                solver = solver_steps[0]
                steps = solver_steps[1] if len(solver_steps) > 1 else "---"
            else:
                solver = "---"
                steps = "---"

        rows.append({
            "config": config_name,
            "cfg_scale": cfg,
            "solver": solver,
            "steps": steps,
            "knn_top1": metrics.get("knn_top1", 0),
            "knn_top5": metrics.get("knn_top5", 0),
            "steering": metrics.get("steering", 0),
            "diversity_ratio": metrics.get("diversity_ratio", 0),
            "linear_acc": metrics.get("linear_acc", 0),
            "fd": metrics.get("fd", 0),
        })

    # Fine-grained CFG sweep
    sweep_rows = []
    for entry in eval_data.get("cfg_sweep", []):
        sweep_rows.append({
            "config": f"CFG={entry['cfg']} Euler-{entry['steps']}",
            "cfg_scale": str(entry["cfg"]),
            "solver": "Euler",
            "steps": str(entry["steps"]),
            "knn_top1": entry.get("knn1", 0),
            "knn_top5": entry.get("knn5", 0),
            "steering": entry.get("steering", 0),
            "diversity_ratio": entry.get("div_ratio", 0),
            "linear_acc": "---",
            "fd": entry.get("fd", 0),
        })

    return rows, sweep_rows


def generate_latex(dataset_rows, val_rows, val_total_cells,
                   cfg_rows, cfg_sweep_rows, eval_metadata):
    """Generate the complete LaTeX file with all three supplementary tables."""

    lines = []
    lines.append(r"% Supplementary Tables for CLOP-DiT manuscript")
    lines.append(r"% Auto-generated by scripts/analysis/generate_supplementary_tables.py")
    lines.append(r"% Do not edit manually — re-run the script to regenerate.")
    lines.append(r"")
    lines.append(r"\RequirePackage{booktabs}")
    lines.append(r"\RequirePackage{longtable}")
    lines.append(r"\RequirePackage{array}")
    lines.append(r"\RequirePackage{caption}")
    lines.append(r"")

    # ── Table S1: All 80 datasets ──
    lines.append(r"% ============================================================")
    lines.append(r"% Table S1: All 80 GEO Dataset Identifiers")
    lines.append(r"% ============================================================")
    lines.append(r"\begin{longtable}{rlp{6.5cm}rr}")
    lines.append(r"\caption{All 80 GEO datasets used in CLOP-DiT training and validation. "
                 r"Datasets were collected from the Gene Expression Omnibus (GEO) and "
                 r"preprocessed to a uniform format of 2{,}000 highly variable genes per dataset. "
                 r"The Split column indicates whether the dataset was used for training (Train) "
                 r"or held out for validation (Val).}")
    lines.append(r"\label{tab:s1_datasets} \\")
    lines.append(r"\toprule")
    lines.append(r"\# & GEO Accession & Description & $n_{\text{cells}}$ & Split \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"\# & GEO Accession & Description & $n_{\text{cells}}$ & Split \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{r}{\textit{Continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    total_cells = 0
    for row in dataset_rows:
        desc = escape_latex(row["description"])
        n = row["n_cells"]
        total_cells += n if isinstance(n, int) else 0
        lines.append(
            f"  {row['idx']} & {row['geo_accession']} & "
            f"{desc} & "
            f"{n:,} & {row['split']} \\\\"
        )

    lines.append(r"\midrule")
    lines.append(f"  & \\textbf{{Total}} & & \\textbf{{{total_cells:,}}} & \\\\")
    lines.append(r"\end{longtable}")
    lines.append(r"")

    # ── Table S2: 8 validation datasets ──
    lines.append(r"% ============================================================")
    lines.append(r"% Table S2: 8 Held-out Validation Datasets")
    lines.append(r"% ============================================================")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{The 8 held-out validation datasets, selected by study-level split "
                 r"(seed\,=\,42, val\_split\,=\,0.1). No cells from these studies were seen "
                 r"during CLOP or DiT training, ensuring unbiased evaluation of "
                 r"cross-study generalization.}")
    lines.append(r"\label{tab:s2_validation}")
    lines.append(r"\begin{tabular}{rlp{7cm}r}")
    lines.append(r"\toprule")
    lines.append(r"\# & GEO Accession & Description & $n_{\text{cells}}$ \\")
    lines.append(r"\midrule")

    for row in val_rows:
        desc = escape_latex(row["description"])
        lines.append(
            f"  {row['rank']} & {row['geo_accession']} & "
            f"{desc} & "
            f"{row['n_cells']:,} \\\\"
        )

    lines.append(r"\midrule")
    lines.append(f"  & \\textbf{{Total}} & & \\textbf{{{val_total_cells:,}}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"")

    # ── Table S3: CFG sweep results ──
    lines.append(r"% ============================================================")
    lines.append(r"% Table S3: Full CFG Scale / Solver / Steps Sweep Results")
    lines.append(r"% ============================================================")
    lines.append(r"\begin{longtable}{lllrrrrrr}")
    lines.append(r"\caption{Full classifier-free guidance (CFG) scale sweep results. "
                 r"Each configuration was evaluated by generating 200 cells for each of "
                 f"100 evaluation groups ({eval_metadata.get('n_cells', 220304):,} total reference cells). "
                 r"KNN-1 and KNN-5 measure type-identity preservation (higher is better), "
                 r"Steering measures text--cell alignment, DivR is the diversity ratio "
                 r"(1.0\,=\,real data diversity), LinAcc is linear classifier accuracy on "
                 r"generated cells, and FD is the Fr\'{e}chet distance to real data "
                 r"(lower is better, 0\,=\,identical).}")
    lines.append(r"\label{tab:s3_cfg_sweep} \\")
    lines.append(r"\toprule")
    lines.append(r"CFG & Solver & Steps & KNN-1 & KNN-5 & Steer. & DivR & LinAcc & FD \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"CFG & Solver & Steps & KNN-1 & KNN-5 & Steer. & DivR & LinAcc & FD \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{r}{\textit{Continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    # Section header: Named configurations
    lines.append(r"\multicolumn{9}{l}{\textit{Primary configurations (from Table~2 in main text)}} \\")
    lines.append(r"\midrule")

    for row in cfg_rows:
        la = row["linear_acc"]
        la_str = f"{la:.4f}" if isinstance(la, float) else str(la)
        fd = row["fd"]
        fd_str = f"{fd:.2f}" if isinstance(fd, float) else str(fd)
        dr = row["diversity_ratio"]
        dr_str = f"{dr:.4f}" if isinstance(dr, float) else str(dr)

        lines.append(
            f"  {row['cfg_scale']} & {row['solver']} & {row['steps']} & "
            f"{row['knn_top1']:.4f} & {row['knn_top5']:.4f} & "
            f"{row['steering']:.3f} & {dr_str} & {la_str} & {fd_str} \\\\"
        )

    # Section header: Fine-grained sweep
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{l}{\textit{Fine-grained CFG sweep (Euler solver)}} \\")
    lines.append(r"\midrule")

    for row in cfg_sweep_rows:
        la = row["linear_acc"]
        la_str = f"{la:.4f}" if isinstance(la, float) else str(la)
        fd = row["fd"]
        fd_str = f"{fd:.2f}" if isinstance(fd, float) else str(fd)
        dr = row["diversity_ratio"]
        dr_str = f"{dr:.4f}" if isinstance(dr, float) else str(dr)

        lines.append(
            f"  {row['cfg_scale']} & {row['solver']} & {row['steps']} & "
            f"{row['knn_top1']:.4f} & {row['knn_top5']:.4f} & "
            f"{row['steering']:.3f} & {dr_str} & {la_str} & {fd_str} \\\\"
        )

    lines.append(r"\end{longtable}")
    lines.append(r"")

    return "\n".join(lines) + "\n"


def generate_json(dataset_rows, val_rows, val_total_cells,
                  cfg_rows, cfg_sweep_rows, train_ids, val_ids):
    """Generate the supplementary data JSON."""
    data = {
        "table_s1_all_datasets": [
            {
                "index": r["idx"],
                "geo_accession": r["geo_accession"],
                "dataset_key": r["dataset_key"],
                "description": r["full_description"],
                "n_cells": r["n_cells"],
                "split": r["split"],
            }
            for r in dataset_rows
        ],
        "table_s2_validation_datasets": {
            "seed": 42,
            "val_split": 0.1,
            "split_strategy": "dataset-level (by study)",
            "n_val_datasets": len(val_ids),
            "total_val_cells": val_total_cells,
            "val_dataset_indices": val_ids,
            "datasets": [
                {
                    "rank": r["rank"],
                    "geo_accession": r["geo_accession"],
                    "dataset_key": r["dataset_key"],
                    "description": r["full_description"],
                    "n_cells": r["n_cells"],
                    "sample_idx": r["sample_idx"],
                }
                for r in val_rows
            ],
        },
        "table_s3_cfg_sweep": {
            "primary_configs": [
                {
                    "config": r["config"],
                    "cfg_scale": r["cfg_scale"],
                    "solver": r["solver"],
                    "steps": r["steps"],
                    "knn_top1": r["knn_top1"],
                    "knn_top5": r["knn_top5"],
                    "steering": r["steering"],
                    "diversity_ratio": r["diversity_ratio"],
                    "linear_acc": r["linear_acc"],
                    "fd": r["fd"],
                }
                for r in cfg_rows
            ],
            "fine_grained_sweep": [
                {
                    "config": r["config"],
                    "cfg_scale": r["cfg_scale"],
                    "solver": r["solver"],
                    "steps": r["steps"],
                    "knn_top1": r["knn_top1"],
                    "knn_top5": r["knn_top5"],
                    "steering": r["steering"],
                    "diversity_ratio": r["diversity_ratio"],
                    "fd": r["fd"],
                }
                for r in cfg_sweep_rows
            ],
        },
    }
    return data


def main():
    print("=" * 60)
    print("Generating Supplementary Tables for CLOP-DiT")
    print("=" * 60)

    # ── Load data ──
    print("\n[1/5] Loading metadata...")
    metadata = load_metadata()
    print(f"  Loaded {len(metadata)} datasets from metadata_structured.json")

    print("\n[2/5] Computing dataset-level split (seed=42, val_split=0.1)...")
    train_ids, val_ids = compute_dataset_split(
        n_datasets=len(DATASET_ORDER), val_split=0.1, seed=42
    )
    print(f"  Train: {len(train_ids)} datasets, Val: {len(val_ids)} datasets")
    print(f"  Validation dataset indices: {val_ids}")
    print(f"  Validation datasets:")
    for vid in val_ids:
        dk = DATASET_ORDER[vid]
        geo = extract_geo_accession(dk)
        nc = metadata.get(dk, {}).get("n_cells", "?")
        print(f"    [{vid}] {geo} ({dk}) — {nc} cells")

    print("\n[3/5] Loading evaluation results...")
    eval_data = load_eval_results()
    eval_metadata = eval_data.get("metadata", {})
    print(f"  {len(eval_data.get('results', {}))} primary configs")
    print(f"  {len(eval_data.get('cfg_sweep', []))} sweep entries")

    # ── Build tables ──
    print("\n[4/5] Building tables...")
    dataset_rows = build_dataset_table(metadata, train_ids, val_ids)
    val_rows, val_total_cells = build_val_table(metadata, val_ids)
    cfg_rows, cfg_sweep_rows = build_cfg_table(eval_data)

    # Verify cell counts match article
    train_cells = sum(r["n_cells"] for r in dataset_rows if r["split"] == "Train")
    print(f"  Table S1: {len(dataset_rows)} datasets")
    print(f"    Train: {train_cells:,} cells, Val: {val_total_cells:,} cells")
    print(f"    Total: {train_cells + val_total_cells:,} cells")
    print(f"  Table S2: {len(val_rows)} validation datasets ({val_total_cells:,} cells)")
    print(f"  Table S3: {len(cfg_rows)} primary + {len(cfg_sweep_rows)} sweep configs")

    # ── Write outputs ──
    print("\n[5/5] Writing output files...")

    latex_content = generate_latex(
        dataset_rows, val_rows, val_total_cells,
        cfg_rows, cfg_sweep_rows, eval_metadata,
    )
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(latex_content, encoding="utf-8")
    print(f"  LaTeX: {OUTPUT_TEX}")

    json_data = generate_json(
        dataset_rows, val_rows, val_total_cells,
        cfg_rows, cfg_sweep_rows, train_ids, val_ids,
    )
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  JSON:  {OUTPUT_JSON}")

    print("\nDone.")


if __name__ == "__main__":
    main()
