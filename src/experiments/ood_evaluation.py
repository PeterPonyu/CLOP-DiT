#!/usr/bin/env python3
# ood_evaluation.py — Out-of-distribution prompt robustness evaluation
"""
Evaluate CLOP-DiT generation quality on out-of-distribution text prompts.

This module tests two OOD scenarios:
  1. Novel cell types not seen during training (evaluate_novel_types).
  2. Free-form / informal text descriptions vs. structured templates
     (evaluate_free_form).

Produces:
  - results/ood_evaluation/ood_results.json   (quantitative metrics)
  - results/ood_evaluation/fig_ood_robustness.pdf  (heatmap summary)

Usage (CLI):
    python -m src.experiments.ood_evaluation \
        --prompts_yaml configs/ood_test_prompts.yaml \
        --reference_h5ad data/processed_h5ad/reference.h5ad \
        --output_dir results/ood_evaluation \
        --num_cells 50
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import anndata as ad
import scanpy as sc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score

# ── Project path setup ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.visualization.style import COLORS as VIZ_COLORS, apply_style, save_with_vcd

# ── Load CLOPDiTInference from numbered script via importlib ───────────
_INFERENCE_SCRIPT = _PROJECT_ROOT / "scripts" / "05_inference.py"
_spec = importlib.util.spec_from_file_location("inference", str(_INFERENCE_SCRIPT))
_inference_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_inference_mod)
CLOPDiTInference = _inference_mod.CLOPDiTInference

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _to_dense(x: np.ndarray) -> np.ndarray:
    """Convert sparse matrix to dense if necessary."""
    return x.A if hasattr(x, "A") else np.asarray(x)


def _load_prompts_yaml(path: str | Path) -> dict:
    """Load OOD prompts from a YAML config file.

    Expected YAML structure::

        novel_types:
          - name: "Megakaryocyte"
            prompt: "Megakaryocytes from human bone marrow ..."
            reference_type: "Platelets"
        free_form:
          - name: "informal_tcell"
            prompt: "those killer T cells in a lung tumor"
            structured_equivalent: "CD8+ cytotoxic T lymphocytes ..."
            reference_type: "CD8+ T cells"

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed prompt configuration.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts YAML not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_embedding_distance(
    generated_emb: np.ndarray,
    prototype: np.ndarray,
) -> dict:
    """Compute distance metrics between generated embeddings and a training prototype.

    Parameters
    ----------
    generated_emb : (N, D) array
        Generated cell embeddings.
    prototype : (D,) array
        Mean real cell embedding for the reference type.

    Returns
    -------
    dict with "cosine_sim_mean", "cosine_sim_std", "l2_dist_mean", "l2_dist_std".
    """
    # Normalise for cosine similarity
    gen_norm = generated_emb / (np.linalg.norm(generated_emb, axis=1, keepdims=True) + 1e-12)
    proto_norm = prototype / (np.linalg.norm(prototype) + 1e-12)

    cosine_sims = gen_norm @ proto_norm
    l2_dists = np.linalg.norm(generated_emb - prototype[None, :], axis=1)

    return {
        "cosine_sim_mean": float(np.mean(cosine_sims)),
        "cosine_sim_std": float(np.std(cosine_sims)),
        "l2_dist_mean": float(np.mean(l2_dists)),
        "l2_dist_std": float(np.std(l2_dists)),
    }


def _compute_expression_metrics(
    generated_adata: ad.AnnData,
    reference_adata: ad.AnnData,
    reference_type: str,
    cell_type_key: str = "cell_type",
) -> dict:
    """Compute Pearson r and R-squared between generated and reference mean expression.

    Parameters
    ----------
    generated_adata : AnnData
        Generated cells with gene expression in .X.
    reference_adata : AnnData
        Reference dataset.
    reference_type : str
        Cell type label to filter reference cells.
    cell_type_key : str
        Column in reference_adata.obs with cell type labels.

    Returns
    -------
    dict with "pearson_r", "pearson_p", "r2", "num_genes_compared".
    """
    # Get reference cells of matching type
    if cell_type_key not in reference_adata.obs.columns:
        logger.warning(
            "cell_type_key '%s' not found in reference. Falling back to all cells.",
            cell_type_key,
        )
        ref_subset = reference_adata
    else:
        mask = reference_adata.obs[cell_type_key] == reference_type
        if mask.sum() == 0:
            logger.warning(
                "No cells found for type '%s'. Falling back to all cells.", reference_type,
            )
            ref_subset = reference_adata
        else:
            ref_subset = reference_adata[mask]

    # Align genes
    shared_genes = list(set(generated_adata.var_names) & set(ref_subset.var_names))
    if len(shared_genes) < 10:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "r2": float("nan"),
            "num_genes_compared": len(shared_genes),
        }

    gen_expr = _to_dense(generated_adata[:, shared_genes].X)
    ref_expr = _to_dense(ref_subset[:, shared_genes].X)

    gen_mean = np.mean(gen_expr, axis=0)
    ref_mean = np.mean(ref_expr, axis=0)

    r, p = stats.pearsonr(gen_mean, ref_mean)
    r2 = r2_score(ref_mean, gen_mean)

    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "r2": float(r2),
        "num_genes_compared": len(shared_genes),
    }


def _build_training_prototypes(
    reference_adata: ad.AnnData,
    embedding_key: str = "X_clop_dit",
    cell_type_key: str = "cell_type",
) -> dict[str, np.ndarray]:
    """Compute mean embedding per cell type from reference data.

    Parameters
    ----------
    reference_adata : AnnData
        Reference dataset with cell embeddings in .obsm[embedding_key].
    embedding_key : str
        Key in .obsm for cell embeddings.
    cell_type_key : str
        Column in .obs for cell type labels.

    Returns
    -------
    dict mapping cell type name to (D,) mean embedding.
    """
    if embedding_key not in reference_adata.obsm:
        raise KeyError(
            f"Embedding key '{embedding_key}' not found in reference_adata.obsm. "
            f"Available keys: {list(reference_adata.obsm.keys())}"
        )

    prototypes = {}
    for ct in reference_adata.obs[cell_type_key].unique():
        mask = reference_adata.obs[cell_type_key] == ct
        embs = reference_adata.obsm[embedding_key][mask.values]
        prototypes[ct] = np.mean(embs, axis=0)

    logger.info("Built %d training prototypes from reference data.", len(prototypes))
    return prototypes


# ═══════════════════════════════════════════════════════════════════════
# OODEvaluator
# ═══════════════════════════════════════════════════════════════════════

class OODEvaluator:
    """Evaluate CLOP-DiT robustness to out-of-distribution text prompts.

    Parameters
    ----------
    pipeline : CLOPDiTInference
        Loaded inference pipeline.
    """

    def __init__(self, pipeline: CLOPDiTInference):
        self.pipeline = pipeline

    @classmethod
    def from_checkpoints(
        cls,
        dit_checkpoint: str = "models/checkpoints/dit_best.pth",
        clop_checkpoint: str = "models/checkpoints/clop_best.pth",
        scgpt_model_dir: str = "models/scgpt_pancancer",
        device: str = "cuda",
    ) -> "OODEvaluator":
        """Convenience constructor that loads the inference pipeline.

        Parameters
        ----------
        dit_checkpoint : str
            Path to DiT checkpoint.
        clop_checkpoint : str
            Path to CLOP checkpoint.
        scgpt_model_dir : str
            Path to scGPT model directory.
        device : str
            Compute device.

        Returns
        -------
        OODEvaluator
        """
        pipeline = CLOPDiTInference(
            dit_checkpoint=dit_checkpoint,
            clop_checkpoint=clop_checkpoint,
            scgpt_model_dir=scgpt_model_dir,
            device=device,
        )
        return cls(pipeline)

    # ── Core evaluation methods ──────────────────────────────────────

    def evaluate_novel_types(
        self,
        prompts: list[dict],
        reference_adata: ad.AnnData,
        num_cells: int = 50,
        cell_type_key: str = "cell_type",
        embedding_key: str = "X_clop_dit",
    ) -> dict[str, dict]:
        """Evaluate generation quality for novel / unseen cell type prompts.

        Parameters
        ----------
        prompts : list of dict
            Each dict must have keys: "name", "prompt", "reference_type".
        reference_adata : AnnData
            Reference dataset with embeddings and expression.
        num_cells : int
            Number of cells to generate per prompt.
        cell_type_key : str
            Column in reference_adata.obs for cell type labels.
        embedding_key : str
            Key in reference_adata.obsm for cell embeddings.

        Returns
        -------
        dict mapping prompt name to metrics dict.
        """
        prototypes = _build_training_prototypes(
            reference_adata, embedding_key=embedding_key, cell_type_key=cell_type_key,
        )

        results = {}
        for entry in prompts:
            name = entry["name"]
            prompt_text = entry["prompt"]
            ref_type = entry.get("reference_type", None)

            logger.info("OOD novel-type evaluation: '%s'", name)

            # Generate cells
            gen_adata = self.pipeline.generate_adata(
                prompt=prompt_text,
                num_cells=num_cells,
                decode_expression=True,
                reference_adata=reference_adata,
            )

            metrics: Dict[str, Any] = {"prompt": prompt_text, "reference_type": ref_type}

            # Embedding distance to closest prototype
            if "X_clop_dit" in gen_adata.obsm:
                gen_emb = gen_adata.obsm["X_clop_dit"]

                if ref_type and ref_type in prototypes:
                    metrics["embedding_distance"] = _compute_embedding_distance(
                        gen_emb, prototypes[ref_type],
                    )
                else:
                    # Compute distance to all prototypes, report closest
                    best_type, best_dist = None, float("inf")
                    for ct, proto in prototypes.items():
                        d = np.mean(np.linalg.norm(gen_emb - proto[None, :], axis=1))
                        if d < best_dist:
                            best_type, best_dist = ct, d
                    metrics["closest_prototype"] = best_type
                    metrics["embedding_distance"] = _compute_embedding_distance(
                        gen_emb, prototypes[best_type],
                    )

            # Expression metrics
            if ref_type:
                metrics["expression_metrics"] = _compute_expression_metrics(
                    gen_adata, reference_adata, ref_type, cell_type_key=cell_type_key,
                )

            results[name] = metrics

        return results

    def evaluate_free_form(
        self,
        prompts: list[dict],
        reference_adata: ad.AnnData,
        num_cells: int = 50,
        cell_type_key: str = "cell_type",
        embedding_key: str = "X_clop_dit",
    ) -> dict[str, dict]:
        """Evaluate free-form / informal prompts vs. structured equivalents.

        Parameters
        ----------
        prompts : list of dict
            Each dict must have: "name", "prompt", "structured_equivalent",
            "reference_type".
        reference_adata : AnnData
            Reference dataset.
        num_cells : int
            Cells per prompt.
        cell_type_key : str
            Column in reference_adata.obs for cell type labels.
        embedding_key : str
            Key in reference_adata.obsm for cell embeddings.

        Returns
        -------
        dict mapping prompt name to metrics dict with keys
        "free_form_metrics", "structured_metrics", and "comparison".
        """
        prototypes = _build_training_prototypes(
            reference_adata, embedding_key=embedding_key, cell_type_key=cell_type_key,
        )

        results = {}
        for entry in prompts:
            name = entry["name"]
            free_text = entry["prompt"]
            struct_text = entry["structured_equivalent"]
            ref_type = entry.get("reference_type", None)

            logger.info("OOD free-form evaluation: '%s'", name)

            # Generate from free-form prompt
            gen_free = self.pipeline.generate_adata(
                prompt=free_text,
                num_cells=num_cells,
                decode_expression=True,
                reference_adata=reference_adata,
            )

            # Generate from structured equivalent
            gen_struct = self.pipeline.generate_adata(
                prompt=struct_text,
                num_cells=num_cells,
                decode_expression=True,
                reference_adata=reference_adata,
            )

            entry_results: Dict[str, Any] = {
                "free_form_prompt": free_text,
                "structured_prompt": struct_text,
                "reference_type": ref_type,
            }

            # Embedding metrics for free-form
            free_metrics: Dict[str, Any] = {}
            struct_metrics: Dict[str, Any] = {}

            if ref_type and ref_type in prototypes:
                proto = prototypes[ref_type]

                if "X_clop_dit" in gen_free.obsm:
                    free_metrics["embedding_distance"] = _compute_embedding_distance(
                        gen_free.obsm["X_clop_dit"], proto,
                    )
                if "X_clop_dit" in gen_struct.obsm:
                    struct_metrics["embedding_distance"] = _compute_embedding_distance(
                        gen_struct.obsm["X_clop_dit"], proto,
                    )

            # Expression metrics
            if ref_type:
                free_metrics["expression_metrics"] = _compute_expression_metrics(
                    gen_free, reference_adata, ref_type, cell_type_key=cell_type_key,
                )
                struct_metrics["expression_metrics"] = _compute_expression_metrics(
                    gen_struct, reference_adata, ref_type, cell_type_key=cell_type_key,
                )

            entry_results["free_form_metrics"] = free_metrics
            entry_results["structured_metrics"] = struct_metrics

            # Comparison: relative drop in quality
            comparison: Dict[str, Any] = {}
            if "expression_metrics" in free_metrics and "expression_metrics" in struct_metrics:
                fr = free_metrics["expression_metrics"].get("pearson_r", float("nan"))
                sr = struct_metrics["expression_metrics"].get("pearson_r", float("nan"))
                if not (np.isnan(fr) or np.isnan(sr)):
                    comparison["pearson_r_delta"] = float(fr - sr)
                    comparison["pearson_r_ratio"] = float(fr / sr) if sr != 0 else float("nan")

                f_r2 = free_metrics["expression_metrics"].get("r2", float("nan"))
                s_r2 = struct_metrics["expression_metrics"].get("r2", float("nan"))
                if not (np.isnan(f_r2) or np.isnan(s_r2)):
                    comparison["r2_delta"] = float(f_r2 - s_r2)

            if "embedding_distance" in free_metrics and "embedding_distance" in struct_metrics:
                f_cos = free_metrics["embedding_distance"]["cosine_sim_mean"]
                s_cos = struct_metrics["embedding_distance"]["cosine_sim_mean"]
                comparison["cosine_sim_delta"] = float(f_cos - s_cos)

            entry_results["comparison"] = comparison

            results[name] = entry_results

        return results

    # ── Reporting ────────────────────────────────────────────────────

    def generate_report(
        self,
        results: dict,
        output_dir: str | Path,
    ) -> Path:
        """Save JSON results and generate a heatmap summary figure.

        Parameters
        ----------
        results : dict
            Combined results from evaluate_novel_types and/or evaluate_free_form.
            Expected structure: {"novel_types": {...}, "free_form": {...}}.
        output_dir : str or Path
            Directory for outputs.

        Returns
        -------
        Path to the saved figure.
        """
        apply_style()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Save JSON ────────────────────────────────────────────────
        json_path = output_dir / "ood_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Saved OOD results to %s", json_path)

        # ── Build heatmap data ───────────────────────────────────────
        metric_keys = ["pearson_r", "r2", "cosine_sim_mean", "l2_dist_mean"]
        metric_labels = ["Pearson r", "R-squared", "Cosine Sim.", "L2 Distance"]

        all_rows: list[dict] = []

        # Collect novel_types results
        for name, m in results.get("novel_types", {}).items():
            row: Dict[str, Any] = {"prompt": name}
            if "expression_metrics" in m:
                row["pearson_r"] = m["expression_metrics"].get("pearson_r", float("nan"))
                row["r2"] = m["expression_metrics"].get("r2", float("nan"))
            if "embedding_distance" in m:
                row["cosine_sim_mean"] = m["embedding_distance"].get(
                    "cosine_sim_mean", float("nan"),
                )
                row["l2_dist_mean"] = m["embedding_distance"].get(
                    "l2_dist_mean", float("nan"),
                )
            all_rows.append(row)

        # Collect free_form results (use free-form metrics)
        for name, m in results.get("free_form", {}).items():
            row = {"prompt": f"{name} (free-form)"}
            fm = m.get("free_form_metrics", {})
            if "expression_metrics" in fm:
                row["pearson_r"] = fm["expression_metrics"].get("pearson_r", float("nan"))
                row["r2"] = fm["expression_metrics"].get("r2", float("nan"))
            if "embedding_distance" in fm:
                row["cosine_sim_mean"] = fm["embedding_distance"].get(
                    "cosine_sim_mean", float("nan"),
                )
                row["l2_dist_mean"] = fm["embedding_distance"].get(
                    "l2_dist_mean", float("nan"),
                )
            all_rows.append(row)

        if not all_rows:
            logger.warning("No results to plot. Skipping figure generation.")
            return json_path

        # ── Assemble matrix ──────────────────────────────────────────
        prompt_names = [r["prompt"] for r in all_rows]
        data_matrix = np.full((len(all_rows), len(metric_keys)), float("nan"))
        for i, row in enumerate(all_rows):
            for j, key in enumerate(metric_keys):
                data_matrix[i, j] = row.get(key, float("nan"))

        # ── Plot heatmap ─────────────────────────────────────────────
        fig, ax = plt.subplots(
            figsize=(max(5, len(metric_keys) * 1.5), max(3, len(prompt_names) * 0.6)),
        )

        # Custom diverging colourmap: red-white-blue for similarity metrics
        cmap = sns.color_palette("RdYlGn", as_cmap=True)

        sns.heatmap(
            data_matrix,
            ax=ax,
            xticklabels=metric_labels,
            yticklabels=prompt_names,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Metric Value"},
        )

        ax.set_title("OOD Prompt Robustness", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

        fig_path = output_dir / "fig_ood_robustness.pdf"
        save_with_vcd(fig, fig_path, close=True)
        logger.info("Saved OOD robustness figure to %s", fig_path)

        return fig_path


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """Command-line interface for OOD evaluation."""
    parser = argparse.ArgumentParser(
        description="CLOP-DiT Out-of-Distribution Prompt Evaluation",
    )
    parser.add_argument(
        "--prompts_yaml",
        type=str,
        default="configs/ood_test_prompts.yaml",
        help="Path to OOD prompts YAML config.",
    )
    parser.add_argument(
        "--reference_h5ad",
        type=str,
        required=True,
        help="Path to reference AnnData (.h5ad) file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ood_evaluation",
        help="Directory for output files.",
    )
    parser.add_argument("--num_cells", type=int, default=50, help="Cells per prompt.")
    parser.add_argument("--cell_type_key", type=str, default="cell_type")
    parser.add_argument("--embedding_key", type=str, default="X_clop_dit")
    parser.add_argument(
        "--dit_checkpoint",
        type=str,
        default="models/checkpoints/dit_best.pth",
    )
    parser.add_argument(
        "--clop_checkpoint",
        type=str,
        default="models/checkpoints/clop_best.pth",
    )
    parser.add_argument(
        "--scgpt_model_dir",
        type=str,
        default="models/scgpt_pancancer",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    setup_logging()
    logger.info("Starting OOD evaluation.")

    # Load prompts
    prompt_config = _load_prompts_yaml(args.prompts_yaml)
    novel_prompts = prompt_config.get("novel_types", [])
    free_form_prompts = prompt_config.get("free_form", [])

    if not novel_prompts and not free_form_prompts:
        logger.error("No prompts found in %s. Exiting.", args.prompts_yaml)
        sys.exit(1)

    # Load reference data
    logger.info("Loading reference data: %s", args.reference_h5ad)
    reference_adata = sc.read_h5ad(args.reference_h5ad)

    # Initialise evaluator
    evaluator = OODEvaluator.from_checkpoints(
        dit_checkpoint=args.dit_checkpoint,
        clop_checkpoint=args.clop_checkpoint,
        scgpt_model_dir=args.scgpt_model_dir,
        device=args.device,
    )

    results: Dict[str, Any] = {}

    # Evaluate novel types
    if novel_prompts:
        logger.info("Evaluating %d novel-type prompts...", len(novel_prompts))
        results["novel_types"] = evaluator.evaluate_novel_types(
            prompts=novel_prompts,
            reference_adata=reference_adata,
            num_cells=args.num_cells,
            cell_type_key=args.cell_type_key,
            embedding_key=args.embedding_key,
        )

    # Evaluate free-form prompts
    if free_form_prompts:
        logger.info("Evaluating %d free-form prompts...", len(free_form_prompts))
        results["free_form"] = evaluator.evaluate_free_form(
            prompts=free_form_prompts,
            reference_adata=reference_adata,
            num_cells=args.num_cells,
            cell_type_key=args.cell_type_key,
            embedding_key=args.embedding_key,
        )

    # Generate report
    evaluator.generate_report(results, args.output_dir)
    logger.info("OOD evaluation complete. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
