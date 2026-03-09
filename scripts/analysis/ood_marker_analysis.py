#!/usr/bin/env python3
"""OOD marker gene analysis — check whether generated cells express expected markers.

For each OOD prompt that mentions specific marker genes, generate cells and
check whether those markers are up-regulated in the decoded expression profile.
This is the key test: if the model is truly conditioning on text, marker genes
should appear at higher expression than background.

Produces results/ood_evaluation/ood_marker_analysis.json
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logging_config import setup_logging

# Load CLOPDiTInference from numbered script
_INFERENCE_SCRIPT = _PROJECT_ROOT / "scripts" / "inference" / "05_inference.py"
_spec = importlib.util.spec_from_file_location("inference", str(_INFERENCE_SCRIPT))
_inference_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_inference_mod)
CLOPDiTInference = _inference_mod.CLOPDiTInference

logger = logging.getLogger(__name__)

# ── Prompt definitions with expected markers ─────────────────────────
# Each entry: (name, prompt, expected_marker_genes)
NOVEL_TYPE_PROMPTS = [
    (
        "Goblet cells",
        "Mucus-secreting goblet cells from human intestinal epithelium. These specialized "
        "secretory cells produce and release gel-forming mucins that form the protective "
        "mucus layer of the gastrointestinal tract, expressing MUC2, TFF3, and FCGBP.",
        ["MUC2", "TFF3", "FCGBP"],
    ),
    (
        "Purkinje cells",
        "Purkinje neurons from human cerebellar cortex. Large GABAergic inhibitory neurons "
        "in the Purkinje layer of the cerebellum, characterized by extensive dendritic "
        "arbors and expression of CALB1, PCP2, and ITPR1.",
        ["CALB1", "PCP2", "ITPR1"],
    ),
    (
        "Podocytes",
        "Glomerular podocytes from human kidney. Highly specialized epithelial cells "
        "wrapping the capillaries of the renal glomerulus, forming the slit diaphragm "
        "filtration barrier and expressing NPHS1, NPHS2, WT1, and PODXL.",
        ["NPHS1", "NPHS2", "WT1", "PODXL"],
    ),
    (
        "Leydig cells",
        "Interstitial Leydig cells from human testis. Steroidogenic cells located in the "
        "interstitial compartment between seminiferous tubules, responsible for testosterone "
        "biosynthesis and expressing STAR, CYP11A1, and HSD3B2.",
        ["STAR", "CYP11A1", "HSD3B2"],
    ),
    (
        "Club cells",
        "Non-ciliated bronchiolar club cells (formerly Clara cells) from human distal "
        "airways. Secretory epithelial cells expressing SCGB1A1 and CYP2F1, contributing "
        "to airway surface liquid and xenobiotic metabolism in the terminal bronchioles.",
        ["SCGB1A1", "CYP2F1"],
    ),
    (
        "Merkel cells",
        "Merkel cells from human skin epidermis. Neuroendocrine mechanoreceptor cells in "
        "the basal layer of the epidermis, associated with light-touch sensation and "
        "expressing KRT20, ATOH1, and ISL1.",
        ["KRT20", "ATOH1", "ISL1"],
    ),
]

FREE_FORM_PROMPTS = [
    ("informal_cd8", "immune cells that kill infected cells in the lung"),
    ("informal_macrophage", "the big cells that eat bacteria and dead cells in inflamed tissue"),
    ("informal_stem", "undifferentiated cells that can become many different cell types in the bone marrow"),
    ("informal_neuron", "brain cells that fire electrical signals and talk to each other with chemicals"),
    ("verbose_treg", "I am looking for a type of T cell that suppresses immune responses rather than promoting them. They are often found in tumors and express FOXP3 and CD25. Sometimes called regulatory T cells."),
    ("question_style", "what would a fibroblast in a wound healing context look like at the single-cell level"),
    ("shorthand_nk", "NK cells, cytotoxic, blood"),
    ("clinical_note_style", "Patient sample shows atypical large B-cells consistent with diffuse large B-cell lymphoma, germinal center subtype"),
]

# Structured equivalents for free-form prompts (template format matching training)
STRUCTURED_EQUIVALENTS = {
    "informal_cd8": "CD8+ cytotoxic T lymphocytes from human lung tissue. Adaptive immune cells expressing CD8A, CD8B, GZMB, PRF1, and IFNG.",
    "informal_macrophage": "Macrophages from human inflamed tissue. Innate immune phagocytic cells expressing CD68, CD163, CSF1R, and MRC1.",
    "informal_stem": "Hematopoietic stem cells from human bone marrow. Multipotent progenitor cells expressing CD34, KIT, THY1, and CXCR4.",
    "informal_neuron": "Excitatory neurons from human cerebral cortex. Glutamatergic neurons expressing SLC17A7, RBFOX3, SNAP25, and SYN1.",
    "verbose_treg": "CD4+ regulatory T cells from human tumor microenvironment. Immunosuppressive T lymphocytes expressing FOXP3, IL2RA, CTLA4, and IKZF2.",
    "question_style": "Activated fibroblasts from human wound healing tissue. Mesenchymal cells expressing COL1A1, ACTA2, FAP, and POSTN.",
    "shorthand_nk": "Natural killer cells from human peripheral blood. Cytotoxic innate lymphocytes expressing NCAM1, KLRD1, NKG7, and GNLY.",
    "clinical_note_style": "Diffuse large B-cell lymphoma cells, germinal center subtype, from human lymph node. Malignant B lymphocytes expressing BCL6, CD10, LMO2, and CD20.",
}


def _find_marker_genes(gene_names: list[str], markers: list[str]) -> dict[str, int | None]:
    """Map marker gene names to indices in gene_names (case-insensitive)."""
    name_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    return {m: name_to_idx.get(m.upper()) for m in markers}


def main() -> None:
    setup_logging()
    import scanpy as sc

    output_dir = Path("results/ood_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference for gene-name setup
    ref_path = "data/processed_h5ad/lung_processed.h5ad"
    logger.info("Loading reference: %s", ref_path)
    reference = sc.read_h5ad(ref_path)

    # Init pipeline
    logger.info("Loading inference pipeline...")
    pipeline = CLOPDiTInference(
        dit_checkpoint="models/checkpoints/DiT/best/dit_best.pth",
        clop_checkpoint="models/checkpoints/CLOP/best/clop_best.pth",
        scgpt_model_dir="models/scgpt_pancancer",
        device="cuda",
    )

    num_cells = 100
    results = {"novel_types": {}, "free_form_vs_structured": {}, "summary": {}}

    # ── 1. Novel-type marker analysis ──────────────────────────────
    logger.info("=== Novel-type marker gene analysis ===")
    marker_hit_rates = []

    for name, prompt, expected_markers in NOVEL_TYPE_PROMPTS:
        logger.info("Generating: %s", name)
        gen = pipeline.generate_adata(
            prompt=prompt,
            num_cells=num_cells,
            decode_expression=True,
            reference_adata=reference,
        )

        gene_names = gen.var_names.tolist()
        marker_map = _find_marker_genes(gene_names, expected_markers)

        expr = gen.X if not hasattr(gen.X, "A") else gen.X.A
        all_mean = np.mean(expr, axis=0)  # background mean per gene
        all_std = np.std(np.mean(expr, axis=0))

        marker_results = {}
        markers_found = 0
        markers_upregulated = 0

        for marker, idx in marker_map.items():
            if idx is None:
                marker_results[marker] = {"status": "not_in_vocabulary"}
                continue
            markers_found += 1
            marker_mean = float(np.mean(expr[:, idx]))
            marker_pct_nonzero = float(np.mean(expr[:, idx] > 0))
            bg_mean = float(all_mean[idx])
            # Percentile rank of this gene among all genes
            pct_rank = float(np.mean(all_mean <= marker_mean) * 100)

            is_upregulated = pct_rank > 75  # top 25%
            if is_upregulated:
                markers_upregulated += 1

            marker_results[marker] = {
                "mean_expression": round(marker_mean, 4),
                "pct_nonzero": round(marker_pct_nonzero, 4),
                "percentile_rank": round(pct_rank, 1),
                "upregulated_top25pct": is_upregulated,
            }

        hit_rate = markers_upregulated / max(markers_found, 1)
        marker_hit_rates.append(hit_rate)

        # Embedding diversity (intra-sample)
        if hasattr(gen, "obsm") and "X_generated" in gen.obsm:
            embs = gen.obsm["X_generated"]
        else:
            embs = expr  # fallback to expression space
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        cosine_mat = (embs / norms) @ (embs / norms).T
        np.fill_diagonal(cosine_mat, np.nan)
        intra_cos = float(np.nanmean(cosine_mat))

        results["novel_types"][name] = {
            "prompt": prompt,
            "expected_markers": expected_markers,
            "markers_in_vocabulary": markers_found,
            "markers_upregulated": markers_upregulated,
            "marker_hit_rate": round(hit_rate, 3),
            "marker_details": marker_results,
            "intra_sample_cosine_sim": round(intra_cos, 4),
            "num_genes_decoded": len(gene_names),
        }
        logger.info(
            "  %s: %d/%d markers found, %d upregulated (hit_rate=%.2f)",
            name, markers_found, len(expected_markers), markers_upregulated, hit_rate,
        )

    # ── 2. Free-form vs. structured comparison ────────────────────
    logger.info("=== Free-form vs. structured comparison ===")
    cosine_deltas = []

    for name, free_prompt in FREE_FORM_PROMPTS:
        struct_prompt = STRUCTURED_EQUIVALENTS.get(name)
        if not struct_prompt:
            continue

        logger.info("Generating free-form: %s", name)
        gen_free = pipeline.generate_adata(
            prompt=free_prompt, num_cells=num_cells,
            decode_expression=True, reference_adata=reference,
        )
        logger.info("Generating structured: %s", name)
        gen_struct = pipeline.generate_adata(
            prompt=struct_prompt, num_cells=num_cells,
            decode_expression=True, reference_adata=reference,
        )

        # Compare mean expression profiles
        free_expr = gen_free.X if not hasattr(gen_free.X, "A") else gen_free.X.A
        struct_expr = gen_struct.X if not hasattr(gen_struct.X, "A") else gen_struct.X.A

        free_mean = np.mean(free_expr, axis=0)
        struct_mean = np.mean(struct_expr, axis=0)

        from scipy import stats
        r, p = stats.pearsonr(free_mean, struct_mean)

        # Cosine similarity between mean expression profiles
        cos_sim = float(
            np.dot(free_mean, struct_mean)
            / (np.linalg.norm(free_mean) * np.linalg.norm(struct_mean) + 1e-12)
        )
        cosine_deltas.append(cos_sim)

        results["free_form_vs_structured"][name] = {
            "free_form_prompt": free_prompt,
            "structured_prompt": struct_prompt,
            "expression_pearson_r": round(float(r), 4),
            "expression_pearson_p": float(p),
            "expression_cosine_sim": round(cos_sim, 4),
        }
        logger.info("  %s: pearson_r=%.4f, cosine_sim=%.4f", name, r, cos_sim)

    # ── Summary ───────────────────────────────────────────────────
    results["summary"] = {
        "novel_types_mean_marker_hit_rate": round(float(np.mean(marker_hit_rates)), 3) if marker_hit_rates else None,
        "novel_types_marker_hit_rates": {
            NOVEL_TYPE_PROMPTS[i][0]: round(marker_hit_rates[i], 3)
            for i in range(len(marker_hit_rates))
        },
        "free_form_mean_expression_cosine": round(float(np.mean(cosine_deltas)), 4) if cosine_deltas else None,
        "free_form_expression_cosines": {
            FREE_FORM_PROMPTS[i][0]: round(cosine_deltas[i], 4)
            for i in range(len(cosine_deltas))
        },
        "interpretation": (
            "Marker hit rate measures whether genes named in the prompt appear in the top 25% "
            "of expressed genes in the generated cells. A high rate suggests text-conditional "
            "generation; a low rate suggests the model ignores specific gene references and "
            "falls back to generic expression profiles. "
            "Expression cosine similarity between free-form and structured equivalents measures "
            "whether informal prompts produce similar profiles to template-style prompts."
        ),
    }

    out_path = output_dir / "ood_marker_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved OOD marker analysis to %s", out_path)


if __name__ == "__main__":
    main()
