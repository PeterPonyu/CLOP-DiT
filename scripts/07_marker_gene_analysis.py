#!/usr/bin/env python3
# 07_marker_gene_analysis.py — Marker gene visualization for CLOP-DiT v0.4
"""
Biologically-grounded visualization of generated single-cell gene expression.
Demonstrates that CLOP-DiT generated cells recapitulate known marker gene patterns.

This script:
1. Generates cells for specific cell types with biological prompts
2. Decodes gene expression via scGPT
3. Visualizes expression of canonical marker genes (violin/box plots)
4. Compares real vs generated marker gene distributions
5. Performs pathway-level analysis

Marker genes are curated from established literature:
  - CD8+ T cells: CD8A, CD8B, GZMB, PRF1, IFNG
  - CD4+ T cells: CD4, IL7R, TCF7, FOXP3
  - NK cells: NKG7, KLRD1, GNLY, NCR1
  - Macrophages: CD68, CD163, CSF1R, MRC1
  - Cancer markers: MKI67, TOP2A, PCNA, CDK1
  - Stem/progenitor: CD34, KIT, FLT3, PROM1

Usage:
    python scripts/07_marker_gene_analysis.py \
        --output_dir figures/marker_genes \
        --reference_h5ad data/processed_h5ad/GSE123902_LungAdreHmCancer_processed.h5ad
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ============================================================================
#  Canonical marker genes by cell type
# ============================================================================

MARKER_GENES = {
    "CD8+ Cytotoxic T cells": {
        "markers": ["CD8A", "CD8B", "GZMB", "PRF1", "IFNG", "NKG7", "CD3E", "CD3D"],
        "description": "Cytotoxic lymphocytes that kill target cells",
    },
    "CD4+ Helper T cells": {
        "markers": ["CD4", "IL7R", "TCF7", "FOXP3", "CD3E", "CD3D", "CD28", "ICOS"],
        "description": "Helper T lymphocytes coordinating immune response",
    },
    "NK cells": {
        "markers": ["NKG7", "KLRD1", "GNLY", "NCR1", "NCAM1", "KLRF1", "FCGR3A", "CD160"],
        "description": "Natural killer cells of innate immunity",
    },
    "Macrophages / Monocytes": {
        "markers": ["CD68", "CD163", "CSF1R", "MRC1", "CD14", "FCGR1A", "MSR1", "MARCO"],
        "description": "Myeloid phagocytes in tissue",
    },
    "Proliferating cells": {
        "markers": ["MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "TYMS", "MCM2", "BIRC5"],
        "description": "Actively dividing cells (cell cycle markers)",
    },
    "Epithelial cells": {
        "markers": ["EPCAM", "KRT18", "KRT8", "CDH1", "KRT19", "MUC1", "CLDN4", "TJP1"],
        "description": "Epithelial tissue cells",
    },
    "Fibroblasts / Stroma": {
        "markers": ["COL1A1", "COL1A2", "DCN", "LUM", "FAP", "PDGFRA", "VIM", "FN1"],
        "description": "Connective tissue stromal cells",
    },
}

# Cell-type specific prompts for generation
CELL_TYPE_PROMPTS = {
    "CD8_T": (
        "CD8+ cytotoxic T lymphocytes from human lung adenocarcinoma tumor microenvironment. "
        "Tumor-infiltrating CD8+ T cells expressing cytotoxic effector molecules and exhaustion markers."
    ),
    "Macrophage": (
        "Tumor-associated macrophages from human lung adenocarcinoma. "
        "Myeloid macrophage populations in the cancer microenvironment with mixed M1/M2 polarization states."
    ),
    "Epithelial_tumor": (
        "Malignant epithelial cells from human lung adenocarcinoma. "
        "Cancer cells of epithelial origin expressing proliferative and lung adenocarcinoma-specific markers."
    ),
    "Fibroblast": (
        "Cancer-associated fibroblasts from human lung adenocarcinoma stroma. "
        "Stromal fibroblast populations supporting the tumor microenvironment."
    ),
}


def load_inference_pipeline(clop_ckpt, dit_ckpt, scgpt_dir, device="cuda"):
    """Load full inference pipeline."""
    from src.architecture.clop import CLOPAligner
    from src.architecture.dit import DiT1D
    from transformers import AutoTokenizer, AutoModel

    # Load CLOP
    ckpt = torch.load(clop_ckpt, map_location=device, weights_only=False)
    clop_config = ckpt.get("config", {})
    text_dim = clop_config.get("text_dim", 1024)
    cell_dim = clop_config.get("cell_dim", 512)
    proj_dim = clop_config.get("proj_dim", 256)

    clop = CLOPAligner(text_dim=text_dim, cell_dim=cell_dim, proj_dim=proj_dim).to(device)
    clop.load_state_dict(ckpt["model_state_dict"])
    clop.eval()

    # Load DiT
    dit_ckpt_data = torch.load(dit_ckpt, map_location=device, weights_only=False)
    dit_config = dit_ckpt_data.get("config", {})
    dit = DiT1D(
        latent_dim=dit_config.get("latent_dim", cell_dim),
        hidden_dim=dit_config.get("hidden_dim", 384),
        cond_dim=proj_dim,
        num_tokens=dit_config.get("num_tokens", 16),
    ).to(device)
    state = dit_ckpt_data.get("ema_state_dict", dit_ckpt_data["model_state_dict"])
    dit.load_state_dict(state)
    dit.eval()

    # Text encoder
    model_id = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract" if text_dim == 1024 \
        else "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text_model = AutoModel.from_pretrained(model_id).to(device).eval()

    return clop, dit, tokenizer, text_model, cell_dim


def encode_text(prompt, tokenizer, text_model, clop, device="cuda"):
    """Encode text prompt → CLOP projected condition."""
    with torch.no_grad():
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=512, padding=True).to(device)
        text_emb = text_model(**tokens).last_hidden_state[:, 0, :]  # [CLS]
        cond = clop.project_text(text_emb)  # (1, proj_dim)
    return cond


def generate_cells(dit, cond, num_cells, cell_dim, num_steps=4, cfg_scale=3.0):
    """Generate cell embeddings via DiT flow matching."""
    with torch.no_grad():
        cond_expanded = cond.expand(num_cells, -1)
        z = dit.sample(cond_expanded, num_steps=num_steps, cfg_scale=cfg_scale)
    return z


def main():
    parser = argparse.ArgumentParser(description="CLOP-DiT v0.4 Marker Gene Analysis")
    parser.add_argument("--output_dir", type=str, default="figures/marker_genes")
    parser.add_argument("--reference_h5ad", type=str,
                        default="data/processed_h5ad/GSE123902_LungAdreHmCancer_processed.h5ad")
    parser.add_argument("--clop_checkpoint", type=str, default="models/checkpoints/clop_best.pth")
    parser.add_argument("--dit_checkpoint", type=str, default="models/checkpoints/dit_best.pth")
    parser.add_argument("--scgpt_model_dir", type=str, default="models/scgpt_pancancer")
    parser.add_argument("--num_cells", type=int, default=200,
                        help="Cells to generate per cell type")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────
    logger.info("Loading inference pipeline...")
    clop, dit, tokenizer, text_model, cell_dim = load_inference_pipeline(
        args.clop_checkpoint, args.dit_checkpoint, args.scgpt_model_dir, args.device
    )

    # ── Load scGPT decoder ──────────────────────────────────────────
    import scanpy as sc
    from src.architecture.decoder import ScGPTDecoder

    ref_adata = sc.read_h5ad(args.reference_h5ad)
    logger.info(f"Reference: {ref_adata.shape}")

    scgpt_decoder = ScGPTDecoder(
        model_dir=args.scgpt_model_dir,
        device=torch.device(args.device),
    )
    # Encode reference to set up gene vocabulary
    ref_embs = scgpt_decoder.encode(ref_adata)
    logger.info(f"Reference embeddings: {ref_embs.shape}")
    # The gene vocabulary is now set inside scgpt_decoder from this encode() call

    # ── Generate cells for each cell type ────────────────────────────
    all_generated = {}
    for ct_name, prompt_text in CELL_TYPE_PROMPTS.items():
        logger.info(f"\nGenerating {args.num_cells} cells for: {ct_name}")

        cond = encode_text(prompt_text, tokenizer, text_model, clop, args.device)
        cell_embs = generate_cells(dit, cond, args.num_cells, cell_dim)

        # Decode to gene expression
        decoded = scgpt_decoder.decode(cell_embs.cpu().numpy())
        all_generated[ct_name] = {
            "expression": decoded["expression"],
            "gene_names": decoded["gene_names"],
            "prompt": prompt_text,
        }
        logger.info(f"  Decoded: {decoded['expression'].shape}")

    # Also decode real reference embeddings
    logger.info("\nDecoding real reference cells...")
    # Take a subsample
    if len(ref_embs) > 500:
        idx = np.random.choice(len(ref_embs), 500, replace=False)
        real_embs_sub = ref_embs[idx]
    else:
        real_embs_sub = ref_embs
    real_decoded = scgpt_decoder.decode(real_embs_sub)

    # ── Visualization ────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    gene_names = all_generated[list(all_generated.keys())[0]]["gene_names"]
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # 1. Marker gene violin plots per cell type
    logger.info("\nPlotting marker gene violin plots...")
    for ct_name, markers_info in MARKER_GENES.items():
        markers = markers_info["markers"]
        available_markers = [m for m in markers if m in gene_to_idx]

        if len(available_markers) < 2:
            logger.warning(f"  {ct_name}: only {len(available_markers)} markers available, skipping")
            continue

        fig, axes = plt.subplots(1, len(available_markers), figsize=(3*len(available_markers), 5))
        if len(available_markers) == 1:
            axes = [axes]

        for ax_i, marker in enumerate(available_markers):
            midx = gene_to_idx[marker]
            data = []
            labels = []

            # Real reference
            data.append(real_decoded["expression"][:, midx])
            labels.append("Real\n(Reference)")

            # Each generated cell type
            for gen_ct in all_generated:
                data.append(all_generated[gen_ct]["expression"][:, midx])
                labels.append(f"Gen\n({gen_ct})")

            parts = axes[ax_i].violinplot(data, showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_alpha(0.7)
            axes[ax_i].set_xticks(range(1, len(labels)+1))
            axes[ax_i].set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
            axes[ax_i].set_title(marker, fontweight="bold", fontsize=10)
            axes[ax_i].set_ylabel("Expression level")

        fig.suptitle(f"Marker Gene Expression: {ct_name}\n({markers_info['description']})",
                     fontweight="bold", fontsize=12)
        plt.tight_layout()
        safe_name = ct_name.replace(" ", "_").replace("/", "_").replace("+", "plus")
        fig.savefig(output_dir / f"markers_{safe_name}.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: markers_{safe_name}.png ({len(available_markers)} markers)")

    # 2. Cross-cell-type marker heatmap
    logger.info("\nPlotting cross-cell-type marker heatmap...")
    all_markers = []
    for ct_name, info in MARKER_GENES.items():
        all_markers.extend([m for m in info["markers"] if m in gene_to_idx])
    all_markers = list(dict.fromkeys(all_markers))  # unique, ordered

    if len(all_markers) >= 5:
        gen_types = list(all_generated.keys())
        heatmap_data = np.zeros((len(gen_types) + 1, len(all_markers)))

        # Real reference means
        for j, marker in enumerate(all_markers):
            midx = gene_to_idx[marker]
            heatmap_data[0, j] = real_decoded["expression"][:, midx].mean()

        # Generated means
        for i, ct in enumerate(gen_types):
            for j, marker in enumerate(all_markers):
                midx = gene_to_idx[marker]
                heatmap_data[i+1, j] = all_generated[ct]["expression"][:, midx].mean()

        # Normalize per marker (z-score across cell types)
        heatmap_normalized = np.zeros_like(heatmap_data)
        for j in range(len(all_markers)):
            col = heatmap_data[:, j]
            if col.std() > 0:
                heatmap_normalized[:, j] = (col - col.mean()) / col.std()

        fig, ax = plt.subplots(figsize=(max(12, len(all_markers)*0.5), 6))
        im = ax.imshow(heatmap_normalized, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
        ax.set_xticks(range(len(all_markers)))
        ax.set_xticklabels(all_markers, rotation=90, fontsize=7)
        ax.set_yticks(range(len(gen_types) + 1))
        ax.set_yticklabels(["Real (Ref)"] + [f"Gen ({ct})" for ct in gen_types], fontsize=9)
        plt.colorbar(im, label="Z-score (per marker)")
        ax.set_title("Cross-Cell-Type Marker Gene Expression (Z-scored)",
                     fontweight="bold", fontsize=12)

        # Add cell type group separators
        cumulative = 0
        for ct_name, info in MARKER_GENES.items():
            avail = [m for m in info["markers"] if m in gene_to_idx]
            if avail:
                mid = cumulative + len(avail) / 2 - 0.5
                ax.text(mid, -0.8, ct_name, ha="center", fontsize=6, color="gray",
                        rotation=0, style="italic")
                cumulative += len(avail)

        plt.tight_layout()
        fig.savefig(output_dir / "marker_heatmap_cross_celltype.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: marker_heatmap_cross_celltype.png ({len(all_markers)} markers)")

    # 3. Cell-type discrimination: are generated cell types distinguishable?
    logger.info("\nPlotting cell-type discrimination UMAP...")
    try:
        from sklearn.decomposition import PCA
        import umap

        all_expr = []
        all_labels = []

        # Real cells
        all_expr.append(real_decoded["expression"][:200])
        all_labels.extend(["Real (Reference)"] * min(200, len(real_decoded["expression"])))

        # Generated cells
        for ct in gen_types:
            expr = all_generated[ct]["expression"][:200]
            all_expr.append(expr)
            all_labels.extend([f"Gen: {ct}"] * len(expr))

        combined = np.vstack(all_expr)

        # PCA → UMAP
        pca = PCA(n_components=30)
        pca_emb = pca.fit_transform(combined)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
        umap_emb = reducer.fit_transform(pca_emb)

        fig, ax = plt.subplots(figsize=(10, 8))
        unique_labels = list(dict.fromkeys(all_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = np.array(all_labels) == label
            marker = "o" if "Real" in label else "x"
            alpha = 0.6 if "Real" in label else 0.8
            size = 20 if "Real" in label else 30
            ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1],
                       c=[colors[i]], label=label, marker=marker,
                       alpha=alpha, s=size, edgecolors="none")

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title("Generated Cell Types in Gene Expression Space\n"
                     "(CLOP-DiT text-conditioned generation → scGPT decode)",
                     fontweight="bold", fontsize=11)
        plt.tight_layout()
        fig.savefig(output_dir / "celltype_discrimination_umap.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: celltype_discrimination_umap.png")

    except ImportError:
        logger.warning("  UMAP not available, skipping discrimination plot")

    # 4. Top discriminative genes between generated cell types
    logger.info("\nIdentifying top discriminative genes...")
    if len(gen_types) >= 2:
        fig, axes = plt.subplots(1, len(gen_types), figsize=(5*len(gen_types), 6))
        if len(gen_types) == 1:
            axes = [axes]

        for i, ct in enumerate(gen_types):
            expr_ct = all_generated[ct]["expression"]
            # Compare against all others pooled
            other_expr = np.vstack([all_generated[o]["expression"] for o in gen_types if o != ct])

            # Log fold change
            mean_ct = expr_ct.mean(axis=0) + 1e-8
            mean_other = other_expr.mean(axis=0) + 1e-8
            lfc = np.log2(mean_ct / mean_other)

            # Top upregulated genes
            top_up = np.argsort(lfc)[-15:][::-1]
            top_down = np.argsort(lfc)[:10]

            top_idx = np.concatenate([top_up, top_down])
            top_names = [gene_names[j] for j in top_idx]
            top_lfc = lfc[top_idx]

            colors_bar = ["#d62728" if v > 0 else "#1f77b4" for v in top_lfc]
            axes[i].barh(range(len(top_names)), top_lfc, color=colors_bar, alpha=0.8)
            axes[i].set_yticks(range(len(top_names)))
            axes[i].set_yticklabels(top_names, fontsize=7)
            axes[i].set_xlabel("log₂ fold change")
            axes[i].set_title(f"{ct}\n(vs others)", fontweight="bold", fontsize=10)
            axes[i].axvline(0, color="gray", linewidth=0.5)

        fig.suptitle("Top Discriminative Genes per Generated Cell Type\n"
                    "(log₂ fold change vs pooled other types)",
                    fontweight="bold", fontsize=12)
        plt.tight_layout()
        fig.savefig(output_dir / "discriminative_genes.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: discriminative_genes.png")

    # 5. Summary statistics
    summary = {
        "num_cell_types_generated": len(gen_types),
        "cells_per_type": args.num_cells,
        "num_genes_decoded": len(gene_names),
        "reference_dataset": args.reference_h5ad,
        "marker_gene_coverage": {},
        "generated_prompts": CELL_TYPE_PROMPTS,
    }

    for ct_name, info in MARKER_GENES.items():
        available = [m for m in info["markers"] if m in gene_to_idx]
        summary["marker_gene_coverage"][ct_name] = {
            "total": len(info["markers"]),
            "available": len(available),
            "markers": available,
        }

    with open(output_dir / "marker_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Marker Gene Analysis Complete")
    logger.info(f"{'='*60}")
    logger.info(f"  Cell types generated: {len(gen_types)}")
    logger.info(f"  Cells per type: {args.num_cells}")
    logger.info(f"  Genes decoded: {len(gene_names)}")
    logger.info(f"  Figures saved to: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
