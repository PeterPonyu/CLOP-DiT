#!/usr/bin/env python3
# 07_marker_gene_analysis.py — Improved marker gene visualization for CLOP-DiT v0.4
"""
Biologically-grounded visualization of generated vs real single-cell gene expression.

Key improvements over v0.3:
  1. Grid layout (max 4 per row) instead of single-row cramming
  2. Proper reference vs generated paired comparison
  3. Real cells decoded through same scGPT pipeline for fair comparison
  4. Statistical tests (KS test) for distribution similarity
  5. Improved aesthetics: publication-quality figures

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

    ckpt = torch.load(clop_ckpt, map_location=device, weights_only=False)
    clop_config = ckpt.get("config", {})
    text_dim = clop_config.get("text_dim", 1024)
    cell_dim = clop_config.get("cell_dim", 512)
    proj_dim = clop_config.get("proj_dim", 256)

    clop = CLOPAligner(text_dim=text_dim, cell_dim=cell_dim, proj_dim=proj_dim).to(device)
    clop.load_state_dict(ckpt["model_state_dict"])
    clop.eval()

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
        text_emb = text_model(**tokens).last_hidden_state[:, 0, :]
        cond = clop.project_text(text_emb)
    return cond


def generate_cells(dit, cond, num_cells, cell_dim, num_steps=20, cfg_scale=3.0):
    """Generate cell embeddings via DiT flow matching."""
    with torch.no_grad():
        cond_expanded = cond.expand(num_cells, -1)
        z = dit.sample(cond_expanded, num_steps=num_steps, cfg_scale=cfg_scale)
    return z


def plot_marker_violin_grid(
    gene_names, gene_to_idx, real_expr, gen_expr_dict,
    ct_name, markers, description, output_dir, max_cols=4
):
    """
    Plot marker gene violin plots in a grid layout (max_cols per row).
    Shows Real vs Generated side-by-side for each marker.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    available = [m for m in markers if m in gene_to_idx]
    if len(available) < 2:
        logger.warning(f"  {ct_name}: only {len(available)} markers available, skipping")
        return

    n_markers = len(available)
    n_cols = min(max_cols, n_markers)
    n_rows = (n_markers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    gen_types = list(gen_expr_dict.keys())
    colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]

    for idx, marker in enumerate(available):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        midx = gene_to_idx[marker]

        # Collect data
        positions = []
        data_list = []
        tick_labels = []
        pos = 1

        # Real reference
        real_vals = real_expr[:, midx]
        data_list.append(real_vals)
        positions.append(pos)
        tick_labels.append("Real\n(Ref)")
        pos += 1

        # Generated cell types
        for gi, gen_ct in enumerate(gen_types):
            gen_vals = gen_expr_dict[gen_ct]["expression"][:, midx]
            data_list.append(gen_vals)
            positions.append(pos)
            tick_labels.append(f"Gen\n({gen_ct})")
            pos += 1

        # Box plot with swarm overlay
        bp = ax.boxplot(data_list, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))

        # Color boxes
        box_colors = ["#B3E5FC"] + [colors[i % len(colors)] for i in range(len(gen_types))]
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # KS test: real vs best matching generated
        ks_stats = []
        for gen_ct in gen_types:
            gen_vals = gen_expr_dict[gen_ct]["expression"][:, midx]
            ks_stat, ks_p = stats.ks_2samp(real_vals, gen_vals)
            ks_stats.append((gen_ct, ks_stat, ks_p))

        # Show best KS p-value
        best_ks = min(ks_stats, key=lambda x: x[1])
        ax.text(0.95, 0.95, f"KS={best_ks[1]:.3f}\np={best_ks[2]:.2e}",
                transform=ax.transAxes, fontsize=6, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=0)
        ax.set_title(marker, fontweight="bold", fontsize=11)
        ax.set_ylabel("Expression", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Hide empty subplots
    for idx in range(n_markers, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Marker Gene Expression: {ct_name}\n{description}",
        fontweight="bold", fontsize=13, y=1.02
    )
    plt.tight_layout()
    safe_name = ct_name.replace(" ", "_").replace("/", "_").replace("+", "plus")
    fig.savefig(output_dir / f"markers_{safe_name}.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: markers_{safe_name}.png ({n_markers} markers, {n_rows}x{n_cols} grid)")


def plot_cross_celltype_heatmap(
    gene_names, gene_to_idx, real_expr, gen_expr_dict, output_dir
):
    """Cross-cell-type heatmap with improved layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_markers = []
    marker_groups = []
    for ct_name, info in MARKER_GENES.items():
        avail = [m for m in info["markers"] if m in gene_to_idx]
        all_markers.extend(avail)
        marker_groups.extend([ct_name] * len(avail))
    all_markers_unique = list(dict.fromkeys(all_markers))

    if len(all_markers_unique) < 5:
        return

    gen_types = list(gen_expr_dict.keys())
    row_labels = ["Real (Reference)"] + [f"Gen: {ct}" for ct in gen_types]
    n_rows = len(row_labels)

    heatmap_data = np.zeros((n_rows, len(all_markers_unique)))

    # Real means
    for j, marker in enumerate(all_markers_unique):
        midx = gene_to_idx[marker]
        heatmap_data[0, j] = real_expr[:, midx].mean()

    # Generated means
    for i, ct in enumerate(gen_types):
        for j, marker in enumerate(all_markers_unique):
            midx = gene_to_idx[marker]
            heatmap_data[i + 1, j] = gen_expr_dict[ct]["expression"][:, midx].mean()

    # Z-score per marker
    heatmap_z = np.zeros_like(heatmap_data)
    for j in range(len(all_markers_unique)):
        col = heatmap_data[:, j]
        if col.std() > 0:
            heatmap_z[:, j] = (col - col.mean()) / col.std()

    # Plot
    fig_width = max(14, len(all_markers_unique) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, 4 + n_rows * 0.5))

    im = ax.imshow(heatmap_z, cmap="RdBu_r", aspect="auto", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(all_markers_unique)))
    ax.set_xticklabels(all_markers_unique, rotation=90, fontsize=7, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Cell values
    for i in range(n_rows):
        for j in range(len(all_markers_unique)):
            val = heatmap_z[i, j]
            color = "white" if abs(val) > 1.5 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=5, color=color)

    plt.colorbar(im, label="Z-score (normalized per marker)", shrink=0.8)
    ax.set_title("Cross-Cell-Type Marker Gene Expression (Z-scored)\n"
                 "Real reference vs CLOP-DiT generated cells",
                 fontweight="bold", fontsize=12)

    # Group separators
    cumulative = 0
    prev_ct = None
    for m, ct in zip(all_markers, marker_groups):
        if m not in all_markers_unique:
            continue
        if ct != prev_ct and prev_ct is not None:
            pos = all_markers_unique.index(m)
            ax.axvline(pos - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        prev_ct = ct
        cumulative += 1

    plt.tight_layout()
    fig.savefig(output_dir / "marker_heatmap_cross_celltype.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: marker_heatmap_cross_celltype.png ({len(all_markers_unique)} markers)")


def plot_celltype_umap(real_expr, gen_expr_dict, output_dir):
    """UMAP of decoded gene expression colored by source."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from sklearn.decomposition import PCA
        import umap
    except ImportError:
        logger.warning("  UMAP not available, skipping discrimination plot")
        return

    gen_types = list(gen_expr_dict.keys())
    all_expr = []
    all_labels = []

    # Real cells
    n_real = min(200, len(real_expr))
    all_expr.append(real_expr[:n_real])
    all_labels.extend(["Real (Reference)"] * n_real)

    # Generated cells
    for ct in gen_types:
        expr = gen_expr_dict[ct]["expression"][:200]
        all_expr.append(expr)
        all_labels.extend([f"Gen: {ct}"] * len(expr))

    combined = np.vstack(all_expr)

    pca = PCA(n_components=min(30, combined.shape[1] - 1))
    pca_emb = pca.fit_transform(combined)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
    umap_emb = reducer.fit_transform(pca_emb)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = list(dict.fromkeys(all_labels))
    cmap_colors = ["#1976D2", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, label in enumerate(unique_labels):
        mask = np.array(all_labels) == label
        marker = "o" if "Real" in label else "^"
        alpha = 0.5 if "Real" in label else 0.7
        size = 25 if "Real" in label else 35
        color = "#607D8B" if "Real" in label else cmap_colors[i % len(cmap_colors)]
        ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1],
                   c=color, label=label, marker=marker,
                   alpha=alpha, s=size, edgecolors="white", linewidths=0.3)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9,
              frameon=True, fancybox=True, shadow=True)
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title("Cell Type Discrimination in Decoded Gene Expression Space\n"
                 "CLOP-DiT text-conditioned generation → scGPT decode",
                 fontweight="bold", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "celltype_discrimination_umap.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: celltype_discrimination_umap.png")


def plot_discriminative_genes_grid(gene_names, gen_expr_dict, output_dir, max_cols=2):
    """Top discriminative genes per cell type in grid layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen_types = list(gen_expr_dict.keys())
    if len(gen_types) < 2:
        return

    n_types = len(gen_types)
    n_cols = min(max_cols, n_types)
    n_rows = (n_types + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, ct in enumerate(gen_types):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        expr_ct = gen_expr_dict[ct]["expression"]
        other_expr = np.vstack([gen_expr_dict[o]["expression"] for o in gen_types if o != ct])

        mean_ct = expr_ct.mean(axis=0) + 1e-8
        mean_other = other_expr.mean(axis=0) + 1e-8
        lfc = np.log2(mean_ct / mean_other)

        top_up = np.argsort(lfc)[-10:][::-1]
        top_down = np.argsort(lfc)[:5]
        top_idx = np.concatenate([top_up, top_down])
        top_names = [gene_names[j] for j in top_idx]
        top_lfc = lfc[top_idx]

        colors_bar = ["#d32f2f" if v > 0 else "#1565C0" for v in top_lfc]
        y_pos = range(len(top_names) - 1, -1, -1)
        ax.barh(list(y_pos), top_lfc, color=colors_bar, alpha=0.85, height=0.7)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(top_names, fontsize=8, fontweight="bold")
        ax.set_xlabel("log₂ fold change", fontsize=10)
        ax.set_title(f"{ct}\n(vs all other types)", fontweight="bold", fontsize=11)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide empty subplots
    for idx in range(n_types, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle("Top Discriminative Genes per Generated Cell Type\n"
                 "(log₂ fold change vs pooled other types)",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "discriminative_genes.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: discriminative_genes.png ({n_rows}x{n_cols} grid)")


def plot_distribution_comparison(
    gene_names, gene_to_idx, real_expr, gen_expr_dict, output_dir
):
    """
    Side-by-side distribution comparison: Real vs Generated for key markers.
    Shows density plots overlaid for direct visual comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    # Select key markers that are available
    key_markers = []
    for ct_info in MARKER_GENES.values():
        for m in ct_info["markers"][:3]:
            if m in gene_to_idx and m not in key_markers:
                key_markers.append(m)
    key_markers = key_markers[:16]  # max 16

    if len(key_markers) < 4:
        return

    n_cols = 4
    n_rows = (len(key_markers) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    gen_types = list(gen_expr_dict.keys())

    for idx, marker in enumerate(key_markers):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        midx = gene_to_idx[marker]

        # Real distribution
        real_vals = real_expr[:, midx]
        ax.hist(real_vals, bins=30, alpha=0.4, density=True, color="#1976D2", label="Real")

        # Generated distributions (overlay best matching type)
        for gi, gen_ct in enumerate(gen_types):
            gen_vals = gen_expr_dict[gen_ct]["expression"][:, midx]
            ax.hist(gen_vals, bins=30, alpha=0.3, density=True,
                    color=["#E91E63", "#4CAF50", "#FF9800", "#9C27B0"][gi % 4],
                    label=f"Gen:{gen_ct}")

        ax.set_title(marker, fontweight="bold", fontsize=10)
        ax.set_xlabel("Expression", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=5, loc="upper right")

    # Hide empty
    for idx in range(len(key_markers), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle("Expression Distribution Comparison: Real vs Generated\n"
                 "(density histograms overlaid for key markers)",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "distribution_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: distribution_comparison.png ({len(key_markers)} markers)")


def plot_correlation_scatter(real_expr, gen_expr_dict, gene_names, output_dir, max_cols=2):
    """Scatter plot of mean gene expression: Real vs Generated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    gen_types = list(gen_expr_dict.keys())
    n_types = len(gen_types)
    n_cols = min(max_cols, n_types)
    n_rows = (n_types + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    real_mean = real_expr.mean(axis=0)

    for i, ct in enumerate(gen_types):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        gen_mean = gen_expr_dict[ct]["expression"].mean(axis=0)

        # Pearson correlation
        r, p = stats.pearsonr(real_mean, gen_mean)

        ax.scatter(real_mean, gen_mean, s=3, alpha=0.3, c="#1976D2", edgecolors="none")

        # Diagonal line
        max_val = max(real_mean.max(), gen_mean.max())
        ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, linewidth=1)

        # Annotate top divergent genes
        diff = np.abs(real_mean - gen_mean)
        top_diff = np.argsort(diff)[-5:]
        for j in top_diff:
            ax.annotate(gene_names[j], (real_mean[j], gen_mean[j]),
                        fontsize=5, alpha=0.7, color="red")

        ax.set_xlabel("Real mean expression", fontsize=10)
        ax.set_ylabel("Generated mean expression", fontsize=10)
        ax.set_title(f"Gen: {ct}\nPearson r = {r:.4f} (p = {p:.2e})",
                     fontweight="bold", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)

    for idx in range(n_types, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle("Gene Expression Correlation: Real vs Generated\n"
                 "(per-gene mean expression across cells)",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: correlation_scatter.png")


def main():
    parser = argparse.ArgumentParser(description="CLOP-DiT v0.4 Marker Gene Analysis")
    parser.add_argument("--output_dir", type=str, default="figures/marker_genes")
    parser.add_argument("--reference_h5ad", type=str,
                        default="data/processed_h5ad/GSE123902_LungAdreHmCancer_processed.h5ad")
    parser.add_argument("--clop_checkpoint", type=str, default="models/checkpoints/clop_best.pth")
    parser.add_argument("--dit_checkpoint", type=str, default="models/checkpoints/dit_best.pth")
    parser.add_argument("--scgpt_model_dir", type=str, default="models/scgpt_pancancer")
    parser.add_argument("--num_cells", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ──
    logger.info("Loading inference pipeline...")
    clop, dit, tokenizer, text_model, cell_dim = load_inference_pipeline(
        args.clop_checkpoint, args.dit_checkpoint, args.scgpt_model_dir, args.device
    )

    # ── Load scGPT decoder ──
    import scanpy as sc
    from src.architecture.decoder import ScGPTDecoder

    ref_adata = sc.read_h5ad(args.reference_h5ad)
    logger.info(f"Reference: {ref_adata.shape}")

    scgpt_decoder = ScGPTDecoder(
        model_dir=args.scgpt_model_dir,
        device=torch.device(args.device),
    )
    ref_embs = scgpt_decoder.encode(ref_adata)
    logger.info(f"Reference embeddings: {ref_embs.shape}")

    # ── Generate cells for each cell type ──
    all_generated = {}
    for ct_name, prompt_text in CELL_TYPE_PROMPTS.items():
        logger.info(f"\nGenerating {args.num_cells} cells for: {ct_name}")
        cond = encode_text(prompt_text, tokenizer, text_model, clop, args.device)
        cell_embs = generate_cells(dit, cond, args.num_cells, cell_dim)
        decoded = scgpt_decoder.decode(cell_embs.cpu().numpy())
        all_generated[ct_name] = {
            "expression": decoded["expression"],
            "gene_names": decoded["gene_names"],
            "prompt": prompt_text,
        }
        logger.info(f"  Decoded: {decoded['expression'].shape}")

    # ── Decode real reference cells ──
    logger.info("\nDecoding real reference cells...")
    if len(ref_embs) > 500:
        idx = np.random.choice(len(ref_embs), 500, replace=False)
        real_embs_sub = ref_embs[idx]
    else:
        real_embs_sub = ref_embs
    real_decoded = scgpt_decoder.decode(real_embs_sub)
    real_expr = real_decoded["expression"]

    # ── Visualization ──
    gene_names = all_generated[list(all_generated.keys())[0]]["gene_names"]
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # 1. Marker gene violin plots (grid layout)
    logger.info("\n=== Plotting marker gene violins (grid layout) ===")
    for ct_name, markers_info in MARKER_GENES.items():
        plot_marker_violin_grid(
            gene_names, gene_to_idx, real_expr, all_generated,
            ct_name, markers_info["markers"], markers_info["description"],
            output_dir, max_cols=4
        )

    # 2. Cross-cell-type heatmap
    logger.info("\n=== Plotting cross-cell-type heatmap ===")
    plot_cross_celltype_heatmap(gene_names, gene_to_idx, real_expr, all_generated, output_dir)

    # 3. UMAP discrimination
    logger.info("\n=== Plotting cell-type UMAP ===")
    plot_celltype_umap(real_expr, all_generated, output_dir)

    # 4. Discriminative genes (grid layout)
    logger.info("\n=== Plotting discriminative genes ===")
    plot_discriminative_genes_grid(gene_names, all_generated, output_dir, max_cols=2)

    # 5. Distribution comparison
    logger.info("\n=== Plotting distribution comparison ===")
    plot_distribution_comparison(gene_names, gene_to_idx, real_expr, all_generated, output_dir)

    # 6. Correlation scatter
    logger.info("\n=== Plotting correlation scatter ===")
    plot_correlation_scatter(real_expr, all_generated, gene_names, output_dir, max_cols=2)

    # 7. Summary
    summary = {
        "num_cell_types_generated": len(all_generated),
        "cells_per_type": args.num_cells,
        "num_genes_decoded": len(gene_names),
        "reference_dataset": args.reference_h5ad,
        "reference_cells_decoded": len(real_expr),
        "figures_generated": [
            "markers_*.png (per cell type, grid layout)",
            "marker_heatmap_cross_celltype.png",
            "celltype_discrimination_umap.png",
            "discriminative_genes.png",
            "distribution_comparison.png",
            "correlation_scatter.png",
        ],
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
    logger.info(f"Marker Gene Analysis Complete (v0.4 improved)")
    logger.info(f"{'='*60}")
    logger.info(f"  Cell types generated: {len(all_generated)}")
    logger.info(f"  Cells per type: {args.num_cells}")
    logger.info(f"  Genes decoded: {len(gene_names)}")
    logger.info(f"  Reference cells: {len(real_expr)}")
    logger.info(f"  Figures saved to: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
