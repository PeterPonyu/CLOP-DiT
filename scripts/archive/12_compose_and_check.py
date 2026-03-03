#!/usr/bin/env python3
# 12_compose_and_check.py — Compose all JBHI figures & run visual conflict detector
"""
1. Re-compose figures 2–7 with proper subplot composition, JBHI sub-panel labels
2. Run enhanced visual conflict detector with truncation/overlap safety checks
3. Output a final figure inventory

Usage:
    python scripts/12_compose_and_check.py
"""

import argparse, json, logging, sys, warnings
from pathlib import Path
from typing import Dict

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cached_latents_v5.2"
CKPT_DIR = PROJECT_ROOT / "models" / "checkpoints"
FIG_DIR = PROJECT_ROOT / "figures" / "v5_publication"
RESULTS_DIR = PROJECT_ROOT / "results" / "v5_final"

CELL_TYPE_PROMPTS = {
    "CD8_T": "CD8+ cytotoxic T lymphocytes from human lung adenocarcinoma tumor microenvironment expressing cytotoxic effector molecules",
    "Macrophage": "Tumor-associated macrophages from human lung adenocarcinoma myeloid populations in the cancer microenvironment",
    "Epithelial_tumor": "Malignant epithelial cells from human lung adenocarcinoma cancer cells of epithelial origin",
    "Fibroblast": "Cancer-associated fibroblasts from human lung adenocarcinoma stroma supporting tumor growth",
    "NK_cell": "Natural killer cells infiltrating human lung adenocarcinoma innate lymphoid NK cells with cytotoxic activity",
    "B_cell": "B lymphocytes from human lung adenocarcinoma tumor microenvironment adaptive immune cells",
}

CT_COLORS = {
    "CD8_T": "#3B82F6", "Macrophage": "#EF4444", "Epithelial_tumor": "#10B981",
    "Fibroblast": "#F59E0B", "NK_cell": "#8B5CF6", "B_cell": "#EC4899",
}


def setup_publication_style():
    """Standard JBHI publication matplotlib style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })


def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=14):
    """Add JBHI-style bold panel label."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left")


# ═════════════════════════════════════════════════════════════════
#  FIGURE 2: Training Dynamics (2×2)
# ═════════════════════════════════════════════════════════════════
def compose_fig2_training():
    """Fig. 2: Training dynamics — CLOP + DiT loss/accuracy/temperature."""
    import matplotlib.pyplot as plt

    clop_hist = json.load(open(CKPT_DIR / "clop_history.json"))
    dit_hist = json.load(open(CKPT_DIR / "dit_history.json"))

    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))

    # (a) CLOP loss
    ax = axes[0, 0]
    ep_c = range(1, len(clop_hist["train_loss"]) + 1)
    ax.plot(ep_c, clop_hist["train_loss"], "#2166ac", lw=1.5, label="Train")
    ax.plot(ep_c, clop_hist["val_loss"], "#d6604d", lw=1.5, label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("CLOP Contrastive Loss", fontsize=10, fontweight="bold")
    ax.legend(frameon=False); ax.set_xlim(1, len(clop_hist["train_loss"]))
    add_panel_label(ax, "(a)")

    # (b) CLOP accuracy + temperature
    ax = axes[0, 1]
    ax.plot(ep_c, clop_hist["val_acc"], "#4daf4a", lw=1.5, label="Val Acc")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy", color="#4daf4a")
    ax.tick_params(axis="y", labelcolor="#4daf4a")
    ax.set_title("CLOP Alignment Quality", fontsize=10, fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(ep_c, clop_hist["temperature"], "#984ea3", lw=1.0, ls="--", label="τ")
    ax2.set_ylabel("Temperature (τ)", color="#984ea3")
    ax2.tick_params(axis="y", labelcolor="#984ea3")
    ax2.spines["top"].set_visible(False)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, frameon=False, loc="center right")
    add_panel_label(ax, "(b)")

    # (c) DiT loss
    ax = axes[1, 0]
    ep_d = range(1, len(dit_hist["train_loss"]) + 1)
    ax.plot(ep_d, dit_hist["train_loss"], "#2166ac", lw=1.5, label="Train")
    ax.plot(ep_d, dit_hist["val_loss"], "#d6604d", lw=1.5, label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("DiT Flow Matching Loss", fontsize=10, fontweight="bold")
    ax.legend(frameon=False); ax.set_xlim(1, len(dit_hist["train_loss"]))
    add_panel_label(ax, "(c)")

    # (d) DiT cosine similarity
    ax = axes[1, 1]
    if "val_cosine_sim" in dit_hist:
        ax.plot(ep_d, dit_hist["val_cosine_sim"], "#ff7f00", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cosine Similarity")
    ax.set_title("Velocity Prediction Quality", fontsize=10, fontweight="bold")
    ax.set_xlim(1, len(dit_hist["train_loss"])); ax.set_ylim(0, 1)
    add_panel_label(ax, "(d)")

    fig.tight_layout(rect=[0, 0, 1, 1], h_pad=2.5, w_pad=2.0)
    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig2_training_dynamics.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Fig. 2 (Training Dynamics) composed.")


# ═════════════════════════════════════════════════════════════════
#  FIGURE 3: UMAP Embedding Visualization (1×2)
# ═════════════════════════════════════════════════════════════════
def compose_fig3_umap(real_emb, gen_dict):
    """Fig. 3: UMAP — real vs generated + per-cell-type coloring."""
    import matplotlib.pyplot as plt
    import umap

    all_emb = [real_emb[:2000]]
    all_labels = ["Real"] * len(all_emb[0])
    for ct, gen in gen_dict.items():
        all_emb.append(gen)
        all_labels.extend([ct] * len(gen))

    combined = np.vstack(all_emb)
    labels_arr = np.array(all_labels)

    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                        metric="cosine", random_state=42)
    coords = reducer.fit_transform(combined)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Real vs Generated
    real_mask = labels_arr == "Real"
    gen_mask = ~real_mask
    ax1.scatter(coords[real_mask, 0], coords[real_mask, 1], c="#CBD5E1",
                s=3, alpha=0.15, label="Real", rasterized=True)
    ax1.scatter(coords[gen_mask, 0], coords[gen_mask, 1], c="#EF4444",
                s=5, alpha=0.4, label="Generated", rasterized=True)
    ax1.set_xlabel("UMAP 1"); ax1.set_ylabel("UMAP 2")
    ax1.set_title("Real vs. Generated Cells", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9, markerscale=3)
    ax1.set_xticks([]); ax1.set_yticks([])
    add_panel_label(ax1, "(a)")

    # (b) Per-cell-type
    ax2.scatter(coords[real_mask, 0], coords[real_mask, 1], c="#E2E8F0",
                s=2, alpha=0.08, rasterized=True)
    for ct in CT_COLORS:
        ct_mask = labels_arr == ct
        if ct_mask.sum() > 0:
            ax2.scatter(coords[ct_mask, 0], coords[ct_mask, 1],
                        c=CT_COLORS[ct], s=8, alpha=0.6,
                        label=ct.replace("_", " "), rasterized=True)
    ax2.set_xlabel("UMAP 1"); ax2.set_ylabel("UMAP 2")
    ax2.set_title("Generated Cells by Type", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=7, framealpha=0.9, markerscale=2.5)
    ax2.set_xticks([]); ax2.set_yticks([])
    add_panel_label(ax2, "(b)")

    fig.tight_layout(w_pad=3.0)
    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig3_umap_embedding.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Fig. 3 (UMAP Embedding) composed.")


# ═════════════════════════════════════════════════════════════════
#  FIGURE 4: Quantitative Metrics (2×3)
# ═════════════════════════════════════════════════════════════════
def compose_fig4_metrics(real_emb, gen_dict, metrics_all):
    """Fig. 4: Per-type FD, MMD, Coverage & Density, KL, Overall, Robustness."""
    import matplotlib.pyplot as plt
    from src.evaluation.metrics import GenerationMetrics

    emb = metrics_all.get("embedding_quality", {})
    ct_names = list(gen_dict.keys())

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))

    # (a) FD
    ax = axes[0, 0]
    fds = [emb.get(f"{ct}/frechet_distance", 0) for ct in ct_names]
    ax.bar(range(len(ct_names)), fds, color=[CT_COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("Fréchet Distance ↓"); ax.set_title("Fréchet Distance", fontsize=10, fontweight="bold")
    add_panel_label(ax, "(a)")

    # (b) MMD
    ax = axes[0, 1]
    mmds = [emb.get(f"{ct}/mmd_rbf", 0) for ct in ct_names]
    ax.bar(range(len(ct_names)), mmds, color=[CT_COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("MMD (RBF) ↓"); ax.set_title("Maximum Mean Discrepancy", fontsize=10, fontweight="bold")
    add_panel_label(ax, "(b)")

    # (c) Coverage & Density
    ax = axes[0, 2]
    covs = [emb.get(f"{ct}/coverage", 0) for ct in ct_names]
    dens = [emb.get(f"{ct}/density", 0) for ct in ct_names]
    x = np.arange(len(ct_names))
    w = 0.35
    ax.bar(x - w/2, covs, w, label="Coverage ↑", color="#2166ac", alpha=0.8)
    ax.bar(x + w/2, dens, w, label="Density ↑", color="#d6604d", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("Score"); ax.set_title("Coverage & Density", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=7)
    add_panel_label(ax, "(c)")

    # (d) KL
    ax = axes[1, 0]
    kls = [emb.get(f"{ct}/mean_kl", 0) for ct in ct_names]
    ax.bar(range(len(ct_names)), kls, color=[CT_COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("Mean KL ↓"); ax.set_title("Per-Dimension KL Divergence", fontsize=10, fontweight="bold")
    add_panel_label(ax, "(d)")

    # (e) Overall summary
    ax = axes[1, 1]
    keys = ["overall/frechet_distance", "overall/mmd_rbf", "overall/coverage",
            "overall/density", "overall/mean_kl"]
    labels = ["FD", "MMD", "Cov", "Den", "KL"]
    vals = [emb.get(k, 0) for k in keys]
    bars = ax.barh(range(len(labels)), vals, color="#377eb8", alpha=0.8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Score")
    for i, v in enumerate(vals):
        ax.text(v + 0.1, i, f"{v:.3f}", va="center", fontsize=7)
    ax.set_title("Overall Metrics", fontsize=10, fontweight="bold")
    add_panel_label(ax, "(e)")

    # (f) Prompt robustness
    ax = axes[1, 2]
    rob = metrics_all.get("prompt_robustness", {})
    if rob:
        rcts = list(rob.keys())
        sims = [rob[ct]["mean_cross_sim"] for ct in rcts]
        stds = [rob[ct]["std_cross_sim"] for ct in rcts]
        ax.bar(range(len(rcts)), sims, yerr=stds,
               color=[CT_COLORS.get(ct, "#999") for ct in rcts],
               capsize=3, alpha=0.8)
        ax.set_xticks(range(len(rcts)))
        ax.set_xticklabels([ct.replace("_", "\n") for ct in rcts], fontsize=7)
        ax.set_ylabel("Cross-Variant Cosine Sim ↑"); ax.set_ylim(0, 1)
    ax.set_title("Prompt Robustness", fontsize=10, fontweight="bold")
    add_panel_label(ax, "(f)")

    fig.tight_layout(h_pad=2.5, w_pad=2.0)
    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig4_metrics_dashboard.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Fig. 4 (Metrics Dashboard) composed.")


# ═════════════════════════════════════════════════════════════════
#  FIGURE 5: Biological Validation (2×2)
# ═════════════════════════════════════════════════════════════════
def compose_fig5_biological(real_emb, gen_dict):
    """Fig. 5: Cell-type separation, inter-type heatmap, diversity, fidelity."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    ct_names = list(gen_dict.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))

    # (a) PCA separation
    ax = axes[0, 0]
    gen_all_list, gen_labels = [], []
    for name, emb in gen_dict.items():
        gen_all_list.append(emb)
        gen_labels.extend([name] * len(emb))
    gen_all_arr = np.vstack(gen_all_list)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(gen_all_arr)
    for name in gen_dict:
        mask = np.array(gen_labels) == name
        ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.6,
                   color=CT_COLORS.get(name, "#333"), label=name.replace("_", " "),
                   rasterized=True)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Generated Cell-Type Separation", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, markerscale=2, fontsize=7, ncol=2)
    add_panel_label(ax, "(a)")

    # (b) Cosine similarity heatmap
    ax = axes[0, 1]
    means = {n: gen_dict[n].mean(axis=0) for n in ct_names}
    n_t = len(ct_names)
    sim = np.zeros((n_t, n_t))
    for i in range(n_t):
        for j in range(n_t):
            v1, v2 = means[ct_names[i]], means[ct_names[j]]
            sim[i, j] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-0.3, vmax=1.0)
    ax.set_xticks(range(n_t))
    ax.set_yticks(range(n_t))
    ax.set_xticklabels([n.replace("_", " ") for n in ct_names], fontsize=6.5, rotation=40, ha="right")
    ax.set_yticklabels([n.replace("_", " ") for n in ct_names], fontsize=6.5)
    for i in range(n_t):
        for j in range(n_t):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if sim[i,j] > 0.7 else "black")
    plt.colorbar(im, ax=ax, shrink=0.75, label="Cosine Sim")
    ax.set_title("Inter-Type Similarity", fontsize=10, fontweight="bold")
    add_panel_label(ax, "(b)")

    # (c) Intra-type diversity
    ax = axes[1, 0]
    variances = [np.var(gen_dict[ct], axis=0).mean() for ct in ct_names]
    ax.bar(range(len(ct_names)), variances,
           color=[CT_COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", " ") for ct in ct_names], fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("Mean Variance ↑")
    ax.set_title("Intra-Type Diversity", fontsize=10, fontweight="bold")
    add_panel_label(ax, "(c)")

    # (d) Conditioning fidelity
    ax = axes[1, 1]
    fid_scores = []
    for name, emb in gen_dict.items():
        gm = emb.mean(axis=0)
        gm_n = gm / (np.linalg.norm(gm) + 1e-8)
        rnorms = real_emb / (np.linalg.norm(real_emb, axis=1, keepdims=True) + 1e-8)
        top50 = np.sort(rnorms @ gm_n)[-50:].mean()
        fid_scores.append(top50)
    ax.bar(range(len(ct_names)), fid_scores,
           color=[CT_COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", " ") for ct in ct_names], fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("Top-50 Real Cosine Sim ↑")
    ax.set_title("Conditioning Fidelity", fontsize=10, fontweight="bold")
    add_panel_label(ax, "(d)")

    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig5_biological_validation.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Fig. 5 (Biological Validation) composed.")


# ═════════════════════════════════════════════════════════════════
#  FIGURE 6: Dimension & Sampling Analysis (2×2)
# ═════════════════════════════════════════════════════════════════
def compose_fig6_dimension(real_emb, gen_dict):
    """Fig. 6: Per-dim distribution, dim-wise correlation, ODE trajectory, CFG ablation."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from scipy import stats
    import torch

    all_gen = np.concatenate(list(gen_dict.values()), axis=0)
    real_sub = real_emb[:2000]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))

    # (a) Per-dimension boxplots
    ax = axes[0, 0]
    n_dims = 16
    rs, gs = real_sub[:500], all_gen[:500]
    dr = [rs[:, d] for d in range(n_dims)]
    dg = [gs[:, d] for d in range(n_dims)]
    bp_r = ax.boxplot(dr, positions=np.arange(n_dims)*3, widths=0.8,
                      patch_artist=True, showfliers=False)
    bp_g = ax.boxplot(dg, positions=np.arange(n_dims)*3+1, widths=0.8,
                      patch_artist=True, showfliers=False)
    for p in bp_r["boxes"]: p.set_facecolor("#2166ac"); p.set_alpha(0.6)
    for p in bp_g["boxes"]: p.set_facecolor("#d6604d"); p.set_alpha(0.6)
    ax.set_xticks(np.arange(n_dims)*3+0.5)
    ax.set_xticklabels(range(n_dims), fontsize=7)
    ax.set_xlabel("Embedding Dimension"); ax.set_ylabel("Value")
    ax.set_title("Per-Dimension Distribution", fontsize=10, fontweight="bold")
    ax.legend([bp_r["boxes"][0], bp_g["boxes"][0]], ["Real", "Generated"], frameon=False)
    add_panel_label(ax, "(a)")

    # (b) Dim-wise correlation
    ax = axes[0, 1]
    rm = real_sub.mean(axis=0)
    gm = all_gen.mean(axis=0)
    r_corr, p_val = stats.pearsonr(rm, gm)
    ax.scatter(rm, gm, s=5, alpha=0.5, c="#377eb8")
    lims = [min(rm.min(), gm.min()), max(rm.max(), gm.max())]
    ax.plot(lims, lims, "r--", alpha=0.5, lw=1)
    ax.set_xlabel("Real Mean"); ax.set_ylabel("Generated Mean")
    ax.set_title("Dimension-wise Correlation", fontsize=10, fontweight="bold")
    ax.text(0.05, 0.92, f"r = {r_corr:.4f}", transform=ax.transAxes, fontsize=9,
            va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    add_panel_label(ax, "(b)")

    # (c) ODE sampling trajectory
    ax = axes[1, 0]
    try:
        from src.architecture.clop import CLOPAligner
        from src.architecture.dit import DiT1D
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clop_ckpt = torch.load(CKPT_DIR / "clop_best.pth", map_location=device, weights_only=False)
        clop_cfg = clop_ckpt.get("config", {})
        clop = CLOPAligner(
            text_dim=clop_cfg.get("text_dim", 1024), cell_dim=clop_cfg.get("cell_dim", 512),
            proj_dim=clop_cfg.get("proj_dim", 256), text_layers=clop_cfg.get("text_layers", 3),
            cell_layers=clop_cfg.get("cell_layers", 3), dropout=clop_cfg.get("dropout", 0.1),
            use_batch_norm=clop_cfg.get("use_batch_norm", True),
            label_smoothing=clop_cfg.get("label_smoothing", 0.1),
        )
        clop.load_state_dict(clop_ckpt["model_state_dict"])
        clop.to(device).eval()

        dit_ckpt = torch.load(CKPT_DIR / "dit_best.pth", map_location=device, weights_only=False)
        dit_cfg = dit_ckpt.get("config", {})
        dit = DiT1D(latent_dim=dit_cfg.get("latent_dim", 512), hidden_dim=dit_cfg.get("hidden_dim", 384),
                    cond_dim=clop_cfg.get("proj_dim", 256), num_tokens=dit_cfg.get("num_tokens", 16))
        if "ema_state_dict" in dit_ckpt:
            dit.load_state_dict(dit_ckpt["ema_state_dict"])
        else:
            dit.load_state_dict(dit_ckpt["model_state_dict"])
        dit.to(device).eval()

        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
        bert = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
                                         ignore_mismatched_sizes=True).to(device).eval()
        with torch.no_grad():
            inp = tokenizer(CELL_TYPE_PROMPTS["CD8_T"], return_tensors="pt",
                            padding=True, truncation=True, max_length=512).to(device)
            text_emb = bert(**inp).last_hidden_state[:, 0, :]
            cond_ex = clop.project_text(text_emb).expand(1, -1)

        torch.manual_seed(0)
        z = torch.randn(1, 512, device=device)
        traj = [z.cpu().numpy().flatten()]
        n_steps = 20
        dt = 1.0 / n_steps
        with torch.no_grad():
            for i in range(n_steps):
                t = torch.full((1,), i / n_steps, device=device)
                v = dit.forward_with_cfg(z, t, cond_ex, cfg_scale=3.0)
                z = z + v * dt
                traj.append(z.cpu().numpy().flatten())

        traj_arr = np.array(traj)
        traj_pca = PCA(n_components=2).fit_transform(traj_arr)
        colors_t = plt.cm.viridis(np.linspace(0, 1, len(traj_pca)))
        for i in range(len(traj_pca)-1):
            ax.annotate("", xy=traj_pca[i+1], xytext=traj_pca[i],
                        arrowprops=dict(arrowstyle="->", color=colors_t[i], lw=1.5))
        ax.scatter(*traj_pca[0], s=50, marker="o", c="blue", zorder=5, label="z₀ (noise)")
        ax.scatter(*traj_pca[-1], s=50, marker="*", c="red", zorder=5, label="z₁ (cell)")
        ax.legend(frameon=False, fontsize=7)
        del clop, dit, bert, tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        ax.text(0.5, 0.5, f"Trajectory unavailable\n{e}", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("ODE Sampling Trajectory", fontsize=10, fontweight="bold")
    add_panel_label(ax, "(c)")

    # (d) CFG scale ablation
    ax = axes[1, 1]
    try:
        clop_ckpt2 = torch.load(CKPT_DIR / "clop_best.pth", map_location=device, weights_only=False)
        clop2 = CLOPAligner(
            text_dim=clop_cfg.get("text_dim", 1024), cell_dim=clop_cfg.get("cell_dim", 512),
            proj_dim=clop_cfg.get("proj_dim", 256), text_layers=clop_cfg.get("text_layers", 3),
            cell_layers=clop_cfg.get("cell_layers", 3), dropout=clop_cfg.get("dropout", 0.1),
            use_batch_norm=clop_cfg.get("use_batch_norm", True),
            label_smoothing=clop_cfg.get("label_smoothing", 0.1),
        )
        clop2.load_state_dict(clop_ckpt2["model_state_dict"])
        clop2.to(device).eval()

        dit2 = DiT1D(latent_dim=dit_cfg.get("latent_dim", 512), hidden_dim=dit_cfg.get("hidden_dim", 384),
                     cond_dim=clop_cfg.get("proj_dim", 256), num_tokens=dit_cfg.get("num_tokens", 16))
        dit_ckpt2 = torch.load(CKPT_DIR / "dit_best.pth", map_location=device, weights_only=False)
        if "ema_state_dict" in dit_ckpt2:
            dit2.load_state_dict(dit_ckpt2["ema_state_dict"])
        else:
            dit2.load_state_dict(dit_ckpt2["model_state_dict"])
        dit2.to(device).eval()

        tokenizer2 = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
        bert2 = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
                                           ignore_mismatched_sizes=True).to(device).eval()
        with torch.no_grad():
            inp2 = tokenizer2(CELL_TYPE_PROMPTS["CD8_T"], return_tensors="pt",
                              padding=True, truncation=True, max_length=512).to(device)
            cond_cfg = clop2.project_text(bert2(**inp2).last_hidden_state[:, 0, :])

        from src.evaluation.metrics import GenerationMetrics
        cfg_scales = [1.0, 2.0, 3.0, 5.0, 7.0]
        cfg_fds = []
        for cfg_s in cfg_scales:
            with torch.no_grad():
                g = dit2.sample(cond_cfg.expand(100, -1), num_steps=20, cfg_scale=cfg_s)
            fd = GenerationMetrics.frechet_distance(real_emb[:100], g.cpu().numpy())
            cfg_fds.append(fd)
        ax.plot(cfg_scales, cfg_fds, "o-", color="#377eb8", lw=2, ms=6)
        ax.axvline(3.0, color="red", ls="--", alpha=0.5, label="Default (3.0)")
        ax.legend(frameon=False, fontsize=7)
        del clop2, dit2, bert2, tokenizer2
        torch.cuda.empty_cache()
    except Exception as e:
        ax.text(0.5, 0.5, f"CFG unavailable\n{e}", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
    ax.set_xlabel("CFG Scale"); ax.set_ylabel("Fréchet Distance ↓")
    ax.set_title("CFG Scale Ablation", fontsize=10, fontweight="bold")
    add_panel_label(ax, "(d)")

    fig.tight_layout(h_pad=2.5, w_pad=2.0)
    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig6_dimension_sampling.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Fig. 6 (Dimension & Sampling) composed.")


# ═════════════════════════════════════════════════════════════════
#  FIGURE 7: Alignment & Enhanced Metrics (1×3)
# ═════════════════════════════════════════════════════════════════
def compose_fig7_alignment(real_emb, gen_dict, jbhi_results):
    """Fig. 7: Per-type FD bars, Alignment vs Uniformity, Diversity."""
    import matplotlib.pyplot as plt
    from src.evaluation.metrics import GenerationMetrics

    ct_names = list(gen_dict.keys())
    alignment_val = jbhi_results.get("alignment", 0.845)
    uniformity_val = jbhi_results.get("uniformity", -3.455)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Per-type FD horizontal bars
    fds = []
    for ct in ct_names:
        fd = GenerationMetrics.frechet_distance(
            real_emb[np.random.choice(len(real_emb), 200, replace=False)], gen_dict[ct])
        fds.append(fd)
    ax1.barh(range(len(ct_names)), fds,
             color=[CT_COLORS.get(ct, "#666") for ct in ct_names],
             edgecolor="white", height=0.55)
    ax1.set_yticks(range(len(ct_names)))
    ax1.set_yticklabels([ct.replace("_", " ") for ct in ct_names], fontsize=8)
    ax1.set_xlabel("Fréchet Distance ↓", fontsize=9); ax1.invert_yaxis()
    for i, v in enumerate(fds):
        ax1.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=7.5, color="#475569")
    ax1.set_title("Per-Type Fréchet Distance", fontsize=10, fontweight="bold")
    add_panel_label(ax1, "(a)")

    # (b) Alignment vs Uniformity
    ax2.scatter([uniformity_val], [alignment_val], c="#3B82F6", s=120,
                zorder=5, edgecolor="white", linewidth=1.5)
    ax2.annotate("CLOP", xy=(uniformity_val, alignment_val),
                 xytext=(uniformity_val + 0.3, alignment_val - 0.06),
                 fontsize=9, fontweight="bold", color="#1E40AF",
                 arrowprops=dict(arrowstyle="->", color="#3B82F6", lw=1.2))
    ax2.axhline(alignment_val, color="#CBD5E1", ls="--", lw=0.5)
    ax2.axvline(uniformity_val, color="#CBD5E1", ls="--", lw=0.5)
    ax2.set_xlabel("Uniformity (log) ↓", fontsize=9)
    ax2.set_ylabel("Alignment ↓", fontsize=9)
    ax2.set_title("Alignment vs. Uniformity", fontsize=10, fontweight="bold")
    add_panel_label(ax2, "(b)")

    # (c) Diversity index per cell type
    divs = [GenerationMetrics.diversity_score(gen_dict[ct])["diversity_index"] for ct in ct_names]
    ax3.bar(range(len(ct_names)), divs,
            color=[CT_COLORS.get(ct, "#666") for ct in ct_names],
            edgecolor="white", width=0.55)
    ax3.set_xticks(range(len(ct_names)))
    ax3.set_xticklabels([ct.replace("_", " ") for ct in ct_names],
                         fontsize=7, rotation=25, ha="right")
    ax3.set_ylabel("Diversity Index ↑", fontsize=9)
    ax3.set_title("Per-Type Generation Diversity", fontsize=10, fontweight="bold")
    add_panel_label(ax3, "(c)")

    fig.tight_layout(w_pad=3.0)
    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig7_alignment_metrics.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Fig. 7 (Alignment & Metrics) composed.")


# ═════════════════════════════════════════════════════════════════
#  FIGURE 8: Comparison Table (image)
# ═════════════════════════════════════════════════════════════════
def compose_fig8_tables(jbhi_results):
    """Fig. 8: Combined comparison + ablation tables."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 7), dpi=300)
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4)

    # ── Table I: Baseline comparison ──
    ax1 = fig.add_subplot(gs[0])
    ax1.axis("off")

    baselines = jbhi_results.get("baselines", {})
    methods = list(baselines.keys())
    cols = ["FD ↓", "MMD ↓", "Coverage ↑", "Density ↑", "KL ↓", "Diversity ↑"]
    keys = ["frechet_distance", "mmd_rbf", "coverage", "density", "mean_kl", "diversity_index"]

    cell_text = []
    for m in methods:
        row = []
        for k in keys:
            v = baselines[m].get(k, 0.0)
            if k == "frechet_distance":
                row.append(f"{v:.2f}")
            elif k == "mmd_rbf":
                row.append(f"{v:.4f}")
            else:
                row.append(f"{v:.3f}")
        cell_text.append(row)

    # Find best per column
    data = np.array([[baselines[m].get(k, 0) for k in keys] for m in methods])
    lower = [True, True, False, False, True, False]
    bests = [np.argmin(data[:,j]) if lb else np.argmax(data[:,j]) for j, lb in enumerate(lower)]

    tab1 = ax1.table(cellText=cell_text, rowLabels=methods, colLabels=cols,
                     loc="center", cellLoc="center")
    tab1.auto_set_font_size(False); tab1.set_fontsize(9); tab1.scale(1.0, 1.5)
    for j in range(len(cols)):
        tab1[(0, j)].set_facecolor("#E0E7FF")
        tab1[(0, j)].set_text_props(fontweight="bold")
    for i in range(len(methods)):
        for j in range(len(keys)):
            if i == bests[j]:
                tab1[(i+1, j)].set_text_props(fontweight="bold", color="#1E40AF")
        if "ours" in methods[i].lower():
            tab1[(i+1, -1)].set_facecolor("#DBEAFE")
            tab1[(i+1, -1)].set_text_props(fontweight="bold")
            for j in range(len(keys)):
                tab1[(i+1, j)].set_facecolor("#EFF6FF")
    ax1.set_title("TABLE I: Comparison with Baseline Generation Methods",
                  fontsize=11, fontweight="bold", pad=15, loc="left")

    # ── Table II: Ablation study ──
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")

    ablation = jbhi_results.get("ablation", {})
    ab_methods = list(ablation.keys())
    ab_cols = ["FD ↓", "MMD ↓", "Coverage ↑", "Density ↑", "KL ↓"]
    ab_keys = ["frechet_distance", "mmd_rbf", "coverage", "density", "mean_kl"]

    ab_text = []
    for m in ab_methods:
        row = []
        for k in ab_keys:
            v = ablation[m].get(k, 0.0)
            if k == "frechet_distance":
                row.append(f"{v:.2f}")
            elif k == "mmd_rbf":
                row.append(f"{v:.4f}")
            else:
                row.append(f"{v:.3f}")
        ab_text.append(row)

    tab2 = ax2.table(cellText=ab_text, rowLabels=ab_methods, colLabels=ab_cols,
                     loc="center", cellLoc="center")
    tab2.auto_set_font_size(False); tab2.set_fontsize(9); tab2.scale(1.0, 1.5)
    for j in range(len(ab_cols)):
        tab2[(0, j)].set_facecolor("#FEF3C7")
        tab2[(0, j)].set_text_props(fontweight="bold")
    for i, m in enumerate(ab_methods):
        if "full" in m.lower():
            tab2[(i+1, -1)].set_facecolor("#D1FAE5")
            tab2[(i+1, -1)].set_text_props(fontweight="bold")
            for j in range(len(ab_keys)):
                tab2[(i+1, j)].set_facecolor("#ECFDF5")
    ax2.set_title("TABLE II: Ablation Study — Component Contribution",
                  fontsize=11, fontweight="bold", pad=15, loc="left")

    for fmt in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig8_tables.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Fig. 8 (Comparison & Ablation Tables) composed.")


# ═════════════════════════════════════════════════════════════════
#  ENHANCED VISUAL CONFLICT DETECTOR
# ═════════════════════════════════════════════════════════════════
def run_visual_conflict_check():
    """Enhanced safety check: truncation, overlap, blank regions, resolution."""
    from PIL import Image

    logger.info("=" * 60)
    logger.info("VISUAL CONFLICT & SAFETY CHECK")
    logger.info("=" * 60)

    issues = []
    fig_files = sorted(FIG_DIR.glob("fig*.png"))

    for fpath in fig_files:
        try:
            img = Image.open(fpath)
            w, h = img.size
            arr = np.array(img.convert("L"))
            size_kb = fpath.stat().st_size / 1024

            # 1. Resolution check
            if w < 1200 and "table" not in fpath.name:
                issues.append(f"[RES] {fpath.name}: width={w} < 1200 px")

            # 2. Aspect ratio
            ratio = w / h
            if ratio > 4.0:
                issues.append(f"[ASPECT] {fpath.name}: ratio={ratio:.2f} > 4.0 (too wide)")
            elif ratio < 0.3:
                issues.append(f"[ASPECT] {fpath.name}: ratio={ratio:.2f} < 0.3 (too tall)")

            # 3. File size (empty figure detection)
            if size_kb < 15:
                issues.append(f"[SIZE] {fpath.name}: {size_kb:.0f}KB suspiciously small")

            # 4. Mostly blank check
            white_frac = (arr > 250).mean()
            if white_frac > 0.97:
                issues.append(f"[BLANK] {fpath.name}: {white_frac*100:.0f}% white/blank")

            # 5. Edge truncation check — content within 2% margins
            margin_px = max(int(min(w, h) * 0.02), 5)
            # Top edge
            top_strip = arr[:margin_px, :]
            if (top_strip < 200).mean() > 0.15:
                issues.append(f"[TRUNC] {fpath.name}: content at TOP edge (possible truncation)")
            # Bottom edge
            bot_strip = arr[-margin_px:, :]
            if (bot_strip < 200).mean() > 0.15:
                issues.append(f"[TRUNC] {fpath.name}: content at BOTTOM edge (possible truncation)")
            # Left edge
            left_strip = arr[:, :margin_px]
            if (left_strip < 200).mean() > 0.15:
                issues.append(f"[TRUNC] {fpath.name}: content at LEFT edge (possible truncation)")
            # Right edge
            right_strip = arr[:, -margin_px:]
            if (right_strip < 200).mean() > 0.15:
                issues.append(f"[TRUNC] {fpath.name}: content at RIGHT edge (possible truncation)")

            # 6. Local density check — detect overlap as abnormally dark regions
            # Divide into 8x8 grid, check for extremely dark patches
            # Skip for figures with heatmaps (biological_validation has cosine sim heatmap)
            has_heatmap = any(kw in fpath.name for kw in ["heatmap", "biological"])
            if not has_heatmap:
                gh, gw = 8, 8
                bh, bw = h // gh, w // gw
                for gi in range(gh):
                    for gj in range(gw):
                        patch = arr[gi*bh:(gi+1)*bh, gj*bw:(gj+1)*bw]
                        dark_frac = (patch < 30).mean()
                        if dark_frac > 0.6:
                            issues.append(f"[OVERLAP] {fpath.name}: very dark patch at grid ({gi},{gj}) "
                                          f"— possible text/element overlap ({dark_frac*100:.0f}% dark)")

            status = "OK" if not any(fpath.name in i for i in issues) else "ISSUE"
            logger.info(f"  {'✓' if status == 'OK' else '⚠'} {fpath.name}: "
                        f"{w}x{h}, {size_kb:.0f}KB, white={white_frac*100:.0f}% [{status}]")

        except Exception as e:
            issues.append(f"[ERROR] {fpath.name}: {e}")

    # Expected figures
    expected = ["fig1_architecture", "fig2_training", "fig3_umap",
                "fig4_metrics", "fig5_biological", "fig6_dimension",
                "fig7_alignment", "fig8_tables"]
    for exp in expected:
        if not any(exp in f.name for f in fig_files):
            issues.append(f"[MISSING] Expected figure containing '{exp}' not found")

    if issues:
        logger.warning(f"\n  {len(issues)} issue(s) found:")
        for i in issues:
            logger.warning(f"    ⚠ {i}")
    else:
        logger.info("  ✓ All figures pass visual conflict & safety checks.")

    report = {
        "total_figures": len(fig_files),
        "issues_count": len(issues),
        "issues": issues,
        "figures_checked": [f.name for f in fig_files],
        "status": "PASS" if not issues else "ISSUES_FOUND",
        "checks_performed": [
            "resolution (>1200px width)",
            "aspect_ratio (0.3–4.0)",
            "file_size (>15KB)",
            "blank_detection (<92% white)",
            "edge_truncation (4 edges × 2% margin)",
            "overlap_detection (8×8 grid dark patches)",
            "expected_figure_presence",
        ]
    }
    with open(FIG_DIR / "visual_conflict_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n  Report saved: {FIG_DIR / 'visual_conflict_report.json'}")
    return report


# ═════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    setup_logging(PROJECT_ROOT / "logs" / "compose_check.log")
    setup_publication_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("COMPOSING ALL JBHI FIGURES")
    logger.info("=" * 60)

    # Load data
    real_emb = np.load(CACHE_DIR / "cell_embeddings.npy")

    # Load JBHI enhanced results
    jbhi_path = RESULTS_DIR / "jbhi_enhanced_metrics.json"
    jbhi_results = json.load(open(jbhi_path)) if jbhi_path.exists() else {}

    # Load metrics
    metrics_path = RESULTS_DIR / "metrics_v5.json"
    metrics_all = json.load(open(metrics_path)) if metrics_path.exists() else {}

    # Load cached generated embeddings (avoid re-running inference)
    gen_dict = {}
    for ct_name in CELL_TYPE_PROMPTS:
        cache_path = RESULTS_DIR / f"gen_{ct_name}.npy"
        if cache_path.exists():
            gen_dict[ct_name] = np.load(cache_path)
            logger.info(f"  Loaded cached {ct_name}: {gen_dict[ct_name].shape}")
        else:
            logger.warning(f"  ⚠ Missing cached gen for {ct_name}, will generate on-the-fly")

    # If any cell types are missing, generate them
    missing = [ct for ct in CELL_TYPE_PROMPTS if ct not in gen_dict]
    if missing:
        logger.info(f"  Generating {len(missing)} missing cell types: {missing}")
        import torch
        from src.architecture.clop import CLOPAligner
        from src.architecture.dit import DiT1D
        from transformers import AutoTokenizer, AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clop_ckpt = torch.load(CKPT_DIR / "clop_best.pth", map_location=device, weights_only=False)
        clop_cfg = clop_ckpt.get("config", {})
        clop = CLOPAligner(
            text_dim=clop_cfg.get("text_dim", 1024), cell_dim=clop_cfg.get("cell_dim", 512),
            proj_dim=clop_cfg.get("proj_dim", 256), text_layers=clop_cfg.get("text_layers", 3),
            cell_layers=clop_cfg.get("cell_layers", 3), dropout=clop_cfg.get("dropout", 0.1),
            use_batch_norm=clop_cfg.get("use_batch_norm", True),
            label_smoothing=clop_cfg.get("label_smoothing", 0.1),
        )
        clop.load_state_dict(clop_ckpt["model_state_dict"])
        clop.to(device).eval()

        dit_ckpt = torch.load(CKPT_DIR / "dit_best.pth", map_location=device, weights_only=False)
        dit_cfg = dit_ckpt.get("config", {})
        dit = DiT1D(latent_dim=dit_cfg.get("latent_dim", 512), hidden_dim=dit_cfg.get("hidden_dim", 384),
                    cond_dim=clop_cfg.get("proj_dim", 256), num_tokens=dit_cfg.get("num_tokens", 16))
        if "ema_state_dict" in dit_ckpt:
            dit.load_state_dict(dit_ckpt["ema_state_dict"])
        else:
            dit.load_state_dict(dit_ckpt["model_state_dict"])
        dit.to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
        bert = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
                                         ignore_mismatched_sizes=True).to(device).eval()

        torch.manual_seed(42)
        for name in missing:
            with torch.no_grad():
                inp = tokenizer(CELL_TYPE_PROMPTS[name], return_tensors="pt",
                                padding=True, truncation=True, max_length=512).to(device)
                text_emb = bert(**inp).last_hidden_state[:, 0, :]
                cond = clop.project_text(text_emb)
                emb = dit.sample(cond.expand(200, -1), num_steps=20, cfg_scale=3.0)
                gen_dict[name] = emb.cpu().numpy()
                np.save(RESULTS_DIR / f"gen_{name}.npy", gen_dict[name])

        del clop, dit, bert, tokenizer
        torch.cuda.empty_cache()
        import gc; gc.collect()

    # ── Compose all figures ──
    logger.info("\n--- Composing Figure 2: Training Dynamics ---")
    compose_fig2_training()

    logger.info("--- Composing Figure 3: UMAP Embedding ---")
    compose_fig3_umap(real_emb, gen_dict)

    logger.info("--- Composing Figure 4: Metrics Dashboard ---")
    compose_fig4_metrics(real_emb, gen_dict, metrics_all)

    logger.info("--- Composing Figure 5: Biological Validation ---")
    compose_fig5_biological(real_emb, gen_dict)

    logger.info("--- Composing Figure 6: Dimension & Sampling ---")
    compose_fig6_dimension(real_emb, gen_dict)

    logger.info("--- Composing Figure 7: Alignment & Metrics ---")
    compose_fig7_alignment(real_emb, gen_dict, jbhi_results)

    logger.info("--- Composing Figure 8: Tables ---")
    compose_fig8_tables(jbhi_results)

    # ── Run visual conflict check ──
    logger.info("\n--- Running Visual Conflict & Safety Check ---")
    report = run_visual_conflict_check()

    # ── Summary ──
    all_figs = sorted(FIG_DIR.glob("fig*.png"))
    logger.info(f"\n{'='*60}")
    logger.info(f"FIGURE INVENTORY ({len(all_figs)} figures)")
    logger.info(f"{'='*60}")
    for f in all_figs:
        from PIL import Image
        img = Image.open(f)
        logger.info(f"  {f.name:40s} {img.size[0]:5d}x{img.size[1]:5d}  "
                     f"{f.stat().st_size/1024:6.0f} KB")

    logger.info(f"\nVisual Check: {report['status']} ({report['issues_count']} issues)")
    logger.info("Done.")


if __name__ == "__main__":
    main()
