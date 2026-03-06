# full_pipeline_figures.py — Publication figures for 10_full_pipeline (5 multi-panel figures).
"""Generate training dynamics, embedding space, metrics dashboard, biological validation, and sampling figures."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.utils.paths import FIG_DIR, CHECKPOINT_DIR
from src.evaluation.run_metrics import load_models_for_pipeline

logger = logging.getLogger(__name__)

CD8_PROMPT = (
    "CD8+ cytotoxic T lymphocytes from human lung adenocarcinoma tumor microenvironment "
    "expressing cytotoxic effector molecules"
)


def _encode_text(text: str, clop, device: str = "cuda"):
    """Encode text to CLOP condition vector (for figure generation)."""
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
    )
    model = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        ignore_mismatched_sizes=True,
    ).to(device).eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        text_emb = outputs.last_hidden_state[:, 0, :]
        cond = clop.project_text(text_emb)
    return cond.squeeze(0)

def generate_all_figures(metrics, gen_dict, real_emb, projected_text, sample_ids):
    """Generate 5 publication-quality multi-panel figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    from scipy import stats
    from sklearn.decomposition import PCA

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Publication style
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
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    COLORS = {
        "real": "#2166ac",
        "generated": "#d6604d",
        "CD8_T": "#e41a1c",
        "Macrophage": "#377eb8",
        "Epithelial_tumor": "#4daf4a",
        "NK_cell": "#984ea3",
        "Fibroblast": "#ff7f00",
        "B_cell": "#a65628",
    }

    # ── Load training histories ──
    clop_hist = json.load(open(CHECKPOINT_DIR / "clop_history.json"))
    dit_hist = json.load(open(CHECKPOINT_DIR / "dit_history.json"))

    # ──────────────────────────────────────────────────────────────────────
    # FIGURE 1: Training Dynamics (2×2 panel)
    # a) CLOP loss curves  b) CLOP accuracy + temperature
    # c) DiT loss curves   d) DiT velocity cosine similarity
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Generating Figure 1: Training Dynamics...")
    fig1, axes1 = plt.subplots(2, 2, figsize=(8, 6.5))

    # 1a: CLOP loss
    ax = axes1[0, 0]
    epochs_c = range(1, len(clop_hist["train_loss"]) + 1)
    ax.plot(epochs_c, clop_hist["train_loss"], color="#2166ac", lw=1.5, label="Train")
    ax.plot(epochs_c, clop_hist["val_loss"], color="#d6604d", lw=1.5, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) CLOP Contrastive Loss", fontweight="bold")
    ax.legend(frameon=False)
    ax.set_xlim(1, len(clop_hist["train_loss"]))

    # 1b: CLOP accuracy + temperature
    ax = axes1[0, 1]
    ax.plot(epochs_c, clop_hist["val_acc"], color="#4daf4a", lw=1.5, label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy", color="#4daf4a")
    ax.tick_params(axis="y", labelcolor="#4daf4a")
    ax.set_title("(b) CLOP Alignment Quality", fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(epochs_c, clop_hist["temperature"], color="#984ea3", lw=1.0, ls="--",
             label="Temperature")
    ax2.set_ylabel("Temperature", color="#984ea3")
    ax2.tick_params(axis="y", labelcolor="#984ea3")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center right")

    # 1c: DiT loss
    ax = axes1[1, 0]
    epochs_d = range(1, len(dit_hist["train_loss"]) + 1)
    ax.plot(epochs_d, dit_hist["train_loss"], color="#2166ac", lw=1.5, label="Train")
    ax.plot(epochs_d, dit_hist["val_loss"], color="#d6604d", lw=1.5, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("(c) DiT Flow Matching Loss", fontweight="bold")
    ax.legend(frameon=False)
    ax.set_xlim(1, len(dit_hist["train_loss"]))

    # 1d: DiT cosine similarity
    ax = axes1[1, 1]
    if "val_cosine_sim" in dit_hist:
        ax.plot(epochs_d, dit_hist["val_cosine_sim"], color="#ff7f00", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("(d) Velocity Prediction Quality", fontweight="bold")
    ax.set_xlim(1, len(dit_hist["train_loss"]))
    ax.set_ylim(0, 1)

    fig1.suptitle("Figure 1: CLOP-DiT Training Dynamics", fontsize=13, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(FIG_DIR / "fig1_training_dynamics.pdf", dpi=300, bbox_inches="tight")
    fig1.savefig(FIG_DIR / "fig1_training_dynamics.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    logger.info("  Figure 1 saved.")

    # ──────────────────────────────────────────────────────────────────────
    # FIGURE 2: Embedding Space Analysis (1×3 panel)
    # a) PCA: Real vs Generated  b) Per-cell-type PCA  c) Norm distributions
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Generating Figure 2: Embedding Space Analysis...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4.5))

    # Subsample real for speed
    n_sub = min(2000, len(real_emb))
    idx_sub = np.random.choice(len(real_emb), n_sub, replace=False)
    real_sub = real_emb[idx_sub]

    all_gen = np.concatenate(list(gen_dict.values()), axis=0)
    combined = np.vstack([real_sub, all_gen])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)
    n_real = len(real_sub)

    # 2a: Real vs Generated
    ax = axes2[0]
    ax.scatter(coords[:n_real, 0], coords[:n_real, 1], s=3, alpha=0.3,
               color=COLORS["real"], label="Real", rasterized=True)
    ax.scatter(coords[n_real:, 0], coords[n_real:, 1], s=5, alpha=0.5,
               color=COLORS["generated"], label="Generated", rasterized=True)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("(a) Real vs Generated", fontweight="bold")
    ax.legend(frameon=False, markerscale=3)

    # 2b: Per-cell-type
    ax = axes2[1]
    ax.scatter(coords[:n_real, 0], coords[:n_real, 1], s=2, alpha=0.15,
               color="#cccccc", label="Real (all)", rasterized=True)
    offset = n_real
    for name in gen_dict:
        n = len(gen_dict[name])
        ax.scatter(coords[offset:offset+n, 0], coords[offset:offset+n, 1],
                   s=8, alpha=0.7, color=COLORS.get(name, "#333333"),
                   label=name, rasterized=True)
        offset += n
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("(b) Cell-Type Specific Generation", fontweight="bold")
    ax.legend(frameon=False, markerscale=2, fontsize=7, ncol=2)

    # 2c: Norm distributions
    ax = axes2[2]
    real_norms = np.linalg.norm(real_sub, axis=1)
    gen_norms = np.linalg.norm(all_gen, axis=1)
    ax.hist(real_norms, bins=50, alpha=0.6, color=COLORS["real"],
            label=f"Real (μ={real_norms.mean():.1f})", density=True)
    ax.hist(gen_norms, bins=50, alpha=0.6, color=COLORS["generated"],
            label=f"Gen (μ={gen_norms.mean():.1f})", density=True)
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Density")
    ax.set_title("(c) Embedding Norm Distribution", fontweight="bold")
    ax.legend(frameon=False)

    fig2.suptitle("Figure 2: Embedding Space Analysis", fontsize=13, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(FIG_DIR / "fig2_embedding_space.pdf", dpi=300, bbox_inches="tight")
    fig2.savefig(FIG_DIR / "fig2_embedding_space.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    logger.info("  Figure 2 saved.")

    # ──────────────────────────────────────────────────────────────────────
    # FIGURE 3: Quantitative Metrics Dashboard (2×3 panel)
    # a) FD by cell type  b) MMD by cell type  c) Coverage & Density
    # d) KL divergence    e) Cosine similarity  f) Dim-wise correlation
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Generating Figure 3: Metrics Dashboard...")
    fig3, axes3 = plt.subplots(2, 3, figsize=(14, 8))

    emb_metrics = metrics.get("embedding_quality", {})
    ct_names = list(gen_dict.keys())

    # 3a: FD by cell type
    ax = axes3[0, 0]
    fd_values = [emb_metrics.get(f"{ct}/frechet_distance", 0) for ct in ct_names]
    bars = ax.bar(range(len(ct_names)), fd_values, color=[COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("Fréchet Distance ↓")
    ax.set_title("(a) Fréchet Distance", fontweight="bold")

    # 3b: MMD by cell type
    ax = axes3[0, 1]
    mmd_values = [emb_metrics.get(f"{ct}/mmd_rbf", 0) for ct in ct_names]
    ax.bar(range(len(ct_names)), mmd_values, color=[COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("MMD (RBF) ↓")
    ax.set_title("(b) Maximum Mean Discrepancy", fontweight="bold")

    # 3c: Coverage & Density
    ax = axes3[0, 2]
    cov_values = [emb_metrics.get(f"{ct}/coverage", 0) for ct in ct_names]
    den_values = [emb_metrics.get(f"{ct}/density", 0) for ct in ct_names]
    x = np.arange(len(ct_names))
    w = 0.35
    ax.bar(x - w/2, cov_values, w, label="Coverage ↑", color="#2166ac", alpha=0.8)
    ax.bar(x + w/2, den_values, w, label="Density ↑", color="#d6604d", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title("(c) Coverage & Density", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)

    # 3d: KL divergence
    ax = axes3[1, 0]
    kl_values = [emb_metrics.get(f"{ct}/mean_kl", 0) for ct in ct_names]
    ax.bar(range(len(ct_names)), kl_values, color=[COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("Mean KL ↓")
    ax.set_title("(d) Per-Dimension KL Divergence", fontweight="bold")

    # 3e: Overall metrics summary
    ax = axes3[1, 1]
    overall_keys = ["overall/frechet_distance", "overall/mmd_rbf",
                    "overall/coverage", "overall/density", "overall/mean_kl"]
    labels = ["FD", "MMD", "Coverage", "Density", "KL"]
    values = [emb_metrics.get(k, 0) for k in overall_keys]
    # Normalize for radar-like viz
    ax.barh(range(len(labels)), values, color="#377eb8", alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Score")
    ax.set_title("(e) Overall Metrics Summary", fontweight="bold")

    # 3f: Prompt robustness
    ax = axes3[1, 2]
    rob = metrics.get("prompt_robustness", {})
    if rob:
        rob_cts = list(rob.keys())
        rob_sims = [rob[ct]["mean_cross_sim"] for ct in rob_cts]
        rob_stds = [rob[ct]["std_cross_sim"] for ct in rob_cts]
        ax.bar(range(len(rob_cts)), rob_sims, yerr=rob_stds,
               color=[COLORS.get(ct, "#999") for ct in rob_cts],
               capsize=3, alpha=0.8)
        ax.set_xticks(range(len(rob_cts)))
        ax.set_xticklabels([ct.replace("_", "\n") for ct in rob_cts], fontsize=7)
        ax.set_ylabel("Cross-Variant Cosine Sim ↑")
        ax.set_ylim(0, 1)
    ax.set_title("(f) Prompt Robustness", fontweight="bold")

    fig3.suptitle("Figure 3: Generation Quality Metrics", fontsize=13, fontweight="bold")
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig(FIG_DIR / "fig3_metrics_dashboard.pdf", dpi=300, bbox_inches="tight")
    fig3.savefig(FIG_DIR / "fig3_metrics_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)
    logger.info("  Figure 3 saved.")

    # ──────────────────────────────────────────────────────────────────────
    # FIGURE 4: Biological Validation (2×2 panel)
    # a) Per-cell-type embedding separation (UMAP or PCA)
    # b) Inter-type cosine distance heatmap
    # c) Diversity: intra-type variance
    # d) Conditioning fidelity: condition→embedding correlation
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Generating Figure 4: Biological Validation...")
    fig4, axes4 = plt.subplots(2, 2, figsize=(10, 9))

    # 4a: Cell-type separation (PCA with generated only)
    ax = axes4[0, 0]
    gen_labels = []
    gen_all_list = []
    for name, emb in gen_dict.items():
        gen_all_list.append(emb)
        gen_labels.extend([name] * len(emb))
    gen_all_arr = np.vstack(gen_all_list)
    pca_gen = PCA(n_components=2)
    coords_gen = pca_gen.fit_transform(gen_all_arr)

    for name in gen_dict:
        mask = np.array(gen_labels) == name
        ax.scatter(coords_gen[mask, 0], coords_gen[mask, 1], s=10, alpha=0.6,
                   color=COLORS.get(name, "#333"), label=name, rasterized=True)
    ax.set_xlabel(f"PC1 ({pca_gen.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_gen.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("(a) Generated Cell-Type Separation", fontweight="bold")
    ax.legend(frameon=False, markerscale=2, fontsize=7, ncol=2)

    # 4b: Inter-type cosine distance heatmap
    ax = axes4[0, 1]
    type_means = {}
    for name, emb in gen_dict.items():
        type_means[name] = emb.mean(axis=0)
    names = list(type_means.keys())
    n_types = len(names)
    sim_matrix = np.zeros((n_types, n_types))
    for i in range(n_types):
        for j in range(n_types):
            v1, v2 = type_means[names[i]], type_means[names[j]]
            sim_matrix[i, j] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(n_types))
    ax.set_yticks(range(n_types))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    for i in range(n_types):
        for j in range(n_types):
            ax.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine Sim")
    ax.set_title("(b) Inter-Type Similarity", fontweight="bold")

    # 4c: Intra-type diversity (variance)
    ax = axes4[1, 0]
    intra_vars = []
    for name in gen_dict:
        v = np.var(gen_dict[name], axis=0).mean()
        intra_vars.append(v)
    ax.bar(range(len(ct_names)), intra_vars, color=[COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("Mean Variance ↑")
    ax.set_title("(c) Intra-Type Diversity", fontweight="bold")

    # 4d: Conditioning fidelity (condition → nearest real cosine)
    ax = axes4[1, 1]
    fidelity_scores = []
    for name, emb in gen_dict.items():
        gen_mean = emb.mean(axis=0)
        gen_mean_n = gen_mean / (np.linalg.norm(gen_mean) + 1e-8)
        # Find nearest real neighbors
        real_norms_mat = real_emb / (np.linalg.norm(real_emb, axis=1, keepdims=True) + 1e-8)
        sims_to_real = real_norms_mat @ gen_mean_n
        top_k_sim = np.sort(sims_to_real)[-50:].mean()
        fidelity_scores.append(top_k_sim)

    ax.bar(range(len(ct_names)), fidelity_scores, color=[COLORS.get(ct, "#999") for ct in ct_names])
    ax.set_xticks(range(len(ct_names)))
    ax.set_xticklabels([ct.replace("_", "\n") for ct in ct_names], fontsize=7)
    ax.set_ylabel("Top-50 Real Cosine Sim ↑")
    ax.set_title("(d) Conditioning Fidelity", fontweight="bold")

    fig4.suptitle("Figure 4: Biological Validation", fontsize=13, fontweight="bold")
    fig4.tight_layout(rect=[0, 0, 1, 0.96])
    fig4.savefig(FIG_DIR / "fig4_biological_validation.pdf", dpi=300, bbox_inches="tight")
    fig4.savefig(FIG_DIR / "fig4_biological_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)
    logger.info("  Figure 4 saved.")

    # ──────────────────────────────────────────────────────────────────────
    # FIGURE 5: Dimension-Level Analysis & ODE Integration (2×2 panel)
    # a) Per-dim distribution comparison (violin)
    # b) Dim-wise correlation real vs gen
    # c) Sampling trajectory visualization
    # d) CFG scale ablation 
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Generating Figure 5: Dimension & Sampling Analysis...")
    fig5, axes5 = plt.subplots(2, 2, figsize=(9, 7.5))

    # 5a: Per-dimension distribution (first 20 dims)
    ax = axes5[0, 0]
    n_dims_show = 16
    real_sub_small = real_sub[:500]
    gen_sub_small = all_gen[:500]
    positions = []
    data_real, data_gen = [], []
    for d in range(n_dims_show):
        data_real.append(real_sub_small[:, d])
        data_gen.append(gen_sub_small[:, d])
    bp_r = ax.boxplot(data_real, positions=np.arange(n_dims_show)*3,
                      widths=0.8, patch_artist=True, showfliers=False)
    bp_g = ax.boxplot(data_gen, positions=np.arange(n_dims_show)*3 + 1,
                      widths=0.8, patch_artist=True, showfliers=False)
    for patch in bp_r["boxes"]:
        patch.set_facecolor(COLORS["real"])
        patch.set_alpha(0.6)
    for patch in bp_g["boxes"]:
        patch.set_facecolor(COLORS["generated"])
        patch.set_alpha(0.6)
    ax.set_xticks(np.arange(n_dims_show)*3 + 0.5)
    ax.set_xticklabels([str(i) for i in range(n_dims_show)], fontsize=7)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Value")
    ax.set_title("(a) Per-Dimension Distribution", fontweight="bold")
    ax.legend([bp_r["boxes"][0], bp_g["boxes"][0]], ["Real", "Generated"], frameon=False)

    # 5b: Dim-wise mean correlation
    ax = axes5[0, 1]
    real_dim_mean = real_sub.mean(axis=0)
    gen_dim_mean = all_gen.mean(axis=0)
    r_corr, p_val = stats.pearsonr(real_dim_mean, gen_dim_mean)
    ax.scatter(real_dim_mean, gen_dim_mean, s=5, alpha=0.5, color="#377eb8")
    lims = [min(real_dim_mean.min(), gen_dim_mean.min()),
            max(real_dim_mean.max(), gen_dim_mean.max())]
    ax.plot(lims, lims, "r--", alpha=0.5, lw=1)
    ax.set_xlabel("Real Mean")
    ax.set_ylabel("Generated Mean")
    ax.set_title("(b) Dimension-wise Correlation", fontweight="bold")
    ax.text(0.05, 0.95, f"r = {r_corr:.4f}\np = {p_val:.2e}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # 5c: Sampling trajectory visualization
    ax = axes5[1, 0]
    # Show how z evolves from noise to data for a single example
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        clop_m, dit_m, _, _ = load_models_for_pipeline(
            CHECKPOINT_DIR / "clop_best.pth", CHECKPOINT_DIR / "dit_best.pth", device
        )
        cond_ex = _encode_text(CD8_PROMPT, clop_m, device)
        cond_batch = cond_ex.expand(1, -1)
        z = torch.randn(1, 512, device=device)
        trajectory = [z.cpu().numpy().flatten()]
        n_show_steps = 20
        dt = 1.0 / n_show_steps
        for i in range(n_show_steps):
            t_val = i / n_show_steps
            t = torch.full((1,), t_val, device=device)
            v = dit_m.forward_with_cfg(z, t, cond_batch, cfg_scale=3.0)
            z = z + v * dt
            trajectory.append(z.cpu().numpy().flatten())
        del clop_m, dit_m
        torch.cuda.empty_cache()

        traj = np.array(trajectory)
        traj_pca = PCA(n_components=2).fit_transform(traj)
        colors_traj = plt.cm.viridis(np.linspace(0, 1, len(traj_pca)))
        for i in range(len(traj_pca)-1):
            ax.annotate("", xy=traj_pca[i+1], xytext=traj_pca[i],
                        arrowprops=dict(arrowstyle="->", color=colors_traj[i], lw=1.5))
        ax.scatter(traj_pca[0, 0], traj_pca[0, 1], s=50, marker="o",
                   color="blue", zorder=5, label="z₀ (noise)")
        ax.scatter(traj_pca[-1, 0], traj_pca[-1, 1], s=50, marker="*",
                   color="red", zorder=5, label="z₁ (cell)")
        ax.legend(frameon=False)
    except Exception as e:
        ax.text(0.5, 0.5, f"Trajectory unavailable\n{e}", transform=ax.transAxes,
                ha="center", va="center")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("(c) ODE Sampling Trajectory", fontweight="bold")

    # 5d: CFG scale ablation
    ax = axes5[1, 1]
    cfg_scales = [1.0, 2.0, 3.0, 5.0, 7.0]
    cfg_fds = []
    try:
        clop_m, dit_m, _, _ = load_models_for_pipeline(
            CHECKPOINT_DIR / "clop_best.pth", CHECKPOINT_DIR / "dit_best.pth", device
        )
        cond_ex = _encode_text(CD8_PROMPT, clop_m, device)
        for cfg in cfg_scales:
            cond_batch = cond_ex.expand(100, -1)
            gen_cfg = dit_m.sample(cond_batch, num_steps=20, cfg_scale=cfg)
            gen_np = gen_cfg.cpu().numpy()
            # Compute FD against real
            from src.evaluation.metrics import GenerationMetrics
            fd = GenerationMetrics.frechet_distance(real_emb[:100], gen_np)
            cfg_fds.append(fd)
        del clop_m, dit_m
        torch.cuda.empty_cache()
        ax.plot(cfg_scales, cfg_fds, "o-", color="#377eb8", lw=2, markersize=6)
        ax.axvline(x=3.0, color="red", ls="--", alpha=0.5, label="default (3.0)")
        ax.legend(frameon=False)
    except Exception as e:
        ax.text(0.5, 0.5, f"CFG ablation unavailable\n{e}", transform=ax.transAxes,
                ha="center", va="center")
    ax.set_xlabel("CFG Scale")
    ax.set_ylabel("Fréchet Distance ↓")
    ax.set_title("(d) CFG Scale Ablation", fontweight="bold")

    fig5.suptitle("Figure 5: Dimension & Sampling Analysis", fontsize=13, fontweight="bold")
    fig5.tight_layout(rect=[0, 0, 1, 0.96])
    fig5.savefig(FIG_DIR / "fig5_dimension_sampling.pdf", dpi=300, bbox_inches="tight")
    fig5.savefig(FIG_DIR / "fig5_dimension_sampling.png", dpi=300, bbox_inches="tight")
    plt.close(fig5)
    logger.info("  Figure 5 saved.")

    logger.info(f"All 5 figures saved to {FIG_DIR}")
    return True

