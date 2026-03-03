#!/usr/bin/env python3
"""
14_biological_evaluation.py — Proper Biological Evaluation

Core insight: Previous evaluations used OUT-OF-DISTRIBUTION text prompts that the
model never saw during training. This script evaluates using the ACTUAL training
conditions and biologically meaningful metrics.

Key question: Does the model generate cells that are biologically plausible?
- Do generated cells belong to the right cell type cluster?
- Does changing the condition change what cell type is generated?
- How does the model compare to simple baselines?

Evaluation approach:
1. Use IN-DISTRIBUTION conditions (actual training text embeddings)
2. KNN-based cell type classification  
3. Cross-group controllability (steering test)
4. Per-group centroid & diversity matching
5. Centered Fréchet Distance (removing global mean bias)
6. PCA visualization
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys, json, time, gc
from pathlib import Path
from collections import Counter, defaultdict

PROJECT = Path("/home/zeyufu/Desktop/CLOP-DiT")
sys.path.insert(0, str(PROJECT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = PROJECT / "data/cached_latents_v5.2"
CKPT = PROJECT / "models/checkpoints"
RESULTS = PROJECT / "results/v5_final"
FIGURES = PROJECT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def seed_all(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
#  Fréchet Distance
# ============================================================================

def frechet_distance(real, gen):
    """Compute FD between two sets of embeddings."""
    from scipy.linalg import sqrtm
    mu_r, mu_g = real.mean(0), gen.mean(0)
    sigma_r = np.cov(real, rowvar=False)
    sigma_g = np.cov(gen, rowvar=False)
    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fd)


# ============================================================================
#  Main Evaluation
# ============================================================================

def main():
    seed_all(42)
    from src.architecture.dit import DiT1D
    
    print("=" * 70)
    print("  BIOLOGICAL EVALUATION OF CLOP-DiT CELL GENERATION")
    print("=" * 70)
    
    # ── 1. Load Data ──────────────────────────────────────────────────
    print("\n[1] Loading data...")
    cell_emb = np.load(CACHE / "cell_embeddings.npy")      # (220304, 512)
    text_emb = np.load(CACHE / "text_embeddings.npy")       # (220304, 1024)
    proj_text = np.load(CACHE / "projected_text.npy")       # (220304, 256)
    sample_ids = np.load(CACHE / "sample_ids.npy")          # (220304,)
    
    N, cell_dim = cell_emb.shape
    print(f"    {N:,} cells, {cell_dim}-dim")
    
    # Group cells by unique text embedding
    _, text_labels = np.unique(
        np.round(text_emb[:, :50], 4), axis=0, return_inverse=True
    )
    n_groups = len(np.unique(text_labels))
    print(f"    {n_groups} unique text groups")
    
    # Build group info
    group_sizes = Counter(text_labels.tolist())
    group_indices = defaultdict(list)
    for i, g in enumerate(text_labels):
        group_indices[g].append(i)
    group_indices = {g: np.array(idx) for g, idx in group_indices.items()}
    
    # Select groups with >= 50 cells
    valid_groups = [g for g, s in group_sizes.items() if s >= 50]
    valid_groups.sort(key=lambda g: group_sizes[g], reverse=True)
    print(f"    {len(valid_groups)} groups with >= 50 cells")
    
    # Use top 100 for evaluation
    eval_groups = valid_groups[:100]
    
    # Precompute group info
    group_centroids = {}
    group_stds = {}
    group_cond = {}  # CLOP-projected condition for each group
    for g in eval_groups:
        idx = group_indices[g]
        group_centroids[g] = cell_emb[idx].mean(axis=0)
        group_stds[g] = cell_emb[idx].std(axis=0).mean()
        group_cond[g] = proj_text[idx[0]]  # Same for all cells in group
    
    global_mean = cell_emb.mean(axis=0)
    global_std = cell_emb.std(axis=0)
    
    # ── 2. Train KNN Classifier ──────────────────────────────────────
    print("\n[2] Training KNN classifier on real data...")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score
    
    # Center the data (remove global mean → focus on inter-group differences)
    cell_centered = cell_emb - global_mean
    
    # PCA to 50 dims for KNN efficiency
    pca = PCA(n_components=50, random_state=42)
    cell_pca = pca.fit_transform(cell_centered)
    print(f"    PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Only use cells from eval groups for cleaner evaluation
    eval_mask = np.isin(text_labels, eval_groups)
    eval_cell_pca = cell_pca[eval_mask]
    eval_labels = text_labels[eval_mask]
    n_eval = eval_mask.sum()
    print(f"    Eval cells: {n_eval:,} from {len(eval_groups)} groups")
    
    # Split 80/20 for KNN train/test
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_eval)
    split = int(n_eval * 0.8)
    knn_train_idx = perm[:split]
    knn_test_idx = perm[split:]
    
    knn = KNeighborsClassifier(n_neighbors=15, metric='cosine', n_jobs=-1)
    knn.fit(eval_cell_pca[knn_train_idx], eval_labels[knn_train_idx])
    
    # Test on held-out real data
    real_preds = knn.predict(eval_cell_pca[knn_test_idx])
    real_acc = accuracy_score(eval_labels[knn_test_idx], real_preds)
    
    # Also compute top-5 accuracy
    real_proba = knn.predict_proba(eval_cell_pca[knn_test_idx])
    knn_classes = knn.classes_
    real_top5 = 0
    for i, true_label in enumerate(eval_labels[knn_test_idx]):
        top5_classes = knn_classes[np.argsort(real_proba[i])[-5:]]
        if true_label in top5_classes:
            real_top5 += 1
    real_top5_acc = real_top5 / len(knn_test_idx)
    
    print(f"    KNN top-1 accuracy on real data: {real_acc:.4f}")
    print(f"    KNN top-5 accuracy on real data: {real_top5_acc:.4f}")
    print(f"    Random baseline: {1/len(eval_groups):.5f}")
    
    # ── 3. Load Original DiT ─────────────────────────────────────────
    print("\n[3] Loading Original DiT model...")
    dit = DiT1D(latent_dim=512, hidden_dim=384, cond_dim=256,
                num_blocks=8, num_heads=6, cond_drop_prob=0.15).to(DEVICE)
    ckpt = torch.load(CKPT / "dit_best.pth", map_location=DEVICE, weights_only=False)
    dit.load_state_dict(ckpt["model_state_dict"])
    dit.eval()
    print(f"    Loaded epoch {ckpt.get('epoch', '?')}")
    
    # ── 4. Generate Cells ────────────────────────────────────────────
    N_GEN = 200  # per group
    print(f"\n[4] Generating {N_GEN} cells for {len(eval_groups)} groups...")
    
    gen_cond = {}     # conditioned generation (CFG=3.0)
    gen_mild = {}     # mild conditioning (CFG=1.0)
    gen_uncond = {}   # unconditional (CFG=0.0)
    
    seed_all(42)
    for i, g in enumerate(eval_groups):
        cond = torch.tensor(group_cond[g], dtype=torch.float32, device=DEVICE)
        cond_batch = cond.unsqueeze(0).expand(N_GEN, -1)
        
        with torch.no_grad():
            gen_cond[g] = dit.sample(cond_batch, cfg_scale=3.0, num_steps=20).cpu().numpy()
            gen_mild[g] = dit.sample(cond_batch, cfg_scale=1.0, num_steps=20).cpu().numpy()
            gen_uncond[g] = dit.sample(cond_batch, cfg_scale=0.0, num_steps=20).cpu().numpy()
        
        if (i + 1) % 25 == 0:
            print(f"    Generated for {i+1}/{len(eval_groups)} groups")
    
    print(f"    Total: {N_GEN * len(eval_groups):,} cells per mode")
    
    # Also generate a Gaussian baseline (sample from data's N(mu, Sigma))
    print("    Generating Gaussian baseline...")
    cov_approx = np.cov(cell_emb[rng.choice(N, 10000, replace=False)], rowvar=False)
    L = np.linalg.cholesky(cov_approx + np.eye(cell_dim) * 1e-6)
    gauss_all = (rng.standard_normal((N_GEN * len(eval_groups), cell_dim)) @ L.T) + global_mean
    gen_gauss = {}
    for i, g in enumerate(eval_groups):
        gen_gauss[g] = gauss_all[i * N_GEN:(i + 1) * N_GEN]
    
    # ── 5. Evaluate ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    
    results = {}
    
    for mode_name, gen_data in [("CFG=3.0", gen_cond), ("CFG=1.0", gen_mild),
                                 ("CFG=0.0", gen_uncond), ("Gaussian", gen_gauss)]:
        print(f"\n  ── Mode: {mode_name} ──")
        
        # Stack all generated cells
        all_gen = np.vstack([gen_data[g] for g in eval_groups])
        all_gen_labels = np.repeat(eval_groups, N_GEN)
        
        # ── 5a. KNN Classification ──
        gen_centered = all_gen - global_mean
        gen_pca = pca.transform(gen_centered)
        
        gen_preds = knn.predict(gen_pca)
        gen_acc = accuracy_score(all_gen_labels, gen_preds)
        
        gen_proba = knn.predict_proba(gen_pca)
        gen_top5 = 0
        for j, true_label in enumerate(all_gen_labels):
            top5_classes = knn_classes[np.argsort(gen_proba[j])[-5:]]
            if true_label in top5_classes:
                gen_top5 += 1
        gen_top5_acc = gen_top5 / len(all_gen_labels)
        
        print(f"    KNN top-1: {gen_acc:.4f}  top-5: {gen_top5_acc:.4f}")
        
        # ── 5b. Centroid Cosine ──
        centroid_cos = []
        for g in eval_groups:
            real_c = group_centroids[g]
            gen_c = gen_data[g].mean(axis=0)
            cos = np.dot(real_c, gen_c) / (np.linalg.norm(real_c) * np.linalg.norm(gen_c) + 1e-8)
            centroid_cos.append(cos)
        
        # Centered centroid cosine (more discriminative)
        centroid_cos_centered = []
        for g in eval_groups:
            real_c = group_centroids[g] - global_mean
            gen_c = gen_data[g].mean(axis=0) - global_mean
            cos = np.dot(real_c, gen_c) / (np.linalg.norm(real_c) * np.linalg.norm(gen_c) + 1e-8)
            centroid_cos_centered.append(cos)
        
        print(f"    Raw centroid cos:      {np.mean(centroid_cos):.5f}")
        print(f"    Centered centroid cos: {np.mean(centroid_cos_centered):.5f} ± {np.std(centroid_cos_centered):.5f}")
        
        # ── 5c. Cross-Group Steering ──
        n_pairs = 1000
        correct = 0
        correct_centered = 0
        rng2 = np.random.default_rng(123)
        for _ in range(n_pairs):
            a, b = rng2.choice(eval_groups, size=2, replace=False)
            gen_a_c = gen_data[a].mean(0)
            real_a = group_centroids[a]
            real_b = group_centroids[b]
            
            # Raw space
            d_aa = np.dot(gen_a_c, real_a) / (np.linalg.norm(gen_a_c) * np.linalg.norm(real_a))
            d_ab = np.dot(gen_a_c, real_b) / (np.linalg.norm(gen_a_c) * np.linalg.norm(real_b))
            if d_aa > d_ab:
                correct += 1
            
            # Centered
            gen_a_cc = gen_a_c - global_mean
            real_a_c = real_a - global_mean
            real_b_c = real_b - global_mean
            d_aa_c = np.dot(gen_a_cc, real_a_c) / (np.linalg.norm(gen_a_cc) * np.linalg.norm(real_a_c) + 1e-8)
            d_ab_c = np.dot(gen_a_cc, real_b_c) / (np.linalg.norm(gen_a_cc) * np.linalg.norm(real_b_c) + 1e-8)
            if d_aa_c > d_ab_c:
                correct_centered += 1
        
        steer_raw = correct / n_pairs
        steer_centered = correct_centered / n_pairs
        print(f"    Steering (raw):      {steer_raw:.4f}")
        print(f"    Steering (centered): {steer_centered:.4f}  (random=0.500)")
        
        # ── 5d. Diversity ──
        gen_group_stds = [gen_data[g].std(axis=0).mean() for g in eval_groups]
        real_group_stds = [group_stds[g] for g in eval_groups]
        diversity_ratio = np.mean(gen_group_stds) / np.mean(real_group_stds)
        print(f"    Diversity ratio:     {diversity_ratio:.4f}  (real std={np.mean(real_group_stds):.5f})")
        
        # ── 5e. FD (overall) ──
        real_sub = cell_emb[rng.choice(N, min(20000, N), replace=False)]
        fd = frechet_distance(real_sub, all_gen)
        print(f"    Overall FD:          {fd:.3f}")
        
        # ── 5f. Centered FD ──
        real_sub_c = real_sub - global_mean
        all_gen_c = all_gen - global_mean
        fd_centered = frechet_distance(real_sub_c, all_gen_c)
        print(f"    Centered FD:         {fd_centered:.3f}")
        
        # ── 5g. Per-group FD (sampled) ──
        per_group_fds = []
        for g in eval_groups[:20]:  # Top 20 groups
            idx = group_indices[g]
            if len(idx) >= 50 and len(gen_data[g]) >= 50:
                pgfd = frechet_distance(cell_emb[idx], gen_data[g])
                per_group_fds.append(pgfd)
        if per_group_fds:
            print(f"    Per-group FD (top20): {np.mean(per_group_fds):.3f} ± {np.std(per_group_fds):.3f}")
        
        results[mode_name] = {
            "knn_top1": float(gen_acc),
            "knn_top5": float(gen_top5_acc),
            "centroid_cos_raw": float(np.mean(centroid_cos)),
            "centroid_cos_centered": float(np.mean(centroid_cos_centered)),
            "steering_raw": float(steer_raw),
            "steering_centered": float(steer_centered),
            "diversity_ratio": float(diversity_ratio),
            "overall_fd": float(fd),
            "centered_fd": float(fd_centered),
        }
    
    # Add real data baseline
    results["Real_holdout"] = {
        "knn_top1": float(real_acc),
        "knn_top5": float(real_top5_acc),
    }
    
    # ── 6. Summary Table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    
    header = f"{'Method':<14} {'KNN-1↑':>8} {'KNN-5↑':>8} {'CtrCos↑':>8} {'Steer↑':>8} {'DivR':>8} {'FD↓':>8} {'cFD↓':>8}"
    print(header)
    print("-" * len(header))
    
    for mode in ["Real_holdout", "CFG=3.0", "CFG=1.0", "CFG=0.0", "Gaussian"]:
        r = results[mode]
        if mode == "Real_holdout":
            print(f"{'Real (test)':<14} {r['knn_top1']:>8.4f} {r['knn_top5']:>8.4f} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8}")
        else:
            print(f"{mode:<14} {r['knn_top1']:>8.4f} {r['knn_top5']:>8.4f} "
                  f"{r['centroid_cos_centered']:>8.4f} {r['steering_centered']:>8.4f} "
                  f"{r['diversity_ratio']:>8.4f} {r['overall_fd']:>8.2f} {r['centered_fd']:>8.3f}")
    
    print(f"\n  Random KNN baseline: {1/len(eval_groups):.5f}")
    
    # ── 7. Key Diagnostic: What does the model ACTUALLY generate? ────
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC: GENERATION STRUCTURE ANALYSIS")
    print("=" * 70)
    
    for mode_name, gen_data in [("CFG=3.0", gen_cond), ("CFG=0.0", gen_uncond)]:
        all_gen = np.vstack([gen_data[g] for g in eval_groups])
        
        gen_mean = all_gen.mean(0)
        gen_std_per_dim = all_gen.std(0)
        
        # How similar is generated mean to real mean?
        mean_cos = np.dot(gen_mean, global_mean) / (np.linalg.norm(gen_mean) * np.linalg.norm(global_mean))
        
        # Per-group mean variation (how much do group means differ?)
        group_means = np.array([gen_data[g].mean(0) for g in eval_groups])
        gm_pairwise = group_means @ group_means.T
        gm_norms = np.linalg.norm(group_means, axis=1, keepdims=True)
        gm_cos = gm_pairwise / (gm_norms @ gm_norms.T)
        gm_offdiag = gm_cos[np.triu_indices(len(eval_groups), k=1)]
        
        print(f"\n  [{mode_name}]")
        print(f"    Gen mean norm: {np.linalg.norm(gen_mean):.3f}  (real: {np.linalg.norm(global_mean):.3f})")
        print(f"    Gen-Real mean cosine: {mean_cos:.6f}")
        print(f"    Gen per-dim std (mean): {gen_std_per_dim.mean():.5f}  (real: {global_std.mean():.5f})")
        print(f"    Gen inter-group cosine: {gm_offdiag.mean():.6f}  (real: 0.9939)")
        print(f"    Gen inter-group cos min: {gm_offdiag.min():.6f}")
    
    # ── 8. PCA Visualization ─────────────────────────────────────────
    print("\n[8] Creating PCA visualization...")
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Select top-10 groups for visualization
    vis_groups = eval_groups[:10]
    vis_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Subsample real cells from these groups
    vis_real = []
    vis_real_labels = []
    for i, g in enumerate(vis_groups):
        idx = group_indices[g]
        sub = rng.choice(idx, min(200, len(idx)), replace=False)
        vis_real.append(cell_emb[sub])
        vis_real_labels.extend([i] * len(sub))
    vis_real = np.vstack(vis_real)
    vis_real_labels = np.array(vis_real_labels)
    
    # Generated cells
    vis_gen = np.vstack([gen_cond[g] for g in vis_groups])
    vis_gen_labels = np.repeat(np.arange(10), N_GEN)
    
    # Combined PCA (centered)
    combined = np.vstack([vis_real - global_mean, vis_gen - global_mean])
    pca_2d = PCA(n_components=2, random_state=42)
    combined_2d = pca_2d.fit_transform(combined)
    
    n_real = len(vis_real)
    real_2d = combined_2d[:n_real]
    gen_2d = combined_2d[n_real:]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Real cells
    for i in range(10):
        mask = vis_real_labels == i
        axes[0].scatter(real_2d[mask, 0], real_2d[mask, 1], c=[vis_colors[i]],
                       alpha=0.4, s=8, label=f"G{vis_groups[i]}")
    axes[0].set_title("Real Cells (top-10 groups)")
    axes[0].legend(fontsize=6, ncol=2)
    
    # Plot 2: Generated cells (CFG=3.0)
    for i in range(10):
        mask = vis_gen_labels == i
        axes[1].scatter(gen_2d[mask, 0], gen_2d[mask, 1], c=[vis_colors[i]],
                       alpha=0.4, s=8, label=f"G{vis_groups[i]}")
    axes[1].set_title("Generated Cells (CFG=3.0)")
    axes[1].legend(fontsize=6, ncol=2)
    
    # Plot 3: Overlay
    axes[2].scatter(real_2d[:, 0], real_2d[:, 1], c='gray', alpha=0.2, s=6, label="Real")
    for i in range(10):
        mask = vis_gen_labels == i
        axes[2].scatter(gen_2d[mask, 0], gen_2d[mask, 1], c=[vis_colors[i]],
                       alpha=0.5, s=10, label=f"Gen G{vis_groups[i]}")
    axes[2].set_title("Overlay (gray=Real)")
    axes[2].legend(fontsize=5, ncol=2)
    
    plt.tight_layout()
    plt.savefig(FIGURES / "biological_eval_pca.png", dpi=150, bbox_inches='tight')
    print(f"    Saved: {FIGURES / 'biological_eval_pca.png'}")
    
    # ── 9. CRITICAL TEST: Centered-Space Classification ──────────────
    # The real test: in the CENTERED space, can a classifier distinguish
    # cell types? And do generated cells match?
    print("\n" + "=" * 70)
    print("  CRITICAL: CENTERED-SPACE ANALYSIS")
    print("=" * 70)
    
    # How separable are the real cell types in centered space?
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Use PCA-50 centered features
    scaler = StandardScaler()
    eval_pca_scaled = scaler.fit_transform(eval_cell_pca)
    
    # Quick linear classifier on real data
    lr = LogisticRegression(
        max_iter=500, C=1.0, solver='saga', n_jobs=-1,
        random_state=42
    )
    lr.fit(eval_pca_scaled[knn_train_idx], eval_labels[knn_train_idx])
    lr_real_acc = lr.score(eval_pca_scaled[knn_test_idx], eval_labels[knn_test_idx])
    print(f"    Linear classifier on real data: {lr_real_acc:.4f}")
    
    # Classify generated cells with linear classifier
    for mode_name, gen_data in [("CFG=3.0", gen_cond), ("CFG=1.0", gen_mild),
                                 ("CFG=0.0", gen_uncond), ("Gaussian", gen_gauss)]:
        all_gen = np.vstack([gen_data[g] for g in eval_groups])
        all_gen_labels = np.repeat(eval_groups, N_GEN)
        gen_c = all_gen - global_mean
        gen_pca_t = pca.transform(gen_c)
        gen_pca_s = scaler.transform(gen_pca_t)
        lr_gen_acc = lr.score(gen_pca_s, all_gen_labels)
        print(f"    Linear classifier on {mode_name}: {lr_gen_acc:.4f}")
    
    # ── 10. CONCLUSION ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    
    cfg3 = results["CFG=3.0"]
    cfg0 = results["CFG=0.0"]
    gauss = results["Gaussian"]
    
    print(f"""
    DATA GEOMETRY:
      Real cell groups differ by ~0.6% in cosine similarity (0.994 vs 1.0).
      The inter-group signal lives in a tiny subspace after centering.
    
    MODEL ASSESSMENT:
      KNN accuracy: CFG=3.0 → {cfg3['knn_top1']:.4f} vs CFG=0.0 → {cfg0['knn_top1']:.4f}
      Steering:     CFG=3.0 → {cfg3['steering_centered']:.4f} vs random 0.500
      
      If KNN_cond >> KNN_uncond: conditioning WORKS (model learned cell types)
      If KNN_cond ≈ KNN_uncond: conditioning has no effect (model ignores cond)
      
      Diversity ratio: CFG=3.0 → {cfg3['diversity_ratio']:.4f}
      If << 1.0: mode collapse. If ≈ 1.0: appropriate diversity.
    
    COMPARISON TO GAUSSIAN BASELINE:
      Gaussian KNN: {gauss['knn_top1']:.4f}
      If model >> Gaussian: model learned meaningful structure beyond N(mu,Sigma)
      If model ≈ Gaussian: model only learned mean+covariance (trivial)
    """)
    
    # ── 11. Save Results ─────────────────────────────────────────────
    save_path = RESULTS / "biological_evaluation.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Results saved to {save_path}")
    
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
