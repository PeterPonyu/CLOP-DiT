#!/usr/bin/env python3
"""
14b_cfg_sweep.py — Find optimal CFG scale and ODE steps

The initial evaluation showed conditioning WORKS (35% KNN vs 1% random).
Now sweep CFG scale (0.5 to 5.0) and ODE steps (10, 20, 50, 100) to find
the best trade-off between accuracy and diversity.
"""

import numpy as np
import torch
import sys, json
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

PROJECT = Path("/home/zeyufu/Desktop/CLOP-DiT")
sys.path.insert(0, str(PROJECT))
DEVICE = torch.device("cuda")
CACHE = PROJECT / "data/cached_latents_v5.2"
CKPT = PROJECT / "models/checkpoints"

def seed_all(s=42):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

print("Loading data & classifier...")
cell_emb = np.load(CACHE / "cell_embeddings.npy")
text_emb = np.load(CACHE / "text_embeddings.npy")
proj_text = np.load(CACHE / "projected_text.npy")

N, D = cell_emb.shape
global_mean = cell_emb.mean(0)

# Group by text
_, text_labels = np.unique(np.round(text_emb[:, :50], 4), axis=0, return_inverse=True)
group_sizes = Counter(text_labels.tolist())
group_indices = defaultdict(list)
for i, g in enumerate(text_labels):
    group_indices[g].append(i)
group_indices = {g: np.array(v) for g, v in group_indices.items()}

valid_groups = sorted([g for g, s in group_sizes.items() if s >= 50],
                      key=lambda g: group_sizes[g], reverse=True)[:100]

group_cond = {g: proj_text[group_indices[g][0]] for g in valid_groups}
group_centroids = {g: cell_emb[group_indices[g]].mean(0) for g in valid_groups}
group_stds_val = {g: cell_emb[group_indices[g]].std(0).mean() for g in valid_groups}

# KNN classifier
cell_centered = cell_emb - global_mean
pca = PCA(n_components=50, random_state=42)
cell_pca = pca.fit_transform(cell_centered)

eval_mask = np.isin(text_labels, valid_groups)
eval_pca = cell_pca[eval_mask]
eval_labels = text_labels[eval_mask]
rng = np.random.default_rng(42)
perm = rng.permutation(eval_mask.sum())
split = int(len(perm) * 0.8)

knn = KNeighborsClassifier(n_neighbors=15, metric='cosine', n_jobs=-1)
knn.fit(eval_pca[perm[:split]], eval_labels[perm[:split]])

# Load model
from src.architecture.dit import DiT1D
dit = DiT1D(latent_dim=512, hidden_dim=384, cond_dim=256,
            num_blocks=8, num_heads=6).to(DEVICE)
ckpt = torch.load(CKPT / "dit_best.pth", map_location=DEVICE, weights_only=False)
dit.load_state_dict(ckpt["model_state_dict"])
dit.eval()

N_GEN = 200  # per group
N_EVAL_GROUPS = 50  # use top 50 for speed
eval_grps = valid_groups[:N_EVAL_GROUPS]

# Sweep parameters
CFG_SCALES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
ODE_STEPS = [10, 20, 50]

print(f"\nSweeping {len(CFG_SCALES)} CFG × {len(ODE_STEPS)} ODE steps...")
print(f"  {N_EVAL_GROUPS} groups × {N_GEN} cells = {N_EVAL_GROUPS * N_GEN} cells per config\n")

header = f"{'CFG':>5} {'Steps':>6} {'KNN-1':>8} {'KNN-5':>8} {'Steer':>8} {'DivR':>8} {'FD':>8}"
print(header)
print("-" * len(header))

best_knn = 0
best_config = None
all_results = []

for n_steps in ODE_STEPS:
    for cfg in CFG_SCALES:
        seed_all(42)
        
        gen_data = {}
        for g in eval_grps:
            cond_t = torch.tensor(group_cond[g], dtype=torch.float32, device=DEVICE)
            cond_batch = cond_t.unsqueeze(0).expand(N_GEN, -1)
            with torch.no_grad():
                gen_data[g] = dit.sample(cond_batch, cfg_scale=cfg, num_steps=n_steps).cpu().numpy()
        
        # Evaluate
        all_gen = np.vstack([gen_data[g] for g in eval_grps])
        all_labels = np.repeat(eval_grps, N_GEN)
        
        gen_c = all_gen - global_mean
        gen_pca = pca.transform(gen_c)
        
        # KNN
        preds = knn.predict(gen_pca)
        acc1 = accuracy_score(all_labels, preds)
        
        proba = knn.predict_proba(gen_pca)
        knn_classes = knn.classes_
        top5 = sum(1 for j, tl in enumerate(all_labels)
                   if tl in knn_classes[np.argsort(proba[j])[-5:]]) / len(all_labels)
        
        # Steering
        n_pairs = 500
        correct = 0
        rng2 = np.random.default_rng(123)
        for _ in range(n_pairs):
            a, b = rng2.choice(eval_grps, 2, replace=False)
            ga = gen_data[a].mean(0) - global_mean
            ra = group_centroids[a] - global_mean
            rb = group_centroids[b] - global_mean
            daa = np.dot(ga, ra) / (np.linalg.norm(ga) * np.linalg.norm(ra) + 1e-8)
            dab = np.dot(ga, rb) / (np.linalg.norm(ga) * np.linalg.norm(rb) + 1e-8)
            if daa > dab: correct += 1
        steer = correct / n_pairs
        
        # Diversity
        divr = np.mean([gen_data[g].std(0).mean() for g in eval_grps]) / np.mean([group_stds_val[g] for g in eval_grps])
        
        # FD
        from scipy.linalg import sqrtm
        real_sub = cell_emb[rng.choice(N, 15000, replace=False)]
        mu_r, mu_g = real_sub.mean(0), all_gen.mean(0)
        sig_r = np.cov(real_sub, rowvar=False)
        sig_g = np.cov(all_gen, rowvar=False)
        cm = sqrtm(sig_r @ sig_g)
        if np.iscomplexobj(cm): cm = cm.real
        fd = float((mu_r - mu_g) @ (mu_r - mu_g) + np.trace(sig_r + sig_g - 2 * cm))
        
        print(f"{cfg:>5.1f} {n_steps:>6} {acc1:>8.4f} {top5:>8.4f} {steer:>8.4f} {divr:>8.4f} {fd:>8.2f}")
        
        result = {"cfg": cfg, "steps": n_steps, "knn1": acc1, "knn5": top5,
                  "steering": steer, "div_ratio": divr, "fd": fd}
        all_results.append(result)
        
        if acc1 > best_knn:
            best_knn = acc1
            best_config = result

print(f"\n{'='*70}")
print(f"  BEST CONFIG: CFG={best_config['cfg']}, Steps={best_config['steps']}")
print(f"    KNN-1={best_config['knn1']:.4f}, KNN-5={best_config['knn5']:.4f}")
print(f"    Steering={best_config['steering']:.4f}, DivR={best_config['div_ratio']:.4f}")
print(f"    FD={best_config['fd']:.2f}")
print(f"{'='*70}")

# Also find best by steering
best_steer = max(all_results, key=lambda r: r['steering'])
print(f"\n  BEST STEERING: CFG={best_steer['cfg']}, Steps={best_steer['steps']}")
print(f"    Steering={best_steer['steering']:.4f}, KNN-1={best_steer['knn1']:.4f}")

# Best balanced (KNN * DivR)
best_bal = max(all_results, key=lambda r: r['knn1'] * min(r['div_ratio'], 1.0))
print(f"\n  BEST BALANCED: CFG={best_bal['cfg']}, Steps={best_bal['steps']}")
print(f"    KNN-1={best_bal['knn1']:.4f}, DivR={best_bal['div_ratio']:.4f}")
print(f"    Score={best_bal['knn1'] * min(best_bal['div_ratio'], 1.0):.4f}")

with open(PROJECT / "results/v5_final/cfg_sweep.json", "w") as f:
    json.dump([{k: float(v) if hasattr(v, 'item') else v for k, v in r.items()} for r in all_results], f, indent=2)
print(f"\n  Saved to results/v5_final/cfg_sweep.json")
