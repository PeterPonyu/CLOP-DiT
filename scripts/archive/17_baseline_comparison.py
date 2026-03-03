#!/usr/bin/env python3
"""
17_baseline_comparison.py — Phase 2 Decoder Architecture Comparison

All models use the same Stage 1 (CLOP) text projections as conditioning.
This is a fair comparison of decoder architectures for converting
256-d CLOP conditioning into 512-d scGPT cell embeddings.

DECODERS:
  --- Oracle baselines ---
  1. Per-Type Gaussian     — Oracle: sample from per-group N(μ_k, diag(σ²_k))
  2. Retrieval + Jitter    — Oracle: retrieve real cells + noise
  --- Feed-forward (1-pass, no iterative sampling) ---
  3. Conditional VAE       — Encoder-decoder with latent z
  4. Conditional GAN       — WGAN-GP adversarial
  --- Diffusion / Flow-matching (iterative ODE sampling) ---
  5. Flow MLP              — Flow matching with MLP velocity net
  6. CLOP-DiT              — Flow matching with Transformer velocity net (from ckpt)
  --- Ablation: Is ODE/diffusion necessary? ---
  7. DiT Step Ablation     — Same DiT ckpt, 1/2/5/10/20/50 ODE steps
  8. Direct Transformer    — Same DiT architecture, trained as direct predictor (no ODE)
  9. DDPM-DiT              — Same DiT architecture, discrete denoising (not flow matching)

METRICS:
  Classification: KNN-1, KNN-5, Steering, Linear accuracy
  Distributional: Per-group FD, Mean Correlation, Coverage
  Novelty:        Novelty score (distance to nearest real)
  Downstream:     Train-on-fake Test-on-real accuracy

OUTPUT:
  results/v5_final/baseline_comparison.json
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, json, time
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT = Path("/home/zeyufu/Desktop/CLOP-DiT")
sys.path.insert(0, str(PROJECT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = PROJECT / "data/cached_latents_v5.2"
CKPT = PROJECT / "models/checkpoints"
RESULTS = PROJECT / "results/v5_final"
RESULTS.mkdir(parents=True, exist_ok=True)


def seed_all(s=42):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ═══════════════════════════════════════════════════════════════════
#  EVALUATION — Classification + Distributional + Novelty + Downstream
# ═══════════════════════════════════════════════════════════════════

def _per_group_fd(real_cells, gen_cells):
    """Fréchet Distance between two sets of vectors (diagonal covariance)."""
    mu_r, mu_g = real_cells.mean(0), gen_cells.mean(0)
    std_r, std_g = real_cells.std(0) + 1e-8, gen_cells.std(0) + 1e-8
    mean_diff = np.sum((mu_r - mu_g) ** 2)
    cov_term = np.sum(std_r ** 2 + std_g ** 2 - 2 * std_r * std_g)
    return float(mean_diff + cov_term)


def evaluate_generation(gen_data, eval_groups, group_centroids, group_stds,
                        global_mean, knn, pca, knn_classes, scaler, lr_model,
                        N_GEN, cell_emb, group_indices):
    """Comprehensive evaluation: classification + distributional + novelty + downstream."""
    all_gen = np.vstack([gen_data[g] for g in eval_groups])
    all_labels = np.repeat(eval_groups, N_GEN)

    # ── Classification metrics ──
    gen_c = all_gen - global_mean
    gen_pca = pca.transform(gen_c)
    preds = knn.predict(gen_pca)
    knn1 = accuracy_score(all_labels, preds)
    proba = knn.predict_proba(gen_pca)
    knn5 = sum(1 for j, tl in enumerate(all_labels)
               if tl in knn_classes[np.argsort(proba[j])[-5:]]) / len(all_labels)

    # Steering (centered)
    rng2 = np.random.default_rng(123)
    correct, n_pairs = 0, 1000
    for _ in range(n_pairs):
        a, b = rng2.choice(eval_groups, 2, replace=False)
        ga = gen_data[a].mean(0) - global_mean
        ra = group_centroids[a] - global_mean
        rb = group_centroids[b] - global_mean
        daa = np.dot(ga, ra) / (np.linalg.norm(ga) * np.linalg.norm(ra) + 1e-8)
        dab = np.dot(ga, rb) / (np.linalg.norm(ga) * np.linalg.norm(rb) + 1e-8)
        if daa > dab:
            correct += 1
    steer = correct / n_pairs

    # Diversity ratio
    divr = np.mean([gen_data[g].std(0).mean() for g in eval_groups]) / \
           np.mean([group_stds[g] for g in eval_groups])

    # Linear classifier
    gen_pca_s = scaler.transform(gen_pca)
    lr_acc = lr_model.score(gen_pca_s, all_labels)

    # ── Distributional metrics ──
    per_group_fds = []
    for g in eval_groups:
        real_g = cell_emb[group_indices[g]]
        gen_g = gen_data[g]
        per_group_fds.append(_per_group_fd(real_g, gen_g))
    mean_group_fd = float(np.mean(per_group_fds))

    # Mean correlation (per-dimension mean match across groups)
    dim_means_real = np.array([cell_emb[group_indices[g]].mean(0) for g in eval_groups])
    dim_means_gen = np.array([gen_data[g].mean(0) for g in eval_groups])
    mean_corr = float(np.corrcoef(dim_means_real.flatten(), dim_means_gen.flatten())[0, 1])

    # ── Novelty & Coverage ──
    real_eval_idx = np.concatenate([group_indices[g] for g in eval_groups])
    real_eval = cell_emb[real_eval_idx]
    real_eval_pca = pca.transform(real_eval - global_mean)

    rng_n = np.random.default_rng(42)
    n_sub = min(5000, len(all_gen))
    gen_sub_idx = rng_n.choice(len(all_gen), n_sub, replace=False)
    gen_sub_pca = gen_pca[gen_sub_idx]

    n_sub_r = min(5000, len(real_eval_pca))
    real_sub_idx = rng_n.choice(len(real_eval_pca), n_sub_r, replace=False)
    real_sub_pca = real_eval_pca[real_sub_idx]

    nn_r = NearestNeighbors(n_neighbors=1, metric='cosine').fit(real_sub_pca)
    dists_gen_to_real, _ = nn_r.kneighbors(gen_sub_pca)
    novelty = float(np.mean(dists_gen_to_real))

    nn_g = NearestNeighbors(n_neighbors=1, metric='cosine').fit(gen_sub_pca)
    dists_real_to_gen, _ = nn_g.kneighbors(real_sub_pca)
    threshold = float(np.median(dists_gen_to_real))
    coverage = float(np.mean(dists_real_to_gen <= threshold))

    # ── Downstream utility: train on fake, test on real ──
    scaler_ds = StandardScaler()
    gen_pca_ds = scaler_ds.fit_transform(gen_pca)
    lr_ds = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    lr_ds.fit(gen_pca_ds, all_labels)
    real_pca_test = scaler_ds.transform(real_eval_pca)
    real_labels_test = np.concatenate([
        np.full(len(group_indices[g]), g) for g in eval_groups
    ])
    ds_acc = float(lr_ds.score(real_pca_test, real_labels_test))

    return {
        "knn_top1": float(knn1),
        "knn_top5": float(knn5),
        "steering": float(steer),
        "diversity_ratio": float(divr),
        "linear_acc": float(lr_acc),
        "per_group_fd": mean_group_fd,
        "mean_corr": mean_corr,
        "novelty": novelty,
        "coverage": coverage,
        "downstream_acc": ds_acc,
    }


# ═══════════════════════════════════════════════════════════════════
#  BASELINE 1: Per-Type Gaussian (Oracle)
# ═══════════════════════════════════════════════════════════════════

def generate_per_type_gaussian(cell_emb, group_indices, eval_groups, N_GEN):
    rng = np.random.default_rng(42)
    gen_data = {}
    for g in eval_groups:
        cells = cell_emb[group_indices[g]]
        mu = cells.mean(0)
        std = cells.std(0) + 1e-8
        gen_data[g] = rng.normal(mu, std, size=(N_GEN, cell_emb.shape[1]))
    return gen_data


# ═══════════════════════════════════════════════════════════════════
#  BASELINE 2: Retrieval + Jitter (Oracle)
# ═══════════════════════════════════════════════════════════════════

def generate_retrieval(cell_emb, group_indices, eval_groups, N_GEN, noise_scale=0.1):
    rng = np.random.default_rng(42)
    gen_data = {}
    for g in eval_groups:
        real = cell_emb[group_indices[g]]
        idx = rng.choice(len(real), N_GEN, replace=True)
        sampled = real[idx].copy()
        noise = rng.normal(0, noise_scale * real.std(0), size=sampled.shape)
        gen_data[g] = sampled + noise
    return gen_data


# ═══════════════════════════════════════════════════════════════════
#  DECODER A: Conditional VAE (Feed-forward)
# ═══════════════════════════════════════════════════════════════════

class ConditionalVAE(nn.Module):
    def __init__(self, x_dim=512, c_dim=256, h_dim=512, z_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + c_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
            nn.Linear(h_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
        )
        self.mu_head = nn.Linear(h_dim, z_dim)
        self.logvar_head = nn.Linear(h_dim, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + c_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
            nn.Linear(h_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
            nn.Linear(h_dim, x_dim),
        )

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], -1))
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], -1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    @torch.no_grad()
    def generate(self, c, n=200, z_dim=128):
        z = torch.randn(n, z_dim, device=c.device)
        c_exp = c.unsqueeze(0).expand(n, -1) if c.dim() == 1 else c
        return self.decode(z, c_exp)


def train_cvae(cell_emb, proj_text, text_labels, group_indices, valid_groups,
               epochs=100, batch_size=512, lr=3e-4):
    print("    Training cVAE...")
    model = ConditionalVAE(x_dim=512, c_dim=256, h_dim=512, z_dim=128).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    mask = np.isin(text_labels, valid_groups)
    x_train = torch.tensor(cell_emb[mask], dtype=torch.float32)
    c_train = torch.tensor(proj_text[mask], dtype=torch.float32)
    n = len(x_train)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x = x_train[idx].to(DEVICE)
            c = c_train[idx].to(DEVICE)
            x_hat, mu, logvar = model(x, c)
            recon = F.mse_loss(x_hat, x, reduction='sum') / len(idx)
            kld = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / len(idx)
            loss = recon + 0.1 * kld
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(idx)
        scheduler.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"      Epoch {ep+1:3d}: loss={total_loss/n:.4f}")
    model.eval()
    return model


def generate_cvae(model, group_cond, eval_groups, N_GEN):
    gen_data = {}
    for g in eval_groups:
        c = torch.tensor(group_cond[g], dtype=torch.float32, device=DEVICE)
        gen_data[g] = model.generate(c, n=N_GEN, z_dim=128).cpu().numpy()
    return gen_data


# ═══════════════════════════════════════════════════════════════════
#  DECODER B: Conditional GAN (Feed-forward, WGAN-GP)
# ═══════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    def __init__(self, z_dim=128, c_dim=256, h_dim=512, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + c_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
            nn.Linear(h_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
            nn.Linear(h_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
            nn.Linear(h_dim, out_dim),
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], -1))


class Critic(nn.Module):
    def __init__(self, x_dim=512, c_dim=256, h_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + c_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
            nn.Linear(h_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, x, c):
        return self.net(torch.cat([x, c], -1)).squeeze(-1)


def gradient_penalty(critic, real, fake, c, device):
    alpha = torch.rand(len(real), 1, device=device)
    interp = alpha * real + (1 - alpha) * fake
    interp.requires_grad_(True)
    d_interp = critic(interp, c)
    grads = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True
    )[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def train_cgan(cell_emb, proj_text, text_labels, group_indices, valid_groups,
               epochs=100, batch_size=512, lr=1e-4, n_critic=5):
    print("    Training cGAN (WGAN-GP)...")
    G = Generator(z_dim=128, c_dim=256, h_dim=512, out_dim=512).to(DEVICE)
    D = Critic(x_dim=512, c_dim=256, h_dim=512).to(DEVICE)
    opt_G = torch.optim.AdamW(G.parameters(), lr=lr, betas=(0.0, 0.9), weight_decay=1e-5)
    opt_D = torch.optim.AdamW(D.parameters(), lr=lr, betas=(0.0, 0.9), weight_decay=1e-5)

    mask = np.isin(text_labels, valid_groups)
    x_train = torch.tensor(cell_emb[mask], dtype=torch.float32)
    c_train = torch.tensor(proj_text[mask], dtype=torch.float32)
    n = len(x_train)

    for ep in range(epochs):
        perm = torch.randperm(n)
        total_d, total_g, n_d, n_g = 0, 0, 0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            real = x_train[idx].to(DEVICE)
            cond = c_train[idx].to(DEVICE)
            bs = len(real)

            D.train(); G.eval()
            z = torch.randn(bs, 128, device=DEVICE)
            with torch.no_grad():
                fake = G(z, cond)
            d_real = D(real, cond).mean()
            d_fake = D(fake, cond).mean()
            gp = gradient_penalty(D, real, fake, cond, DEVICE)
            d_loss = d_fake - d_real + 10.0 * gp
            opt_D.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_D.step()
            total_d += d_loss.item(); n_d += 1

            if n_d % n_critic == 0:
                G.train(); D.eval()
                z = torch.randn(bs, 128, device=DEVICE)
                fake = G(z, cond)
                g_loss = -D(fake, cond).mean()
                opt_G.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                opt_G.step()
                total_g += g_loss.item(); n_g += 1

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"      Epoch {ep+1:3d}: D_loss={total_d/max(n_d,1):.4f} "
                  f"G_loss={total_g/max(n_g,1):.4f}")
    G.eval()
    return G


def generate_cgan(model, group_cond, eval_groups, N_GEN):
    gen_data = {}
    for g in eval_groups:
        c = torch.tensor(group_cond[g], dtype=torch.float32, device=DEVICE)
        c_batch = c.unsqueeze(0).expand(N_GEN, -1)
        z = torch.randn(N_GEN, 128, device=DEVICE)
        with torch.no_grad():
            gen_data[g] = model(z, c_batch).cpu().numpy()
    return gen_data


# ═══════════════════════════════════════════════════════════════════
#  DECODER C: Conditional Flow-Matching MLP (same framework as DiT)
# ═══════════════════════════════════════════════════════════════════

class FlowMLP(nn.Module):
    """Flow matching with an MLP velocity network (no transformer).
    Same training objective as DiT: v(z_t, t, c) ≈ z_1 - z_0.
    This isolates the effect of the transformer architecture.
    """
    def __init__(self, x_dim=512, c_dim=256, h_dim=512, t_dim=64):
        super().__init__()
        self.t_embed = nn.Sequential(
            nn.Linear(1, t_dim), nn.GELU(),
            nn.Linear(t_dim, t_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(x_dim + c_dim + t_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
            nn.Linear(h_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
            nn.Linear(h_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
            nn.Linear(h_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
            nn.Linear(h_dim, x_dim),
        )

    def forward(self, z_t, t, c):
        t_emb = self.t_embed(t.unsqueeze(-1))
        return self.net(torch.cat([z_t, c, t_emb], -1))

    @torch.no_grad()
    def sample(self, c, num_steps=10, cfg_scale=0.0):
        n = c.shape[0]
        z = torch.randn(n, 512, device=c.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((n,), i * dt, device=c.device)
            v = self.forward(z, t, c)
            z = z + v * dt
        return z

    @torch.no_grad()
    def sample_midpoint(self, c, num_steps=10, cfg_scale=0.0):
        n = c.shape[0]
        z = torch.randn(n, 512, device=c.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((n,), i * dt, device=c.device)
            v1 = self.forward(z, t, c)
            z_mid = z + 0.5 * dt * v1
            t_mid = torch.full((n,), (i + 0.5) * dt, device=c.device)
            v2 = self.forward(z_mid, t_mid, c)
            z = z + dt * v2
        return z


def train_flow_mlp(cell_emb, proj_text, text_labels, group_indices, valid_groups,
                   epochs=100, batch_size=512, lr=3e-4):
    print("    Training Flow-Matching MLP...")
    model = FlowMLP(x_dim=512, c_dim=256, h_dim=512).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {n_params:,} (vs DiT: 22,100,000)")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    mask = np.isin(text_labels, valid_groups)
    x_train = torch.tensor(cell_emb[mask], dtype=torch.float32)
    c_train = torch.tensor(proj_text[mask], dtype=torch.float32)
    n = len(x_train)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x1 = x_train[idx].to(DEVICE)
            c = c_train[idx].to(DEVICE)
            x0 = torch.randn_like(x1)
            t = torch.rand(len(x1), device=DEVICE)
            z_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
            target_v = x1 - x0
            pred_v = model(z_t, t, c)
            loss = F.mse_loss(pred_v, target_v)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(x1)
        scheduler.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"      Epoch {ep+1:3d}: flow_loss={total_loss/n:.4f}")
    model.eval()
    return model


def generate_flow_mlp(model, group_cond, eval_groups, N_GEN, method="euler"):
    gen_data = {}
    for g in eval_groups:
        c = torch.tensor(group_cond[g], dtype=torch.float32, device=DEVICE)
        c_batch = c.unsqueeze(0).expand(N_GEN, -1)
        if method == "midpoint":
            gen_data[g] = model.sample_midpoint(c_batch, num_steps=10).cpu().numpy()
        else:
            gen_data[g] = model.sample(c_batch, num_steps=10).cpu().numpy()
    return gen_data


# ═══════════════════════════════════════════════════════════════════
#  DECODER D: Direct Transformer (same DiT architecture, NO ODE/flow)
#  Tests: "Is the ODE framework necessary, or does the transformer
#          work better as a direct conditional predictor?"
#  Architecture: same DiT1D (8 blocks, 6 heads, 384 hidden, 16 tokens)
#  Training: condition + noise → cell embedding (MSE, single forward pass)
#  Inference: one forward pass (no iterative ODE sampling)
# ═══════════════════════════════════════════════════════════════════

class DirectTransformer(nn.Module):
    """Same DiT1D architecture but used as a direct predictor.
    Input: noise z ~ N(0,I) concatenated with condition c.
    Output: cell embedding (direct prediction, no velocity/ODE).
    Uses fixed t=0.5 to keep the timestep conditioning active.
    """
    def __init__(self, latent_dim=512, hidden_dim=384, cond_dim=256,
                 num_heads=6, num_blocks=8, num_tokens=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.token_dim = latent_dim // num_tokens

        # Same architecture as DiT1D
        self.input_proj = nn.Linear(self.token_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)

        # Timestep embedder (fixed t=0.5 during inference)
        from src.architecture.dit import TimestepEmbedder, ConditionEmbedder, DiTBlock, DiTFinalLayer
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.c_embedder = ConditionEmbedder(cond_dim, hidden_dim, dropout_prob=0.0)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, 4.0, 0.0, 0.1)
            for _ in range(num_blocks)
        ])
        self.final_layer = DiTFinalLayer(hidden_dim, self.token_dim)

        # Init
        for m in [self.input_proj]:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_noise, t, cond):
        B = z_noise.shape[0]
        x = z_noise.view(B, self.num_tokens, self.token_dim)
        x = self.input_proj(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        c_emb = self.c_embedder(cond)
        combined = t_emb + c_emb
        for block in self.blocks:
            x = block(x, combined)
        x = self.final_layer(x, combined)
        return x.reshape(B, self.latent_dim)

    @torch.no_grad()
    def generate(self, cond, n=200):
        device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        t = torch.full((n,), 0.5, device=device)
        c = cond.unsqueeze(0).expand(n, -1) if cond.dim() == 1 else cond
        return self.forward(z, t, c)


def train_direct_transformer(cell_emb, proj_text, text_labels, group_indices,
                              valid_groups, epochs=100, batch_size=512, lr=2e-4):
    """Train transformer as direct predictor: noise + condition → cell.
    No flow matching, no ODE. Same architecture as DiT (22M params).
    """
    print("    Training Direct Transformer (no ODE)...")
    model = DirectTransformer(latent_dim=512, hidden_dim=384, cond_dim=256,
                               num_heads=6, num_blocks=8, num_tokens=16).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {n_params:,} (same architecture as DiT)")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    mask = np.isin(text_labels, valid_groups)
    x_train = torch.tensor(cell_emb[mask], dtype=torch.float32)
    c_train = torch.tensor(proj_text[mask], dtype=torch.float32)
    n = len(x_train)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x_real = x_train[idx].to(DEVICE)
            c = c_train[idx].to(DEVICE)
            z_noise = torch.randn_like(x_real)
            # Random t for training (keeps timestep embedder active)
            t = torch.rand(len(x_real), device=DEVICE)
            pred = model(z_noise, t, c)
            loss = F.mse_loss(pred, x_real)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(x_real)
        scheduler.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"      Epoch {ep+1:3d}: mse_loss={total_loss/n:.4f}")
    model.eval()
    return model


def generate_direct_transformer(model, group_cond, eval_groups, N_GEN):
    gen_data = {}
    for g in eval_groups:
        c = torch.tensor(group_cond[g], dtype=torch.float32, device=DEVICE)
        gen_data[g] = model.generate(c, n=N_GEN).cpu().numpy()
    return gen_data


# ═══════════════════════════════════════════════════════════════════
#  DECODER E: DDPM-DiT (same architecture, discrete denoising)
#  Tests: "Is flow matching the right diffusion framework?"
#  Architecture: same DiT1D (8 blocks, 6 heads, 384 hidden)
#  Training: predict noise ε from x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε
#  Inference: reverse process with T steps (DDPM-style)
# ═══════════════════════════════════════════════════════════════════

class DDPMDiT(nn.Module):
    """Same DiT architecture but with DDPM discrete denoising.
    Predicts noise ε rather than velocity v.
    Uses cosine noise schedule.
    """
    def __init__(self, latent_dim=512, hidden_dim=384, cond_dim=256,
                 num_heads=6, num_blocks=8, num_tokens=16, T=1000):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.token_dim = latent_dim // num_tokens
        self.T = T

        # Cosine noise schedule
        steps = torch.linspace(0, T, T + 1)
        alpha_bar = torch.cos(((steps / T) + 0.008) / 1.008 * torch.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = betas.clamp(max=0.999)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, 0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1 - alpha_cumprod))

        # Same transformer architecture as DiT
        self.input_proj = nn.Linear(self.token_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)

        from src.architecture.dit import TimestepEmbedder, ConditionEmbedder, DiTBlock, DiTFinalLayer
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.c_embedder = ConditionEmbedder(cond_dim, hidden_dim, dropout_prob=0.15)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, 4.0, 0.0, 0.1)
            for _ in range(num_blocks)
        ])
        self.final_layer = DiTFinalLayer(hidden_dim, self.token_dim)

        for m in [self.input_proj]:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_t, t_idx, cond, force_drop_cond=False):
        """Predict noise ε from noisy input x_t at discrete timestep t_idx."""
        B = x_t.shape[0]
        x = x_t.view(B, self.num_tokens, self.token_dim)
        x = self.input_proj(x) + self.pos_embed
        # Normalize timestep to [0, 1] for the embedder
        t_norm = t_idx.float() / self.T
        t_emb = self.t_embedder(t_norm)
        c_emb = self.c_embedder(cond, force_drop=force_drop_cond)
        combined = t_emb + c_emb
        for block in self.blocks:
            x = block(x, combined)
        x = self.final_layer(x, combined)
        return x.reshape(B, self.latent_dim)

    def forward_with_cfg(self, x_t, t_idx, cond, cfg_scale=2.0):
        eps_cond = self.forward(x_t, t_idx, cond, force_drop_cond=False)
        eps_uncond = self.forward(x_t, t_idx, cond, force_drop_cond=True)
        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    @torch.no_grad()
    def sample(self, cond, num_steps=50, cfg_scale=2.0):
        """DDPM reverse sampling with strided timesteps."""
        device = next(self.parameters()).device
        B = cond.shape[0]
        x = torch.randn(B, self.latent_dim, device=device)

        # Use evenly spaced timesteps for faster sampling
        stride = max(1, self.T // num_steps)
        timesteps = list(range(self.T - 1, -1, -stride))

        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            eps = self.forward_with_cfg(x, t, cond, cfg_scale=cfg_scale)

            alpha = self.alphas[t_val]
            alpha_bar = self.alpha_cumprod[t_val]
            alpha_bar_prev = self.alpha_cumprod[t_val - stride] if t_val >= stride else torch.tensor(1.0)

            # DDPM update
            pred_x0 = (x - self.sqrt_one_minus_alpha_cumprod[t_val] * eps) / self.sqrt_alpha_cumprod[t_val]
            pred_x0 = pred_x0.clamp(-30, 30)  # stability clamp

            # Posterior mean
            coef1 = torch.sqrt(alpha_bar_prev) * (1 - alpha) / (1 - alpha_bar)
            coef2 = torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
            x_prev = coef1 * pred_x0 + coef2 * x

            # Add noise (except at t=0)
            if t_val > 0:
                beta_tilde = (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha)
                noise = torch.randn_like(x)
                x = x_prev + torch.sqrt(beta_tilde.clamp(min=1e-8)) * noise
            else:
                x = x_prev

        return x


def train_ddpm_dit(cell_emb, proj_text, text_labels, group_indices,
                    valid_groups, epochs=100, batch_size=512, lr=2e-4):
    """Train DDPM-DiT: same transformer, discrete denoising (predict noise)."""
    print("    Training DDPM-DiT (discrete denoising)...")
    model = DDPMDiT(latent_dim=512, hidden_dim=384, cond_dim=256,
                     num_heads=6, num_blocks=8, num_tokens=16, T=1000).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Trainable parameters: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    mask = np.isin(text_labels, valid_groups)
    x_train = torch.tensor(cell_emb[mask], dtype=torch.float32)
    c_train = torch.tensor(proj_text[mask], dtype=torch.float32)
    n = len(x_train)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            x0 = x_train[idx].to(DEVICE)
            c = c_train[idx].to(DEVICE)
            # Sample random timestep
            t = torch.randint(0, model.T, (len(x0),), device=DEVICE)
            # Forward diffusion
            eps = torch.randn_like(x0)
            sqrt_ab = model.sqrt_alpha_cumprod[t].unsqueeze(-1)
            sqrt_1_ab = model.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1)
            x_t = sqrt_ab * x0 + sqrt_1_ab * eps
            # Predict noise
            eps_pred = model(x_t, t, c)
            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(x0)
        scheduler.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"      Epoch {ep+1:3d}: noise_mse={total_loss/n:.4f}")
    model.eval()
    return model


def generate_ddpm_dit(model, group_cond, eval_groups, N_GEN, num_steps=50, cfg_scale=2.0):
    gen_data = {}
    for g in eval_groups:
        c = torch.tensor(group_cond[g], dtype=torch.float32, device=DEVICE)
        c_batch = c.unsqueeze(0).expand(N_GEN, -1)
        gen_data[g] = model.sample(c_batch, num_steps=num_steps,
                                    cfg_scale=cfg_scale).cpu().numpy()
    return gen_data


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    seed_all(42)

    print("=" * 76)
    print("  CLOP-DiT: PHASE 2 DECODER ARCHITECTURE COMPARISON")
    print("  All decoders conditioned on identical CLOP text projections (256-d)")
    print("=" * 76)

    # ── 1. Load Data ─────────────────────────────────────────────
    print("\n[1] Loading data...")
    cell_emb = np.load(CACHE / "cell_embeddings.npy")
    text_emb = np.load(CACHE / "text_embeddings.npy")
    proj_text = np.load(CACHE / "projected_text.npy")

    N, D = cell_emb.shape
    global_mean = cell_emb.mean(0)

    _, text_labels = np.unique(np.round(text_emb[:, :50], 4),
                               axis=0, return_inverse=True)
    group_sizes = Counter(text_labels.tolist())
    group_indices = defaultdict(list)
    for i, g in enumerate(text_labels):
        group_indices[g].append(i)
    group_indices = {g: np.array(v) for g, v in group_indices.items()}

    valid_groups = sorted([g for g, s in group_sizes.items() if s >= 50],
                          key=lambda g: group_sizes[g], reverse=True)[:100]

    group_cond = {g: proj_text[group_indices[g][0]] for g in valid_groups}
    group_centroids = {g: cell_emb[group_indices[g]].mean(0) for g in valid_groups}
    group_stds = {g: cell_emb[group_indices[g]].std(0).mean() for g in valid_groups}

    print(f"    {N:,} cells, {D}-dim, {len(valid_groups)} eval groups")

    # ── 2. Build Classifiers ────────────────────────────────────
    print("\n[2] Building classifiers...")
    cell_centered = cell_emb - global_mean
    pca = PCA(n_components=50, random_state=42)
    cell_pca = pca.fit_transform(cell_centered)

    eval_mask = np.isin(text_labels, valid_groups)
    eval_pca = cell_pca[eval_mask]
    eval_labels = text_labels[eval_mask]
    rng = np.random.default_rng(42)
    perm = rng.permutation(eval_mask.sum())
    split = int(len(perm) * 0.8)
    train_idx = perm[:split]

    knn = KNeighborsClassifier(n_neighbors=15, metric='cosine', n_jobs=-1)
    knn.fit(eval_pca[train_idx], eval_labels[train_idx])
    knn_classes = knn.classes_

    scaler_cls = StandardScaler()
    scaler_cls.fit(eval_pca[train_idx])
    lr_model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    lr_model.fit(scaler_cls.transform(eval_pca[train_idx]), eval_labels[train_idx])

    N_GEN = 200

    eval_args = dict(eval_groups=valid_groups, group_centroids=group_centroids,
                     group_stds=group_stds, global_mean=global_mean, knn=knn,
                     pca=pca, knn_classes=knn_classes, scaler=scaler_cls,
                     lr_model=lr_model, N_GEN=N_GEN, cell_emb=cell_emb,
                     group_indices=group_indices)

    # ── 3. CLOP-DiT (generate fresh for distributional metrics) ─
    print("\n[3] Loading DiT model and generating cells...")
    final_eval = json.load(open(RESULTS / "final_evaluation.json"))
    from src.architecture.dit import DiT1D
    dit = DiT1D(latent_dim=512, hidden_dim=384, cond_dim=256,
                num_heads=6, num_blocks=8, num_tokens=16, cond_drop_prob=0.15).to(DEVICE)
    ckpt = torch.load(CKPT / "dit_best.pth", map_location=DEVICE, weights_only=True)
    dit.load_state_dict(ckpt["ema_state_dict"])
    dit.eval()
    n_dit = sum(p.numel() for p in dit.parameters())
    print(f"    DiT parameters: {n_dit:,}")

    gen_dit_acc, gen_dit_bal = {}, {}
    for g in valid_groups:
        cond_t = torch.tensor(group_cond[g], dtype=torch.float32, device=DEVICE)
        cond_batch = cond_t.unsqueeze(0).expand(N_GEN, -1)
        with torch.no_grad():
            gen_dit_acc[g] = dit.sample(cond_batch, cfg_scale=2.0,
                                         num_steps=10).cpu().numpy()
            gen_dit_bal[g] = dit.sample_midpoint(cond_batch, cfg_scale=1.0,
                                                  num_steps=10).cpu().numpy()

    print("    Evaluating DiT (CFG=2.0 Euler)...")
    res_dit_acc = evaluate_generation(gen_dit_acc, **eval_args)
    print(f"      KNN={res_dit_acc['knn_top1']:.4f} DivR={res_dit_acc['diversity_ratio']:.4f} "
          f"FD_g={res_dit_acc['per_group_fd']:.2f} r={res_dit_acc['mean_corr']:.4f} "
          f"DS={res_dit_acc['downstream_acc']:.4f}")

    print("    Evaluating DiT (CFG=1.0 Midpoint)...")
    res_dit_bal = evaluate_generation(gen_dit_bal, **eval_args)
    print(f"      KNN={res_dit_bal['knn_top1']:.4f} DivR={res_dit_bal['diversity_ratio']:.4f} "
          f"FD_g={res_dit_bal['per_group_fd']:.2f} r={res_dit_bal['mean_corr']:.4f} "
          f"DS={res_dit_bal['downstream_acc']:.4f}")

    # ── 3b. ODE Step Count Ablation (uses existing DiT checkpoint) ─
    print("\n[3b] ODE Step Count Ablation (1, 2, 5, 10, 20, 50 steps)...")
    step_counts = [1, 2, 5, 10, 20, 50]
    res_steps = {}
    for nsteps in step_counts:
        seed_all(42)
        gen_step = {}
        for g in valid_groups:
            cond_t = torch.tensor(group_cond[g], dtype=torch.float32, device=DEVICE)
            cond_batch = cond_t.unsqueeze(0).expand(N_GEN, -1)
            with torch.no_grad():
                gen_step[g] = dit.sample(cond_batch, cfg_scale=2.0,
                                          num_steps=nsteps).cpu().numpy()
        r = evaluate_generation(gen_step, **eval_args)
        res_steps[nsteps] = r
        print(f"    Steps={nsteps:2d}: KNN={r['knn_top1']:.4f} DivR={r['diversity_ratio']:.4f} "
              f"FD_g={r['per_group_fd']:.2f} DS={r['downstream_acc']:.4f}")
    print("    → This answers: 'Is the multi-step ODE necessary?'")
    print(f"    → 1-step KNN = {res_steps[1]['knn_top1']:.4f} vs 10-step = {res_steps[10]['knn_top1']:.4f}")

    # ── 4. Per-Type Gaussian (Oracle) ───────────────────────────
    print("\n[4] Per-Type Gaussian (oracle)...")
    seed_all(42)
    gen_ptg = generate_per_type_gaussian(cell_emb, group_indices, valid_groups, N_GEN)
    res_ptg = evaluate_generation(gen_ptg, **eval_args)
    print(f"    KNN={res_ptg['knn_top1']:.4f} DivR={res_ptg['diversity_ratio']:.4f} "
          f"FD_g={res_ptg['per_group_fd']:.2f} r={res_ptg['mean_corr']:.4f} "
          f"DS={res_ptg['downstream_acc']:.4f}")

    # ── 5. Retrieval + Jitter (Oracle) ──────────────────────────
    print("\n[5] Retrieval + Jitter (oracle)...")
    seed_all(42)
    gen_ret = generate_retrieval(cell_emb, group_indices, valid_groups, N_GEN, 0.1)
    res_ret = evaluate_generation(gen_ret, **eval_args)
    print(f"    KNN={res_ret['knn_top1']:.4f} DivR={res_ret['diversity_ratio']:.4f} "
          f"FD_g={res_ret['per_group_fd']:.2f} r={res_ret['mean_corr']:.4f} "
          f"DS={res_ret['downstream_acc']:.4f}")

    # ── 6. Conditional VAE ──────────────────────────────────────
    print("\n[6] Conditional VAE...")
    seed_all(42)
    cvae = train_cvae(cell_emb, proj_text, text_labels, group_indices, valid_groups,
                      epochs=100, batch_size=512, lr=3e-4)
    n_cvae = sum(p.numel() for p in cvae.parameters())
    print(f"    cVAE parameters: {n_cvae:,}")
    gen_cvae = generate_cvae(cvae, group_cond, valid_groups, N_GEN)
    res_cvae = evaluate_generation(gen_cvae, **eval_args)
    print(f"    KNN={res_cvae['knn_top1']:.4f} DivR={res_cvae['diversity_ratio']:.4f} "
          f"FD_g={res_cvae['per_group_fd']:.2f} r={res_cvae['mean_corr']:.4f} "
          f"DS={res_cvae['downstream_acc']:.4f}")

    # ── 7. Conditional GAN ──────────────────────────────────────
    print("\n[7] Conditional GAN (WGAN-GP)...")
    seed_all(42)
    cgan = train_cgan(cell_emb, proj_text, text_labels, group_indices, valid_groups,
                      epochs=100, batch_size=512, lr=1e-4, n_critic=5)
    n_cgan = sum(p.numel() for p in cgan.parameters())
    print(f"    cGAN parameters: {n_cgan:,}")
    gen_cgan = generate_cgan(cgan, group_cond, valid_groups, N_GEN)
    res_cgan = evaluate_generation(gen_cgan, **eval_args)
    print(f"    KNN={res_cgan['knn_top1']:.4f} DivR={res_cgan['diversity_ratio']:.4f} "
          f"FD_g={res_cgan['per_group_fd']:.2f} r={res_cgan['mean_corr']:.4f} "
          f"DS={res_cgan['downstream_acc']:.4f}")

    # ── 8. Flow-Matching MLP ────────────────────────────────────
    print("\n[8] Conditional Flow-Matching MLP...")
    seed_all(42)
    flow_mlp = train_flow_mlp(cell_emb, proj_text, text_labels, group_indices,
                               valid_groups, epochs=100, batch_size=512, lr=3e-4)
    n_fmlp = sum(p.numel() for p in flow_mlp.parameters())
    gen_fmlp_euler = generate_flow_mlp(flow_mlp, group_cond, valid_groups, N_GEN, "euler")
    res_fmlp_euler = evaluate_generation(gen_fmlp_euler, **eval_args)
    gen_fmlp_mid = generate_flow_mlp(flow_mlp, group_cond, valid_groups, N_GEN, "midpoint")
    res_fmlp_mid = evaluate_generation(gen_fmlp_mid, **eval_args)
    print(f"    Flow MLP (Euler):    KNN={res_fmlp_euler['knn_top1']:.4f} "
          f"DivR={res_fmlp_euler['diversity_ratio']:.4f} FD_g={res_fmlp_euler['per_group_fd']:.2f} "
          f"r={res_fmlp_euler['mean_corr']:.4f} DS={res_fmlp_euler['downstream_acc']:.4f}")
    print(f"    Flow MLP (Midpoint): KNN={res_fmlp_mid['knn_top1']:.4f} "
          f"DivR={res_fmlp_mid['diversity_ratio']:.4f} FD_g={res_fmlp_mid['per_group_fd']:.2f} "
          f"r={res_fmlp_mid['mean_corr']:.4f} DS={res_fmlp_mid['downstream_acc']:.4f}")

    # ── 9. Direct Transformer (no ODE) ──────────────────────────
    print("\n[9] Direct Transformer (same architecture, NO ODE)...")
    seed_all(42)
    direct_tf = train_direct_transformer(cell_emb, proj_text, text_labels,
                                          group_indices, valid_groups,
                                          epochs=100, batch_size=512, lr=2e-4)
    n_direct = sum(p.numel() for p in direct_tf.parameters())
    gen_direct = generate_direct_transformer(direct_tf, group_cond, valid_groups, N_GEN)
    res_direct = evaluate_generation(gen_direct, **eval_args)
    print(f"    Direct Transformer:  KNN={res_direct['knn_top1']:.4f} "
          f"DivR={res_direct['diversity_ratio']:.4f} FD_g={res_direct['per_group_fd']:.2f} "
          f"r={res_direct['mean_corr']:.4f} DS={res_direct['downstream_acc']:.4f}")
    print(f"    → This answers: 'Is ODE necessary for the transformer architecture?'")

    # ── 10. DDPM-DiT (discrete denoising) ───────────────────────
    print("\n[10] DDPM-DiT (same architecture, discrete denoising)...")
    seed_all(42)
    ddpm = train_ddpm_dit(cell_emb, proj_text, text_labels, group_indices,
                           valid_groups, epochs=100, batch_size=512, lr=2e-4)
    n_ddpm = sum(p.numel() for p in ddpm.parameters() if p.requires_grad)
    gen_ddpm = generate_ddpm_dit(ddpm, group_cond, valid_groups, N_GEN,
                                  num_steps=50, cfg_scale=2.0)
    res_ddpm = evaluate_generation(gen_ddpm, **eval_args)
    print(f"    DDPM-DiT:            KNN={res_ddpm['knn_top1']:.4f} "
          f"DivR={res_ddpm['diversity_ratio']:.4f} FD_g={res_ddpm['per_group_fd']:.2f} "
          f"r={res_ddpm['mean_corr']:.4f} DS={res_ddpm['downstream_acc']:.4f}")
    print(f"    → This answers: 'Is flow matching better than DDPM for this task?'")

    # ── 11. Summary Table ───────────────────────────────────────
    print("\n" + "=" * 130)
    print("  PHASE 2 DECODER ARCHITECTURE COMPARISON — COMPREHENSIVE RESULTS")
    print("  All decoders use identical CLOP Stage 1 conditioning (256-d projections)")
    print("=" * 130)

    header = (f"{'Decoder':<28} {'Type':<12} {'Params':>10} {'KNN-1↑':>8} "
              f"{'DivR→1':>8} {'FD_g↓':>8} {'r↑':>8} {'Novel':>8} "
              f"{'Cover':>8} {'DS↑':>8}")
    print(header)
    print("-" * len(header))

    rows = [
        ("Per-Type Gaussian", "Oracle", "—", res_ptg),
        ("Retrieval + Jitter", "Oracle", "—", res_ret),
        ("Conditional VAE", "FF", f"{n_cvae:,}", res_cvae),
        ("Conditional GAN", "FF", f"{n_cgan:,}", res_cgan),
        ("Direct Transformer", "FF-T", f"{n_direct:,}", res_direct),
        ("Flow MLP (Euler)", "Diff", f"{n_fmlp:,}", res_fmlp_euler),
        ("DDPM-DiT", "DDPM", f"{n_ddpm:,}", res_ddpm),
        ("CLOP-DiT (CFG=2.0 E)", "Flow", f"{n_dit:,}", res_dit_acc),
        ("CLOP-DiT (CFG=1.0 M)", "Flow", f"{n_dit:,}", res_dit_bal),
    ]
    for name, typ, params, r in rows:
        print(f"{name:<28} {typ:<12} {params:>10} {r['knn_top1']:>8.4f} "
              f"{r['diversity_ratio']:>8.4f} {r['per_group_fd']:>8.2f} "
              f"{r['mean_corr']:>8.4f} {r['novelty']:>8.4f} "
              f"{r['coverage']:>8.4f} {r['downstream_acc']:>8.4f}")

    print("\n" + "=" * 90)
    print("  ODE STEP COUNT ABLATION (DiT CFG=2.0, Euler)")
    print("=" * 90)
    print(f"{'Steps':>6} {'KNN-1↑':>8} {'DivR→1':>8} {'FD_g↓':>8} {'DS↑':>8}")
    print("-" * 42)
    for nsteps in step_counts:
        r = res_steps[nsteps]
        print(f"{nsteps:>6} {r['knn_top1']:>8.4f} {r['diversity_ratio']:>8.4f} "
              f"{r['per_group_fd']:>8.2f} {r['downstream_acc']:>8.4f}")

    # ── 12. Save ────────────────────────────────────────────────
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Phase 2 decoder architecture comparison — expanded with ODE ablation, "
                           "Direct Transformer, and DDPM-DiT baselines",
            "n_eval_groups": len(valid_groups),
            "n_gen_per_group": N_GEN,
            "cvae_epochs": 100,
            "cgan_epochs": 100,
            "flow_mlp_epochs": 100,
            "direct_transformer_epochs": 100,
            "ddpm_dit_epochs": 100,
            "total_time_s": time.time() - t0,
            "model_params": {
                "conditional_vae": n_cvae,
                "conditional_gan_G": n_cgan,
                "flow_mlp": n_fmlp,
                "direct_transformer": n_direct,
                "ddpm_dit": n_ddpm,
                "clop_dit": n_dit,
            },
        },
        "oracle_baselines": {
            "per_type_gaussian": res_ptg,
            "retrieval_jitter": res_ret,
        },
        "feed_forward_decoders": {
            "conditional_vae": res_cvae,
            "conditional_gan": res_cgan,
            "direct_transformer": res_direct,
        },
        "diffusion_decoders": {
            "flow_mlp_euler": res_fmlp_euler,
            "flow_mlp_midpoint": res_fmlp_mid,
            "ddpm_dit": res_ddpm,
            "clop_dit_best_acc": res_dit_acc,
            "clop_dit_balanced": res_dit_bal,
        },
        "ode_step_ablation": {str(k): v for k, v in res_steps.items()},
        "real_data_baseline": final_eval.get("real_data_baseline", {}),
    }

    out_path = RESULTS / "baseline_comparison.json"
    json.dump(output, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
