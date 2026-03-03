#!/usr/bin/env python3
"""
16_publication_figures.py — Generate ALL publication-quality figures for CLOP-DiT

Uses verified v5.2 results (220,304 cells, 80 datasets, 1,088 text groups).
Produces figures for both the Evaluation Report and the JBHI article.

OUTPUTS (saved to figures/v5_publication/):
  fig2_training_dynamics.png  — Training curves for CLOP and DiT (4 panels)
  fig3_pca_embedding.png      — PCA of real vs generated cells
  fig4_metrics_dashboard.png  — Comprehensive metrics (6 panels)
  fig5_biological_validation.png — PCA separation, cosine heatmap, diversity
  fig6_cfg_solver.png         — CFG sweep curves + solver comparison
  fig7_dimension_sampling.png — Dimension analysis, ODE trajectory, norm dist.
  fig8_tables.png             — Comparison & ablation tables as figures
"""

import numpy as np
import torch
import sys, json, re, time
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.neighbors import KNeighborsClassifier
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
FIGURES = PROJECT / "figures/v5_publication"
FIGURES.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

# ── Publication style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette (WCAG-accessible against white background)
COLORS = {
    'blue': '#1976D2',
    'red': '#D32F2F',
    'green': '#2E7D32',
    'orange': '#E65100',
    'purple': '#7B1FA2',
    'teal': '#00796B',
    'amber': '#BF6C00',
    'pink': '#880E4F',
    'grey': '#455A64',
    'indigo': '#283593',
}


def seed_all(s=42):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ═══════════════════════════════════════════════════════════════════
#  VISUAL CONFLICT DETECTION
# ═══════════════════════════════════════════════════════════════════

def _hex_to_rgb(h):
    """Convert hex color to (r,g,b) float tuple."""
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

def _relative_luminance(rgb):
    """Relative luminance per WCAG 2.0."""
    def linearize(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = [linearize(c) for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def _contrast_ratio(c1, c2):
    l1 = _relative_luminance(c1) + 0.05
    l2 = _relative_luminance(c2) + 0.05
    return max(l1, l2) / min(l1, l2)

def _color_distance(c1, c2):
    """Euclidean distance in RGB space."""
    return sum((a - b)**2 for a, b in zip(c1, c2)) ** 0.5

def detect_visual_conflicts():
    """Detect potential visual conflicts across all generated figures.
    
    Checks:
      1. Color accessibility — adjacent palette colors distinguishable?
      2. File integrity — all expected figures exist and are non-trivial?
      3. Contrast — palette colors readable on white background?
      4. Consistency — figure naming vs content alignment?
    """
    print("\n  ═══ VISUAL CONFLICT DETECTION ═══")
    issues = []

    # 1. Palette color accessibility
    palette = list(COLORS.values())
    names = list(COLORS.keys())
    MIN_DIST = 0.25  # Minimum RGB distance for distinguishability
    for i in range(len(palette)):
        for j in range(i+1, len(palette)):
            d = _color_distance(_hex_to_rgb(palette[i]), _hex_to_rgb(palette[j]))
            if d < MIN_DIST:
                issues.append(f"  ⚠ Colors '{names[i]}' and '{names[j]}' may be hard to distinguish (dist={d:.3f} < {MIN_DIST})")
    
    # 2. Contrast against white background
    white = (1.0, 1.0, 1.0)
    MIN_CONTRAST = 3.0  # WCAG AA for large text
    for name, hexc in COLORS.items():
        cr = _contrast_ratio(_hex_to_rgb(hexc), white)
        if cr < MIN_CONTRAST:
            issues.append(f"  ⚠ Color '{name}' ({hexc}) low contrast on white: {cr:.2f} (need ≥{MIN_CONTRAST})")

    # 3. File integrity check
    expected_files = [
        "fig2_training_dynamics.png", "fig2_training_dynamics.pdf",
        "fig3_pca_embedding.png", "fig3_pca_embedding.pdf",
        "fig4_metrics_dashboard.png", "fig4_metrics_dashboard.pdf",
        "fig5_biological_validation.png", "fig5_biological_validation.pdf",
        "fig6_cfg_solver.png", "fig6_cfg_solver.pdf",
        "fig7_dimension_sampling.png", "fig7_dimension_sampling.pdf",
        "fig8_tables.png", "fig8_tables.pdf",
        "fig9_baseline_comparison.png", "fig9_baseline_comparison.pdf",
    ]
    for f in expected_files:
        fp = FIGURES / f
        if not fp.exists():
            issues.append(f"  ✗ Missing expected file: {f}")
        elif fp.stat().st_size < 5000:
            issues.append(f"  ⚠ Suspiciously small file: {f} ({fp.stat().st_size} bytes)")

    summary_fp = PROJECT / "figures/final_eval_summary.png"
    if not summary_fp.exists():
        issues.append("  ✗ Missing summary figure: figures/final_eval_summary.png")

    # 4. Stale / orphaned figures check
    all_files = set(f.name for f in FIGURES.iterdir() if f.is_file())
    expected_set = set(expected_files)
    orphaned = all_files - expected_set
    if orphaned:
        issues.append(f"  ⚠ Orphaned files in v5_publication/: {', '.join(sorted(orphaned))}")

    # 5. Colorblind simulation (simplified check for red-green confusion)
    r_rgb = _hex_to_rgb(COLORS['red'])
    g_rgb = _hex_to_rgb(COLORS['green'])
    # Simulate protanopia: red channel reduced
    r_sim = (r_rgb[0] * 0.2, r_rgb[1], r_rgb[2])
    g_sim = (g_rgb[0] * 0.2, g_rgb[1], g_rgb[2])
    cb_dist = _color_distance(r_sim, g_sim)
    if cb_dist < 0.3:
        issues.append(f"  ⚠ Red/Green may be indistinguishable for colorblind users (protanopia sim dist={cb_dist:.3f})")

    if issues:
        print(f"  Found {len(issues)} potential issue(s):")
        for iss in issues:
            print(iss)
    else:
        print("  ✓ No visual conflicts detected")
    print()
    return issues


# ═══════════════════════════════════════════════════════════════════
#  PARSE TRAINING LOGS
# ═══════════════════════════════════════════════════════════════════

def parse_clop_log(log_path):
    """Extract epoch-level metrics from CLOP training log."""
    epochs, train_loss, val_loss, val_acc, temp = [], [], [], [], []
    with open(log_path) as f:
        lines = f.readlines()
    
    i = 0
    epoch = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Train Loss:"):
            epoch += 1
            tl = float(line.split(":")[1].strip())
            vl = float(lines[i+1].strip().split(":")[1].strip())
            va = float(lines[i+2].strip().split(":")[1].strip())
            tp = float(lines[i+3].strip().split(":")[1].strip())
            epochs.append(epoch)
            train_loss.append(tl)
            val_loss.append(vl)
            val_acc.append(va)
            temp.append(tp)
            i += 4
        else:
            i += 1
    return {
        'epoch': np.array(epochs),
        'train_loss': np.array(train_loss),
        'val_loss': np.array(val_loss),
        'val_acc': np.array(val_acc),
        'temp': np.array(temp),
    }


def parse_dit_log(log_path):
    """Extract epoch-level metrics from DiT training log."""
    epochs, train_loss, val_loss, val_cosine = [], [], [], []
    with open(log_path) as f:
        lines = f.readlines()
    
    i = 0
    epoch = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Train Loss:"):
            epoch += 1
            tl = float(line.split(":")[1].strip())
            vl = float(lines[i+1].strip().split(":")[1].strip())
            vc = float(lines[i+2].strip().split(":")[1].strip())
            epochs.append(epoch)
            train_loss.append(tl)
            val_loss.append(vl)
            val_cosine.append(vc)
            i += 3
        else:
            i += 1
    return {
        'epoch': np.array(epochs),
        'train_loss': np.array(train_loss),
        'val_loss': np.array(val_loss),
        'val_cosine': np.array(val_cosine),
    }


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 2: TRAINING DYNAMICS (4 panels)
# ═══════════════════════════════════════════════════════════════════

def make_fig2_training_dynamics():
    print("  [Fig 2] Training Dynamics...")
    clop = parse_clop_log(PROJECT / "logs/clop_training_v5.2.log")
    dit = parse_dit_log(PROJECT / "logs/dit_training_v5.2.log")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # (a) CLOP Loss
    ax = axes[0, 0]
    ax.plot(clop['epoch'], clop['train_loss'], color=COLORS['blue'], linewidth=1.5, label='Train Loss')
    ax.plot(clop['epoch'], clop['val_loss'], color=COLORS['red'], linewidth=1.5, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('InfoNCE Loss')
    ax.set_title('(a) CLOP Contrastive Loss')
    ax.legend()
    ax.set_xlim(1, 200)
    
    # (b) CLOP Val Accuracy + Temperature
    ax = axes[0, 1]
    ax.plot(clop['epoch'], clop['val_acc'] * 100, color=COLORS['green'], linewidth=1.5, label='Val Accuracy')
    ax.axhline(y=100/256, color=COLORS['red'], linestyle='--', alpha=0.7, linewidth=1, label=f'Random (1/256 = {100/256:.2f}%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)', color=COLORS['green'])
    ax.set_title('(b) CLOP Validation Accuracy & Temperature')
    ax.set_xlim(1, 200)
    
    ax2 = ax.twinx()
    ax2.plot(clop['epoch'], clop['temp'], color=COLORS['orange'], linewidth=1.2, alpha=0.7, label='Temperature τ')
    ax2.set_ylabel('Temperature τ', color=COLORS['orange'])
    ax2.spines['right'].set_visible(True)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)
    
    # (c) DiT Loss
    ax = axes[1, 0]
    ax.plot(dit['epoch'], dit['train_loss'], color=COLORS['blue'], linewidth=1.5, label='Train Loss')
    ax.plot(dit['epoch'], dit['val_loss'], color=COLORS['red'], linewidth=1.5, label='Val Loss (EMA)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Flow Matching MSE Loss')
    ax.set_title('(c) DiT Flow Matching Loss')
    ax.legend()
    ax.set_xlim(1, 200)
    ax.set_yscale('log')
    
    # (d) DiT Val Cosine
    ax = axes[1, 1]
    ax.plot(dit['epoch'], dit['val_cosine'], color=COLORS['indigo'], linewidth=2)
    ax.fill_between(dit['epoch'], 0, dit['val_cosine'], alpha=0.1, color=COLORS['indigo'])
    ax.axhline(y=1.0, color=COLORS['grey'], linestyle=':', alpha=0.5, label='Perfect (1.0)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Velocity Cosine Similarity')
    ax.set_title('(d) DiT Velocity Prediction Quality')
    ax.set_xlim(1, 200)
    ax.set_ylim(0, 1.05)
    
    # Annotate final value
    final_cos = dit['val_cosine'][-1]
    ax.annotate(f'{final_cos:.4f}', xy=(200, final_cos), 
                xytext=(170, final_cos - 0.08),
                fontsize=10, fontweight='bold', color=COLORS['indigo'],
                arrowprops=dict(arrowstyle='->', color=COLORS['indigo'], lw=1.2))
    ax.legend(loc='lower right')
    
    fig.suptitle('Training Dynamics: CLOP (Stage 1) and DiT (Stage 2)', 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_training_dynamics.png", facecolor='white')
    fig.savefig(FIGURES / "fig2_training_dynamics.pdf", facecolor='white')
    plt.close()
    print("    ✓ fig2_training_dynamics.png/pdf")


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 3: UMAP / PCA EMBEDDING VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def make_fig3_embedding(cell_emb, text_labels, group_indices, valid_groups,
                        group_cond, dit, global_mean):
    print("  [Fig 3] Embedding Visualization...")
    
    N_GEN = 200
    TOP_K = 8  # Show top 8 groups
    vis_groups = valid_groups[:TOP_K]
    
    # Generate cells for visualization
    seed_all(42)
    gen_data = {}
    for g in vis_groups:
        cond_t = torch.tensor(group_cond[g], dtype=torch.float32, device=DEVICE)
        cond_batch = cond_t.unsqueeze(0).expand(N_GEN, -1)
        with torch.no_grad():
            gen_data[g] = dit.sample(cond_batch, cfg_scale=2.0, num_steps=10).cpu().numpy()
    
    # Collect real + gen
    rng = np.random.default_rng(42)
    real_pts, real_labels_vis = [], []
    for i, g in enumerate(vis_groups):
        idx = group_indices[g]
        n_sub = min(300, len(idx))
        sub = rng.choice(idx, n_sub, replace=False)
        real_pts.append(cell_emb[sub])
        real_labels_vis.extend([i] * n_sub)
    real_pts = np.vstack(real_pts)
    
    gen_pts = np.vstack([gen_data[g] for g in vis_groups])
    gen_labels_vis = np.repeat(np.arange(TOP_K), N_GEN)
    
    # PCA
    combined = np.vstack([real_pts - global_mean, gen_pts - global_mean])
    pca_2d = PCA(n_components=2, random_state=42)
    combined_2d = pca_2d.fit_transform(combined)
    n_real = len(real_pts)
    
    colors = plt.cm.tab10(np.linspace(0, 1, TOP_K))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # (a) All real (gray) vs all generated (colored)
    ax = axes[0]
    # Background: other real cells
    bg_idx = rng.choice(len(cell_emb), min(5000, len(cell_emb)), replace=False)
    bg_pca = pca_2d.transform(cell_emb[bg_idx] - global_mean)
    ax.scatter(bg_pca[:, 0], bg_pca[:, 1], c='lightgray', alpha=0.15, s=3, rasterized=True)
    
    # Generated cells colored
    for i in range(TOP_K):
        mask = gen_labels_vis == i
        ax.scatter(combined_2d[n_real:][mask, 0], combined_2d[n_real:][mask, 1],
                   c=[colors[i]], alpha=0.6, s=12, marker='^', label=f'Gen Group {i+1}')
    
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    ax.set_title('(a) Real Data (gray) vs Generated (colored)')
    ax.legend(fontsize=7, ncol=2, loc='upper right', markerscale=1.5)
    
    # (b) Real (○) vs Generated (△) per group
    ax = axes[1]
    for i in range(TOP_K):
        mask_r = np.array(real_labels_vis) == i
        mask_g = gen_labels_vis == i
        ax.scatter(combined_2d[:n_real][mask_r, 0], combined_2d[:n_real][mask_r, 1],
                   c=[colors[i]], alpha=0.2, s=6, marker='o')
        ax.scatter(combined_2d[n_real:][mask_g, 0], combined_2d[n_real:][mask_g, 1],
                   c=[colors[i]], alpha=0.5, s=14, marker='^')
    
    ax.scatter([], [], c='gray', marker='o', s=10, alpha=0.5, label='Real')
    ax.scatter([], [], c='gray', marker='^', s=14, alpha=0.7, label='Generated')
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    ax.set_title('(b) Per-Group Real (○) vs Generated (△)')
    ax.legend(fontsize=8, loc='upper right')
    
    fig.suptitle('PCA Visualization of Real and Generated Cell Embeddings (CFG=2.0, Euler-10)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_pca_embedding.png", facecolor='white')
    fig.savefig(FIGURES / "fig3_pca_embedding.pdf", facecolor='white')
    plt.close()
    print("    ✓ fig3_pca_embedding.png/pdf")


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 4: COMPREHENSIVE METRICS DASHBOARD (6 panels)
# ═══════════════════════════════════════════════════════════════════

def make_fig4_metrics_dashboard(results_data):
    print("  [Fig 4] Metrics Dashboard...")
    
    res = results_data['results']
    real = results_data['real_data_baseline']
    random_chance = real['random_chance']
    
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.35)
    
    # (a) KNN Classification Accuracy
    ax = fig.add_subplot(gs[0, 0])
    methods = ['CFG=3.0\nEuler', 'CFG=2.0\nEuler', 'CFG=1.0\nEuler',
               'CFG=1.0\nMidpoint', 'CFG=0.0', 'Gaussian']
    keys = ['CFG=3.0 Euler-10', 'CFG=2.0 Euler-10', 'CFG=1.0 Euler-10',
            'CFG=1.0 Midpoint-10', 'CFG=0.0 Euler-10', 'Gaussian']
    knn1 = [res[k]['knn_top1'] for k in keys]
    knn5 = [res[k]['knn_top5'] for k in keys]
    
    x = np.arange(len(methods))
    w = 0.35
    ax.bar(x - w/2, knn1, w, color=COLORS['blue'], alpha=0.85, label='Top-1')
    ax.bar(x + w/2, knn5, w, color=COLORS['green'], alpha=0.85, label='Top-5')
    ax.axhline(y=random_chance, color='red', linestyle='--', alpha=0.7, linewidth=1,
               label=f'Random ({random_chance:.3f})')
    ax.axhline(y=real['knn_top1'], color=COLORS['amber'], linestyle='--', alpha=0.7,
               linewidth=1, label=f'Real ({real["knn_top1"]:.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title('(a) KNN Classification Accuracy')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, min(1.0, real['knn_top1'] + 0.1))
    
    # (b) Steering Accuracy
    ax = fig.add_subplot(gs[0, 1])
    steer_keys = ['CFG=3.0 Euler-10', 'CFG=2.0 Euler-10', 'CFG=1.0 Euler-10',
                  'CFG=1.0 Midpoint-10', 'CFG=1.0 Euler-50', 'CFG=0.0 Euler-10', 'Gaussian']
    steer_labels = ['CFG=3.0\nEuler-10', 'CFG=2.0\nEuler-10', 'CFG=1.0\nEuler-10',
                    'CFG=1.0\nMid-10', 'CFG=1.0\nEuler-50', 'CFG=0.0', 'Gaussian']
    steer_vals = [res[k]['steering'] for k in steer_keys]
    
    bar_colors = [COLORS['blue'] if v > 0.6 else COLORS['red'] for v in steer_vals]
    ax.bar(np.arange(len(steer_labels)), steer_vals, color=bar_colors, alpha=0.85)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
    ax.set_xticks(np.arange(len(steer_labels)))
    ax.set_xticklabels(steer_labels, fontsize=7)
    ax.set_ylabel('Steering Accuracy')
    ax.set_title('(b) Cross-Group Steering Accuracy')
    ax.set_ylim(0.3, 0.9)
    ax.legend(fontsize=7)
    
    # (c) Diversity Ratio
    ax = fig.add_subplot(gs[0, 2])
    div_methods = ['CFG=3.0\nEuler', 'CFG=2.0\nEuler', 'CFG=1.0\nEuler',
                   'CFG=3.0\nMid', 'CFG=1.0\nMid', 'CFG=0.0', 'Gaussian']
    div_keys = ['CFG=3.0 Euler-10', 'CFG=2.0 Euler-10', 'CFG=1.0 Euler-10',
                'CFG=3.0 Midpoint-10', 'CFG=1.0 Midpoint-10', 'CFG=0.0 Euler-10', 'Gaussian']
    divr = [res[k]['diversity_ratio'] for k in div_keys]
    
    div_colors = []
    for d in divr:
        if 0.8 <= d <= 1.2:
            div_colors.append(COLORS['green'])
        elif d < 0.8:
            div_colors.append(COLORS['orange'])
        else:
            div_colors.append(COLORS['red'])
    ax.bar(np.arange(len(div_methods)), divr, color=div_colors, alpha=0.85)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Ideal (1.0)')
    ax.set_xticks(np.arange(len(div_methods)))
    ax.set_xticklabels(div_methods, fontsize=7)
    ax.set_ylabel('Diversity Ratio')
    ax.set_title('(c) Within-Group Diversity Ratio')
    ax.legend(fontsize=7)
    
    # (d) Linear Accuracy vs KNN Accuracy
    ax = fig.add_subplot(gs[1, 0])
    cond_keys = [k for k in res.keys() if k not in ['Gaussian', 'CFG=0.0 Euler-10']]
    knn_vals = [res[k]['knn_top1'] for k in cond_keys]
    lin_vals = [res[k]['linear_acc'] for k in cond_keys]
    
    ax.scatter(knn_vals, lin_vals, c=COLORS['indigo'], s=60, alpha=0.8, zorder=3)
    for i, k in enumerate(cond_keys):
        short = k.replace('CFG=', '').replace(' Euler-10', 'E').replace(' Midpoint-10', 'M').replace(' Euler-50', 'E50')
        ax.annotate(short, (knn_vals[i], lin_vals[i]), fontsize=7, 
                    textcoords="offset points", xytext=(5, 5))
    
    # Add baselines
    ax.scatter([res['CFG=0.0 Euler-10']['knn_top1']], [res['CFG=0.0 Euler-10']['linear_acc']], 
               c=COLORS['red'], s=80, marker='X', zorder=3, label='Unconditional')
    ax.scatter([res['Gaussian']['knn_top1']], [res['Gaussian']['linear_acc']], 
               c=COLORS['grey'], s=80, marker='D', zorder=3, label='Gaussian')
    ax.scatter([real['knn_top1']], [real['linear_acc']], 
               c=COLORS['amber'], s=100, marker='*', zorder=3, label='Real Data')
    
    ax.set_xlabel('KNN Top-1 Accuracy')
    ax.set_ylabel('Linear Classifier Accuracy')
    ax.set_title('(d) KNN vs Linear Separability')
    ax.legend(fontsize=7)
    
    # (e) Centroid Cosine Similarity  
    ax = fig.add_subplot(gs[1, 1])
    cos_keys = ['CFG=3.0 Euler-10', 'CFG=2.0 Euler-10', 'CFG=1.0 Euler-10',
                'CFG=1.0 Midpoint-10', 'CFG=0.0 Euler-10', 'Gaussian']
    cos_labels = ['CFG=3.0', 'CFG=2.0', 'CFG=1.0', 'CFG=1.0\nMid', 'CFG=0.0', 'Gauss']
    cos_means = [res[k]['centroid_cos_centered'] for k in cos_keys]
    cos_stds = [res[k]['centroid_cos_std'] for k in cos_keys]
    
    x = np.arange(len(cos_labels))
    colors_cos = [COLORS['blue']]*4 + [COLORS['red'], COLORS['grey']]
    ax.bar(x, cos_means, yerr=cos_stds, color=colors_cos, alpha=0.8,
           capsize=3, error_kw={'linewidth': 1})
    ax.set_xticks(x)
    ax.set_xticklabels(cos_labels, fontsize=8)
    ax.set_ylabel('Centered Centroid Cosine')
    ax.set_title('(e) Centroid Alignment (centered)')
    
    # (f) Improvement multiplier over random
    ax = fig.add_subplot(gs[1, 2])
    mult_keys = ['CFG=3.0 Euler-10', 'CFG=2.0 Euler-10', 'CFG=1.0 Euler-10',
                 'CFG=1.0 Midpoint-10', 'CFG=1.0 Euler-50', 'CFG=0.0 Euler-10', 'Gaussian']
    mult_labels = ['CFG=3.0', 'CFG=2.0', 'CFG=1.0', 'CFG=1.0\nMid', 'CFG=1.0\nE-50', 'CFG=0', 'Gauss']
    mults = [res[k]['knn_top1'] / random_chance for k in mult_keys]
    
    mult_colors = [COLORS['blue']]*5 + [COLORS['red'], COLORS['grey']]
    ax.bar(np.arange(len(mult_labels)), mults, color=mult_colors, alpha=0.85)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Random (1×)')
    ax.set_xticks(np.arange(len(mult_labels)))
    ax.set_xticklabels(mult_labels, fontsize=7)
    ax.set_ylabel('KNN / Random Chance (×)')
    ax.set_title('(f) Improvement Over Random')
    ax.legend(fontsize=7)
    
    # Add text annotation for best
    best_mult = max(mults[:5])
    ax.annotate(f'{best_mult:.0f}×', xy=(np.argmax(mults[:5]), best_mult),
                fontsize=12, fontweight='bold', ha='center', va='bottom',
                color=COLORS['blue'])
    
    fig.suptitle('CLOP-DiT: Comprehensive Generation Evaluation (220,304 cells, 100 groups)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_metrics_dashboard.png", facecolor='white')
    fig.savefig(FIGURES / "fig4_metrics_dashboard.pdf", facecolor='white')
    plt.close()
    print("    ✓ fig4_metrics_dashboard.png/pdf")


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 5: BIOLOGICAL VALIDATION (4 panels)
# ═══════════════════════════════════════════════════════════════════

def make_fig5_biological_validation(cell_emb, text_labels, group_indices, 
                                     valid_groups, group_cond, dit, global_mean):
    print("  [Fig 5] Biological Validation...")
    
    N_GEN = 200
    TOP_K = 10
    vis_groups = valid_groups[:TOP_K]
    
    seed_all(42)
    gen_data = {}
    for g in vis_groups:
        cond_t = torch.tensor(group_cond[g], dtype=torch.float32, device=DEVICE)
        cond_batch = cond_t.unsqueeze(0).expand(N_GEN, -1)
        with torch.no_grad():
            gen_data[g] = dit.sample(cond_batch, cfg_scale=2.0, num_steps=10).cpu().numpy()
    
    rng = np.random.default_rng(42)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # (a) PCA showing type separation of generated cells
    ax = axes[0, 0]
    gen_all = np.vstack([gen_data[g] for g in vis_groups])
    gen_labels = np.repeat(np.arange(TOP_K), N_GEN)
    gen_centered = gen_all - global_mean
    pca_2d = PCA(n_components=2, random_state=42)
    gen_2d = pca_2d.fit_transform(gen_centered)
    
    colors = plt.cm.tab10(np.linspace(0, 1, TOP_K))
    for i in range(TOP_K):
        mask = gen_labels == i
        ax.scatter(gen_2d[mask, 0], gen_2d[mask, 1], c=[colors[i]], s=10, alpha=0.5,
                   label=f'Group {i+1}')
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    ax.set_title('(a) PCA of Generated Cells (CFG=2.0)')
    ax.legend(fontsize=6, ncol=2, loc='upper right', markerscale=2)
    
    # (b) Inter-group cosine similarity heatmap
    ax = axes[0, 1]
    gen_centroids = np.array([gen_data[g].mean(0) - global_mean for g in vis_groups])
    gen_centroids_norm = gen_centroids / (np.linalg.norm(gen_centroids, axis=1, keepdims=True) + 1e-8)
    cos_matrix = gen_centroids_norm @ gen_centroids_norm.T
    
    im = ax.imshow(cos_matrix, cmap='RdBu_r', vmin=-0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(TOP_K))
    ax.set_yticks(range(TOP_K))
    ax.set_xticklabels([f'G{i+1}' for i in range(TOP_K)], fontsize=8)
    ax.set_yticklabels([f'G{i+1}' for i in range(TOP_K)], fontsize=8)
    ax.set_title('(b) Generated Inter-Group Cosine Similarity')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Add values
    for i in range(TOP_K):
        for j in range(TOP_K):
            color = 'white' if abs(cos_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{cos_matrix[i,j]:.2f}', ha='center', va='center', 
                    fontsize=5, color=color)
    
    # (c) Intra-group diversity: gen vs real
    ax = axes[1, 0]
    real_divs = []
    gen_divs = []
    for g in vis_groups:
        real_divs.append(cell_emb[group_indices[g]].std(0).mean())
        gen_divs.append(gen_data[g].std(0).mean())
    
    x = np.arange(TOP_K)
    ax.bar(x - 0.2, real_divs, 0.35, color=COLORS['blue'], alpha=0.8, label='Real')
    ax.bar(x + 0.2, gen_divs, 0.35, color=COLORS['orange'], alpha=0.8, label='Generated')
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{i+1}' for i in range(TOP_K)], fontsize=8)
    ax.set_ylabel('Mean Std Dev')
    ax.set_title('(c) Intra-Group Diversity: Real vs Generated')
    ax.legend(fontsize=8)
    
    # (d) Conditioning fidelity: cosine to target centroid
    ax = axes[1, 1]
    fidelities = []
    wrong_fidelities = []
    for i, g in enumerate(vis_groups):
        target_centroid = cell_emb[group_indices[g]].mean(0) - global_mean
        target_norm = target_centroid / (np.linalg.norm(target_centroid) + 1e-8)
        
        gen_centered_g = gen_data[g] - global_mean
        gen_norm = gen_centered_g / (np.linalg.norm(gen_centered_g, axis=1, keepdims=True) + 1e-8)
        cos_to_target = gen_norm @ target_norm
        fidelities.append(cos_to_target.mean())
        
        # Random wrong centroid
        other_g = vis_groups[(i + 1) % TOP_K]
        wrong_centroid = cell_emb[group_indices[other_g]].mean(0) - global_mean
        wrong_norm = wrong_centroid / (np.linalg.norm(wrong_centroid) + 1e-8)
        cos_to_wrong = gen_norm @ wrong_norm
        wrong_fidelities.append(cos_to_wrong.mean())
    
    x = np.arange(TOP_K)
    ax.bar(x - 0.2, fidelities, 0.35, color=COLORS['green'], alpha=0.8, label='To Target')
    ax.bar(x + 0.2, wrong_fidelities, 0.35, color=COLORS['red'], alpha=0.8, label='To Wrong')
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{i+1}' for i in range(TOP_K)], fontsize=8)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('(d) Conditioning Fidelity: Target vs Wrong Centroid')
    ax.legend(fontsize=8)
    
    fig.suptitle('Biological Validation of Generated Cell Embeddings',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig5_biological_validation.png", facecolor='white')
    fig.savefig(FIGURES / "fig5_biological_validation.pdf", facecolor='white')
    plt.close()
    print("    ✓ fig5_biological_validation.png/pdf")


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 7: DIMENSION & SAMPLING ANALYSIS (4 panels)
# ═══════════════════════════════════════════════════════════════════

def make_fig7_dimension_sampling(cell_emb, group_cond, valid_groups, dit, global_mean):
    print("  [Fig 7] Dimension & Sampling Analysis...")
    
    N_GEN = 200
    seed_all(42)
    rng = np.random.default_rng(42)
    
    # Generate cells
    gen_all = []
    for g in valid_groups[:50]:
        cond_t = torch.tensor(group_cond[g], dtype=torch.float32, device=DEVICE)
        cond_batch = cond_t.unsqueeze(0).expand(N_GEN, -1)
        with torch.no_grad():
            gen_all.append(dit.sample(cond_batch, cfg_scale=2.0, num_steps=10).cpu().numpy())
    gen_all = np.vstack(gen_all)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # (a) Per-dimension mean comparison (real vs gen)
    ax = axes[0, 0]
    real_means = cell_emb.mean(0)
    gen_means = gen_all.mean(0)
    
    ax.scatter(real_means, gen_means, s=8, alpha=0.6, c=COLORS['blue'], rasterized=True)
    
    # Perfect correlation line
    lims = [min(real_means.min(), gen_means.min()), max(real_means.max(), gen_means.max())]
    ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=1, label='y = x')
    
    # Correlation
    corr = np.corrcoef(real_means, gen_means)[0, 1]
    ax.set_xlabel('Real Mean')
    ax.set_ylabel('Generated Mean')
    ax.set_title(f'(a) Per-Dimension Mean Correlation (r = {corr:.4f})')
    ax.legend(fontsize=8)
    
    # (b) Per-dimension std comparison
    ax = axes[0, 1]
    real_stds = cell_emb.std(0)
    gen_stds = gen_all.std(0)
    
    ax.scatter(real_stds, gen_stds, s=8, alpha=0.6, c=COLORS['purple'], rasterized=True)
    lims = [min(real_stds.min(), gen_stds.min()), max(real_stds.max(), gen_stds.max())]
    ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=1, label='y = x')
    
    corr_std = np.corrcoef(real_stds, gen_stds)[0, 1]
    ax.set_xlabel('Real Std Dev')
    ax.set_ylabel('Generated Std Dev')
    ax.set_title(f'(b) Per-Dimension Std Correlation (r = {corr_std:.4f})')
    ax.legend(fontsize=8)
    
    # (c) ODE sampling trajectory visualization
    ax = axes[1, 0]
    # Generate a single trajectory
    g = valid_groups[0]
    cond_t = torch.tensor(group_cond[g], dtype=torch.float32, device=DEVICE)
    cond_batch = cond_t.unsqueeze(0).expand(5, -1)
    
    # Manually trace ODE trajectory
    z0 = torch.randn(5, 512, device=DEVICE)
    trajectory = [z0.cpu().numpy()]
    n_vis_steps = 20
    dt = 1.0 / n_vis_steps
    z_t = z0.clone()
    
    with torch.no_grad():
        for step in range(n_vis_steps):
            t_val = step * dt
            t_tensor = torch.full((5,), t_val, device=DEVICE)
            # Get unconditional and conditional velocity
            v_uncond = dit.forward(z_t, t_tensor, torch.zeros_like(cond_batch))
            v_cond = dit.forward(z_t, t_tensor, cond_batch)
            v = v_uncond + 2.0 * (v_cond - v_uncond)  # CFG=2.0
            z_t = z_t + v * dt
            trajectory.append(z_t.cpu().numpy())
    
    trajectory = np.array(trajectory)  # (steps+1, 5, 512)
    # PCA on the entire trajectory
    traj_flat = trajectory.reshape(-1, 512)
    pca_traj = PCA(n_components=2, random_state=42)
    traj_2d = pca_traj.fit_transform(traj_flat).reshape(n_vis_steps+1, 5, 2)
    
    traj_colors = plt.cm.viridis(np.linspace(0, 1, n_vis_steps+1))
    for sample_i in range(5):
        for step in range(n_vis_steps):
            ax.plot([traj_2d[step, sample_i, 0], traj_2d[step+1, sample_i, 0]],
                    [traj_2d[step, sample_i, 1], traj_2d[step+1, sample_i, 1]],
                    color=traj_colors[step], linewidth=1.2, alpha=0.7)
        ax.scatter(traj_2d[0, sample_i, 0], traj_2d[0, sample_i, 1], 
                   c='blue', s=30, marker='o', zorder=5)
        ax.scatter(traj_2d[-1, sample_i, 0], traj_2d[-1, sample_i, 1],
                   c='red', s=30, marker='*', zorder=5)
    
    ax.scatter([], [], c='blue', marker='o', s=30, label='z₀ (noise)')
    ax.scatter([], [], c='red', marker='*', s=30, label='z₁ (cell)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('(c) ODE Sampling Trajectory (CFG=2.0)')
    ax.legend(fontsize=8)
    
    # (d) Norm comparison
    ax = axes[1, 1]
    real_norms = np.linalg.norm(cell_emb[rng.choice(len(cell_emb), 5000, replace=False)], axis=1)
    gen_norms = np.linalg.norm(gen_all[:5000], axis=1)
    
    ax.hist(real_norms, bins=60, alpha=0.6, color=COLORS['blue'], density=True, label=f'Real (μ={real_norms.mean():.2f})')
    ax.hist(gen_norms, bins=60, alpha=0.6, color=COLORS['orange'], density=True, label=f'Generated (μ={gen_norms.mean():.2f})')
    ax.set_xlabel('L2 Norm')
    ax.set_ylabel('Density')
    ax.set_title('(d) Embedding Norm Distribution')
    ax.legend(fontsize=8)
    
    fig.suptitle('Dimension and Sampling Analysis',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig7_dimension_sampling.png", facecolor='white')
    fig.savefig(FIGURES / "fig7_dimension_sampling.pdf", facecolor='white')
    plt.close()
    print("    ✓ fig7_dimension_sampling.png/pdf")


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 6: CFG SWEEP + SOLVER COMPARISON (6 panels)
# ═══════════════════════════════════════════════════════════════════

def make_fig6_cfg_sweep_and_solver(results_data):
    print("  [Fig 6] CFG Sweep & Solver Comparison...")
    
    sweep_data = results_data['cfg_sweep']
    res = results_data['results']
    
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)
    
    # Top row: CFG sweep curves
    metrics_to_plot = [
        ('knn1', 'KNN Top-1 Accuracy ↑'),
        ('steering', 'Steering Accuracy ↑'),
        ('div_ratio', 'Diversity Ratio (→1.0)'),
    ]
    
    step_styles = [(10, '-o', 'Steps=10'), (20, '--s', 'Steps=20'), (50, ':^', 'Steps=50')]
    step_colors = [COLORS['blue'], COLORS['orange'], COLORS['green']]
    
    for ax_idx, (metric, title) in enumerate(metrics_to_plot):
        ax = fig.add_subplot(gs[0, ax_idx])
        for (step_val, style, label), color in zip(step_styles, step_colors):
            sub = sorted([r for r in sweep_data if r['steps'] == step_val], key=lambda r: r['cfg'])
            cfgs = [r['cfg'] for r in sub]
            vals = [r[metric] for r in sub]
            ax.plot(cfgs, vals, style, color=color, label=label, markersize=5, linewidth=1.5)
        
        ax.set_xlabel('CFG Scale')
        ax.set_ylabel(title.split('↑')[0].split('↓')[0].split('(')[0].strip())
        panel_letter = chr(ord('a') + ax_idx)
        ax.set_title(f'({panel_letter}) {title}')
        ax.legend(fontsize=7)
        
        if metric == 'div_ratio':
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.4)
        if metric == 'steering':
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4)
    
    # Bottom row: Euler vs Midpoint comparison
    compare_cfgs = [1.0, 2.0, 3.0]
    euler_keys = [f"CFG={c:.1f} Euler-10" for c in compare_cfgs]
    mid_keys = [f"CFG={c:.1f} Midpoint-10" for c in compare_cfgs]
    
    solver_metrics = [
        ('knn_top1', 'KNN Top-1 Accuracy'),
        ('steering', 'Steering Accuracy'),
        ('diversity_ratio', 'Diversity Ratio'),
    ]
    
    for ax_idx, (m, title) in enumerate(solver_metrics):
        ax = fig.add_subplot(gs[1, ax_idx])
        euler_vals = [res[k][m] for k in euler_keys]
        mid_vals = [res[k][m] for k in mid_keys]
        
        x = np.arange(len(compare_cfgs))
        ax.bar(x - 0.18, euler_vals, 0.32, color=COLORS['blue'], alpha=0.85, label='Euler')
        ax.bar(x + 0.18, mid_vals, 0.32, color=COLORS['orange'], alpha=0.85, label='Midpoint')
        ax.set_xticks(x)
        ax.set_xticklabels([f'CFG={c}' for c in compare_cfgs])
        panel_letter = chr(ord('d') + ax_idx)
        ax.set_title(f'({panel_letter}) {title}')
        ax.legend(fontsize=8)
        
        if m == 'diversity_ratio':
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.4)
        if m == 'steering':
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4)
    
    fig.suptitle('CFG Scale Analysis & ODE Solver Comparison',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig6_cfg_solver.png", facecolor='white')
    fig.savefig(FIGURES / "fig6_cfg_solver.pdf", facecolor='white')
    plt.close()
    print("    ✓ fig6_cfg_solver.png/pdf")


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 8: COMPARISON & ABLATION TABLES (rendered as figure)
# ═══════════════════════════════════════════════════════════════════

def make_fig8_tables(results_data, baseline_data=None):
    print("  [Fig 8] Summary Tables...")
    
    res = results_data['results']
    real = results_data['real_data_baseline']
    rc = real['random_chance']
    
    # Determine layout: 3 tables if baselines available, 2 otherwise
    has_baselines = baseline_data is not None
    n_tables = 3 if has_baselines else 2
    fig, axes = plt.subplots(n_tables, 1, figsize=(14, 4.5 * n_tables))
    if n_tables == 2:
        axes = list(axes)
    
    # TABLE I: Main Generation Results
    ax = axes[0]
    ax.axis('off')
    
    headers = ['Method', 'KNN-1 ↑', 'KNN-5 ↑', 'Steering ↑', 'DivR (→1)', 'LinAcc ↑', 'FD', 'KNN/Rand']
    rows = [
        ['Real Data (holdout)', f'{real["knn_top1"]:.3f}', f'{real["knn_top5"]:.3f}', '—', '1.000', f'{real["linear_acc"]:.3f}', '0.00', f'{real["knn_top1"]/rc:.0f}×'],
        ['CFG=2.0 Euler-10', f'{res["CFG=2.0 Euler-10"]["knn_top1"]:.3f}', f'{res["CFG=2.0 Euler-10"]["knn_top5"]:.3f}', 
         f'{res["CFG=2.0 Euler-10"]["steering"]:.3f}', f'{res["CFG=2.0 Euler-10"]["diversity_ratio"]:.3f}',
         f'{res["CFG=2.0 Euler-10"]["linear_acc"]:.3f}', f'{res["CFG=2.0 Euler-10"]["fd"]:.2f}',
         f'{res["CFG=2.0 Euler-10"]["knn_top1"]/rc:.0f}×'],
        ['CFG=3.0 Euler-10', f'{res["CFG=3.0 Euler-10"]["knn_top1"]:.3f}', f'{res["CFG=3.0 Euler-10"]["knn_top5"]:.3f}',
         f'{res["CFG=3.0 Euler-10"]["steering"]:.3f}', f'{res["CFG=3.0 Euler-10"]["diversity_ratio"]:.3f}',
         f'{res["CFG=3.0 Euler-10"]["linear_acc"]:.3f}', f'{res["CFG=3.0 Euler-10"]["fd"]:.2f}',
         f'{res["CFG=3.0 Euler-10"]["knn_top1"]/rc:.0f}×'],
        ['CFG=1.0 Midpoint-10', f'{res["CFG=1.0 Midpoint-10"]["knn_top1"]:.3f}', f'{res["CFG=1.0 Midpoint-10"]["knn_top5"]:.3f}',
         f'{res["CFG=1.0 Midpoint-10"]["steering"]:.3f}', f'{res["CFG=1.0 Midpoint-10"]["diversity_ratio"]:.3f}',
         f'{res["CFG=1.0 Midpoint-10"]["linear_acc"]:.3f}', f'{res["CFG=1.0 Midpoint-10"]["fd"]:.2f}',
         f'{res["CFG=1.0 Midpoint-10"]["knn_top1"]/rc:.0f}×'],
        ['CFG=1.0 Euler-50', f'{res["CFG=1.0 Euler-50"]["knn_top1"]:.3f}', f'{res["CFG=1.0 Euler-50"]["knn_top5"]:.3f}',
         f'{res["CFG=1.0 Euler-50"]["steering"]:.3f}', f'{res["CFG=1.0 Euler-50"]["diversity_ratio"]:.3f}',
         f'{res["CFG=1.0 Euler-50"]["linear_acc"]:.3f}', f'{res["CFG=1.0 Euler-50"]["fd"]:.2f}',
         f'{res["CFG=1.0 Euler-50"]["knn_top1"]/rc:.0f}×'],
        ['CFG=0.0 (Unconditional)', f'{res["CFG=0.0 Euler-10"]["knn_top1"]:.3f}', f'{res["CFG=0.0 Euler-10"]["knn_top5"]:.3f}',
         f'{res["CFG=0.0 Euler-10"]["steering"]:.3f}', f'{res["CFG=0.0 Euler-10"]["diversity_ratio"]:.3f}',
         f'{res["CFG=0.0 Euler-10"]["linear_acc"]:.3f}', f'{res["CFG=0.0 Euler-10"]["fd"]:.2f}',
         f'{res["CFG=0.0 Euler-10"]["knn_top1"]/rc:.0f}×'],
        ['Gaussian N(μ,Σ)', f'{res["Gaussian"]["knn_top1"]:.3f}', f'{res["Gaussian"]["knn_top5"]:.3f}',
         f'{res["Gaussian"]["steering"]:.3f}', f'{res["Gaussian"]["diversity_ratio"]:.3f}',
         f'{res["Gaussian"]["linear_acc"]:.3f}', f'{res["Gaussian"]["fd"]:.2f}',
         f'{res["Gaussian"]["knn_top1"]/rc:.0f}×'],
    ]
    
    table1 = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 1.5)
    
    # Style headers
    for j in range(len(headers)):
        table1[0, j].set_facecolor('#1565C0')
        table1[0, j].set_text_props(color='white', fontweight='bold')
    
    # Highlight best conditioned row
    for j in range(len(headers)):
        table1[1, j].set_facecolor('#E3F2FD')  # Real data row
        table1[2, j].set_facecolor('#E8F5E9')  # Best accuracy
    
    ax.set_title('TABLE I: Generation Quality Across Configurations (100 eval groups, 200 cells/group)',
                 fontsize=12, fontweight='bold', pad=15)
    
    # TABLE III: Phase 2 Decoder Architecture Comparison (if available)
    ax_idx = 1
    if has_baselines:
        ax = axes[ax_idx]
        ax.axis('off')

        # New format: oracle_baselines, feed_forward_decoders, diffusion_decoders
        oracle = baseline_data.get('oracle_baselines', {})
        ff_dec = baseline_data.get('feed_forward_decoders', {})
        diff_dec = baseline_data.get('diffusion_decoders', {})

        headers3 = ['Decoder', 'Type', 'Params', 'KNN-1 ↑', 'DivR (→1)', 'FD_g ↓', 'r ↑', 'DS ↑']

        def _row(name, dtype, params, d):
            return [name, dtype, params, f'{d["knn_top1"]:.3f}',
                    f'{d["diversity_ratio"]:.3f}', f'{d["per_group_fd"]:.2f}',
                    f'{d["mean_corr"]:.4f}', f'{d["downstream_acc"]:.3f}']

        rows3 = [
            ['Real Data (holdout)', '—', '—', f'{real["knn_top1"]:.3f}', '1.000', '0.00', '1.0000', '—'],
            _row('Per-Type Gaussian', 'Oracle', '—', oracle['per_type_gaussian']),
            _row('Retrieval + Jitter', 'Oracle', '—', oracle['retrieval_jitter']),
            _row('Conditional VAE', 'FF', '1.5M', ff_dec['conditional_vae']),
            _row('Conditional GAN', 'FF', '1.0M', ff_dec['conditional_gan']),
            _row('Flow MLP (Euler)', 'Diff', '1.5M', diff_dec['flow_mlp_euler']),
            _row('CLOP-DiT (CFG=2.0)', 'Diff', '22.1M', diff_dec['clop_dit_best_acc']),
            _row('CLOP-DiT (CFG=1.0 M)', 'Diff', '22.1M', diff_dec['clop_dit_balanced']),
        ]

        table3 = ax.table(cellText=rows3, colLabels=headers3, loc='center', cellLoc='center')
        table3.auto_set_font_size(False)
        table3.set_fontsize(8.5)
        table3.scale(1, 1.5)

        for j in range(len(headers3)):
            table3[0, j].set_facecolor('#1565C0')
            table3[0, j].set_text_props(color='white', fontweight='bold')
        # Highlight oracle rows (light blue)
        for row_idx in [1, 2, 3]:
            for j in range(len(headers3)):
                table3[row_idx, j].set_facecolor('#E3F2FD')
        # Highlight CLOP-DiT rows (light green)
        for row_idx in [7, 8]:
            for j in range(len(headers3)):
                table3[row_idx, j].set_facecolor('#E8F5E9')

        ax.set_title('TABLE III: Phase 2 Decoder Architecture Comparison',
                     fontsize=12, fontweight='bold', pad=15)
        ax_idx += 1
    
    # TABLE II: Ablation-like comparison (conditioned vs unconditioned)
    ax = axes[ax_idx]
    ax.axis('off')
    
    headers2 = ['Comparison', 'KNN-1', 'Steering', 'DivR', 'LinAcc', 'CtrCos', 'Verdict']
    rows2 = [
        ['Best Conditioned (CFG=2.0)', f'{res["CFG=2.0 Euler-10"]["knn_top1"]:.3f}',
         f'{res["CFG=2.0 Euler-10"]["steering"]:.3f}', f'{res["CFG=2.0 Euler-10"]["diversity_ratio"]:.3f}',
         f'{res["CFG=2.0 Euler-10"]["linear_acc"]:.3f}', f'{res["CFG=2.0 Euler-10"]["centroid_cos_centered"]:.3f}',
         '37× random'],
        ['Best Balanced (CFG=1.0 Mid)', f'{res["CFG=1.0 Midpoint-10"]["knn_top1"]:.3f}',
         f'{res["CFG=1.0 Midpoint-10"]["steering"]:.3f}', f'{res["CFG=1.0 Midpoint-10"]["diversity_ratio"]:.3f}',
         f'{res["CFG=1.0 Midpoint-10"]["linear_acc"]:.3f}', f'{res["CFG=1.0 Midpoint-10"]["centroid_cos_centered"]:.3f}',
         '29× random, ideal DivR'],
        ['No CFG (CFG=0.0)', f'{res["CFG=0.0 Euler-10"]["knn_top1"]:.3f}',
         f'{res["CFG=0.0 Euler-10"]["steering"]:.3f}', f'{res["CFG=0.0 Euler-10"]["diversity_ratio"]:.3f}',
         f'{res["CFG=0.0 Euler-10"]["linear_acc"]:.3f}', f'{res["CFG=0.0 Euler-10"]["centroid_cos_centered"]:.3f}',
         '= random'],
        ['Gaussian Baseline', f'{res["Gaussian"]["knn_top1"]:.3f}',
         f'{res["Gaussian"]["steering"]:.3f}', f'{res["Gaussian"]["diversity_ratio"]:.3f}',
         f'{res["Gaussian"]["linear_acc"]:.3f}', f'{res["Gaussian"]["centroid_cos_centered"]:.3f}',
         '= random'],
        ['Euler vs Midpoint (CFG=1.0)', 
         f'{res["CFG=1.0 Euler-10"]["knn_top1"]:.3f} vs {res["CFG=1.0 Midpoint-10"]["knn_top1"]:.3f}',
         f'{res["CFG=1.0 Euler-10"]["steering"]:.3f} vs {res["CFG=1.0 Midpoint-10"]["steering"]:.3f}',
         f'{res["CFG=1.0 Euler-10"]["diversity_ratio"]:.3f} vs {res["CFG=1.0 Midpoint-10"]["diversity_ratio"]:.3f}',
         '—', '—', 'Midpoint: +24% DivR'],
    ]
    
    table2 = ax.table(cellText=rows2, colLabels=headers2, loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.5)
    
    for j in range(len(headers2)):
        table2[0, j].set_facecolor('#1565C0')
        table2[0, j].set_text_props(color='white', fontweight='bold')
    
    for j in range(len(headers2)):
        table2[1, j].set_facecolor('#E8F5E9')  # Best row
        table2[3, j].set_facecolor('#FFEBEE')  # Bad rows
        table2[4, j].set_facecolor('#FFEBEE')
    
    ax.set_title('TABLE II: Ablation Analysis — Conditioning, CFG, and Solver Contributions',
                 fontsize=12, fontweight='bold', pad=15)
    
    plt.tight_layout()
    fig.savefig(FIGURES / "fig8_tables.png", facecolor='white')
    fig.savefig(FIGURES / "fig8_tables.pdf", facecolor='white')
    plt.close()
    print("    ✓ fig8_tables.png/pdf")


def make_fig8b_baseline_table(baseline_data):
    """Generate TABLE III: Comparison with Baseline Generative Models.
    
    Rendered as part of the fig8 table output set.
    """
    # This data is integrated into fig9_baseline_comparison instead
    pass


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 9: BASELINE MODEL COMPARISON (4 panels)
# ═══════════════════════════════════════════════════════════════════

def make_fig9_baseline_comparison(results_data, baseline_data):
    """Phase 2 Decoder Architecture Comparison — 4 panels with expanded baselines."""
    print("  [Fig 9] Phase 2 Decoder Architecture Comparison (expanded)...")

    oracle = baseline_data['oracle_baselines']
    ff = baseline_data['feed_forward_decoders']
    diff = baseline_data['diffusion_decoders']
    real = baseline_data['real_data_baseline']
    rc = real.get('random_chance', 0.01)

    # Model display order: Oracle → Feed-Forward (incl. Direct Transformer) → Diffusion
    models = [
        ('Per-Type\nGaussian',  oracle['per_type_gaussian'],       COLORS['teal'],   'Oracle'),
        ('Retrieval\n+Jitter',  oracle['retrieval_jitter'],        COLORS['purple'],  'Oracle'),
        ('Direct\nTransformer', ff.get('direct_transformer', {}),  COLORS['amber'],   'FF'),
        ('cVAE',                ff['conditional_vae'],             COLORS['orange'],  'FF'),
        ('cGAN\n(WGAN-GP)',     ff['conditional_gan'],             COLORS['green'],   'FF'),
        ('DDPM-DiT',            diff.get('ddpm_dit', {}),          COLORS['grey'],    'DDPM'),
        ('Flow MLP',            diff['flow_mlp_euler'],            COLORS['pink'],    'Diff'),
        ('CLOP-DiT\n(CFG=2.0)', diff['clop_dit_best_acc'],        COLORS['blue'],    'Diff'),
        ('CLOP-DiT\n(CFG=1.0)', diff['clop_dit_balanced'],        COLORS['indigo'],  'Diff'),
    ]

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30)

    labels = [m[0] for m in models]
    colors = [m[2] for m in models]
    x = np.arange(len(models))

    # (a) KNN Top-1 Accuracy
    ax = fig.add_subplot(gs[0, 0])
    vals = [m[1].get('knn_top1', 0) for m in models]
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(y=rc, color='red', linestyle='--', alpha=0.6, linewidth=1,
               label=f'Random chance ({rc:.3f})')
    ax.axhline(y=real.get('knn_top1', 0.89), color=COLORS['red'], linestyle='--',
               alpha=0.6, linewidth=1, label=f'Real data ({real.get("knn_top1", 0.89):.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel('KNN Top-1 Accuracy')
    ax.set_title('(a) Cell Type Classification Accuracy (KNN-1) ↑')
    ax.legend(fontsize=7)
    # Category separators
    ax.axvline(x=1.5, color='grey', linestyle=':', alpha=0.3)
    ax.axvline(x=4.5, color='grey', linestyle=':', alpha=0.3)
    ax.text(0.5, 0.98, 'Oracle', fontsize=7, color='grey', ha='center', va='top')
    ax.text(3.0, 0.98, 'Feed-Forward', fontsize=7, color='grey', ha='center', va='top')
    ax.text(6.5, 0.98, 'Diffusion/Flow', fontsize=7, color='grey', ha='center', va='top')
    for bar, v in zip(bars, vals):
        if v > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.1%}',
                    ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    # (b) Diversity Ratio (capped at 2.5 for readability)
    ax = fig.add_subplot(gs[0, 1])
    vals_divr = [m[1].get('diversity_ratio', 0) for m in models]
    vals_show = [min(v, 2.5) for v in vals_divr]
    bars = ax.bar(x, vals_show, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.6, linewidth=1,
               label='Ideal (1.0)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel('Diversity Ratio')
    ax.set_title('(b) Within-Group Diversity Ratio (→1.0)')
    ax.set_ylim(0, 2.6)
    ax.legend(fontsize=7)
    ax.axvline(x=1.5, color='grey', linestyle=':', alpha=0.3)
    ax.axvline(x=4.5, color='grey', linestyle=':', alpha=0.3)
    for bar, v, v_orig in zip(bars, vals_show, vals_divr):
        lbl = f'{v_orig:.3f}' if v_orig < 1 else (f'{v_orig:.2f}' if v_orig <= 2.5 else f'{v_orig:.1f}↑')
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, lbl,
                ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    # (c) Per-Group FD (log scale)
    ax = fig.add_subplot(gs[1, 0])
    vals_fd = [m[1].get('per_group_fd', 0) for m in models]
    vals_fd_safe = [max(v, 0.005) for v in vals_fd]
    bars = ax.bar(x, vals_fd_safe, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel('Per-Group FD (log scale) ↓')
    ax.set_title('(c) Per-Group Fréchet Distance (FD$_g$) ↓')
    ax.axvline(x=1.5, color='grey', linestyle=':', alpha=0.3)
    ax.axvline(x=4.5, color='grey', linestyle=':', alpha=0.3)
    for bar, v in zip(bars, vals_fd):
        ax.text(bar.get_x() + bar.get_width()/2, max(v, 0.005) * 1.5, f'{v:.2f}',
                ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    # (d) Accuracy–Diversity trade-off scatter
    ax = fig.add_subplot(gs[1, 1])
    for i, (name, data, color, dtype) in enumerate(models):
        knn = data.get('knn_top1', 0)
        divr = data.get('diversity_ratio', 0)
        marker = 'o' if dtype in ('Diff', 'DDPM') else ('D' if dtype == 'Oracle' else 's')
        size = 140 if dtype in ('Diff', 'DDPM') else 80
        ax.scatter(knn, min(divr, 2.5), c=color, s=size, marker=marker, zorder=3,
                   edgecolors='white', linewidth=0.5, label=name.replace('\n', ' '))

    # ODE step ablation if available
    step_abl = baseline_data.get('ode_step_ablation', {})
    if step_abl:
        for step_str, sdata in sorted(step_abl.items(), key=lambda x: int(x[0])):
            knn = sdata.get('knn_top1', 0)
            divr = sdata.get('diversity_ratio', 0)
            ax.scatter(knn, min(divr, 2.5), c=COLORS['blue'], s=30, marker='^',
                       alpha=0.5, zorder=2, edgecolors='white', linewidth=0.3)
            ax.annotate(f'{step_str}s', (knn, min(divr, 2.5)),
                        fontsize=5.5, color=COLORS['blue'], alpha=0.7,
                        textcoords='offset points', xytext=(4, -3))

    # Real data point
    ax.scatter(real.get('knn_top1', 0.89), 1.0, c='black', s=200, marker='*',
               zorder=4, edgecolors='white', linewidth=0.5, label='Real Data')

    ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.4)
    ax.axvline(x=rc, color='red', linestyle=':', alpha=0.4)
    ax.set_xlabel('KNN Top-1 Accuracy ↑')
    ax.set_ylabel('Diversity Ratio (→1.0)')
    ax.set_ylim(0, 2.6)
    ax.set_title('(d) Accuracy–Diversity Trade-off (▲ = ODE steps)')
    ax.legend(fontsize=5.5, ncol=2, loc='upper left', markerscale=0.7)

    from matplotlib.patches import FancyBboxPatch
    ideal = FancyBboxPatch((0.3, 0.8), 0.65, 0.4, boxstyle='round,pad=0.02',
                            facecolor='green', alpha=0.06, edgecolor='green',
                            linestyle='--', linewidth=0.8)
    ax.add_patch(ideal)
    ax.text(0.62, 1.15, 'Ideal Region', fontsize=8, color='green', alpha=0.7,
            ha='center', style='italic')

    fig.suptitle('Phase 2 Decoder Architecture Comparison\n'
                 '(All decoders conditioned on same CLOP text projections, 100 groups × 200 cells)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig9_baseline_comparison.png", facecolor='white')
    fig.savefig(FIGURES / "fig9_baseline_comparison.pdf", facecolor='white')
    plt.close()
    print("    ✓ fig9_baseline_comparison.png/pdf")


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 10: ODE STEP COUNT ABLATION
# ═══════════════════════════════════════════════════════════════════

def make_fig10_ode_steps(baseline_data):
    """ODE integration step count ablation — 3 panels showing the accuracy-diversity trade-off."""
    step_abl = baseline_data.get('ode_step_ablation', {})
    if not step_abl:
        print("  [Fig 10] Skipped (no ode_step_ablation data)")
        return
    print("  [Fig 10] ODE Step Count Ablation...")

    steps_int = sorted([int(k) for k in step_abl.keys()])
    steps_str = [str(s) for s in steps_int]
    data = [step_abl[s] for s in steps_str]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Color gradient from light to dark blue
    n = len(steps_int)
    cmap = plt.cm.Blues
    step_colors = [cmap(0.3 + 0.65 * i / (n - 1)) for i in range(n)]

    # (a) KNN Top-1 Accuracy vs Steps
    ax = axes[0]
    knn_vals = [d.get('knn_top1', 0) for d in data]
    ax.plot(steps_int, knn_vals, 'o-', color=COLORS['blue'], linewidth=2, markersize=8)
    for i, (s, v) in enumerate(zip(steps_int, knn_vals)):
        ax.annotate(f'{v:.1%}', (s, v), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=8, fontweight='bold')
    ax.set_xlabel('ODE Integration Steps')
    ax.set_ylabel('KNN Top-1 Accuracy')
    ax.set_title('(a) Classification Accuracy vs Steps ↑')
    ax.set_xscale('log')
    ax.set_xticks(steps_int)
    ax.set_xticklabels(steps_str)
    ax.grid(True, alpha=0.3)

    # (b) Diversity Ratio vs Steps
    ax = axes[1]
    divr_vals = [d.get('diversity_ratio', 0) for d in data]
    ax.plot(steps_int, divr_vals, 's-', color=COLORS['green'], linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='grey', linestyle='--', alpha=0.5, label='Ideal (1.0)')
    for i, (s, v) in enumerate(zip(steps_int, divr_vals)):
        lbl = f'{v:.3f}' if v < 1 else f'{v:.2f}'
        ax.annotate(lbl, (s, v), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=8, fontweight='bold')
    ax.set_xlabel('ODE Integration Steps')
    ax.set_ylabel('Diversity Ratio')
    ax.set_title('(b) Within-Group Diversity vs Steps (→1.0)')
    ax.set_xscale('log')
    ax.set_xticks(steps_int)
    ax.set_xticklabels(steps_str)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (c) Downstream Accuracy vs Steps
    ax = axes[2]
    ds_vals = [d.get('downstream_acc', 0) for d in data]
    ax.plot(steps_int, ds_vals, 'D-', color=COLORS['orange'], linewidth=2, markersize=8)
    for i, (s, v) in enumerate(zip(steps_int, ds_vals)):
        ax.annotate(f'{v:.1%}', (s, v), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=8, fontweight='bold')
    ax.set_xlabel('ODE Integration Steps')
    ax.set_ylabel('Downstream Accuracy')
    ax.set_title('(c) Downstream Utility vs Steps ↑')
    ax.set_xscale('log')
    ax.set_xticks(steps_int)
    ax.set_xticklabels(steps_str)
    ax.grid(True, alpha=0.3)

    fig.suptitle('ODE Integration Step Count Ablation\n'
                 '(CLOP-DiT with CFG=2.0, Euler integrator, 10-step default)',
                 fontsize=13, fontweight='bold', y=1.04)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig10_ode_steps.png", facecolor='white')
    fig.savefig(FIGURES / "fig10_ode_steps.pdf", facecolor='white')
    plt.close()
    print("    ✓ fig10_ode_steps.png/pdf")


# ═══════════════════════════════════════════════════════════════════
#  ADDITIONAL: Update top-level figures (final_eval_summary, etc.)
# ═══════════════════════════════════════════════════════════════════

def make_summary_figure(results_data):
    """Re-create the main summary figure at figures/ root for the eval report."""
    print("  [Summary] Final Evaluation Summary...")
    
    res = results_data['results']
    real = results_data['real_data_baseline']
    rc = real['random_chance']
    sweep_data = results_data['cfg_sweep']
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    
    # Panel A: KNN accuracy bar chart
    ax = fig.add_subplot(gs[0, 0])
    methods = ["CFG=3.0\nEuler", "CFG=2.0\nEuler", "CFG=1.0\nEuler", 
               "CFG=1.0\nMidpoint", "CFG=0.0", "Gaussian"]
    keys = ["CFG=3.0 Euler-10", "CFG=2.0 Euler-10", "CFG=1.0 Euler-10",
            "CFG=1.0 Midpoint-10", "CFG=0.0 Euler-10", "Gaussian"]
    knn1 = [res[k]['knn_top1'] for k in keys]
    knn5 = [res[k]['knn_top5'] for k in keys]
    
    x = np.arange(len(methods))
    w = 0.35
    ax.bar(x - w/2, knn1, w, color=COLORS['blue'], alpha=0.85, label='KNN Top-1')
    ax.bar(x + w/2, knn5, w, color=COLORS['green'], alpha=0.85, label='KNN Top-5')
    ax.axhline(y=rc, color='red', linestyle='--', alpha=0.7, label=f'Random ({rc:.3f})')
    ax.axhline(y=real['knn_top1'], color=COLORS['amber'], linestyle='--', alpha=0.7, 
               label=f'Real ({real["knn_top1"]:.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_title('(a) Cell Type Classification Accuracy')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, min(1.0, real['knn_top1'] + 0.1))
    
    # Panel B: Steering & Diversity
    ax = fig.add_subplot(gs[0, 1])
    sd_keys = ['CFG=3.0 Euler-10', 'CFG=2.0 Euler-10', 'CFG=1.0 Euler-10',
               'CFG=1.0 Midpoint-10', 'CFG=0.0 Euler-10', 'Gaussian']
    sd_labels = ['CFG=3.0', 'CFG=2.0', 'CFG=1.0', 'CFG=1.0\nMid', 'CFG=0.0', 'Gauss']
    steer = [res[k]['steering'] for k in sd_keys]
    divr = [res[k]['diversity_ratio'] for k in sd_keys]
    
    x = np.arange(len(sd_labels))
    ax2 = ax.twinx()
    b1 = ax.bar(x - 0.15, steer, 0.3, color=COLORS['orange'], alpha=0.85, label='Steering')
    b2 = ax2.bar(x + 0.15, divr, 0.3, color=COLORS['purple'], alpha=0.85, label='DivR')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
    ax2.axhline(y=1.0, color='purple', linestyle=':', alpha=0.4, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sd_labels, fontsize=8)
    ax.set_ylabel('Steering', color=COLORS['orange'])
    ax2.set_ylabel('DivR', color=COLORS['purple'])
    ax2.spines['right'].set_visible(True)
    ax.set_ylim(0.3, 0.9)
    ax2.set_ylim(0, 2.5)
    ax.set_title('(b) Controllability & Diversity')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')
    
    # Panel C: CFG Sweep
    ax = fig.add_subplot(gs[1, 0])
    sweep_10 = sorted([r for r in sweep_data if r['steps'] == 10], key=lambda r: r['cfg'])
    cfgs_s = [r['cfg'] for r in sweep_10]
    knn_s = [r['knn1']/rc for r in sweep_10]
    divr_s = [r['div_ratio'] for r in sweep_10]
    
    ax_d = ax.twinx()
    ax.plot(cfgs_s, knn_s, '-o', color=COLORS['blue'], linewidth=2, markersize=6, label='KNN/Random (×)')
    ax_d.plot(cfgs_s, divr_s, '--s', color=COLORS['red'], linewidth=1.5, markersize=5, alpha=0.8, label='DivR')
    ax_d.axhline(y=1.0, color='red', linestyle=':', alpha=0.3)
    ax.set_xlabel('CFG Scale')
    ax.set_ylabel('KNN / Random (×)', color=COLORS['blue'])
    ax_d.set_ylabel('Diversity Ratio', color=COLORS['red'])
    ax_d.spines['right'].set_visible(True)
    ax.set_title('(c) CFG Scale: Accuracy–Diversity Trade-off')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_d.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='center right')
    
    # Panel D: Summary text
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    
    best_knn_key = max([k for k in res if k not in ['Gaussian', 'CFG=0.0 Euler-10'] and 'Euler-10' in k],
                       key=lambda k: res[k]['knn_top1'])
    best = res[best_knn_key]
    
    summary_text = (
        f"CLOP-DiT v5.2 — Final Verified Results\n"
        f"{'─'*44}\n\n"
        f"Dataset:     220,304 cells, 80 datasets\n"
        f"Eval:        100 groups × 200 cells\n\n"
        f"Best Accuracy ({best_knn_key}):\n"
        f"  KNN Top-1:   {best['knn_top1']:.1%}  ({best['knn_top1']/rc:.0f}× random)\n"
        f"  KNN Top-5:   {best['knn_top5']:.1%}\n"
        f"  Steering:    {best['steering']:.1%}  (random = 50%)\n"
        f"  Linear Acc:  {best['linear_acc']:.1%}\n\n"
        f"Best Balanced (CFG=1.0 Midpoint-10):\n"
        f"  KNN Top-1:   {res['CFG=1.0 Midpoint-10']['knn_top1']:.1%}\n"
        f"  DivR:        {res['CFG=1.0 Midpoint-10']['diversity_ratio']:.3f}\n\n"
        f"Controls:\n"
        f"  Unconditional KNN: {res['CFG=0.0 Euler-10']['knn_top1']:.1%} = random\n"
        f"  Gaussian KNN:      {res['Gaussian']['knn_top1']:.1%} = random\n"
        f"  Real Data KNN:     {real['knn_top1']:.1%}"
    )
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#F5F5F5', edgecolor='#BDBDBD'))
    ax.set_title('(d) Key Results Summary', fontsize=12, fontweight='bold')
    
    fig.suptitle('CLOP-DiT: Conditional Single-Cell Generation — Final Evaluation',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(PROJECT / "figures/final_eval_summary.png", dpi=300, facecolor='white')
    plt.close()
    print("    ✓ figures/final_eval_summary.png")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    seed_all(42)
    
    print("=" * 70)
    print("  CLOP-DiT: Publication Figure Generation")
    print("=" * 70)
    
    # Load results
    with open(RESULTS / "final_evaluation.json") as f:
        results_data = json.load(f)
    print(f"  Loaded results: {len(results_data['results'])} configs")
    
    # ── Figure 2: Training dynamics (no GPU needed) ──
    make_fig2_training_dynamics()
    
    # ── Figure 4: Metrics dashboard (no GPU needed) ──
    make_fig4_metrics_dashboard(results_data)
    
    # ── Figure 6: CFG sweep + solver (no GPU needed) ──
    make_fig6_cfg_sweep_and_solver(results_data)
    
    # ── Figure 8: Tables (no GPU needed) ──
    baseline_path = RESULTS / "baseline_comparison.json"
    baseline_data = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_data = json.load(f)
    make_fig8_tables(results_data, baseline_data)
    
    # ── Figure 9: Baseline comparison (no GPU needed) ──
    if baseline_data is not None:
        make_fig9_baseline_comparison(results_data, baseline_data)
        make_fig10_ode_steps(baseline_data)
    else:
        print("  [Fig 9-10] SKIPPED — run 17_baseline_comparison.py first")
    
    # ── Summary figure (no GPU needed) ──
    make_summary_figure(results_data)
    
    # ── GPU-dependent figures ──
    print("\n  Loading data and model for GPU-dependent figures...")
    cell_emb = np.load(CACHE / "cell_embeddings.npy")
    text_emb = np.load(CACHE / "text_embeddings.npy")
    proj_text = np.load(CACHE / "projected_text.npy")
    
    N, D = cell_emb.shape
    global_mean = cell_emb.mean(0)
    
    _, text_labels = np.unique(np.round(text_emb[:, :50], 4), axis=0, return_inverse=True)
    group_sizes = Counter(text_labels.tolist())
    group_indices = defaultdict(list)
    for i, g in enumerate(text_labels):
        group_indices[g].append(i)
    group_indices = {g: np.array(v) for g, v in group_indices.items()}
    
    valid_groups = sorted([g for g, s in group_sizes.items() if s >= 50],
                         key=lambda g: group_sizes[g], reverse=True)[:100]
    group_cond = {g: proj_text[group_indices[g][0]] for g in valid_groups}
    
    print(f"  {N:,} cells, {len(valid_groups)} eval groups")
    
    # Load DiT
    from src.architecture.dit import DiT1D
    dit = DiT1D(latent_dim=512, hidden_dim=384, cond_dim=256,
                num_blocks=8, num_heads=6).to(DEVICE)
    ckpt = torch.load(CKPT / "dit_best.pth", map_location=DEVICE, weights_only=False)
    dit.load_state_dict(ckpt["model_state_dict"])
    dit.eval()
    print(f"  DiT loaded: {dit.count_parameters()['trainable_M']} params")
    
    # ── Figure 3: Embedding visualization ──
    make_fig3_embedding(cell_emb, text_labels, group_indices, valid_groups,
                        group_cond, dit, global_mean)
    
    # ── Figure 5: Biological validation ──
    make_fig5_biological_validation(cell_emb, text_labels, group_indices,
                                     valid_groups, group_cond, dit, global_mean)
    
    # ── Figure 7: Dimension & sampling ──
    make_fig7_dimension_sampling(cell_emb, group_cond, valid_groups, dit, global_mean)
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  All figures generated in {elapsed:.0f}s")
    print(f"  Output: {FIGURES}")
    print(f"{'=' * 70}")
    
    # ── Clean stale figures (old naming convention) ──
    stale = ["fig3_umap_embedding", "fig6_dimension_sampling", "fig7_alignment_metrics",
             "fig2_umap_embedding", "fig3_enhanced_metrics", "fig6_comparison_table", 
             "fig7_ablation_table"]
    for name in stale:
        for ext in (".png", ".pdf"):
            fp = FIGURES / f"{name}{ext}"
            if fp.exists():
                fp.unlink()
                print(f"  [cleanup] Removed stale {fp.name}")
    
    # ── Visual conflict detection ──
    detect_visual_conflicts()
    
    # List all outputs
    for f in sorted(FIGURES.glob("*")):
        size = f.stat().st_size / 1024
        print(f"    {f.name:<40} {size:>8.1f} KB")


if __name__ == "__main__":
    main()
