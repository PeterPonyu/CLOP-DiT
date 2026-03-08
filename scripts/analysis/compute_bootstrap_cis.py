#!/usr/bin/env python3
"""
compute_bootstrap_cis.py — Bootstrap 95% confidence intervals for headline metrics.

Strategy: Precompute per-type metric components ONCE (expensive KNN/LR prediction),
then resample types B=1000 times (with replacement) to build CIs using only the
precomputed per-type summaries. This avoids re-running KNN prediction on 167K
training points for each bootstrap iteration.

Outputs results/bootstrap_cis.json.
"""

import numpy as np
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT = Path("/home/zeyufu/Desktop/CLOP-DiT")

B = 1000  # bootstrap iterations
SEED = 42
N_STEER_PAIRS = 500


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # ── Load data ──
    print("[1] Loading data...")
    cell_emb = np.load(PROJECT / "data/cached_latents_v5.2/cell_embeddings_dedup_preprocessed.npy")
    group_ids = np.load(PROJECT / "data/cached_latents_v5.2/text_group_ids_dedup.npy")
    gen_emb = np.load(PROJECT / "results/generated_embeddings.npy")
    gen_labels = np.load(PROJECT / "results/generated_labels.npy")

    N, D = cell_emb.shape
    global_mean = cell_emb.mean(0)
    print(f"    Real: {N} cells, {D}-dim")
    print(f"    Gen:  {gen_emb.shape[0]} cells, {gen_emb.shape[1]}-dim")

    # ── Build group structures ──
    group_indices = defaultdict(list)
    for i, g in enumerate(group_ids):
        group_indices[g].append(i)
    group_indices = {g: np.array(v) for g, v in group_indices.items()}

    all_groups = sorted(group_indices.keys())
    group_centroids = {g: cell_emb[group_indices[g]].mean(0) for g in all_groups}
    group_stds = {g: cell_emb[group_indices[g]].std(0).mean() for g in all_groups}

    # ── Organise generated data by type ──
    gen_by_type = {}
    for t in np.unique(gen_labels):
        gen_by_type[int(t)] = gen_emb[gen_labels == t]

    eval_groups = sorted([g for g in all_groups if g in gen_by_type])
    n_types = len(eval_groups)
    print(f"    {n_types} types in common")

    # ── Train classifiers once on real data ──
    print("[2] Training classifiers...")
    cell_centered = cell_emb - global_mean
    pca = PCA(n_components=50, random_state=42)
    cell_pca = pca.fit_transform(cell_centered)

    eval_mask = np.isin(group_ids, eval_groups)
    eval_pca = cell_pca[eval_mask]
    eval_labels = group_ids[eval_mask]

    perm = rng.permutation(eval_mask.sum())
    split = int(len(perm) * 0.8)
    train_idx, test_idx = perm[:split], perm[split:]

    knn = KNeighborsClassifier(n_neighbors=15, metric="cosine", n_jobs=-1)
    knn.fit(eval_pca[train_idx], eval_labels[train_idx])
    knn_classes = knn.classes_
    print(f"    KNN fitted. Training LogReg...")

    scaler = StandardScaler()
    eval_pca_scaled = scaler.fit_transform(eval_pca)
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="saga", random_state=42)
    lr.fit(eval_pca_scaled[train_idx], eval_labels[train_idx])

    real_knn_acc = accuracy_score(eval_labels[test_idx], knn.predict(eval_pca[test_idx]))
    random_chance = 1.0 / n_types
    print(f"    Real KNN acc: {real_knn_acc:.4f}, Random chance: {random_chance:.5f}")

    # ── Precompute per-type metric components (one-time KNN/LR prediction) ──
    print("[3] Precomputing per-type metrics...")

    all_gen = np.vstack([gen_by_type[g] for g in eval_groups])
    all_labels = np.concatenate([np.full(len(gen_by_type[g]), g) for g in eval_groups])

    gen_c = all_gen - global_mean
    gen_pca_all = pca.transform(gen_c)
    knn_preds = knn.predict(gen_pca_all)
    knn_proba = knn.predict_proba(gen_pca_all)

    gen_pca_scaled = scaler.transform(gen_pca_all)
    lr_preds = lr.predict(gen_pca_scaled)

    per_type = {}
    offset = 0
    for g in eval_groups:
        n_g = len(gen_by_type[g])
        sl = slice(offset, offset + n_g)

        # KNN top-1 correct count
        knn1_correct = int(np.sum(knn_preds[sl] == g))

        # KNN top-5 correct count
        knn5_correct = int(sum(
            1 for j in range(n_g)
            if g in knn_classes[np.argsort(knn_proba[offset + j])[-5:]]
        ))

        # LR correct count
        lr_correct = int(np.sum(lr_preds[sl] == g))

        # Diversity: gen std for this type
        gen_std = float(gen_by_type[g].std(0).mean())

        # Centroid cosine (centered)
        gc = gen_by_type[g].mean(0) - global_mean
        rc = group_centroids[g] - global_mean
        ctr_cos = float(np.dot(rc, gc) / (np.linalg.norm(rc) * np.linalg.norm(gc) + 1e-8))

        per_type[g] = {
            "n": n_g,
            "knn1_correct": knn1_correct,
            "knn5_correct": knn5_correct,
            "lr_correct": lr_correct,
            "gen_std": gen_std,
            "real_std": float(group_stds[g]),
            "ctr_cos": ctr_cos,
            "gen_centroid_centered": gc,
            "real_centroid_centered": rc,
        }
        offset += n_g

    print(f"    Per-type precompute done ({time.time() - t0:.1f}s)")

    # ── Point estimates ──
    total_cells = sum(per_type[g]["n"] for g in eval_groups)
    knn1_point = sum(per_type[g]["knn1_correct"] for g in eval_groups) / total_cells
    knn5_point = sum(per_type[g]["knn5_correct"] for g in eval_groups) / total_cells
    lr_point = sum(per_type[g]["lr_correct"] for g in eval_groups) / total_cells
    divr_point = np.mean([per_type[g]["gen_std"] for g in eval_groups]) / \
                 np.mean([per_type[g]["real_std"] for g in eval_groups])

    # Steering point estimate (1000 pairs)
    rng_steer = np.random.default_rng(123)
    steer_correct = 0
    for _ in range(1000):
        a, b_ = rng_steer.choice(eval_groups, 2, replace=False)
        ga = per_type[a]["gen_centroid_centered"]
        ra = per_type[a]["real_centroid_centered"]
        rb = per_type[b_]["real_centroid_centered"]
        daa = np.dot(ga, ra) / (np.linalg.norm(ga) * np.linalg.norm(ra) + 1e-8)
        dab = np.dot(ga, rb) / (np.linalg.norm(ga) * np.linalg.norm(rb) + 1e-8)
        if daa > dab:
            steer_correct += 1
    steer_point = steer_correct / 1000
    ctr_cos_point = float(np.mean([per_type[g]["ctr_cos"] for g in eval_groups]))

    print(f"\n[4] Point estimates:")
    print(f"    KNN-1={knn1_point:.4f}  KNN-5={knn5_point:.4f}  "
          f"Steer={steer_point:.4f}  DivR={divr_point:.4f}  LinAcc={lr_point:.4f}")

    # ── Bootstrap over types (fast: uses precomputed per-type values) ──
    print(f"\n[5] Running {B} bootstrap iterations (resampling {n_types} types)...")
    eval_groups_arr = np.array(eval_groups)

    boot_knn1 = np.zeros(B)
    boot_knn5 = np.zeros(B)
    boot_lr = np.zeros(B)
    boot_divr = np.zeros(B)
    boot_steer = np.zeros(B)
    boot_ctr_cos = np.zeros(B)

    for b in range(B):
        if (b + 1) % 200 == 0:
            print(f"    Iteration {b+1}/{B}  ({time.time() - t0:.0f}s)")

        # Resample types with replacement
        boot_types = list(rng.choice(eval_groups_arr, n_types, replace=True))
        unique_boot = list(set(boot_types))

        # Aggregate per-type precomputed values (weighted by count)
        total_n = sum(per_type[g]["n"] for g in boot_types)
        boot_knn1[b] = sum(per_type[g]["knn1_correct"] for g in boot_types) / total_n
        boot_knn5[b] = sum(per_type[g]["knn5_correct"] for g in boot_types) / total_n
        boot_lr[b] = sum(per_type[g]["lr_correct"] for g in boot_types) / total_n
        boot_divr[b] = np.mean([per_type[g]["gen_std"] for g in boot_types]) / \
                        np.mean([per_type[g]["real_std"] for g in boot_types])
        boot_ctr_cos[b] = np.mean([per_type[g]["ctr_cos"] for g in boot_types])

        # Steering with resampled types (only unique types for pair selection)
        if len(unique_boot) >= 2:
            rng_s = np.random.default_rng(123 + b)
            correct = 0
            for _ in range(N_STEER_PAIRS):
                a, b_ = rng_s.choice(unique_boot, 2, replace=False)
                ga = per_type[a]["gen_centroid_centered"]
                ra = per_type[a]["real_centroid_centered"]
                rb = per_type[b_]["real_centroid_centered"]
                daa = np.dot(ga, ra) / (np.linalg.norm(ga) * np.linalg.norm(ra) + 1e-8)
                dab = np.dot(ga, rb) / (np.linalg.norm(ga) * np.linalg.norm(rb) + 1e-8)
                if daa > dab:
                    correct += 1
            boot_steer[b] = correct / N_STEER_PAIRS
        else:
            boot_steer[b] = np.nan

    # ── Compute CIs ──
    print(f"\n[6] Computing 95% CIs...")
    point_dict = {
        "knn_top1": knn1_point,
        "knn_top5": knn5_point,
        "steering": steer_point,
        "diversity_ratio": divr_point,
        "linear_acc": lr_point,
        "centroid_cosine": ctr_cos_point,
    }
    boot_dict = {
        "knn_top1": boot_knn1,
        "knn_top5": boot_knn5,
        "steering": boot_steer[~np.isnan(boot_steer)],
        "diversity_ratio": boot_divr,
        "linear_acc": boot_lr,
        "centroid_cosine": boot_ctr_cos,
    }

    ci_results = {}
    for name in point_dict:
        pt = point_dict[name]
        vals = boot_dict[name]
        lo = float(np.percentile(vals, 2.5))
        hi = float(np.percentile(vals, 97.5))
        ci_results[name] = {
            "point_estimate": float(pt),
            "ci_95_lower": lo,
            "ci_95_upper": hi,
            "bootstrap_std": float(np.std(vals)),
            "bootstrap_median": float(np.median(vals)),
            "n_bootstrap": int(len(vals)),
        }
        print(f"    {name:>18s}: {pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    # ── Save ──
    output = {
        "metadata": {
            "n_bootstrap": B,
            "n_types": n_types,
            "n_gen_per_type": int(len(gen_emb) / n_types),
            "resampling_unit": "cell_types",
            "ci_method": "percentile",
            "seed": SEED,
            "generation_config": {
                "cfg_scale": 1.5,
                "num_steps": 20,
                "solver": "midpoint",
            },
            "note": "CIs computed by resampling cell types with replacement. "
                    "Classifier weights are fixed; only the set of types included "
                    "in each bootstrap varies.",
            "real_data_baseline_knn1": float(real_knn_acc),
            "random_chance": float(random_chance),
            "elapsed_seconds": time.time() - t0,
        },
        "metrics": ci_results,
    }

    out_path = PROJECT / "results/bootstrap_cis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n    Saved: {out_path}")
    print(f"    Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
