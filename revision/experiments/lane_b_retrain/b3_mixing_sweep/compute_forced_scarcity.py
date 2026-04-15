"""B3 extension — Forced-scarcity mixing sweep.

The natural-scarcity B3 result showed no strategy improves rare-class F1
because baseline is already 0.92-0.94 (little headroom). A reviewer may
push back: "of course augmentation doesn't help at 92% — what about
genuinely rare types?" This extension artificially reduces the rare-class
training set to 30 cells, creating genuine headroom, and tests whether
CLOP-DiT augmentation fills the gap.

Reuses the B3 infrastructure (same strategies, ratios, classifier) but
with artificial scarcity.

Inputs (same as B3):
  data/cached_latents/cell_embeddings_dedup_preprocessed.npy
  data/cached_latents/text_group_ids_dedup.npy
  results/generated_embeddings.npy
  results/generated_labels.npy
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]

REAL_LATENT = REPO_ROOT / "data" / "cached_latents" / "cell_embeddings_dedup_preprocessed.npy"
REAL_LABELS = REPO_ROOT / "data" / "cached_latents" / "text_group_ids_dedup.npy"
GEN_LATENT = REPO_ROOT / "results" / "generated_embeddings.npy"
GEN_LABELS = REPO_ROOT / "results" / "generated_labels.npy"

RARE_GIDS = [51, 64]  # Megakaryocytes, Ameloblasts
MAJORITY_CAP = 300
SEED = 0
RATIOS = [1, 2, 5, 10]
FORCED_RARE_TRAIN = 30  # Artificially keep only 30 rare training cells
TEST_FRAC = 0.30


def smote(minority: np.ndarray, n_synth: int,
          k: int, rng: np.random.Generator) -> np.ndarray:
    n, d = minority.shape
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(minority)
    _, idx = nn.kneighbors(minority)
    idx = idx[:, 1:]
    anchors = rng.integers(0, n, n_synth)
    neighbors = idx[anchors, rng.integers(0, k_eff, n_synth)]
    alpha = rng.uniform(0, 1, (n_synth, 1)).astype(minority.dtype)
    return minority[anchors] + alpha * (minority[neighbors] - minority[anchors])


def assemble_majority(real, real_l, rare_gid, test_frac, rng):
    X_train, y_train, X_test, y_test = [], [], [], []
    for gid in np.unique(real_l):
        if gid == rare_gid:
            continue
        idx_all = np.where(real_l == gid)[0]
        n_keep = min(MAJORITY_CAP, idx_all.size)
        idx = rng.choice(idx_all, n_keep, replace=False)
        tr, te = train_test_split(idx, test_size=test_frac,
                                  random_state=int(rng.integers(0, 2**31)))
        X_train.append(real[tr]); y_train.extend([gid] * tr.size)
        X_test.append(real[te]); y_test.extend([gid] * te.size)
    return (np.concatenate(X_train), np.asarray(y_train, dtype=real_l.dtype),
            np.concatenate(X_test), np.asarray(y_test, dtype=real_l.dtype))


def evaluate_strategy(X_tr, y_tr, X_te, y_te, rare_gid):
    clf = LogisticRegression(solver="lbfgs", max_iter=120,
                             n_jobs=-1, multi_class="auto")
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return {
        "rare_f1": float(f1_score(y_te, pred, labels=[rare_gid], average=None)[0]),
        "macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        "accuracy": float((pred == y_te).mean()),
        "train_n": int(X_tr.shape[0]),
        "fit_seconds": round(time.time() - t0, 2),
    }


def run_forced_scarcity(real, real_l, gen, gen_l, rare_gid):
    rng = np.random.default_rng(SEED + rare_gid)
    X_maj_tr, y_maj_tr, X_te_maj, y_te_maj = assemble_majority(
        real, real_l, rare_gid, TEST_FRAC, rng
    )

    rare_real = real[real_l == rare_gid]
    rare_gen = gen[gen_l == rare_gid]

    # Split rare cells: use full test set for meaningful evaluation
    rare_indices = np.arange(rare_real.shape[0])
    rng_split = np.random.default_rng(SEED)
    rng_split.shuffle(rare_indices)

    # Keep a proper test set (30% of all rare), then subsample train to FORCED_RARE_TRAIN
    n_test = max(int(rare_real.shape[0] * TEST_FRAC), 50)
    rare_te_idx = rare_indices[:n_test]
    rare_pool_idx = rare_indices[n_test:]

    # Artificially subsample train to FORCED_RARE_TRAIN
    if rare_pool_idx.size > FORCED_RARE_TRAIN:
        rare_tr_idx = rng.choice(rare_pool_idx, FORCED_RARE_TRAIN, replace=False)
    else:
        rare_tr_idx = rare_pool_idx

    X_rare_tr_real = rare_real[rare_tr_idx]
    X_rare_te = rare_real[rare_te_idx]

    X_te = np.concatenate([X_te_maj, X_rare_te])
    y_te = np.concatenate([y_te_maj, np.full(X_rare_te.shape[0], rare_gid,
                                              dtype=y_maj_tr.dtype)])

    n_rare_tr = X_rare_tr_real.shape[0]

    # Baseline
    X_tr = np.concatenate([X_maj_tr, X_rare_tr_real])
    y_tr = np.concatenate([y_maj_tr, np.full(n_rare_tr, rare_gid, dtype=y_maj_tr.dtype)])
    base = evaluate_strategy(X_tr, y_tr, X_te, y_te, rare_gid)

    out = {
        "rare_gid": int(rare_gid),
        "forced_train_n": int(n_rare_tr),
        "n_real_rare_total": int(rare_real.shape[0]),
        "n_real_rare_test": int(X_rare_te.shape[0]),
        "n_clop_rare_available": int(rare_gen.shape[0]),
        "majority_train_size": int(X_maj_tr.shape[0]),
        "test_total_size": int(X_te.shape[0]),
        "baseline": base,
        "sweep": {},
    }

    strategies = ("oversampling", "smote", "clop", "clop+oversamp", "clop+smote")

    for strategy in strategies:
        out["sweep"][strategy] = {}
        for ratio in RATIOS:
            n_synth = n_rare_tr * ratio
            rng_local = np.random.default_rng(SEED + rare_gid * 100 + ratio)

            if strategy == "oversampling":
                synth = X_rare_tr_real[rng_local.choice(n_rare_tr, n_synth, replace=True)]

            elif strategy == "smote":
                synth = smote(X_rare_tr_real, n_synth, k=5, rng=rng_local)

            elif strategy == "clop":
                n_avail = rare_gen.shape[0]
                if n_avail == 0:
                    continue
                synth = rare_gen[rng_local.choice(n_avail, n_synth, replace=(n_synth > n_avail))]

            elif strategy == "clop+oversamp":
                half = n_synth // 2
                n_avail = rare_gen.shape[0]
                if n_avail == 0:
                    continue
                synth = np.concatenate([
                    rare_gen[rng_local.choice(n_avail, half, replace=(half > n_avail))],
                    X_rare_tr_real[rng_local.choice(n_rare_tr, n_synth - half, replace=True)]
                ])

            elif strategy == "clop+smote":
                half = n_synth // 2
                n_avail = rare_gen.shape[0]
                if n_avail == 0:
                    continue
                synth = np.concatenate([
                    rare_gen[rng_local.choice(n_avail, half, replace=(half > n_avail))],
                    smote(X_rare_tr_real, n_synth - half, k=5, rng=rng_local)
                ])
            else:
                continue

            X_tr = np.concatenate([X_maj_tr, X_rare_tr_real, synth])
            y_tr = np.concatenate([
                y_maj_tr,
                np.full(n_rare_tr + synth.shape[0], rare_gid, dtype=y_maj_tr.dtype)
            ])
            res = evaluate_strategy(X_tr, y_tr, X_te, y_te, rare_gid)
            res["n_synth"] = int(synth.shape[0])
            out["sweep"][strategy][f"{ratio}x"] = res

    return out


def main():
    real = np.load(REAL_LATENT).astype(np.float32)
    real_l = np.load(REAL_LABELS)
    gen = np.load(GEN_LATENT).astype(np.float32)
    gen_l = np.load(GEN_LABELS)

    per_rare = {}
    for rare_gid in RARE_GIDS:
        print(f"\n=== forced-scarcity gid={rare_gid} (train={FORCED_RARE_TRAIN}) ===")
        out = run_forced_scarcity(real, real_l, gen, gen_l, rare_gid)
        per_rare[int(rare_gid)] = out
        b = out["baseline"]
        print(f"  baseline rare-F1 = {b['rare_f1']:.3f}, macro-F1 = {b['macro_f1']:.3f}")
        for s, ratios in out["sweep"].items():
            line = f"  {s:18s}"
            for r in (f"{rr}x" for rr in RATIOS):
                if r in ratios:
                    line += f"  {r}:{ratios[r]['rare_f1']:.3f}"
            print(line)

    summary = {
        "seed": SEED,
        "forced_rare_train": FORCED_RARE_TRAIN,
        "majority_cap": MAJORITY_CAP,
        "ratios": RATIOS,
        "rare_gids": RARE_GIDS,
        "per_rare_type": per_rare,
    }
    (HERE / "forced_scarcity_sweep.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: float(o)
                   if isinstance(o, np.floating) else None) + "\n"
    )

    # Preview
    lines = [
        "B3 extension — forced-scarcity mixing sweep",
        "=" * 50,
        f"Forced rare training cells: {FORCED_RARE_TRAIN}",
        f"seed={SEED}, majority_cap={MAJORITY_CAP}, ratios={RATIOS}",
        "",
    ]
    for rare_gid, out in per_rare.items():
        lines.append(
            f"=== gid {rare_gid} (forced train={out['forced_train_n']}, "
            f"test={out['n_real_rare_test']}, "
            f"clop_avail={out['n_clop_rare_available']}) ==="
        )
        b = out["baseline"]
        lines.append(f"  baseline                  rare-F1={b['rare_f1']:.3f}  "
                     f"macro-F1={b['macro_f1']:.3f}")
        lines.append(f"  {'strategy':18s} {'1x':>11s} {'2x':>11s} "
                     f"{'5x':>11s} {'10x':>11s}")
        for s, ratios in out["sweep"].items():
            row = f"  {s:18s}"
            for r in (f"{rr}x" for rr in RATIOS):
                if r in ratios:
                    row += f"  {r}:{ratios[r]['rare_f1']:.3f}"
                else:
                    row += "  " + " " * 9 + "-"
            lines.append(row)
        lines.append("")
    preview = "\n".join(lines) + "\n"
    (HERE / "forced_scarcity_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
